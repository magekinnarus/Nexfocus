#!/usr/bin/env python3
"""Local stdlib-only CLI for the loopback creative-document Agent driver.

Examples:
  python tools/creative_document_driver.py --capability-file driver.json status
  python tools/creative_document_driver.py --capability-file driver.json inspect --request inspect.json
  python tools/creative_document_driver.py --capability-file driver.json preview --expected-revision 4 --output preview.png

The capability file is the one-time download from the editor's Local Agent
driver controls. Keep it local and revoke the grant in the editor when done.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


MAX_CAPABILITY_BYTES = 4096
MAX_COMMAND_BYTES = 1 * 1024 * 1024
ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9._:-]{0,127}$")
TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{32,256}$")
CAPABILITY_FIELDS = {"schemaVersion", "baseUrl", "documentId", "actorId", "scopes", "expiresAt", "token"}
ENVELOPE_FIELDS = {
    "schemaVersion", "commandId", "documentId", "intent", "expectedRevision",
    "commandType", "targetIds", "coordinateSpace", "payload", "transaction",
}


class InputError(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InputError("duplicate JSON field")
        result[key] = value
    return result


def _strict_json(data: bytes, maximum: int) -> Any:
    if len(data) > maximum:
        raise InputError("input exceeds the size limit")
    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _: (_ for _ in ()).throw(InputError("non-finite JSON number")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise InputError("input is not valid UTF-8 JSON") from exc
    _validate_tree(value)
    return value


def _validate_tree(value: Any) -> None:
    nodes = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > 100_000 or depth > 32:
            raise InputError("input structure exceeds the limit")
        if item is None or type(item) in {str, bool, int}:
            return
        if type(item) is float:
            if item != item or item in (float("inf"), float("-inf")):
                raise InputError("non-finite JSON number")
            return
        if isinstance(item, list):
            for child in item:
                visit(child, depth + 1)
            return
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise InputError("JSON object keys must be strings")
            for child in item.values():
                visit(child, depth + 1)
            return
        raise InputError("input contains a non-JSON value")

    visit(value, 0)


def _contains(value: Any, needle: str) -> bool:
    if isinstance(value, str):
        return needle in value
    if isinstance(value, list):
        return any(_contains(child, needle) for child in value)
    if isinstance(value, dict):
        return any(_contains(key, needle) or _contains(child, needle) for key, child in value.items())
    return False


def _safe_output(value: Any, secret: str) -> Any:
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret and secret in value else value
    if isinstance(value, list):
        return [_safe_output(child, secret) for child in value]
    if isinstance(value, dict):
        return {
            (key.replace(secret, "[REDACTED]") if secret and secret in key else key): _safe_output(child, secret)
            for key, child in value.items()
        }
    return value


def _read_capability(path: Path) -> dict[str, Any]:
    try:
        if path.stat().st_size > MAX_CAPABILITY_BYTES:
            raise InputError("capability file exceeds the size limit")
        raw = path.read_bytes()
    except InputError:
        raise
    except OSError as exc:
        raise InputError("capability file could not be read") from exc
    capability = _strict_json(raw, MAX_CAPABILITY_BYTES)
    if not isinstance(capability, dict) or set(capability) != CAPABILITY_FIELDS:
        raise InputError("capability file has an unexpected shape")
    if type(capability["schemaVersion"]) is not int or capability["schemaVersion"] != 1:
        raise InputError("unsupported capability schema")
    if not isinstance(capability["documentId"], str) or not ID_PATTERN.fullmatch(capability["documentId"]):
        raise InputError("capability document ID is invalid")
    if not isinstance(capability["actorId"], str) or not ID_PATTERN.fullmatch(capability["actorId"]):
        raise InputError("capability actor ID is invalid")
    if type(capability["expiresAt"]) is not int:
        raise InputError("capability expiry is invalid")
    if capability["expiresAt"] <= int(time.time()):
        raise InputError("capability has expired")
    scopes = capability["scopes"]
    if (not isinstance(scopes, list) or not scopes
            or any(not isinstance(scope, str) or scope not in {"inspect", "propose", "mutate"} for scope in scopes)
            or len(scopes) != len(set(scopes))):
        raise InputError("capability scopes are invalid")
    token = capability["token"]
    if not isinstance(token, str) or not TOKEN_PATTERN.fullmatch(token):
        raise InputError("capability token is malformed")
    base_url = capability["baseUrl"]
    if not isinstance(base_url, str):
        raise InputError("capability URL is invalid")
    parts = urllib.parse.urlsplit(base_url)
    if (parts.scheme != "http" or parts.username is not None or parts.password is not None
            or parts.query or parts.fragment or parts.path not in {"", "/"}):
        raise InputError("capability URL must be plain HTTP loopback")
    try:
        address = ipaddress.ip_address(parts.hostname or "")
        port = parts.port
    except ValueError as exc:
        raise InputError("capability URL must use a loopback IP literal") from exc
    if not address.is_loopback or port is None or not 1 <= port <= 65535:
        raise InputError("capability URL must use a loopback IP literal and valid port")
    normalized_host = f"[{address.compressed}]" if isinstance(address, ipaddress.IPv6Address) else address.compressed
    capability["baseUrl"] = f"http://{normalized_host}:{port}"
    return capability


def _read_request(path: str) -> Any:
    if path == "-":
        raw = sys.stdin.buffer.read(MAX_COMMAND_BYTES + 1)
    else:
        try:
            source = Path(path)
            if source.stat().st_size > MAX_COMMAND_BYTES:
                raise InputError("command request exceeds the size limit")
            raw = source.read_bytes()
        except InputError:
            raise
        except OSError as exc:
            raise InputError("command request file could not be read") from exc
    return _strict_json(raw, MAX_COMMAND_BYTES)


def _validate_envelope(value: Any, command: str, capability: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != ENVELOPE_FIELDS:
        raise InputError("command envelope must use the exact v1 field set")
    if type(value["schemaVersion"]) is not int or value["schemaVersion"] != 1:
        raise InputError("unsupported command schema")
    if value["intent"] != command:
        raise InputError(f"request intent must be {command}")
    if value["documentId"] != capability["documentId"]:
        raise InputError("request document does not match the capability")
    for field in ("commandId", "documentId"):
        if not isinstance(value[field], str) or not ID_PATTERN.fullmatch(value[field]):
            raise InputError(f"{field} is invalid")
    if not isinstance(value["commandType"], str) or not value["commandType"]:
        raise InputError("commandType is invalid")
    if value["intent"] == "inspect":
        if value["expectedRevision"] is not None and (type(value["expectedRevision"]) is not int or value["expectedRevision"] < 0):
            raise InputError("expectedRevision is invalid")
    elif type(value["expectedRevision"]) is not int or value["expectedRevision"] < 0:
        raise InputError("expectedRevision is required")
    if not isinstance(value["coordinateSpace"], str):
        raise InputError("coordinateSpace is invalid")
    if not isinstance(value["targetIds"], list) or any(not isinstance(item, str) or not ID_PATTERN.fullmatch(item) for item in value["targetIds"]):
        raise InputError("targetIds is invalid")
    if not isinstance(value["payload"], dict):
        raise InputError("payload must be an object")
    if not isinstance(value["transaction"], (dict, type(None))):
        raise InputError("transaction is invalid")
    if _contains(value, capability["token"]):
        raise InputError("the bearer capability must not appear in the command envelope")
    return value


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request: urllib.request.Request, response: Any, code: int,
                         message: str, headers: Any, new_url: str) -> None:
        return None


def _request(url: str, token: str, *, method: str = "GET", body: bytes | None = None,
             maximum: int = MAX_COMMAND_BYTES) -> tuple[int, Any, dict[str, str]]:
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=10) as response:
            raw = response.read(maximum + 1)
            if len(raw) > maximum:
                raise InputError("server response exceeds the size limit")
            response_headers = {key.lower(): value for key, value in response.headers.items()}
            return response.status, raw, response_headers
    except urllib.error.HTTPError as exc:
        raw = exc.read(maximum + 1)
        if len(raw) > maximum:
            raise InputError("server response exceeds the size limit") from exc
        response_headers = {key.lower(): value for key, value in exc.headers.items()}
        return exc.code, raw, response_headers


def _print_json(value: Any, secret: str) -> None:
    print(json.dumps(_safe_output(value, secret), ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capability-file", required=True, type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    for intent in ("inspect", "propose", "mutate"):
        child = commands.add_parser(intent)
        child.add_argument("--request", default="-", help="JSON envelope file; defaults to stdin")
    preview = commands.add_parser("preview")
    preview.add_argument("--expected-revision", required=True, type=int)
    preview.add_argument("--output", required=True, type=Path)
    commands.add_parser("status")
    args = parser.parse_args(argv)

    try:
        capability = _read_capability(args.capability_file)
    except InputError as exc:
        print(f"malformed: {exc}", file=sys.stderr)
        return 4
    token = capability["token"]
    base = capability["baseUrl"]
    document_id = urllib.parse.quote(capability["documentId"], safe="")
    if args.command == "status":
        url = f"{base}/creative_document_agent/v1/documents/{document_id}/status"
        body = None
    elif args.command == "preview":
        if "inspect" not in capability["scopes"]:
            print("refused: capability does not include inspect", file=sys.stderr)
            return 2
        if args.expected_revision < 0:
            print("malformed: expected revision must be non-negative", file=sys.stderr)
            return 4
        url = (f"{base}/creative_document_agent/v1/documents/{document_id}/preview"
               f"?expectedRevision={args.expected_revision}")
        body = None
    else:
        if args.command not in capability["scopes"]:
            print(f"refused: capability does not include {args.command}", file=sys.stderr)
            return 2
        try:
            envelope = _validate_envelope(_read_request(args.request), args.command, capability)
        except InputError as exc:
            print(f"malformed: {exc}", file=sys.stderr)
            return 4
        url = f"{base}/creative_document_agent/v1/documents/{document_id}/commands"
        body = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")

    try:
        status, response_body, response_headers = _request(url, token, method="POST" if body is not None else "GET", body=body,
                                                           maximum=12 * 1024 * 1024 if args.command == "preview" else 4 * 1024 * 1024)
    except InputError as exc:
        print(f"malformed: {exc}", file=sys.stderr)
        return 4
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        message = str(getattr(exc, "reason", "local driver transport failed"))
        if token in message:
            message = message.replace(token, "[REDACTED]")
        print(f"transport: {message}", file=sys.stderr)
        return 3

    if status in {301, 302, 303, 307, 308}:
        print("transport: redirect refused", file=sys.stderr)
        return 3

    if args.command == "preview" and status == 200:
        try:
            if response_headers.get("content-type", "").split(";", 1)[0].lower() != "image/png":
                raise InputError("preview response is not PNG")
            if response_headers.get("x-document-id") != capability["documentId"]:
                raise InputError("preview response document does not match")
            if response_headers.get("x-document-revision") != str(args.expected_revision):
                raise InputError("preview response revision does not match the captured revision")
            if response_headers.get("x-snapshot-token") != f"{capability['documentId']}@{args.expected_revision}":
                raise InputError("preview response snapshot token does not match")
            args.output.write_bytes(response_body)
        except (InputError, OSError) as exc:
            print(f"malformed: {exc}", file=sys.stderr)
            return 4
        _print_json({"status": "preview-saved", "revision": args.expected_revision, "path": str(args.output)}, token)
        return 0

    try:
        result = _strict_json(response_body, 4 * 1024 * 1024)
    except InputError as exc:
        print(f"malformed: invalid JSON response ({exc})", file=sys.stderr)
        return 4
    if _contains(result, token):
        result = _safe_output(result, token)
    _print_json(result, token)
    if status >= 500:
        return 3
    if status >= 400 or (isinstance(result, dict) and result.get("status") in {"error", "conflict", "refused"}):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
