"""Loopback-only W04 transport for the shared creative-document service."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any

import args_manager
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from modules.creative_document.command_service import (
    ActorContext,
    CommandEnvelope,
    CommandServiceError,
    MAX_ENVELOPE_BYTES,
    parse_envelope,
    command_service,
    refusal_receipt,
    strict_json_loads,
)
from modules.creative_document_editor_api import (
    SessionGrant,
    _authorized,
    _handle,
    _invalidate_document_views,
    _principal_id,
    creative_document_runtime,
)
from modules.creative_document.ids import make_id


creative_document_command_router = APIRouter()
MAX_DRIVER_LIFETIME_SECONDS = 60 * 60
MAX_DRIVER_REQUESTS = 4
_driver_work = threading.BoundedSemaphore(MAX_DRIVER_REQUESTS)


@dataclass(frozen=True)
class AgentGrant:
    grant_id: str
    document_id: str
    owner_principal: str
    actor_id: str
    scopes: frozenset[str]
    expires_at: int
    secret_digest: str


@dataclass(frozen=True)
class AgentRequestContext:
    actor: ActorContext
    owner_principal: str
    grant_id: str
    expires_at: int


class AgentGrantRegistry:
    """Ephemeral grants; only SHA-256 secret digests survive in process memory."""

    def __init__(self) -> None:
        self._by_secret: dict[str, AgentGrant] = {}
        self._lock = threading.RLock()

    def issue(
        self,
        *,
        document_id: str,
        owner_principal: str,
        scopes: list[str],
        lifetime_seconds: int,
        base_url: str,
    ) -> dict[str, Any]:
        if not isinstance(scopes, list) or not scopes or any(scope not in {"inspect", "propose", "mutate"} for scope in scopes):
            raise AgentGrantError("INVALID_SCOPES")
        if len(set(scopes)) != len(scopes):
            raise AgentGrantError("INVALID_SCOPES")
        if type(lifetime_seconds) is not int or not 60 <= lifetime_seconds <= MAX_DRIVER_LIFETIME_SECONDS:
            raise AgentGrantError("INVALID_LIFETIME")
        secret = secrets.token_urlsafe(32)
        digest = hashlib.sha256(secret.encode("ascii")).hexdigest()
        now = int(time.time())
        grant = AgentGrant(
            grant_id=make_id("grant"),
            document_id=document_id,
            owner_principal=owner_principal,
            actor_id=make_id("agent"),
            scopes=frozenset(scopes),
            expires_at=now + lifetime_seconds,
            secret_digest=digest,
        )
        with self._lock:
            self._prune()
            # One active grant per document/owner keeps UI status and revoke explicit.
            for key, current in list(self._by_secret.items()):
                if current.document_id == document_id and current.owner_principal == owner_principal:
                    self._by_secret.pop(key, None)
            self._by_secret[digest] = grant
        return {
            "status": "enabled",
            "grantId": grant.grant_id,
            "documentId": grant.document_id,
            "actorId": grant.actor_id,
            "scopes": sorted(grant.scopes),
            "expiresAt": grant.expires_at,
            "capability": {
                "schemaVersion": 1,
                "baseUrl": base_url,
                "documentId": grant.document_id,
                "actorId": grant.actor_id,
                "scopes": sorted(grant.scopes),
                "expiresAt": grant.expires_at,
                "token": secret,
            },
        }

    def status(self, *, document_id: str, owner_principal: str) -> dict[str, Any]:
        with self._lock:
            self._prune()
            grant = next((item for item in self._by_secret.values()
                          if item.document_id == document_id and item.owner_principal == owner_principal), None)
        if grant is None:
            return {"status": "disabled", "grantId": None, "documentId": document_id,
                    "scopes": [], "expiresAt": None, "actorId": None}
        return {"status": "enabled", "grantId": grant.grant_id, "documentId": document_id,
                "scopes": sorted(grant.scopes), "expiresAt": grant.expires_at, "actorId": grant.actor_id}

    def revoke(self, *, document_id: str, owner_principal: str, grant_id: str) -> bool:
        with self._lock:
            self._prune()
            for key, grant in list(self._by_secret.items()):
                if (grant.document_id == document_id and grant.owner_principal == owner_principal
                        and secrets.compare_digest(grant.grant_id, grant_id)):
                    self._by_secret.pop(key, None)
                    return True
        return False

    def authorize(self, request: Request, *, document_id: str, scope: str | None) -> AgentRequestContext:
        _require_local_transport(request)
        authorization = request.headers.get("authorization", "")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise AgentGrantError("AUTHORIZATION_REQUIRED")
        # Browser-origin requests, cookies, or inherited editor capabilities are not Agent auth.
        browser_headers = ("origin", "referer", "cookie", "sec-fetch-site", "sec-fetch-mode", "sec-fetch-dest",
                           "x-gradio-session", "x-editor-owner-key")
        if any(request.headers.get(name) for name in browser_headers):
            raise AgentGrantError("BROWSER_ORIGIN_REFUSED")
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._lock:
            self._prune()
            grant = self._by_secret.get(digest)
        if grant is None:
            raise AgentGrantError("AUTHORIZATION_REQUIRED")
        if grant.document_id != document_id:
            raise AgentGrantError("WRONG_DOCUMENT")
        if scope is not None and scope not in grant.scopes:
            raise AgentGrantError("SCOPE_REQUIRED")
        actor = ActorContext("agent", grant.actor_id, grant.scopes)
        return AgentRequestContext(actor, grant.owner_principal, grant.grant_id, grant.expires_at)

    def _prune(self) -> None:
        now = int(time.time())
        for key, grant in list(self._by_secret.items()):
            if grant.expires_at <= now:
                self._by_secret.pop(key, None)


class AgentGrantError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


agent_grants = AgentGrantRegistry()


def _peer_is_loopback(request: Request) -> bool:
    client = request.client
    if client is None or not client.host:
        return False
    try:
        address = ipaddress.ip_address(client.host.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_loopback


def _configured_local_only() -> bool:
    if bool(getattr(args_manager.args, "share", False)):
        return False
    bind_host = getattr(args_manager.args, "listen", None)
    if not bind_host:
        return True
    host = str(bind_host).strip().strip("[]")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host.lower() == "localhost"
    return address.is_loopback


def _require_local_transport(request: Request) -> None:
    if not _configured_local_only():
        raise AgentGrantError("LOCAL_DRIVER_DISABLED")
    if not _peer_is_loopback(request):
        # Forwarded-client headers are intentionally ignored.
        raise AgentGrantError("LOOPBACK_REQUIRED")


def _local_base_url(request: Request) -> str:
    server = request.scope.get("server")
    if not isinstance(server, (tuple, list)) or len(server) < 2 or type(server[1]) is not int:
        raise AgentGrantError("LOCAL_DRIVER_DISABLED")
    host = request.client.host if request.client is not None else ""
    try:
        address = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError as exc:
        raise AgentGrantError("LOOPBACK_REQUIRED") from exc
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    if not address.is_loopback:
        raise AgentGrantError("LOOPBACK_REQUIRED")
    literal = "[::1]" if isinstance(address, ipaddress.IPv6Address) else "127.0.0.1"
    return f"http://{literal}:{server[1]}"


async def _bounded_json(request: Request, maximum: int = MAX_ENVELOPE_BYTES) -> Any:
    length = request.headers.get("content-length")
    if length is not None:
        try:
            content_length = int(length)
        except ValueError as exc:
            raise CommandServiceError("INVALID_CONTENT_LENGTH") from exc
        if content_length < 0 or content_length > maximum:
            raise CommandServiceError("REQUEST_TOO_LARGE")
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > maximum:
            raise CommandServiceError("REQUEST_TOO_LARGE")
        chunks.append(chunk)
    return strict_json_loads(b"".join(chunks), max_bytes=maximum)


def _contains_secret(value: Any, secret: str) -> bool:
    if isinstance(value, str):
        return bool(secret) and secret in value
    if isinstance(value, dict):
        return any(_contains_secret(key, secret) or _contains_secret(item, secret) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_secret(item, secret) for item in value)
    return False


def _error_receipt(code: str, envelope: CommandEnvelope | None = None, *, actor_id: str | None = None) -> dict[str, Any]:
    mapping = {
        "AUTHORIZATION_REQUIRED": "AUTHORIZATION_REQUIRED",
        "BROWSER_ORIGIN_REFUSED": "AUTHORIZATION_REQUIRED",
        "WRONG_DOCUMENT": "AUTHORIZATION_REQUIRED",
        "LOCAL_DRIVER_DISABLED": "AUTHORIZATION_REQUIRED",
        "LOOPBACK_REQUIRED": "AUTHORIZATION_REQUIRED",
        "SCOPE_REQUIRED": "SCOPE_REQUIRED",
    }
    return refusal_receipt(code=mapping.get(code, code), envelope=envelope,
                           actor_kind="agent", actor_id=actor_id)


@creative_document_command_router.post("/creative_document_api/documents/{document_id}/agent-grants")
async def issue_agent_grant(document_id: str, request: Request) -> Response:
    session = _authorized(request)
    handle = _handle(document_id, request, session)
    try:
        _require_local_transport(request)
        payload = await _bounded_json(request, 4096)
        if not isinstance(payload, dict) or set(payload) != {"scopes", "lifetimeSeconds"}:
            raise AgentGrantError("INVALID_GRANT_REQUEST")
        principal = _principal_id(session, request.headers.get("x-editor-owner-key"))
        grant = agent_grants.issue(
            document_id=document_id,
            owner_principal=principal,
            scopes=payload["scopes"],
            lifetime_seconds=payload["lifetimeSeconds"],
            base_url=_local_base_url(request),
        )
    except (CommandServiceError, AgentGrantError) as exc:
        return JSONResponse({"status": "refused", "code": getattr(exc, "code", "INVALID_GRANT_REQUEST"),
                             "message": "Local driver grant could not be enabled."}, status_code=422,
                            headers={"Cache-Control": "no-store"})
    _ = handle
    return JSONResponse(grant, headers={"Cache-Control": "no-store", "Pragma": "no-cache"})


@creative_document_command_router.get("/creative_document_api/documents/{document_id}/agent-grants/status")
async def get_agent_grant_status(document_id: str, request: Request) -> dict[str, Any]:
    session = _authorized(request)
    _handle(document_id, request, session)
    principal = _principal_id(session, request.headers.get("x-editor-owner-key"))
    return agent_grants.status(document_id=document_id, owner_principal=principal)


@creative_document_command_router.delete("/creative_document_api/documents/{document_id}/agent-grants/{grant_id}")
async def revoke_agent_grant(document_id: str, grant_id: str, request: Request) -> Response:
    session = _authorized(request)
    _handle(document_id, request, session)
    principal = _principal_id(session, request.headers.get("x-editor-owner-key"))
    revoked = agent_grants.revoke(document_id=document_id, owner_principal=principal, grant_id=grant_id)
    return JSONResponse({"status": "revoked" if revoked else "disabled", "documentId": document_id,
                         "grantId": grant_id if revoked else None}, headers={"Cache-Control": "no-store"})


@creative_document_command_router.get("/creative_document_agent/v1/documents/{document_id}/status")
async def agent_status(document_id: str, request: Request) -> Response:
    try:
        context = agent_grants.authorize(request, document_id=document_id, scope=None)
    except AgentGrantError as exc:
        return JSONResponse({"status": "refused", "code": exc.code,
                             "message": "The local driver grant is unavailable."}, status_code=401)
    return JSONResponse({"status": "enabled", "documentId": document_id,
                         "actorKind": "agent", "actorId": context.actor.actor_id,
                         "scopes": sorted(context.actor.scopes), "expiresAt": context.expires_at},
                        headers={"Cache-Control": "no-store"})


@creative_document_command_router.post("/creative_document_agent/v1/documents/{document_id}/commands")
async def agent_command(document_id: str, request: Request) -> Response:
    raw: Any = None
    envelope: CommandEnvelope | None = None
    try:
        raw = await _bounded_json(request)
        envelope = parse_envelope(raw)
    except CommandServiceError as exc:
        return JSONResponse(refusal_receipt(code=exc.code, actor_kind="agent"))
    scope = envelope.intent
    try:
        context = agent_grants.authorize(request, document_id=document_id, scope=scope)
    except AgentGrantError as exc:
        return JSONResponse(_error_receipt(exc.code, envelope), status_code=401,
                            headers={"Cache-Control": "no-store"})
    if _contains_secret(raw, request.headers.get("authorization", "").partition(" ")[2]):
        return JSONResponse(refusal_receipt(code="SECRET_IN_COMMAND", envelope=envelope,
                                            actor_kind="agent", actor_id=context.actor.actor_id),
                            headers={"Cache-Control": "no-store"})
    if envelope.document_id != document_id:
        return JSONResponse(refusal_receipt(code="UNKNOWN_DOCUMENT", envelope=envelope,
                                            actor_kind="agent", actor_id=context.actor.actor_id),
                            headers={"Cache-Control": "no-store"})
    try:
        handle = creative_document_runtime.get(document_id, context.owner_principal)
    except Exception:
        return JSONResponse(refusal_receipt(code="UNKNOWN_DOCUMENT", envelope=envelope,
                                            actor_kind="agent", actor_id=context.actor.actor_id),
                            headers={"Cache-Control": "no-store"})
    if not _driver_work.acquire(blocking=False):
        return JSONResponse(refusal_receipt(code="DRIVER_BUSY", envelope=envelope,
                                            actor_kind="agent", actor_id=context.actor.actor_id,
                                            current_revision=handle.document.current_revision),
                            status_code=429, headers={"Cache-Control": "no-store"})
    try:
        receipt = command_service.execute(handle, raw, context.actor)
    finally:
        _driver_work.release()
    if receipt.get("status") in {"committed", "saved"}:
        _invalidate_document_views(document_id)
    if receipt.get("status") == "saved":
        handle.recovery_notice = "Explicit save checkpoint is available."
    try:
        if len(json.dumps(receipt, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) > 4 * 1024 * 1024:
            return JSONResponse(refusal_receipt(code="RESPONSE_TOO_LARGE", envelope=envelope,
                                                actor_kind="agent", actor_id=context.actor.actor_id,
                                                current_revision=handle.document.current_revision),
                                headers={"Cache-Control": "no-store"})
    except (TypeError, ValueError):
        return JSONResponse(refusal_receipt(code="INTERNAL_ERROR", envelope=envelope,
                                            actor_kind="agent", actor_id=context.actor.actor_id,
                                            current_revision=handle.document.current_revision),
                            headers={"Cache-Control": "no-store"})
    return JSONResponse(receipt, headers={"Cache-Control": "no-store"})


@creative_document_command_router.get("/creative_document_agent/v1/documents/{document_id}/preview")
async def agent_preview(document_id: str, request: Request) -> Response:
    try:
        context = agent_grants.authorize(request, document_id=document_id, scope="inspect")
    except AgentGrantError as exc:
        return JSONResponse({"status": "refused", "code": exc.code,
                             "message": "The local driver grant is unavailable."}, status_code=401,
                            headers={"Cache-Control": "no-store"})
    expected = request.query_params.get("expectedRevision")
    try:
        if expected is None or not expected.isascii() or not expected.isdecimal() or len(expected) > 12:
            raise ValueError
        expected_revision = int(expected)
    except ValueError:
        return JSONResponse({"status": "refused", "code": "INVALID_REVISION",
                             "message": "A captured preview revision is required."}, status_code=400,
                            headers={"Cache-Control": "no-store"})
    try:
        handle = creative_document_runtime.get(document_id, context.owner_principal)
    except Exception:
        return JSONResponse({"status": "refused", "code": "UNKNOWN_DOCUMENT",
                             "message": "Document does not exist."}, status_code=404,
                            headers={"Cache-Control": "no-store"})
    if not _driver_work.acquire(blocking=False):
        return JSONResponse({"status": "refused", "code": "DRIVER_BUSY",
                             "message": "The local driver is at its work limit."}, status_code=429,
                            headers={"Cache-Control": "no-store"})
    try:
        image, revision, _digest = command_service.preview(handle, expected_revision=expected_revision)
    except CommandServiceError as exc:
        return JSONResponse({"status": "conflict" if exc.code == "STALE_DOCUMENT_REVISION" else "refused",
                             "code": exc.code,
                             "message": "The preview revision is stale." if exc.code == "STALE_DOCUMENT_REVISION" else "The preview is unavailable.",
                             "currentRevision": handle.document.current_revision,
                             "writes": 0}, status_code=409 if exc.code == "STALE_DOCUMENT_REVISION" else 422,
                            headers={"Cache-Control": "no-store"})
    finally:
        _driver_work.release()
    return Response(image, media_type="image/png", headers={
        "X-Document-ID": document_id,
        "X-Document-Revision": str(revision),
        "X-Snapshot-Token": f"{document_id}@{revision}",
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
    })
