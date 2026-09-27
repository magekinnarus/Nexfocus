"""Stable identifiers and deterministic content identity for creative documents."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import PurePosixPath
from typing import Any


ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._:-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class InvalidIdentity(ValueError):
    """Raised when a durable identifier or digest is malformed."""


def make_id(prefix: str = "id") -> str:
    """Create an opaque, stable identifier suitable for a durable record."""

    if not prefix or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", prefix):
        raise InvalidIdentity(f"invalid id prefix: {prefix!r}")
    return f"{prefix}-{uuid.uuid4().hex}"


new_id = make_id


def validate_id(value: str, *, field: str = "id") -> str:
    if not isinstance(value, str) or not ID_RE.fullmatch(value):
        raise InvalidIdentity(f"{field} must be an opaque durable identifier")
    return value


def validate_sha256(value: str, *, field: str = "sha256") -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value.lower()):
        raise InvalidIdentity(f"{field} must be a lowercase SHA-256 digest")
    return value.lower()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | bytes) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    """Serialize JSON with stable ordering and no transient whitespace."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def content_digest(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def digest_json(value: Any) -> str:
    return content_digest(value)


def safe_relative_asset_path(digest: str, extension: str = "bin") -> str:
    """Return the canonical project-relative path for an embedded asset."""

    validate_sha256(digest)
    extension = extension.lstrip(".").lower() or "bin"
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,15}", extension):
        raise ValueError("asset extension contains unsafe characters")
    prefix = digest[:2]
    return str(PurePosixPath("assets", "sha256", prefix, f"{digest}.{extension}"))
