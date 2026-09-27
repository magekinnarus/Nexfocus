"""Content-addressed immutable assets and explicit external relinking."""

from __future__ import annotations

import os
import re
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from .ids import make_id, safe_relative_asset_path, sha256_bytes, sha256_file, validate_sha256
from .schema import AssetRecord


class AssetStoreError(ValueError):
    """Base error for asset-store operations."""


class AssetHashMismatch(AssetStoreError):
    """Raised when bytes do not match the expected content identity."""


class MissingAsset(AssetStoreError):
    """Raised when an embedded asset is absent."""


class AssetStore:
    """Store immutable project blobs below ``assets/sha256``.

    The store never replaces a blob at an existing digest with new bytes. A
    same-content put is a deduplicating no-op.
    """

    def __init__(self, project_root: str | os.PathLike[str]) -> None:
        self.project_root = Path(project_root)
        self.asset_root = self.project_root / "assets" / "sha256"
        self.asset_root.mkdir(parents=True, exist_ok=True)

    def _path(self, digest: str, extension: str = "bin") -> Path:
        validate_sha256(digest)
        relative = safe_relative_asset_path(digest, extension)
        return self.project_root / Path(*relative.split("/"))

    def put_bytes(
        self,
        data: bytes,
        *,
        media_type: str = "application/octet-stream",
        extension: str = "bin",
        asset_id: str | None = None,
        width: int | None = None,
        height: int | None = None,
        has_alpha: bool | None = None,
        color_space: str | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> AssetRecord:
        if not isinstance(data, bytes):
            raise TypeError("asset bytes must be bytes")
        digest = sha256_bytes(data)
        path = self._path(digest, extension)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if sha256_file(str(path)) != digest:
                raise AssetHashMismatch(f"immutable asset path is corrupt: {path}")
        else:
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            temporary.write_bytes(data)
            os.replace(temporary, path)
        return AssetRecord(
            asset_id=asset_id or make_id("asset"),
            content_hash=digest,
            media_type=media_type,
            storage_uri=path.relative_to(self.project_root).as_posix(),
            extension=extension.lstrip(".").lower() or "bin",
            byte_length=len(data),
            width=width,
            height=height,
            has_alpha=has_alpha,
            color_space=color_space,
            provenance=dict(provenance or {}),
        )

    def put_file(self, path: str | os.PathLike[str], **kwargs: Any) -> AssetRecord:
        source = Path(path)
        return self.put_bytes(source.read_bytes(), extension=source.suffix.lstrip(".") or "bin", **kwargs)

    def verify(self, asset: AssetRecord) -> bool:
        asset.validate()
        if asset.external_uri is not None and asset.external_status != "embedded":
            path = Path(asset.external_uri)
            if not path.exists():
                return False
            return sha256_file(str(path)) == asset.expected_hash
        relative = PurePosixPath(asset.storage_uri.replace("\\", "/"))
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise AssetStoreError(f"embedded asset URI is not project-relative: {asset.storage_uri}")
        path = self.project_root / Path(*relative.parts)
        if not path.exists():
            raise MissingAsset(asset.storage_uri)
        actual = sha256_file(str(path))
        if actual != asset.content_hash:
            raise AssetHashMismatch(f"asset {asset.asset_id} expected {asset.content_hash}, got {actual}")
        if asset.byte_length is not None and path.stat().st_size != asset.byte_length:
            raise AssetHashMismatch(f"asset {asset.asset_id} byte length changed")
        return True

    def read_bytes(self, asset: AssetRecord) -> bytes:
        self.verify(asset)
        if asset.external_uri is not None and asset.external_status != "embedded":
            return Path(asset.external_uri).read_bytes()
        relative = PurePosixPath(asset.storage_uri.replace("\\", "/"))
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise AssetStoreError(f"embedded asset URI is not project-relative: {asset.storage_uri}")
        return (self.project_root / Path(*relative.parts)).read_bytes()

    def external_reference(
        self,
        path_or_uri: str | os.PathLike[str],
        *,
        media_type: str = "application/octet-stream",
        extension: str = "bin",
        asset_id: str | None = None,
        width: int | None = None,
        height: int | None = None,
        has_alpha: bool | None = None,
        color_space: str | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> AssetRecord:
        path = Path(path_or_uri)
        expected = sha256_file(str(path)) if path.exists() else None
        if expected is None:
            # An external record must still carry the expected identity. The
            # caller may provide it later through ``external_reference_with_hash``.
            raise MissingAsset(str(path))
        return AssetRecord(
            asset_id=asset_id or make_id("asset"),
            content_hash=expected,
            media_type=media_type,
            storage_uri=path.as_posix(),
            extension=extension.lstrip(".").lower() or "bin",
            byte_length=path.stat().st_size,
            width=width,
            height=height,
            has_alpha=has_alpha,
            color_space=color_space,
            provenance=dict(provenance or {}),
            external_uri=path.as_posix(),
            expected_hash=expected,
            external_status="available",
        )

    def external_reference_with_hash(
        self,
        uri: str,
        expected_hash: str,
        *,
        media_type: str = "application/octet-stream",
        extension: str = "bin",
        asset_id: str | None = None,
        width: int | None = None,
        height: int | None = None,
        has_alpha: bool | None = None,
        color_space: str | None = None,
    ) -> AssetRecord:
        validate_sha256(expected_hash, field="expectedHash")
        path = Path(uri)
        available = path.exists() and sha256_file(str(path)) == expected_hash
        return AssetRecord(
            asset_id=asset_id or make_id("asset"),
            content_hash=expected_hash,
            media_type=media_type,
            storage_uri=uri,
            extension=extension.lstrip(".").lower() or "bin",
            byte_length=path.stat().st_size if path.exists() else None,
            width=width,
            height=height,
            has_alpha=has_alpha,
            color_space=color_space,
            external_uri=uri,
            expected_hash=expected_hash,
            external_status="available" if available else "missing",
        )

    def relink(self, asset: AssetRecord, new_path: str | os.PathLike[str]) -> AssetRecord:
        if asset.external_uri is None or asset.expected_hash is None:
            raise AssetStoreError("only external assets can be relinked")
        path = Path(new_path)
        if not path.exists():
            raise MissingAsset(str(path))
        actual = sha256_file(str(path))
        if actual != asset.expected_hash:
            raise AssetHashMismatch(f"relink bytes differ for {asset.asset_id}; create a new asset identity explicitly")
        asset.external_uri = path.as_posix()
        asset.storage_uri = path.as_posix()
        asset.external_status = "relinked"
        asset.byte_length = path.stat().st_size
        return asset

    put = put_bytes
    add_external_reference = external_reference_with_hash
    relink_external = relink


def verify_asset_hash(path: str | os.PathLike[str], expected_hash: str) -> bool:
    return sha256_file(str(path)) == validate_sha256(expected_hash)
