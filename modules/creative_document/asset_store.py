"""Content-addressed immutable assets and explicit external relinking."""

from __future__ import annotations

import os
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Iterator, Sequence

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
        self.project_root = Path(project_root).resolve()
        self.asset_root = self.project_root / "assets" / "sha256"
        self.asset_root.mkdir(parents=True, exist_ok=True)
        self._publication_lock = threading.RLock()

    @contextmanager
    def _exclusive_publication(self) -> Iterator[None]:
        """Serialize a batch across threads and processes sharing a project."""

        lock_path = self.project_root / "assets" / ".publication.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._publication_lock, lock_path.open("a+b") as lock_file:
            if lock_file.tell() == 0:
                lock_file.write(b"0")
                lock_file.flush()
            lock_file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _assert_asset_path(self, path: Path) -> None:
        root = self.project_root.resolve()
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise AssetStoreError("asset path escapes the project root") from exc
        current = root
        for component in relative.parts:
            current = current / component
            if current.is_symlink():
                raise AssetStoreError("asset path contains a link-like component")
        resolved = path.resolve()
        if os.path.commonpath([str(root), str(resolved)]) != str(root):
            raise AssetStoreError("asset path escapes the project root")

    def put_batch(self, entries: Sequence[tuple[bytes, AssetRecord]]) -> list[AssetRecord]:
        """Stage and verify a complete asset batch before publishing any blob.

        If publication fails, remove only target links this call created and
        only while their inode and content still match the staged file. Existing
        content-addressed blobs are never removed.
        """

        if not entries:
            return []
        staged: dict[Path, Path | None] = {}
        records: list[AssetRecord] = []
        published: list[tuple[Path, Path, str, int, int, int]] = []
        temporary_paths: list[Path] = []
        try:
            with self._exclusive_publication():
                # Validate and stage every new byte sequence before publishing.
                for data, record in entries:
                    if not isinstance(data, bytes):
                        raise AssetStoreError("asset bytes must be bytes")
                    record.validate()
                    if sha256_bytes(data) != record.content_hash:
                        raise AssetHashMismatch("pending asset bytes do not match their declared identity")
                    if record.byte_length is not None and len(data) != record.byte_length:
                        raise AssetHashMismatch("pending asset byte length does not match its declared identity")
                    target = self._path(record.content_hash, record.extension)
                    self._assert_asset_path(target)
                    canonical = target.relative_to(self.project_root).as_posix()
                    if record.storage_uri.replace("\\", "/") != canonical:
                        raise AssetStoreError("pending asset URI is not canonical")
                    if target not in staged:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        self._assert_asset_path(target)
                        if target.exists():
                            if target.is_symlink() or sha256_file(str(target)) != record.content_hash:
                                raise AssetHashMismatch("existing immutable asset is corrupt")
                            staged[target] = None
                        else:
                            temporary = target.with_name(f".{target.name}.{os.getpid()}.{make_id('tmp')}.stage")
                            temporary_paths.append(temporary)
                            with temporary.open("xb") as handle:
                                handle.write(data)
                                handle.flush()
                                os.fsync(handle.fileno())
                            if sha256_file(str(temporary)) != record.content_hash or temporary.stat().st_size != len(data):
                                raise AssetHashMismatch("staged immutable asset failed verification")
                            staged[target] = temporary
                    records.append(record)

                for target, temporary in staged.items():
                    if temporary is None:
                        continue
                    temporary_stat = temporary.stat()
                    staged_identity = (sha256_file(str(temporary)), temporary_stat.st_size,
                                       temporary_stat.st_dev, temporary_stat.st_ino)
                    try:
                        os.link(temporary, target)
                    except FileExistsError:
                        if target.is_symlink() or sha256_file(str(target)) != sha256_file(str(temporary)):
                            raise AssetHashMismatch("immutable asset raced with different bytes")
                    else:
                        # Record rollback authority immediately after the
                        # atomic link succeeds, before any verification can
                        # raise and leave an untracked new target behind.
                        published.append((target, temporary, *staged_identity))
                    self._assert_asset_path(target)
                    if not target.is_file() or sha256_file(str(target)) != target.name.split(".", 1)[0]:
                        raise AssetHashMismatch("published immutable asset failed verification")
                    matching = next(record for record in records if self._path(record.content_hash, record.extension) == target)
                    if matching.byte_length is not None and target.stat().st_size != matching.byte_length:
                        raise AssetHashMismatch("published immutable asset byte length changed")
                return records
        except Exception as exc:
            for target, temporary, digest, size, device, inode in reversed(published):
                try:
                    target_stat = target.stat()
                    if (not target.is_symlink() and target_stat.st_dev == device and target_stat.st_ino == inode
                            and target_stat.st_size == size and sha256_file(str(target)) == digest):
                        target.unlink()
                except OSError:
                    pass
            if isinstance(exc, AssetStoreError):
                raise
            raise AssetStoreError("asset batch publication failed") from exc
        finally:
            for temporary in temporary_paths:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

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
        extension = extension.lstrip(".").lower() or "bin"
        if not extension.isascii() or not extension.isalnum():
            raise AssetStoreError("asset extension must be alphanumeric")
        digest = sha256_bytes(data)
        path = self._path(digest, extension)
        record = AssetRecord(
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
        return self.put_batch([(data, record)])[0]

    def put_file(self, path: str | os.PathLike[str], **kwargs: Any) -> AssetRecord:
        source = Path(path)
        return self.put_bytes(source.read_bytes(), extension=source.suffix.lstrip(".") or "bin", **kwargs)

    def verify(self, asset: AssetRecord) -> bool:
        asset.validate()
        if asset.external_uri is not None and asset.external_status != "embedded":
            path = Path(asset.external_uri)
            if not path.exists():
                return False
            return (sha256_file(str(path)) == asset.expected_hash
                    and (asset.byte_length is None or path.stat().st_size == asset.byte_length))
        relative = PurePosixPath(asset.storage_uri.replace("\\", "/"))
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise AssetStoreError(f"embedded asset URI is not project-relative: {asset.storage_uri}")
        expected = PurePosixPath(safe_relative_asset_path(asset.content_hash, asset.extension))
        if relative != expected:
            raise AssetStoreError(f"embedded asset URI is not canonical: {asset.storage_uri}")
        path = self.project_root / Path(*relative.parts)
        root = self.project_root.resolve()
        if os.path.commonpath([str(root), str(path.resolve())]) != str(root):
            raise AssetStoreError(f"embedded asset resolves outside the project: {asset.storage_uri}")
        if any(part.is_symlink() for part in [self.project_root / Path(*relative.parts[:index]) for index in range(1, len(relative.parts) + 1)]):
            raise AssetStoreError(f"embedded asset path contains a link-like component: {asset.storage_uri}")
        if not path.exists():
            raise MissingAsset(asset.storage_uri)
        actual = sha256_file(str(path))
        if actual != asset.content_hash:
            raise AssetHashMismatch(f"asset {asset.asset_id} expected {asset.content_hash}, got {actual}")
        if asset.byte_length is not None and path.stat().st_size != asset.byte_length:
            raise AssetHashMismatch(f"asset {asset.asset_id} byte length changed")
        return True

    def read_bytes(self, asset: AssetRecord) -> bytes:
        if asset.external_uri is not None and asset.external_status != "embedded":
            asset.validate()
            path = Path(asset.external_uri)
            try:
                data = path.read_bytes()
            except FileNotFoundError as exc:
                raise MissingAsset(str(path)) from exc
            except OSError as exc:
                raise AssetStoreError(f"cannot read external asset {asset.asset_id}: {exc}") from exc
            if sha256_bytes(data) != asset.expected_hash:
                raise AssetHashMismatch(f"external asset {asset.asset_id} bytes do not match its expected identity")
            if asset.byte_length is not None and len(data) != asset.byte_length:
                raise AssetHashMismatch(f"external asset {asset.asset_id} byte length does not match its record")
            return data
        self.verify(asset)
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
        exists = path.exists()
        if exists and sha256_file(str(path)) != expected_hash:
            raise AssetHashMismatch(f"external bytes do not match expected identity for {uri}")
        available = exists
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
