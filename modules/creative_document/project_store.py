"""Native ``.nexscene`` directory projects, recovery, and safe portability."""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping

from .asset_store import AssetHashMismatch, AssetStore, MissingAsset
from .history import HistoryValidationError, TransactionRecord, validate_history
from .ids import canonical_json, content_digest, make_id, sha256_file, validate_id
from .migrations import MigrationError, migrate_manifest
from .schema import Document, SchemaValidationError


class ProjectStoreError(RuntimeError):
    """Base project persistence error."""


class CorruptProject(ProjectStoreError):
    """Raised when no retained revision witness validates."""


class InjectedInterruption(ProjectStoreError):
    """Deterministic fault-injection interruption for save tests."""

    def __init__(self, point: str) -> None:
        super().__init__(f"interrupted at {point}")
        self.point = point


@dataclass(frozen=True)
class SaveResult:
    document: Document
    revision: int
    transaction_id: str
    witness_path: Path
    pointer_path: Path
    pointer_refreshed: bool


class FaultInjector:
    """Callable interruption helper; pass ``points`` or a callback."""

    def __init__(self, points: Iterable[str] = ()) -> None:
        self.points = set(points)

    def __call__(self, point: str) -> None:
        if point in self.points:
            raise InjectedInterruption(point)


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{make_id('tmp')}.tmp")
    with temporary.open("wb") as handle:
        handle.write(canonical_json(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _immutable_json_write(path: Path, value: Mapping[str, Any]) -> None:
    """Durably publish canonical JSON once; retained authority is never replaced."""

    encoded = canonical_json(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise ProjectStoreError(f"immutable record already exists with different content: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{make_id('tmp')}.tmp")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        # Hard-link creation is an atomic, no-replace publication on the same
        # volume; Windows/NTFS supports it, and it avoids a race that could
        # otherwise replace an already retained record.
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise ProjectStoreError(f"immutable record raced with different content: {path}")
    finally:
        temporary.unlink(missing_ok=True)
    if path.read_bytes() != encoded:
        raise ProjectStoreError(f"immutable record failed re-read verification: {path}")


def _json_read(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise CorruptProject(f"cannot read JSON record {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CorruptProject(f"JSON record is not an object: {path}")
    return value


def _safe_member(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    raw_parts = normalized[:-1].split("/") if normalized.endswith("/") else normalized.split("/")
    if (path.is_absolute() or any(part in {"", ".", ".."} for part in raw_parts)
            or tuple(raw_parts) != path.parts):
        raise ProjectStoreError(f"unsafe archive member path: {name}")
    if any(":" in part or "\x00" in part for part in path.parts):
        raise ProjectStoreError(f"unsafe archive member path: {name}")
    return path


class ProjectStore:
    """Persistence authority for a single directory-form editable scene."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.revisions_dir = self.path / "revisions"
        self.pending_dir = self.path / "pending"
        self.history_dir = self.path / "history" / "transactions"
        self.checkpoints_dir = self.path / "checkpoints"
        self.revisions_dir.mkdir(parents=True, exist_ok=True)
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.assets = AssetStore(self.path)
        self.pointer_path = self.path / "manifest.json"

    def _interrupt(self, interrupt: Callable[[str], None] | None, point: str) -> None:
        if interrupt is not None:
            interrupt(point)

    def _history_records(self, document: Document) -> list[dict[str, Any]]:
        try:
            records = [TransactionRecord.from_dict(record) for record in document.history]
            return [record.to_dict() for record in validate_history(records, document.current_revision)]
        except (HistoryValidationError, TypeError, ValueError) as exc:
            raise ProjectStoreError(f"invalid durable transaction history: {exc}") from exc

    def _validate_assets(self, document: Document, *, reconcile_missing: bool = False) -> None:
        for asset in document.assets.values():
            if asset.external_uri is not None and asset.external_status != "embedded":
                external_path = Path(asset.external_uri)
                if not external_path.exists():
                    if reconcile_missing:
                        asset.external_status = "missing"
                    continue
                if not self.assets.verify(asset):
                    raise ProjectStoreError(f"external asset is present but does not match its expected identity: {asset.asset_id}")
                if reconcile_missing and asset.external_status == "missing":
                    asset.external_status = "available"
                continue
            try:
                self.assets.verify(asset)
            except (MissingAsset, AssetHashMismatch) as exc:
                raise ProjectStoreError(str(exc)) from exc

    def _witness_path(self, revision: int) -> Path:
        return self.revisions_dir / str(revision) / "manifest.json"

    def _write_history(self, records: list[dict[str, Any]]) -> list[str]:
        refs: list[str] = []
        for record in records:
            transaction = TransactionRecord.from_dict(record)
            transaction_id = transaction.transaction_id
            resulting_revision = transaction.resulting_revision
            path = self.history_dir / f"{resulting_revision}-{transaction_id}.json"
            _immutable_json_write(path, transaction.to_dict())
            if TransactionRecord.from_dict(_json_read(path)).to_dict() != transaction.to_dict():
                raise ProjectStoreError(f"history re-read mismatch: {path}")
            refs.append(transaction_id)
        return refs

    def _write_checkpoint(self, document_value: dict[str, Any]) -> str:
        revision = int(document_value["currentRevision"])
        path = self.checkpoints_dir / str(revision) / "manifest.json"
        checkpoint_body = {
            "checkpointFormat": "nexfocus.nexscene.checkpoint",
            "checkpointVersion": 1,
            "documentId": document_value["documentId"],
            "revision": revision,
            "documentDigest": document_value["revisionDigest"],
            "historyRefs": document_value["historyRefs"],
            "document": document_value,
        }
        checkpoint_value = {**checkpoint_body, "checkpointDigest": content_digest(checkpoint_body)}
        _immutable_json_write(path, checkpoint_value)
        if _json_read(path) != checkpoint_value:
            raise ProjectStoreError(f"checkpoint re-read mismatch: {path}")
        return path.relative_to(self.path).as_posix()

    def save(
        self,
        document: Document,
        *,
        transaction_id: str | None = None,
        checkpoint: bool = False,
        interrupt: Callable[[str], None] | None = None,
        interrupt_at: str | None = None,
    ) -> SaveResult:
        """Publish one validated document revision using the W01 order.

        The supported interruption points are ``before_assets``,
        ``after_assets``, ``before_history``, ``after_history``,
        ``before_pending``, ``after_pending``, ``before_witness``,
        ``after_witness``, ``before_pointer``, and ``after_pointer``.
        """

        if interrupt is None and interrupt_at is not None:
            interrupt = FaultInjector([interrupt_at])
        records = self._history_records(document)
        expected_history_refs = [record["transactionId"] for record in records]
        if document.history_refs and document.history_refs != expected_history_refs:
            raise ProjectStoreError("document history references do not match transaction history")
        # The history-reference list is part of the durable manifest digest,
        # so it must be finalized before the digest is computed.
        document.history_refs = expected_history_refs
        if checkpoint:
            document.checkpoint_refs = [f"checkpoints/{document.current_revision}/manifest.json"]
        elif any(PurePosixPath(path).parts[1:2] != (str(document.current_revision),) for path in document.checkpoint_refs):
            # Checkpoints bind one exact document revision; do not carry a
            # previous revision's checkpoint forward as current authority.
            document.checkpoint_refs = []
        document.validate()
        self._validate_assets(document, reconcile_missing=True)
        document.refresh_digest()
        value = document.to_dict()
        tx_id = transaction_id or (records[-1]["transactionId"] if records else make_id("save"))
        validate_id(tx_id, field="transactionId")
        witness = self._witness_path(document.current_revision)
        if witness.exists():
            existing = _json_read(witness)
            if existing == value:
                self._validate_witness(witness)
                self._refresh_pointer(document, witness)
                return SaveResult(document, document.current_revision, tx_id, witness, self.pointer_path, True)
            raise ProjectStoreError(f"revision witness already exists with different content: {witness}")

        self._interrupt(interrupt, "before_assets")
        # Asset publication happens at put/references time. Verification above
        # is deliberately repeated before history publication as the first
        # durable boundary for records that may have been externally changed.
        self._validate_assets(document, reconcile_missing=True)
        self._interrupt(interrupt, "after_assets")

        self._interrupt(interrupt, "before_history")
        history_refs = self._write_history(records)
        value["historyRefs"] = history_refs
        if checkpoint:
            self._write_checkpoint(value)
        self._interrupt(interrupt, "after_history")

        pending_path = self.pending_dir / tx_id / "manifest.json"
        # The pending manifest is intentionally the same typed manifest that
        # will become the retained witness. Its directory location marks it
        # uncommitted; keeping the bytes identical makes the commit boundary a
        # single same-volume move with no post-publication rewrite.
        pending_value = value
        self._interrupt(interrupt, "before_pending")
        _json_write(pending_path, pending_value)
        if _json_read(pending_path) != pending_value:
            raise ProjectStoreError("pending candidate re-read mismatch")
        self._validate_manifest_records(Document.from_dict(_json_read(pending_path)))
        self._interrupt(interrupt, "after_pending")

        self._interrupt(interrupt, "before_witness")
        witness.parent.mkdir(parents=True, exist_ok=True)
        # Atomic same-volume no-replace publication of the flushed candidate.
        # On Windows, os.rename is atomic and fails if the destination exists.
        # On platforms whose rename overwrites, an atomic hard-link create
        # provides the same no-replace boundary before removing the pending
        # name. An existing witness is never replaced, including under a race.
        try:
            if os.name == "nt":
                os.rename(pending_path, witness)
            else:
                os.link(pending_path, witness)
                pending_path.unlink()
        except FileExistsError as exc:
            raise ProjectStoreError(f"revision witness was concurrently published: {witness}") from exc
        if _json_read(witness) != value:
            raise ProjectStoreError("published witness re-read mismatch")
        self._interrupt(interrupt, "after_witness")

        self._interrupt(interrupt, "before_pointer")
        self._refresh_pointer(document, witness)
        self._interrupt(interrupt, "after_pointer")
        return SaveResult(document, document.current_revision, tx_id, witness, self.pointer_path, True)

    def _refresh_pointer(self, document: Document, witness: Path) -> None:
        pointer = {
            "formatId": document.format_id,
            "schemaVersion": document.schema_version,
            "documentId": document.document_id,
            "revision": document.current_revision,
            "witness": witness.relative_to(self.path).as_posix(),
            "witnessDigest": sha256_file(str(witness)),
        }
        _json_write(self.pointer_path, pointer)
        if _json_read(self.pointer_path) != pointer:
            raise ProjectStoreError("root pointer re-read mismatch")

    def checkpoint(self, document: Document) -> Path:
        records = self._history_records(document)
        expected_refs = [record["transactionId"] for record in records]
        if document.history_refs and document.history_refs != expected_refs:
            raise ProjectStoreError("checkpoint history references do not match transaction history")
        document.history_refs = expected_refs
        document.validate()
        self._validate_assets(document, reconcile_missing=True)
        self._validate_history_files(document)
        document.refresh_digest()
        return self.path / self._write_checkpoint(document.to_dict())

    def _validate_history_files(self, document: Document) -> None:
        records = self._history_records(document)
        refs = list(document.history_refs) or [str(record["transactionId"]) for record in records]
        if [str(record["transactionId"]) for record in records] != refs:
            raise CorruptProject("history references do not match the manifest chain")
        for record in records:
            transaction = TransactionRecord.from_dict(record)
            path = self.history_dir / f"{transaction.resulting_revision}-{transaction.transaction_id}.json"
            try:
                stored = TransactionRecord.from_dict(_json_read(path))
            except (CorruptProject, HistoryValidationError, TypeError, ValueError) as exc:
                raise CorruptProject(f"missing or invalid transaction record: {transaction.transaction_id}") from exc
            if stored.to_dict() != transaction.to_dict():
                raise CorruptProject(f"transaction record differs from manifest: {transaction.transaction_id}")

    def _validate_checkpoint(self, path: Path, expected: Document | None = None) -> Document:
        checkpoint = _json_read(path)
        required = {"checkpointFormat", "checkpointVersion", "documentId", "revision", "documentDigest", "historyRefs", "document", "checkpointDigest"}
        if set(checkpoint) != required:
            raise CorruptProject(f"checkpoint has unknown or missing fields: {path}")
        body = {key: value for key, value in checkpoint.items() if key != "checkpointDigest"}
        try:
            calculated_checkpoint_digest = content_digest(body)
        except (TypeError, ValueError) as exc:
            raise CorruptProject(f"checkpoint payload is not canonical JSON: {path}") from exc
        if (checkpoint["checkpointFormat"] != "nexfocus.nexscene.checkpoint"
                or type(checkpoint["checkpointVersion"]) is not int or checkpoint["checkpointVersion"] != 1
                or type(checkpoint["revision"]) is not int or not isinstance(checkpoint["documentId"], str)
                or not isinstance(checkpoint["documentDigest"], str)
                or not isinstance(checkpoint["historyRefs"], list)
                or not all(isinstance(value, str) for value in checkpoint["historyRefs"])
                or not isinstance(checkpoint["checkpointDigest"], str)
                or calculated_checkpoint_digest != checkpoint["checkpointDigest"]):
            raise CorruptProject(f"checkpoint identity/digest is invalid: {path}")
        expected_relative = PurePosixPath("checkpoints") / str(checkpoint["revision"]) / "manifest.json"
        if path.resolve() != (self.path / Path(*expected_relative.parts)).resolve():
            raise CorruptProject(f"checkpoint path does not match its revision identity: {path}")
        try:
            document = Document.from_dict(migrate_manifest(checkpoint["document"]))
        except (MigrationError, SchemaValidationError, TypeError, ValueError) as exc:
            raise CorruptProject(f"checkpoint document is invalid: {exc}") from exc
        if (document.document_id != checkpoint["documentId"] or document.current_revision != checkpoint["revision"]
                or document.revision_digest != checkpoint["documentDigest"] or document.history_refs != checkpoint["historyRefs"]):
            raise CorruptProject("checkpoint envelope does not match its document/revision/history")
        if expected is not None and (document.document_id != expected.document_id
                or document.current_revision != expected.current_revision
                or document.revision_digest != expected.revision_digest
                or document.history_refs != expected.history_refs):
            raise CorruptProject("referenced checkpoint does not match its retained witness")
        self._validate_manifest_records(document, validate_checkpoints=False)
        return document

    def _validate_manifest_records(self, document: Document, *, validate_checkpoints: bool = True) -> None:
        self._validate_assets(document, reconcile_missing=False)
        self._validate_history_files(document)
        if validate_checkpoints:
            for checkpoint_ref in document.checkpoint_refs:
                checkpoint_member = _safe_member(checkpoint_ref)
                checkpoint_path = self.path / Path(*checkpoint_member.parts)
                if not checkpoint_path.is_file():
                    raise CorruptProject(f"missing referenced checkpoint: {checkpoint_ref}")
                self._validate_checkpoint(checkpoint_path, expected=document)

    def _validate_witness(self, path: Path) -> Document:
        raw = _json_read(path)
        try:
            migrated = migrate_manifest(raw)
            document = Document.from_dict(migrated)
        except (MigrationError, SchemaValidationError, ValueError, TypeError) as exc:
            raise CorruptProject(f"invalid retained witness {path}: {exc}") from exc
        if document.current_revision != int(path.parent.name):
            raise CorruptProject(f"witness revision mismatch: {path}")
        self._validate_manifest_records(document)
        return document

    def _retained_revisions(self) -> list[int]:
        values: list[int] = []
        for entry in self.revisions_dir.iterdir():
            if entry.is_dir() and entry.name.isdigit():
                values.append(int(entry.name))
        return sorted(values, reverse=True)

    def open(self) -> Document:
        for revision in self._retained_revisions():
            witness = self._witness_path(revision)
            if not witness.exists():
                continue
            try:
                document = self._validate_witness(witness)
                previous_statuses = [asset.external_status for asset in document.assets.values()]
                self._validate_assets(document, reconcile_missing=True)
                if previous_statuses != [asset.external_status for asset in document.assets.values()]:
                    document.revision_digest = None
                    document.refresh_digest()
            except (CorruptProject, ProjectStoreError):
                # Rejected artifacts are intentionally preserved for inspection.
                continue
            pointer_valid = False
            if self.pointer_path.exists():
                try:
                    pointer = _json_read(self.pointer_path)
                    pointer_valid = (
                        pointer.get("formatId") == document.format_id
                        and pointer.get("schemaVersion") == document.schema_version
                        and pointer.get("documentId") == document.document_id
                        and pointer.get("revision") == revision
                        and pointer.get("witness") == witness.relative_to(self.path).as_posix()
                        and pointer.get("witnessDigest") == sha256_file(str(witness))
                    )
                except CorruptProject:
                    pointer_valid = False
            if not pointer_valid:
                self._refresh_pointer(document, witness)
            return document
        raise CorruptProject(f"no fully valid retained revision in {self.path}")

    load = open
    recover = open

    def recover_checkpoint(self, revision: int | None = None) -> Document:
        candidates = [revision] if revision is not None else sorted(
            (int(p.name) for p in self.checkpoints_dir.iterdir() if p.is_dir() and p.name.isdigit()), reverse=True
        )
        for candidate in candidates:
            if candidate is None:
                continue
            path = self.checkpoints_dir / str(candidate) / "manifest.json"
            if not path.exists():
                continue
            try:
                return self._validate_checkpoint(path)
            except (CorruptProject, ProjectStoreError, ValueError, TypeError, MigrationError, SchemaValidationError):
                continue
        raise CorruptProject("no valid checkpoint")

    def _portable_document(self, document: Document, *, embed_external: bool) -> Document:
        clone = Document.from_dict(document.to_dict())
        clone.checkpoint_refs = []
        clone.revision_digest = None
        if not embed_external:
            return clone
        for asset in clone.assets.values():
            if asset.external_uri is None or asset.external_status == "embedded":
                continue
            source = Path(asset.external_uri)
            if not source.exists():
                raise MissingAsset(str(source))
            data = source.read_bytes()
            actual = sha256_file(str(source))
            if actual != asset.expected_hash:
                raise AssetHashMismatch(f"cannot embed mismatched external asset {asset.asset_id}")
            # The caller creates the portable store in a new root, so the
            # content hash/identity survives while the absolute source path is
            # removed from the packed manifest.
            asset.external_uri = None
            asset.expected_hash = None
            asset.external_status = None
            asset.storage_uri = f"assets/sha256/{actual[:2]}/{actual}.{asset.extension}"
            asset.content_hash = actual
            asset.byte_length = len(data)
            asset.provenance = {**asset.provenance, "embeddedFromExternal": True}
        return clone

    def pack_portable(self, archive_path: str | os.PathLike[str], *, embed_external: bool = False) -> Path:
        document = self.open()
        archive_path = Path(archive_path)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        external_bytes: dict[str, bytes] = {}
        if embed_external:
            for asset in document.assets.values():
                if asset.external_uri is None or asset.external_status == "embedded":
                    continue
                source = Path(asset.external_uri)
                if not source.exists():
                    raise MissingAsset(str(source))
                data = source.read_bytes()
                if sha256_file(str(source)) != asset.expected_hash:
                    raise AssetHashMismatch(f"cannot embed mismatched external asset {asset.asset_id}")
                external_bytes[asset.asset_id] = data
        with tempfile.TemporaryDirectory(prefix="nexscene-pack-") as temporary:
            staging = ProjectStore(Path(temporary) / "project.nexscene")
            clone = self._portable_document(document, embed_external=embed_external)
            # Copy referenced bytes into the staging asset store, including
            # external bytes only when explicitly requested.
            for asset in clone.assets.values():
                if asset.external_uri is None and not asset.storage_uri.startswith("assets/"):
                    continue
                if asset.external_uri is not None:
                    if not embed_external:
                        continue
                    data = external_bytes[asset.asset_id]
                else:
                    if asset.asset_id in external_bytes:
                        data = external_bytes[asset.asset_id]
                    else:
                        source = self.path / Path(asset.storage_uri.replace("/", os.sep))
                        if not source.exists():
                            raise MissingAsset(str(source))
                        data = source.read_bytes()
                staged_asset = staging.assets.put_bytes(
                    data, media_type=asset.media_type, extension=asset.extension, asset_id=asset.asset_id,
                    width=asset.width, height=asset.height, has_alpha=asset.has_alpha,
                    color_space=asset.color_space, provenance=asset.provenance,
                )
                if staged_asset.content_hash != asset.content_hash or staged_asset.storage_uri != asset.storage_uri:
                    raise AssetHashMismatch(asset.asset_id)
            # A portable package is self-sufficient at the current saved
            # revision. Its retained history is copied through save.
            clone.history_refs = []
            staging.save(clone)
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for file in sorted(staging.path.rglob("*")):
                    if not file.is_file():
                        continue
                    member = file.relative_to(staging.path).as_posix()
                    _safe_member(member)
                    archive.write(file, member)
        return archive_path

    pack = pack_portable

    @staticmethod
    def unpack_portable(archive_path: str | os.PathLike[str], destination: str | os.PathLike[str]) -> "ProjectStore":
        archive_path = Path(archive_path)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ProjectStoreError("portable unpack destination must not already exist")
        with tempfile.TemporaryDirectory(prefix="nexscene-unpack-", dir=destination.parent) as temporary:
            staging = Path(temporary) / "project"
            staging.mkdir()
            with zipfile.ZipFile(archive_path) as archive:
                members = archive.infolist()
                normalized: dict[str, zipfile.ZipInfo] = {}
                files: set[str] = set()
                for info in members:
                    member = _safe_member(info.filename)
                    member_name = "/".join(member.parts)
                    if member_name in normalized:
                        raise ProjectStoreError(f"duplicate archive member: {info.filename}")
                    normalized[member_name] = info
                    mode = (info.external_attr >> 16) & 0xFFFF
                    file_kind = stat.S_IFMT(mode)
                    if file_kind not in {0, stat.S_IFREG, stat.S_IFDIR}:
                        raise ProjectStoreError(f"link-like or special archive member: {info.filename}")
                    if info.is_dir() != (file_kind == stat.S_IFDIR) and file_kind != 0:
                        raise ProjectStoreError(f"archive member type mismatch: {info.filename}")
                    if not info.is_dir():
                        files.add(member_name)
                for member_name in normalized:
                    parts = PurePosixPath(member_name).parts
                    if any("/".join(parts[:index]) in files for index in range(1, len(parts))):
                        raise ProjectStoreError(f"archive file is also a parent directory: {member_name}")
                for member_name, info in normalized.items():
                    target = staging / Path(*PurePosixPath(member_name).parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if info.is_dir():
                        target.mkdir(exist_ok=True)
                        continue
                    with archive.open(info) as source, target.open("xb") as sink:
                        shutil.copyfileobj(source, sink)
            ProjectStore(staging).open()  # validate all records and embedded hashes before publication
            os.rename(staging, destination)
        store = ProjectStore(destination)
        store.open()
        return store

    unpack = unpack_portable

    save_scene = save
    open_scene = open
    pack_scene_portable = pack_portable
    unpack_scene_portable = unpack_portable
