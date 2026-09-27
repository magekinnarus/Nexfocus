"""Native ``.nexscene`` directory projects, recovery, and safe portability."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping

from .asset_store import AssetHashMismatch, AssetStore, MissingAsset
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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(canonical_json(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _json_read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorruptProject(f"cannot read JSON record {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CorruptProject(f"JSON record is not an object: {path}")
    return value


def _safe_member(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ProjectStoreError(f"unsafe archive member path: {name}")
    if ":" in path.parts[0] or any("\x00" in part for part in path.parts):
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
        records = [dict(record) for record in document.history]
        if document.current_revision > 0 and not records:
            raise ProjectStoreError("a nonzero document revision must retain transaction history")
        previous = 0
        refs: list[str] = []
        for record in records:
            if not isinstance(record, dict):
                raise ProjectStoreError("history entries must be JSON objects")
            transaction_id = record.get("transactionId")
            if not isinstance(transaction_id, str):
                raise ProjectStoreError("history record lacks transactionId")
            validate_id(transaction_id, field="transactionId")
            if int(record.get("previousRevision", -1)) != previous:
                raise ProjectStoreError("history chain does not start or advance monotonically")
            resulting = int(record.get("resultingRevision", -1))
            if resulting <= previous:
                raise ProjectStoreError("history resulting revisions must advance")
            previous = resulting
            refs.append(transaction_id)
        if previous != document.current_revision:
            raise ProjectStoreError("history chain does not end at document revision")
        return records

    def _validate_assets(self, document: Document) -> None:
        for asset in document.assets.values():
            if asset.external_uri is not None and asset.external_status != "embedded":
                # A missing external reference is a visible project state, not
                # an invalid scene. Mismatched available bytes still fail.
                if Path(asset.external_uri).exists():
                    self.assets.verify(asset)
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
            transaction_id = str(record["transactionId"])
            resulting_revision = int(record["resultingRevision"])
            path = self.history_dir / f"{resulting_revision}-{transaction_id}.json"
            _json_write(path, record)
            if _json_read(path) != record:
                raise ProjectStoreError(f"history re-read mismatch: {path}")
            refs.append(transaction_id)
        return refs

    def _write_checkpoint(self, document_value: dict[str, Any]) -> str:
        revision = int(document_value["currentRevision"])
        path = self.checkpoints_dir / str(revision) / "manifest.json"
        _json_write(path, document_value)
        if _json_read(path) != document_value:
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
        document.validate()
        self._validate_assets(document)
        records = self._history_records(document)
        # The history-reference list is part of the durable manifest digest,
        # so it must be finalized before the digest is computed.
        document.history_refs = [str(record["transactionId"]) for record in records]
        if checkpoint:
            document.checkpoint_refs = [f"checkpoints/{document.current_revision}/manifest.json"]
        document.refresh_digest()
        value = document.to_dict()
        tx_id = transaction_id or (records[-1]["transactionId"] if records else make_id("save"))
        validate_id(tx_id, field="transactionId")
        witness = self._witness_path(document.current_revision)
        if witness.exists():
            existing = _json_read(witness)
            if existing.get("revisionDigest") == value.get("revisionDigest"):
                self._refresh_pointer(document, witness)
                return SaveResult(document, document.current_revision, tx_id, witness, self.pointer_path, True)
            raise ProjectStoreError(f"revision witness already exists with different content: {witness}")

        self._interrupt(interrupt, "before_assets")
        # Asset publication happens at put/references time. Verification above
        # is deliberately repeated before history publication as the first
        # durable boundary for records that may have been externally changed.
        self._validate_assets(document)
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
        # single same-volume rename with no post-rename rewrite.
        pending_value = value
        self._interrupt(interrupt, "before_pending")
        _json_write(pending_path, pending_value)
        if _json_read(pending_path) != pending_value:
            raise ProjectStoreError("pending candidate re-read mismatch")
        self._interrupt(interrupt, "after_pending")

        self._interrupt(interrupt, "before_witness")
        witness.parent.mkdir(parents=True, exist_ok=True)
        # Same-volume replace of the pending file is the commit witness.
        os.replace(pending_path, witness)
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
        document.validate()
        document.refresh_digest()
        return self.path / self._write_checkpoint(document.to_dict())

    def _validate_history_files(self, document: Document) -> None:
        records = self._history_records(document)
        refs = list(document.history_refs) or [str(record["transactionId"]) for record in records]
        if [str(record["transactionId"]) for record in records] != refs:
            raise CorruptProject("history references do not match the manifest chain")
        loaded = []
        for record in records:
            transaction_id = str(record["transactionId"])
            matches = list(self.history_dir.glob(f"*-{transaction_id}.json"))
            if len(matches) != 1:
                raise CorruptProject(f"missing or ambiguous transaction record: {transaction_id}")
            stored = _json_read(matches[0])
            if stored != record:
                raise CorruptProject(f"transaction record differs from manifest: {transaction_id}")
            loaded.append(stored)

    def _validate_witness(self, path: Path) -> Document:
        raw = _json_read(path)
        try:
            migrated = migrate_manifest(raw)
            document = Document.from_dict(migrated)
        except (MigrationError, SchemaValidationError, ValueError, TypeError) as exc:
            raise CorruptProject(f"invalid retained witness {path}: {exc}") from exc
        if document.current_revision != int(path.parent.name):
            raise CorruptProject(f"witness revision mismatch: {path}")
        self._validate_assets(document)
        self._validate_history_files(document)
        for checkpoint in document.checkpoint_refs:
            checkpoint_member = _safe_member(checkpoint)
            checkpoint_path = self.path / Path(*checkpoint_member.parts)
            if not checkpoint_path.exists():
                raise CorruptProject(f"missing referenced checkpoint: {checkpoint}")
            _json_read(checkpoint_path)
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
            except (CorruptProject, ProjectStoreError):
                # Rejected artifacts are intentionally preserved for inspection.
                continue
            pointer_valid = False
            if self.pointer_path.exists():
                try:
                    pointer = _json_read(self.pointer_path)
                    pointer_valid = (
                        pointer.get("revision") == revision
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
                return Document.from_dict(migrate_manifest(_json_read(path)))
            except (ValueError, TypeError, MigrationError, SchemaValidationError):
                continue
        raise CorruptProject("no valid checkpoint")

    def _portable_document(self, document: Document, *, embed_external: bool) -> Document:
        clone = Document.from_dict(document.to_dict())
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
                target = staging.path / Path(asset.storage_uri.replace("/", os.sep))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                if sha256_file(str(target)) != asset.content_hash:
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
        destination.mkdir(parents=True, exist_ok=True)
        root = destination.resolve()
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            for info in members:
                member = _safe_member(info.filename)
                # Reject Unix symlinks and other link-like entries.
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ProjectStoreError(f"link-like archive member: {info.filename}")
                target = (destination / Path(*member.parts)).resolve()
                if os.path.commonpath([str(root), str(target)]) != str(root):
                    raise ProjectStoreError("archive member escapes destination")
            for info in members:
                member = _safe_member(info.filename)
                target = destination / Path(*member.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
        store = ProjectStore(destination)
        store.open()  # fail closed on malformed manifest or content hash
        return store

    unpack = unpack_portable

    save_scene = save
    open_scene = open
    pack_scene_portable = pack_portable
    unpack_scene_portable = unpack_portable
