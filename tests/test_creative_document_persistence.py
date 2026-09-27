from __future__ import annotations

import json
from pathlib import Path

import pytest

from modules.creative_document import (
    AssetHashMismatch,
    AssetStore,
    CorruptProject,
    Document,
    InjectedInterruption,
    LayerRecord,
    ProjectStore,
    RevisionManager,
    TransactionRecord,
)


def test_undo_and_redo_are_new_monotonic_revisions() -> None:
    manager = RevisionManager({"value": 0})
    edit = manager.commit(0, {"value": 1}, actor_id="director", actor_kind="human", command_ids=["cmd-edit"])
    undo = manager.undo(1, actor_id="director")
    redo = manager.redo(2, actor_id="director")
    assert (edit.resulting_revision, undo.resulting_revision, redo.resulting_revision) == (1, 2, 3)
    assert manager.state == {"value": 1}
    assert [record.kind for record in manager.records] == ["edit", "undo", "redo"]


def _document(revision: int = 0, history: list[dict] | None = None) -> Document:
    return Document(
        "doc-persist", 32, 24, current_revision=revision, root_layer_ids=["layer-root"],
        layers={"layer-root": LayerRecord("layer-root", "Scene", "group")}, history=history or [],
    )


def test_asset_store_deduplicates_and_rejects_mismatched_relink(tmp_path: Path) -> None:
    store = AssetStore(tmp_path / "scene.nexscene")
    first = store.put_bytes(b"same", extension="bin")
    second = store.put_bytes(b"same", extension="bin")
    assert first.content_hash == second.content_hash
    assert first.storage_uri == second.storage_uri

    external = tmp_path / "external.bin"
    external.write_bytes(b"expected")
    record = store.external_reference(external)
    with pytest.raises(AssetHashMismatch):
        changed = tmp_path / "changed.bin"
        changed.write_bytes(b"different")
        store.relink(record, changed)
    store.relink(record, external)
    assert record.external_status == "relinked"


def test_pending_candidate_is_not_recovered_but_committed_witness_is(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "scene.nexscene")
    store.save(_document())
    tx = TransactionRecord("txn-1", "group-1", "director", "human", ["cmd-1"], 0, 1, before={"v": 0}, after={"v": 1})
    candidate = _document(1, [tx.to_dict()])
    with pytest.raises(InjectedInterruption):
        store.save(candidate, interrupt_at="after_pending")
    assert store.open().current_revision == 0
    assert list((store.path / "pending").rglob("manifest.json"))

    with pytest.raises(InjectedInterruption):
        store.save(candidate, interrupt_at="after_witness")
    (store.path / "manifest.json").write_text("{corrupt", encoding="utf-8")
    assert store.open().current_revision == 1


def test_corrupt_newest_witness_falls_back_and_repairs_pointer(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "scene.nexscene")
    store.save(_document())
    tx = TransactionRecord("txn-2", "group-2", "director", "human", ["cmd-2"], 0, 1, before={"v": 0}, after={"v": 1})
    document = _document(1, [tx.to_dict()])
    store.save(document)
    (store.path / "revisions" / "1" / "manifest.json").write_text("{bad", encoding="utf-8")
    assert store.open().current_revision == 0
    pointer = json.loads((store.path / "manifest.json").read_text(encoding="utf-8"))
    assert pointer["revision"] == 0


def test_portable_pack_reopens_and_rejects_traversal(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "scene.nexscene")
    store.save(_document())
    archive = store.pack_portable(tmp_path / "scene.zip")
    unpacked = ProjectStore.unpack_portable(archive, tmp_path / "unpacked")
    assert unpacked.open().document_id == "doc-persist"

    import zipfile
    malicious = tmp_path / "malicious.zip"
    with zipfile.ZipFile(malicious, "w") as archive_file:
        archive_file.writestr("../escape.txt", b"no")
    with pytest.raises(Exception):
        ProjectStore.unpack_portable(malicious, tmp_path / "bad")


def test_portable_pack_rejects_embedded_hash_mismatch(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "scene.nexscene")
    asset = store.assets.put_bytes(b"verified", asset_id="asset-1")
    document = _document()
    document.assets[asset.asset_id] = asset
    document.layers["layer-root"].asset_ids = [asset.asset_id]
    store.save(document)
    archive = store.pack_portable(tmp_path / "hash.zip")

    import zipfile
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(tampered, "w") as target:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename.startswith("assets/sha256/"):
                data = b"tampered"
            target.writestr(info, data)
    with pytest.raises(Exception):
        ProjectStore.unpack_portable(tampered, tmp_path / "tampered-out")
