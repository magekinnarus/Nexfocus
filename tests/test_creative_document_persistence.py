from __future__ import annotations

import json
from pathlib import Path
import stat
import zipfile
import copy

import pytest

from modules.creative_document import (
    AffineTransform,
    AssetHashMismatch,
    AssetStore,
    BBOperation,
    BBox,
    CandidateRecord,
    CorruptProject,
    CoordinateTransform,
    DepthComposite,
    Document,
    ExtractionDerivative,
    HandoffNote,
    InjectedInterruption,
    LayerRecord,
    MaskRecord,
    MissingAsset,
    ProjectStore,
    ProjectStoreError,
    RevisionManager,
    TransactionRecord,
)
from modules.creative_document.history import HistoryValidationError, validate_history


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
        layers={"layer-root": LayerRecord("layer-root", "Scene", "group", revision=revision)}, history=history or [],
    )


def _history_to(revision: int) -> list[dict]:
    return [
        TransactionRecord(f"txn-{step}", f"group-{step}", "director", "director", [f"cmd-{step}"],
                          step - 1, step, before={"revision": step - 1}, after={"revision": step}).to_dict()
        for step in range(1, revision + 1)
    ]


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


def test_retained_transaction_records_are_immutable_and_identical_saves_are_idempotent(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "immutable.nexscene")
    store.save(_document())
    history = _history_to(1)
    revision_one = _document(1, history)
    store.save(revision_one)
    record_path = store.path / "history" / "transactions" / "1-txn-1.json"
    original = record_path.read_bytes()

    store.save(revision_one)
    assert record_path.read_bytes() == original
    tampered = json.loads(original)
    tampered["after"] = {"revision": "rewritten"}
    candidate_history = [tampered, *_history_to(2)[1:]]
    candidate = _document(2, candidate_history)
    with pytest.raises(ProjectStoreError, match="immutable record"):
        store.save(candidate, interrupt_at="after_history")
    assert record_path.read_bytes() == original
    assert store.open().current_revision == 1


@pytest.mark.parametrize(
    ("point", "expected_revision"),
    [
        ("before_assets", 0), ("after_assets", 0), ("before_history", 0), ("after_history", 0),
        ("before_pending", 0), ("after_pending", 0), ("before_witness", 0),
        ("after_witness", 1), ("before_pointer", 1), ("after_pointer", 1),
    ],
)
def test_each_save_interruption_recovers_the_correct_commit_boundary(tmp_path: Path, point: str, expected_revision: int) -> None:
    store = ProjectStore(tmp_path / point / "scene.nexscene")
    store.save(_document())
    candidate = _document(1, _history_to(1))
    with pytest.raises(InjectedInterruption):
        store.save(candidate, interrupt_at=point)
    assert store.open().current_revision == expected_revision


def test_external_asset_missing_reconciles_but_present_mismatch_fails_closed(tmp_path: Path) -> None:
    external_path = tmp_path / "source.bin"
    external_path.write_bytes(b"expected bytes")
    store = ProjectStore(tmp_path / "external.nexscene")
    asset = store.assets.external_reference(external_path, asset_id="asset-external")
    document = _document()
    document.assets[asset.asset_id] = asset
    document.layers["layer-root"].asset_ids = [asset.asset_id]
    store.save(document)

    external_path.unlink()
    reopened = store.open()
    assert reopened.assets[asset.asset_id].external_status == "missing"
    assert reopened.assets[asset.asset_id].content_hash == asset.content_hash

    external_path.write_bytes(b"different bytes")
    with pytest.raises(CorruptProject, match="fully valid retained revision"):
        store.open()


def test_external_mismatch_cannot_be_saved_and_correct_relink_is_hash_safe(tmp_path: Path) -> None:
    original = tmp_path / "original.bin"
    wrong = tmp_path / "wrong.bin"
    original.write_bytes(b"the exact bytes")
    wrong.write_bytes(b"different bytes")
    store = ProjectStore(tmp_path / "relink.nexscene")
    record = store.assets.external_reference(original, asset_id="asset-relink")
    document = _document()
    document.assets[record.asset_id] = record
    document.layers["layer-root"].asset_ids = [record.asset_id]
    original.write_bytes(b"tampered")
    with pytest.raises(ProjectStoreError, match="does not match"):
        store.save(document)
    with pytest.raises(AssetHashMismatch):
        store.assets.relink(record, wrong)
    original.write_bytes(b"the exact bytes")
    store.assets.relink(record, original)
    assert record.external_status == "relinked"


def test_checkpoint_envelope_is_bound_to_witness_and_recovery_is_non_authoritative(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "checkpoint.nexscene")
    document = _document()
    store.save(document, checkpoint=True)
    saved_path = store.pointer_path.read_bytes()
    recovered = store.recover_checkpoint(0)
    assert recovered.current_revision == 0
    assert store.open().current_revision == 0
    assert store.pointer_path.read_bytes() == saved_path

    checkpoint_path = store.checkpoints_dir / "0" / "manifest.json"
    checkpoint_path.write_text("{}", encoding="utf-8")
    with pytest.raises(CorruptProject):
        store.open()


def test_newest_bad_checkpoint_falls_back_to_previous_fully_valid_witness(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "checkpoint-fallback.nexscene")
    store.save(_document())
    store.save(_document(1, _history_to(1)), checkpoint=True)
    (store.checkpoints_dir / "1" / "manifest.json").write_text("{}", encoding="utf-8")
    assert store.open().current_revision == 0


def test_portable_unpack_embeds_external_bytes_and_publishes_only_after_validation(tmp_path: Path) -> None:
    external_path = tmp_path / "external-image.bin"
    external_path.write_bytes(b"portable external bytes")
    store = ProjectStore(tmp_path / "portable-source.nexscene")
    asset = store.assets.external_reference(external_path, asset_id="asset-portable")
    document = _document()
    document.assets[asset.asset_id] = asset
    document.layers["layer-root"].asset_ids = [asset.asset_id]
    store.save(document)
    archive = store.pack_portable(tmp_path / "portable.zip", embed_external=True)
    external_path.unlink()
    unpacked = ProjectStore.unpack_portable(archive, tmp_path / "portable-unpacked.nexscene")
    restored = unpacked.open().assets[asset.asset_id]
    assert restored.external_uri is None
    assert unpacked.assets.read_bytes(restored) == b"portable external bytes"

    invalid = tmp_path / "invalid.zip"
    with zipfile.ZipFile(invalid, "w") as zip_file:
        zip_file.writestr("../escape", b"bad")
    failed_destination = tmp_path / "failed-unpack.nexscene"
    with pytest.raises(ProjectStoreError):
        ProjectStore.unpack_portable(invalid, failed_destination)
    assert not failed_destination.exists()


@pytest.mark.parametrize("member", ["/absolute.txt", "C:/absolute.txt", "nested/../escape.txt", "nested/./file.txt"])
def test_portable_unpack_rejects_absolute_and_noncanonical_paths_without_mutation(tmp_path: Path, member: str) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr(member, b"bad")
    destination = tmp_path / "unsafe-destination.nexscene"
    with pytest.raises(ProjectStoreError):
        ProjectStore.unpack_portable(archive, destination)
    assert not destination.exists()


def test_portable_unpack_rejects_link_like_members_without_mutation(tmp_path: Path) -> None:
    archive = tmp_path / "link.zip"
    link = zipfile.ZipInfo("link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr(link, "target")
    destination = tmp_path / "link-destination.nexscene"
    with pytest.raises(ProjectStoreError, match="link-like"):
        ProjectStore.unpack_portable(archive, destination)
    assert not destination.exists()


def test_complete_edit_undo_redo_history_reopens_with_reversible_material(tmp_path: Path) -> None:
    manager = RevisionManager({"value": 0})
    manager.commit(0, {"value": 1}, actor_id="director", actor_kind="human", command_ids=["cmd-edit"])
    manager.undo(1, actor_id="director")
    manager.redo(2, actor_id="director")
    document = _document(3, [record.to_dict() for record in manager.records])
    store = ProjectStore(tmp_path / "history-roundtrip.nexscene")
    store.save(document)
    reopened = store.open()
    assert [record["kind"] for record in reopened.history] == ["edit", "undo", "redo"]
    assert all(record["before"] is not None and record["after"] is not None for record in reopened.history)
    assert reopened.history_refs == [record["transactionId"] for record in reopened.history]


def test_transaction_parser_rejects_incomplete_typed_and_nonreversible_records() -> None:
    valid = _history_to(1)[0]
    mutations = [
        lambda value: value.pop("actorId"),
        lambda value: value.update(extra="not-allowed"),
        lambda value: value.update(actorId=42),
        lambda value: value.update(previousRevision=True),
        lambda value: value.update(resultingRevision=2),
        lambda value: value.update(commandIds=[]),
        lambda value: value.update(kind="unknown-kind"),
        lambda value: value.update(before=None, after=None, delta=None),
    ]
    for mutate in mutations:
        candidate = copy.deepcopy(valid)
        mutate(candidate)
        with pytest.raises(HistoryValidationError):
            TransactionRecord.from_dict(candidate)

    duplicate = TransactionRecord.from_dict(valid)
    second = TransactionRecord.from_dict(_history_to(2)[1])
    second.transaction_id = duplicate.transaction_id
    with pytest.raises(HistoryValidationError, match="duplicate transaction"):
        validate_history([duplicate, second], 2)


def test_corrupt_referenced_transaction_invalidates_newest_revision_and_falls_back(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "transaction-fallback.nexscene")
    store.save(_document())
    store.save(_document(1, _history_to(1)))
    record_path = store.path / "history" / "transactions" / "1-txn-1.json"
    record_path.write_text("{}", encoding="utf-8")
    assert store.open().current_revision == 0


@pytest.mark.parametrize("point", ["before_assets", "after_assets"])
def test_asset_publication_boundary_keeps_new_blob_durable_but_uncommitted(tmp_path: Path, point: str) -> None:
    store = ProjectStore(tmp_path / point / "assets.nexscene")
    store.save(_document())
    asset = store.assets.put_bytes(b"new immutable bytes", asset_id="asset-new")
    candidate = _document(1, _history_to(1))
    candidate.assets[asset.asset_id] = asset
    candidate.layers["layer-root"].asset_ids = [asset.asset_id]
    with pytest.raises(InjectedInterruption):
        store.save(candidate, interrupt_at=point)
    assert store.assets.verify(asset)
    assert store.open().current_revision == 0


def _candidate_document(asset, *, current_revision: int = 0, history: list[dict] | None = None) -> Document:
    placement = CoordinateTransform.from_forward("document", "native-bb", (10, 10), (10, 10), AffineTransform.identity())
    crop_to_native = CoordinateTransform.from_forward("source-crop", "native-bb", (10, 10), (10, 10), AffineTransform.identity(), 1)
    native_to_crop = CoordinateTransform.from_forward("native-bb", "source-crop", (10, 10), (10, 10), AffineTransform.identity(), 1)
    plate = LayerRecord("plate-layer", "Complete plate", "raster", asset_ids=[asset.asset_id])
    depth = LayerRecord("depth-layer", "Registered depth element", "depth", depth_element=True,
                        complete_plate_id="plate-layer", visible=False, revision=current_revision)
    context = MaskRecord("context-mask", "context", "source-crop", 0, asset.asset_id,
                         "bb-operation", content_hash=asset.content_hash)
    matte = MaskRecord("extraction-matte", "extraction", "native-bb", 0, asset.asset_id,
                       "bb-operation", content_hash=asset.content_hash)
    composite = DepthComposite("depth-composite", 0, [{"layerId": "plate-layer", "revision": 0}],
                               asset.asset_id, asset.content_hash, "plate-layer")
    candidate = CandidateRecord("candidate-context", "full_crop_contextual", "bb-operation", 0, True, None,
                                placement, asset_id=asset.asset_id)
    extraction = ExtractionDerivative("rgba-derivative", candidate.candidate_id, "bb-operation",
                                      matte.mask_id, placement, 0, asset.asset_id,
                                      {"blend": "preserve-source-alpha", "stitch": "registered"})
    operation = BBOperation(
        "bb-operation", 2, "doc-persist", 0, composite.composite_id,
        [{"layerId": "plate-layer", "revision": 0}], context.mask_id, asset.content_hash,
        BBox(0, 10, 0, 10), (10, 10), (10, 10), crop_to_native, native_to_crop,
        asset.asset_id, candidate_ids=[candidate.candidate_id], extraction_ids=[extraction.derivative_id],
        operation_revision=1,
    )
    return Document("doc-persist", 10, 10, current_revision=current_revision,
                    root_layer_ids=[plate.layer_id, depth.layer_id], layers={plate.layer_id: plate, depth.layer_id: depth},
                    assets={asset.asset_id: asset}, masks={context.mask_id: context, matte.mask_id: matte},
                    depth_composites={composite.composite_id: composite}, bb_operations={operation.operation_id: operation},
                    candidates={candidate.candidate_id: candidate}, extractions={extraction.derivative_id: extraction},
                    history=history or [])


def test_candidate_extraction_and_depth_plate_hide_show_survive_project_save_open(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "scene-relationships.nexscene")
    asset = store.assets.put_bytes(b"small deterministic image bytes", asset_id="asset-scene", extension="bin")
    document = _candidate_document(asset)
    store.save(document)
    hidden = store.open()
    assert hidden.layers["depth-layer"].visible is False
    assert hidden.layers["depth-layer"].complete_plate_id == "plate-layer"
    assert hidden.candidates["candidate-context"].kind == "full_crop_contextual"
    assert hidden.extractions["rgba-derivative"].parent_candidate_id == "candidate-context"
    assert hidden.extractions["rgba-derivative"].placement_transform.to_dict() == hidden.candidates["candidate-context"].placement_transform.to_dict()

    shown = copy.deepcopy(hidden)
    shown.current_revision = 1
    shown.history = _history_to(1)
    shown.layers["depth-layer"].revision = 1
    shown.layers["depth-layer"].visible = True
    store.save(shown)
    reopened = store.open()
    assert reopened.layers["depth-layer"].visible is True
    assert reopened.layers["depth-layer"].complete_plate_id == "plate-layer"
    assert reopened.layers["plate-layer"].asset_ids == [asset.asset_id]
    assert reopened.candidates["candidate-context"].candidate_id != reopened.extractions["rgba-derivative"].derivative_id
