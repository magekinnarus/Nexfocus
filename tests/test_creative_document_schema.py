from __future__ import annotations

import json
import copy

import pytest

from modules.creative_document import (
    AffineTransform,
    AssetRecord,
    BBOperation,
    BBox,
    CandidateRecord,
    CollaborationState,
    CoordinateTransform,
    CorrectivePatch,
    DepthComposite,
    Document,
    ExternalRoundTrip,
    ExtractionDerivative,
    GuideLifecycle,
    GuideRecord,
    HandoffNote,
    InteractionGroup,
    LayerRecord,
    MaskRecord,
    ObjectRecord,
    OperationRecord,
    PrivateProxy,
    SchemaValidationError,
    SelectionRevisionConflict,
    SelectionRecord,
    VariantSet,
)
from modules.creative_document.history import TransactionRecord
from modules.creative_document.ids import content_digest


def _transform() -> CoordinateTransform:
    return CoordinateTransform.from_forward(
        "document", "native-bb", (100, 100), (50, 50), AffineTransform.scale(0.5)
    )


def _complete_fixture() -> Document:
    transform = _transform()
    asset = AssetRecord("asset-bg", "0" * 64, "image/png", "assets/sha256/00/" + "0" * 64 + ".png", extension="png", byte_length=4)
    layer_root = LayerRecord("layer-root", "Scene", "group", child_ids=["layer-subject"], asset_ids=["asset-bg"])
    layer_subject = LayerRecord("layer-subject", "Subject", "raster", parent_id="layer-root", object_ids=["obj-guide"], mask_ids=["mask-context"], depth_element=True, complete_plate_id="layer-root")
    obj = ObjectRecord("obj-guide", "layer-subject", "path", geometry={"points": [[1, 2], [3, 4]]})
    context = MaskRecord("mask-context", "context", "document", 0, asset_id="asset-bg", owner_id="layer-subject", content_hash="0" * 64)
    matte = MaskRecord("mask-matte", "extraction", "native-bb", 0, asset_id="asset-bg", owner_id="bb-op")
    composite = DepthComposite("cmp-1", 0, [{"layerId": "layer-root", "revision": 0}], content_asset_id="asset-bg", content_hash="0" * 64, complete_plate_layer_id="layer-root")
    crop_transform = CoordinateTransform.from_forward(
        "source-crop", "native-bb", (50, 50), (50, 50), AffineTransform.identity(), 1
    )
    inverse_transform = CoordinateTransform.from_forward(
        "native-bb", "source-crop", (50, 50), (50, 50), AffineTransform.identity(), 1
    )
    bb = BBOperation(
        "bb-op", 2, "doc-1", 0, "cmp-1", [{"layerId": "layer-root", "revision": 0}], "mask-context", "0" * 64,
        BBox(0, 50, 0, 50), (50, 50), (50, 50), crop_transform, inverse_transform, "asset-bg",
        candidate_ids=["candidate-1"], extraction_ids=["derivative-1"], operation_revision=1,
    )
    candidate = CandidateRecord("candidate-1", "full_crop_contextual", "bb-op", 0, True, None, transform, asset_id="asset-bg")
    derivative = ExtractionDerivative("derivative-1", "candidate-1", "bb-op", "mask-matte", transform, 0, "asset-bg", {"edge": "feather"})
    selection = SelectionRecord("selection-1", "layer-subject", 0, "0" * 64, 1, "current", "mask-context")
    proxy = PrivateProxy("proxy-1", "asset-bg", "0" * 64, "slot-1", transform)
    guide = GuideRecord("guide-1", "Focal anchor", lifecycle=GuideLifecycle.PROPOSED.value)
    document = Document(
        "doc-1", 100, 100, root_layer_ids=["layer-root"],
        layers={"layer-root": layer_root, "layer-subject": layer_subject},
        objects={"obj-guide": obj}, assets={"asset-bg": asset},
        masks={"mask-context": context, "mask-matte": matte}, selections={"selection-1": selection},
        depth_composites={"cmp-1": composite}, bb_operations={"bb-op": bb},
        candidates={"candidate-1": candidate}, extractions={"derivative-1": derivative},
        guides={"guide-1": guide}, private_proxies={"proxy-1": proxy},
    )
    return document


def _reference_matrix_fixture() -> Document:
    document = _complete_fixture()
    document.variants["variant-1"] = VariantSet(
        "variant-1", "subject", ["layer-subject"], "layer-subject"
    )
    document.interaction_groups["interaction-1"] = InteractionGroup(
        "interaction-1", ["layer-subject"], _transform(), [{"above": "layer-root"}], "candidate-1"
    )
    document.operations["op-parent"] = OperationRecord(
        "op-parent", "import", 2, [{"kind": "layer", "id": "layer-root"}],
        produced_ids=["patch-1"], child_ids=["op-child"], actor_id="director",
    )
    document.operations["op-child"] = OperationRecord(
        "op-child", "refine", 2, [{"recordType": "object", "recordId": "obj-guide"}],
        parent_ids=["op-parent"], actor_id="director",
    )
    exchange = ExternalRoundTrip(
        "exchange-1", document.document_id, ["patch-1"], 0, "document", None, None,
        _transform(), "0" * 64,
    )
    document.external_round_trips[exchange.exchange_id] = exchange
    document.patches["patch-1"] = CorrectivePatch(
        "patch-1", "layer-root", AffineTransform.identity(), 0,
        mask_id="mask-context", external_exchange_id="exchange-1",
    )
    return document


def test_nontrivial_document_round_trips_and_digest_is_canonical() -> None:
    document = _complete_fixture()
    first = document.to_dict()
    second = json.loads(json.dumps(first))
    assert Document.from_dict(second).to_dict() == first
    assert document.canonical_digest() == Document.from_dict(second).canonical_digest()


@pytest.mark.parametrize(
    ("case", "mutate"),
    [
        ("depth-composite asset/hash", lambda raw: raw["depthComposites"]["cmp-1"].update(contentHash="1" * 64)),
        ("BB source document", lambda raw: raw["bbOperations"]["bb-op"].update(sourceDocumentId="other-document")),
        ("BB layer manifest", lambda raw: raw["bbOperations"]["bb-op"]["sourceLayerManifest"][0].update(layerId="missing-layer")),
        ("BB guidance", lambda raw: raw["bbOperations"]["bb-op"].update(guidanceIds=["missing-guide"])),
        ("candidate operation", lambda raw: raw["candidates"]["candidate-1"].update(operationId="missing-operation")),
        ("extraction matte", lambda raw: raw["extractions"]["derivative-1"].update(extractionMatteId="missing-mask")),
        ("private proxy hash", lambda raw: raw["privateProxies"]["proxy-1"].update(proxyContentHash="1" * 64)),
        ("guide replacement", lambda raw: raw["guides"]["guide-1"].update(lifecycle="replacement-pending", replacementId="missing-guide")),
        ("selection source", lambda raw: raw["selections"]["selection-1"].update(sourceId="missing-source")),
    ],
)
def test_document_referential_matrix_rejects_dangling_or_inconsistent_ids(case, mutate) -> None:
    raw = copy.deepcopy(_complete_fixture().to_dict())
    mutate(raw)
    raw.pop("revisionDigest", None)
    raw["revisionDigest"] = content_digest(raw)
    with pytest.raises(SchemaValidationError):
        Document.from_dict(raw)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: setattr(doc.depth_composites["cmp-1"], "content_asset_id", "missing-asset"),
        lambda doc: setattr(doc.depth_composites["cmp-1"], "complete_plate_layer_id", "missing-layer"),
        lambda doc: doc.depth_composites["cmp-1"].layer_manifest[0].update(layerId="missing-layer"),
        lambda doc: setattr(doc.bb_operations["bb-op"], "source_composite_id", "missing-composite"),
        lambda doc: setattr(doc.bb_operations["bb-op"], "context_mask_id", "missing-mask"),
        lambda doc: setattr(doc.bb_operations["bb-op"], "native_bb_asset_id", "missing-asset"),
        lambda doc: setattr(doc.bb_operations["bb-op"], "generation_mask_id", "missing-mask"),
        lambda doc: setattr(doc.bb_operations["bb-op"], "blend_mask_id", "missing-mask"),
        lambda doc: setattr(doc.bb_operations["bb-op"], "guidance_ids", ["missing-guide"]),
        lambda doc: setattr(doc.bb_operations["bb-op"], "candidate_ids", ["missing-candidate"]),
        lambda doc: setattr(doc.bb_operations["bb-op"], "extraction_ids", ["missing-derivative"]),
        lambda doc: setattr(doc.selections["selection-1"], "mask_id", "missing-mask"),
        lambda doc: setattr(doc.selections["selection-1"], "derived_layer_ids", ["missing-layer"]),
        lambda doc: setattr(doc.objects["obj-guide"], "layer_id", "layer-root"),
        lambda doc: setattr(doc.masks["mask-context"], "owner_id", "layer-root"),
        lambda doc: setattr(doc.variants["variant-1"], "member_ids", ["missing-member"]),
        lambda doc: setattr(doc.interaction_groups["interaction-1"], "member_layer_ids", ["missing-layer"]),
        lambda doc: setattr(doc.interaction_groups["interaction-1"], "parent_candidate_id", "missing-candidate"),
        lambda doc: setattr(doc.candidates["candidate-1"], "blend_mask_id", "missing-mask"),
        lambda doc: setattr(doc.candidates["candidate-1"], "asset_id", "missing-asset"),
        lambda doc: setattr(doc.candidates["candidate-1"], "parent_candidate_id", "missing-candidate"),
        lambda doc: setattr(doc.extractions["derivative-1"], "operation_id", "missing-operation"),
        lambda doc: setattr(doc.extractions["derivative-1"], "extraction_matte_id", "missing-mask"),
        lambda doc: setattr(doc.extractions["derivative-1"], "rgba_asset_id", "missing-asset"),
        lambda doc: setattr(doc.guides["guide-1"], "object_ids", ["missing-object"]),
        lambda doc: setattr(doc.patches["patch-1"], "parent_id", "missing-parent"),
        lambda doc: setattr(doc.patches["patch-1"], "mask_id", "missing-mask"),
        lambda doc: setattr(doc.patches["patch-1"], "external_exchange_id", "missing-exchange"),
        lambda doc: setattr(doc.external_round_trips["exchange-1"], "document_id", "another-document"),
        lambda doc: setattr(doc.external_round_trips["exchange-1"], "target_ids", ["missing-target"]),
        lambda doc: setattr(doc.external_round_trips["exchange-1"], "exported_revision", 1),
        lambda doc: setattr(doc.external_round_trips["exchange-1"], "export_asset_hash", "1" * 64),
        lambda doc: setattr(doc.private_proxies["proxy-1"], "proxy_asset_id", "missing-asset"),
        lambda doc: setattr(doc.private_proxies["proxy-1"], "coverage_matte_id", "missing-mask"),
        lambda doc: setattr(doc.operations["op-child"], "input_refs", [{"kind": "asset", "id": "missing-asset"}]),
        lambda doc: setattr(doc.operations["op-parent"], "produced_ids", ["missing-record"]),
        lambda doc: setattr(doc.operations["op-child"], "parent_ids", ["missing-operation"]),
        lambda doc: setattr(doc.operations["op-parent"], "child_ids", ["missing-operation"]),
    ],
)
def test_full_referential_matrix_rejects_dangling_ownership_and_lineage(mutate) -> None:
    document = _reference_matrix_fixture()
    mutate(document)
    with pytest.raises(SchemaValidationError):
        document.validate()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw.update(width=True),
        lambda raw: raw["layers"]["layer-subject"].update(visible="false"),
        lambda raw: raw["bbOperations"]["bb-op"].update(sourceRevision=False),
        lambda raw: raw["selections"]["selection-1"].update(seedGeometry="not-an-array"),
        lambda raw: raw["metadata"].update(note=float("nan")),
    ],
)
def test_schema_rejects_incompatible_primitives_and_non_json_metadata(mutate) -> None:
    raw = copy.deepcopy(_complete_fixture().to_dict())
    raw.pop("revisionDigest", None)
    mutate(raw)
    with pytest.raises(SchemaValidationError):
        Document.from_dict(raw)


def test_unknown_fields_dangling_refs_and_cycles_fail_closed() -> None:
    document = Document("doc-1", 20, 20, root_layer_ids=["root"], layers={"root": LayerRecord("root", "Root", "group")})
    raw = document.to_dict()
    raw["unexpected"] = True
    with pytest.raises(SchemaValidationError):
        Document.from_dict(raw)

    cyclic = Document("doc-2", 20, 20, root_layer_ids=["root"], layers={
        "root": LayerRecord("root", "Root", "group", child_ids=["child"]),
        "child": LayerRecord("child", "Child", "group", parent_id="root", child_ids=["root"]),
    })
    with pytest.raises(SchemaValidationError):
        cyclic.validate()


def test_guide_proposal_requires_explicit_lifecycle_transition() -> None:
    guide = GuideRecord("guide-1", "Anchor")
    assert guide.lifecycle == GuideLifecycle.PROPOSED
    guide.transition(GuideLifecycle.ACTIVE.value, 2)
    assert guide.state_revision == 2
    guide.transition(GuideLifecycle.REPLACEMENT_PENDING.value, 3, replacement_id="guide-2")
    with pytest.raises(SchemaValidationError):
        guide.transition(GuideLifecycle.PROPOSED.value, 4)


def test_private_proxy_projection_and_depth_hide_show_keep_plate_reference() -> None:
    layer_plate = LayerRecord("plate", "Complete plate", "raster")
    layer_depth = LayerRecord("depth", "Depth element", "depth", depth_element=True, complete_plate_id="plate")
    document = Document("doc-safe", 16, 16, root_layer_ids=["plate", "depth"], layers={"plate": layer_plate, "depth": layer_depth})
    layer_depth.visible = False
    reloaded = Document.from_dict(document.to_dict())
    assert reloaded.layers["depth"].visible is False
    assert reloaded.layers["depth"].complete_plate_id == "plate"
    safe = reloaded.agent_safe_dict()
    serialized = json.dumps(safe)
    assert "private_source" not in serialized
    assert "private-source" not in serialized


def test_schema_rejects_incompatible_boolean_and_dangling_clipping_id() -> None:
    document = Document("doc-type", 16, 16, root_layer_ids=["layer"],
                        layers={"layer": LayerRecord("layer", "Layer", "raster")})
    raw = document.to_dict()
    raw["layers"]["layer"]["visible"] = "false"
    with pytest.raises(SchemaValidationError, match="visible"):
        Document.from_dict(raw)

    document.layers["layer"].clipping_refs = ["missing-layer"]
    with pytest.raises(SchemaValidationError, match="clippingRefs"):
        document.validate()


def test_transaction_history_requires_full_typed_reversible_record() -> None:
    document = Document("doc-history", 16, 16, current_revision=1,
                        root_layer_ids=["layer"], layers={"layer": LayerRecord("layer", "Layer", "raster")},
                        history=[{"transactionId": "txn-1", "previousRevision": 0, "resultingRevision": 1}])
    with pytest.raises(SchemaValidationError, match="history"):
        document.validate()


def test_bb_inverse_matrix_and_dimensions_are_reciprocal() -> None:
    forward = CoordinateTransform.from_forward("source-crop", "native-bb", (20, 10), (40, 30),
                                               AffineTransform((2, 0.2, 3, 0.3, 3, 4, 0, 0, 1)), 1)
    reverse = CoordinateTransform.from_forward("native-bb", "source-crop", (40, 30), (20, 10),
                                               forward.inverse, 1)
    def operation(reverse_transform=reverse, *, operation_revision=1, source_crop_dimensions=(20, 10)):
        return BBOperation("bb-1", 2, "doc-bb", 0, "composite-1", [{"layerId": "layer-1", "revision": 0}],
                           "mask-1", "a" * 64, BBox(0, 10, 0, 20), source_crop_dimensions, (40, 30),
                           forward, reverse_transform, "asset-1", operation_revision=operation_revision)

    operation().validate()
    mapped = forward.map_point(7.5, 4.5)
    restored = reverse.map_point(*mapped)
    assert restored == pytest.approx((7.5, 4.5))

    bad_reverse = CoordinateTransform.from_forward("native-bb", "source-crop", (40, 30), (20, 10),
                                                    AffineTransform.identity(), 1)
    with pytest.raises(SchemaValidationError, match="reciprocal"):
        operation(bad_reverse).validate()
    swapped_dimensions = CoordinateTransform.from_forward("native-bb", "source-crop", (40, 30), (10, 20),
                                                           forward.inverse, 1)
    with pytest.raises(SchemaValidationError, match="dimensions"):
        operation(swapped_dimensions).validate()
    wrong_revision = CoordinateTransform.from_forward("native-bb", "source-crop", (40, 30), (20, 10),
                                                       forward.inverse, 2)
    with pytest.raises(SchemaValidationError, match="operation revision"):
        operation(wrong_revision).validate()
    with pytest.raises(SchemaValidationError, match="source crop dimensions"):
        operation(source_crop_dimensions=(19, 10)).validate()
    wrong_space = CoordinateTransform.from_forward("preview", "source-crop", (40, 30), (20, 10),
                                                    forward.inverse, 1)
    with pytest.raises(SchemaValidationError, match="native-bb"):
        operation(wrong_space).validate()
    with pytest.raises(ValueError, match="accepted named space"):
        CoordinateTransform.from_forward("private-source", "native-bb", (20, 10), (40, 30), forward.forward, 1)

    outside_document = _complete_fixture()
    outside_operation = outside_document.bb_operations["bb-op"]
    outside_operation.source_bbox = BBox(90, 110, 0, 20)
    outside_operation.source_crop_dimensions = (20, 20)
    outside_operation.crop_to_native = CoordinateTransform.from_forward(
        "source-crop", "native-bb", (20, 20), (50, 50), AffineTransform.identity(), 1
    )
    outside_operation.native_to_crop = CoordinateTransform.from_forward(
        "native-bb", "source-crop", (50, 50), (20, 20), AffineTransform.identity(), 1
    )
    with pytest.raises(SchemaValidationError, match="bbox exceeds"):
        outside_document.validate()


def test_private_proxy_metadata_is_typed_and_non_disclosing() -> None:
    proxy = PrivateProxy("proxy-1", "asset-1", "a" * 64, "slot-1", _transform(),
                         {"depthRange": {"near": 0.1, "far": 1.0}, "registrationQuality": 0.9,
                          "sourceDimensions": [100, 100], "coordinateSpace": "document"})
    serialized = json.dumps(proxy.to_dict())
    assert "private-source-path" not in serialized
    bad_values = [
        {"note": {"sourcePath": "C:/private.png"}},
        {"note": {"uri": "file:///private/source.png"}},
        {"note": {"hash": "a" * 64}},
        {"note": {"assetId": "private-asset"}},
        {"note": {"thumbnail": [1, 2, 3]}},
        {"note": {"history": [{"transactionId": "private-transaction"}]}},
        {"note": {"pixels": [[0, 0, 0]]}},
        {"depthRange": {"near": 0.1, "far": 1.0, "sourcePath": "C:/private.png"}},
        {"registrationQuality": {"assetId": "private-asset"}},
        {"coordinateSpace": "private-source-uri"},
    ]
    for metadata in bad_values:
        with pytest.raises(SchemaValidationError):
            PrivateProxy("proxy-1", "asset-1", "a" * 64, "slot-1", _transform(), metadata).to_dict()

    asset = AssetRecord("proxy-asset", "d" * 64, "image/png", f"assets/sha256/dd/{'d' * 64}.png", extension="png")
    layer = LayerRecord("proxy-layer", "Proxy", "raster", asset_ids=["proxy-asset"])
    valid_proxy = PrivateProxy("proxy-real", "proxy-asset", "d" * 64, "slot-safe", _transform(),
                               {"depthRange": {"near": 0.1, "far": 1.0}, "registrationQuality": 0.8})
    document = Document("doc-proxy", 16, 16, root_layer_ids=["proxy-layer"],
                        layers={"proxy-layer": layer}, assets={"proxy-asset": asset},
                        private_proxies={"proxy-real": valid_proxy})
    serialized_document = json.dumps(document.to_dict())
    serialized_projection = json.dumps(document.agent_safe_dict())
    for serialized in (serialized_document, serialized_projection):
        assert "private-source-path" not in serialized
        assert "director-private-hash" not in serialized
    assert PrivateProxy.from_dict(valid_proxy.to_dict()).to_dict() == valid_proxy.to_dict()


def _transaction(revision: int, *, transaction_id: str | None = None) -> TransactionRecord:
    return TransactionRecord(
        transaction_id or f"txn-{revision}", f"group-{revision}", "director", "director",
        [f"cmd-{revision}"], revision - 1, revision, before={"revision": revision - 1}, after={"revision": revision},
    )


def _selection_document() -> Document:
    source = LayerRecord("source", "Source", "raster")
    derived = LayerRecord("derived", "Derived", "raster")
    masks = {
        "mask-old": MaskRecord("mask-old", "editing-selection", "document", 0, editable_source={"path": []}),
        "mask-new": MaskRecord("mask-new", "editing-selection", "document", 0, editable_source={"path": []}),
    }
    selection = SelectionRecord.create(
        selection_id="selection", source_id="source", source_revision=0, source_content_digest="a" * 64,
        mask_id="mask-old",
    )
    return Document("doc-selection", 16, 16, root_layer_ids=["source", "derived"],
                    layers={"source": source, "derived": derived}, masks=masks,
                    selections={"selection": selection})


def test_selection_create_refine_and_duplicate_registration_use_cas() -> None:
    document = _selection_document()
    selection = document.selections.pop("selection")
    created = document.create_selection(selection, expected_document_revision=0, command_id="cmd-create")
    assert created.selection_revision == 1
    assert document.current_revision == 1
    assert document.history[-1]["kind"] == "selection-create"

    before = document.to_dict()
    refined = document.refine_selection("selection", "mask-new", expected_document_revision=1,
                                         expected_selection_revision=1, refinement={"tool": "polygon"})
    assert refined.selection_revision == 2
    assert refined.refinement_history[-1]["previousMaskId"] == "mask-old"
    assert "mask-old" in document.masks

    before_duplicate = document.to_dict()
    duplicated = document.register_selection_derived_layer(
        "selection", "derived", expected_document_revision=2, expected_selection_revision=2,
    )
    assert duplicated.selection_revision == 3
    assert duplicated.duplicate_history[-1]["layerId"] == "derived"
    committed = document.to_dict()
    with pytest.raises(SelectionRevisionConflict):
        document.register_selection_derived_layer(
            "selection", "source", expected_document_revision=3, expected_selection_revision=2,
        )
    assert document.to_dict() == committed
    assert before != document.to_dict()
    assert before_duplicate != committed


@pytest.mark.parametrize(
    ("reason", "expected_state", "digest"),
    [
        ("SOURCE_CONTENT_CHANGED", "stale-source", "b" * 64),
        ("SOURCE_CLIP_CHANGED", "stale-source", None),
        ("SOURCE_TRANSFORM_CHANGED", "needs-rebase", None),
        ("SOURCE_REMOVED", "stale-source", None),
    ],
)
def test_selection_invalidation_preserves_mask_and_records_every_staleness_reason(reason, expected_state, digest) -> None:
    document = _selection_document()
    document.register_selection_derived_layer(
        "selection", "derived", expected_document_revision=0, expected_selection_revision=1,
    )
    selection = document.selections["selection"]
    changed = document.invalidate_selections_for_source(
        "source", expected_document_revision=1, reason=reason, source_revision=2, source_content_digest=digest,
    )
    selection = document.selections["selection"]
    assert changed == ["selection"]
    assert document.current_revision == 2
    assert selection.selection_revision == 2
    assert selection.mask_id == "mask-old"
    assert selection.derived_layer_ids == ["derived"]
    assert selection.state == expected_state
    assert selection.invalidation_history[-1]["reason"] == reason
    assert document.layers["source"].revision == 2


@pytest.mark.parametrize("reason", ["SOURCE_CONTENT_CHANGED", "SOURCE_CLIP_CHANGED", "SOURCE_TRANSFORM_CHANGED"])
def test_selection_rebase_is_explicit_and_mask_safe(reason) -> None:
    document = _selection_document()
    digest = "b" * 64 if reason == "SOURCE_CONTENT_CHANGED" else None
    document.invalidate_selections_for_source("source", expected_document_revision=0, reason=reason,
                                              source_revision=1, source_content_digest=digest)
    old_mask = document.selections["selection"].mask_id
    rebased = document.rebase_selection(
        "selection", 1, expected_document_revision=1, expected_selection_revision=1,
        new_mask_id="mask-new" if reason != "SOURCE_TRANSFORM_CHANGED" else None,
        source_content_digest=digest, geometry_only=(reason == "SOURCE_TRANSFORM_CHANGED"),
    )
    assert rebased.state == "current"
    assert rebased.selection_revision == 2
    assert rebased.mask_id == (old_mask if reason == "SOURCE_TRANSFORM_CHANGED" else "mask-new")
    assert old_mask in document.masks
    Document.from_dict(document.to_dict())


def test_removed_selection_source_refuses_rebase_without_partial_mutation() -> None:
    document = _selection_document()
    document.invalidate_selections_for_source("source", expected_document_revision=0,
                                              reason="SOURCE_REMOVED", source_revision=1)
    before = document.to_dict()
    with pytest.raises(SchemaValidationError, match="removed source"):
        document.rebase_selection("selection", 1, expected_document_revision=1,
                                  expected_selection_revision=1, new_mask_id="mask-new")
    assert document.to_dict() == before


@pytest.mark.parametrize(
    ("geometry_only", "new_mask_id"),
    [(False, None), (True, "mask-new")],
    ids=["non-geometry-transform-rebase", "new-mask-transform-rebase"],
)
def test_transform_staleness_only_allows_geometry_rebase_with_existing_mask(geometry_only, new_mask_id) -> None:
    document = _selection_document()
    document.invalidate_selections_for_source("source", expected_document_revision=0,
                                              reason="SOURCE_TRANSFORM_CHANGED", source_revision=1)
    before = document.to_dict()
    with pytest.raises(SchemaValidationError, match="transform staleness requires geometry-only"):
        document.rebase_selection(
            "selection", 1, expected_document_revision=1, expected_selection_revision=1,
            new_mask_id=new_mask_id, geometry_only=geometry_only,
        )
    assert document.to_dict() == before


def test_persisted_transform_rebase_cannot_claim_a_new_mask() -> None:
    document = _selection_document()
    document.invalidate_selections_for_source("source", expected_document_revision=0,
                                              reason="SOURCE_TRANSFORM_CHANGED", source_revision=1)
    document.rebase_selection("selection", 1, expected_document_revision=1,
                              expected_selection_revision=1, geometry_only=True)
    raw = copy.deepcopy(document.to_dict())
    selection = raw["selections"]["selection"]
    selection["rebaseHistory"][0]["geometryOnly"] = False
    selection["rebaseHistory"][0]["maskId"] = "mask-new"
    selection["maskId"] = "mask-new"
    raw.pop("revisionDigest", None)
    raw["revisionDigest"] = content_digest(raw)
    with pytest.raises(SchemaValidationError, match="transform rebase must be geometry-only"):
        Document.from_dict(raw)


def test_content_rebase_digest_is_bound_to_retained_transaction_snapshot() -> None:
    document = _selection_document()
    document.invalidate_selections_for_source("source", expected_document_revision=0,
                                              reason="SOURCE_CONTENT_CHANGED", source_revision=1,
                                              source_content_digest="b" * 64)
    document.rebase_selection(
        "selection", 1, expected_document_revision=1, expected_selection_revision=1,
        new_mask_id="mask-new", source_content_digest="b" * 64,
    )
    restored = Document.from_dict(document.to_dict())
    assert restored.selections["selection"].source_content_digest == "b" * 64

    raw = copy.deepcopy(document.to_dict())
    raw["selections"]["selection"]["sourceContentDigest"] = "c" * 64
    raw.pop("revisionDigest", None)
    raw["revisionDigest"] = content_digest(raw)
    with pytest.raises(SchemaValidationError, match="final source content digest"):
        Document.from_dict(raw)


def test_guide_replacement_supersession_chain_is_revision_bound() -> None:
    old = GuideRecord("guide-old", "Old", "superseded", 0, 3, replacement_id="guide-new")
    new = GuideRecord("guide-new", "Replacement", "active", 2, 3, supersedes_id="guide-old")
    history = [_transaction(revision).to_dict() for revision in (1, 2, 3)]
    document = Document("doc-guides", 16, 16, current_revision=3, root_layer_ids=["layer"],
                        layers={"layer": LayerRecord("layer", "Layer", "raster", revision=3)},
                        guides={"guide-old": old, "guide-new": new}, history=history)
    reloaded = Document.from_dict(document.to_dict())
    assert reloaded.guides["guide-old"].replacement_id == "guide-new"
    assert reloaded.guides["guide-new"].supersedes_id == "guide-old"

    old.replacement_id = "missing-guide"
    with pytest.raises(SchemaValidationError, match="replacement lineage"):
        document.validate()


def test_complete_guide_proposal_activation_replacement_and_retirement_round_trip() -> None:
    original = GuideRecord("guide-original", "Original", created_revision=0)
    original.transition(GuideLifecycle.ACTIVE.value, 1)
    original.transition(GuideLifecycle.CONSUMED.value, 2)
    original.transition(GuideLifecycle.REPLACEMENT_PENDING.value, 3, replacement_id="guide-replacement")
    replacement = GuideRecord(
        "guide-replacement", "Replacement", GuideLifecycle.ACTIVE.value, 3, 3,
        supersedes_id="guide-original",
    )
    original.transition(GuideLifecycle.SUPERSEDED.value, 4)
    original.transition(GuideLifecycle.SAFE_TO_REMOVE.value, 5)
    document = Document(
        "doc-guide-lifecycle", 16, 16, current_revision=5, root_layer_ids=["layer"],
        layers={"layer": LayerRecord("layer", "Layer", "raster", revision=5)},
        guides={original.guide_id: original, replacement.guide_id: replacement},
        history=[_transaction(revision).to_dict() for revision in range(1, 6)],
    )
    restored = Document.from_dict(document.to_dict())
    assert restored.guides["guide-original"].lifecycle == GuideLifecycle.SAFE_TO_REMOVE.value
    assert restored.guides["guide-original"].state_revision == 5
    assert restored.guides["guide-replacement"].lifecycle == GuideLifecycle.ACTIVE.value


def test_guide_replacement_cycle_and_handoff_dangling_bindings_fail_closed() -> None:
    first = GuideRecord("guide-first", "First", "replacement-pending", 0, 0,
                        supersedes_id="guide-second", replacement_id="guide-second")
    second = GuideRecord("guide-second", "Second", "replacement-pending", 0, 0,
                         supersedes_id="guide-first", replacement_id="guide-first")
    with pytest.raises(SchemaValidationError, match="cycle"):
        Document("doc-guide-cycle", 8, 8, guides={"guide-first": first, "guide-second": second}).validate()

    note = HandoffNote("note-dangling", "director", "director", "Will not reattach", 0,
                       "txn-missing", ["missing-target"], {"missing-target": 4})
    document = Document("doc-dangling-note", 8, 8, root_layer_ids=["layer"],
                        layers={"layer": LayerRecord("layer", "Layer", "raster",
                                                     collaboration=CollaborationState(handoff_notes=[note]))})
    with pytest.raises(SchemaValidationError, match="orphaned"):
        document.validate()

    raw = Document("doc-dangling-note", 8, 8, root_layer_ids=["layer"],
                    layers={"layer": LayerRecord("layer", "Layer", "raster")}).to_dict()
    raw["layers"]["layer"]["collaboration"]["handoffNotes"] = [note.to_dict()]
    raw.pop("revisionDigest", None)
    raw["revisionDigest"] = content_digest(raw)
    loaded = Document.from_dict(raw)
    loaded_note = loaded.layers["layer"].collaboration.handoff_notes[0]
    assert loaded_note.body == "Will not reattach"
    assert loaded_note.state == "orphaned"
    assert "missing-target no longer resolves" in loaded_note.orphan_reason
    assert loaded.revision_digest == loaded.canonical_digest()


def test_handoff_bindings_orphan_explicitly_and_safe_projection_hides_private_notes() -> None:
    tx1 = _transaction(1)
    shared = HandoffNote("note-shared", "director", "director", "Review the layer", 1, "txn-1",
                         ["layer"], {"layer": 1}, "shared")
    private = HandoffNote("note-private", "director", "director", "Private review", 1, "txn-1",
                          ["layer"], {"layer": 1}, "director-only")
    layer = LayerRecord("layer", "Layer", "raster", revision=1,
                        collaboration=CollaborationState(handoff_notes=[shared, private]))
    document = Document("doc-notes", 16, 16, current_revision=1, root_layer_ids=["layer"],
                        layers={"layer": layer}, history=[tx1.to_dict()])
    reloaded = Document.from_dict(document.to_dict())
    safe = json.dumps(reloaded.agent_safe_dict())
    assert "Review the layer" in safe
    assert "Private review" not in safe

    orphaned = HandoffNote("note-orphan", "director", "director", "Keep unresolved", 0, "txn-missing",
                           ["deleted-layer"], {"deleted-layer": 1}, state="orphaned", orphan_reason="target removed")
    orphan_doc = Document("doc-orphan", 16, 16, root_layer_ids=["layer"],
                          layers={"layer": LayerRecord("layer", "Layer", "raster",
                                                       collaboration=CollaborationState(handoff_notes=[orphaned]))})
    Document.from_dict(orphan_doc.to_dict())


def test_handoff_reconciliation_preserves_proven_historical_target_revision() -> None:
    note = HandoffNote("note-1", "director", "director", "Review", 1, "txn-1", ["layer"], {"layer": 1})
    layer = LayerRecord("layer", "Layer", "raster", revision=2,
                        collaboration=CollaborationState(handoff_notes=[note]))
    tx1 = TransactionRecord(
        "txn-1", "group-1", "director", "director", ["cmd-1"], 0, 1,
        affected_ids=["layer"], before={"layer": {"layerId": "layer", "revision": 0}},
        after={"layer": {"layerId": "layer", "revision": 1}},
    )
    tx2 = TransactionRecord(
        "txn-2", "group-2", "director", "director", ["cmd-2"], 1, 2,
        affected_ids=["layer"], before={"layer": {"layerId": "layer", "revision": 1}},
        after={"layer": {"layerId": "layer", "revision": 2}},
    )
    history = [tx1.to_dict(), tx2.to_dict()]
    document = Document("doc-orphaning", 16, 16, current_revision=2, root_layer_ids=["layer"],
                        layers={"layer": layer}, history=history)
    changed = document.reconcile_handoffs(expected_document_revision=2)
    assert changed == []
    assert document.layers["layer"].collaboration.handoff_notes[0].state == "open"
    assert document.current_revision == 2
    reloaded = Document.from_dict(document.to_dict())
    assert reloaded.layers["layer"].collaboration.handoff_notes[0].target_revisions == {"layer": 1}


@pytest.mark.parametrize(
    ("failure", "expected_reason"),
    [
        ("missing-transaction", "transaction binding no longer resolves"),
        ("missing-target", "target deleted-layer no longer resolves"),
        ("unproven-revision", "cannot be proven from retained history"),
    ],
)
def test_handoff_reconciliation_orphans_only_genuinely_lost_bindings(failure, expected_reason) -> None:
    target_id = "deleted-layer" if failure == "missing-target" else "layer"
    target_revision = 1
    note_transaction = "txn-missing" if failure == "missing-transaction" else "txn-1"
    note = HandoffNote("note-lost", "director", "director", "Keep unresolved", 1,
                       note_transaction, [target_id], {target_id: target_revision})
    current_target_revision = 2 if failure == "unproven-revision" else 1
    layer = LayerRecord("layer", "Layer", "raster", revision=current_target_revision,
                        collaboration=CollaborationState(handoff_notes=[note]))
    current_revision = current_target_revision
    history = [_transaction(revision).to_dict() for revision in range(1, current_revision + 1)]
    document = Document("doc-lost-binding", 16, 16, current_revision=current_revision, root_layer_ids=["layer"],
                        layers={"layer": layer}, history=history)

    changed = document.reconcile_handoffs(expected_document_revision=current_revision)
    assert changed == ["note-lost"]
    orphaned = document.layers["layer"].collaboration.handoff_notes[0]
    assert orphaned.state == "orphaned"
    assert expected_reason in orphaned.orphan_reason
    assert document.current_revision == current_revision + 1
    assert Document.from_dict(document.to_dict()).layers["layer"].collaboration.handoff_notes[0].state == "orphaned"


def test_operation_inputs_outputs_parent_child_and_exchange_patch_links_resolve() -> None:
    asset = AssetRecord("asset", "c" * 64, "application/octet-stream", f"assets/sha256/cc/{'c' * 64}.bin")
    layer = LayerRecord("layer", "Layer", "raster", object_ids=["object"], asset_ids=["asset"])
    obj = ObjectRecord("object", "layer", "path")
    parent = OperationRecord("op-parent", "import", 2, [{"kind": "layer", "id": "layer"}],
                             produced_ids=["object"], child_ids=["op-child"])
    child = OperationRecord("op-child", "refine", 2, [{"recordType": "object", "recordId": "object"}],
                            parent_ids=["op-parent"])
    exchange = ExternalRoundTrip("exchange", "doc-links", ["patch"], 0, "document", None, None,
                                 _transform(), "c" * 64)
    patch = CorrectivePatch("patch", "layer", AffineTransform.identity(), 0, external_exchange_id="exchange")
    document = Document("doc-links", 16, 16, root_layer_ids=["layer"], layers={"layer": layer},
                        objects={"object": obj}, assets={"asset": asset}, operations={"op-parent": parent, "op-child": child},
                        external_round_trips={"exchange": exchange}, patches={"patch": patch})
    Document.from_dict(document.to_dict())

    child.input_refs = [{"kind": "asset", "id": "missing"}]
    with pytest.raises(SchemaValidationError, match="input reference"):
        document.validate()
