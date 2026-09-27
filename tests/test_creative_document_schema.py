from __future__ import annotations

import json

import pytest

from modules.creative_document import (
    AffineTransform,
    AssetRecord,
    BBOperation,
    BBox,
    CandidateRecord,
    CoordinateTransform,
    DepthComposite,
    Document,
    ExtractionDerivative,
    GuideLifecycle,
    GuideRecord,
    LayerRecord,
    MaskRecord,
    ObjectRecord,
    PrivateProxy,
    SchemaValidationError,
    SelectionRecord,
)


def _transform() -> CoordinateTransform:
    return CoordinateTransform.from_forward(
        "document", "native-bb", (100, 100), (50, 50), AffineTransform.scale(0.5)
    )


def test_nontrivial_document_round_trips_and_digest_is_canonical() -> None:
    transform = _transform()
    asset = AssetRecord("asset-bg", "0" * 64, "image/png", "assets/sha256/00/" + "0" * 64 + ".png", extension="png", byte_length=4)
    layer_root = LayerRecord("layer-root", "Scene", "group", child_ids=["layer-subject"], asset_ids=["asset-bg"])
    layer_subject = LayerRecord("layer-subject", "Subject", "raster", parent_id="layer-root", mask_ids=["mask-context"], depth_element=True, complete_plate_id="layer-root")
    obj = ObjectRecord("obj-guide", "layer-subject", "path", geometry={"points": [[1, 2], [3, 4]]})
    context = MaskRecord("mask-context", "context", "document", 0, asset_id="asset-bg", owner_id="layer-subject", content_hash="0" * 64)
    matte = MaskRecord("mask-matte", "extraction", "native-bb", 0, asset_id="asset-bg", owner_id="bb-op")
    composite = DepthComposite("cmp-1", 0, [{"layerId": "layer-root"}], content_asset_id="asset-bg", complete_plate_layer_id="layer-root")
    crop_transform = CoordinateTransform.from_forward(
        "source-crop", "native-bb", (50, 50), (50, 50), AffineTransform.identity()
    )
    inverse_transform = CoordinateTransform(
        "native-bb", "source-crop", (50, 50), (50, 50), crop_transform.inverse, crop_transform.forward
    )
    bb = BBOperation(
        "bb-op", 1, "doc-1", 0, "cmp-1", [{"layerId": "layer-root", "revision": 0}], "mask-context", "0" * 64,
        BBox(0, 50, 0, 50), (50, 50), (50, 50), crop_transform, inverse_transform, "asset-bg", operation_revision=1,
    )
    candidate = CandidateRecord("candidate-1", "full_crop_contextual", "bb-op", 0, True, None, transform, asset_id="asset-bg")
    derivative = ExtractionDerivative("derivative-1", "candidate-1", "bb-op", "mask-matte", transform, 0, "asset-bg", {"edge": "feather"})
    selection = SelectionRecord("selection-1", "layer-subject", 0, "0" * 64, 1, "current", "mask-context", derived_layer_ids=["layer-subject"])
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
    first = document.to_dict()
    second = json.loads(json.dumps(first))
    assert Document.from_dict(second).to_dict() == first
    assert document.canonical_digest() == Document.from_dict(second).canonical_digest()


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
