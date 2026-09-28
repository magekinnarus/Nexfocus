from __future__ import annotations

import io

import pytest
from PIL import Image

from modules.creative_document import (
    AffineTransform,
    AssetStore,
    Document,
    LayerRecord,
    ObjectRecord,
    make_id,
)
from modules.creative_document.editor_render import RenderError, render_document, render_document_png
from modules.creative_document.editor_view import ViewportMapping, document_to_view


def _png(color: tuple[int, int, int, int], size: tuple[int, int] = (16, 12)) -> bytes:
    output = io.BytesIO()
    Image.new("RGBA", size, color).save(output, format="PNG")
    return output.getvalue()


def _add_raster(document: Document, asset_store: AssetStore, layer_id: str, raw: bytes) -> str:
    record = asset_store.put_bytes(raw, media_type="image/png", extension="png", width=16, height=12, has_alpha=True)
    object_id = make_id("obj")
    layer = document.layers[layer_id]
    layer.asset_ids.append(record.asset_id)
    layer.object_ids.append(object_id)
    obj = ObjectRecord(
        object_id,
        layer_id,
        "raster-placement",
        geometry={"x": 0, "y": 0, "width": 16, "height": 12},
        asset_id=record.asset_id,
    )
    document.objects[object_id] = obj
    document.assets[record.asset_id] = record
    return record.asset_id


def test_viewport_inverse_mapping_covers_fit_actual_zoom_pan_and_dpr() -> None:
    mapping = ViewportMapping(2400, 1792, 1200, 900, zoom=1.0)
    assert mapping.scale == pytest.approx(0.5)
    fit_point = mapping.document_to_preview(840.25, 512.5)
    assert mapping.preview_to_document(*fit_point) == pytest.approx((840.25, 512.5))

    actual = ViewportMapping(2400, 1792, 1200, 900, zoom=2.0, pan_x=51.5, pan_y=-24.0)
    preview = actual.document_to_preview(2399, 1791)
    assert actual.preview_to_document(*preview) == pytest.approx((2399, 1791))
    physical = (preview[0] * 2.5, preview[1] * 2.5)
    assert actual.physical_to_document(*physical, 2.5) == pytest.approx((2399, 1791))
    assert len(actual.matrix()) == 9


def test_nested_document_and_object_transforms_round_trip_through_preview() -> None:
    parent = AffineTransform((1, 0, 14, 0, 1, -3, 0, 0, 1))
    layer = AffineTransform((0, -1, 40, 1, 0, 6, 0, 0, 1))
    obj = AffineTransform((2, 0, 5, 0, 0.5, 9, 0, 0, 1))
    local = (7.25, 11.5)
    document_point = parent.compose(layer).compose(obj).apply_point(*local)
    mapping = ViewportMapping(100, 80, 700, 500, zoom=1.7, pan_x=32, pan_y=-18)
    preview = mapping.document_to_preview(*document_point)
    recovered_document = mapping.preview_to_document(*preview)
    recovered_local = parent.compose(layer).compose(obj).inverse().apply_point(*recovered_document)
    assert recovered_local == pytest.approx(local)


def test_view_projection_omits_storage_paths_and_exposes_only_asset_routes(tmp_path) -> None:
    store = AssetStore(tmp_path / "project")
    layer_id = make_id("layer")
    document = Document(make_id("doc"), 16, 12, root_layer_ids=[layer_id], layers={layer_id: LayerRecord(layer_id, "Base", "raster")})
    _add_raster(document, store, layer_id, _png((200, 30, 20, 255)))
    document.refresh_digest()

    view = document_to_view(document)
    serialized = repr(view)
    assert str(store.project_root) not in serialized
    assert "storageUri" not in serialized
    assert view["assets"][0]["url"].endswith("/content")
    assert view["assets"][0]["thumbnailUrl"].endswith("/thumbnail")
    assert "transform" in view["layers"][0]


def test_render_is_deterministic_respects_layer_order_opacity_and_visibility(tmp_path) -> None:
    store = AssetStore(tmp_path / "project")
    bottom_id = make_id("layer")
    top_id = make_id("layer")
    document = Document(make_id("doc"), 16, 12, root_layer_ids=[bottom_id, top_id], layers={
        bottom_id: LayerRecord(bottom_id, "Red", "raster"),
        top_id: LayerRecord(top_id, "Blue", "raster", opacity=0.5),
    })
    _add_raster(document, store, bottom_id, _png((240, 10, 20, 255)))
    _add_raster(document, store, top_id, _png((10, 20, 240, 255)))
    document.refresh_digest()
    before = document.to_dict()

    first = render_document(document, lambda asset_id: store.read_bytes(document.assets[asset_id]))
    encoded = render_document_png(document, lambda asset_id: store.read_bytes(document.assets[asset_id]))
    assert encoded == render_document_png(document, lambda asset_id: store.read_bytes(document.assets[asset_id]))
    assert document.to_dict() == before
    assert first.getpixel((4, 4)) == (125, 15, 130, 255)
    document.layers[top_id].visible = False
    visible = render_document(document, lambda asset_id: store.read_bytes(document.assets[asset_id]))
    assert visible.getpixel((4, 4)) == (240, 10, 20, 255)


def test_group_opacity_is_applied_after_compositing_children(tmp_path) -> None:
    store = AssetStore(tmp_path / "project")
    group_id, red_id, blue_id = make_id("layer"), make_id("layer"), make_id("layer")
    group = LayerRecord(group_id, "Group", "group", child_ids=[red_id, blue_id], opacity=0.5)
    red = LayerRecord(red_id, "Red", "raster", parent_id=group_id)
    blue = LayerRecord(blue_id, "Blue", "raster", parent_id=group_id)
    document = Document(make_id("doc"), 16, 12, root_layer_ids=[group_id], layers={group_id: group, red_id: red, blue_id: blue})
    _add_raster(document, store, red_id, _png((240, 10, 20, 255)))
    _add_raster(document, store, blue_id, _png((10, 20, 240, 255)))
    document.refresh_digest()

    rendered = render_document(document, lambda asset_id: store.read_bytes(document.assets[asset_id]))
    assert rendered.getpixel((4, 4)) == (10, 20, 240, 128)


def test_unsupported_visible_semantics_fail_with_typed_render_errors() -> None:
    layer_id = make_id("layer")
    document = Document(make_id("doc"), 8, 8, root_layer_ids=[layer_id], layers={
        layer_id: LayerRecord(layer_id, "Unsupported", "text")
    })
    with pytest.raises(RenderError, match="unsupported visible layer kind") as error:
        render_document(document, lambda _asset_id: b"")
    assert error.value.code == "UNSUPPORTED_LAYER_KIND"

    document.layers[layer_id].kind = "paint"
    document.layers[layer_id].blend_mode = "multiply"
    with pytest.raises(RenderError) as error:
        render_document(document, lambda _asset_id: b"")
    assert error.value.code == "UNSUPPORTED_LAYER_SEMANTICS"


def test_render_persists_shapes_paths_and_paint_strokes_as_editable_objects() -> None:
    layer_id = make_id("layer")
    shape_id, path_id, stroke_id = make_id("obj"), make_id("obj"), make_id("obj")
    layer = LayerRecord(layer_id, "Paint and vectors", "paint", object_ids=[shape_id, path_id, stroke_id])
    document = Document(make_id("doc"), 16, 12, root_layer_ids=[layer_id], layers={layer_id: layer}, objects={
        shape_id: ObjectRecord(shape_id, layer_id, "shape", geometry={"shape": "rectangle", "x": 1, "y": 1, "width": 5, "height": 4}, style={"fill": "#EF3322"}),
        path_id: ObjectRecord(path_id, layer_id, "path", geometry={"points": [[1, 8], [6, 8]]}, style={"stroke": "#2266EE", "width": 2}),
        stroke_id: ObjectRecord(stroke_id, layer_id, "paint-stroke", geometry={"points": [[9, 2], [13, 2]]}, style={"color": "#22CC55", "width": 3, "opacity": 0.75, "hardness": 0.5}),
    })
    rendered = render_document(document, lambda _asset_id: b"")
    assert rendered.getpixel((3, 3)) == (239, 51, 34, 255)
    assert rendered.getpixel((3, 8))[2] > 200
    assert rendered.getpixel((11, 2))[1] > 100
    assert len(document.objects) == 3


def test_view_surfaces_typed_browser_fidelity_and_effective_lock_states() -> None:
    parent_id, child_id, warning_layer_id = make_id("layer"), make_id("layer"), make_id("layer")
    unsupported_id, soft_id = make_id("obj"), make_id("obj")
    parent = LayerRecord(parent_id, "Locked group", "group", child_ids=[child_id], locked=True)
    child = LayerRecord(child_id, "Unsupported child", "text", parent_id=parent_id, object_ids=[unsupported_id])
    warning_layer = LayerRecord(warning_layer_id, "Soft paint", "paint", object_ids=[soft_id])
    warning_layer.clipping_refs = [parent_id]
    document = Document(make_id("doc"), 16, 12, root_layer_ids=[parent_id, warning_layer_id],
                        layers={parent_id: parent, child_id: child, warning_layer_id: warning_layer}, objects={
        unsupported_id: ObjectRecord(unsupported_id, child_id, "text-placement"),
        soft_id: ObjectRecord(soft_id, warning_layer_id, "paint-stroke",
                              geometry={"points": [[1, 1], [5, 5]]}, style={"color": "#ffffff", "hardness": 0.35}),
    })

    view = document_to_view(document)
    child_view = next(layer for layer in view["layers"] if layer["id"] == child_id)
    assert child_view["effectiveLocked"] is True
    assert child_view["lockedByAncestor"] is True
    codes = {issue["code"] for issue in view["rendererIssues"]}
    assert {"UNSUPPORTED_LAYER_KIND", "UNSUPPORTED_OBJECT_KIND", "UNSUPPORTED_LAYER_CLIPPING",
            "PAINT_HARDNESS_PREVIEW_APPROXIMATION"}.issubset(codes)
    assert any(issue.get("objectId") == soft_id and issue["severity"] == "warning"
               for issue in view["rendererIssues"])
    assert child_view["rendererIssues"]
