from __future__ import annotations

import io
import hashlib
import os
from pathlib import Path

import pytest
from PIL import Image

from modules.creative_document import AssetRecord, AssetStore, Document, LayerRecord, MaskRecord, ObjectRecord, ProjectStore, SchemaValidationError, make_id
from modules.creative_document import asset_store as asset_store_module
from modules.creative_document.editor_actions import (
    EditorActionError,
    PendingAsset,
    PreparedAction,
    prepare_action,
    prepare_raster_import,
    prepare_undo_redo,
    publish_pending_assets,
)


ACTOR = "director-unit"


def _png() -> bytes:
    image = Image.new("RGBA", (24, 16), (0, 0, 0, 0))
    for x in range(24):
        for y in range(16):
            if x < 12:
                image.putpixel((x, y), (220, 40, 30, 255))
            else:
                image.putpixel((x, y), (25, 90, 220, 255))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _scene(tmp_path):
    project = tmp_path / "scene.nexscene"
    store = AssetStore(project)
    layer_id = make_id("layer")
    layer = LayerRecord(layer_id, "Source raster", "raster")
    document = Document(make_id("doc"), 24, 16, root_layer_ids=[layer_id], layers={layer_id: layer})
    raw = _png()
    asset = store.put_bytes(raw, media_type="image/png", extension="png", width=24, height=16, has_alpha=True)
    object_id = make_id("obj")
    layer.asset_ids.append(asset.asset_id)
    layer.object_ids.append(object_id)
    document.assets[asset.asset_id] = asset
    document.objects[object_id] = ObjectRecord(
        object_id, layer_id, "raster-placement",
        geometry={"x": 0, "y": 0, "width": 24, "height": 16}, asset_id=asset.asset_id,
    )
    document.refresh_digest()
    return document, store, project, layer_id, asset.asset_id, raw


def _payload(document, action_type, data=None, *, actions=None):
    payload = {
        "documentId": document.document_id,
        "expectedRevision": document.current_revision,
        "actorKind": "director",
        "actionType": action_type,
    }
    if actions is not None:
        payload["actions"] = actions
    else:
        payload["data"] = data or {}
    return payload


def _apply(document, store, action_type, data=None, *, actions=None):
    prepared = prepare_action(document, store, _payload(document, action_type, data, actions=actions), actor_id=ACTOR)
    publish_pending_assets(store, prepared)
    return prepared.document, prepared.receipt


def test_layer_batch_undo_redo_and_invalid_batch_are_atomic(tmp_path) -> None:
    document, store, _project, source_id, _asset_id, _raw = _scene(tmp_path)
    original = document.to_dict()
    items = [
        {"actionType": "add_layer", "data": {"kind": "vector", "name": "Ink"}},
        {"actionType": "add_layer", "data": {"kind": "group", "name": "Composition"}},
    ]
    document, receipt = _apply(document, store, "batch", actions=items)
    assert receipt["currentRevision"] == 1
    assert len(document.history) == 1
    assert len(document.root_layer_ids) == 3

    undone = prepare_undo_redo(document, {"documentId": document.document_id, "expectedRevision": 1}, actor_id=ACTOR)
    document = undone.document
    assert document.current_revision == 2
    assert document.root_layer_ids == original["rootLayerIds"]
    redone = prepare_undo_redo(document, {"documentId": document.document_id, "expectedRevision": 2}, actor_id=ACTOR, redo=True)
    document = redone.document
    assert document.current_revision == 3
    assert len(document.root_layer_ids) == 3

    document, _ = _apply(document, store, "set_layer_lock", {"layerId": source_id, "locked": True})
    before = document.to_dict()
    files_before = sorted(path.relative_to(store.project_root).as_posix() for path in store.project_root.rglob("*"))
    invalid = [
        {"actionType": "add_layer", "data": {"kind": "vector", "name": "Must not persist"}},
        {"actionType": "transform", "targetIds": [source_id], "data": {"transform": [1, 0, 0, 0, 1, 0, 0, 0, 1]}},
    ]
    with pytest.raises(EditorActionError, match="locked") as error:
        prepare_action(document, store, _payload(document, "batch", actions=invalid), actor_id=ACTOR)
    assert error.value.code == "LAYER_LOCKED"
    assert document.to_dict() == before
    assert sorted(path.relative_to(store.project_root).as_posix() for path in store.project_root.rglob("*")) == files_before

    stale = _payload(document, "add_layer", {"kind": "vector", "name": "Stale"})
    stale["expectedRevision"] -= 1
    with pytest.raises(EditorActionError) as error:
        prepare_action(document, store, stale, actor_id=ACTOR)
    assert error.value.code == "STALE_DOCUMENT_REVISION"
    assert error.value.current_revision == document.current_revision


def test_selection_refinement_cas_rebase_and_duplicate_preserve_source_bytes(tmp_path) -> None:
    document, store, project_path, source_id, source_asset_id, original_bytes = _scene(tmp_path)
    document, _ = _apply(document, store, "create_selection", {
        "sourceLayerId": source_id,
        "seeds": [{"kind": "box", "geometry": {"x": 1, "y": 2, "width": 8, "height": 9}}],
        "semanticHint": "left subject",
    })
    selection = next(iter(document.selections.values()))
    original_mask_id = selection.mask_id
    assert selection.state == "current"

    duplicate, _ = _apply(document, store, "duplicate_selection_to_layer", {
        "selectionId": selection.selection_id,
        "expectedSelectionRevision": selection.selection_revision,
        "name": "Left subject copy",
    })
    derived = next(layer for layer in duplicate.layers.values() if layer.layer_id != source_id)
    assert derived.lineage["selectionId"] == selection.selection_id
    assert derived.lineage["sourceLayerId"] == source_id
    assert store.read_bytes(duplicate.assets[source_asset_id]) == original_bytes
    assert derived.layer_id in duplicate.selections[selection.selection_id].derived_layer_ids

    with pytest.raises(EditorActionError) as error:
        prepare_action(duplicate, store, _payload(duplicate, "refine_selection", {
            "selectionId": selection.selection_id,
            "expectedSelectionRevision": 1,
            "operation": "grow",
            "radius": 2,
        }), actor_id=ACTOR)
    assert error.value.code == "STALE_SELECTION_REVISION"

    selection = duplicate.selections[selection.selection_id]
    duplicate, _ = _apply(duplicate, store, "refine_selection", {
        "selectionId": selection.selection_id,
        "expectedSelectionRevision": selection.selection_revision,
        "operation": "grow",
        "radius": 1,
    })
    selection = duplicate.selections[selection.selection_id]
    assert selection.mask_id != original_mask_id
    assert selection.selection_revision == 3

    duplicate, _ = _apply(duplicate, store, "transform", {
        "targetIds": [source_id],
        "transform": [1, 0, 2, 0, 1, 1, 0, 0, 1],
    })
    selection = duplicate.selections[selection.selection_id]
    assert selection.state == "needs-rebase"
    with pytest.raises(EditorActionError) as error:
        prepare_action(duplicate, store, _payload(duplicate, "refine_selection", {
            "selectionId": selection.selection_id,
            "expectedSelectionRevision": selection.selection_revision,
            "operation": "invert",
            "radius": 1,
        }), actor_id=ACTOR)
    assert error.value.code == "SELECTION_NEEDS_REBASE"

    duplicate, _ = _apply(duplicate, store, "rebase_selection", {
        "selectionId": selection.selection_id,
        "expectedSelectionRevision": selection.selection_revision,
        "geometryOnly": True,
    })
    selection = duplicate.selections[selection.selection_id]
    assert selection.state == "current"
    assert selection.mask_id != original_mask_id

    duplicate, _ = _apply(duplicate, store, "set_layer_opacity", {"layerId": source_id, "opacity": 0.6})
    selection = duplicate.selections[selection.selection_id]
    assert selection.state == "stale-source"
    assert selection.stale_reason == "SOURCE_CONTENT_CHANGED"
    assert selection.current_source_content_digest
    prior_mask_id = selection.mask_id
    duplicate, _ = _apply(duplicate, store, "rebase_selection", {
        "selectionId": selection.selection_id,
        "expectedSelectionRevision": selection.selection_revision,
        "geometryOnly": False,
        "seeds": [{"kind": "box", "geometry": {"x": 2, "y": 2, "width": 7, "height": 8}}],
    })
    selection = duplicate.selections[selection.selection_id]
    assert selection.state == "current"
    assert selection.mask_id != prior_mask_id

    project = ProjectStore(project_path)
    project.save(duplicate, checkpoint=True)
    reopened = project.open()
    saved = reopened.selections[selection.selection_id]
    assert saved.mask_id == selection.mask_id
    assert saved.selection_revision == selection.selection_revision
    assert saved.derived_layer_ids == selection.derived_layer_ids
    assert project.assets.read_bytes(reopened.assets[source_asset_id]) == original_bytes


def test_guide_lifecycle_and_relational_context_remain_separate(tmp_path) -> None:
    document, store, project_path, source_id, _asset_id, _raw = _scene(tmp_path)
    document, _ = _apply(document, store, "add_layer", {"kind": "vector", "name": "Context reference"})
    reference_id = document.root_layer_ids[-1]
    guide_data = {
        "name": "Pose guide",
        "semanticRole": "pose-anchor",
        "kind": "shape",
        "geometry": {"shape": "ellipse", "x": 3, "y": 2, "width": 10, "height": 12},
        "style": {"fill": "#E7B45A", "stroke": "#FFFFFF"},
    }
    document, _ = _apply(document, store, "create_guide", guide_data)
    old = next(iter(document.guides.values()))
    assert old.lifecycle == "proposed"
    assert document.layers[document.objects[old.object_ids[0]].layer_id].visible
    with pytest.raises((EditorActionError, SchemaValidationError)):
        prepare_action(document, store, _payload(document, "transition_guide", {
            "guideId": old.guide_id, "lifecycle": "superseded",
        }), actor_id=ACTOR)

    document, _ = _apply(document, store, "transition_guide", {"guideId": old.guide_id, "lifecycle": "active"})
    document, _ = _apply(document, store, "create_guide", {**guide_data, "name": "Replacement guide"})
    replacement = next(guide for guide in document.guides.values() if guide.guide_id != old.guide_id)
    document, _ = _apply(document, store, "transition_guide", {
        "guideId": old.guide_id, "lifecycle": "replacement-pending", "replacementId": replacement.guide_id,
    })
    assert document.guides[replacement.guide_id].supersedes_id == old.guide_id
    document, _ = _apply(document, store, "transition_guide", {"guideId": old.guide_id, "lifecycle": "superseded"})
    document, _ = _apply(document, store, "transition_guide", {"guideId": old.guide_id, "lifecycle": "safe-to-remove"})
    assert document.guides[old.guide_id].replacement_id == replacement.guide_id

    document, _ = _apply(document, store, "create_context_mask", {
        "seeds": [{"kind": "box", "geometry": {"x": 1, "y": 1, "width": 14, "height": 12}}],
    })
    context_mask = next(mask for mask in document.masks.values() if mask.purpose == "context")
    document, _ = _apply(document, store, "set_relational_context", {
        "contextMaskId": context_mask.mask_id,
        "referenceIds": [reference_id],
        "dilationPx": 7,
        "editMaskId": None,
        "editTargetIds": [source_id],
    })
    lineage = document.masks[context_mask.mask_id].lineage
    assert lineage["relationalReferences"] == [reference_id]
    assert lineage["dilationPx"] == 7
    assert lineage["editMaskId"] is None
    assert lineage["editTargetIds"] == [source_id]

    before = document.to_dict()
    with pytest.raises(EditorActionError) as error:
        prepare_action(document, store, _payload(document, "set_relational_context", {
            "contextMaskId": context_mask.mask_id,
            "referenceIds": [source_id],
            "dilationPx": 0,
            "editMaskId": None,
            "editTargetIds": [source_id],
        }), actor_id=ACTOR)
    assert error.value.code == "CONTEXT_EDIT_SCOPE_OVERLAP"
    assert document.to_dict() == before

    project = ProjectStore(project_path)
    project.save(document, checkpoint=True)
    reopened = project.open()
    assert reopened.guides[replacement.guide_id].supersedes_id == old.guide_id
    assert reopened.masks[context_mask.mask_id].lineage["dilationPx"] == 7


def test_proposed_guide_cannot_be_consumed_without_writing(tmp_path) -> None:
    document, store, _project_path, _source_id, _asset_id, _raw = _scene(tmp_path)
    document, _ = _apply(document, store, "create_guide", {
        "name": "Unactivated guide",
        "semanticRole": "pose-anchor",
        "kind": "shape",
        "geometry": {"shape": "ellipse", "x": 3, "y": 2, "width": 10, "height": 12},
        "style": {"fill": "#E7B45A", "stroke": "#FFFFFF"},
    })
    guide = next(iter(document.guides.values()))
    assert guide.lifecycle == "proposed"
    before_document = document.to_dict()
    before_files = {
        path.relative_to(store.project_root).as_posix(): path.read_bytes()
        for path in store.project_root.rglob("*") if path.is_file()
    }

    with pytest.raises(SchemaValidationError, match="invalid guide transition proposed -> consumed"):
        prepare_action(document, store, _payload(document, "transition_guide", {
            "guideId": guide.guide_id, "lifecycle": "consumed",
        }), actor_id=ACTOR)

    assert document.to_dict() == before_document
    assert {
        path.relative_to(store.project_root).as_posix(): path.read_bytes()
        for path in store.project_root.rglob("*") if path.is_file()
    } == before_files


def test_active_guide_can_be_consumed_and_state_survives_save_open(tmp_path) -> None:
    document, store, project_path, _source_id, _asset_id, _raw = _scene(tmp_path)
    document, _ = _apply(document, store, "create_guide", {
        "name": "Applied pose guide",
        "semanticRole": "pose-anchor",
        "kind": "shape",
        "geometry": {"shape": "ellipse", "x": 3, "y": 2, "width": 10, "height": 12},
        "style": {"fill": "#E7B45A", "stroke": "#FFFFFF"},
    })
    guide_id = next(iter(document.guides))
    document, _ = _apply(document, store, "transition_guide", {"guideId": guide_id, "lifecycle": "active"})
    revision_before_consume = document.current_revision

    document, receipt = _apply(document, store, "transition_guide", {"guideId": guide_id, "lifecycle": "consumed"})
    guide = document.guides[guide_id]
    assert guide.lifecycle == "consumed"
    assert document.current_revision == revision_before_consume + 1
    assert guide.state_revision == document.current_revision == receipt["currentRevision"]

    project = ProjectStore(project_path)
    project.save(document, checkpoint=True)
    reopened = project.open()
    saved_guide = reopened.guides[guide_id]
    assert saved_guide.lifecycle == "consumed"
    assert saved_guide.state_revision == guide.state_revision
    assert reopened.current_revision == document.current_revision


def test_relational_context_respects_owner_locks_and_keeps_failed_updates_atomic(tmp_path) -> None:
    document, store, _project, source_id, _asset_id, _raw = _scene(tmp_path)
    document, _ = _apply(document, store, "create_context_mask", {
        "sourceLayerId": source_id,
        "seeds": [{"kind": "box", "geometry": {"x": 1, "y": 1, "width": 8, "height": 6}}],
    })
    mask = next(item for item in document.masks.values() if item.purpose == "context")
    document, _ = _apply(document, store, "set_layer_lock", {"layerId": source_id, "locked": True})

    before_document = document.to_dict()
    before_files = {
        path.relative_to(store.project_root).as_posix(): path.read_bytes()
        for path in store.project_root.rglob("*") if path.is_file()
    }
    with pytest.raises(EditorActionError) as error:
        prepare_action(document, store, _payload(document, "set_relational_context", {
            "contextMaskId": mask.mask_id, "referenceIds": [], "dilationPx": 3,
            "editMaskId": None, "editTargetIds": [],
        }), actor_id=ACTOR)
    assert error.value.code == "LAYER_LOCKED"
    assert document.to_dict() == before_document
    assert {
        path.relative_to(store.project_root).as_posix(): path.read_bytes()
        for path in store.project_root.rglob("*") if path.is_file()
    } == before_files


def test_relational_context_respects_locked_ancestor_and_object_owner(tmp_path) -> None:
    document, store, _project, source_id, _asset_id, _raw = _scene(tmp_path)
    document, _ = _apply(document, store, "add_layer", {"kind": "group", "name": "Locked parent"})
    group_id = document.root_layer_ids[-1]
    document, _ = _apply(document, store, "reparent_layer", {"layerId": source_id, "parentId": group_id})
    document, _ = _apply(document, store, "create_context_mask", {
        "sourceLayerId": source_id,
        "seeds": [{"kind": "box", "geometry": {"x": 1, "y": 1, "width": 8, "height": 6}}],
    })
    mask = next(item for item in document.masks.values() if item.purpose == "context")
    document, _ = _apply(document, store, "set_layer_lock", {"layerId": group_id, "locked": True})
    before = document.to_dict()
    with pytest.raises(EditorActionError) as error:
        prepare_action(document, store, _payload(document, "set_relational_context", {
            "contextMaskId": mask.mask_id, "referenceIds": [], "dilationPx": 0,
            "editMaskId": None, "editTargetIds": [],
        }), actor_id=ACTOR)
    assert error.value.code == "LAYER_LOCKED"
    assert document.to_dict() == before

    # An object owner resolves through its own layer, so its ancestor lock also
    # refuses the context-lineage mutation.
    document, store, _project, source_id, _asset_id, _raw = _scene(tmp_path / "object-owner")
    object_id = next(iter(document.objects))
    document, _ = _apply(document, store, "create_context_mask", {
        "seeds": [{"kind": "box", "geometry": {"x": 1, "y": 1, "width": 8, "height": 6}}],
    })
    mask = next(item for item in document.masks.values() if item.purpose == "context")
    document.masks[mask.mask_id].owner_id = object_id
    document, _ = _apply(document, store, "set_layer_lock", {"layerId": source_id, "locked": True})
    before = document.to_dict()
    with pytest.raises(EditorActionError) as error:
        prepare_action(document, store, _payload(document, "set_relational_context", {
            "contextMaskId": mask.mask_id, "referenceIds": [], "dilationPx": 0,
            "editMaskId": None, "editTargetIds": [],
        }), actor_id=ACTOR)
    assert error.value.code == "LAYER_LOCKED"
    assert document.to_dict() == before


def test_relational_context_allows_unlocked_owner_locked_reference_and_ownerless_mask(tmp_path) -> None:
    document, store, _project, source_id, _asset_id, _raw = _scene(tmp_path)
    document, _ = _apply(document, store, "add_layer", {"kind": "vector", "name": "Locked reference"})
    reference_id = document.root_layer_ids[-1]
    document, _ = _apply(document, store, "create_context_mask", {
        "sourceLayerId": source_id,
        "seeds": [{"kind": "box", "geometry": {"x": 1, "y": 1, "width": 8, "height": 6}}],
    })
    owned_mask = next(item for item in document.masks.values() if item.purpose == "context")
    document, _ = _apply(document, store, "set_layer_lock", {"layerId": reference_id, "locked": True})
    document, _ = _apply(document, store, "set_relational_context", {
        "contextMaskId": owned_mask.mask_id, "referenceIds": [reference_id], "dilationPx": 2,
        "editMaskId": None, "editTargetIds": [source_id],
    })
    assert document.masks[owned_mask.mask_id].lineage["relationalReferences"] == [reference_id]

    document, _ = _apply(document, store, "create_context_mask", {
        "seeds": [{"kind": "box", "geometry": {"x": 3, "y": 2, "width": 5, "height": 4}}],
    })
    ownerless = next(item for item in document.masks.values() if item.purpose == "context" and item.owner_id is None)
    document, _ = _apply(document, store, "set_relational_context", {
        "contextMaskId": ownerless.mask_id, "referenceIds": [], "dilationPx": 4,
        "editMaskId": None, "editTargetIds": [],
    })
    assert document.masks[ownerless.mask_id].lineage["dilationPx"] == 4


def test_imported_image_can_be_a_proposed_semantic_guide(tmp_path) -> None:
    document, store, _project, _source_id, _asset_id, raw = _scene(tmp_path)
    prepared = prepare_raster_import(
        document,
        store,
        _payload(document, "import_image", {"asGuide": True, "semanticRole": "color-reference"}),
        raw,
        actor_id=ACTOR,
        filename="reference.png",
    )
    publish_pending_assets(store, prepared)
    imported = prepared.document
    guide = next(iter(imported.guides.values()))
    layer = imported.layers[imported.objects[guide.object_ids[0]].layer_id]
    assert guide.lifecycle == "proposed"
    assert guide.semantic_role == "color-reference"
    assert layer.kind == "guide"
    assert layer.role == "color-reference"


def _locked_action(action_type, layer_id, *, data=None):
    payload_data = {"layerId": layer_id, **(data or {})}
    return {"actionType": action_type, "data": payload_data}


def test_locked_targets_and_ancestors_refuse_mutations_except_self_unlock(tmp_path) -> None:
    document, store, _project, source_id, _asset_id, _raw = _scene(tmp_path)
    document, _ = _apply(document, store, "create_selection", {
        "sourceLayerId": source_id,
        "seeds": [{"kind": "box", "geometry": {"x": 1, "y": 1, "width": 8, "height": 8}}],
        "semanticHint": "protected subject",
    })
    selection = next(iter(document.selections.values()))
    document, _ = _apply(document, store, "add_layer", {"kind": "group", "name": "Protected group"})
    group_id = document.root_layer_ids[-1]
    document, _ = _apply(document, store, "add_layer", {
        "kind": "vector", "name": "Protected child", "parentId": group_id,
    })
    child_id = document.layers[group_id].child_ids[0]
    document, _ = _apply(document, store, "set_layer_lock", {"layerId": source_id, "locked": True})
    before_locked_source = document.to_dict()
    blocked_source_actions = [
        _locked_action("rename_layer", source_id, data={"name": "Changed"}),
        _locked_action("set_layer_visibility", source_id, data={"visible": False}),
        _locked_action("set_layer_opacity", source_id, data={"opacity": 0.5}),
        _locked_action("reorder_layer", source_id, data={"direction": "up"}),
        {"actionType": "transform",
         "data": {"targetIds": [source_id], "transform": [1, 0, 1, 0, 1, 1, 0, 0, 1]}},
        _locked_action("duplicate_layer", source_id),
        _locked_action("delete_layer", source_id, data={"confirmed": True}),
        {"actionType": "create_object",
         "data": {"layerId": source_id, "kind": "shape",
                  "geometry": {"shape": "rectangle", "x": 1, "y": 1, "width": 4, "height": 4},
                  "style": {"stroke": "#FFFFFF"}}},
        _locked_action("refine_selection", source_id, data={"selectionId": selection.selection_id,
                       "expectedSelectionRevision": selection.selection_revision, "operation": "grow", "radius": 1}),
        _locked_action("rebase_selection", source_id, data={"selectionId": selection.selection_id,
                       "expectedSelectionRevision": selection.selection_revision, "geometryOnly": True}),
        _locked_action("duplicate_selection_to_layer", source_id, data={"selectionId": selection.selection_id,
                       "expectedSelectionRevision": selection.selection_revision}),
    ]
    for action in blocked_source_actions:
        with pytest.raises(EditorActionError) as error:
            prepare_action(document, store, _payload(document, action["actionType"], action.get("data"),
                                                     actions=action if action["actionType"] == "batch" else None),
                           actor_id=ACTOR)
        assert error.value.code == "LAYER_LOCKED"
        assert document.to_dict() == before_locked_source

    # An explicitly targeted locked layer can be unlocked when no ancestor is
    # locked; no other edit receives this exception.
    document, _ = _apply(document, store, "set_layer_lock", {"layerId": source_id, "locked": False})
    assert not document.layers[source_id].locked

    document, _ = _apply(document, store, "set_layer_lock", {"layerId": group_id, "locked": True})
    before_locked_group = document.to_dict()
    with pytest.raises(EditorActionError) as error:
        prepare_action(document, store, _payload(document, "set_layer_lock", {
            "layerId": child_id, "locked": True,
        }), actor_id=ACTOR)
    assert error.value.code == "LAYER_LOCKED"
    assert document.to_dict() == before_locked_group
    with pytest.raises(EditorActionError) as error:
        prepare_action(document, store, _payload(document, "set_layer_lock", {
            "layerId": child_id, "locked": False,
        }), actor_id=ACTOR)
    assert error.value.code == "LAYER_LOCKED"
    assert document.to_dict() == before_locked_group

    # Unlocking the group itself is the narrow exception; children can then be
    # locked and unlocked under the now-unlocked parent.
    document, _ = _apply(document, store, "set_layer_lock", {"layerId": group_id, "locked": False})
    document, _ = _apply(document, store, "set_layer_lock", {"layerId": child_id, "locked": True})
    document, _ = _apply(document, store, "set_layer_lock", {"layerId": child_id, "locked": False})
    assert not document.layers[child_id].locked


def _pending_asset(data: bytes, *, asset_id: str) -> PendingAsset:
    digest = hashlib.sha256(data).hexdigest()
    record = AssetRecord(
        asset_id, digest, "application/octet-stream",
        f"assets/sha256/{digest[:2]}/{digest}.bin", "bin", len(data),
    )
    return PendingAsset(data, record)


def test_asset_batch_rolls_back_only_new_blobs_and_can_retry(tmp_path, monkeypatch) -> None:
    _document, store, project_path, _layer_id, _asset_id, _raw = _scene(tmp_path)
    first = _pending_asset(b"new-first-asset", asset_id=make_id("asset"))
    second = _pending_asset(b"new-second-asset", asset_id=make_id("asset"))
    prepared = PreparedAction(_document, {}, [first, second])
    real_link = os.link
    calls = 0

    def fail_second_link(source, destination, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publication failure")
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(asset_store_module.os, "link", fail_second_link)
    with pytest.raises(EditorActionError) as error:
        publish_pending_assets(store, prepared)
    assert error.value.code == "ASSET_PUBLICATION_FAILED"
    for pending in (first, second):
        path = project_path / pending.record.storage_uri
        assert not path.exists()
    assert not list(project_path.rglob("*.stage"))

    monkeypatch.setattr(asset_store_module.os, "link", real_link)
    publish_pending_assets(store, prepared)
    assert store.read_bytes(first.record) == first.content
    assert store.read_bytes(second.record) == second.content


def test_asset_batch_failure_never_removes_preexisting_deduplicated_content(tmp_path, monkeypatch) -> None:
    _document, store, project_path, _layer_id, _asset_id, _raw = _scene(tmp_path)
    existing = _pending_asset(b"already-published", asset_id=make_id("asset"))
    fresh = _pending_asset(b"second-fresh-asset", asset_id=make_id("asset"))
    store.put_bytes(
        existing.content,
        media_type=existing.record.media_type,
        extension=existing.record.extension,
        asset_id=existing.record.asset_id,
    )
    existing_path = project_path / existing.record.storage_uri
    original = existing_path.read_bytes()
    real_link = os.link

    def fail_first_new_link(source, destination, *args, **kwargs):
        if Path(destination).name.startswith(fresh.record.content_hash):
            raise OSError("injected publication failure")
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(asset_store_module.os, "link", fail_first_new_link)
    with pytest.raises(EditorActionError) as error:
        publish_pending_assets(store, PreparedAction(_document, {}, [existing, fresh]))
    assert error.value.code == "ASSET_PUBLICATION_FAILED"
    assert existing_path.read_bytes() == original
    assert not (project_path / fresh.record.storage_uri).exists()


def test_asset_batch_rolls_back_when_verification_fails_after_all_links(tmp_path, monkeypatch) -> None:
    _document, store, project_path, _layer_id, _asset_id, _raw = _scene(tmp_path)
    pending = [
        _pending_asset(b"post-link-first", asset_id=make_id("asset")),
        _pending_asset(b"post-link-second", asset_id=make_id("asset")),
    ]
    real_assert = store._assert_asset_path
    second_path = project_path / pending[1].record.storage_uri

    def fail_after_second_link(path):
        real_assert(path)
        if Path(path) == second_path and second_path.exists():
            raise OSError("injected post-link verification failure")

    monkeypatch.setattr(store, "_assert_asset_path", fail_after_second_link)
    with pytest.raises(EditorActionError) as error:
        publish_pending_assets(store, PreparedAction(_document, {}, pending))
    assert error.value.code == "ASSET_PUBLICATION_FAILED"
    assert all(not (project_path / item.record.storage_uri).exists() for item in pending)
    assert not list(project_path.rglob("*.stage"))


def test_selection_seed_allowlist_paint_refinement_and_semantic_hint(tmp_path) -> None:
    document, store, _project, source_id, source_asset_id, _raw = _scene(tmp_path)
    layer = document.layers[source_id]
    allowed_id = make_id("mask")
    blocked_id = make_id("mask")
    layer.mask_ids.extend([allowed_id, blocked_id])
    document.masks[allowed_id] = MaskRecord(allowed_id, "layer-alpha", "document", 0, asset_id=source_asset_id,
                                            owner_id=source_id, content_hash=document.assets[source_asset_id].content_hash)
    document.masks[blocked_id] = MaskRecord(blocked_id, "context", "document", 0, asset_id=source_asset_id,
                                            owner_id=source_id, content_hash=document.assets[source_asset_id].content_hash)

    document, _ = _apply(document, store, "create_selection", {
        "sourceLayerId": source_id,
        "seeds": [{"kind": "paint", "geometry": {"points": [[3, 3], [8, 8]], "size": 5}}],
        "semanticHint": "painted foreground subject",
    })
    selection = next(iter(document.selections.values()))
    assert selection.semantic_hint == "painted foreground subject"
    assert selection.bounds is not None
    document, _ = _apply(document, store, "refine_selection", {
        "selectionId": selection.selection_id,
        "expectedSelectionRevision": selection.selection_revision,
        "operation": "subtract",
        "seed": {"kind": "paint", "geometry": {"points": [[5, 4], [5, 7]], "size": 2}},
    })
    selection = document.selections[selection.selection_id]
    assert selection.refinement_history[-1]["details"]["operation"] == "subtract"
    assert selection.selection_revision == 2

    document, _ = _apply(document, store, "create_selection", {
        "sourceLayerId": source_id, "seeds": [{"kind": "alpha"}], "semanticHint": "alpha region",
    })
    document, _ = _apply(document, store, "create_selection", {
        "sourceLayerId": source_id, "seeds": [{"kind": "mask", "maskId": allowed_id}],
    })
    before = document.to_dict()
    with pytest.raises(EditorActionError) as error:
        prepare_action(document, store, _payload(document, "create_selection", {
            "sourceLayerId": source_id, "seeds": [{"kind": "mask", "maskId": blocked_id}],
        }), actor_id=ACTOR)
    assert error.value.code == "INVALID_SELECTION_SEED"
    assert document.to_dict() == before
