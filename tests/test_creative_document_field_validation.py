from __future__ import annotations

import copy
import hashlib
import io
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from modules.creative_document import (
    AffineTransform,
    CoordinateTransform,
    Document,
    LayerRecord,
    MaskRecord,
    ObjectRecord,
    ProjectStore,
    SchemaValidationError,
    make_id,
)
from modules.creative_document.command_service import ActorContext, CommandService, CommandServiceError, _canonical_target_ids
from modules.creative_document.editor_render import render_document_png


AGENT = ActorContext("agent", "agent-w05-field-test", frozenset({"inspect", "propose", "mutate"}))
DIRECTOR = ActorContext("human", "director-w05-field-test", frozenset({"inspect", "propose", "mutate"}))


def _png(size: tuple[int, int], color: tuple[int, int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGBA", size, color).save(output, format="PNG")
    return output.getvalue()


def _mask_png(size: tuple[int, int]) -> bytes:
    image = Image.new("L", size, 0)
    ImageDraw = __import__("PIL.ImageDraw", fromlist=["ImageDraw"]).Draw
    ImageDraw(image).rectangle((8, 6, size[0] - 5, size[1] - 5), fill=255)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _scene(tmp_path: Path) -> SimpleNamespace:
    width, height = 64, 48
    store = ProjectStore(tmp_path / "scene.nexscene")
    plate_id, depth_id, reference_id = make_id("layer"), make_id("layer"), make_id("layer")
    plate_asset = store.assets.put_bytes(_png((width, height), (30, 75, 118, 255)),
                                         media_type="image/png", extension="png",
                                         width=width, height=height, has_alpha=True)
    depth_asset = store.assets.put_bytes(_png((28, 20), (190, 122, 46, 255)),
                                         media_type="image/png", extension="png",
                                         width=28, height=20, has_alpha=True)
    edit_mask_asset = store.assets.put_bytes(_mask_png((width, height)),
                                             media_type="image/png", extension="png",
                                             width=width, height=height, has_alpha=False)

    plate = LayerRecord(plate_id, "Complete plate", "raster", role="complete-plate",
                        asset_ids=[plate_asset.asset_id])
    depth_object_id = make_id("obj")
    depth = LayerRecord(
        depth_id, "Registered depth element", "raster", opacity=0.72,
        object_ids=[depth_object_id], role="depth-element", depth_element=True,
        complete_plate_id=plate_id,
        registration=CoordinateTransform.from_forward(
            "source-crop", "document", (28, 20), (width, height),
            AffineTransform.translation(18, 14), operation_revision=0,
        ),
    )
    depth_object = ObjectRecord(
        depth_object_id, depth_id, "raster-placement", "document",
        {"x": 18, "y": 14, "width": 28, "height": 20}, asset_id=depth_asset.asset_id,
    )
    reference_shape_id = make_id("obj")
    reference = LayerRecord(reference_id, "Relational reference", "vector", role="context-reference",
                            object_ids=[reference_shape_id])
    reference_shape = ObjectRecord(
        reference_shape_id, reference_id, "shape", "document",
        {"shape": "ellipse", "x": 6, "y": 8, "width": 10, "height": 10},
        style={"fill": "#E8C85A", "stroke": "#30260A", "strokeWidth": 1, "opacity": 1},
    )
    edit_mask_id = make_id("mask")
    edit_mask = MaskRecord(
        edit_mask_id, "generation", "document", 0, asset_id=edit_mask_asset.asset_id,
        owner_id=depth_id, lineage={"fixture": "w05-explicit-edit-permission"},
        content_hash=edit_mask_asset.content_hash,
    )
    depth.mask_ids.append(edit_mask_id)
    document = Document(
        make_id("doc"), width, height,
        root_layer_ids=[plate_id, depth_id, reference_id],
        layers={plate_id: plate, depth_id: depth, reference_id: reference},
        objects={depth_object_id: depth_object, reference_shape_id: reference_shape},
        assets={asset.asset_id: asset for asset in (plate_asset, depth_asset, edit_mask_asset)},
        masks={edit_mask_id: edit_mask},
    )
    document.refresh_digest()
    document.validate()
    return SimpleNamespace(
        document=document,
        store=store,
        committed_revision=0,
        owner_principal="owner-w05-test-principal",
        lock=threading.RLock(),
        recovery_notice=None,
        plate_id=plate_id,
        depth_id=depth_id,
        reference_id=reference_id,
        edit_mask_id=edit_mask_id,
    )


def _envelope(
    handle: SimpleNamespace,
    command_type: str,
    *,
    data: dict | None = None,
    intent: str = "mutate",
    expected_revision: int | None = None,
    actions: list[dict] | None = None,
) -> dict:
    is_inspect = intent == "inspect"
    if is_inspect:
        payload: dict = {}
        transaction = None
        targets: list[str] = []
    elif command_type in {"undo", "redo", "save_scene"}:
        payload = {}
        transaction = {"transactionId": make_id("txn"), "groupId": make_id("grp"), "phase": "commit"}
        targets = []
    elif command_type == "batch":
        payload = {"actions": actions or []}
        transaction = {"transactionId": make_id("txn"), "groupId": make_id("grp"), "phase": "commit"}
        targets = list(_canonical_target_ids(command_type, payload, ()))
    else:
        payload = {"data": data or {}}
        transaction = {"transactionId": make_id("txn"), "groupId": make_id("grp"), "phase": "commit"}
        targets = list(_canonical_target_ids(command_type, payload, ()))
    return {
        "schemaVersion": 1,
        "commandId": make_id("cmd"),
        "documentId": handle.document.document_id,
        "intent": intent,
        "expectedRevision": None if is_inspect else (
            handle.document.current_revision if expected_revision is None else expected_revision
        ),
        "commandType": command_type,
        "targetIds": targets,
        "coordinateSpace": "document",
        "payload": payload,
        "transaction": transaction,
    }


def _execute(
    service: CommandService,
    handle: SimpleNamespace,
    command_type: str,
    *,
    data: dict | None = None,
    intent: str = "mutate",
    actor: ActorContext = AGENT,
    expected_revision: int | None = None,
    actions: list[dict] | None = None,
) -> dict:
    return service.execute(handle, _envelope(
        handle, command_type, data=data, intent=intent, expected_revision=expected_revision, actions=actions,
    ), actor)


def _files(path: Path) -> list[tuple[str, str]]:
    return sorted(
        (item.relative_to(path).as_posix(), hashlib.sha256(item.read_bytes()).hexdigest())
        for item in path.rglob("*") if item.is_file()
    ) if path.exists() else []


def _render(handle: SimpleNamespace) -> bytes:
    return render_document_png(handle.document, lambda identity: handle.store.assets.read_bytes(handle.document.assets[identity]))


def _guide_data(name: str, role: str, geometry: dict, *, fill: str) -> dict:
    return {
        "name": name,
        "semanticRole": role,
        "kind": "shape",
        "geometry": geometry,
        "style": {"fill": fill, "stroke": "#141414", "strokeWidth": 1, "opacity": 0.9},
    }


def test_marker_and_semantic_anchor_are_distinct_and_lifecycle_is_explicit(tmp_path: Path) -> None:
    handle = _scene(tmp_path)
    service = CommandService()
    marker = _execute(service, handle, "create_guide", actor=AGENT, data=_guide_data(
        "Position marker", "position only; no subject identity",
        {"shape": "ellipse", "x": 22, "y": 12, "width": 12, "height": 12}, fill="#F2D64B",
    ))
    anchor = _execute(service, handle, "create_guide", actor=AGENT, data=_guide_data(
        "Focal person anchor", "a person-shaped focal anchor that nearby figures attend to",
        {"shape": "polygon", "points": [[42, 10], [49, 10], [52, 16], [50, 20], [55, 29], [47, 27], [43, 34], [40, 25], [36, 28], [39, 19], [37, 15]]},
        fill="#47C78B",
    ))
    assert marker["status"] == anchor["status"] == "committed"
    marker_id = next(identity for identity in marker["createdIds"] if identity.startswith("guide-"))
    anchor_id = next(identity for identity in anchor["createdIds"] if identity.startswith("guide-"))
    marker_record = handle.document.guides[marker_id]
    anchor_record = handle.document.guides[anchor_id]
    assert marker_record.object_ids != anchor_record.object_ids
    assert marker_record.semantic_role != anchor_record.semantic_role
    assert marker_record.lifecycle == anchor_record.lifecycle == "proposed"

    inspected = _execute(service, handle, "inspect_document", intent="inspect", actor=AGENT)
    projection = inspected["result"]
    assert inspected["writes"] == 0 and inspected["observedRevision"] == 2
    assert projection["guides"][marker_id]["lifecycle"] == "proposed"
    assert projection["guides"][marker_id]["semanticRole"] == "position only; no subject identity"
    assert projection["guides"][anchor_id]["lifecycle"] == "proposed"
    assert projection["guides"][anchor_id]["semanticRole"].startswith("a person-shaped focal anchor")
    marker_object = projection["objects"][marker_record.object_ids[0]]
    anchor_object = projection["objects"][anchor_record.object_ids[0]]
    assert marker_object["geometry"] != anchor_object["geometry"]

    _execute(service, handle, "set_layer_visibility", actor=DIRECTOR, data={
        "layerId": handle.document.objects[marker_record.object_ids[0]].layer_id, "visible": False,
    })
    saved = _execute(service, handle, "save_scene", actor=DIRECTOR)
    assert saved["status"] == "saved"
    assert handle.document.guides[marker_id].lifecycle == "proposed"
    assert handle.document.guides[anchor_id].lifecycle == "proposed"

    conversational = _envelope(handle, "save_scene", expected_revision=handle.document.current_revision)
    conversational["commandType"] = "conversational_text"
    conversational["payload"] = {"text": f"activate guide {anchor_id}"}
    before_document, before_files = handle.document.to_dict(), _files(handle.store.path)
    refusal = service.execute(handle, conversational, AGENT)
    assert refusal["error"]["code"] == "UNSUPPORTED_COMMAND" and refusal["writes"] == 0
    assert handle.document.to_dict() == before_document and _files(handle.store.path) == before_files

    active = _execute(service, handle, "transition_guide", actor=DIRECTOR,
                      data={"guideId": anchor_id, "lifecycle": "active"})
    assert active["newRevision"] == before_document["currentRevision"] + 1
    replacement = _execute(service, handle, "create_guide", actor=DIRECTOR, data=_guide_data(
        "Tighter focal guide", "same focal person, tighter pose cue",
        {"shape": "polygon", "points": [[44, 12], [49, 12], [52, 19], [48, 23], [43, 21]]}, fill="#66D6A0",
    ))
    replacement_id = next(identity for identity in replacement["createdIds"] if identity.startswith("guide-"))
    pending = _execute(service, handle, "transition_guide", actor=DIRECTOR, data={
        "guideId": anchor_id, "lifecycle": "replacement-pending", "replacementId": replacement_id,
    })
    activated_replacement = _execute(service, handle, "transition_guide", actor=DIRECTOR,
                                     data={"guideId": replacement_id, "lifecycle": "active"})
    retired = _execute(service, handle, "transition_guide", actor=DIRECTOR,
                       data={"guideId": anchor_id, "lifecycle": "safe-to-remove"})
    assert [item["newRevision"] for item in (pending, activated_replacement, retired)] == sorted(
        item["newRevision"] for item in (pending, activated_replacement, retired)
    )
    assert handle.document.guides[anchor_id].lifecycle == "safe-to-remove"
    assert handle.document.guides[anchor_id].replacement_id == replacement_id
    assert handle.document.guides[replacement_id].lifecycle == "active"
    assert handle.document.guides[replacement_id].supersedes_id == anchor_id

    _execute(service, handle, "save_scene", actor=DIRECTOR)
    reopened = ProjectStore(handle.store.path).open()
    assert reopened.guides[marker_id].lifecycle == "proposed"
    assert reopened.guides[anchor_id].lifecycle == "safe-to-remove"
    assert reopened.guides[replacement_id].lifecycle == "active"
    assert reopened.guides[replacement_id].supersedes_id == anchor_id
    assert reopened.guides[anchor_id].replacement_id == replacement_id
    assert [tx["resultingRevision"] for tx in reopened.history] == sorted(tx["resultingRevision"] for tx in reopened.history)
    assert {tx["actorId"] for tx in reopened.history}.issuperset({AGENT.actor_id, DIRECTOR.actor_id})


def test_relational_context_and_edit_scope_are_distinct_and_refusals_write_nothing(tmp_path: Path) -> None:
    handle = _scene(tmp_path)
    service = CommandService()
    context = _execute(service, handle, "create_context_mask", actor=AGENT, data={
        "sourceLayerId": None,
        "seeds": [{"kind": "box", "geometry": {"x": 4, "y": 3, "width": 54, "height": 40}}],
    })
    context_id = next(identity for identity in context["createdIds"] if identity.startswith("mask-"))

    _execute(service, handle, "set_layer_lock", actor=DIRECTOR, data={
        "layerId": handle.reference_id, "locked": True,
    })
    configured = _execute(service, handle, "set_relational_context", actor=AGENT, data={
        "contextMaskId": context_id,
        "referenceIds": [handle.reference_id],
        "dilationPx": 24,
        "editMaskId": handle.edit_mask_id,
        "editTargetIds": [handle.depth_id],
    })
    assert configured["status"] == "committed"
    lineage = handle.document.masks[context_id].lineage
    assert lineage["relationalReferences"] == [handle.reference_id]
    assert lineage["editTargetIds"] == [handle.depth_id]
    assert lineage["editMaskId"] == handle.edit_mask_id
    assert lineage["dilationPx"] == 24
    assert handle.document.layers[handle.reference_id].locked is True

    inspected = _execute(service, handle, "inspect_document", intent="inspect", actor=AGENT)
    summary = next(item for item in inspected["result"]["relationalContext"] if item["contextMaskId"] == context_id)
    assert summary["referenceIds"] == [handle.reference_id]
    assert summary["editTargetIds"] == [handle.depth_id]
    assert summary["dilationPx"] == 24

    def assert_zero_write_refusal(receipt: dict, before: dict, files_before: list[tuple[str, str]]) -> None:
        assert receipt["writes"] == 0
        assert handle.document.to_dict() == before
        assert _files(handle.store.path) == files_before

    invalid_cases = [
        ({"referenceIds": [handle.depth_id], "editTargetIds": [handle.depth_id]}, "CONTEXT_EDIT_SCOPE_OVERLAP"),
        ({"referenceIds": [make_id("layer")], "editTargetIds": [handle.depth_id]}, "INVALID_CONTEXT_REFERENCES"),
        ({"referenceIds": "malformed", "editTargetIds": [handle.depth_id]}, "INVALID_TARGETS"),
    ]
    for update, code in invalid_cases:
        data = {
            "contextMaskId": context_id,
            "referenceIds": [handle.reference_id],
            "dilationPx": 24,
            "editMaskId": handle.edit_mask_id,
            "editTargetIds": [handle.depth_id],
            **update,
        }
        before, files_before = handle.document.to_dict(), _files(handle.store.path)
        refused = _execute(service, handle, "set_relational_context", actor=AGENT, data=data)
        assert refused["error"]["code"] == code
        assert_zero_write_refusal(refused, before, files_before)

    stale_revision = handle.document.current_revision - 1
    before, files_before = handle.document.to_dict(), _files(handle.store.path)
    stale = _execute(service, handle, "set_relational_context", actor=AGENT,
                     expected_revision=stale_revision, data={
                         "contextMaskId": context_id, "referenceIds": [handle.reference_id],
                         "dilationPx": 25, "editMaskId": handle.edit_mask_id,
                         "editTargetIds": [handle.depth_id],
                     })
    assert stale["error"]["code"] == "STALE_DOCUMENT_REVISION"
    assert_zero_write_refusal(stale, before, files_before)

    owned = _execute(service, handle, "create_context_mask", actor=AGENT, data={
        "sourceLayerId": handle.plate_id,
        "seeds": [{"kind": "box", "geometry": {"x": 5, "y": 4, "width": 40, "height": 32}}],
    })
    owned_id = next(identity for identity in owned["createdIds"] if identity.startswith("mask-"))
    _execute(service, handle, "set_layer_lock", actor=DIRECTOR, data={"layerId": handle.plate_id, "locked": True})
    before, files_before = handle.document.to_dict(), _files(handle.store.path)
    locked = _execute(service, handle, "set_relational_context", actor=AGENT, data={
        "contextMaskId": owned_id, "referenceIds": [handle.reference_id], "dilationPx": 10,
        "editMaskId": handle.edit_mask_id, "editTargetIds": [handle.depth_id],
    })
    assert locked["error"]["code"] == "LAYER_LOCKED"
    assert_zero_write_refusal(locked, before, files_before)


def test_broad_to_tight_exchange_refreshes_exact_revision_and_preserves_lineage(tmp_path: Path) -> None:
    handle = _scene(tmp_path)
    service = CommandService()
    initial = _execute(service, handle, "inspect_document", intent="inspect", actor=AGENT)
    assert initial["result"]["observedRevision"] == 0
    broad = _execute(service, handle, "batch", actor=AGENT, actions=[
        {"actionType": "create_guide", "data": _guide_data(
            "Broad scene anchor", "broad focal person and nearby relationship",
            {"shape": "polygon", "points": [[12, 6], [30, 6], [42, 19], [29, 37], [11, 32]]}, fill="#DF9850",
        )},
        {"actionType": "create_context_mask", "data": {
            "sourceLayerId": None,
            "seeds": [{"kind": "box", "geometry": {"x": 4, "y": 3, "width": 54, "height": 40}}],
        }},
    ])
    broad_guide_id = next(identity for identity in broad["createdIds"] if identity.startswith("guide-"))
    context_id = next(identity for identity in broad["createdIds"] if identity.startswith("mask-"))
    assert broad["previousRevision"] == 0 and broad["newRevision"] == 1

    activated = _execute(service, handle, "transition_guide", actor=DIRECTOR, data={
        "guideId": broad_guide_id, "lifecycle": "active",
    })
    exact_director_view = _execute(service, handle, "inspect_document", intent="inspect", actor=AGENT,
                                   expected_revision=activated["newRevision"])
    assert exact_director_view["status"] == "ok"
    assert exact_director_view["result"]["guides"][broad_guide_id]["lifecycle"] == "active"
    assert exact_director_view["observedRevision"] == 2

    tight = _execute(service, handle, "batch", actor=AGENT, actions=[
        {"actionType": "create_guide", "data": _guide_data(
            "Tight pose refinement", "tight pose cue within the retained broad composition",
            {"shape": "polygon", "points": [[21, 11], [27, 11], [31, 20], [26, 29], [20, 24]]}, fill="#63D8A2",
        )},
        {"actionType": "set_relational_context", "data": {
            "contextMaskId": context_id, "referenceIds": [handle.reference_id],
            "dilationPx": 18, "editMaskId": handle.edit_mask_id,
            "editTargetIds": [handle.depth_id],
        }},
    ])
    tight_guide_id = next(identity for identity in tight["createdIds"] if identity.startswith("guide-"))
    assert tight["previousRevision"] == 2 and tight["newRevision"] == 3
    assert handle.document.guides[broad_guide_id].lifecycle == "active"
    assert handle.document.guides[tight_guide_id].lifecycle == "proposed"

    intervening = _execute(service, handle, "set_layer_visibility", actor=DIRECTOR, data={
        "layerId": handle.depth_id, "visible": False,
    })
    before, files_before = handle.document.to_dict(), _files(handle.store.path)
    stale = _execute(service, handle, "create_guide", actor=AGENT, expected_revision=3,
                     data=_guide_data("Stale attempt", "must be refused", {
                         "shape": "ellipse", "x": 5, "y": 5, "width": 5, "height": 5,
                     }, fill="#FFFFFF"))
    assert stale["error"]["code"] == "STALE_DOCUMENT_REVISION"
    assert stale["writes"] == 0 and handle.document.to_dict() == before
    assert _files(handle.store.path) == files_before

    refreshed = _execute(service, handle, "inspect_document", intent="inspect", actor=AGENT,
                         expected_revision=intervening["newRevision"])
    assert refreshed["result"]["guides"][broad_guide_id]["lifecycle"] == "active"
    assert refreshed["result"]["guides"][tight_guide_id]["lifecycle"] == "proposed"
    continuation = _execute(service, handle, "create_guide", actor=AGENT,
                            expected_revision=refreshed["observedRevision"], data=_guide_data(
                                "Refreshed continuation", "continues the tightened focal cue",
                                {"shape": "ellipse", "x": 23, "y": 15, "width": 4, "height": 5}, fill="#9CDDF2",
                            ))
    continuation_id = next(identity for identity in continuation["createdIds"] if identity.startswith("guide-"))
    assert continuation["previousRevision"] == 4 and continuation["newRevision"] == 5

    relation = handle.document.masks[context_id].lineage
    assert relation["dilationPx"] == 18 and relation["editTargetIds"] == [handle.depth_id]
    _execute(service, handle, "save_scene", actor=DIRECTOR)
    reopened = ProjectStore(handle.store.path).open()
    assert {broad_guide_id, tight_guide_id, continuation_id}.issubset(reopened.guides)
    assert reopened.guides[broad_guide_id].lifecycle == "active"
    assert reopened.guides[tight_guide_id].lifecycle == "proposed"
    assert reopened.guides[continuation_id].created_revision == 5
    assert reopened.masks[context_id].lineage["editTargetIds"] == [handle.depth_id]
    assert reopened.masks[context_id].lineage["dilationPx"] == 18
    revisions = [(tx["previousRevision"], tx["resultingRevision"]) for tx in reopened.history]
    assert revisions == sorted(revisions) and all(new == old + 1 for old, new in revisions)
    actor_ids = {tx["actorId"] for tx in reopened.history}
    assert actor_ids == {AGENT.actor_id, DIRECTOR.actor_id}


def test_complete_plate_depth_visibility_undo_redo_stale_and_two_reopens(tmp_path: Path) -> None:
    handle = _scene(tmp_path)
    service = CommandService()
    plate = handle.document.layers[handle.plate_id]
    depth = handle.document.layers[handle.depth_id]
    assert depth.depth_element is True
    assert depth.complete_plate_id == plate.layer_id != depth.layer_id
    assert depth.complete_plate_id in handle.document.layers
    assert handle.document.layers[depth.complete_plate_id].depth_element is False
    assert depth.registration is not None
    registration = copy.deepcopy(depth.registration.to_dict())
    element_transform = copy.deepcopy(depth.transform.to_dict())
    element_geometry = copy.deepcopy(handle.document.objects[depth.object_ids[0]].geometry)

    with pytest.raises(SchemaValidationError, match="cannot be its own complete plate"):
        malformed = copy.deepcopy(handle.document)
        malformed.layers[handle.depth_id].complete_plate_id = handle.depth_id
        malformed.validate()
    with pytest.raises(SchemaValidationError, match="complete plate"):
        malformed = copy.deepcopy(handle.document)
        malformed.layers[handle.depth_id].complete_plate_id = make_id("layer")
        malformed.validate()
    with pytest.raises(SchemaValidationError, match="another depth element"):
        malformed = copy.deepcopy(handle.document)
        other_id = make_id("layer")
        malformed.layers[other_id] = LayerRecord(other_id, "Other depth", "raster", depth_element=True,
                                                 complete_plate_id=handle.plate_id)
        malformed.root_layer_ids.append(other_id)
        malformed.layers[handle.depth_id].complete_plate_id = other_id
        malformed.validate()

    shown_before = _render(handle)
    shown_digest = hashlib.sha256(shown_before).hexdigest()
    hide = _execute(service, handle, "set_layer_visibility", actor=AGENT,
                    data={"layerId": handle.depth_id, "visible": False})
    hidden = _render(handle)
    with Image.open(io.BytesIO(hidden)) as image:
        assert image.mode == "RGBA" and image.getchannel("A").getextrema() == (255, 255)
    assert hashlib.sha256(hidden).hexdigest() != shown_digest

    files_before = _files(handle.store.path)
    stale = _execute(service, handle, "set_layer_visibility", actor=AGENT, expected_revision=0,
                     data={"layerId": handle.depth_id, "visible": True})
    assert stale["error"]["code"] == "STALE_DOCUMENT_REVISION" and stale["writes"] == 0
    assert _files(handle.store.path) == files_before
    assert handle.document.layers[handle.depth_id].visible is False

    undo = _execute(service, handle, "undo", actor=DIRECTOR)
    assert undo["status"] == "committed" and undo["newRevision"] == hide["newRevision"] + 1
    assert handle.document.layers[handle.depth_id].visible is True
    assert _render(handle) == shown_before
    redo = _execute(service, handle, "redo", actor=DIRECTOR)
    assert redo["status"] == "committed" and redo["newRevision"] == undo["newRevision"] + 1
    assert handle.document.layers[handle.depth_id].visible is False
    assert _render(handle) == hidden

    hidden_save = _execute(service, handle, "save_scene", actor=DIRECTOR)
    assert hidden_save["status"] == "saved"
    reopened_hidden = ProjectStore(handle.store.path).open()
    hidden_layer = reopened_hidden.layers[handle.depth_id]
    assert hidden_layer.visible is False and hidden_layer.complete_plate_id == handle.plate_id
    assert hidden_layer.registration.to_dict() == registration
    assert reopened_hidden.objects[hidden_layer.object_ids[0]].geometry == element_geometry
    handle.document = reopened_hidden
    show = _execute(service, handle, "set_layer_visibility", actor=AGENT,
                    data={"layerId": handle.depth_id, "visible": True})
    shown_after = _render(handle)
    assert shown_after == shown_before
    assert show["newRevision"] == hidden_save["currentRevision"] + 1
    assert handle.document.layers[handle.depth_id].transform.to_dict() == element_transform
    assert handle.document.layers[handle.depth_id].registration.to_dict() == registration

    shown_save = _execute(service, handle, "save_scene", actor=DIRECTOR)
    assert shown_save["status"] == "saved"
    reopened_shown = ProjectStore(handle.store.path).open()
    shown_layer = reopened_shown.layers[handle.depth_id]
    assert shown_layer.visible is True and shown_layer.complete_plate_id == handle.plate_id
    assert shown_layer.registration.to_dict() == registration
    assert reopened_shown.objects[shown_layer.object_ids[0]].geometry == element_geometry
    handle.document = reopened_shown
    assert hashlib.sha256(_render(handle)).hexdigest() == shown_digest


def _make_saved_blank_w03_scene(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    import secrets

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import modules.config
    import modules.creative_document_editor_api as editor_api
    from modules.creative_document_editor_api import CreativeDocumentRuntime

    monkeypatch.setattr(modules.config, "path_outputs", str(tmp_path))
    runtime = CreativeDocumentRuntime()
    monkeypatch.setattr(editor_api, "creative_document_runtime", runtime)
    app = FastAPI()
    app.include_router(editor_api.creative_document_router)
    client = TestClient(app)
    owner_key = secrets.token_urlsafe(32)
    session_id = "director-w05-preparation"
    capability = runtime.sessions.issue(session_id, None, remote_exposure=False)
    headers = {
        "Authorization": f"Bearer {capability['token']}",
        "X-Gradio-Session": session_id,
        "X-Editor-Owner-Key": owner_key,
        "Origin": "http://testserver",
        "Sec-Fetch-Site": "same-origin",
    }
    created = client.post("/creative_document_api/documents", headers=headers,
                          json={"width": 32, "height": 24, "name": "Director field scene"})
    assert created.status_code == 200, created.text
    document_id = created.json()["documentId"]
    saved = client.post(f"/creative_document_api/documents/{document_id}/save", headers=headers,
                        json={"expectedRevision": 0})
    assert saved.status_code == 200, saved.text
    project_path = runtime.project_path(document_id)
    owner_path = project_path.parent / f"{document_id}.owner.json"
    assert project_path.is_dir() and owner_path.is_file()
    return SimpleNamespace(
        client=client, runtime=runtime, owner_key=owner_key, document_id=document_id,
        project_path=project_path, owner_path=owner_path, editor_api=editor_api,
    )


def _unit_w05_fixture_builder(output_root: Path, asset_root: Path, owner_key: str | None, *,
                              document_id: str, projects_root_override: Path) -> tuple[str, dict[str, object]]:
    from modules.creative_document import Document, ProjectStore, make_id

    del output_root, asset_root, owner_key
    width, height = 2950, 1770
    plate_id, depth_id, riders_id, direction_id, comparison_id = (make_id("layer") for _ in range(5))
    plate_object_id, depth_object_id, riders_object_id, direction_object_id, comparison_object_id = (
        make_id("obj") for _ in range(5)
    )
    store = ProjectStore(projects_root_override / f"{document_id}.nexscene")
    image_records = [store.assets.put_bytes(
        _png((16, 12), (index * 24, 80, 128, 255)), media_type="image/png", extension="png",
        width=16, height=12, has_alpha=True,
    ) for index in range(5)]
    complete_plate = store.assets.put_bytes(
        _png((width, height), (39, 49, 61, 255)), media_type="image/png", extension="png",
        width=width, height=height, has_alpha=True,
    )
    edit_mask_asset = store.assets.put_bytes(
        _mask_png((width, height)), media_type="image/png", extension="png",
        width=width, height=height, has_alpha=False,
    )
    edit_mask_id = make_id("mask")
    layers = {
        plate_id: LayerRecord(plate_id, "Complete plate", "raster", role="complete-plate",
                              object_ids=[plate_object_id]),
        depth_id: LayerRecord(
            depth_id, "Registered depth element", "raster", opacity=0.88,
            object_ids=[depth_object_id], mask_ids=[edit_mask_id], role="depth-element",
            depth_element=True, complete_plate_id=plate_id,
            registration=CoordinateTransform.from_forward(
                "source-crop", "document", (16, 12), (width, height),
                AffineTransform.translation(580, 280), operation_revision=0,
            ),
            lineage={"completePlateId": plate_id, "registrationPreparedFor": document_id},
        ),
        riders_id: LayerRecord(riders_id, "Relational reference: riders", "raster", role="context-reference",
                               object_ids=[riders_object_id]),
        direction_id: LayerRecord(direction_id, "Direction reference", "raster", role="context-reference",
                                  object_ids=[direction_object_id]),
        comparison_id: LayerRecord(comparison_id, "Scene comparison reference", "raster",
                                   role="context-reference", object_ids=[comparison_object_id]),
    }
    objects = {
        plate_object_id: ObjectRecord(plate_object_id, plate_id, "raster-placement", "document",
                                      {"x": 0, "y": 0, "width": width, "height": height},
                                      asset_id=complete_plate.asset_id),
        depth_object_id: ObjectRecord(depth_object_id, depth_id, "raster-placement", "document",
                                      {"x": 580, "y": 280, "width": 16, "height": 12},
                                      asset_id=image_records[1].asset_id,
                                      lineage={"registrationLayerId": depth_id, "completePlateId": plate_id}),
        riders_object_id: ObjectRecord(riders_object_id, riders_id, "raster-placement", "document",
                                       {"x": 110, "y": 1300, "width": 16, "height": 12},
                                       asset_id=image_records[2].asset_id),
        direction_object_id: ObjectRecord(direction_object_id, direction_id, "raster-placement", "document",
                                          {"x": 2220, "y": 70, "width": 16, "height": 12},
                                          asset_id=image_records[3].asset_id),
        comparison_object_id: ObjectRecord(comparison_object_id, comparison_id, "raster-placement", "document",
                                           {"x": 2220, "y": 475, "width": 16, "height": 12},
                                           asset_id=image_records[4].asset_id),
    }
    generation_mask = MaskRecord(
        edit_mask_id, "generation", "document", 0, asset_id=edit_mask_asset.asset_id,
        owner_id=depth_id, lineage={"derivation": "unit W05 edit permission", "ownerLayerId": depth_id},
        content_hash=edit_mask_asset.content_hash,
    )
    document = Document(
        document_id, width, height,
        root_layer_ids=[plate_id, depth_id, riders_id, direction_id, comparison_id],
        layers=layers, objects=objects,
        assets={asset.asset_id: asset for asset in [*image_records, complete_plate, edit_mask_asset]},
        masks={edit_mask_id: generation_mask},
        metadata={"validationFixture": "P6-M01-W05", "lineageMode": "unit test"},
    )
    document.refresh_digest()
    document.validate()
    store.save(document, checkpoint=True)
    ids = {
        "documentId": document_id,
        "layers": {"completePlate": plate_id, "depthElement": depth_id, "ridersReference": riders_id,
                   "directionReference": direction_id, "comparisonReference": comparison_id},
        "objects": {"completePlate": plate_object_id, "depthElement": depth_object_id,
                    "ridersReference": riders_object_id, "directionReference": direction_object_id,
                    "comparisonReference": comparison_object_id},
        "masks": {"generationEditScope": edit_mask_id},
        "completePlateId": plate_id,
        "depthElementId": depth_id,
        "editTargetId": depth_id,
        "editMaskId": edit_mask_id,
    }
    return document_id, {
        "ids": ids,
        "fixtures": {"unit-test": {"sha256": "0" * 64, "dimensions": [width, height]}},
        "construction": {"reconstructionOrSegmentation": False},
    }


def test_director_owned_scene_survives_preparation_and_runtime_teardown(tmp_path: Path,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib
    import json

    from tools.prepare_w05_director_field_scene import prepare_director_scene

    blank = _make_saved_blank_w03_scene(tmp_path, monkeypatch)
    owner_descriptor_before = blank.owner_path.read_bytes()
    owner_principal = hashlib.sha256(f"browser:{blank.owner_key}".encode("utf-8")).hexdigest()
    result = prepare_director_scene(
        document_id=blank.document_id,
        output_root=tmp_path,
        asset_root=tmp_path / "unused-private-assets",
        fixture_builder=_unit_w05_fixture_builder,
    )
    assert result["revision"] == 17
    field_state = result["fieldState"]
    assert field_state["state"] == "complete" and field_state["historyTransactions"] == 17
    assert field_state["preparation"]["actorKind"] == "agent"
    assert field_state["preparation"]["directorAuthoredSetupActions"] == 0
    assert field_state["guides"]["marker"]["lifecycle"] == "proposed"
    assert field_state["guides"]["broad"]["lifecycle"] == "safe-to-remove"
    assert field_state["guides"]["tight"]["lifecycle"] == "active"
    assert field_state["context"] == {
        "referenceIds": [result["ids"]["layers"]["ridersReference"]],
        "editTargetIds": [result["ids"]["layers"]["depthElement"]],
        "editMaskId": result["ids"]["masks"]["generationEditScope"],
        "dilationPx": 24,
        "revision": 11,
    }
    assert field_state["depth"]["visible"] is True and field_state["depth"]["locked"] is False
    assert field_state["history"]["actorKinds"] == ["agent"]
    assert field_state["negativeChecksPassed"] is True
    assert len(field_state["negativeChecks"]) == 6
    assert all(check["passed"] and check["writes"] == 0 for check in field_state["negativeChecks"])
    receipt_text = json.dumps({"scene": result}, sort_keys=True)
    assert result["ownerDescriptorPreservedByteForByte"] is True
    assert result["ownerPrincipalRecorded"] is False
    assert blank.owner_path.read_bytes() == owner_descriptor_before
    assert blank.owner_key not in receipt_text and owner_principal not in receipt_text
    assert all(blank.owner_key.encode("ascii") not in path.read_bytes()
               for path in blank.project_path.rglob("*") if path.is_file())

    # A new runtime and session model the app, server, and temporary-runner teardown.
    from modules.creative_document_editor_api import CreativeDocumentRuntime

    restarted_runtime = CreativeDocumentRuntime()
    monkeypatch.setattr(blank.editor_api, "creative_document_runtime", restarted_runtime)
    session_id = "director-w05-after-restart"
    capability = restarted_runtime.sessions.issue(session_id, None, remote_exposure=False)
    director_headers = {
        "Authorization": f"Bearer {capability['token']}",
        "X-Gradio-Session": session_id,
        "X-Editor-Owner-Key": blank.owner_key,
        "Origin": "http://testserver",
        "Sec-Fetch-Site": "same-origin",
    }
    opened = blank.client.post(
        f"/creative_document_api/documents/{blank.document_id}/open", headers=director_headers,
        json={"discardUnsaved": False},
    )
    assert opened.status_code == 200, opened.text
    assert opened.json()["revision"] == 17
    retained = ProjectStore(blank.project_path).open()
    assert retained.current_revision == 17
    assert len(retained.guides) == 3 and len(retained.masks) == 2
    assert len(retained.history) == 17
    assert {record["actorKind"] for record in retained.history} == {"agent"}
    edited = blank.client.post(
        f"/creative_document_api/documents/{blank.document_id}/actions", headers=director_headers,
        json={"expectedRevision": 17, "actorKind": "director", "actionType": "add_layer",
              "data": {"kind": "vector", "name": "Director field edit"}},
    )
    assert edited.status_code == 200, edited.text
    assert edited.json()["currentRevision"] == 18
    saved = blank.client.post(
        f"/creative_document_api/documents/{blank.document_id}/save", headers=director_headers,
        json={"expectedRevision": 18},
    )
    assert saved.status_code == 200, saved.text
    reopened = blank.client.post(
        f"/creative_document_api/documents/{blank.document_id}/open", headers=director_headers,
        json={"discardUnsaved": False},
    )
    assert reopened.status_code == 200 and reopened.json()["revision"] == 18

    unrelated_session = "unrelated-w05-principal"
    unrelated = restarted_runtime.sessions.issue(unrelated_session, None, remote_exposure=False)
    unrelated_headers = {
        "Authorization": f"Bearer {unrelated['token']}",
        "X-Gradio-Session": unrelated_session,
        "X-Editor-Owner-Key": "x" * 43,
        "Origin": "http://testserver",
        "Sec-Fetch-Site": "same-origin",
    }
    denied = blank.client.get(
        f"/creative_document_api/documents/{blank.document_id}/view", headers=unrelated_headers,
    )
    assert denied.status_code == 404
    assert denied.json()["detail"]["code"] == "UNKNOWN_DOCUMENT"
    assert blank.document_id not in denied.text


def test_director_scene_preparation_refuses_nonblank_scene_without_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.prepare_w05_director_field_scene import FieldScenePreparationError, prepare_director_scene

    blank = _make_saved_blank_w03_scene(tmp_path, monkeypatch)
    capability = blank.runtime.sessions.issue("director-w05-preparation", None, remote_exposure=False)
    headers = {"Authorization": f"Bearer {capability['token']}",
               "X-Gradio-Session": "director-w05-preparation", "X-Editor-Owner-Key": blank.owner_key,
               "Origin": "http://testserver", "Sec-Fetch-Site": "same-origin"}
    changed = blank.client.post(
        f"/creative_document_api/documents/{blank.document_id}/actions",
        headers=headers,
        json={"expectedRevision": 0, "actorKind": "director", "actionType": "add_layer",
              "data": {"kind": "vector", "name": "Existing content"}},
    )
    assert changed.status_code == 200, changed.text
    saved = blank.client.post(f"/creative_document_api/documents/{blank.document_id}/save", headers=headers,
                              json={"expectedRevision": 1})
    assert saved.status_code == 200, saved.text
    before = {path.relative_to(blank.project_path): path.read_bytes()
              for path in blank.project_path.rglob("*") if path.is_file()}
    owner_before = blank.owner_path.read_bytes()
    with pytest.raises(FieldScenePreparationError, match="empty revision-0"):
        prepare_director_scene(
            document_id=blank.document_id, output_root=tmp_path,
            asset_root=tmp_path / "unused-private-assets", fixture_builder=_unit_w05_fixture_builder,
        )
    after = {path.relative_to(blank.project_path): path.read_bytes()
             for path in blank.project_path.rglob("*") if path.is_file()}
    assert after == before
    assert blank.owner_path.read_bytes() == owner_before
    assert not list(blank.project_path.parent.glob(f".w05-field-prep-{blank.document_id}-*"))


def test_director_scene_preparation_rolls_back_failed_install(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.prepare_w05_director_field_scene as preparation
    from tools.prepare_w05_director_field_scene import prepare_director_scene

    blank = _make_saved_blank_w03_scene(tmp_path, monkeypatch)
    project_before = {path.relative_to(blank.project_path): path.read_bytes()
                      for path in blank.project_path.rglob("*") if path.is_file()}
    owner_before = blank.owner_path.read_bytes()
    real_replace = preparation.os.replace
    failed_once = False

    def fail_staged_install(source: object, destination: object) -> None:
        nonlocal failed_once
        source_path, destination_path = Path(source), Path(destination)
        if (not failed_once and source_path.name == f"{blank.document_id}.nexscene"
                and source_path.parent.name.startswith(f".w05-field-prep-{blank.document_id}-")
                and destination_path == blank.project_path):
            failed_once = True
            raise OSError("injected staged install failure")
        real_replace(source, destination)

    monkeypatch.setattr(preparation.os, "replace", fail_staged_install)
    with pytest.raises(OSError, match="injected staged install failure"):
        prepare_director_scene(
            document_id=blank.document_id, output_root=tmp_path,
            asset_root=tmp_path / "unused-private-assets", fixture_builder=_unit_w05_fixture_builder,
        )

    project_after = {path.relative_to(blank.project_path): path.read_bytes()
                     for path in blank.project_path.rglob("*") if path.is_file()}
    assert failed_once and project_after == project_before
    assert blank.owner_path.read_bytes() == owner_before
    assert not list(blank.project_path.parent.glob(f".w05-field-prep-{blank.document_id}-*"))
