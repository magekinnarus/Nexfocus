"""Replay the complete W05 field baseline through the semantic command service.

This is offline fixture preparation. All recorded changes are attributed to a
distinct W05 preparation Agent actor; none are represented as Director actions.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any


class W05FieldStateBuildError(RuntimeError):
    """A redacted failure while replaying the validated W05 field baseline."""


def _runner_helpers(repo: Path) -> Any:
    browser_dir = repo / "tests" / "browser"
    for entry in (str(repo), str(browser_dir)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    import run_creative_document_field_validation as runner

    return runner


def _execute(service: Any, handle: Any, actor: Any, envelope_builder: Any, command_type: str,
             revision: int, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    envelope = envelope_builder(handle.document.document_id, "mutate", command_type,
                                revision=revision, payload=payload)
    receipt = service.execute(handle, envelope, actor)
    if receipt.get("status") != "committed" or receipt.get("newRevision") != revision + 1:
        code = (receipt.get("error") or {}).get("code")
        raise W05FieldStateBuildError(f"validated W05 {command_type} command failed ({code or 'unknown'}).")
    return receipt


def _save(service: Any, handle: Any, actor: Any, envelope_builder: Any, revision: int) -> None:
    envelope = envelope_builder(handle.document.document_id, "mutate", "save_scene", revision=revision)
    receipt = service.execute(handle, envelope, actor)
    if (receipt.get("status") != "saved" or receipt.get("revision") != revision
            or receipt.get("committedRevision") != revision):
        code = (receipt.get("error") or {}).get("code")
        raise W05FieldStateBuildError(f"validated W05 save failed at revision {revision} ({code or 'unknown'}).")
    handle.committed_revision = revision


def _refuse(service: Any, handle: Any, actor: Any, envelope_builder: Any, *, name: str,
            command_type: str, expected_revision: int, payload: dict[str, Any], expected_code: str,
            observed_revision: int) -> dict[str, Any]:
    envelope = envelope_builder(handle.document.document_id, "mutate", command_type,
                                revision=expected_revision, payload=payload)
    receipt = service.execute(handle, envelope, actor)
    error = receipt.get("error") if isinstance(receipt.get("error"), dict) else {}
    passed = (receipt.get("status") in {"error", "conflict"} and error.get("code") == expected_code
              and receipt.get("writes") == 0
              and receipt.get("currentRevision") in {None, observed_revision}
              and handle.document.current_revision == observed_revision)
    evidence = {"name": name, "passed": passed, "expectedCode": expected_code,
                "code": error.get("code"), "expectedRevision": expected_revision,
                "expectedCurrentRevision": observed_revision,
                "receiptCurrentRevision": receipt.get("currentRevision"),
                "observedCurrentRevision": handle.document.current_revision,
                "writes": receipt.get("writes")}
    if not passed:
        raise W05FieldStateBuildError(f"W05 negative check failed: {name}.")
    return evidence


def _guide(document: Any, identity: str) -> Any:
    guide = document.guides.get(identity)
    if guide is None:
        raise W05FieldStateBuildError("a required W05 guide is absent after replay")
    return guide


def validate_w05_field_state(document: Any, fixture_info: dict[str, Any], *, actor_id: str) -> dict[str, Any]:
    """Assert the complete final W05 scene and return its redacted state receipt."""
    ids = fixture_info.get("ids")
    if not isinstance(ids, dict):
        raise W05FieldStateBuildError("the base W05 fixture identities are unavailable")
    layers, masks = ids.get("layers"), ids.get("masks")
    if not isinstance(layers, dict) or not isinstance(masks, dict):
        raise W05FieldStateBuildError("the base W05 layer or mask identities are unavailable")
    try:
        depth_id = layers["depthElement"]
        plate_id = ids["completePlateId"]
        riders_id = layers["ridersReference"]
        edit_mask_id = masks["generationEditScope"]
        marker_id = ids["fieldState"]["markerGuideId"]
        broad_id = ids["fieldState"]["broadGuideId"]
        tight_id = ids["fieldState"]["tightGuideId"]
        context_id = ids["fieldState"]["contextMaskId"]
    except (KeyError, TypeError) as exc:
        raise W05FieldStateBuildError("the field-state identities are incomplete") from exc

    marker, broad, tight = (_guide(document, identity) for identity in (marker_id, broad_id, tight_id))
    depth = document.layers.get(depth_id)
    plate = document.layers.get(plate_id)
    context = document.masks.get(context_id)
    edit_mask = document.masks.get(edit_mask_id)
    history = document.history
    revisions = [(record.get("previousRevision"), record.get("resultingRevision")) for record in history]
    actor_ids = sorted({record.get("actorId") for record in history})
    actor_kinds = sorted({record.get("actorKind") for record in history})
    context_lineage = context.lineage if context is not None else {}
    depth_ok = (depth is not None and depth.depth_element is True and depth.visible is True
                and depth.complete_plate_id == plate_id and depth.registration is not None
                and depth.locked is False and plate is not None and plate.role == "complete-plate")
    guides_ok = (marker.lifecycle == "proposed" and marker.semantic_role == "position only; no subject identity"
                 and len(marker.object_ids) == 1 and not document.layers[document.objects[marker.object_ids[0]].layer_id].visible
                 and broad.lifecycle == "safe-to-remove" and broad.replacement_id == tight_id
                 and tight.lifecycle == "active" and tight.supersedes_id == broad_id
                 and broad.object_ids != tight.object_ids)
    context_ok = (context is not None and context.purpose == "context" and context.owner_id == depth_id
                  and context_lineage.get("relationalReferences") == [riders_id]
                  and context_lineage.get("editTargetIds") == [depth_id]
                  and context_lineage.get("editMaskId") == edit_mask_id
                  and context_lineage.get("dilationPx") == 24
                  and edit_mask is not None and edit_mask.purpose == "generation")
    history_ok = (document.current_revision == 17 and len(history) == 17
                  and revisions == [(revision, revision + 1) for revision in range(17)]
                  and actor_ids == [actor_id] and actor_kinds == ["agent"]
                  and all(isinstance(record.get("metadata", {}).get("semanticCommand"), dict) for record in history))
    if not (depth_ok and guides_ok and context_ok and history_ok):
        raise W05FieldStateBuildError("the replayed W05 field state failed its final-state assertions")

    return {
        "state": "complete",
        "documentId": document.document_id,
        "revision": document.current_revision,
        "historyTransactions": len(history),
        "preparation": {
            "method": "validated CommandService.execute replay",
            "actorKind": "agent",
            "actorId": actor_id,
            "directorAuthoredSetupActions": 0,
        },
        "ids": {
            "markerGuideId": marker_id,
            "markerLayerId": document.objects[marker.object_ids[0]].layer_id,
            "broadGuideId": broad_id,
            "tightGuideId": tight_id,
            "contextMaskId": context_id,
            "depthElementId": depth_id,
            "completePlateId": plate_id,
            "ridersReferenceLayerId": riders_id,
            "generationEditMaskId": edit_mask_id,
        },
        "guides": {
            "marker": {"lifecycle": marker.lifecycle, "stateRevision": marker.state_revision,
                       "semanticRole": marker.semantic_role, "visible": False},
            "broad": {"lifecycle": broad.lifecycle, "stateRevision": broad.state_revision,
                      "replacementId": broad.replacement_id},
            "tight": {"lifecycle": tight.lifecycle, "stateRevision": tight.state_revision,
                      "supersedesId": tight.supersedes_id},
            "objectIdsDistinct": broad.object_ids != tight.object_ids,
        },
        "context": {"referenceIds": list(context_lineage["relationalReferences"]),
                    "editTargetIds": list(context_lineage["editTargetIds"]),
                    "editMaskId": context_lineage["editMaskId"], "dilationPx": context_lineage["dilationPx"],
                    "revision": context.revision},
        "depth": {"visible": depth.visible, "locked": depth.locked,
                  "completePlateId": depth.complete_plate_id,
                  "registration": depth.registration.to_dict(),
                  "layerTransform": depth.transform.to_dict()},
        "history": {"revisionSequence": [record.get("resultingRevision") for record in history],
                    "actorKinds": actor_kinds, "actorIds": actor_ids,
                    "allMutationsUsedSemanticReceipts": True, "revisionsMonotonic": True},
    }


def build_w05_field_state(project_path: Path, fixture_info: dict[str, Any], *, repo: Path) -> dict[str, Any]:
    """Replay the live-runner state sequence on a staged W05 base fixture."""
    from modules.creative_document import ProjectStore, make_id
    from modules.creative_document.command_service import ActorContext, CommandService

    runner = _runner_helpers(repo.resolve())
    store = ProjectStore(project_path)
    document = store.open()
    actor_id = make_id("agent")
    actor = ActorContext(actor_kind="agent", actor_id=actor_id,
                         scopes=frozenset({"inspect", "propose", "mutate"}))
    service = CommandService()
    handle = SimpleNamespace(document=document, store=store, committed_revision=0,
                              lock=threading.RLock(), recovery_notice=None)
    ids = fixture_info["ids"]
    layers = ids["layers"]
    masks = ids["masks"]
    depth_id = layers["depthElement"]
    plate_id = ids["completePlateId"]
    riders_id = layers["ridersReference"]
    edit_mask_id = masks["generationEditScope"]

    marker = _execute(service, handle, actor, runner._envelope, "create_guide", 0, {"data": {
        "name": "Position marker", "semanticRole": "position only; no subject identity",
        "kind": "shape", "geometry": {"shape": "ellipse", "x": 980, "y": 350, "width": 130, "height": 130},
        "style": {"fill": "#F2D64B", "stroke": "#342D0D", "strokeWidth": 5, "opacity": 0.92},
    }})
    marker_id = next(identity for identity in marker["createdIds"] if identity.startswith("guide-"))
    broad = _execute(service, handle, actor, runner._envelope, "create_guide", 1, {"data": {
        "name": "Focal relation anchor", "semanticRole": "focal person attended to by the nearby group",
        "kind": "shape", "geometry": {"shape": "polygon", "points": [[1320, 340], [1385, 345], [1425, 410],
            [1410, 465], [1450, 540], [1375, 525], [1335, 600], [1305, 520], [1260, 550], [1285, 445], [1265, 390]]},
        "style": {"fill": "#47C78B", "stroke": "#153B2A", "strokeWidth": 5, "opacity": 0.78},
    }})
    broad_id = next(identity for identity in broad["createdIds"] if identity.startswith("guide-"))
    marker_layer_id = handle.document.objects[handle.document.guides[marker_id].object_ids[0]].layer_id
    _execute(service, handle, actor, runner._envelope, "set_layer_visibility", 2,
             {"data": {"layerId": marker_layer_id, "visible": False}})
    _save(service, handle, actor, runner._envelope, 3)
    _execute(service, handle, actor, runner._envelope, "transition_guide", 3,
             {"data": {"guideId": broad_id, "lifecycle": "active"}})
    tight = _execute(service, handle, actor, runner._envelope, "create_guide", 4, {"data": {
        "name": "Tighter focal relation", "semanticRole": "same focal person; tighter head-and-shoulder placement cue",
        "kind": "shape", "geometry": {"shape": "polygon", "points": [[1350, 360], [1390, 365], [1415, 405],
            [1407, 448], [1430, 485], [1380, 478], [1350, 515], [1328, 468], [1300, 480], [1315, 414]]},
        "style": {"fill": "#63D7A2", "stroke": "#153B2A", "strokeWidth": 4, "opacity": 0.72},
    }})
    tight_id = next(identity for identity in tight["createdIds"] if identity.startswith("guide-"))
    _execute(service, handle, actor, runner._envelope, "transition_guide", 5,
             {"data": {"guideId": broad_id, "lifecycle": "replacement-pending", "replacementId": tight_id}})
    stale_tight = _refuse(service, handle, actor, runner._envelope, name="stale_tight_create_zero_write",
        command_type="create_guide", expected_revision=5,
        payload={"data": {"name": "Stale guide must not appear", "semanticRole": "stale probe",
                 "kind": "shape", "geometry": {"shape": "ellipse", "x": 1200, "y": 780, "width": 30, "height": 30},
                 "style": {"fill": "#FF0000", "stroke": "#000000", "width": 2}}},
        expected_code="STALE_DOCUMENT_REVISION", observed_revision=6)
    context = _execute(service, handle, actor, runner._envelope, "create_context_mask", 6, {"data": {
        "sourceLayerId": depth_id,
        "seeds": [{"kind": "box", "geometry": {"x": 600, "y": 300, "width": 1450, "height": 960}}],
    }})
    context_id = next(identity for identity in context["createdIds"] if identity.startswith("mask-"))
    _execute(service, handle, actor, runner._envelope, "transition_guide", 7,
             {"data": {"guideId": broad_id, "lifecycle": "superseded"}})
    _execute(service, handle, actor, runner._envelope, "transition_guide", 8,
             {"data": {"guideId": tight_id, "lifecycle": "active"}})
    _execute(service, handle, actor, runner._envelope, "transition_guide", 9,
             {"data": {"guideId": broad_id, "lifecycle": "safe-to-remove"}})
    context_data = {"contextMaskId": context_id, "referenceIds": [riders_id], "dilationPx": 24,
                    "editMaskId": edit_mask_id, "editTargetIds": [depth_id]}
    _execute(service, handle, actor, runner._envelope, "set_relational_context", 10,
             {"data": context_data})
    overlap = _refuse(service, handle, actor, runner._envelope, name="context_overlap_zero_write",
        command_type="set_relational_context", expected_revision=11,
        payload={"data": {**context_data, "referenceIds": [depth_id]}},
        expected_code="CONTEXT_EDIT_SCOPE_OVERLAP", observed_revision=11)
    malformed = _refuse(service, handle, actor, runner._envelope, name="malformed_context_zero_write",
        command_type="set_relational_context", expected_revision=11,
        payload={"data": {**context_data, "referenceIds": "malformed"}},
        expected_code="INVALID_TARGETS", observed_revision=11)
    stale_context = _refuse(service, handle, actor, runner._envelope, name="stale_context_zero_write",
        command_type="set_relational_context", expected_revision=10,
        payload={"data": context_data}, expected_code="STALE_DOCUMENT_REVISION", observed_revision=11)
    _execute(service, handle, actor, runner._envelope, "set_layer_lock", 11,
             {"data": {"layerId": depth_id, "locked": True}})
    locked = _refuse(service, handle, actor, runner._envelope, name="locked_context_owner_zero_write",
        command_type="set_relational_context", expected_revision=12,
        payload={"data": {**context_data, "dilationPx": 25}},
        expected_code="LAYER_LOCKED", observed_revision=12)
    _execute(service, handle, actor, runner._envelope, "set_layer_lock", 12,
             {"data": {"layerId": depth_id, "locked": False}})
    _save(service, handle, actor, runner._envelope, 13)
    _execute(service, handle, actor, runner._envelope, "set_layer_visibility", 13,
             {"data": {"layerId": depth_id, "visible": False}})
    stale_depth = _refuse(service, handle, actor, runner._envelope, name="stale_depth_show_zero_write",
        command_type="set_layer_visibility", expected_revision=13,
        payload={"data": {"layerId": depth_id, "visible": True}},
        expected_code="STALE_DOCUMENT_REVISION", observed_revision=14)
    _execute(service, handle, actor, runner._envelope, "undo", 14, {})
    _save(service, handle, actor, runner._envelope, 15)
    _execute(service, handle, actor, runner._envelope, "redo", 15, {})
    _save(service, handle, actor, runner._envelope, 16)
    hidden_reopen = store.open()
    hidden_depth = hidden_reopen.layers[depth_id]
    if hidden_reopen.current_revision != 16 or hidden_depth.visible is not False:
        raise W05FieldStateBuildError("the saved hidden depth state failed reopen verification")
    _execute(service, handle, actor, runner._envelope, "set_layer_visibility", 16,
             {"data": {"layerId": depth_id, "visible": True}})
    _save(service, handle, actor, runner._envelope, 17)
    final_document = store.open()
    shown_depth = final_document.layers[depth_id]
    if final_document.current_revision != 17 or shown_depth.visible is not True:
        raise W05FieldStateBuildError("the final shown depth state failed reopen verification")

    fixture_info["ids"]["fieldState"] = {"markerGuideId": marker_id, "broadGuideId": broad_id,
        "tightGuideId": tight_id, "contextMaskId": context_id}
    state = validate_w05_field_state(final_document, fixture_info, actor_id=actor_id)
    state["negativeChecks"] = [stale_tight, overlap, malformed, stale_context, locked, stale_depth]
    state["negativeChecksPassed"] = all(item["passed"] and item["writes"] == 0 for item in state["negativeChecks"])
    state["successfulCommandCount"] = 22
    if not state["negativeChecksPassed"]:
        raise W05FieldStateBuildError("one or more W05 zero-write checks failed")
    return state
