"""Explicit Agent-safe document projections and preview rendering.

The Agent projection is deliberately built from record-level allowlists. It is
not a serialized document with private fields removed afterward. The companion
preview is rendered from a second, short-lived document containing only the
visible allowlisted layer/object/asset graph.
"""

from __future__ import annotations

import json
import io
import math
import re
from copy import deepcopy
from typing import Any, Callable, Iterable, Mapping

from PIL import Image

from .editor_render import RenderError, render_document_png
from .ids import canonical_json, sha256_bytes
from .schema import (
    AssetRecord,
    CollaborationState,
    Document,
    HandoffNote,
    LayerRecord,
    ObjectRecord,
    SchemaValidationError,
)


MAX_PROJECTED_RECORDS = 20_000
MAX_PROJECTED_BYTES = 4 * 1024 * 1024
MAX_PREVIEW_BYTES = 12 * 1024 * 1024
MAX_PREVIEW_SIZE = (1600, 1200)

_SAFE_LABEL_RE = re.compile(r"^[^\x00-\x1f\x7f/\\:]{1,120}$")
_HANDOFF_ACTIONS = {"review", "continue", "preserve", "refine", "replace", "align", "hold", "compare"}
_HANDOFF_FOCUSES = {
    "composition", "placement", "scale", "silhouette", "color", "depth", "edge",
    "lighting", "context", "selection", "mask", "guide", "layer", "visibility", "registration",
}
_HANDOFF_DETAILS = {
    "keep", "soften", "sharpen", "enlarge", "reduce", "raise", "lower", "move-left",
    "move-right", "move-up", "move-down", "hide", "show", "activate", "retire", "compare",
}
_GEOMETRY_FIELDS = {"shape", "x", "y", "width", "height", "points", "closed"}
_RASTER_GEOMETRY_FIELDS = {"x", "y", "width", "height"}
_STYLE_FIELDS = {"fill", "stroke", "color", "width", "strokeWidth", "opacity", "mode", "hardness"}
_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}(?:[0-9A-Fa-f]{2})?$")
_SAFE_MEDIA_TYPES = {
    "application/octet-stream",
    "image/bmp",
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/webp",
}


class AgentProjectionError(ValueError):
    """A safe projection could not be built without risking disclosure."""

    def __init__(self, code: str = "SAFE_VIEW_UNAVAILABLE") -> None:
        super().__init__(code)
        self.code = code


def _safe_label(value: Any, fallback: str = "") -> str:
    if not isinstance(value, str) or not _SAFE_LABEL_RE.fullmatch(value.strip()):
        return fallback
    return value.strip()


def _safe_blend_mode(value: Any) -> str:
    # W03's renderer supports normal blending only. Keep that fact visible while
    # avoiding arbitrary persisted strings in an Agent-facing summary.
    return "normal" if value == "normal" else "unsupported"


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _safe_geometry(obj: ObjectRecord) -> dict[str, Any]:
    geometry = obj.geometry
    if not isinstance(geometry, Mapping):
        raise AgentProjectionError("SAFE_GEOMETRY_UNAVAILABLE")
    allowed = _RASTER_GEOMETRY_FIELDS if obj.kind == "raster-placement" else _GEOMETRY_FIELDS
    if set(geometry) - allowed:
        raise AgentProjectionError("SAFE_GEOMETRY_UNAVAILABLE")
    result: dict[str, Any] = {}
    for key in ("shape", "x", "y", "width", "height", "points", "closed"):
        if key not in geometry:
            continue
        value = geometry[key]
        if key == "shape":
            if value not in {"line", "rectangle", "ellipse", "polygon"}:
                raise AgentProjectionError("SAFE_GEOMETRY_UNAVAILABLE")
            result[key] = value
        elif key == "points":
            if not isinstance(value, list) or len(value) > 20_000:
                raise AgentProjectionError("SAFE_GEOMETRY_UNAVAILABLE")
            points: list[list[int | float]] = []
            for point in value:
                if not isinstance(point, (list, tuple)) or len(point) != 2 or any(not _finite_number(v) for v in point):
                    raise AgentProjectionError("SAFE_GEOMETRY_UNAVAILABLE")
                points.append([point[0], point[1]])
            result[key] = points
        elif key == "closed":
            if type(value) is not bool:
                raise AgentProjectionError("SAFE_GEOMETRY_UNAVAILABLE")
            result[key] = value
        else:
            if not _finite_number(value):
                raise AgentProjectionError("SAFE_GEOMETRY_UNAVAILABLE")
            result[key] = value
    return result


def _safe_style(obj: ObjectRecord) -> dict[str, Any]:
    style = obj.style
    if not isinstance(style, Mapping) or set(style) - _STYLE_FIELDS:
        raise AgentProjectionError("SAFE_STYLE_UNAVAILABLE")
    result: dict[str, Any] = {}
    for key, value in style.items():
        if key in {"fill", "stroke", "color"}:
            if value is not None and (not isinstance(value, str) or not _COLOR_RE.fullmatch(value)):
                raise AgentProjectionError("SAFE_STYLE_UNAVAILABLE")
        elif key == "mode":
            if value not in {"paint", "erase"}:
                raise AgentProjectionError("SAFE_STYLE_UNAVAILABLE")
        elif not _finite_number(value):
            raise AgentProjectionError("SAFE_STYLE_UNAVAILABLE")
        result[key] = value
    return result


def _safe_handoff_body(body: str) -> str | None:
    """Accept only the fixed, non-free-text shared handoff vocabulary.

    A shared note body is UTF-8 JSON with exactly four fields. No user sentence,
    URI, asset identifier, path, hash, or pixel payload is accepted. Directors
    can keep unrestricted notes as `director-only`, which this projection never
    reads into its output.
    """

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate field")
            value[key] = item
        return value

    try:
        value = json.loads(body, object_pairs_hook=unique_object,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite")))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or set(value) != {"schemaVersion", "action", "focus", "detail"}:
        return None
    if (type(value["schemaVersion"]) is not int or value["schemaVersion"] != 1
            or value["action"] not in _HANDOFF_ACTIONS
            or value["focus"] not in _HANDOFF_FOCUSES
            or value["detail"] not in _HANDOFF_DETAILS):
        return None
    return canonical_json(value).decode("utf-8")


def _effective_visibility(document: Document) -> dict[str, bool]:
    memo: dict[str, bool] = {}

    def visit(layer_id: str) -> bool:
        if layer_id in memo:
            return memo[layer_id]
        layer = document.layers[layer_id]
        visible = layer.visible and not layer.metadata.get("deleted", False)
        if layer.parent_id is not None:
            visible = visible and visit(layer.parent_id)
        memo[layer_id] = visible
        return visible

    for layer_id in document.layers:
        visit(layer_id)
    return memo


def _director_only_layer_ids(document: Document) -> set[str]:
    """Exclude explicitly Director-only layer subtrees from Agent surfaces."""
    excluded = {
        layer_id for layer_id, layer in document.layers.items()
        if layer.metadata.get("directorOnly") is True
    }
    pending = list(excluded)
    while pending:
        layer = document.layers[pending.pop()]
        for child_id in layer.child_ids:
            if child_id not in excluded:
                excluded.add(child_id)
                pending.append(child_id)
    return excluded


def _layer_projection(
    document: Document,
    layer: LayerRecord,
    visible: bool,
    *,
    safe_ids: set[str],
    transaction_ids: set[str],
) -> dict[str, Any]:
    safe_notes: list[dict[str, Any]] = []
    for note in layer.collaboration.handoff_notes:
        if note.visibility != "shared":
            continue
        if (not set(note.target_ids).issubset(safe_ids)
                or (note.transaction_id is not None and note.transaction_id not in transaction_ids)):
            continue
        body = _safe_handoff_body(note.body)
        if body is None:
            continue
        safe_notes.append({
            "noteId": note.note_id,
            "actorKind": note.actor_kind,
            "body": body,
            "createdAtRevision": note.created_at_revision,
            "transactionId": note.transaction_id,
            "targetIds": list(note.target_ids),
            "targetRevisions": {identity: revision for identity, revision in note.target_revisions.items()
                                if identity in safe_ids},
            "state": note.state,
        })
    last_editor_kind = None
    if layer.collaboration.last_editor is not None:
        last_editor_kind = layer.collaboration.last_editor["actorKind"]
    result = {
        "layerId": layer.layer_id,
        "name": _safe_label(layer.name, "Layer"),
        "kind": layer.kind,
        "role": _safe_label(layer.role, "") or None,
        "parentId": layer.parent_id if layer.parent_id in safe_ids else None,
        "childIds": [identity for identity in layer.child_ids if identity in safe_ids],
        "objectIds": [identity for identity in layer.object_ids if identity in safe_ids],
        "assetIds": [identity for identity in layer.asset_ids if identity in safe_ids],
        "maskIds": [identity for identity in layer.mask_ids if identity in safe_ids],
        "visible": layer.visible,
        "effectivelyVisible": visible,
        "locked": layer.locked,
        "opacity": layer.opacity,
        "blendMode": _safe_blend_mode(layer.blend_mode),
        "transform": layer.transform.to_dict(),
        "revision": layer.revision,
        "depthElement": layer.depth_element,
        "completePlateId": layer.complete_plate_id if layer.complete_plate_id in safe_ids else None,
        "workStatus": layer.collaboration.work_status,
        "lastEditorKind": last_editor_kind,
        "sharedHandoffNotes": safe_notes,
    }
    return result


def _asset_summary(asset: AssetRecord) -> dict[str, Any]:
    return {
        "assetId": asset.asset_id,
        "mediaType": asset.media_type if asset.media_type in _SAFE_MEDIA_TYPES else "application/octet-stream",
        "extension": asset.extension,
        "byteLength": asset.byte_length,
        "width": asset.width,
        "height": asset.height,
        "hasAlpha": asset.has_alpha,
        "availability": "embedded" if asset.external_uri is None else asset.external_status,
    }


def build_agent_projection(document: Document, *, transaction_limit: int = 50) -> dict[str, Any]:
    """Build one recursive allowlisted view at the document's current revision."""

    try:
        document.validate()
        if len(document.layers) + len(document.objects) + len(document.assets) > MAX_PROJECTED_RECORDS:
            raise AgentProjectionError("SAFE_VIEW_TOO_LARGE")
        if type(transaction_limit) is not int or not 0 <= transaction_limit <= 100:
            raise AgentProjectionError("SAFE_VIEW_LIMIT_INVALID")

        visibility = _effective_visibility(document)
        director_only_layers = _director_only_layer_ids(document)
        safe_layer_records = {
            identity: layer for identity, layer in document.layers.items()
            if identity not in director_only_layers
        }
        safe_object_records = {
            identity: obj for identity, obj in document.objects.items()
            if obj.layer_id in safe_layer_records
        }
        director_only_asset_ids = {
            asset_id for layer_id in director_only_layers
            for asset_id in document.layers[layer_id].asset_ids
        }
        director_only_asset_ids.update(
            obj.asset_id for obj in document.objects.values()
            if obj.layer_id in director_only_layers and obj.asset_id is not None
        )
        safe_asset_ids: set[str] = set()
        for layer in safe_layer_records.values():
            safe_asset_ids.update(layer.asset_ids)
        for obj in safe_object_records.values():
            if obj.asset_id is not None:
                safe_asset_ids.add(obj.asset_id)
        for proxy in document.private_proxies.values():
            safe_asset_ids.add(proxy.proxy_asset_id)
        director_only_asset_ids.difference_update(safe_asset_ids)
        safe_mask_records = {
            identity: mask for identity, mask in document.masks.items()
            if mask.owner_id not in director_only_layers and mask.asset_id not in director_only_asset_ids
        }
        safe_selection_records = {
            identity: selection for identity, selection in document.selections.items()
            if (selection.source_id in safe_layer_records and selection.mask_id in safe_mask_records
                and set(selection.derived_layer_ids).issubset(safe_layer_records))
        }
        safe_guide_records = {
            identity: guide for identity, guide in document.guides.items()
            if set(guide.object_ids).issubset(safe_object_records)
        }
        safe_ids = {
            document.document_id, *safe_layer_records.keys(), *safe_object_records.keys(), *safe_asset_ids,
            *safe_mask_records.keys(), *safe_selection_records.keys(), *safe_guide_records.keys(),
            *document.private_proxies.keys(),
            *(proxy.destination_slot_id for proxy in document.private_proxies.values()),
        }
        transaction_ids = {raw["transactionId"] for raw in document.history}
        layers = {
            layer_id: _layer_projection(document, layer, visibility[layer_id], safe_ids=safe_ids,
                                        transaction_ids=transaction_ids)
            for layer_id, layer in safe_layer_records.items()
        }

        objects: dict[str, Any] = {}
        for object_id, obj in safe_object_records.items():
            layer = document.layers[obj.layer_id]
            objects[object_id] = {
                "objectId": obj.object_id,
                "layerId": obj.layer_id,
                "kind": obj.kind,
                "coordinateSpace": obj.coordinate_space,
                "geometry": _safe_geometry(obj),
                "transform": obj.transform.to_dict(),
                "assetId": obj.asset_id,
                "style": _safe_style(obj),
                "revision": obj.revision,
                "effectivelyVisible": visibility[layer.layer_id],
            }

        assets = {identity: _asset_summary(document.assets[identity])
                  for identity in sorted(safe_asset_ids) if identity in document.assets}

        masks: dict[str, Any] = {}
        relational_context: list[dict[str, Any]] = []
        for mask_id, mask in safe_mask_records.items():
            masks[mask_id] = {
                "maskId": mask.mask_id,
                "purpose": mask.purpose,
                "coordinateSpace": mask.coordinate_space,
                "revision": mask.revision,
                "ownerId": mask.owner_id if mask.owner_id in safe_ids else None,
                "available": mask.asset_id in document.assets or mask.editable_source is not None,
            }
            if mask.purpose == "context":
                lineage = mask.lineage
                references = lineage.get("relationalReferences", [])
                edit_targets = lineage.get("editTargetIds", [])
                dilation = lineage.get("dilationPx", 0)
                if (not isinstance(references, list) or not all(isinstance(item, str) for item in references)
                        or not isinstance(edit_targets, list) or not all(isinstance(item, str) for item in edit_targets)
                        or type(dilation) is not int or not 0 <= dilation <= 256):
                    raise AgentProjectionError("SAFE_CONTEXT_UNAVAILABLE")
                relational_context.append({
                    "contextMaskId": mask.mask_id,
                    "ownerId": mask.owner_id if mask.owner_id in safe_ids else None,
                    "referenceIds": [identity for identity in references if identity in safe_ids],
                    "editTargetIds": [identity for identity in edit_targets if identity in safe_ids],
                    "dilationPx": dilation,
                    "revision": mask.revision,
                })

        selections = {
            selection.selection_id: {
                "selectionId": selection.selection_id,
                "sourceId": selection.source_id,
                "sourceRevision": selection.source_revision,
                "selectionRevision": selection.selection_revision,
                "state": selection.state,
                "bounds": None if selection.bounds is None else selection.bounds.to_dict(),
                "derivedLayerIds": list(selection.derived_layer_ids),
                "staleReason": selection.stale_reason,
            }
            for selection in safe_selection_records.values()
        }

        guides = {
            guide.guide_id: {
                "guideId": guide.guide_id,
                "name": _safe_label(guide.name, "Guide"),
                "lifecycle": guide.lifecycle,
                "createdRevision": guide.created_revision,
                "stateRevision": guide.state_revision,
                "supersedesId": guide.supersedes_id if guide.supersedes_id in safe_guide_records else None,
                "replacementId": guide.replacement_id if guide.replacement_id in safe_guide_records else None,
                "semanticRole": _safe_label(guide.semantic_role, "") or None,
                "objectIds": list(guide.object_ids),
            }
            for guide in safe_guide_records.values()
        }

        transactions = []
        for raw in document.history[-transaction_limit:] if transaction_limit else []:
            transaction = {
                "transactionId": raw["transactionId"],
                "groupId": raw["groupId"],
                "actorKind": raw["actorKind"],
                "commandIds": list(raw["commandIds"]),
                "previousRevision": raw["previousRevision"],
                "resultingRevision": raw["resultingRevision"],
                "timestamp": raw["timestamp"],
                "affectedIds": [identity for identity in raw["affectedIds"] if identity in safe_ids],
                "kind": raw["kind"],
            }
            transactions.append(transaction)

        private_proxies: dict[str, Any] = {}
        for proxy_id, proxy in document.private_proxies.items():
            asset = document.assets.get(proxy.proxy_asset_id)
            if asset is None or asset.content_hash != proxy.proxy_content_hash:
                raise AgentProjectionError("SAFE_PROXY_UNAVAILABLE")
            coverage = document.masks.get(proxy.coverage_matte_id) if proxy.coverage_matte_id else None
            private_proxies[proxy_id] = {
                "proxyId": proxy.proxy_id,
                "proxyAssetId": proxy.proxy_asset_id,
                "destinationSlotId": proxy.destination_slot_id,
                "registration": proxy.registration.to_dict(),
                "permittedDepthMetadata": deepcopy(proxy.permitted_depth_metadata),
                "coverageMatte": None if coverage is None else {
                    "present": True,
                    "purpose": "coverage",
                    "revision": coverage.revision,
                },
            }

        snapshot_token = f"{document.document_id}@{document.current_revision}"
        result = {
            "schemaVersion": 1,
            "documentId": document.document_id,
            "revision": document.current_revision,
            "snapshotToken": snapshot_token,
            "width": document.width,
            "height": document.height,
            "colorSpaceIntent": _safe_label(document.color_space_intent, "srgb"),
            "rootLayerIds": [identity for identity in document.root_layer_ids if identity in safe_layer_records],
            "layers": layers,
            "objects": objects,
            "assets": assets,
            "masks": masks,
            "selections": selections,
            "guides": guides,
            "relationalContext": relational_context,
            "recentTransactions": transactions,
            "privateProxies": private_proxies,
        }
        encoded = canonical_json(result)
        if len(encoded) > MAX_PROJECTED_BYTES:
            raise AgentProjectionError("SAFE_VIEW_TOO_LARGE")
        return result
    except AgentProjectionError:
        raise
    except (SchemaValidationError, KeyError, TypeError, ValueError, OverflowError) as exc:
        raise AgentProjectionError("SAFE_VIEW_UNAVAILABLE") from exc


def _visible_layer_ids(document: Document) -> tuple[set[str], dict[str, bool]]:
    visibility = _effective_visibility(document)
    director_only_layers = _director_only_layer_ids(document)
    return {identity for identity, visible in visibility.items()
            if visible and identity not in director_only_layers}, visibility


def _render_document(document: Document, visible_ids: set[str]) -> Document:
    """Construct a renderer-only document with no hidden or unallowlisted data."""

    layers: dict[str, LayerRecord] = {}
    objects: dict[str, ObjectRecord] = {}
    referenced_assets: set[str] = set()
    for layer_id in visible_ids:
        original = document.layers[layer_id]
        object_ids = [item for item in original.object_ids if item in document.objects]
        safe_objects: list[str] = []
        for object_id in object_ids:
            source = document.objects[object_id]
            geometry = _safe_geometry(source)
            style = _safe_style(source)
            clone = ObjectRecord(
                source.object_id, source.layer_id, source.kind, source.coordinate_space,
                geometry, source.transform, source.asset_id, style, 0, {}, {},
            )
            objects[object_id] = clone
            safe_objects.append(object_id)
            if clone.asset_id is not None:
                referenced_assets.add(clone.asset_id)
        layer_assets = [asset_id for asset_id in original.asset_ids if asset_id in document.assets]
        referenced_assets.update(layer_assets)
        children = [child for child in original.child_ids if child in visible_ids]
        layers[layer_id] = LayerRecord(
            original.layer_id, _safe_label(original.name, "Layer"), original.kind,
            original.parent_id if original.parent_id in visible_ids else None,
            children, True, original.locked, original.opacity, original.blend_mode,
            original.transform, safe_objects, layer_assets, [],
            _safe_label(original.role, "") or None, [], CollaborationState(), 0,
            False, None, None, {}, {},
        )

    proxy_layer_ids: list[str] = []
    for proxy_id, proxy in sorted(document.private_proxies.items()):
        asset = document.assets.get(proxy.proxy_asset_id)
        registration = proxy.registration
        if (asset is None or asset.external_uri is not None or asset.width is None or asset.height is None
                or registration.to_space != "document"
                or registration.source_dimensions != (asset.width, asset.height)
                or registration.target_dimensions != (document.width, document.height)):
            raise AgentProjectionError("SAFE_PROXY_UNAVAILABLE")
        proxy_key = sha256_bytes(proxy_id.encode("utf-8"))[:24]
        layer_id = f"safe-proxy-{proxy_key}"
        object_id = f"safe-proxy-object-{proxy_key}"
        if layer_id in document.layers or object_id in document.objects or layer_id in objects or object_id in layers:
            raise AgentProjectionError("SAFE_PROXY_UNAVAILABLE")
        layers[layer_id] = LayerRecord(
            layer_id=layer_id,
            name="Private source proxy",
            kind="raster",
            object_ids=[object_id],
            asset_ids=[asset.asset_id],
            role="safe-proxy",
            locked=True,
        )
        objects[object_id] = ObjectRecord(
            object_id=object_id,
            layer_id=layer_id,
            kind="raster-placement",
            coordinate_space="document",
            geometry={"x": 0, "y": 0, "width": asset.width, "height": asset.height},
            transform=registration.forward,
            asset_id=asset.asset_id,
        )
        proxy_layer_ids.append(layer_id)
        referenced_assets.add(asset.asset_id)

    assets: dict[str, AssetRecord] = {}
    for asset_id in referenced_assets:
        asset = document.assets.get(asset_id)
        if asset is None or asset.external_uri is not None:
            raise AgentProjectionError("SAFE_PREVIEW_UNAVAILABLE")
        assets[asset_id] = deepcopy(asset)

    roots = proxy_layer_ids + [identity for identity in document.root_layer_ids if identity in visible_ids]
    # Root children of a hidden parent cannot be visible; preserve only actual roots.
    safe_document = Document(
        document.document_id, document.width, document.height,
        color_space_intent=_safe_label(document.color_space_intent, "srgb"),
        root_layer_ids=roots, layers=layers, objects=objects, assets=assets,
    )
    try:
        safe_document.validate()
        safe_document.refresh_digest()
    except (SchemaValidationError, TypeError, ValueError) as exc:
        raise AgentProjectionError("SAFE_PREVIEW_UNAVAILABLE") from exc
    return safe_document


def render_agent_safe_preview(
    document: Document,
    asset_loader: Callable[[AssetRecord], bytes],
) -> tuple[bytes, int]:
    """Render a PNG from the visible allowlisted scene at one locked revision."""

    projection = build_agent_projection(document, transaction_limit=0)
    del projection
    visible_ids, _ = _visible_layer_ids(document)
    safe_document = _render_document(document, visible_ids)
    try:
        png = render_document_png(safe_document, lambda asset_id: asset_loader(safe_document.assets[asset_id]))
        with Image.open(io.BytesIO(png)) as image:
            image.thumbnail(MAX_PREVIEW_SIZE, Image.Resampling.LANCZOS)
            output = io.BytesIO()
            image.save(output, format="PNG", optimize=False, compress_level=9)
            content = output.getvalue()
    except AgentProjectionError:
        raise
    except (RenderError, OSError, KeyError, TypeError, ValueError) as exc:
        raise AgentProjectionError("SAFE_PREVIEW_UNAVAILABLE") from exc
    if len(content) > MAX_PREVIEW_BYTES:
        raise AgentProjectionError("SAFE_PREVIEW_TOO_LARGE")
    return content, document.current_revision
