"""Renderer-neutral human-editor mutations over the W02 document records."""

from __future__ import annotations

import io
import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

from PIL import Image, ImageChops, ImageDraw, ImageFilter

from .asset_store import AssetStore
from .editor_render import RenderError, image_bytes, render_layer
from .history import HistoryValidationError, RevisionConflict, TransactionRecord
from .ids import canonical_json, make_id, safe_relative_asset_path, sha256_bytes, validate_id
from .schema import (
    AssetRecord,
    CollaborationState,
    Document,
    GuideRecord,
    LayerRecord,
    MaskRecord,
    ObjectRecord,
    SchemaValidationError,
    SelectionRecord,
    SelectionState,
    VariantSet,
)
from .transforms import AffineTransform, BBox


class EditorActionError(ValueError):
    def __init__(self, code: str, message: str, *, current_revision: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.current_revision = current_revision


@dataclass
class PendingAsset:
    content: bytes
    record: AssetRecord


@dataclass
class PreparedAction:
    document: Document
    receipt: dict[str, Any]
    pending_assets: list[PendingAsset] = field(default_factory=list)


_MUTABLE_SCENE_FIELDS = {
    "width", "height", "colorSpaceIntent", "rootLayerIds", "layers", "objects", "assets", "masks",
    "selections", "variants", "interactionGroups", "operations", "depthComposites", "bbOperations",
    "candidates", "extractions", "guides", "patches", "externalRoundTrips", "privateProxies",
    "visualProfile", "metadata",
}


def _snapshot(document: Document) -> dict[str, Any]:
    # Serialize records individually so an in-flight candidate may contain
    # records stamped with the next revision before its transaction is
    # appended. Document.to_dict() correctly rejects that intermediate state.
    value = {
        "formatId": document.format_id,
        "schemaVersion": document.schema_version,
        "documentId": document.document_id,
        "width": document.width,
        "height": document.height,
        "colorSpaceIntent": document.color_space_intent,
        "rootLayerIds": list(document.root_layer_ids),
        "layers": {key: record.to_dict() for key, record in document.layers.items()},
        "objects": {key: record.to_dict() for key, record in document.objects.items()},
        "assets": {key: record.to_dict() for key, record in document.assets.items()},
        "masks": {key: record.to_dict() for key, record in document.masks.items()},
        "selections": {key: record.to_dict() for key, record in document.selections.items()},
        "variants": {key: record.to_dict() for key, record in document.variants.items()},
        "interactionGroups": {key: record.to_dict() for key, record in document.interaction_groups.items()},
        "operations": {key: record.to_dict() for key, record in document.operations.items()},
        "depthComposites": {key: record.to_dict() for key, record in document.depth_composites.items()},
        "bbOperations": {key: record.to_dict() for key, record in document.bb_operations.items()},
        "candidates": {key: record.to_dict() for key, record in document.candidates.items()},
        "extractions": {key: record.to_dict() for key, record in document.extractions.items()},
        "guides": {key: record.to_dict() for key, record in document.guides.items()},
        "patches": {key: record.to_dict() for key, record in document.patches.items()},
        "externalRoundTrips": {key: record.to_dict() for key, record in document.external_round_trips.items()},
        "privateProxies": {key: record.to_dict() for key, record in document.private_proxies.items()},
        "visualProfile": document.visual_profile.to_dict(),
        "metadata": deepcopy(document.metadata),
    }
    metadata = value.get("metadata", {})
    metadata.pop("editorUndoStack", None)
    metadata.pop("editorRedoStack", None)
    return value


def _restore_snapshot(document: Document, snapshot: Mapping[str, Any]) -> Document:
    """Restore scene fields while retaining monotonic document history."""

    current = document.to_dict(include_digest=False)
    value = deepcopy(dict(snapshot))
    value.update({
        "formatId": current["formatId"],
        "schemaVersion": current["schemaVersion"],
        "documentId": current["documentId"],
        "currentRevision": current["currentRevision"],
        "history": current["history"],
        "historyRefs": current["historyRefs"],
        "checkpointRefs": current["checkpointRefs"],
        "revisionDigest": None,
    })
    metadata = dict(value.get("metadata", {}))
    for field in ("editorUndoStack", "editorRedoStack"):
        if field in current["metadata"]:
            metadata[field] = current["metadata"][field]
    value["metadata"] = metadata
    return Document.from_dict(value)


def _record_value(document: Document, identity: str) -> Any:
    for collection in (
        document.layers, document.objects, document.masks, document.selections, document.guides,
        document.variants, document.assets, document.interaction_groups,
    ):
        if identity in collection:
            return collection[identity].to_dict()
    return None


def _transaction(
    before: Document,
    candidate: Document,
    *,
    action_type: str,
    actor_id: str,
    actor_kind: str,
    transaction_id: str,
    group_id: str,
    affected_ids: list[str],
    created_ids: list[str],
    invalidated_ids: list[str],
    undoable: bool,
    history_kind: str = "edit",
    extra_metadata: Mapping[str, Any] | None = None,
    stamp_ids: list[str] | None = None,
    command_id: str | None = None,
) -> tuple[Document, dict[str, Any]]:
    previous_revision = before.current_revision
    new_revision = previous_revision + 1
    candidate.current_revision = new_revision
    # A checkpoint reference is revision-local by the W02 schema. Preserve the
    # checkpoint on its retained on-disk manifest, then let the next explicit
    # save publish a new reference for this revision.
    candidate.checkpoint_refs = []
    for identity in (affected_ids if stamp_ids is None else stamp_ids):
        if identity in candidate.layers:
            candidate.layers[identity].revision = new_revision
        if identity in candidate.objects:
            candidate.objects[identity].revision = new_revision
        if identity in candidate.masks:
            candidate.masks[identity].revision = new_revision
        if identity in candidate.guides:
            candidate.guides[identity].state_revision = new_revision
        if identity in candidate.variants:
            candidate.variants[identity].revision = new_revision

    undo_stack = list(before.metadata.get("editorUndoStack", []))
    redo_stack = list(before.metadata.get("editorRedoStack", []))
    if action_type in {"undo", "redo"}:
        undo_stack = list(candidate.metadata.get("editorUndoStack", []))
        redo_stack = list(candidate.metadata.get("editorRedoStack", []))
    elif undoable:
        undo_stack.append(transaction_id)
        redo_stack = []
    else:
        # Do not allow a snapshot undo to cross a semantic transition that has
        # its own monotonic selection/guide lineage.
        undo_stack = []
        redo_stack = []
    candidate.metadata["editorUndoStack"] = undo_stack
    candidate.metadata["editorRedoStack"] = redo_stack

    scene_after = _snapshot(candidate)
    retained_after: Mapping[str, Any] = scene_after
    retained_metadata = {"editorAction": action_type, "undoable": undoable, **dict(extra_metadata or {})}
    if history_kind == "selection-rebase":
        content_rebased = next((
            selection for selection in candidate.selections.values()
            if selection.selection_id in affected_ids
            and selection.rebase_history
            and selection.rebase_history[-1]["reason"] == "SOURCE_CONTENT_CHANGED"
        ), None)
        if content_rebased is not None:
            # W02 requires a retained selection record after a content rebase so
            # the new mask and source digest remain independently auditable.
            retained_after = content_rebased.to_dict()
            retained_metadata["sceneAfter"] = scene_after

    transaction = TransactionRecord(
        transaction_id=transaction_id,
        group_id=group_id,
        actor_id=actor_id,
        actor_kind=actor_kind,
        command_ids=[command_id or make_id("cmd")],
        previous_revision=previous_revision,
        resulting_revision=new_revision,
        affected_ids=list(dict.fromkeys(affected_ids)),
        before=_snapshot(before),
        after=retained_after,
        delta={"actionType": action_type, "createdIds": created_ids, "invalidatedIds": invalidated_ids},
        kind=history_kind,
        metadata=retained_metadata,
    )
    transaction.validate(previous=TransactionRecord.from_dict(before.history[-1]) if before.history else None)
    candidate.history.append(transaction.to_dict())
    candidate.history_refs = [entry["transactionId"] for entry in candidate.history]
    candidate.revision_digest = None
    try:
        candidate.validate()
        candidate.refresh_digest()
    except (SchemaValidationError, ValueError, TypeError) as exc:
        raise EditorActionError("VALIDATION_FAILED", str(exc), current_revision=before.current_revision) from exc
    return candidate, {
        "status": "committed",
        "actionId": transaction.command_ids[0],
        "transactionId": transaction_id,
        "groupId": group_id,
        "documentId": candidate.document_id,
        "previousRevision": previous_revision,
        "currentRevision": new_revision,
        "newRevision": new_revision,
        "affectedIds": list(dict.fromkeys(affected_ids)),
        "createdIds": created_ids,
        "invalidatedDerivedIds": invalidated_ids,
        "code": None,
        "refreshHint": None,
    }


def _new_asset(data: bytes, *, media_type: str, extension: str, width: int | None = None,
               height: int | None = None, has_alpha: bool | None = None) -> PendingAsset:
    digest = sha256_bytes(data)
    asset_id = make_id("asset")
    extension = extension.lower().lstrip(".") or "bin"
    record = AssetRecord(
        asset_id=asset_id,
        content_hash=digest,
        media_type=media_type,
        storage_uri=safe_relative_asset_path(digest, extension),
        extension=extension,
        byte_length=len(data),
        width=width,
        height=height,
        has_alpha=has_alpha,
    )
    record.validate()
    return PendingAsset(data, record)


def _image_asset(pending: PendingAsset) -> bytes:
    return pending.content


def _add_pending(candidate: Document, pending: PendingAsset, pending_assets: list[PendingAsset]) -> AssetRecord:
    candidate.assets[pending.record.asset_id] = pending.record
    pending_assets.append(pending)
    return pending.record


def _load_image(asset_store: AssetStore, document: Document, asset_id: str) -> Image.Image:
    asset = document.assets.get(asset_id)
    if asset is None:
        raise EditorActionError("UNKNOWN_ASSET", "asset does not exist")
    if asset.external_uri is not None:
        raise EditorActionError("EXTERNAL_ASSET_UNAVAILABLE", "external assets must be embedded before editor use")
    try:
        with Image.open(io.BytesIO(asset_store.read_bytes(asset))) as opened:
            return opened.convert("RGBA")
    except EditorActionError:
        raise
    except Exception as exc:
        raise EditorActionError("INVALID_IMAGE_ASSET", "asset could not be decoded as an image") from exc


def _raster_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def _source_image(document: Document, asset_store: AssetStore, layer_id: str) -> Image.Image:
    try:
        return render_layer(document, layer_id, lambda asset_id: asset_store.read_bytes(document.assets[asset_id]))
    except RenderError as exc:
        raise EditorActionError(exc.code, str(exc)) from exc


def _selection_source(document: Document, asset_store: AssetStore, layer_id: str) -> tuple[Image.Image, str, int]:
    _require_layer(document, layer_id)
    layer = document.layers.get(layer_id)
    if layer is None or layer.kind != "raster" or layer.metadata.get("deleted", False):
        raise EditorActionError("UNSUPPORTED_SELECTION_SOURCE", "selection source must be an active raster layer")
    image = _source_image(document, asset_store, layer_id)
    digest = sha256_bytes(_raster_bytes(image))
    return image, digest, layer.revision


def _descendant_layer_ids(document: Document, layer_id: str) -> set[str]:
    result = {layer_id}
    pending = list(document.layers[layer_id].child_ids)
    while pending:
        child_id = pending.pop()
        if child_id not in result:
            result.add(child_id)
            pending.extend(document.layers[child_id].child_ids)
    return result


def _invalidate_source_selections(
    document: Document,
    asset_store: AssetStore,
    source_layer_ids: set[str],
    reason: str,
    next_revision: int,
) -> list[str]:
    invalidated = []
    for selection in document.selections.values():
        if selection.source_id in source_layer_ids and selection.state == SelectionState.CURRENT.value:
            digest = None
            if reason == "SOURCE_CONTENT_CHANGED":
                _, digest, _ = _selection_source(document, asset_store, selection.source_id)
            selection.mark_source_changed(reason, next_revision, digest)
            invalidated.append(selection.selection_id)
    return invalidated


def _mask_from_geometry(width: int, height: int, seed: Mapping[str, Any]) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    kind = seed.get("kind")
    geometry = seed.get("geometry", seed)
    if not isinstance(geometry, Mapping):
        raise EditorActionError("INVALID_SELECTION_SEED", "selection geometry must be an object")
    if kind in {"box", "rectangle"}:
        x, y = float(geometry.get("x", 0)), float(geometry.get("y", 0))
        w, h = float(geometry.get("width", 0)), float(geometry.get("height", 0))
        if w <= 0 or h <= 0:
            raise EditorActionError("INVALID_SELECTION_SEED", "selection box must have positive dimensions")
        draw.rectangle((x, y, x + w, y + h), fill=255)
    elif kind == "polygon":
        points = geometry.get("points")
        if not isinstance(points, list) or len(points) < 3:
            raise EditorActionError("INVALID_SELECTION_SEED", "polygon selection requires at least three points")
        draw.polygon([(float(p[0]), float(p[1])) for p in points], fill=255)
    elif kind in {"paint", "brush"}:
        points = geometry.get("points")
        radius = float(geometry.get("size", geometry.get("radius", 12)))
        if not isinstance(points, list) or not points or not 1 <= radius <= 512:
            raise EditorActionError("INVALID_SELECTION_SEED", "paint selection requires points and a bounded size")
        values = [(float(p[0]), float(p[1])) for p in points]
        draw.line(values, fill=255, width=max(1, round(radius)), joint="curve")
        for x, y in values:
            draw.ellipse((x - radius / 2, y - radius / 2, x + radius / 2, y + radius / 2), fill=255)
    else:
        raise EditorActionError("INVALID_SELECTION_SEED", "unsupported selection seed kind")
    return mask


def _flood_mask(source: Image.Image, seed: Mapping[str, Any]) -> Image.Image:
    x, y = float(seed.get("x", 0)), float(seed.get("y", 0))
    px, py = math.floor(x), math.floor(y)
    if px < 0 or py < 0 or px >= source.width or py >= source.height:
        raise EditorActionError("INVALID_SELECTION_SEED", "point seed is outside the document")
    tolerance = seed.get("tolerance", 24)
    if type(tolerance) is not int or not 0 <= tolerance <= 96:
        raise EditorActionError("INVALID_SELECTION_SEED", "point tolerance must be an integer from 0 to 96")
    original = source.convert("RGB")
    work = original.copy()
    colors = original.getcolors(maxcolors=131072)
    known = {color for _, color in colors} if colors is not None else set()
    sentinel = next((color for color in ((255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 0, 0), (255, 255, 255)) if color not in known), (1, 0, 1))
    ImageDraw.floodfill(work, (px, py), sentinel, thresh=tolerance)
    difference = ImageChops.difference(work, original)
    return difference.convert("L").point(lambda value: 255 if value else 0)


def _combine_seed(mask: Image.Image, seed_mask: Image.Image, mode: str) -> Image.Image:
    if mode in {"replace", "add"}:
        return seed_mask if mode == "replace" else ImageChops.lighter(mask, seed_mask)
    if mode == "subtract":
        return ImageChops.subtract(mask, seed_mask)
    if mode == "intersect":
        return ImageChops.darker(mask, seed_mask)
    raise EditorActionError("INVALID_SELECTION_OPERATION", "seed mode must be replace, add, subtract, or intersect")


def _seed_mask(document: Document, asset_store: AssetStore, source_image: Image.Image,
               seeds: list[Mapping[str, Any]]) -> Image.Image:
    result = Image.new("L", (document.width, document.height), 0)
    seen = False
    for seed in seeds:
        if not isinstance(seed, Mapping):
            raise EditorActionError("INVALID_SELECTION_SEED", "selection seeds must be objects")
        kind = seed.get("kind")
        if kind == "point":
            part = _flood_mask(source_image, seed)
        elif kind in {"box", "rectangle", "polygon", "paint", "brush"}:
            part = _mask_from_geometry(document.width, document.height, seed)
        elif kind == "alpha":
            part = source_image.getchannel("A")
        elif kind == "mask":
            mask_id = seed.get("maskId")
            mask_record = document.masks.get(mask_id) if isinstance(mask_id, str) else None
            if mask_record is None or mask_record.asset_id is None or mask_record.purpose not in {"editing-selection", "layer-alpha", "visibility"}:
                raise EditorActionError("INVALID_SELECTION_SEED", "seed mask is missing or has an unsupported purpose")
            part = _load_image(asset_store, document, mask_record.asset_id).convert("L")
            if part.size != result.size:
                raise EditorActionError("INVALID_SELECTION_SEED", "seed mask dimensions differ from the document")
        else:
            raise EditorActionError("INVALID_SELECTION_SEED", "unsupported selection seed kind")
        mode = seed.get("mode", "replace" if not seen else "add")
        result = _combine_seed(result, part, mode)
        seen = True
    if not seen or result.getbbox() is None:
        raise EditorActionError("EMPTY_SELECTION", "selection seeds did not select any document pixels")
    return result


def _selection_mask(document: Document, asset_store: AssetStore, selection: SelectionRecord) -> Image.Image:
    mask_record = document.masks.get(selection.mask_id)
    if mask_record is None or mask_record.asset_id is None:
        raise EditorActionError("MISSING_SELECTION_MASK", "selection mask is unavailable")
    image = _load_image(asset_store, document, mask_record.asset_id).convert("L")
    if image.size != (document.width, document.height):
        raise EditorActionError("INVALID_SELECTION_MASK", "selection mask dimensions differ from the document")
    return image


def _bounds(mask: Image.Image) -> BBox:
    bbox = mask.getbbox()
    if bbox is None:
        raise EditorActionError("EMPTY_SELECTION", "selection mask is empty")
    x1, y1, x2, y2 = bbox
    return BBox(y1, y2, x1, x2)


def _selection_refinement(mask: Image.Image, operation: str, data: Mapping[str, Any]) -> Image.Image:
    if operation in {"add", "subtract", "intersect", "paint"}:
        seed = data.get("seed")
        if not isinstance(seed, Mapping):
            raise EditorActionError("INVALID_SELECTION_SEED", "refinement requires one seed")
        if seed.get("kind") == "point":
            raise EditorActionError("INVALID_SELECTION_SEED", "point assist is available when creating a selection")
        part = _mask_from_geometry(mask.width, mask.height, seed)
        if operation in {"add", "paint"}:
            return ImageChops.lighter(mask, part)
        if operation == "subtract":
            return ImageChops.subtract(mask, part)
        return ImageChops.darker(mask, part)
    if operation == "invert":
        return ImageChops.invert(mask)
    radius = data.get("radius", 1)
    if type(radius) is not int or not 1 <= radius <= 128:
        raise EditorActionError("INVALID_MASK_RADIUS", "mask radius must be an integer from 1 to 128 document pixels")
    size = radius * 2 + 1
    if operation == "grow":
        return mask.filter(ImageFilter.MaxFilter(size))
    if operation == "shrink":
        return mask.filter(ImageFilter.MinFilter(size))
    if operation == "feather":
        return mask.filter(ImageFilter.GaussianBlur(radius))
    if operation == "clean":
        return mask.filter(ImageFilter.MinFilter(size)).filter(ImageFilter.MaxFilter(size))
    raise EditorActionError("INVALID_SELECTION_OPERATION", "unsupported mask refinement")


def _asset_data(document: Document, asset_store: AssetStore, asset_id: str) -> bytes:
    record = document.assets.get(asset_id)
    if record is None:
        raise EditorActionError("UNKNOWN_ASSET", "asset does not exist")
    if record.external_uri is not None:
        raise EditorActionError("EXTERNAL_ASSET_UNAVAILABLE", "external assets must be embedded before editor use")
    try:
        return asset_store.read_bytes(record)
    except Exception as exc:
        raise EditorActionError("ASSET_READ_FAILED", "asset could not be read") from exc


def _require_layer(
    document: Document,
    layer_id: str,
    *,
    allow_locked: bool = False,
    allow_target_locked: bool = False,
) -> LayerRecord:
    layer = document.layers.get(layer_id)
    if layer is None or layer.metadata.get("deleted", False):
        raise EditorActionError("UNKNOWN_LAYER", "layer does not exist")
    current = layer
    while current is not None:
        if current.locked and not allow_locked and not (allow_target_locked and current is layer):
            raise EditorActionError("LAYER_LOCKED", "target layer or parent group is locked")
        current = document.layers.get(current.parent_id) if current.parent_id else None
    return layer


def _add_layer(document: Document, layer: LayerRecord) -> None:
    parent_id = layer.parent_id
    if parent_id is None:
        document.root_layer_ids.append(layer.layer_id)
    else:
        parent = _require_layer(document, parent_id)
        if parent.kind != "group":
            raise EditorActionError("INVALID_LAYER_PARENT", "layers may only be parented under a group")
        parent.child_ids.append(layer.layer_id)
    document.layers[layer.layer_id] = layer


def _sibling_list(document: Document, layer: LayerRecord) -> list[str]:
    return document.root_layer_ids if layer.parent_id is None else document.layers[layer.parent_id].child_ids


def _matrix_from_payload(value: object) -> AffineTransform:
    if not isinstance(value, (list, tuple)) or len(value) != 9:
        raise EditorActionError("INVALID_TRANSFORM", "transform must contain nine affine values")
    try:
        return AffineTransform.from_values(value)
    except Exception as exc:
        raise EditorActionError("INVALID_TRANSFORM", "transform is singular or malformed") from exc


def _rgba_color(value: object) -> str:
    if not isinstance(value, str) or len(value) not in {7, 9} or not value.startswith("#"):
        raise EditorActionError("INVALID_COLOR", "color must be #RRGGBB or #RRGGBBAA")
    try:
        bytes.fromhex(value[1:])
    except ValueError as exc:
        raise EditorActionError("INVALID_COLOR", "color is malformed") from exc
    return value.upper()


def _check_json(value: Any) -> None:
    try:
        canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise EditorActionError("INVALID_PAYLOAD", "action data must be finite JSON") from exc


def _apply_action(
    document: Document,
    asset_store: AssetStore,
    action: Mapping[str, Any],
    pending_assets: list[PendingAsset],
) -> tuple[list[str], list[str], list[str], bool, str, dict[str, Any]]:
    action_type = action.get("actionType")
    data = action.get("data", {})
    if not isinstance(action_type, str) or not isinstance(data, Mapping):
        raise EditorActionError("INVALID_PAYLOAD", "action type and document-space data are required")
    _check_json(data)
    affected: list[str] = []
    created: list[str] = []
    invalidated: list[str] = []
    undoable = True
    history_kind = "edit"
    tx_metadata: dict[str, Any] = {}

    if action_type == "add_layer":
        kind = data.get("kind", "vector")
        if kind not in {"raster", "paint", "vector", "group", "guide"}:
            raise EditorActionError("INVALID_LAYER_KIND", "unsupported layer kind")
        layer_id = make_id("layer")
        parent_id = data.get("parentId")
        layer = LayerRecord(layer_id, str(data.get("name", "Layer")), kind, parent_id=parent_id,
                            role=data.get("role"), metadata={"createdBy": "human-editor"})
        _add_layer(document, layer)
        affected.append(layer_id)
        created.append(layer_id)
    elif action_type == "rename_layer":
        layer_id = data.get("layerId")
        layer = _require_layer(document, layer_id)
        name = data.get("name")
        if not isinstance(name, str) or not name.strip() or len(name) > 120:
            raise EditorActionError("INVALID_LAYER_NAME", "layer name must contain 1 to 120 characters")
        layer.name = name.strip()
        affected.append(layer_id)
    elif action_type in {"set_layer_visibility", "set_layer_lock", "set_layer_opacity"}:
        layer_id = data.get("layerId")
        layer = _require_layer(
            document,
            layer_id,
            allow_target_locked=(action_type == "set_layer_lock"),
        )
        if action_type == "set_layer_visibility":
            if type(data.get("visible")) is not bool:
                raise EditorActionError("INVALID_VISIBILITY", "visible must be boolean")
            layer.visible = data["visible"]
        elif action_type == "set_layer_lock":
            if type(data.get("locked")) is not bool:
                raise EditorActionError("INVALID_LOCK_STATE", "locked must be boolean")
            layer.locked = data["locked"]
        else:
            opacity = data.get("opacity")
            if type(opacity) not in (int, float) or not 0 <= float(opacity) <= 1:
                raise EditorActionError("INVALID_OPACITY", "opacity must be between zero and one")
            layer.opacity = float(opacity)
        affected.append(layer_id)
        if action_type == "set_layer_opacity":
            invalidated.extend(_invalidate_source_selections(
                document, asset_store, _descendant_layer_ids(document, layer_id), "SOURCE_CONTENT_CHANGED", document.current_revision + 1
            ))
            if invalidated:
                affected.extend(invalidated)
                undoable = False
    elif action_type == "reorder_layer":
        layer_id = data.get("layerId")
        layer = _require_layer(document, layer_id)
        siblings = _sibling_list(document, layer)
        old_index = siblings.index(layer_id)
        direction = data.get("direction")
        index = data.get("index")
        if index is None:
            index = old_index + (1 if direction == "up" else -1 if direction == "down" else 0)
        if type(index) is not int or not 0 <= index < len(siblings):
            raise EditorActionError("INVALID_LAYER_ORDER", "requested layer position is outside its sibling group")
        siblings.pop(old_index)
        siblings.insert(index, layer_id)
        affected.append(layer_id)
    elif action_type == "reparent_layer":
        layer_id, parent_id = data.get("layerId"), data.get("parentId")
        layer = _require_layer(document, layer_id)
        if parent_id == layer_id:
            raise EditorActionError("INVALID_LAYER_PARENT", "a layer cannot parent itself")
        siblings = _sibling_list(document, layer)
        siblings.remove(layer_id)
        if parent_id is None:
            document.root_layer_ids.append(layer_id)
        else:
            parent = _require_layer(document, parent_id)
            if parent.kind != "group" or parent.locked:
                raise EditorActionError("INVALID_LAYER_PARENT", "new parent must be an unlocked group")
            parent.child_ids.append(layer_id)
            affected.append(parent_id)
        layer.parent_id = parent_id
        affected.append(layer_id)
        invalidated.extend(_invalidate_source_selections(
            document, asset_store, _descendant_layer_ids(document, layer_id), "SOURCE_CONTENT_CHANGED", document.current_revision + 1
        ))
        if invalidated:
            affected.extend(invalidated)
            undoable = False
    elif action_type == "delete_layer":
        layer_id = data.get("layerId")
        layer = _require_layer(document, layer_id)
        if data.get("confirmed") is not True:
            raise EditorActionError("CONFIRMATION_REQUIRED", "layer deletion requires explicit confirmation")
        if layer.kind == "group" and layer.child_ids:
            raise EditorActionError("GROUP_NOT_EMPTY", "move or delete group children before deleting the group")
        layer.metadata["deleted"] = True
        layer.visible = False
        layer.locked = True
        selections = [selection for selection in document.selections.values() if selection.source_id == layer_id and selection.state == SelectionState.CURRENT.value]
        if selections:
            new_revision = document.current_revision + 1
            for selection in selections:
                selection.mark_source_changed("SOURCE_REMOVED", new_revision)
                invalidated.append(selection.selection_id)
            undoable = False
        affected.extend([layer_id, *invalidated])
    elif action_type == "duplicate_layer":
        layer_id = data.get("layerId")
        source = _require_layer(document, layer_id)
        if source.kind == "group" and source.child_ids:
            raise EditorActionError("GROUP_DUPLICATE_UNSUPPORTED", "duplicate one logical layer at a time")
        duplicate_id = make_id("layer")
        duplicate = deepcopy(source)
        duplicate.layer_id = duplicate_id
        duplicate.name = str(data.get("name", f"{source.name} copy"))[:120]
        duplicate.revision = document.current_revision + 1
        duplicate.metadata = {**duplicate.metadata, "duplicatedFrom": source.layer_id}
        new_objects = []
        for old_object_id in source.object_ids:
            old_object = document.objects[old_object_id]
            object_id = make_id("obj")
            copied = deepcopy(old_object)
            copied.object_id = object_id
            copied.layer_id = duplicate_id
            duplicate.object_ids[duplicate.object_ids.index(old_object_id)] = object_id
            document.objects[object_id] = copied
            new_objects.append(object_id)
        _add_layer(document, duplicate)
        affected.extend([source.layer_id, duplicate_id, *new_objects])
        created.extend([duplicate_id, *new_objects])
    elif action_type == "transform":
        target_ids = action.get("targetIds", data.get("targetIds", []))
        if not isinstance(target_ids, list) or not target_ids or len(set(target_ids)) != len(target_ids):
            raise EditorActionError("INVALID_TARGETS", "transform requires unique target IDs")
        transform = _matrix_from_payload(data.get("transform"))
        for target_id in target_ids:
            if target_id in document.layers:
                layer = _require_layer(document, target_id)
                layer.transform = transform
                affected.append(target_id)
                dependent = _invalidate_source_selections(
                    document, asset_store, _descendant_layer_ids(document, target_id), "SOURCE_TRANSFORM_CHANGED", document.current_revision + 1
                )
                if dependent:
                    invalidated.extend(dependent)
                    undoable = False
                    affected.extend(dependent)
            elif target_id in document.objects:
                obj = document.objects[target_id]
                _require_layer(document, obj.layer_id)
                obj.transform = transform
                affected.extend([target_id, obj.layer_id])
                dependent = _invalidate_source_selections(
                    document, asset_store, {obj.layer_id}, "SOURCE_CONTENT_CHANGED", document.current_revision + 1
                )
                if dependent:
                    invalidated.extend(dependent)
                    undoable = False
                    affected.extend(dependent)
            else:
                raise EditorActionError("UNKNOWN_TARGET", "transform target does not exist")
    elif action_type == "create_object":
        layer_id = data.get("layerId")
        layer = _require_layer(document, layer_id)
        kind = data.get("kind")
        if kind not in {"shape", "path", "paint-stroke", "guide"}:
            raise EditorActionError("INVALID_OBJECT_KIND", "unsupported editable object kind")
        if layer.kind not in {"paint", "vector", "shape", "guide"}:
            raise EditorActionError("INVALID_OBJECT_LAYER", "editable shapes and strokes require a paint, vector, shape, or guide layer")
        geometry, style = data.get("geometry"), data.get("style", {})
        if not isinstance(geometry, Mapping) or not isinstance(style, Mapping):
            raise EditorActionError("INVALID_GEOMETRY", "object geometry and style must be objects")
        _validate_geometry(document, kind, geometry, style)
        object_id = make_id("obj")
        obj = ObjectRecord(object_id, layer_id, kind, "document", dict(geometry), AffineTransform.identity(),
                           None, dict(style), document.current_revision + 1)
        document.objects[object_id] = obj
        layer.object_ids.append(object_id)
        affected.extend([layer_id, object_id])
        created.append(object_id)
        invalidated.extend(_invalidate_source_selections(
            document, asset_store, {layer_id}, "SOURCE_CONTENT_CHANGED", document.current_revision + 1
        ))
        if invalidated:
            affected.extend(invalidated)
            undoable = False
    elif action_type == "import_image":
        raw = action.get("fileBytes")
        filename = data.get("filename", "import.png")
        if not isinstance(raw, bytes) or not isinstance(filename, str):
            raise EditorActionError("INVALID_IMAGE_UPLOAD", "image import bytes and filename are required")
        if len(raw) > 100 * 1024 * 1024:
            raise EditorActionError("IMAGE_TOO_LARGE", "image uploads are limited to 100 MiB")
        try:
            with Image.open(io.BytesIO(raw)) as opened:
                opened.verify()
            with Image.open(io.BytesIO(raw)) as opened:
                width, height = opened.size
                if width <= 0 or height <= 0 or width * height > 80_000_000:
                    raise EditorActionError("IMAGE_DIMENSIONS_UNSUPPORTED", "image dimensions exceed the supported limit")
                has_alpha = "A" in opened.getbands()
                media = Image.MIME.get(opened.format or "", "application/octet-stream")
                image_format = (opened.format or "PNG").lower()
        except EditorActionError:
            raise
        except Exception as exc:
            raise EditorActionError("INVALID_IMAGE_UPLOAD", "uploaded file is not a valid supported image") from exc
        pending = _new_asset(raw, media_type=media, extension=image_format, width=width, height=height, has_alpha=has_alpha)
        record = _add_pending(document, pending, pending_assets)
        layer_id, object_id = make_id("layer"), make_id("obj")
        parent_id = data.get("parentId")
        as_guide = data.get("asGuide") is True
        semantic_role = data.get("semanticRole")
        guide_id = make_id("guide") if as_guide else None
        layer_kind = "guide" if as_guide else "raster"
        layer_metadata = {"sourceFilename": filename[:180], "imported": True}
        if guide_id:
            layer_metadata["guideId"] = guide_id
        layer = LayerRecord(layer_id, data.get("name", filename[:96] or "Imported image"), layer_kind,
                            parent_id=parent_id, asset_ids=[record.asset_id],
                            role=semantic_role if as_guide else None, metadata=layer_metadata)
        raster = ObjectRecord(object_id, layer_id, "raster-placement", "document",
                              {"x": 0, "y": 0, "width": width, "height": height},
                              AffineTransform.identity(), record.asset_id, {}, document.current_revision + 1)
        layer.object_ids.append(object_id)
        _add_layer(document, layer)
        document.objects[object_id] = raster
        affected.extend([record.asset_id, layer_id, object_id])
        created.extend([record.asset_id, layer_id, object_id])
        if as_guide:
            if not isinstance(semantic_role, str) or not semantic_role.strip() or len(semantic_role) > 120:
                raise EditorActionError("INVALID_GUIDE_ROLE", "imported guide needs a semantic role from 1 to 120 characters")
            guide = GuideRecord(guide_id, layer.name, "proposed", document.current_revision + 1,
                               document.current_revision + 1, semantic_role=semantic_role.strip(), object_ids=[object_id],
                               metadata={"source": "imported-image"})
            document.guides[guide_id] = guide
            affected.append(guide_id)
            created.append(guide_id)
            undoable = False
    elif action_type == "create_selection":
        source_id = data.get("sourceLayerId")
        source, source_digest, source_revision = _selection_source(document, asset_store, source_id)
        seeds = data.get("seeds")
        if not isinstance(seeds, list):
            raise EditorActionError("INVALID_SELECTION_SEED", "selection requires a seed array")
        mask = _seed_mask(document, asset_store, source, seeds)
        pending = _new_asset(_raster_bytes(mask), media_type="image/png", extension="png",
                             width=document.width, height=document.height, has_alpha=False)
        mask_record = _add_pending(document, pending, pending_assets)
        selection_id, mask_id = make_id("sel"), make_id("mask")
        selection_mask = MaskRecord(mask_id, "editing-selection", "document", document.current_revision + 1,
                                    asset_id=mask_record.asset_id, owner_id=source_id,
                                    lineage={"selectionId": selection_id, "derivation": "human-seed"},
                                    content_hash=mask_record.content_hash)
        document.masks[mask_id] = selection_mask
        document.layers[source_id].mask_ids.append(mask_id)
        selection = SelectionRecord(
            selection_id, source_id, source_revision, source_digest, 1, SelectionState.CURRENT.value,
            mask_id, coordinate_space="document", seed_geometry=[dict(seed) for seed in seeds],
            semantic_hint=data.get("semanticHint"), bounds=_bounds(mask),
        )
        document.selections[selection_id] = selection
        affected.extend([source_id, mask_id, selection_id, mask_record.asset_id])
        created.extend([selection_id, mask_id, mask_record.asset_id])
        undoable = False
        history_kind = "selection-create"
    elif action_type == "refine_selection":
        selection_id = data.get("selectionId")
        selection = document.selections.get(selection_id)
        if selection is None:
            raise EditorActionError("UNKNOWN_SELECTION", "selection does not exist")
        _require_layer(document, selection.source_id)
        if selection.state != SelectionState.CURRENT.value:
            raise EditorActionError("STALE_SOURCE_CONFLICT" if selection.state == SelectionState.STALE_SOURCE.value else "SELECTION_NEEDS_REBASE",
                                    "stale selections must be explicitly rebased before refinement")
        expected = data.get("expectedSelectionRevision")
        if type(expected) is not int or expected != selection.selection_revision:
            raise EditorActionError("STALE_SELECTION_REVISION", "selection revision changed", current_revision=document.current_revision)
        current_mask = _selection_mask(document, asset_store, selection)
        new_mask = _selection_refinement(current_mask, data.get("operation"), data)
        if new_mask.getbbox() is None:
            raise EditorActionError("EMPTY_SELECTION", "refinement cannot produce an empty selection")
        pending = _new_asset(_raster_bytes(new_mask), media_type="image/png", extension="png",
                             width=document.width, height=document.height, has_alpha=False)
        asset = _add_pending(document, pending, pending_assets)
        mask_id = make_id("mask")
        document.masks[mask_id] = MaskRecord(mask_id, "editing-selection", "document", document.current_revision + 1,
                                             asset_id=asset.asset_id, owner_id=selection.source_id,
                                             lineage={"selectionId": selection_id, "derivation": data["operation"]},
                                             content_hash=asset.content_hash)
        document.layers[selection.source_id].mask_ids.append(mask_id)
        selection.refine(mask_id, expected_selection_revision=expected,
                         refinement={"operation": data["operation"], "radius": data.get("radius"), "details": data.get("details", {})})
        selection.bounds = _bounds(new_mask)
        affected.extend([selection_id, mask_id, asset.asset_id])
        created.extend([mask_id, asset.asset_id])
        undoable = False
        history_kind = "selection-refine"
    elif action_type == "rebase_selection":
        selection_id = data.get("selectionId")
        selection = document.selections.get(selection_id)
        if selection is None:
            raise EditorActionError("UNKNOWN_SELECTION", "selection does not exist")
        _require_layer(document, selection.source_id)
        expected = data.get("expectedSelectionRevision")
        if type(expected) is not int or expected != selection.selection_revision:
            raise EditorActionError("STALE_SELECTION_REVISION", "selection revision changed", current_revision=document.current_revision)
        source_id = selection.source_id
        if source_id not in document.layers or document.layers[source_id].metadata.get("deleted", False):
            raise EditorActionError("SOURCE_REMOVED", "removed selection sources cannot be rebased")
        source, source_digest, source_revision = _selection_source(document, asset_store, source_id)
        geometry_only = data.get("geometryOnly") is True
        new_mask_id = None
        if not geometry_only:
            seeds = data.get("seeds")
            if not isinstance(seeds, list):
                raise EditorActionError("MASK_REDERIVATION_REQUIRED", "content or clip rebases require new mask seeds")
            new_mask = _seed_mask(document, asset_store, source, seeds)
            pending = _new_asset(_raster_bytes(new_mask), media_type="image/png", extension="png",
                                 width=document.width, height=document.height, has_alpha=False)
            asset = _add_pending(document, pending, pending_assets)
            new_mask_id = make_id("mask")
            document.masks[new_mask_id] = MaskRecord(new_mask_id, "editing-selection", "document", document.current_revision + 1,
                                                     asset_id=asset.asset_id, owner_id=source_id,
                                                     lineage={"selectionId": selection_id, "derivation": "explicit-rebase"},
                                                     content_hash=asset.content_hash)
            document.layers[source_id].mask_ids.append(new_mask_id)
            selection.bounds = _bounds(new_mask)
            created.extend([new_mask_id, asset.asset_id])
        selection.explicit_rebase(
            source_revision,
            expected_selection_revision=expected,
            new_mask_id=new_mask_id,
            source_content_digest=(source_digest if selection.stale_reason == "SOURCE_CONTENT_CHANGED" else None),
            geometry_only=geometry_only,
        )
        affected.append(selection_id)
        if new_mask_id:
            affected.append(new_mask_id)
        undoable = False
        history_kind = "selection-rebase"
    elif action_type == "duplicate_selection_to_layer":
        selection_id = data.get("selectionId")
        selection = document.selections.get(selection_id)
        if selection is None:
            raise EditorActionError("UNKNOWN_SELECTION", "selection does not exist")
        _require_layer(document, selection.source_id)
        expected = data.get("expectedSelectionRevision")
        if type(expected) is not int or expected != selection.selection_revision:
            raise EditorActionError("STALE_SELECTION_REVISION", "selection revision changed", current_revision=document.current_revision)
        if selection.state != SelectionState.CURRENT.value:
            raise EditorActionError("STALE_SOURCE_CONFLICT" if selection.state == SelectionState.STALE_SOURCE.value else "SELECTION_NEEDS_REBASE",
                                    "stale selections cannot be duplicated")
        source = _source_image(document, asset_store, selection.source_id)
        mask = _selection_mask(document, asset_store, selection)
        source.putalpha(ImageChops.multiply(source.getchannel("A"), mask))
        pending = _new_asset(_raster_bytes(source), media_type="image/png", extension="png",
                             width=document.width, height=document.height, has_alpha=True)
        asset = _add_pending(document, pending, pending_assets)
        layer_id, object_id = make_id("layer"), make_id("obj")
        layer = LayerRecord(layer_id, str(data.get("name", "Selection copy"))[:120], "raster",
                            parent_id=data.get("parentId"), asset_ids=[asset.asset_id],
                            lineage={"selectionId": selection_id, "sourceLayerId": selection.source_id,
                                     "selectionRevision": selection.selection_revision, "derivation": "duplicate-selection"})
        obj = ObjectRecord(object_id, layer_id, "raster-placement", "document",
                           {"x": 0, "y": 0, "width": document.width, "height": document.height},
                           AffineTransform.identity(), asset.asset_id,
                           lineage={"selectionId": selection_id, "sourceLayerId": selection.source_id})
        layer.object_ids.append(object_id)
        _add_layer(document, layer)
        document.objects[object_id] = obj
        selection.register_derived_layer(layer_id, expected_selection_revision=expected)
        affected.extend([selection_id, layer_id, object_id, asset.asset_id])
        created.extend([layer_id, object_id, asset.asset_id])
        undoable = False
        history_kind = "selection-duplicate"
    elif action_type == "create_context_mask":
        seeds = data.get("seeds")
        if not isinstance(seeds, list):
            raise EditorActionError("INVALID_CONTEXT_MASK", "context mask requires box, polygon, or paint seeds")
        source_id = data.get("sourceLayerId")
        if source_id is not None:
            _require_layer(document, source_id)
            source = _source_image(document, asset_store, source_id)
        else:
            source = Image.new("RGBA", (document.width, document.height), (0, 0, 0, 255))
        mask = _seed_mask(document, asset_store, source, seeds)
        pending = _new_asset(_raster_bytes(mask), media_type="image/png", extension="png",
                             width=document.width, height=document.height, has_alpha=False)
        asset = _add_pending(document, pending, pending_assets)
        mask_id = make_id("mask")
        document.masks[mask_id] = MaskRecord(mask_id, "context", "document", document.current_revision + 1,
                                             asset_id=asset.asset_id, owner_id=source_id,
                                             lineage={"relationalReferences": [], "dilationPx": 0, "editMaskId": None},
                                             content_hash=asset.content_hash)
        if source_id is not None:
            document.layers[source_id].mask_ids.append(mask_id)
        affected.extend([mask_id, asset.asset_id])
        created.extend([mask_id, asset.asset_id])
        undoable = False
    elif action_type == "set_relational_context":
        mask_id = data.get("contextMaskId")
        mask = document.masks.get(mask_id)
        if mask is None or mask.purpose != "context":
            raise EditorActionError("INVALID_CONTEXT_MASK", "target must be a context mask")
        # Updating the lineage mutates the context mask owned by this layer or
        # object. A locked owner (or any locked ancestor of its layer) refuses
        # the write. Reference layers are read-only inputs and may be locked.
        if mask.owner_id in document.layers:
            _require_layer(document, mask.owner_id)
        elif mask.owner_id in document.objects:
            _require_layer(document, document.objects[mask.owner_id].layer_id)
        references = data.get("referenceIds")
        dilation = data.get("dilationPx")
        edit_mask_id = data.get("editMaskId")
        edit_target_ids = data.get("editTargetIds", [])
        if (not isinstance(references, list) or any(not isinstance(value, str) for value in references)
                or len(set(references)) != len(references)):
            raise EditorActionError("INVALID_CONTEXT_REFERENCES", "reference IDs must be a unique array")
        if type(dilation) is not int or not 0 <= dilation <= 256:
            raise EditorActionError("INVALID_CONTEXT_DILATION", "context dilation must be 0 to 256 document pixels")
        if not isinstance(edit_target_ids, list) or any(not isinstance(value, str) for value in edit_target_ids):
            raise EditorActionError("INVALID_CONTEXT_REFERENCES", "edit target IDs must be an array")
        if len(set(edit_target_ids)) != len(edit_target_ids):
            raise EditorActionError("INVALID_CONTEXT_REFERENCES", "edit target IDs must be unique")
        if set(references) & set(edit_target_ids):
            raise EditorActionError("CONTEXT_EDIT_SCOPE_OVERLAP", "relational references must be outside the edit targets")
        for identity in edit_target_ids:
            if identity not in document.layers and identity not in document.objects:
                raise EditorActionError("INVALID_CONTEXT_EDIT_TARGET", "edit target does not exist")
        def layer_scope(identity: str) -> set[str]:
            if identity in document.layers:
                root_id = identity
                scope = {root_id}
                pending = list(document.layers[root_id].child_ids)
                while pending:
                    child_id = pending.pop()
                    if child_id not in scope:
                        scope.add(child_id)
                        pending.extend(document.layers[child_id].child_ids)
                return scope
            owner_id = document.objects[identity].layer_id
            scope = set()
            current = document.layers[owner_id]
            while current is not None:
                scope.add(current.layer_id)
                current = document.layers.get(current.parent_id) if current.parent_id else None
            return scope
        edit_scope = set().union(*(layer_scope(identity) for identity in edit_target_ids)) if edit_target_ids else set()
        for identity in references:
            layer = document.layers.get(identity)
            obj = document.objects.get(identity)
            reference_scope = layer_scope(identity)
            if reference_scope & edit_scope:
                raise EditorActionError("CONTEXT_EDIT_SCOPE_OVERLAP", "relational references must be outside the edit targets")
            if layer is not None:
                current = layer
                visible = True
                while current is not None:
                    visible = visible and current.visible and not current.metadata.get("deleted", False)
                    current = document.layers.get(current.parent_id) if current.parent_id else None
                if not visible:
                    raise EditorActionError("INVALID_CONTEXT_REFERENCES", "context references must be visible")
            elif obj is not None:
                current = document.layers[obj.layer_id]
                visible = True
                while current is not None:
                    visible = visible and current.visible and not current.metadata.get("deleted", False)
                    current = document.layers.get(current.parent_id) if current.parent_id else None
                if not visible:
                    raise EditorActionError("INVALID_CONTEXT_REFERENCES", "context references must be visible")
            else:
                raise EditorActionError("INVALID_CONTEXT_REFERENCES", "context reference does not exist")
        if edit_mask_id is not None:
            edit_mask = document.masks.get(edit_mask_id)
            if edit_mask is None or edit_mask.purpose != "generation":
                raise EditorActionError("INVALID_EDIT_MASK", "edit mask must be a separate generation-purpose mask")
        mask.lineage = {**mask.lineage, "relationalReferences": list(references),
                        "dilationPx": dilation, "editMaskId": edit_mask_id,
                        "editTargetIds": list(edit_target_ids)}
        affected.append(mask_id)
        undoable = False
    elif action_type == "create_guide":
        layer_id, object_id, guide_id = make_id("layer"), make_id("obj"), make_id("guide")
        kind = data.get("kind", "path")
        geometry, style = data.get("geometry"), data.get("style", {})
        if kind not in {"shape", "path", "paint-stroke"} or not isinstance(geometry, Mapping) or not isinstance(style, Mapping):
            raise EditorActionError("INVALID_GUIDE", "guide requires supported editable geometry and style")
        _validate_geometry(document, kind, geometry, style)
        layer = LayerRecord(layer_id, str(data.get("name", "Guide"))[:120], "guide", role=data.get("semanticRole"),
                            metadata={"guideId": guide_id})
        obj = ObjectRecord(object_id, layer_id, kind, "document", dict(geometry), AffineTransform.identity(),
                           style=dict(style), revision=document.current_revision + 1,
                           lineage={"guideId": guide_id})
        layer.object_ids.append(object_id)
        _add_layer(document, layer)
        document.objects[object_id] = obj
        guide = GuideRecord(guide_id, layer.name, "proposed", document.current_revision + 1,
                            document.current_revision + 1, semantic_role=data.get("semanticRole"), object_ids=[object_id])
        document.guides[guide_id] = guide
        affected.extend([layer_id, object_id, guide_id])
        created.extend([layer_id, object_id, guide_id])
        undoable = False
    elif action_type == "transition_guide":
        guide_id = data.get("guideId")
        guide = document.guides.get(guide_id)
        if guide is None:
            raise EditorActionError("UNKNOWN_GUIDE", "guide does not exist")
        def require_guide_unlocked(record: GuideRecord) -> None:
            for object_id in record.object_ids:
                obj = document.objects.get(object_id)
                if obj is not None:
                    _require_layer(document, obj.layer_id)
        require_guide_unlocked(guide)
        lifecycle = data.get("lifecycle")
        replacement_id = data.get("replacementId")
        if lifecycle == "replacement-pending":
            replacement = document.guides.get(replacement_id)
            if replacement is None or replacement.guide_id == guide_id:
                raise EditorActionError("INVALID_GUIDE_REPLACEMENT", "replacement must be a different proposed guide")
            require_guide_unlocked(replacement)
            replacement.supersedes_id = guide_id
            affected.append(replacement_id)
        if lifecycle == "superseded":
            replacement_id = guide.replacement_id
            replacement = document.guides.get(replacement_id) if replacement_id else None
            if replacement is None or replacement.supersedes_id != guide_id:
                raise EditorActionError("INVALID_GUIDE_REPLACEMENT", "supersession requires reciprocal replacement lineage")
        if lifecycle == "active":
            if guide.lifecycle == "replacement-pending":
                old = next((item for item in document.guides.values() if item.replacement_id == guide_id), None)
                if old is not None:
                    require_guide_unlocked(old)
                    old.transition("superseded", document.current_revision + 1, replacement_id=guide_id)
                    affected.append(old.guide_id)
            guide.transition("active", document.current_revision + 1)
        else:
            guide.transition(lifecycle, document.current_revision + 1, replacement_id=replacement_id)
        affected.append(guide_id)
        undoable = False
    elif action_type == "undo":
        raise EditorActionError("UNDO_ACTION_REQUIRES_HISTORY_HANDLER", "undo must use the history endpoint")
    elif action_type == "redo":
        raise EditorActionError("REDO_ACTION_REQUIRES_HISTORY_HANDLER", "redo must use the history endpoint")
    else:
        raise EditorActionError("UNKNOWN_ACTION", "unsupported editor action")

    return list(dict.fromkeys(affected)), created, invalidated, undoable, history_kind, tx_metadata


def _validate_geometry(document: Document, kind: str, geometry: Mapping[str, Any], style: Mapping[str, Any]) -> None:
    _check_json({"geometry": geometry, "style": style})
    if kind == "shape":
        shape = geometry.get("shape")
        if shape not in {"line", "rectangle", "ellipse", "polygon"}:
            raise EditorActionError("INVALID_SHAPE", "shape must be line, rectangle, ellipse, or polygon")
        if shape == "polygon":
            points = geometry.get("points")
            if not isinstance(points, list) or len(points) < 3:
                raise EditorActionError("INVALID_SHAPE", "polygon requires at least three points")
        else:
            values = [geometry.get(field) for field in ("x", "y", "width", "height")]
            if any(type(value) not in (int, float) for value in values):
                raise EditorActionError("INVALID_SHAPE", "shape bounds must be numeric")
            if shape != "line" and (float(values[2]) <= 0 or float(values[3]) <= 0):
                raise EditorActionError("INVALID_SHAPE", "shape bounds must have positive dimensions")
    elif kind in {"path", "paint-stroke", "guide"}:
        points = geometry.get("points")
        if not isinstance(points, list) or not points:
            raise EditorActionError("INVALID_PATH", "path and stroke geometry require points")
        if kind == "path" and len(points) < 2:
            raise EditorActionError("INVALID_PATH", "editable path requires at least two points")
        if kind == "paint-stroke":
            width = style.get("width", 1)
            hardness = style.get("hardness", 1)
            opacity = style.get("opacity", 1)
            if type(width) not in (int, float) or not 1 <= float(width) <= 256:
                raise EditorActionError("INVALID_BRUSH_SIZE", "brush size must be between 1 and 256 document pixels")
            if type(hardness) not in (int, float) or not 0 <= float(hardness) <= 1:
                raise EditorActionError("INVALID_BRUSH_HARDNESS", "brush hardness must be between zero and one")
            if type(opacity) not in (int, float) or not 0 <= float(opacity) <= 1:
                raise EditorActionError("INVALID_BRUSH_OPACITY", "brush opacity must be between zero and one")
        if "closed" in geometry and type(geometry["closed"]) is not bool:
            raise EditorActionError("INVALID_PATH", "closed must be boolean")
    for point in geometry.get("points", []):
        if not isinstance(point, (list, tuple)) or len(point) != 2 or any(type(v) not in (int, float) for v in point):
            raise EditorActionError("INVALID_GEOMETRY", "points must be numeric x/y pairs")
        if abs(float(point[0])) > document.width * 4 or abs(float(point[1])) > document.height * 4:
            raise EditorActionError("GEOMETRY_OUT_OF_BOUNDS", "geometry is outside the supported document margin")
    for field in ("fill", "stroke", "color"):
        if field in style and style[field] is not None:
            _rgba_color(style[field])


def prepare_action(
    document: Document,
    asset_store: AssetStore,
    payload: Mapping[str, Any],
    *,
    actor_id: str,
    actor_kind: str | None = None,
    command_id: str | None = None,
    transaction_id: str | None = None,
    group_id: str | None = None,
) -> PreparedAction:
    """Validate and prepare an atomic revision; no files are written here."""

    if not isinstance(payload, Mapping):
        raise EditorActionError("INVALID_PAYLOAD", "request body must be an object")
    document_id = payload.get("documentId")
    if document_id != document.document_id:
        raise EditorActionError("UNKNOWN_DOCUMENT", "document does not exist")
    expected = payload.get("expectedRevision")
    if type(expected) is not int:
        raise EditorActionError("INVALID_REVISION", "expectedRevision must be an integer")
    if expected != document.current_revision:
        raise EditorActionError("STALE_DOCUMENT_REVISION", "document revision changed", current_revision=document.current_revision)
    actor_kind = actor_kind or payload.get("actorKind", "director")
    if actor_kind not in {"human", "director", "agent"}:
        raise EditorActionError("INVALID_ACTOR_KIND", "editor actor kind is unsupported")
    try:
        actor_id = validate_id(actor_id, field="actorId")
        if command_id is not None:
            validate_id(command_id, field="commandId")
    except (TypeError, ValueError) as exc:
        raise EditorActionError("INVALID_ACTOR", "editor actor identity is invalid") from exc
    transaction_id = transaction_id or payload.get("transactionId") or make_id("txn")
    group_id = group_id or payload.get("groupId") or make_id("grp")
    try:
        validate_id(transaction_id, field="transactionId")
        validate_id(group_id, field="groupId")
    except (TypeError, ValueError) as exc:
        raise EditorActionError("INVALID_TRANSACTION", "transaction identity is invalid") from exc
    action_type = payload.get("actionType")
    if action_type == "batch":
        actions = payload.get("actions")
        if not isinstance(actions, list) or not actions or len(actions) > 100:
            raise EditorActionError("INVALID_BATCH", "batch must contain 1 to 100 actions")
        action_items = actions
    else:
        action_items = [payload]
    candidate = deepcopy(document)
    pending_assets: list[PendingAsset] = []
    affected: list[str] = []
    created: list[str] = []
    invalidated: list[str] = []
    undoable = True
    kinds: list[str] = []
    action_names: list[str] = []
    for item in action_items:
        if not isinstance(item, Mapping):
            raise EditorActionError("INVALID_BATCH", "batch entries must be objects")
        item_affected, item_created, item_invalidated, item_undoable, history_kind, _ = _apply_action(
            candidate, asset_store, item, pending_assets
        )
        affected.extend(item_affected)
        created.extend(item_created)
        invalidated.extend(item_invalidated)
        undoable = undoable and item_undoable
        kinds.append(history_kind)
        action_names.append(str(item.get("actionType")))
    history_kind = kinds[0] if len(set(kinds)) == 1 else "edit"
    if len(action_items) == 1 and action_names[0] == "import_image":
        # The multipart bridge supplies bytes in an internal field; they are
        # removed from all durable metadata and transaction material.
        action_names[0] = "import_image"
    candidate, receipt = _transaction(
        document,
        candidate,
        action_type=action_type if action_type == "batch" else action_names[0],
        actor_id=actor_id,
        actor_kind=actor_kind,
        transaction_id=transaction_id,
        group_id=group_id,
        affected_ids=list(dict.fromkeys(affected)),
        created_ids=list(dict.fromkeys(created)),
        invalidated_ids=list(dict.fromkeys(invalidated)),
        undoable=undoable,
        history_kind=history_kind,
        extra_metadata={"batchActions": action_names} if action_type == "batch" else None,
        stamp_ids=_stamp_ids(action_items, affected),
        command_id=command_id,
    )
    return PreparedAction(candidate, receipt, pending_assets)


def _stamp_ids(actions: list[Mapping[str, Any]], affected: list[str]) -> list[str]:
    """Return record identities whose record revision changes in this batch."""

    identity_ids = list(dict.fromkeys(affected))
    preserve = {"create_selection", "refine_selection", "rebase_selection", "duplicate_selection_to_layer", "duplicate_layer",
                "create_context_mask", "set_relational_context"}
    non_stamped_sources: set[str] = set()
    for action in actions:
        kind = action.get("actionType")
        data = action.get("data", {})
        if kind in preserve and isinstance(data, Mapping):
            for field in ("sourceLayerId", "selectionId"):
                value = data.get(field)
                if isinstance(value, str):
                    non_stamped_sources.add(value)
            if kind == "duplicate_layer" and isinstance(data.get("layerId"), str):
                non_stamped_sources.add(data["layerId"])
    return [identity for identity in identity_ids if identity not in non_stamped_sources]


def prepare_undo_redo(
    document: Document,
    payload: Mapping[str, Any],
    *,
    actor_id: str,
    actor_kind: str | None = None,
    command_id: str | None = None,
    transaction_id: str | None = None,
    group_id: str | None = None,
    redo: bool = False,
) -> PreparedAction:
    expected = payload.get("expectedRevision")
    if type(expected) is not int:
        raise EditorActionError("INVALID_REVISION", "expectedRevision must be an integer")
    if expected != document.current_revision:
        raise EditorActionError("STALE_DOCUMENT_REVISION", "document revision changed", current_revision=document.current_revision)
    if payload.get("documentId") != document.document_id:
        raise EditorActionError("UNKNOWN_DOCUMENT", "document does not exist")
    candidate = deepcopy(document)
    undo_stack = list(candidate.metadata.get("editorUndoStack", []))
    redo_stack = list(candidate.metadata.get("editorRedoStack", []))
    stack = redo_stack if redo else undo_stack
    if not stack:
        raise EditorActionError("NOTHING_TO_REDO" if redo else "NOTHING_TO_UNDO", "no compatible edit is available")
    target_id = stack[-1]
    original = next((TransactionRecord.from_dict(item) for item in candidate.history if item.get("transactionId") == target_id), None)
    if original is None or original.metadata.get("undoable") is not True:
        raise EditorActionError("HISTORY_ENTRY_UNAVAILABLE", "history entry cannot be replayed")
    snapshot = original.after if redo else original.before
    if not isinstance(snapshot, Mapping):
        raise EditorActionError("HISTORY_ENTRY_UNAVAILABLE", "history entry has no retained scene snapshot")
    previous_snapshot = _snapshot(candidate)
    candidate = _restore_snapshot(candidate, snapshot)
    undo_stack = list(document.metadata.get("editorUndoStack", []))
    redo_stack = list(document.metadata.get("editorRedoStack", []))
    if redo:
        redo_stack.pop()
        undo_stack.append(target_id)
    else:
        undo_stack.pop()
        redo_stack.append(target_id)
    candidate.metadata["editorUndoStack"] = undo_stack
    candidate.metadata["editorRedoStack"] = redo_stack
    action_type = "redo" if redo else "undo"
    transaction_id = transaction_id or payload.get("transactionId") or make_id("txn")
    group_id = group_id or payload.get("groupId") or make_id("grp")
    previous_revision = document.current_revision
    candidate.current_revision = previous_revision
    candidate.revision_digest = None
    candidate, receipt = _transaction(
        document,
        candidate,
        action_type=action_type,
        actor_id=actor_id,
        actor_kind=actor_kind or payload.get("actorKind", "director"),
        transaction_id=transaction_id,
        group_id=group_id,
        affected_ids=list(original.affected_ids),
        created_ids=[],
        invalidated_ids=[],
        undoable=False,
        history_kind=action_type,
        extra_metadata={"replayOf": target_id, "undoSnapshot": previous_snapshot},
        command_id=command_id,
    )
    return PreparedAction(candidate, receipt)


def prepare_raster_import(
    document: Document,
    asset_store: AssetStore,
    payload: Mapping[str, Any],
    image_bytes_value: bytes,
    *,
    actor_id: str,
    actor_kind: str | None = None,
    command_id: str | None = None,
    filename: str,
) -> PreparedAction:
    body = dict(payload)
    body["actionType"] = "import_image"
    data = dict(body.get("data", {}))
    data["filename"] = filename
    body["data"] = data
    body["fileBytes"] = image_bytes_value
    return prepare_action(document, asset_store, body, actor_id=actor_id, actor_kind=actor_kind,
                          command_id=command_id)


def publish_pending_assets(asset_store: AssetStore, prepared: PreparedAction) -> None:
    """Publish immutable bytes after all action validation has succeeded."""

    if not prepared.pending_assets:
        return
    try:
        stored = asset_store.put_batch([
            (pending.content, pending.record) for pending in prepared.pending_assets
        ])
    except Exception as exc:
        raise EditorActionError(
            "ASSET_PUBLICATION_FAILED",
            "one or more immutable assets could not be published; no document changes were committed",
        ) from exc
    # AssetStore.put_batch validates each digest, canonical URI, size, and the
    # published target before returning. There is deliberately no fallible
    # post-publication check here: a refusal after successful publication
    # would leave new immutable blobs without an adopted document.


def actor_for_session(username: str | None) -> str:
    """Create a schema-safe actor ID without exposing a username in receipts."""

    if username:
        suffix = sha256_bytes(username.encode("utf-8"))[:16]
        return f"director-{suffix}"
    return "director-local"
