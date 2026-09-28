"""Renderer-neutral projections and explicit editor coordinate mappings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema import Document


@dataclass(frozen=True)
class ViewportMapping:
    """Map document coordinates to CSS-pixel preview coordinates.

    Pointer events use CSS pixels. Device-pixel ratio affects canvas backing
    stores only, so it is intentionally not part of the document transform.
    ``physical_to_document`` is provided for probes that start with backing
    store coordinates.
    """

    document_width: int
    document_height: int
    viewport_width: float
    viewport_height: float
    zoom: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0

    def __post_init__(self) -> None:
        if min(self.document_width, self.document_height) <= 0:
            raise ValueError("document dimensions must be positive")
        if self.viewport_width <= 0 or self.viewport_height <= 0:
            raise ValueError("viewport dimensions must be positive")
        if not 0.1 <= self.zoom <= 8.0:
            raise ValueError("zoom must be between 0.1 and 8")

    @property
    def fit_scale(self) -> float:
        return min(self.viewport_width / self.document_width, self.viewport_height / self.document_height)

    @property
    def scale(self) -> float:
        return self.fit_scale * self.zoom

    @property
    def fit_offset(self) -> tuple[float, float]:
        scale = self.fit_scale
        return (
            (self.viewport_width - self.document_width * scale) / 2.0,
            (self.viewport_height - self.document_height * scale) / 2.0,
        )

    def document_to_preview(self, x: float, y: float) -> tuple[float, float]:
        ox, oy = self.fit_offset
        return ox + self.pan_x + x * self.scale, oy + self.pan_y + y * self.scale

    def preview_to_document(self, x: float, y: float) -> tuple[float, float]:
        ox, oy = self.fit_offset
        return ((x - ox - self.pan_x) / self.scale, (y - oy - self.pan_y) / self.scale)

    def physical_to_document(self, x: float, y: float, device_pixel_ratio: float) -> tuple[float, float]:
        if device_pixel_ratio <= 0:
            raise ValueError("device pixel ratio must be positive")
        return self.preview_to_document(x / device_pixel_ratio, y / device_pixel_ratio)

    def matrix(self) -> list[float]:
        ox, oy = self.fit_offset
        return [self.scale, 0.0, ox + self.pan_x, 0.0, self.scale, oy + self.pan_y, 0.0, 0.0, 1.0]


def document_to_view(document: Document) -> dict[str, Any]:
    """Return a JSON-safe allowlisted view; never expose storage paths."""

    document.validate()
    layers = []
    for layer in document.layers.values():
        ancestor = document.layers.get(layer.parent_id) if layer.parent_id else None
        locked_by_ancestor = False
        while ancestor is not None:
            locked_by_ancestor = locked_by_ancestor or ancestor.locked
            ancestor = document.layers.get(ancestor.parent_id) if ancestor.parent_id else None
        layers.append({
            "id": layer.layer_id,
            "name": layer.name,
            "kind": layer.kind,
            "parentId": layer.parent_id,
            "childIds": list(layer.child_ids),
            "objectIds": list(layer.object_ids),
            "assetIds": list(layer.asset_ids),
            "visible": layer.visible,
            "locked": layer.locked,
            "lockedByAncestor": locked_by_ancestor,
            "effectiveLocked": layer.locked or locked_by_ancestor,
            "opacity": layer.opacity,
            "blendMode": layer.blend_mode,
            "clippingRefs": list(layer.clipping_refs),
            "transform": layer.transform.to_dict(),
            "role": layer.role,
            "revision": layer.revision,
            "depthElement": layer.depth_element,
            "completePlateId": layer.complete_plate_id,
            "deleted": bool(layer.metadata.get("deleted", False)),
            "workStatus": layer.collaboration.work_status,
            "stale": any(
                selection.source_id == layer.layer_id and selection.state != "current"
                for selection in document.selections.values()
            ),
        })
    objects = []
    for obj in document.objects.values():
        objects.append({
            "id": obj.object_id,
            "layerId": obj.layer_id,
            "kind": obj.kind,
            "coordinateSpace": obj.coordinate_space,
            "geometry": obj.geometry,
            "transform": obj.transform.to_dict(),
            "assetId": obj.asset_id,
            "style": obj.style,
            "revision": obj.revision,
        })
    assets = []
    for asset in document.assets.values():
        assets.append({
            "id": asset.asset_id,
            "contentHash": asset.content_hash,
            "mediaType": asset.media_type,
            "extension": asset.extension,
            "byteLength": asset.byte_length,
            "width": asset.width,
            "height": asset.height,
            "hasAlpha": asset.has_alpha,
            "external": asset.external_uri is not None,
            "externalStatus": asset.external_status,
            "thumbnailUrl": f"/creative_document_api/documents/{document.document_id}/assets/{asset.asset_id}/thumbnail",
            "url": f"/creative_document_api/documents/{document.document_id}/assets/{asset.asset_id}/content",
        })
    masks = [
        {
            "id": mask.mask_id,
            "purpose": mask.purpose,
            "coordinateSpace": mask.coordinate_space,
            "revision": mask.revision,
            "assetId": mask.asset_id,
            "ownerId": mask.owner_id,
            "lineage": mask.lineage,
            "contentHash": mask.content_hash,
        }
        for mask in document.masks.values()
    ]
    selections = [
        {
            "id": selection.selection_id,
            "sourceId": selection.source_id,
            "sourceRevision": selection.source_revision,
            "sourceContentDigest": selection.source_content_digest,
            "selectionRevision": selection.selection_revision,
            "state": selection.state,
            "maskId": selection.mask_id,
            "bounds": None if selection.bounds is None else selection.bounds.to_dict(),
            "seedGeometry": selection.seed_geometry,
            "semanticHint": selection.semantic_hint,
            "derivedLayerIds": list(selection.derived_layer_ids),
            "refinementHistory": selection.refinement_history,
            "staleReason": selection.stale_reason,
            "invalidationHistory": selection.invalidation_history,
            "rebaseHistory": selection.rebase_history,
            "duplicateHistory": selection.duplicate_history,
        }
        for selection in document.selections.values()
    ]
    result = {
        "documentId": document.document_id,
        "revision": document.current_revision,
        "revisionDigest": document.revision_digest,
        "width": document.width,
        "height": document.height,
        "colorSpaceIntent": document.color_space_intent,
        "rootLayerIds": list(document.root_layer_ids),
        "layers": layers,
        "objects": objects,
        "assets": assets,
        "masks": masks,
        "selections": selections,
        "guides": [guide.to_dict() for guide in document.guides.values()],
        "interactionGroups": [group.to_dict() for group in document.interaction_groups.values()],
        "variants": [variant.to_dict() for variant in document.variants.values()],
        "dirty": document.current_revision != int(document.metadata.get("lastExplicitSaveRevision", -1)),
        "checkpointNotice": bool(document.checkpoint_refs),
        "recoveryNotice": document.metadata.get("recoveryNotice"),
    }
    issues = _renderer_issues(document)
    result["rendererIssues"] = issues
    for layer in result["layers"]:
        layer["rendererIssues"] = [issue for issue in issues if issue.get("layerId") == layer["id"]]
    for obj in result["objects"]:
        obj["rendererIssues"] = [issue for issue in issues if issue.get("objectId") == obj["id"]]
    return result


def _renderer_issues(document: Document) -> list[dict[str, Any]]:
    """Project browser fidelity limitations as typed, path-free view data."""

    issues: list[dict[str, Any]] = []
    supported_layers = {
        "raster", "paint", "vector", "shape", "guide", "candidate", "patch",
        "proxy", "source-composite", "depth", "object", "group",
    }
    supported_objects = {"raster-placement", "paint-stroke", "path", "guide", "shape"}
    supported_shapes = {"line", "ellipse", "rectangle", "polygon"}
    effective_visibility: dict[str, bool] = {}
    def is_effectively_visible(layer_id: str) -> bool:
        if layer_id in effective_visibility:
            return effective_visibility[layer_id]
        layer = document.layers[layer_id]
        parent_visible = is_effectively_visible(layer.parent_id) if layer.parent_id else True
        result = parent_visible and layer.visible and not layer.metadata.get("deleted", False)
        effective_visibility[layer_id] = result
        return result
    for layer_id in document.layers:
        is_effectively_visible(layer_id)
    for layer in document.layers.values():
        if not effective_visibility[layer.layer_id]:
            continue
        if layer.kind not in supported_layers:
            issues.append({"code": "UNSUPPORTED_LAYER_KIND", "severity": "error", "layerId": layer.layer_id,
                           "message": f"Visible layer kind '{layer.kind}' is not rendered in the editor."})
        if layer.blend_mode != "normal":
            issues.append({"code": "UNSUPPORTED_LAYER_BLEND_MODE", "severity": "error", "layerId": layer.layer_id,
                           "message": f"Blend mode '{layer.blend_mode}' is not rendered in the editor."})
        if layer.clipping_refs:
            issues.append({"code": "UNSUPPORTED_LAYER_CLIPPING", "severity": "error", "layerId": layer.layer_id,
                           "message": "This layer has clipping references that the editor cannot render."})
        for object_id in layer.object_ids:
            obj = document.objects[object_id]
            if obj.coordinate_space != "document":
                issues.append({"code": "UNSUPPORTED_OBJECT_COORDINATE_SPACE", "severity": "error", "layerId": layer.layer_id,
                               "objectId": obj.object_id, "message": f"Object coordinate space '{obj.coordinate_space}' is not rendered in the editor."})
            if obj.kind not in supported_objects:
                issues.append({"code": "UNSUPPORTED_OBJECT_KIND", "severity": "error", "layerId": layer.layer_id,
                               "objectId": obj.object_id, "message": f"Object kind '{obj.kind}' is not rendered in the editor."})
            elif obj.kind == "shape" and obj.geometry.get("shape") not in supported_shapes:
                issues.append({"code": "UNSUPPORTED_OBJECT_SHAPE", "severity": "error", "layerId": layer.layer_id,
                               "objectId": obj.object_id, "message": f"Shape '{obj.geometry.get('shape')}' is not rendered in the editor."})
            if obj.kind == "paint-stroke" and float(obj.style.get("hardness", 1.0)) < 1.0:
                issues.append({"code": "PAINT_HARDNESS_PREVIEW_APPROXIMATION", "severity": "warning",
                               "layerId": layer.layer_id, "objectId": obj.object_id,
                               "message": "Soft brush edges use an approximate canvas halo; preview and export edges may differ."})
            if obj.kind == "raster-placement":
                if obj.asset_id is None or obj.asset_id not in document.assets:
                    issues.append({"code": "MISSING_RASTER_ASSET", "severity": "error", "layerId": layer.layer_id,
                                   "objectId": obj.object_id, "message": "Raster placement has no embedded image asset."})
                elif document.assets[obj.asset_id].external_uri is not None and document.assets[obj.asset_id].external_status != "embedded":
                    issues.append({"code": "EXTERNAL_ASSET_UNAVAILABLE", "severity": "error", "layerId": layer.layer_id,
                                   "objectId": obj.object_id, "assetId": obj.asset_id,
                                   "message": "This external image must be embedded before editor rendering."})
    return issues
