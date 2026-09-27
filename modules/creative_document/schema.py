"""Versioned, renderer-independent records for an editable Nexfocus scene.

The schema deliberately stores meaning and provenance, not Konva/Gradio
objects.  Binary pixels are represented by :class:`AssetRecord` references;
they are owned by ``asset_store.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, Mapping, Sequence

from .ids import content_digest, validate_id, validate_sha256
from .transforms import AffineTransform, BBox, CoordinateTransform
from .tree import TreeValidationError, validate_layer_tree


FORMAT_IDENTIFIER = "nexfocus.nexscene"
FORMAT_ID = FORMAT_IDENTIFIER
SCHEMA_VERSION = 1
CURRENT_SCHEMA_VERSION = SCHEMA_VERSION


class SchemaValidationError(ValueError):
    """Raised when a durable record would have ambiguous meaning."""


class GuideLifecycle(StrEnum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    CONSUMED = "consumed"
    REPLACEMENT_PENDING = "replacement-pending"
    SUPERSEDED = "superseded"
    SAFE_TO_REMOVE = "safe-to-remove"


class SelectionState(StrEnum):
    CURRENT = "current"
    STALE_SOURCE = "stale-source"
    NEEDS_REBASE = "needs-rebase"


def _strict(data: Mapping[str, Any], allowed: set[str], required: set[str], name: str) -> None:
    keys = set(data)
    missing = required - keys
    unknown = keys - allowed
    if missing:
        raise SchemaValidationError(f"{name} missing fields: {sorted(missing)}")
    if unknown:
        raise SchemaValidationError(f"{name} has unknown fields: {sorted(unknown)}")


def _id(value: Any, field_name: str) -> str:
    try:
        return validate_id(str(value), field=field_name)
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(str(exc)) from exc


def _digest(value: Any, field_name: str = "contentHash") -> str:
    try:
        return validate_sha256(str(value), field=field_name)
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(str(exc)) from exc


def _transform(value: Any, field_name: str = "transform") -> AffineTransform:
    try:
        return value if isinstance(value, AffineTransform) else AffineTransform.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(f"invalid {field_name}: {exc}") from exc


def _coordinate_transform(value: Any) -> CoordinateTransform:
    try:
        return value if isinstance(value, CoordinateTransform) else CoordinateTransform.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(f"invalid coordinate transform: {exc}") from exc


def _bbox(value: Any) -> BBox:
    try:
        return value if isinstance(value, BBox) else BBox.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(f"invalid bbox: {exc}") from exc


def _dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SchemaValidationError("metadata must be an object")
    return dict(value)


def _ids(values: Sequence[Any], name: str) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise SchemaValidationError(f"{name} must be an array")
    result = [_id(value, name) for value in values]
    if len(set(result)) != len(result):
        raise SchemaValidationError(f"{name} contains duplicate IDs")
    return result


@dataclass
class VisualProfile:
    palette: dict[str, Any] = field(default_factory=dict)
    lighting: dict[str, Any] = field(default_factory=dict)
    style: dict[str, Any] = field(default_factory=dict)
    prompts: list[str] = field(default_factory=list)
    model_choices: dict[str, Any] = field(default_factory=dict)
    inference_settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"palette": self.palette, "lighting": self.lighting, "style": self.style,
                "prompts": list(self.prompts), "modelChoices": self.model_choices,
                "inferenceSettings": self.inference_settings}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VisualProfile":
        _strict(value, {"palette", "lighting", "style", "prompts", "modelChoices", "inferenceSettings"}, {"palette", "lighting", "style", "prompts", "modelChoices", "inferenceSettings"}, "visualProfile")
        return cls(dict(value["palette"]), dict(value["lighting"]), dict(value["style"]),
                   [str(v) for v in value["prompts"]], dict(value["modelChoices"]), dict(value["inferenceSettings"]))


@dataclass
class HandoffNote:
    note_id: str
    actor_id: str
    actor_kind: str
    body: str
    created_at_revision: int
    transaction_id: str
    target_ids: list[str]
    target_revisions: dict[str, int]
    visibility: str = "shared"
    state: str = "open"
    orphan_reason: str | None = None

    def validate(self) -> None:
        _id(self.note_id, "noteId")
        _id(self.transaction_id, "transactionId")
        if self.actor_kind not in {"human", "director", "agent", "system"}:
            raise SchemaValidationError("invalid handoff actor kind")
        if self.visibility not in {"shared", "director-only"}:
            raise SchemaValidationError("invalid handoff visibility")
        if self.state not in {"open", "acknowledged", "resolved", "orphaned"}:
            raise SchemaValidationError("invalid handoff state")
        if self.created_at_revision < 0 or not self.body:
            raise SchemaValidationError("handoff body and revision are required")
        if set(self.target_revisions) != set(self.target_ids):
            raise SchemaValidationError("handoff target revision binding is incomplete")
        if any(int(revision) < 0 for revision in self.target_revisions.values()):
            raise SchemaValidationError("handoff target revisions must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"noteId": self.note_id, "actorId": self.actor_id, "actorKind": self.actor_kind,
                "body": self.body, "createdAtRevision": self.created_at_revision,
                "transactionId": self.transaction_id, "targetIds": list(self.target_ids),
                "targetRevisions": dict(self.target_revisions), "visibility": self.visibility,
                "state": self.state, "orphanReason": self.orphan_reason}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HandoffNote":
        allowed = {"noteId", "actorId", "actorKind", "body", "createdAtRevision", "transactionId",
                   "targetIds", "targetRevisions", "visibility", "state", "orphanReason"}
        _strict(value, allowed, allowed, "handoffNote")
        note = cls(_id(value["noteId"], "noteId"), str(value["actorId"]), str(value["actorKind"]),
                   str(value["body"]), int(value["createdAtRevision"]), _id(value["transactionId"], "transactionId"),
                   _ids(value["targetIds"], "targetIds"), {str(k): int(v) for k, v in value["targetRevisions"].items()},
                   str(value["visibility"]), str(value["state"]), value["orphanReason"])
        note.validate()
        return note


@dataclass
class CollaborationState:
    work_status: str = "untouched"
    last_editor: dict[str, Any] | None = None
    handoff_notes: list[HandoffNote] = field(default_factory=list)

    WORK_STATUSES: ClassVar[set[str]] = {
        "untouched", "agent-draft", "agent-review", "director-review", "director-edited",
        "locked-for-review", "ready",
    }

    def validate(self) -> None:
        if self.work_status not in self.WORK_STATUSES:
            raise SchemaValidationError(f"invalid collaboration workStatus: {self.work_status}")
        if self.last_editor is not None:
            if set(self.last_editor) != {"actorId", "actorKind", "atRevision"}:
                raise SchemaValidationError("lastEditor has unknown or missing fields")
            if self.last_editor["actorKind"] not in {"human", "director", "agent", "system"}:
                raise SchemaValidationError("invalid lastEditor actorKind")
            if int(self.last_editor["atRevision"]) < 0:
                raise SchemaValidationError("lastEditor revision must be non-negative")
        for note in self.handoff_notes:
            note.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"workStatus": self.work_status, "lastEditor": self.last_editor,
                "handoffNotes": [note.to_dict() for note in self.handoff_notes]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CollaborationState":
        _strict(value, {"workStatus", "lastEditor", "handoffNotes"}, {"workStatus", "lastEditor", "handoffNotes"}, "collaboration")
        result = cls(str(value["workStatus"]), None if value["lastEditor"] is None else dict(value["lastEditor"]),
                     [HandoffNote.from_dict(v) for v in value["handoffNotes"]])
        result.validate()
        return result


@dataclass
class AssetRecord:
    asset_id: str
    content_hash: str
    media_type: str
    storage_uri: str
    extension: str = "bin"
    byte_length: int | None = None
    width: int | None = None
    height: int | None = None
    has_alpha: bool | None = None
    color_space: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    external_uri: str | None = None
    expected_hash: str | None = None
    external_status: str | None = None

    def validate(self) -> None:
        _id(self.asset_id, "assetId")
        _digest(self.content_hash)
        if not self.media_type or not self.storage_uri:
            raise SchemaValidationError("asset media type and storage URI are required")
        if self.byte_length is not None and self.byte_length < 0:
            raise SchemaValidationError("asset byte length cannot be negative")
        for value in (self.width, self.height):
            if value is not None and value <= 0:
                raise SchemaValidationError("asset dimensions must be positive")
        if self.external_uri is not None:
            if self.expected_hash is None:
                raise SchemaValidationError("external asset requires expected hash")
            _digest(self.expected_hash, "expectedHash")
            if self.content_hash != self.expected_hash:
                raise SchemaValidationError("external asset content and expected hashes must agree")
            if self.external_status not in {"missing", "available", "relinked", "embedded"}:
                raise SchemaValidationError("invalid external asset status")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"assetId": self.asset_id, "contentHash": self.content_hash, "mediaType": self.media_type,
                "storageUri": self.storage_uri, "extension": self.extension, "byteLength": self.byte_length,
                "width": self.width, "height": self.height, "hasAlpha": self.has_alpha,
                "colorSpace": self.color_space, "provenance": self.provenance, "externalUri": self.external_uri,
                "expectedHash": self.expected_hash, "externalStatus": self.external_status}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AssetRecord":
        allowed = {"assetId", "contentHash", "mediaType", "storageUri", "extension", "byteLength", "width", "height",
                   "hasAlpha", "colorSpace", "provenance", "externalUri", "expectedHash", "externalStatus"}
        _strict(value, allowed, allowed, "asset")
        record = cls(_id(value["assetId"], "assetId"), _digest(value["contentHash"]), str(value["mediaType"]),
                     str(value["storageUri"]), str(value["extension"]), value["byteLength"], value["width"], value["height"],
                     value["hasAlpha"], value["colorSpace"], _dict(value["provenance"]), value["externalUri"],
                     value["expectedHash"], value["externalStatus"])
        record.validate()
        return record


@dataclass
class LayerRecord:
    layer_id: str
    name: str
    kind: str
    parent_id: str | None = None
    child_ids: list[str] = field(default_factory=list)
    visible: bool = True
    locked: bool = False
    opacity: float = 1.0
    blend_mode: str = "normal"
    transform: AffineTransform = field(default_factory=AffineTransform.identity)
    object_ids: list[str] = field(default_factory=list)
    asset_ids: list[str] = field(default_factory=list)
    mask_ids: list[str] = field(default_factory=list)
    role: str | None = None
    clipping_refs: list[str] = field(default_factory=list)
    collaboration: CollaborationState = field(default_factory=CollaborationState)
    revision: int = 0
    depth_element: bool = False
    complete_plate_id: str | None = None
    registration: CoordinateTransform | None = None
    lineage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    KINDS: ClassVar[set[str]] = {
        "group", "raster", "vector", "shape", "paint", "guide", "mask", "text", "candidate",
        "patch", "proxy", "source-composite", "interaction", "external", "depth", "object",
    }

    def validate(self) -> None:
        _id(self.layer_id, "layerId")
        if not self.name or self.kind not in self.KINDS:
            raise SchemaValidationError("layer name and kind are required")
        if self.parent_id is not None:
            _id(self.parent_id, "parentId")
        _ids(self.child_ids, "childIds")
        _ids(self.object_ids, "objectIds")
        _ids(self.asset_ids, "assetIds")
        _ids(self.mask_ids, "maskIds")
        _ids(self.clipping_refs, "clippingRefs")
        if not 0.0 <= float(self.opacity) <= 1.0:
            raise SchemaValidationError("layer opacity must be between 0 and 1")
        if self.revision < 0:
            raise SchemaValidationError("layer revision cannot be negative")
        self.collaboration.validate()
        if self.registration is not None:
            _coordinate_transform(self.registration)
        if self.depth_element and not self.complete_plate_id:
            raise SchemaValidationError("depth element must identify its complete underlying plate")
        if self.complete_plate_id is not None:
            _id(self.complete_plate_id, "completePlateId")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"layerId": self.layer_id, "name": self.name, "kind": self.kind, "parentId": self.parent_id,
                "childIds": list(self.child_ids), "visible": self.visible, "locked": self.locked,
                "opacity": self.opacity, "blendMode": self.blend_mode, "transform": self.transform.to_dict(),
                "objectIds": list(self.object_ids), "assetIds": list(self.asset_ids), "maskIds": list(self.mask_ids),
                "role": self.role, "clippingRefs": list(self.clipping_refs), "collaboration": self.collaboration.to_dict(),
                "revision": self.revision, "depthElement": self.depth_element, "completePlateId": self.complete_plate_id,
                "registration": None if self.registration is None else self.registration.to_dict(),
                "lineage": self.lineage, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LayerRecord":
        allowed = {"layerId", "name", "kind", "parentId", "childIds", "visible", "locked", "opacity", "blendMode",
                   "transform", "objectIds", "assetIds", "maskIds", "role", "clippingRefs", "collaboration", "revision",
                   "depthElement", "completePlateId", "registration", "lineage", "metadata"}
        _strict(value, allowed, allowed, "layer")
        result = cls(_id(value["layerId"], "layerId"), str(value["name"]), str(value["kind"]),
                     None if value["parentId"] is None else _id(value["parentId"], "parentId"), _ids(value["childIds"], "childIds"),
                     bool(value["visible"]), bool(value["locked"]), float(value["opacity"]), str(value["blendMode"]),
                     _transform(value["transform"]), _ids(value["objectIds"], "objectIds"), _ids(value["assetIds"], "assetIds"),
                     _ids(value["maskIds"], "maskIds"), value["role"], _ids(value["clippingRefs"], "clippingRefs"),
                     CollaborationState.from_dict(value["collaboration"]), int(value["revision"]), bool(value["depthElement"]),
                     None if value["completePlateId"] is None else _id(value["completePlateId"], "completePlateId"),
                     None if value["registration"] is None else _coordinate_transform(value["registration"]),
                     _dict(value["lineage"]), _dict(value["metadata"]))
        result.validate()
        return result


@dataclass
class ObjectRecord:
    object_id: str
    layer_id: str
    kind: str
    coordinate_space: str = "document"
    geometry: dict[str, Any] = field(default_factory=dict)
    transform: AffineTransform = field(default_factory=AffineTransform.identity)
    asset_id: str | None = None
    style: dict[str, Any] = field(default_factory=dict)
    revision: int = 0
    lineage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    KINDS: ClassVar[set[str]] = {"path", "shape", "paint-stroke", "text-placement", "raster-placement", "patch", "guide", "mask"}

    def validate(self) -> None:
        _id(self.object_id, "objectId")
        _id(self.layer_id, "layerId")
        if self.kind not in self.KINDS:
            raise SchemaValidationError("invalid object kind")
        if self.coordinate_space not in {"document", "source-crop", "native-bb", "model", "preview"}:
            raise SchemaValidationError("invalid object coordinate space")
        if self.asset_id is not None:
            _id(self.asset_id, "assetId")
        if self.revision < 0:
            raise SchemaValidationError("object revision cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"objectId": self.object_id, "layerId": self.layer_id, "kind": self.kind,
                "coordinateSpace": self.coordinate_space, "geometry": self.geometry,
                "transform": self.transform.to_dict(), "assetId": self.asset_id, "style": self.style,
                "revision": self.revision, "lineage": self.lineage, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ObjectRecord":
        allowed = {"objectId", "layerId", "kind", "coordinateSpace", "geometry", "transform", "assetId", "style", "revision", "lineage", "metadata"}
        _strict(value, allowed, allowed, "object")
        result = cls(_id(value["objectId"], "objectId"), _id(value["layerId"], "layerId"), str(value["kind"]),
                     str(value["coordinateSpace"]), _dict(value["geometry"]), _transform(value["transform"]),
                     None if value["assetId"] is None else _id(value["assetId"], "assetId"), _dict(value["style"]),
                     int(value["revision"]), _dict(value["lineage"]), _dict(value["metadata"]))
        result.validate()
        return result


@dataclass
class MaskRecord:
    mask_id: str
    purpose: str
    coordinate_space: str
    revision: int
    asset_id: str | None = None
    owner_id: str | None = None
    editable_source: dict[str, Any] | None = None
    lineage: dict[str, Any] = field(default_factory=dict)
    content_hash: str | None = None

    PURPOSES: ClassVar[set[str]] = {"layer-alpha", "visibility", "protection", "editing-selection", "generation", "context", "extraction", "coverage", "blend"}

    def validate(self) -> None:
        _id(self.mask_id, "maskId")
        if self.purpose not in self.PURPOSES:
            raise SchemaValidationError(f"invalid mask purpose: {self.purpose}")
        if self.coordinate_space not in {"document", "source-crop", "native-bb", "model", "preview"}:
            raise SchemaValidationError("invalid mask coordinate space")
        if self.revision < 0 or (self.asset_id is None and self.editable_source is None):
            raise SchemaValidationError("mask needs an asset or editable source")
        if self.asset_id is not None:
            _id(self.asset_id, "assetId")
        if self.owner_id is not None:
            _id(self.owner_id, "ownerId")
        if self.content_hash is not None:
            _digest(self.content_hash)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"maskId": self.mask_id, "purpose": self.purpose, "coordinateSpace": self.coordinate_space,
                "revision": self.revision, "assetId": self.asset_id, "ownerId": self.owner_id,
                "editableSource": self.editable_source, "lineage": self.lineage, "contentHash": self.content_hash}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MaskRecord":
        allowed = {"maskId", "purpose", "coordinateSpace", "revision", "assetId", "ownerId", "editableSource", "lineage", "contentHash"}
        _strict(value, allowed, allowed, "mask")
        result = cls(_id(value["maskId"], "maskId"), str(value["purpose"]), str(value["coordinateSpace"]), int(value["revision"]),
                     None if value["assetId"] is None else _id(value["assetId"], "assetId"),
                     None if value["ownerId"] is None else _id(value["ownerId"], "ownerId"),
                     None if value["editableSource"] is None else dict(value["editableSource"]), _dict(value["lineage"]), value["contentHash"])
        result.validate()
        return result


@dataclass
class SelectionRecord:
    selection_id: str
    source_id: str
    source_revision: int
    source_content_digest: str
    selection_revision: int
    state: str
    mask_id: str
    coordinate_space: str = "document"
    seed_geometry: list[dict[str, Any]] = field(default_factory=list)
    semantic_hint: str | None = None
    bounds: BBox | None = None
    mapping: CoordinateTransform | None = None
    refinement_history: list[dict[str, Any]] = field(default_factory=list)
    derived_layer_ids: list[str] = field(default_factory=list)
    stale_reason: str | None = None
    current_source_content_digest: str | None = None

    def validate(self) -> None:
        _id(self.selection_id, "selectionId")
        _id(self.source_id, "sourceId")
        _id(self.mask_id, "maskId")
        _ids(self.derived_layer_ids, "derivedLayerIds")
        _digest(self.source_content_digest, "sourceContentDigest")
        if self.source_revision < 0 or self.selection_revision < 1:
            raise SchemaValidationError("selection revisions are invalid")
        if self.state not in {state.value for state in SelectionState}:
            raise SchemaValidationError("invalid selection state")
        if self.bounds is not None:
            self.bounds.validate()
        if self.mapping is not None:
            _coordinate_transform(self.mapping)
        if self.state != SelectionState.CURRENT and not self.stale_reason:
            raise SchemaValidationError("stale selection requires a reason")
        if self.current_source_content_digest is not None:
            _digest(self.current_source_content_digest, "currentSourceContentDigest")

    def mark_source_changed(self, reason: str, source_revision: int, source_content_digest: str | None = None) -> None:
        """Mark a dependent selection stale without rewriting its mask lineage."""

        if reason == "SOURCE_TRANSFORM_CHANGED":
            self.state = SelectionState.NEEDS_REBASE.value
        elif reason in {"SOURCE_CONTENT_CHANGED", "SOURCE_CLIP_CHANGED", "SOURCE_REMOVED"}:
            self.state = SelectionState.STALE_SOURCE.value
        else:
            raise SchemaValidationError(f"unknown selection invalidation reason: {reason}")
        self.stale_reason = reason
        if source_content_digest is not None:
            self.current_source_content_digest = _digest(source_content_digest, "currentSourceContentDigest")

    def register_derived_layer(self, layer_id: str) -> int:
        """Register a duplicate-link mutation and advance selection revision."""

        self.validate()
        if self.state != SelectionState.CURRENT.value:
            raise SchemaValidationError("cannot duplicate a stale selection")
        layer_id = _id(layer_id, "derivedLayerId")
        if layer_id not in self.derived_layer_ids:
            self.derived_layer_ids.append(layer_id)
            self.selection_revision += 1
        return self.selection_revision

    def explicit_rebase(
        self,
        target_source_revision: int,
        *,
        new_mask_id: str | None = None,
        source_content_digest: str | None = None,
        geometry_only: bool = False,
    ) -> int:
        """Resolve a stale/rebased selection only through an explicit command."""

        self.validate()
        if target_source_revision < 0:
            raise SchemaValidationError("target source revision cannot be negative")
        if new_mask_id is not None:
            self.mask_id = _id(new_mask_id, "maskId")
        if source_content_digest is not None:
            self.source_content_digest = _digest(source_content_digest, "sourceContentDigest")
        self.source_revision = target_source_revision
        self.state = SelectionState.CURRENT.value
        self.stale_reason = None
        self.current_source_content_digest = None
        self.selection_revision += 1
        return self.selection_revision

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"selectionId": self.selection_id, "sourceId": self.source_id, "sourceRevision": self.source_revision,
                "sourceContentDigest": self.source_content_digest, "selectionRevision": self.selection_revision,
                "state": self.state, "maskId": self.mask_id, "coordinateSpace": self.coordinate_space,
                "seedGeometry": self.seed_geometry, "semanticHint": self.semantic_hint,
                "bounds": None if self.bounds is None else self.bounds.to_dict(),
                "mapping": None if self.mapping is None else self.mapping.to_dict(),
                "refinementHistory": self.refinement_history, "derivedLayerIds": list(self.derived_layer_ids),
                "staleReason": self.stale_reason, "currentSourceContentDigest": self.current_source_content_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SelectionRecord":
        allowed = {"selectionId", "sourceId", "sourceRevision", "sourceContentDigest", "selectionRevision", "state", "maskId", "coordinateSpace",
                   "seedGeometry", "semanticHint", "bounds", "mapping", "refinementHistory", "derivedLayerIds", "staleReason", "currentSourceContentDigest"}
        _strict(value, allowed, allowed, "selection")
        result = cls(_id(value["selectionId"], "selectionId"), _id(value["sourceId"], "sourceId"), int(value["sourceRevision"]),
                     _digest(value["sourceContentDigest"], "sourceContentDigest"), int(value["selectionRevision"]), str(value["state"]),
                     _id(value["maskId"], "maskId"), str(value["coordinateSpace"]), list(value["seedGeometry"]), value["semanticHint"],
                     None if value["bounds"] is None else _bbox(value["bounds"]),
                     None if value["mapping"] is None else _coordinate_transform(value["mapping"]), list(value["refinementHistory"]),
                     _ids(value["derivedLayerIds"], "derivedLayerIds"), value["staleReason"], value["currentSourceContentDigest"])
        result.validate()
        return result


@dataclass
class VariantSet:
    variant_set_id: str
    semantic_role: str
    member_ids: list[str]
    active_member_id: str | None = None
    shared_anchor: dict[str, Any] = field(default_factory=dict)
    placement_frame: CoordinateTransform | None = None
    lineage: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _id(self.variant_set_id, "variantSetId")
        _ids(self.member_ids, "memberIds")
        if not self.member_ids:
            raise SchemaValidationError("variant set needs a member")
        if self.active_member_id is not None and self.active_member_id not in self.member_ids:
            raise SchemaValidationError("active variant is not a member")
        if self.placement_frame is not None:
            _coordinate_transform(self.placement_frame)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"variantSetId": self.variant_set_id, "semanticRole": self.semantic_role, "memberIds": list(self.member_ids),
                "activeMemberId": self.active_member_id, "sharedAnchor": self.shared_anchor,
                "placementFrame": None if self.placement_frame is None else self.placement_frame.to_dict(), "lineage": self.lineage}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VariantSet":
        allowed = {"variantSetId", "semanticRole", "memberIds", "activeMemberId", "sharedAnchor", "placementFrame", "lineage"}
        _strict(value, allowed, allowed, "variantSet")
        result = cls(_id(value["variantSetId"], "variantSetId"), str(value["semanticRole"]), _ids(value["memberIds"], "memberIds"),
                     value["activeMemberId"], _dict(value["sharedAnchor"]), None if value["placementFrame"] is None else _coordinate_transform(value["placementFrame"]), _dict(value["lineage"]))
        result.validate()
        return result


@dataclass
class InteractionGroup:
    group_id: str
    member_layer_ids: list[str]
    registration_frame: CoordinateTransform
    relation_order: list[dict[str, Any]]
    parent_candidate_id: str | None = None
    overlap_notes: str | None = None

    def validate(self) -> None:
        _id(self.group_id, "groupId")
        _ids(self.member_layer_ids, "memberLayerIds")
        _coordinate_transform(self.registration_frame)
        if self.parent_candidate_id is not None:
            _id(self.parent_candidate_id, "parentCandidateId")
        if not self.relation_order:
            raise SchemaValidationError("interaction group needs explicit relation order")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"groupId": self.group_id, "memberLayerIds": list(self.member_layer_ids), "registrationFrame": self.registration_frame.to_dict(),
                "relationOrder": self.relation_order, "parentCandidateId": self.parent_candidate_id, "overlapNotes": self.overlap_notes}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InteractionGroup":
        allowed = {"groupId", "memberLayerIds", "registrationFrame", "relationOrder", "parentCandidateId", "overlapNotes"}
        _strict(value, allowed, allowed, "interactionGroup")
        result = cls(_id(value["groupId"], "groupId"), _ids(value["memberLayerIds"], "memberLayerIds"), _coordinate_transform(value["registrationFrame"]),
                     list(value["relationOrder"]), value["parentCandidateId"], value["overlapNotes"])
        result.validate()
        return result


@dataclass
class DepthComposite:
    composite_id: str
    source_revision: int
    layer_manifest: list[dict[str, Any]]
    content_asset_id: str | None = None
    content_hash: str | None = None
    complete_plate_layer_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _id(self.composite_id, "compositeId")
        if self.source_revision < 0 or not self.layer_manifest:
            raise SchemaValidationError("depth composite needs a source revision and layer manifest")
        if self.content_asset_id is not None:
            _id(self.content_asset_id, "contentAssetId")
        if self.content_hash is not None:
            _digest(self.content_hash)
        if self.complete_plate_layer_id is not None:
            _id(self.complete_plate_layer_id, "completePlateLayerId")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"compositeId": self.composite_id, "sourceRevision": self.source_revision, "layerManifest": self.layer_manifest,
                "contentAssetId": self.content_asset_id, "contentHash": self.content_hash,
                "completePlateLayerId": self.complete_plate_layer_id, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DepthComposite":
        allowed = {"compositeId", "sourceRevision", "layerManifest", "contentAssetId", "contentHash", "completePlateLayerId", "metadata"}
        _strict(value, allowed, allowed, "depthComposite")
        result = cls(_id(value["compositeId"], "compositeId"), int(value["sourceRevision"]), list(value["layerManifest"]),
                     value["contentAssetId"], value["contentHash"], value["completePlateLayerId"], _dict(value["metadata"]))
        result.validate()
        return result


@dataclass
class GuideRecord:
    guide_id: str
    name: str
    lifecycle: str = GuideLifecycle.PROPOSED.value
    created_revision: int = 0
    state_revision: int = 0
    supersedes_id: str | None = None
    replacement_id: str | None = None
    semantic_role: str | None = None
    object_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _id(self.guide_id, "guideId")
        if self.lifecycle not in {state.value for state in GuideLifecycle}:
            raise SchemaValidationError("invalid guide lifecycle")
        if self.created_revision < 0 or self.state_revision < self.created_revision:
            raise SchemaValidationError("guide revisions are invalid")
        if self.supersedes_id is not None:
            _id(self.supersedes_id, "supersedesId")
        if self.replacement_id is not None:
            _id(self.replacement_id, "replacementId")
        _ids(self.object_ids, "objectIds")

    def transition(self, lifecycle: str, revision: int, *, replacement_id: str | None = None) -> "GuideRecord":
        self.validate()
        allowed = {
            GuideLifecycle.PROPOSED.value: {GuideLifecycle.ACTIVE.value, GuideLifecycle.REPLACEMENT_PENDING.value, GuideLifecycle.SAFE_TO_REMOVE.value},
            GuideLifecycle.ACTIVE.value: {GuideLifecycle.CONSUMED.value, GuideLifecycle.REPLACEMENT_PENDING.value, GuideLifecycle.SUPERSEDED.value, GuideLifecycle.SAFE_TO_REMOVE.value},
            GuideLifecycle.CONSUMED.value: {GuideLifecycle.REPLACEMENT_PENDING.value, GuideLifecycle.SUPERSEDED.value, GuideLifecycle.SAFE_TO_REMOVE.value},
            GuideLifecycle.REPLACEMENT_PENDING.value: {GuideLifecycle.ACTIVE.value, GuideLifecycle.SUPERSEDED.value, GuideLifecycle.SAFE_TO_REMOVE.value},
            GuideLifecycle.SUPERSEDED.value: {GuideLifecycle.SAFE_TO_REMOVE.value},
            GuideLifecycle.SAFE_TO_REMOVE.value: set(),
        }
        if lifecycle not in allowed[self.lifecycle]:
            raise SchemaValidationError(f"invalid guide transition {self.lifecycle} -> {lifecycle}")
        if revision < self.state_revision:
            raise SchemaValidationError("guide lifecycle revisions must be monotonic")
        self.lifecycle = lifecycle
        self.state_revision = revision
        if replacement_id is not None:
            self.replacement_id = _id(replacement_id, "replacementId")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"guideId": self.guide_id, "name": self.name, "lifecycle": self.lifecycle, "createdRevision": self.created_revision,
                "stateRevision": self.state_revision, "supersedesId": self.supersedes_id, "replacementId": self.replacement_id,
                "semanticRole": self.semantic_role, "objectIds": list(self.object_ids), "metadata": self.metadata}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GuideRecord":
        allowed = {"guideId", "name", "lifecycle", "createdRevision", "stateRevision", "supersedesId", "replacementId", "semanticRole", "objectIds", "metadata"}
        _strict(value, allowed, allowed, "guide")
        result = cls(_id(value["guideId"], "guideId"), str(value["name"]), str(value["lifecycle"]), int(value["createdRevision"]), int(value["stateRevision"]),
                     value["supersedesId"], value["replacementId"], value["semanticRole"], _ids(value["objectIds"], "objectIds"), _dict(value["metadata"]))
        result.validate()
        return result


@dataclass
class BBOperation:
    operation_id: str
    schema_version: int
    source_document_id: str
    source_revision: int
    source_composite_id: str
    source_layer_manifest: list[dict[str, Any]]
    context_mask_id: str
    context_mask_hash: str
    source_bbox: BBox
    source_crop_dimensions: tuple[int, int]
    native_target_size: tuple[int, int]
    crop_to_native: CoordinateTransform
    native_to_crop: CoordinateTransform
    native_bb_asset_id: str
    generation_mask_id: str | None = None
    generation_mask_hash: str | None = None
    guidance_ids: list[str] = field(default_factory=list)
    blend_mask_id: str | None = None
    blend_policy: dict[str, Any] = field(default_factory=dict)
    outpaint: dict[str, Any] | None = None
    candidate_ids: list[str] = field(default_factory=list)
    extraction_ids: list[str] = field(default_factory=list)
    state: str = "current"
    stale_reason: str | None = None
    operation_revision: int = 1

    def validate(self) -> None:
        _id(self.operation_id, "operationId")
        _id(self.source_document_id, "sourceDocumentId")
        _id(self.source_composite_id, "sourceCompositeId")
        _id(self.context_mask_id, "contextMaskId")
        _digest(self.context_mask_hash, "contextMaskHash")
        _id(self.native_bb_asset_id, "nativeBBAssetId")
        self.source_bbox.validate()
        if self.source_revision < 0 or self.operation_revision < 1:
            raise SchemaValidationError("BB revisions are invalid")
        if len(self.source_crop_dimensions) != 2 or any(v <= 0 for v in self.source_crop_dimensions):
            raise SchemaValidationError("source crop dimensions are invalid")
        if len(self.native_target_size) != 2 or any(v <= 0 for v in self.native_target_size):
            raise SchemaValidationError("native target size is invalid")
        _coordinate_transform(self.crop_to_native)
        _coordinate_transform(self.native_to_crop)
        if self.crop_to_native.from_space != "source-crop" or self.crop_to_native.to_space != "native-bb":
            raise SchemaValidationError("BB crop transform must map source-crop to native-bb")
        if self.native_to_crop.from_space != "native-bb" or self.native_to_crop.to_space != "source-crop":
            raise SchemaValidationError("BB inverse transform must map native-bb to source-crop")
        if self.crop_to_native.source_dimensions != (self.source_bbox.width, self.source_bbox.height):
            raise SchemaValidationError("BB crop transform dimensions do not match source bbox")
        if self.crop_to_native.target_dimensions != self.native_target_size:
            raise SchemaValidationError("BB crop transform dimensions do not match native target")
        if self.generation_mask_id is not None:
            _id(self.generation_mask_id, "generationMaskId")
        if self.generation_mask_hash is not None:
            _digest(self.generation_mask_hash, "generationMaskHash")
        if self.blend_mask_id is not None:
            _id(self.blend_mask_id, "blendMaskId")
        _ids(self.guidance_ids, "guidanceIds")
        _ids(self.candidate_ids, "candidateIds")
        _ids(self.extraction_ids, "extractionIds")
        if self.state not in {"current", "stale"}:
            raise SchemaValidationError("invalid BB operation state")
        if self.state == "stale" and not self.stale_reason:
            raise SchemaValidationError("stale BB operation requires a reason")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"operationId": self.operation_id, "schemaVersion": self.schema_version, "sourceDocumentId": self.source_document_id,
                "sourceRevision": self.source_revision, "sourceCompositeId": self.source_composite_id,
                "sourceLayerManifest": self.source_layer_manifest, "contextMaskId": self.context_mask_id,
                "contextMaskHash": self.context_mask_hash, "sourceBbox": self.source_bbox.to_dict(),
                "sourceCropDimensions": list(self.source_crop_dimensions), "nativeTargetSize": list(self.native_target_size),
                "cropToNative": self.crop_to_native.to_dict(), "nativeToCrop": self.native_to_crop.to_dict(),
                "nativeBBAssetId": self.native_bb_asset_id, "generationMaskId": self.generation_mask_id,
                "generationMaskHash": self.generation_mask_hash, "guidanceIds": list(self.guidance_ids),
                "blendMaskId": self.blend_mask_id, "blendPolicy": self.blend_policy, "outpaint": self.outpaint,
                "candidateIds": list(self.candidate_ids), "extractionIds": list(self.extraction_ids), "state": self.state,
                "staleReason": self.stale_reason, "operationRevision": self.operation_revision}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BBOperation":
        allowed = {"operationId", "schemaVersion", "sourceDocumentId", "sourceRevision", "sourceCompositeId", "sourceLayerManifest", "contextMaskId", "contextMaskHash", "sourceBbox", "sourceCropDimensions", "nativeTargetSize", "cropToNative", "nativeToCrop", "nativeBBAssetId", "generationMaskId", "generationMaskHash", "guidanceIds", "blendMaskId", "blendPolicy", "outpaint", "candidateIds", "extractionIds", "state", "staleReason", "operationRevision"}
        _strict(value, allowed, allowed, "bbOperation")
        result = cls(_id(value["operationId"], "operationId"), int(value["schemaVersion"]), _id(value["sourceDocumentId"], "sourceDocumentId"), int(value["sourceRevision"]), _id(value["sourceCompositeId"], "sourceCompositeId"), list(value["sourceLayerManifest"]), _id(value["contextMaskId"], "contextMaskId"), _digest(value["contextMaskHash"], "contextMaskHash"), _bbox(value["sourceBbox"]), tuple(int(v) for v in value["sourceCropDimensions"]), tuple(int(v) for v in value["nativeTargetSize"]), _coordinate_transform(value["cropToNative"]), _coordinate_transform(value["nativeToCrop"]), _id(value["nativeBBAssetId"], "nativeBBAssetId"), value["generationMaskId"], value["generationMaskHash"], _ids(value["guidanceIds"], "guidanceIds"), value["blendMaskId"], _dict(value["blendPolicy"]), value["outpaint"], _ids(value["candidateIds"], "candidateIds"), _ids(value["extractionIds"], "extractionIds"), str(value["state"]), value["staleReason"], int(value["operationRevision"]))
        result.validate()
        return result


@dataclass
class CandidateRecord:
    candidate_id: str
    kind: str
    operation_id: str
    source_revision: int
    full_crop: bool
    blend_mask_id: str | None
    placement_transform: CoordinateTransform
    status: str = "proposed"
    asset_id: str | None = None
    parent_candidate_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _id(self.candidate_id, "candidateId")
        _id(self.operation_id, "operationId")
        if self.kind != "full_crop_contextual" or not self.full_crop:
            raise SchemaValidationError("contextual candidates must be full_crop_contextual")
        if self.source_revision < 0:
            raise SchemaValidationError("candidate source revision invalid")
        _coordinate_transform(self.placement_transform)
        if self.blend_mask_id is not None:
            _id(self.blend_mask_id, "blendMaskId")
        if self.asset_id is not None:
            _id(self.asset_id, "assetId")
        if self.parent_candidate_id is not None:
            _id(self.parent_candidate_id, "parentCandidateId")
        if self.status not in {"proposed", "accepted-for-extraction", "rejected", "extracted"}:
            raise SchemaValidationError("invalid candidate status")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"candidateId": self.candidate_id, "kind": self.kind, "operationId": self.operation_id,
                "sourceRevision": self.source_revision, "fullCrop": self.full_crop, "blendMaskId": self.blend_mask_id,
                "placementTransform": self.placement_transform.to_dict(), "status": self.status, "assetId": self.asset_id,
                "parentCandidateId": self.parent_candidate_id, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateRecord":
        allowed = {"candidateId", "kind", "operationId", "sourceRevision", "fullCrop", "blendMaskId", "placementTransform", "status", "assetId", "parentCandidateId", "metadata"}
        _strict(value, allowed, allowed, "candidate")
        result = cls(_id(value["candidateId"], "candidateId"), str(value["kind"]), _id(value["operationId"], "operationId"), int(value["sourceRevision"]), bool(value["fullCrop"]), value["blendMaskId"], _coordinate_transform(value["placementTransform"]), str(value["status"]), value["assetId"], value["parentCandidateId"], _dict(value["metadata"]))
        result.validate()
        return result


@dataclass
class ExtractionDerivative:
    derivative_id: str
    parent_candidate_id: str
    operation_id: str
    extraction_matte_id: str
    placement_transform: CoordinateTransform
    source_revision: int
    rgba_asset_id: str
    blend_stitch_metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "registered"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for value, name in ((self.derivative_id, "derivativeId"), (self.parent_candidate_id, "parentCandidateId"), (self.operation_id, "operationId"), (self.extraction_matte_id, "extractionMatteId"), (self.rgba_asset_id, "rgbaAssetId")):
            _id(value, name)
        _coordinate_transform(self.placement_transform)
        if self.source_revision < 0:
            raise SchemaValidationError("derivative source revision invalid")
        if self.status not in {"registered", "accepted", "rejected"}:
            raise SchemaValidationError("invalid derivative status")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"derivativeId": self.derivative_id, "parentCandidateId": self.parent_candidate_id, "operationId": self.operation_id,
                "extractionMatteId": self.extraction_matte_id, "placementTransform": self.placement_transform.to_dict(),
                "sourceRevision": self.source_revision, "rgbaAssetId": self.rgba_asset_id,
                "blendStitchMetadata": self.blend_stitch_metadata, "status": self.status, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExtractionDerivative":
        allowed = {"derivativeId", "parentCandidateId", "operationId", "extractionMatteId", "placementTransform", "sourceRevision", "rgbaAssetId", "blendStitchMetadata", "status", "metadata"}
        _strict(value, allowed, allowed, "extractionDerivative")
        result = cls(_id(value["derivativeId"], "derivativeId"), _id(value["parentCandidateId"], "parentCandidateId"), _id(value["operationId"], "operationId"), _id(value["extractionMatteId"], "extractionMatteId"), _coordinate_transform(value["placementTransform"]), int(value["sourceRevision"]), _id(value["rgbaAssetId"], "rgbaAssetId"), _dict(value["blendStitchMetadata"]), str(value["status"]), _dict(value["metadata"]))
        result.validate()
        return result


@dataclass
class CorrectivePatch:
    patch_id: str
    parent_id: str
    transform: AffineTransform
    revision: int
    mask_id: str | None = None
    visible: bool = True
    external_exchange_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _id(self.patch_id, "patchId")
        _id(self.parent_id, "parentId")
        if self.mask_id is not None:
            _id(self.mask_id, "maskId")
        if self.external_exchange_id is not None:
            _id(self.external_exchange_id, "externalExchangeId")
        if self.revision < 0:
            raise SchemaValidationError("patch revision invalid")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"patchId": self.patch_id, "parentId": self.parent_id, "transform": self.transform.to_dict(), "revision": self.revision,
                "maskId": self.mask_id, "visible": self.visible, "externalExchangeId": self.external_exchange_id, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CorrectivePatch":
        allowed = {"patchId", "parentId", "transform", "revision", "maskId", "visible", "externalExchangeId", "metadata"}
        _strict(value, allowed, allowed, "correctivePatch")
        result = cls(_id(value["patchId"], "patchId"), _id(value["parentId"], "parentId"), _transform(value["transform"]), int(value["revision"]), value["maskId"], bool(value["visible"]), value["externalExchangeId"], _dict(value["metadata"]))
        result.validate()
        return result


@dataclass
class ExternalRoundTrip:
    exchange_id: str
    document_id: str
    target_ids: list[str]
    exported_revision: int
    coordinate_space: str
    crop_origin: tuple[int, int] | None
    bbox: BBox | None
    transform: CoordinateTransform
    export_asset_hash: str
    import_mode: str = "new-revision"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _id(self.exchange_id, "exchangeId")
        _id(self.document_id, "documentId")
        _ids(self.target_ids, "targetIds")
        _digest(self.export_asset_hash, "exportAssetHash")
        _coordinate_transform(self.transform)
        if self.import_mode not in {"new-revision", "director-selected-replacement"}:
            raise SchemaValidationError("invalid external import mode")
        if self.bbox is not None:
            self.bbox.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"exchangeId": self.exchange_id, "documentId": self.document_id, "targetIds": list(self.target_ids), "exportedRevision": self.exported_revision,
                "coordinateSpace": self.coordinate_space, "cropOrigin": None if self.crop_origin is None else list(self.crop_origin),
                "bbox": None if self.bbox is None else self.bbox.to_dict(), "transform": self.transform.to_dict(), "exportAssetHash": self.export_asset_hash,
                "importMode": self.import_mode, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExternalRoundTrip":
        allowed = {"exchangeId", "documentId", "targetIds", "exportedRevision", "coordinateSpace", "cropOrigin", "bbox", "transform", "exportAssetHash", "importMode", "metadata"}
        _strict(value, allowed, allowed, "externalRoundTrip")
        result = cls(_id(value["exchangeId"], "exchangeId"), _id(value["documentId"], "documentId"), _ids(value["targetIds"], "targetIds"), int(value["exportedRevision"]), str(value["coordinateSpace"]), None if value["cropOrigin"] is None else tuple(int(v) for v in value["cropOrigin"]), None if value["bbox"] is None else _bbox(value["bbox"]), _coordinate_transform(value["transform"]), _digest(value["exportAssetHash"], "exportAssetHash"), str(value["importMode"]), _dict(value["metadata"]))
        result.validate()
        return result


@dataclass
class PrivateProxy:
    """An Agent-addressable proxy with no private-source backlink."""

    proxy_id: str
    proxy_asset_id: str
    proxy_content_hash: str
    destination_slot_id: str
    registration: CoordinateTransform
    permitted_depth_metadata: dict[str, Any] = field(default_factory=dict)
    coverage_matte_id: str | None = None

    def validate(self) -> None:
        for value, name in ((self.proxy_id, "proxyId"), (self.proxy_asset_id, "proxyAssetId"), (self.destination_slot_id, "destinationSlotId")):
            _id(value, name)
        _digest(self.proxy_content_hash, "proxyContentHash")
        _coordinate_transform(self.registration)
        if self.coverage_matte_id is not None:
            _id(self.coverage_matte_id, "coverageMatteId")
        for key in self.permitted_depth_metadata:
            if any(token in key.lower() for token in ("private", "sourcepath", "source_path", "thumbnail", "history", "pixel")):
                raise SchemaValidationError("private proxy metadata contains a forbidden backlink")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        # Deliberately no private-source fields exist in this serialized shape.
        return {"proxyId": self.proxy_id, "proxyAssetId": self.proxy_asset_id, "proxyContentHash": self.proxy_content_hash,
                "destinationSlotId": self.destination_slot_id, "registration": self.registration.to_dict(),
                "permittedDepthMetadata": self.permitted_depth_metadata, "coverageMatteId": self.coverage_matte_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PrivateProxy":
        allowed = {"proxyId", "proxyAssetId", "proxyContentHash", "destinationSlotId", "registration", "permittedDepthMetadata", "coverageMatteId"}
        _strict(value, allowed, allowed, "privateProxy")
        result = cls(_id(value["proxyId"], "proxyId"), _id(value["proxyAssetId"], "proxyAssetId"), _digest(value["proxyContentHash"], "proxyContentHash"), _id(value["destinationSlotId"], "destinationSlotId"), _coordinate_transform(value["registration"]), _dict(value["permittedDepthMetadata"]), value["coverageMatteId"])
        result.validate()
        return result


@dataclass
class OperationRecord:
    operation_id: str
    operation_type: str
    schema_version: int
    input_refs: list[dict[str, Any]] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    produced_ids: list[str] = field(default_factory=list)
    actor_id: str = "system"
    state: str = "complete"
    parent_ids: list[str] = field(default_factory=list)
    child_ids: list[str] = field(default_factory=list)

    def validate(self) -> None:
        _id(self.operation_id, "operationId")
        if not self.operation_type or self.schema_version < 1:
            raise SchemaValidationError("operation type and version are required")
        _ids(self.produced_ids, "producedIds")
        _ids(self.parent_ids, "parentIds")
        _ids(self.child_ids, "childIds")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"operationId": self.operation_id, "operationType": self.operation_type, "schemaVersion": self.schema_version,
                "inputRefs": self.input_refs, "settings": self.settings, "producedIds": list(self.produced_ids), "actorId": self.actor_id,
                "state": self.state, "parentIds": list(self.parent_ids), "childIds": list(self.child_ids)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OperationRecord":
        allowed = {"operationId", "operationType", "schemaVersion", "inputRefs", "settings", "producedIds", "actorId", "state", "parentIds", "childIds"}
        _strict(value, allowed, allowed, "operation")
        result = cls(_id(value["operationId"], "operationId"), str(value["operationType"]), int(value["schemaVersion"]), list(value["inputRefs"]), _dict(value["settings"]), _ids(value["producedIds"], "producedIds"), str(value["actorId"]), str(value["state"]), _ids(value["parentIds"], "parentIds"), _ids(value["childIds"], "childIds"))
        result.validate()
        return result


@dataclass
class Document:
    document_id: str
    width: int
    height: int
    color_space_intent: str = "sRGB"
    current_revision: int = 0
    schema_version: int = SCHEMA_VERSION
    format_id: str = FORMAT_IDENTIFIER
    root_layer_ids: list[str] = field(default_factory=list)
    layers: dict[str, LayerRecord] = field(default_factory=dict)
    objects: dict[str, ObjectRecord] = field(default_factory=dict)
    assets: dict[str, AssetRecord] = field(default_factory=dict)
    masks: dict[str, MaskRecord] = field(default_factory=dict)
    selections: dict[str, SelectionRecord] = field(default_factory=dict)
    variants: dict[str, VariantSet] = field(default_factory=dict)
    interaction_groups: dict[str, InteractionGroup] = field(default_factory=dict)
    operations: dict[str, OperationRecord] = field(default_factory=dict)
    depth_composites: dict[str, DepthComposite] = field(default_factory=dict)
    bb_operations: dict[str, BBOperation] = field(default_factory=dict)
    candidates: dict[str, CandidateRecord] = field(default_factory=dict)
    extractions: dict[str, ExtractionDerivative] = field(default_factory=dict)
    guides: dict[str, GuideRecord] = field(default_factory=dict)
    patches: dict[str, CorrectivePatch] = field(default_factory=dict)
    external_round_trips: dict[str, ExternalRoundTrip] = field(default_factory=dict)
    private_proxies: dict[str, PrivateProxy] = field(default_factory=dict)
    visual_profile: VisualProfile = field(default_factory=VisualProfile)
    history: list[dict[str, Any]] = field(default_factory=list)
    history_refs: list[str] = field(default_factory=list)
    checkpoint_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    revision_digest: str | None = None

    def validate(self) -> "Document":
        _id(self.document_id, "documentId")
        if self.format_id != FORMAT_IDENTIFIER:
            raise SchemaValidationError("unknown document format")
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaValidationError(f"unsupported schema version: {self.schema_version}")
        if self.width <= 0 or self.height <= 0 or self.current_revision < 0:
            raise SchemaValidationError("document dimensions or revision are invalid")
        for key, record in self.layers.items():
            if key != record.layer_id:
                raise SchemaValidationError("layer map key does not match layerId")
            record.validate()
            if any(object_id not in self.objects for object_id in record.object_ids):
                raise SchemaValidationError("layer references unknown object")
            if any(asset_id not in self.assets for asset_id in record.asset_ids):
                raise SchemaValidationError("layer references unknown asset")
            if any(mask_id not in self.masks for mask_id in record.mask_ids):
                raise SchemaValidationError("layer references unknown mask")
        try:
            validate_layer_tree(self.layers, self.root_layer_ids)
        except TreeValidationError as exc:
            raise SchemaValidationError(str(exc)) from exc
        for key, record in self.objects.items():
            if key != record.object_id:
                raise SchemaValidationError("object map key does not match objectId")
            record.validate()
            if record.layer_id not in self.layers:
                raise SchemaValidationError("object references unknown layer")
            if record.asset_id is not None and record.asset_id not in self.assets:
                raise SchemaValidationError("object references unknown asset")
        for key, record in self.assets.items():
            if key != record.asset_id:
                raise SchemaValidationError("asset map key does not match assetId")
            record.validate()
        for key, record in self.masks.items():
            if key != record.mask_id:
                raise SchemaValidationError("mask map key does not match maskId")
            record.validate()
            if record.asset_id is not None and record.asset_id not in self.assets:
                raise SchemaValidationError("mask references unknown asset")
            if record.owner_id is not None and record.owner_id not in self.layers and record.owner_id not in self.objects and record.owner_id not in self.bb_operations and record.owner_id not in self.operations:
                raise SchemaValidationError("mask references unknown owner")
        for key, record in self.selections.items():
            if key != record.selection_id:
                raise SchemaValidationError("selection map key does not match selectionId")
            record.validate()
            if record.source_id not in self.layers and record.source_id not in self.depth_composites:
                raise SchemaValidationError("selection references unknown source")
            if record.mask_id not in self.masks:
                raise SchemaValidationError("selection references unknown mask")
            if any(layer_id not in self.layers for layer_id in record.derived_layer_ids):
                raise SchemaValidationError("selection has unknown derived layer")
        for key, record in self.variants.items():
            if key != record.variant_set_id:
                raise SchemaValidationError("variant map key does not match variantSetId")
            record.validate()
            if any(member_id not in self.layers and member_id not in self.candidates and member_id not in self.extractions for member_id in record.member_ids):
                raise SchemaValidationError("variant set references unknown member")
        for key, record in self.interaction_groups.items():
            if key != record.group_id:
                raise SchemaValidationError("interaction group map key does not match groupId")
            record.validate()
            if any(layer_id not in self.layers for layer_id in record.member_layer_ids):
                raise SchemaValidationError("interaction group references unknown layer")
            if record.parent_candidate_id is not None and record.parent_candidate_id not in self.candidates:
                raise SchemaValidationError("interaction group references unknown candidate")
        for key, record in self.operations.items():
            if key != record.operation_id:
                raise SchemaValidationError("operation map key does not match operationId")
            record.validate()
        for collection, identity_field in (
            (self.depth_composites, "composite_id"), (self.bb_operations, "operation_id"),
            (self.candidates, "candidate_id"), (self.extractions, "derivative_id"),
            (self.guides, "guide_id"), (self.patches, "patch_id"),
            (self.external_round_trips, "exchange_id"), (self.private_proxies, "proxy_id"),
        ):
            for key, record in collection.items():
                record.validate()
                record_id = getattr(record, identity_field)
                if key != record_id:
                    raise SchemaValidationError("record map key does not match record identity")
        for composite in self.depth_composites.values():
            for item in composite.layer_manifest:
                layer_id = item.get("layerId") if isinstance(item, Mapping) else None
                if layer_id is not None and layer_id not in self.layers:
                    raise SchemaValidationError("depth composite references unknown layer")
        for operation in self.bb_operations.values():
            if operation.source_composite_id not in self.depth_composites:
                raise SchemaValidationError("BB operation references unknown depth composite")
            if operation.context_mask_id not in self.masks:
                raise SchemaValidationError("BB operation references unknown context mask")
            if operation.native_bb_asset_id not in self.assets:
                raise SchemaValidationError("BB operation references unknown native BB asset")
            if operation.generation_mask_id is not None and operation.generation_mask_id not in self.masks:
                raise SchemaValidationError("BB operation references unknown generation mask")
            if operation.blend_mask_id is not None and operation.blend_mask_id not in self.masks:
                raise SchemaValidationError("BB operation references unknown blend mask")
            if any(candidate_id not in self.candidates for candidate_id in operation.candidate_ids):
                raise SchemaValidationError("BB operation references unknown candidate")
            if any(extraction_id not in self.extractions for extraction_id in operation.extraction_ids):
                raise SchemaValidationError("BB operation references unknown extraction")
        for candidate in self.candidates.values():
            if candidate.operation_id not in self.bb_operations and candidate.operation_id not in self.operations:
                raise SchemaValidationError("candidate references unknown operation")
            if candidate.blend_mask_id is not None and candidate.blend_mask_id not in self.masks:
                raise SchemaValidationError("candidate references unknown blend mask")
            if candidate.asset_id is not None and candidate.asset_id not in self.assets:
                raise SchemaValidationError("candidate references unknown asset")
            if candidate.parent_candidate_id is not None and candidate.parent_candidate_id not in self.candidates:
                raise SchemaValidationError("candidate references unknown parent candidate")
        for extraction in self.extractions.values():
            if extraction.parent_candidate_id not in self.candidates:
                raise SchemaValidationError("extraction references unknown candidate")
            if extraction.operation_id not in self.bb_operations and extraction.operation_id not in self.operations:
                raise SchemaValidationError("extraction references unknown operation")
            if extraction.extraction_matte_id not in self.masks:
                raise SchemaValidationError("extraction references unknown matte")
            if extraction.rgba_asset_id not in self.assets:
                raise SchemaValidationError("extraction references unknown RGBA asset")
        for guide in self.guides.values():
            if any(object_id not in self.objects for object_id in guide.object_ids):
                raise SchemaValidationError("guide references unknown object")
        for patch in self.patches.values():
            if patch.parent_id not in self.layers and patch.parent_id not in self.objects and patch.parent_id not in self.interaction_groups:
                raise SchemaValidationError("patch references unknown parent")
            if patch.mask_id is not None and patch.mask_id not in self.masks:
                raise SchemaValidationError("patch references unknown mask")
        for exchange in self.external_round_trips.values():
            if any(target_id not in self.layers and target_id not in self.objects and target_id not in self.patches for target_id in exchange.target_ids):
                raise SchemaValidationError("external round trip references unknown target")
        for proxy in self.private_proxies.values():
            if proxy.proxy_asset_id not in self.assets:
                raise SchemaValidationError("private proxy references unknown proxy asset")
            if proxy.coverage_matte_id is not None and proxy.coverage_matte_id not in self.masks:
                raise SchemaValidationError("private proxy references unknown coverage matte")
        previous_revision = 0
        transaction_ids: list[str] = []
        for transaction in self.history:
            if not isinstance(transaction, Mapping):
                raise SchemaValidationError("history records must be objects")
            transaction_id = _id(transaction.get("transactionId"), "transactionId")
            if int(transaction.get("previousRevision", -1)) != previous_revision:
                raise SchemaValidationError("history is not a contiguous revision chain")
            resulting_revision = int(transaction.get("resultingRevision", -1))
            if resulting_revision <= previous_revision:
                raise SchemaValidationError("history revisions must advance")
            previous_revision = resulting_revision
            transaction_ids.append(transaction_id)
        if previous_revision != self.current_revision:
            raise SchemaValidationError("history does not end at current document revision")
        if self.history_refs and self.history_refs != transaction_ids:
            raise SchemaValidationError("historyRefs do not match retained history")
        if len(set(self.checkpoint_refs)) != len(self.checkpoint_refs):
            raise SchemaValidationError("duplicate checkpoint reference")
        for layer in self.layers.values():
            if layer.complete_plate_id is not None and layer.complete_plate_id not in self.layers and layer.complete_plate_id not in self.assets:
                raise SchemaValidationError("depth layer references missing complete plate")
        if self.revision_digest is not None:
            _digest(self.revision_digest, "revisionDigest")
        return self

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        self.validate()
        value = {
            "formatId": self.format_id, "schemaVersion": self.schema_version, "documentId": self.document_id,
            "width": self.width, "height": self.height, "colorSpaceIntent": self.color_space_intent,
            "currentRevision": self.current_revision, "rootLayerIds": list(self.root_layer_ids),
            "layers": {key: record.to_dict() for key, record in self.layers.items()},
            "objects": {key: record.to_dict() for key, record in self.objects.items()},
            "assets": {key: record.to_dict() for key, record in self.assets.items()},
            "masks": {key: record.to_dict() for key, record in self.masks.items()},
            "selections": {key: record.to_dict() for key, record in self.selections.items()},
            "variants": {key: record.to_dict() for key, record in self.variants.items()},
            "interactionGroups": {key: record.to_dict() for key, record in self.interaction_groups.items()},
            "operations": {key: record.to_dict() for key, record in self.operations.items()},
            "depthComposites": {key: record.to_dict() for key, record in self.depth_composites.items()},
            "bbOperations": {key: record.to_dict() for key, record in self.bb_operations.items()},
            "candidates": {key: record.to_dict() for key, record in self.candidates.items()},
            "extractions": {key: record.to_dict() for key, record in self.extractions.items()},
            "guides": {key: record.to_dict() for key, record in self.guides.items()},
            "patches": {key: record.to_dict() for key, record in self.patches.items()},
            "externalRoundTrips": {key: record.to_dict() for key, record in self.external_round_trips.items()},
            "privateProxies": {key: record.to_dict() for key, record in self.private_proxies.items()},
            "visualProfile": self.visual_profile.to_dict(), "history": self.history,
            "historyRefs": list(self.history_refs), "checkpointRefs": list(self.checkpoint_refs),
            "metadata": self.metadata,
        }
        if include_digest:
            value["revisionDigest"] = self.revision_digest or content_digest(value)
        return value

    def canonical_digest(self) -> str:
        return content_digest(self.to_dict(include_digest=False))

    def to_json(self) -> str:
        from .ids import canonical_json
        return canonical_json(self.to_dict()).decode("utf-8")

    def refresh_digest(self) -> str:
        self.revision_digest = self.canonical_digest()
        return self.revision_digest

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Document":
        # Migration is intentionally opt-in at the project boundary; the
        # registry can turn a prior version into this exact shape before here.
        allowed = {"formatId", "schemaVersion", "documentId", "width", "height", "colorSpaceIntent", "currentRevision", "rootLayerIds", "layers", "objects", "assets", "masks", "selections", "variants", "interactionGroups", "operations", "depthComposites", "bbOperations", "candidates", "extractions", "guides", "patches", "externalRoundTrips", "privateProxies", "visualProfile", "history", "historyRefs", "checkpointRefs", "metadata", "revisionDigest"}
        _strict(value, allowed, allowed - {"revisionDigest"}, "document")
        def parse_map(raw: Mapping[str, Any], parser: Any) -> dict[str, Any]:
            if not isinstance(raw, Mapping):
                raise SchemaValidationError("record collection must be an object")
            return {str(key): parser(record) for key, record in raw.items()}
        result = cls(
            _id(value["documentId"], "documentId"), int(value["width"]), int(value["height"]), str(value["colorSpaceIntent"]), int(value["currentRevision"]), int(value["schemaVersion"]), str(value["formatId"]), _ids(value["rootLayerIds"], "rootLayerIds"),
            parse_map(value["layers"], LayerRecord.from_dict), parse_map(value["objects"], ObjectRecord.from_dict), parse_map(value["assets"], AssetRecord.from_dict), parse_map(value["masks"], MaskRecord.from_dict), parse_map(value["selections"], SelectionRecord.from_dict), parse_map(value["variants"], VariantSet.from_dict), parse_map(value["interactionGroups"], InteractionGroup.from_dict), parse_map(value["operations"], OperationRecord.from_dict), parse_map(value["depthComposites"], DepthComposite.from_dict), parse_map(value["bbOperations"], BBOperation.from_dict), parse_map(value["candidates"], CandidateRecord.from_dict), parse_map(value["extractions"], ExtractionDerivative.from_dict), parse_map(value["guides"], GuideRecord.from_dict), parse_map(value["patches"], CorrectivePatch.from_dict), parse_map(value["externalRoundTrips"], ExternalRoundTrip.from_dict), parse_map(value["privateProxies"], PrivateProxy.from_dict), VisualProfile.from_dict(value["visualProfile"]), list(value["history"]), _ids(value["historyRefs"], "historyRefs"), [str(v) for v in value["checkpointRefs"]], _dict(value["metadata"]), value.get("revisionDigest"),
        )
        result.validate()
        if result.revision_digest is not None and result.revision_digest != result.canonical_digest():
            raise SchemaValidationError("revision digest does not match canonical document")
        return result

    @classmethod
    def from_json(cls, value: str) -> "Document":
        import json
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(f"invalid document JSON: {exc}") from exc
        if not isinstance(parsed, Mapping):
            raise SchemaValidationError("document JSON must contain an object")
        return cls.from_dict(parsed)

    def transition_guide(self, guide_id: str, lifecycle: str, revision: int, *, replacement_id: str | None = None) -> GuideRecord:
        guide = self.guides[_id(guide_id, "guideId")]
        guide.transition(lifecycle, revision, replacement_id=replacement_id)
        return guide

    def invalidate_selections_for_source(
        self,
        source_id: str,
        *,
        reason: str,
        source_revision: int,
        source_content_digest: str | None = None,
    ) -> list[str]:
        """Apply source-side selection state changes in the source transaction."""

        source_id = _id(source_id, "sourceId")
        changed: list[str] = []
        for selection in self.selections.values():
            if selection.source_id == source_id:
                selection.mark_source_changed(reason, source_revision, source_content_digest)
                changed.append(selection.selection_id)
        return changed

    def agent_safe_dict(self) -> dict[str, Any]:
        """Return an allowlisted projection without private-source payloads."""

        self.validate()
        layers: dict[str, Any] = {}
        for layer_id, layer in self.layers.items():
            layers[layer_id] = {
                "layerId": layer.layer_id,
                "name": layer.name,
                "kind": layer.kind,
                "parentId": layer.parent_id,
                "childIds": list(layer.child_ids),
                "visible": layer.visible,
                "locked": layer.locked,
                "opacity": layer.opacity,
                "blendMode": layer.blend_mode,
                "transform": layer.transform.to_dict(),
                "collaboration": {
                    "workStatus": layer.collaboration.work_status,
                    "lastEditorKind": None if layer.collaboration.last_editor is None else layer.collaboration.last_editor["actorKind"],
                    "handoffNotes": [
                        {
                            "noteId": note.note_id,
                            "actorKind": note.actor_kind,
                            "body": note.body,
                            "createdAtRevision": note.created_at_revision,
                            "transactionId": note.transaction_id,
                            "targetIds": list(note.target_ids),
                            "targetRevisions": dict(note.target_revisions),
                            "state": note.state,
                        }
                        for note in layer.collaboration.handoff_notes
                        if note.visibility == "shared" and not any(token in note.body.lower() for token in ("private", "source_path", "source path", "thumbnail", "pixel"))
                    ],
                },
            }
        return {
            "formatId": self.format_id,
            "schemaVersion": self.schema_version,
            "documentId": self.document_id,
            "revision": self.current_revision,
            "width": self.width,
            "height": self.height,
            "layers": layers,
            "privateProxies": {key: proxy.to_dict() for key, proxy in self.private_proxies.items()},
        }


# Friendly aliases used by callers that prefer explicit record terminology.
DocumentRecord = Document
Layer = LayerRecord
Object = ObjectRecord
Mask = MaskRecord
Selection = SelectionRecord
BBOperationRecord = BBOperation
Candidate = CandidateRecord
ExtractionRecord = ExtractionDerivative
PrivateProxyRecord = PrivateProxy
TransformRecord = CoordinateTransform
