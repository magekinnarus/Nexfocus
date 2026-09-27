"""Versioned, renderer-independent records for an editable Nexfocus scene.

The schema deliberately stores meaning and provenance, not Konva/Gradio
objects.  Binary pixels are represented by :class:`AssetRecord` references;
they are owned by ``asset_store.py``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, ClassVar, Mapping, Sequence

from .history import (HistoryValidationError, RevisionConflict, SelectionRevisionConflict,
                      TransactionRecord, validate_history)
from .ids import content_digest, make_id, validate_id, validate_sha256
from .transforms import AffineTransform, BBox, CoordinateTransform
from .tree import TreeValidationError, validate_layer_tree


FORMAT_IDENTIFIER = "nexfocus.nexscene"
FORMAT_ID = FORMAT_IDENTIFIER
SCHEMA_VERSION = 2
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


_S = str
_I = int
_B = bool
_D = dict
_L = list
_N = (int, float)
_NS = (str, type(None))
_STRINGS = ("array", str)
_OBJECTS = ("array", dict)
_INTEGERS = ("array", int)
_NUMBERS = ("array", float)
_IDOCUMENT_TYPES: dict[str, dict[str, Any]] = {
    "visualProfile": {"palette": _D, "lighting": _D, "style": _D, "prompts": _STRINGS, "modelChoices": _D, "inferenceSettings": _D},
    "handoffNote": {"noteId": _S, "actorId": _S, "actorKind": _S, "body": _S, "createdAtRevision": _I,
                    "transactionId": _S, "targetIds": _STRINGS, "targetRevisions": ("object-values", _I), "visibility": _S, "state": _S, "orphanReason": _NS},
    "collaboration": {"workStatus": _S, "lastEditor": (_D, type(None)), "handoffNotes": _OBJECTS},
    "asset": {"assetId": _S, "contentHash": _S, "mediaType": _S, "storageUri": _S, "extension": _S,
              "byteLength": (_I, type(None)), "width": (_I, type(None)), "height": (_I, type(None)),
              "hasAlpha": (_B, type(None)), "colorSpace": _NS, "provenance": _D,
              "externalUri": _NS, "expectedHash": _NS, "externalStatus": _NS},
    "layer": {"layerId": _S, "name": _S, "kind": _S, "parentId": _NS, "childIds": _L, "visible": _B,
              "locked": _B, "opacity": _N, "blendMode": _S, "transform": _NUMBERS, "objectIds": _STRINGS,
              "assetIds": _STRINGS, "maskIds": _STRINGS, "role": _NS, "clippingRefs": _STRINGS, "collaboration": _D,
              "revision": _I, "depthElement": _B, "completePlateId": _NS,
              "registration": (_D, type(None)), "lineage": _D, "metadata": _D},
    "object": {"objectId": _S, "layerId": _S, "kind": _S, "coordinateSpace": _S, "geometry": _D,
               "transform": _NUMBERS, "assetId": _NS, "style": _D, "revision": _I, "lineage": _D, "metadata": _D},
    "mask": {"maskId": _S, "purpose": _S, "coordinateSpace": _S, "revision": _I, "assetId": _NS,
             "ownerId": _NS, "editableSource": (_D, type(None)), "lineage": _D, "contentHash": _NS},
    "selection": {"selectionId": _S, "sourceId": _S, "sourceRevision": _I, "sourceContentDigest": _S,
                  "selectionRevision": _I, "state": _S, "maskId": _S, "coordinateSpace": _S,
                  "seedGeometry": _OBJECTS, "semanticHint": _NS, "bounds": (_D, type(None)),
                  "mapping": (_D, type(None)), "refinementHistory": _OBJECTS, "derivedLayerIds": _STRINGS,
                  "staleReason": _NS, "currentSourceContentDigest": _NS, "invalidationHistory": _OBJECTS,
                  "rebaseHistory": _OBJECTS, "duplicateHistory": _OBJECTS},
    "variantSet": {"variantSetId": _S, "semanticRole": _S, "memberIds": _L, "activeMemberId": _NS,
                   "sharedAnchor": _D, "placementFrame": (_D, type(None)), "lineage": _D},
    "interactionGroup": {"groupId": _S, "memberLayerIds": _STRINGS, "registrationFrame": _D, "relationOrder": _OBJECTS,
                         "parentCandidateId": _NS, "overlapNotes": _S},
    "depthComposite": {"compositeId": _S, "sourceRevision": _I, "layerManifest": _OBJECTS,
                       "contentAssetId": _NS, "contentHash": _NS, "completePlateLayerId": _NS, "metadata": _D},
    "guide": {"guideId": _S, "name": _S, "lifecycle": _S, "createdRevision": _I, "stateRevision": _I,
              "supersedesId": _NS, "replacementId": _NS, "semanticRole": _NS, "objectIds": _STRINGS, "metadata": _D},
    "bbOperation": {"operationId": _S, "schemaVersion": _I, "sourceDocumentId": _S, "sourceRevision": _I,
                    "sourceCompositeId": _S, "sourceLayerManifest": _OBJECTS, "contextMaskId": _S, "contextMaskHash": _S,
                    "sourceBbox": _D, "sourceCropDimensions": _INTEGERS, "nativeTargetSize": _INTEGERS, "cropToNative": _D,
                    "nativeToCrop": _D, "nativeBBAssetId": _S, "generationMaskId": _NS, "generationMaskHash": _NS,
                    "guidanceIds": _STRINGS, "blendMaskId": _NS, "blendPolicy": _D, "outpaint": (_D, type(None)),
                    "candidateIds": _STRINGS, "extractionIds": _STRINGS, "state": _S, "staleReason": _NS, "operationRevision": _I},
    "candidate": {"candidateId": _S, "kind": _S, "operationId": _S, "sourceRevision": _I, "fullCrop": _B,
                  "blendMaskId": _NS, "placementTransform": _D, "status": _S, "assetId": _NS,
                  "parentCandidateId": _NS, "metadata": _D},
    "extractionDerivative": {"derivativeId": _S, "parentCandidateId": _S, "operationId": _S,
                              "extractionMatteId": _S, "placementTransform": _D, "sourceRevision": _I,
                              "rgbaAssetId": _S, "blendStitchMetadata": _D, "status": _S, "metadata": _D},
    "correctivePatch": {"patchId": _S, "parentId": _S, "transform": _L, "revision": _I, "maskId": _NS,
                        "visible": _B, "externalExchangeId": _NS, "metadata": _D},
    "externalRoundTrip": {"exchangeId": _S, "documentId": _S, "targetIds": _L, "exportedRevision": _I,
                          "coordinateSpace": _S, "cropOrigin": (_INTEGERS, type(None)), "bbox": (_D, type(None)),
                          "transform": _D, "exportAssetHash": _S, "importMode": _S, "metadata": _D},
    "privateProxy": {"proxyId": _S, "proxyAssetId": _S, "proxyContentHash": _S, "destinationSlotId": _S,
                     "registration": _D, "permittedDepthMetadata": _D, "coverageMatteId": _NS},
    "operation": {"operationId": _S, "operationType": _S, "schemaVersion": _I, "inputRefs": _OBJECTS,
                  "settings": _D, "producedIds": _STRINGS, "actorId": _S, "state": _S, "parentIds": _STRINGS, "childIds": _STRINGS},
    "document": {"formatId": _S, "schemaVersion": _I, "documentId": _S, "width": _I, "height": _I,
                 "colorSpaceIntent": _S, "currentRevision": _I, "rootLayerIds": _STRINGS,
                 "layers": _D, "objects": _D, "assets": _D, "masks": _D, "selections": _D,
                 "variants": _D, "interactionGroups": _D, "operations": _D, "depthComposites": _D,
                 "bbOperations": _D, "candidates": _D, "extractions": _D, "guides": _D,
                 "patches": _D, "externalRoundTrips": _D, "privateProxies": _D, "visualProfile": _D,
                 "history": _OBJECTS, "historyRefs": _STRINGS, "checkpointRefs": _STRINGS, "metadata": _D,
                 "revisionDigest": _NS},
}


def _matches_type(value: Any, expected: Any) -> bool:
    if isinstance(expected, tuple) and expected and expected[0] == "array":
        return isinstance(value, list) and all(_matches_type(item, expected[1]) for item in value)
    if isinstance(expected, tuple) and expected and expected[0] == "object-values":
        return isinstance(value, dict) and all(isinstance(k, str) and _matches_type(v, expected[1]) for k, v in value.items())
    if expected is int:
        return type(value) is int
    if expected is bool:
        return type(value) is bool
    if expected is float:
        return type(value) in (int, float)
    if expected == (int, float):
        return type(value) in (int, float)
    if isinstance(expected, tuple):
        return any(_matches_type(value, option) for option in expected)
    return isinstance(value, expected)


def _validate_json_value(value: Any, field_name: str = "value") -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        from math import isfinite
        if not isfinite(value):
            raise SchemaValidationError(f"{field_name} contains a non-finite number")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item, field_name)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaValidationError(f"{field_name} contains a non-string object key")
            _validate_json_value(item, field_name)
        return
    raise SchemaValidationError(f"{field_name} contains a non-JSON value")


def _strict(data: Mapping[str, Any], allowed: set[str], required: set[str], name: str) -> None:
    if not isinstance(data, Mapping):
        raise SchemaValidationError(f"{name} must be a JSON object")
    keys = set(data)
    missing = required - keys
    unknown = keys - allowed
    if missing:
        raise SchemaValidationError(f"{name} missing fields: {sorted(missing)}")
    if unknown:
        raise SchemaValidationError(f"{name} has unknown fields: {sorted(unknown)}")
    for field_name, expected in _IDOCUMENT_TYPES[name].items():
        if field_name in data and not _matches_type(data[field_name], expected):
            raise SchemaValidationError(f"{name}.{field_name} has an incompatible JSON type")
    _validate_json_value(dict(data), name)


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
        _id(self.actor_id, "actorId")
        _id(self.transaction_id, "transactionId")
        if not isinstance(self.body, str) or not self.body:
            raise SchemaValidationError("handoff body is required")
        if self.actor_kind not in {"human", "director", "agent", "system"}:
            raise SchemaValidationError("invalid handoff actor kind")
        if self.visibility not in {"shared", "director-only"}:
            raise SchemaValidationError("invalid handoff visibility")
        if self.state not in {"open", "acknowledged", "resolved", "orphaned"}:
            raise SchemaValidationError("invalid handoff state")
        if type(self.created_at_revision) is not int or self.created_at_revision < 0:
            raise SchemaValidationError("handoff body and revision are required")
        _ids(self.target_ids, "targetIds")
        if not isinstance(self.target_revisions, dict) or any(not isinstance(key, str) for key in self.target_revisions):
            raise SchemaValidationError("handoff target revisions must be an object")
        if set(self.target_revisions) != set(self.target_ids):
            raise SchemaValidationError("handoff target revision binding is incomplete")
        if any(type(revision) is not int or revision < 0 for revision in self.target_revisions.values()):
            raise SchemaValidationError("handoff target revisions must be non-negative")
        if self.state == "orphaned" and not self.orphan_reason:
            raise SchemaValidationError("orphaned handoff note requires an orphan reason")
        if self.state != "orphaned" and self.orphan_reason is not None:
            raise SchemaValidationError("only orphaned handoff notes may carry an orphan reason")

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
        if not isinstance(value["targetRevisions"], dict):
            raise SchemaValidationError("handoff targetRevisions must be an object")
        note = cls(_id(value["noteId"], "noteId"), value["actorId"], value["actorKind"],
                   value["body"], value["createdAtRevision"], _id(value["transactionId"], "transactionId"),
                   _ids(value["targetIds"], "targetIds"), dict(value["targetRevisions"]),
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
            if (not isinstance(self.last_editor["actorId"], str) or type(self.last_editor["atRevision"]) is not int
                    or self.last_editor["atRevision"] < 0):
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
        if not isinstance(self.media_type, str) or not self.media_type or not isinstance(self.storage_uri, str) or not self.storage_uri:
            raise SchemaValidationError("asset media type and storage URI are required")
        if self.byte_length is not None and (type(self.byte_length) is not int or self.byte_length < 0):
            raise SchemaValidationError("asset byte length cannot be negative")
        for value in (self.width, self.height):
            if value is not None and (type(value) is not int or value <= 0):
                raise SchemaValidationError("asset dimensions must be positive")
        if not isinstance(self.extension, str) or not self.extension or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789" for character in self.extension):
            raise SchemaValidationError("asset extension must be a lowercase alphanumeric suffix")
        if not isinstance(self.provenance, dict):
            raise SchemaValidationError("asset provenance must be an object")
        if self.external_uri is not None:
            if not isinstance(self.external_uri, str) or not self.external_uri:
                raise SchemaValidationError("external URI must be a non-empty string")
            if self.expected_hash is None:
                raise SchemaValidationError("external asset requires expected hash")
            _digest(self.expected_hash, "expectedHash")
            if self.content_hash != self.expected_hash:
                raise SchemaValidationError("external asset content and expected hashes must agree")
            if self.external_status not in {"missing", "available", "relinked", "embedded"}:
                raise SchemaValidationError("invalid external asset status")
            if self.external_status == "embedded":
                raise SchemaValidationError("embedded assets cannot retain an external source URI")
        else:
            expected_uri = f"assets/sha256/{self.content_hash[:2]}/{self.content_hash}.{self.extension}"
            if self.storage_uri.replace("\\", "/") != expected_uri:
                raise SchemaValidationError("embedded asset URI must be its canonical content-addressed path")
            if self.expected_hash is not None or self.external_status is not None:
                raise SchemaValidationError("embedded asset cannot carry external verification fields")

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
        if type(self.visible) is not bool or type(self.locked) is not bool or type(self.depth_element) is not bool:
            raise SchemaValidationError("layer visibility, lock, and depth flags must be booleans")
        if type(self.opacity) not in (int, float) or not isinstance(self.blend_mode, str):
            raise SchemaValidationError("layer opacity/blend values have incompatible types")
        if self.parent_id is not None:
            _id(self.parent_id, "parentId")
        _ids(self.child_ids, "childIds")
        _ids(self.object_ids, "objectIds")
        _ids(self.asset_ids, "assetIds")
        _ids(self.mask_ids, "maskIds")
        _ids(self.clipping_refs, "clippingRefs")
        if not 0.0 <= float(self.opacity) <= 1.0:
            raise SchemaValidationError("layer opacity must be between 0 and 1")
        if type(self.revision) is not int or self.revision < 0:
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
        if type(self.revision) is not int or self.revision < 0:
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
        if type(self.revision) is not int or self.revision < 0 or (self.asset_id is None and self.editable_source is None):
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
    invalidation_history: list[dict[str, Any]] = field(default_factory=list)
    rebase_history: list[dict[str, Any]] = field(default_factory=list)
    duplicate_history: list[dict[str, Any]] = field(default_factory=list)

    def validate(self) -> None:
        _id(self.selection_id, "selectionId")
        _id(self.source_id, "sourceId")
        _id(self.mask_id, "maskId")
        _ids(self.derived_layer_ids, "derivedLayerIds")
        _digest(self.source_content_digest, "sourceContentDigest")
        if (type(self.source_revision) is not int or type(self.selection_revision) is not int
                or self.source_revision < 0 or self.selection_revision < 1):
            raise SchemaValidationError("selection revisions are invalid")
        if self.state not in {state.value for state in SelectionState}:
            raise SchemaValidationError("invalid selection state")
        if self.bounds is not None:
            self.bounds.validate()
        if self.mapping is not None:
            _coordinate_transform(self.mapping)
        if self.state != SelectionState.CURRENT and not self.stale_reason:
            raise SchemaValidationError("stale selection requires a reason")
        if self.state == SelectionState.CURRENT and self.stale_reason is not None:
            raise SchemaValidationError("current selection cannot retain a stale reason")
        if self.stale_reason is not None and self.stale_reason not in {
                "SOURCE_CONTENT_CHANGED", "SOURCE_CLIP_CHANGED", "SOURCE_TRANSFORM_CHANGED", "SOURCE_REMOVED"}:
            raise SchemaValidationError("invalid selection staleness reason")
        if self.current_source_content_digest is not None:
            _digest(self.current_source_content_digest, "currentSourceContentDigest")
        if (not isinstance(self.refinement_history, list) or not isinstance(self.invalidation_history, list)
                or not isinstance(self.rebase_history, list) or not isinstance(self.duplicate_history, list)):
            raise SchemaValidationError("selection provenance must be arrays")
        last_source_revision = (self.invalidation_history[0].get("previousSourceRevision")
                                if self.invalidation_history and isinstance(self.invalidation_history[0], dict)
                                else self.source_revision)
        for entry in self.invalidation_history:
            if (not isinstance(entry, dict) or set(entry) != {"previousSourceRevision", "resultingSourceRevision", "reason"}
                    or type(entry.get("previousSourceRevision")) is not int or type(entry.get("resultingSourceRevision")) is not int
                    or entry["previousSourceRevision"] != last_source_revision
                    or entry["resultingSourceRevision"] <= last_source_revision
                    or entry["reason"] not in {"SOURCE_CONTENT_CHANGED", "SOURCE_CLIP_CHANGED", "SOURCE_TRANSFORM_CHANGED", "SOURCE_REMOVED"}):
                raise SchemaValidationError("selection invalidation lineage is malformed or non-monotonic")
            last_source_revision = entry["resultingSourceRevision"]
        for entry in self.refinement_history:
            if (not isinstance(entry, dict) or set(entry) != {"previousSelectionRevision", "selectionRevision", "previousMaskId", "newMaskId", "sourceRevision", "details"}
                    or any(type(entry.get(field)) is not int for field in ("previousSelectionRevision", "selectionRevision", "sourceRevision"))
                    or entry["selectionRevision"] != entry["previousSelectionRevision"] + 1
                    or not isinstance(entry["details"], dict)):
                raise SchemaValidationError("selection refinement lineage is malformed")
            _id(entry["previousMaskId"], "previousMaskId")
            _id(entry["newMaskId"], "newMaskId")
        for entry in self.rebase_history:
            fields = {"previousSelectionRevision", "selectionRevision", "previousSourceRevision", "sourceRevision",
                      "previousMaskId", "maskId", "reason", "geometryOnly"}
            if (not isinstance(entry, dict) or set(entry) != fields
                    or any(type(entry.get(field)) is not int for field in ("previousSelectionRevision", "selectionRevision", "previousSourceRevision", "sourceRevision"))
                    or type(entry["geometryOnly"]) is not bool
                    or entry["selectionRevision"] != entry["previousSelectionRevision"] + 1):
                raise SchemaValidationError("selection rebase lineage is malformed")
            _id(entry["previousMaskId"], "previousMaskId")
            _id(entry["maskId"], "maskId")
        for entry in self.duplicate_history:
            if (not isinstance(entry, dict) or set(entry) != {"previousSelectionRevision", "selectionRevision", "layerId"}
                    or type(entry.get("previousSelectionRevision")) is not int or type(entry.get("selectionRevision")) is not int
                    or entry["selectionRevision"] != entry["previousSelectionRevision"] + 1):
                raise SchemaValidationError("selection duplicate-link lineage is malformed")
            _id(entry["layerId"], "layerId")
        if self.state == SelectionState.CURRENT and self.invalidation_history and last_source_revision != self.source_revision:
            raise SchemaValidationError("current selection source revision does not match its invalidation lineage")
        if self.state != SelectionState.CURRENT and self.invalidation_history:
            if last_source_revision <= self.source_revision or self.invalidation_history[-1]["reason"] != self.stale_reason:
                raise SchemaValidationError("stale selection state does not match its latest invalidation")

    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> "SelectionRecord":
        """Create a selection at its initial selection revision (one)."""

        positional = list(args)
        if len(positional) > 4:
            positional[4] = 1
            kwargs.pop("selection_revision", None)
        else:
            kwargs["selection_revision"] = 1
        if len(positional) > 5:
            if positional[5] != SelectionState.CURRENT.value:
                raise SchemaValidationError("new selections must begin current")
        else:
            kwargs.setdefault("state", SelectionState.CURRENT.value)
        result = cls(*positional, **kwargs)
        result.validate()
        return result

    def mark_source_changed(self, reason: str, source_revision: int, source_content_digest: str | None = None) -> None:
        """Mark a dependent selection stale without rewriting its mask lineage."""

        previous_source_revision = (self.invalidation_history[-1]["resultingSourceRevision"]
                                    if self.invalidation_history else self.source_revision)
        if type(source_revision) is not int or source_revision <= previous_source_revision:
            raise SchemaValidationError("source invalidation must identify a newer integer revision")
        validated_digest = None if source_content_digest is None else _digest(source_content_digest, "currentSourceContentDigest")
        if reason == "SOURCE_CONTENT_CHANGED" and validated_digest is None:
            raise SchemaValidationError("content invalidation requires the new source content digest")
        if reason == "SOURCE_TRANSFORM_CHANGED":
            self.state = SelectionState.NEEDS_REBASE.value
        elif reason in {"SOURCE_CONTENT_CHANGED", "SOURCE_CLIP_CHANGED", "SOURCE_REMOVED"}:
            self.state = SelectionState.STALE_SOURCE.value
        else:
            raise SchemaValidationError(f"unknown selection invalidation reason: {reason}")
        self.stale_reason = reason
        self.invalidation_history.append({
            "previousSourceRevision": previous_source_revision,
            "resultingSourceRevision": source_revision,
            "reason": reason,
        })
        if validated_digest is not None:
            self.current_source_content_digest = validated_digest

    def _check_selection_revision(self, expected_selection_revision: int) -> None:
        if type(expected_selection_revision) is not int:
            raise SchemaValidationError("expected selection revision must be an integer")
        if expected_selection_revision != self.selection_revision:
            raise SelectionRevisionConflict(expected_selection_revision, self.selection_revision)

    def register_derived_layer(self, layer_id: str, *, expected_selection_revision: int) -> int:
        """Register a duplicate-link mutation and advance selection revision."""

        self.validate()
        self._check_selection_revision(expected_selection_revision)
        if self.state != SelectionState.CURRENT.value:
            raise SchemaValidationError("cannot duplicate a stale selection")
        layer_id = _id(layer_id, "derivedLayerId")
        if layer_id in self.derived_layer_ids:
            raise SchemaValidationError("derived layer is already registered")
        previous_revision = self.selection_revision
        self.derived_layer_ids.append(layer_id)
        self.selection_revision += 1
        self.duplicate_history.append({
            "previousSelectionRevision": previous_revision,
            "selectionRevision": self.selection_revision,
            "layerId": layer_id,
        })
        return self.selection_revision

    def refine(
        self,
        new_mask_id: str,
        *,
        expected_selection_revision: int,
        refinement: Mapping[str, Any] | None = None,
    ) -> int:
        """Record a mask refinement as a compare-and-swap selection mutation."""

        self.validate()
        self._check_selection_revision(expected_selection_revision)
        if self.state != SelectionState.CURRENT.value:
            raise SchemaValidationError("cannot refine a stale selection")
        new_mask_id = _id(new_mask_id, "newMaskId")
        if new_mask_id == self.mask_id:
            raise SchemaValidationError("refinement must identify a new mask")
        material = dict(refinement or {})
        # Validate JSON safety before changing the selection record.
        from .ids import canonical_json
        try:
            canonical_json(material)
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError(f"refinement metadata must be finite JSON data: {exc}") from exc
        previous_mask_id = self.mask_id
        next_revision = self.selection_revision + 1
        self.refinement_history.append({
            "previousSelectionRevision": self.selection_revision,
            "selectionRevision": next_revision,
            "previousMaskId": previous_mask_id,
            "newMaskId": new_mask_id,
            "sourceRevision": self.source_revision,
            "details": material,
        })
        self.mask_id = new_mask_id
        self.selection_revision = next_revision
        return next_revision

    def explicit_rebase(
        self,
        target_source_revision: int,
        *,
        expected_selection_revision: int,
        new_mask_id: str | None = None,
        source_content_digest: str | None = None,
        geometry_only: bool = False,
    ) -> int:
        """Resolve a stale/rebased selection only through an explicit command."""

        self.validate()
        self._check_selection_revision(expected_selection_revision)
        if type(target_source_revision) is not int or target_source_revision < 0:
            raise SchemaValidationError("target source revision must be a non-negative integer")
        if self.state == SelectionState.CURRENT.value:
            raise SchemaValidationError("only a stale selection can be explicitly rebased")
        if self.invalidation_history and target_source_revision != self.invalidation_history[-1]["resultingSourceRevision"]:
            raise SchemaValidationError("rebase target must match the latest source invalidation")
        reason = self.stale_reason
        if geometry_only:
            if reason != "SOURCE_TRANSFORM_CHANGED" or new_mask_id is not None or source_content_digest is not None:
                raise SchemaValidationError("geometry-only rebase is valid only for transform staleness and preserves its mask")
        else:
            if reason == "SOURCE_REMOVED":
                raise SchemaValidationError("a removed source cannot be silently rebased")
            if reason in {"SOURCE_CONTENT_CHANGED", "SOURCE_CLIP_CHANGED"} and new_mask_id is None:
                raise SchemaValidationError("content/clip rebase requires a newly derived mask")
        next_mask_id = self.mask_id if new_mask_id is None else _id(new_mask_id, "maskId")
        if not geometry_only and next_mask_id == self.mask_id and reason in {"SOURCE_CONTENT_CHANGED", "SOURCE_CLIP_CHANGED"}:
            raise SchemaValidationError("content/clip rebase must preserve the old mask and bind a distinct new mask")
        next_digest = self.source_content_digest if source_content_digest is None else _digest(source_content_digest, "sourceContentDigest")
        if reason == "SOURCE_CONTENT_CHANGED" and source_content_digest is None:
            raise SchemaValidationError("content rebase requires the new source content digest")
        old_revision = self.source_revision
        old_mask_id = self.mask_id
        next_selection_revision = self.selection_revision + 1
        self.rebase_history.append({
            "previousSelectionRevision": self.selection_revision,
            "selectionRevision": next_selection_revision,
            "previousSourceRevision": old_revision,
            "sourceRevision": target_source_revision,
            "previousMaskId": old_mask_id,
            "maskId": next_mask_id,
            "reason": reason,
            "geometryOnly": geometry_only,
        })
        self.mask_id = next_mask_id
        if source_content_digest is not None:
            self.source_content_digest = next_digest
        self.source_revision = target_source_revision
        self.state = SelectionState.CURRENT.value
        self.stale_reason = None
        self.current_source_content_digest = None
        self.selection_revision = next_selection_revision
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
                "staleReason": self.stale_reason, "currentSourceContentDigest": self.current_source_content_digest,
                "invalidationHistory": self.invalidation_history, "rebaseHistory": self.rebase_history,
                "duplicateHistory": self.duplicate_history}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SelectionRecord":
        allowed = {"selectionId", "sourceId", "sourceRevision", "sourceContentDigest", "selectionRevision", "state", "maskId", "coordinateSpace",
                   "seedGeometry", "semanticHint", "bounds", "mapping", "refinementHistory", "derivedLayerIds", "staleReason", "currentSourceContentDigest",
                   "invalidationHistory", "rebaseHistory", "duplicateHistory"}
        _strict(value, allowed, allowed - {"invalidationHistory", "rebaseHistory", "duplicateHistory"}, "selection")
        result = cls(_id(value["selectionId"], "selectionId"), _id(value["sourceId"], "sourceId"), int(value["sourceRevision"]),
                     _digest(value["sourceContentDigest"], "sourceContentDigest"), int(value["selectionRevision"]), str(value["state"]),
                     _id(value["maskId"], "maskId"), str(value["coordinateSpace"]), list(value["seedGeometry"]), value["semanticHint"],
                     None if value["bounds"] is None else _bbox(value["bounds"]),
                     None if value["mapping"] is None else _coordinate_transform(value["mapping"]), list(value["refinementHistory"]),
                     _ids(value["derivedLayerIds"], "derivedLayerIds"), value["staleReason"], value["currentSourceContentDigest"],
                     list(value.get("invalidationHistory", [])), list(value.get("rebaseHistory", [])),
                     list(value.get("duplicateHistory", [])))
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
        if type(self.source_revision) is not int or self.source_revision < 0 or not self.layer_manifest:
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
        if (type(self.created_revision) is not int or type(self.state_revision) is not int
                or self.created_revision < 0 or self.state_revision < self.created_revision):
            raise SchemaValidationError("guide revisions are invalid")
        if self.supersedes_id is not None:
            _id(self.supersedes_id, "supersedesId")
        if self.replacement_id is not None:
            _id(self.replacement_id, "replacementId")
        _ids(self.object_ids, "objectIds")
        if self.lifecycle in {GuideLifecycle.REPLACEMENT_PENDING.value, GuideLifecycle.SUPERSEDED.value} and self.replacement_id is None:
            raise SchemaValidationError(f"{self.lifecycle} guide requires a replacement guide ID")

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
        if type(revision) is not int or revision < self.state_revision:
            raise SchemaValidationError("guide lifecycle revisions must be monotonic")
        if lifecycle in {GuideLifecycle.REPLACEMENT_PENDING.value, GuideLifecycle.SUPERSEDED.value} and replacement_id is None and self.replacement_id is None:
            raise SchemaValidationError(f"{lifecycle} transition requires a replacement guide ID")
        next_replacement_id = self.replacement_id if replacement_id is None else _id(replacement_id, "replacementId")
        self.lifecycle = lifecycle
        self.state_revision = revision
        self.replacement_id = next_replacement_id
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
        if (type(self.schema_version) is not int or type(self.source_revision) is not int
                or type(self.operation_revision) is not int or self.source_revision < 0 or self.operation_revision < 1):
            raise SchemaValidationError("BB revisions are invalid")
        if len(self.source_crop_dimensions) != 2 or any(type(v) is not int or v <= 0 for v in self.source_crop_dimensions):
            raise SchemaValidationError("source crop dimensions are invalid")
        if self.source_crop_dimensions != (self.source_bbox.width, self.source_bbox.height):
            raise SchemaValidationError("source crop dimensions do not match the half-open source bbox")
        if len(self.native_target_size) != 2 or any(type(v) is not int or v <= 0 for v in self.native_target_size):
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
        if self.native_to_crop.source_dimensions != self.native_target_size:
            raise SchemaValidationError("BB inverse source dimensions do not mirror the forward target")
        if self.native_to_crop.target_dimensions != self.source_crop_dimensions:
            raise SchemaValidationError("BB inverse target dimensions do not mirror the forward source")
        if (self.crop_to_native.operation_revision != self.operation_revision
                or self.native_to_crop.operation_revision != self.operation_revision):
            raise SchemaValidationError("BB transform operation revision does not match its operation")
        identity = AffineTransform.identity().matrix
        for product in (self.crop_to_native.forward.compose(self.native_to_crop.forward),
                        self.native_to_crop.forward.compose(self.crop_to_native.forward)):
            if any(abs(actual - expected) > 1e-8 for actual, expected in zip(product.matrix, identity)):
                raise SchemaValidationError("BB forward and reverse transforms are not reciprocal")
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
        if type(self.source_revision) is not int or self.source_revision < 0:
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
        if type(self.source_revision) is not int or self.source_revision < 0:
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
        if type(self.revision) is not int or self.revision < 0:
            raise SchemaValidationError("patch revision invalid")
        if type(self.visible) is not bool:
            raise SchemaValidationError("patch visibility must be a boolean")

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
        if type(self.exported_revision) is not int or self.exported_revision < 0:
            raise SchemaValidationError("external export revision must be a non-negative integer")
        if self.coordinate_space not in {"document", "source-crop", "native-bb", "model", "preview"}:
            raise SchemaValidationError("invalid external round-trip coordinate space")
        if self.crop_origin is not None and (len(self.crop_origin) != 2 or any(type(v) is not int for v in self.crop_origin)):
            raise SchemaValidationError("external round-trip crop origin must contain two integers")
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

    DEPTH_METADATA_FIELDS: ClassVar[dict[str, Any]] = {
        "depthRange": dict,
        "depthEncoding": str,
        "registrationQuality": (int, float),
        "registrationMethod": str,
        "coordinateSpace": str,
        "sourceDimensions": list,
        "targetDimensions": list,
        "offset": list,
        "scale": list,
    }

    def validate(self) -> None:
        for value, name in ((self.proxy_id, "proxyId"), (self.proxy_asset_id, "proxyAssetId"), (self.destination_slot_id, "destinationSlotId")):
            _id(value, name)
        _digest(self.proxy_content_hash, "proxyContentHash")
        _coordinate_transform(self.registration)
        if self.coverage_matte_id is not None:
            _id(self.coverage_matte_id, "coverageMatteId")
        if not isinstance(self.permitted_depth_metadata, dict):
            raise SchemaValidationError("private proxy depth metadata must be an object")
        if set(self.permitted_depth_metadata) - set(self.DEPTH_METADATA_FIELDS):
            raise SchemaValidationError("private proxy depth metadata contains a field outside its typed allowlist")
        for key, value in self.permitted_depth_metadata.items():
            expected = self.DEPTH_METADATA_FIELDS[key]
            if expected == (int, float):
                valid = type(value) in (int, float)
            else:
                valid = isinstance(value, expected)
            if (not valid or isinstance(value, dict) and key != "depthRange"
                    or isinstance(value, list) and key not in {"sourceDimensions", "targetDimensions", "offset", "scale"}):
                raise SchemaValidationError(f"private proxy metadata field {key} has an incompatible type")
            if key == "depthRange":
                if set(value) != {"near", "far"} or any(type(value[item]) not in (int, float) for item in ("near", "far")):
                    raise SchemaValidationError("depthRange must contain numeric near and far values only")
                if value["far"] <= value["near"]:
                    raise SchemaValidationError("depthRange far must exceed near")
            elif key in {"sourceDimensions", "targetDimensions"}:
                if len(value) != 2 or any(type(part) is not int or part <= 0 for part in value):
                    raise SchemaValidationError(f"{key} must contain two positive integer dimensions")
            elif key in {"offset", "scale"}:
                if len(value) != 2 or any(type(part) not in (int, float) for part in value):
                    raise SchemaValidationError(f"{key} must contain two numeric values")
                if key == "scale" and any(part == 0 for part in value):
                    raise SchemaValidationError("proxy registration scale cannot be zero")
            elif key == "registrationQuality" and not 0.0 <= value <= 1.0:
                raise SchemaValidationError("registration quality must be in [0, 1]")
            elif key == "registrationMethod" and value not in {"affine", "rigid", "manual", "depth-aligned"}:
                raise SchemaValidationError("unknown proxy registration method")
            elif key == "coordinateSpace" and value not in {"document", "source-crop", "native-bb", "model", "preview"}:
                raise SchemaValidationError("unknown proxy metadata coordinate space")
            elif key == "depthEncoding" and value not in {"normalized", "metric", "relative"}:
                raise SchemaValidationError("unknown proxy depth encoding")

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
        if (not isinstance(self.operation_type, str) or not self.operation_type or type(self.schema_version) is not int
                or self.schema_version < 1):
            raise SchemaValidationError("operation type and version are required")
        _id(self.actor_id, "actorId")
        if not isinstance(self.input_refs, list) or not all(isinstance(item, dict) for item in self.input_refs):
            raise SchemaValidationError("operation inputRefs must be an array of typed objects")
        if not isinstance(self.settings, dict) or not isinstance(self.state, str):
            raise SchemaValidationError("operation settings/state have incompatible types")
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
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise SchemaValidationError(f"unsupported schema version: {self.schema_version}")
        if (type(self.width) is not int or type(self.height) is not int or type(self.current_revision) is not int
                or self.width <= 0 or self.height <= 0 or self.current_revision < 0):
            raise SchemaValidationError("document dimensions or revision are invalid")
        _ids(self.root_layer_ids, "rootLayerIds")
        if not isinstance(self.color_space_intent, str) or not self.color_space_intent:
            raise SchemaValidationError("document color-space intent is required")
        if not all(isinstance(value, dict) for value in (self.layers, self.objects, self.assets, self.masks,
                self.selections, self.variants, self.interaction_groups, self.operations, self.depth_composites,
                self.bb_operations, self.candidates, self.extractions, self.guides, self.patches,
                self.external_round_trips, self.private_proxies, self.metadata)):
            raise SchemaValidationError("document record collections and metadata must be objects")
        for key, record in self.layers.items():
            if key != record.layer_id:
                raise SchemaValidationError("layer map key does not match layerId")
            record.validate()
            if record.revision > self.current_revision:
                raise SchemaValidationError("layer revision exceeds the document revision")
            if any(object_id not in self.objects for object_id in record.object_ids):
                raise SchemaValidationError("layer references unknown object")
            if any(asset_id not in self.assets for asset_id in record.asset_ids):
                raise SchemaValidationError("layer references unknown asset")
            if any(mask_id not in self.masks for mask_id in record.mask_ids):
                raise SchemaValidationError("layer references unknown mask")
            if any(ref not in self.layers and ref not in self.objects and ref not in self.masks for ref in record.clipping_refs):
                raise SchemaValidationError("layer clippingRefs contains a dangling reference")
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
            if type(record.revision) is not int or record.revision < 0 or record.revision > self.current_revision:
                raise SchemaValidationError("object revision is outside the document revision range")
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
            if record.revision > self.current_revision:
                raise SchemaValidationError("mask revision exceeds the document revision")
        for key, record in self.selections.items():
            if key != record.selection_id:
                raise SchemaValidationError("selection map key does not match selectionId")
            record.validate()
            if record.source_id not in self.layers and record.source_id not in self.depth_composites:
                raise SchemaValidationError("selection references unknown source")
            if record.source_revision > self.current_revision:
                raise SchemaValidationError("selection source revision exceeds the document revision")
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
        # Every identity is globally unique. A typed ID may only identify one
        # durable record, even when two record classes use the same field name.
        record_collections = {
            "layer": self.layers, "object": self.objects, "asset": self.assets, "mask": self.masks,
            "selection": self.selections, "variant": self.variants, "interactionGroup": self.interaction_groups,
            "operation": self.operations, "depthComposite": self.depth_composites, "bbOperation": self.bb_operations,
            "candidate": self.candidates, "extraction": self.extractions, "guide": self.guides,
            "patch": self.patches, "externalRoundTrip": self.external_round_trips, "privateProxy": self.private_proxies,
        }
        identity_owners: dict[str, str] = {}
        for collection_name, collection in record_collections.items():
            for record_id in collection:
                if record_id in identity_owners:
                    raise SchemaValidationError(f"duplicate durable identity {record_id!r} in {collection_name} and {identity_owners[record_id]}")
                identity_owners[record_id] = collection_name

        # Ownership is reciprocal: object/layer and layer/mask lists cannot
        # disagree with the child record's canonical owner.
        for layer in self.layers.values():
            for object_id in layer.object_ids:
                if self.objects[object_id].layer_id != layer.layer_id:
                    raise SchemaValidationError("layer object membership contradicts the object's owning layer")
            for mask_id in layer.mask_ids:
                if self.masks[mask_id].owner_id != layer.layer_id:
                    raise SchemaValidationError("layer mask membership contradicts the mask owner")
            if layer.depth_element and layer.complete_plate_id == layer.layer_id:
                raise SchemaValidationError("depth element cannot be its own complete plate")
        for object_record in self.objects.values():
            if object_record.object_id not in self.layers[object_record.layer_id].object_ids:
                raise SchemaValidationError("object owner does not list the object in objectIds")
        for mask in self.masks.values():
            if mask.owner_id in self.layers and mask.mask_id not in self.layers[mask.owner_id].mask_ids:
                raise SchemaValidationError("mask owner does not list the mask in maskIds")

        for composite in self.depth_composites.values():
            if type(composite.source_revision) is not int or composite.source_revision > self.current_revision:
                raise SchemaValidationError("depth composite source revision exceeds the document revision")
            if (composite.content_asset_id is None) != (composite.content_hash is None):
                raise SchemaValidationError("depth composite content asset and hash must be paired")
            if composite.content_asset_id is not None:
                asset = self.assets.get(composite.content_asset_id)
                if asset is None or asset.content_hash != composite.content_hash:
                    raise SchemaValidationError("depth composite content asset/hash reference is inconsistent")
            if composite.complete_plate_layer_id is not None and composite.complete_plate_layer_id not in self.layers:
                raise SchemaValidationError("depth composite references an unknown complete plate layer")
            for item in composite.layer_manifest:
                if not isinstance(item, Mapping) or not isinstance(item.get("layerId"), str) or item["layerId"] not in self.layers:
                    raise SchemaValidationError("depth composite layer manifest contains an unknown layer")
                if "revision" in item and (type(item["revision"]) is not int or item["revision"] > composite.source_revision):
                    raise SchemaValidationError("depth composite layer manifest has an impossible revision")
        for operation in self.bb_operations.values():
            if operation.source_document_id != self.document_id:
                raise SchemaValidationError("BB operation source document does not match its owning document")
            if operation.schema_version != self.schema_version or operation.source_revision > self.current_revision:
                raise SchemaValidationError("BB operation schema/source revision binding is invalid")
            try:
                operation.source_bbox.validate(width=self.width, height=self.height)
            except ValueError as exc:
                raise SchemaValidationError(f"BB source bbox is invalid: {exc}") from exc
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
            context_mask = self.masks[operation.context_mask_id]
            context_hash = context_mask.content_hash or (self.assets[context_mask.asset_id].content_hash if context_mask.asset_id else None)
            if context_hash != operation.context_mask_hash:
                raise SchemaValidationError("BB context mask identity/hash does not match its mask record")
            if operation.generation_mask_id is not None and operation.generation_mask_hash is not None:
                generation_mask = self.masks[operation.generation_mask_id]
                generation_hash = generation_mask.content_hash or (self.assets[generation_mask.asset_id].content_hash if generation_mask.asset_id else None)
                if generation_hash != operation.generation_mask_hash:
                    raise SchemaValidationError("BB generation mask identity/hash does not match its mask record")
            if any(guide_id not in self.guides for guide_id in operation.guidance_ids):
                raise SchemaValidationError("BB operation references unknown guide")
            for item in operation.source_layer_manifest:
                if not isinstance(item, Mapping) or not isinstance(item.get("layerId"), str) or item["layerId"] not in self.layers:
                    raise SchemaValidationError("BB source layer manifest contains an unknown layer")
                if type(item.get("revision")) is not int or item["revision"] > operation.source_revision:
                    raise SchemaValidationError("BB source layer manifest has an impossible revision")
        for candidate in self.candidates.values():
            if candidate.operation_id not in self.bb_operations and candidate.operation_id not in self.operations:
                raise SchemaValidationError("candidate references unknown operation")
            if candidate.blend_mask_id is not None and candidate.blend_mask_id not in self.masks:
                raise SchemaValidationError("candidate references unknown blend mask")
            if candidate.asset_id is not None and candidate.asset_id not in self.assets:
                raise SchemaValidationError("candidate references unknown asset")
            if candidate.parent_candidate_id is not None and candidate.parent_candidate_id not in self.candidates:
                raise SchemaValidationError("candidate references unknown parent candidate")
            if candidate.source_revision > self.current_revision:
                raise SchemaValidationError("candidate source revision exceeds the document revision")
            operation = self.bb_operations.get(candidate.operation_id)
            if operation is not None:
                if candidate.source_revision != operation.source_revision or candidate.candidate_id not in operation.candidate_ids:
                    raise SchemaValidationError("candidate and BB operation lineage is inconsistent")
            if candidate.status == "extracted" and not any(item.parent_candidate_id == candidate.candidate_id for item in self.extractions.values()):
                raise SchemaValidationError("extracted candidate has no registered extraction derivative")
        for extraction in self.extractions.values():
            if extraction.parent_candidate_id not in self.candidates:
                raise SchemaValidationError("extraction references unknown candidate")
            if extraction.operation_id not in self.bb_operations and extraction.operation_id not in self.operations:
                raise SchemaValidationError("extraction references unknown operation")
            if extraction.extraction_matte_id not in self.masks:
                raise SchemaValidationError("extraction references unknown matte")
            if extraction.rgba_asset_id not in self.assets:
                raise SchemaValidationError("extraction references unknown RGBA asset")
            candidate = self.candidates[extraction.parent_candidate_id]
            if (extraction.operation_id != candidate.operation_id or extraction.source_revision != candidate.source_revision
                    or extraction.derivative_id not in (self.bb_operations.get(extraction.operation_id).extraction_ids
                        if extraction.operation_id in self.bb_operations else [extraction.derivative_id])):
                raise SchemaValidationError("extraction lineage does not match its parent candidate/operation")
        for guide in self.guides.values():
            if guide.created_revision > self.current_revision or guide.state_revision > self.current_revision:
                raise SchemaValidationError("guide lifecycle revision exceeds the document revision")
            if any(object_id not in self.objects for object_id in guide.object_ids):
                raise SchemaValidationError("guide references unknown object")
            if guide.supersedes_id is not None:
                previous = self.guides.get(guide.supersedes_id)
                if previous is None or previous.replacement_id != guide.guide_id or previous.state_revision < guide.created_revision:
                    raise SchemaValidationError("guide supersession lineage does not resolve reciprocally")
            if guide.lifecycle in {GuideLifecycle.REPLACEMENT_PENDING.value, GuideLifecycle.SUPERSEDED.value}:
                replacement = self.guides.get(guide.replacement_id)
                if (replacement is None or replacement.supersedes_id != guide.guide_id
                        or replacement.created_revision < guide.created_revision or replacement.created_revision > guide.state_revision):
                    raise SchemaValidationError("guide replacement lineage is missing, cyclic, or revision-inconsistent")
        for guide in self.guides.values():
            seen_guides: set[str] = set()
            current = guide
            while current.supersedes_id is not None:
                if current.guide_id in seen_guides:
                    raise SchemaValidationError("guide supersession lineage contains a cycle")
                seen_guides.add(current.guide_id)
                current = self.guides[current.supersedes_id]

        candidate_parent_edges = {candidate.candidate_id: candidate.parent_candidate_id for candidate in self.candidates.values()}
        for candidate_id in candidate_parent_edges:
            seen_candidates: set[str] = set()
            cursor: str | None = candidate_id
            while cursor is not None:
                if cursor in seen_candidates:
                    raise SchemaValidationError("candidate parent lineage contains a cycle")
                seen_candidates.add(cursor)
                cursor = candidate_parent_edges[cursor]
        for patch in self.patches.values():
            if patch.parent_id not in self.layers and patch.parent_id not in self.objects and patch.parent_id not in self.interaction_groups:
                raise SchemaValidationError("patch references unknown parent")
            if patch.mask_id is not None and patch.mask_id not in self.masks:
                raise SchemaValidationError("patch references unknown mask")
            if patch.revision > self.current_revision:
                raise SchemaValidationError("patch revision exceeds the document revision")
            if patch.external_exchange_id is not None:
                exchange = self.external_round_trips.get(patch.external_exchange_id)
                if exchange is None or patch.patch_id not in exchange.target_ids:
                    raise SchemaValidationError("corrective patch exchange link does not resolve reciprocally")
        for exchange in self.external_round_trips.values():
            if exchange.document_id != self.document_id or exchange.exported_revision > self.current_revision:
                raise SchemaValidationError("external round-trip document/revision binding is invalid")
            if any(target_id not in self.layers and target_id not in self.objects and target_id not in self.patches for target_id in exchange.target_ids):
                raise SchemaValidationError("external round trip references unknown target")
            if not any(asset.content_hash == exchange.export_asset_hash for asset in self.assets.values()):
                raise SchemaValidationError("external round trip export hash has no registered asset identity")
        for proxy in self.private_proxies.values():
            if proxy.proxy_asset_id not in self.assets:
                raise SchemaValidationError("private proxy references unknown proxy asset")
            if self.assets[proxy.proxy_asset_id].content_hash != proxy.proxy_content_hash:
                raise SchemaValidationError("private proxy asset/hash identity is inconsistent")
            if proxy.coverage_matte_id is not None and proxy.coverage_matte_id not in self.masks:
                raise SchemaValidationError("private proxy references unknown coverage matte")
        try:
            transactions = validate_history(
                (TransactionRecord.from_dict(item) for item in self.history), self.current_revision
            )
        except (HistoryValidationError, TypeError, ValueError) as exc:
            raise SchemaValidationError(f"invalid durable history: {exc}") from exc
        transaction_ids = [record.transaction_id for record in transactions]
        if self.history_refs and self.history_refs != transaction_ids:
            raise SchemaValidationError("historyRefs do not match retained history")
        if len(set(self.checkpoint_refs)) != len(self.checkpoint_refs):
            raise SchemaValidationError("duplicate checkpoint reference")
        for reference in self.checkpoint_refs:
            path = PurePosixPath(reference.replace("\\", "/")) if isinstance(reference, str) else None
            if (path is None or "\\" in reference or path.is_absolute() or len(path.parts) != 3
                    or path.parts[0] != "checkpoints" or path.parts[2] != "manifest.json"
                    or not path.parts[1].isdigit() or int(path.parts[1]) != self.current_revision):
                raise SchemaValidationError("checkpoint reference must bind this revision's project-relative manifest")
        if self.history_refs:
            _ids(self.history_refs, "historyRefs")
        for layer in self.layers.values():
            if layer.complete_plate_id is not None and layer.complete_plate_id not in self.layers and layer.complete_plate_id not in self.assets:
                raise SchemaValidationError("depth layer references missing complete plate")
            if layer.depth_element and layer.complete_plate_id in self.layers and self.layers[layer.complete_plate_id].depth_element:
                raise SchemaValidationError("depth element cannot claim another depth element as its complete plate")

        transaction_by_id = {record.transaction_id: record for record in transactions}
        target_records = {**self.layers, **self.objects}
        for layer in self.layers.values():
            for note in layer.collaboration.handoff_notes:
                if note.created_at_revision > self.current_revision:
                    raise SchemaValidationError("handoff note revision exceeds the document revision")
                unresolved: list[str] = []
                transaction = transaction_by_id.get(note.transaction_id)
                if transaction is None or transaction.resulting_revision != note.created_at_revision:
                    unresolved.append("transaction binding no longer resolves")
                for target_id, revision in note.target_revisions.items():
                    target = target_records.get(target_id)
                    if target is None:
                        unresolved.append(f"target {target_id} no longer resolves")
                    elif revision > self.current_revision or revision != target.revision:
                        unresolved.append(f"target {target_id} revision no longer resolves")
                if unresolved and note.state != "orphaned":
                    raise SchemaValidationError("unresolved handoff binding must be explicitly orphaned: " + "; ".join(unresolved))
                if note.state == "orphaned" and not note.orphan_reason:
                    raise SchemaValidationError("orphaned handoff note requires an orphan reason")

        operation_types = {
            "layer": self.layers, "object": self.objects, "asset": self.assets, "mask": self.masks,
            "selection": self.selections, "variant": self.variants, "interactionGroup": self.interaction_groups,
            "operation": self.operations, "depthComposite": self.depth_composites, "bbOperation": self.bb_operations,
            "candidate": self.candidates, "extraction": self.extractions, "guide": self.guides,
            "patch": self.patches, "externalRoundTrip": self.external_round_trips, "privateProxy": self.private_proxies,
        }
        for operation in self.operations.values():
            for reference in operation.input_refs:
                if set(reference) == {"kind", "id"}:
                    kind, record_id = reference["kind"], reference["id"]
                elif set(reference) == {"recordType", "recordId"}:
                    kind, record_id = reference["recordType"], reference["recordId"]
                else:
                    raise SchemaValidationError("operation input reference must be a typed {kind,id} or {recordType,recordId} object")
                if kind not in operation_types or not isinstance(record_id, str) or record_id not in operation_types[kind]:
                    raise SchemaValidationError("operation input reference does not resolve to a typed record")
            if any(record_id not in identity_owners for record_id in operation.produced_ids):
                raise SchemaValidationError("operation producedIds contains an unknown record")
            if any(parent_id not in self.operations for parent_id in operation.parent_ids):
                raise SchemaValidationError("operation parentIds contains an unknown operation")
            if any(child_id not in self.operations for child_id in operation.child_ids):
                raise SchemaValidationError("operation childIds contains an unknown operation")
            if any(operation.operation_id not in self.operations[parent_id].child_ids for parent_id in operation.parent_ids):
                raise SchemaValidationError("operation parent/child links are not reciprocal")
            if any(operation.operation_id not in self.operations[child_id].parent_ids for child_id in operation.child_ids):
                raise SchemaValidationError("operation child/parent links are not reciprocal")

        for mask in self.masks.values():
            if mask.owner_id in self.bb_operations:
                operation = self.bb_operations[mask.owner_id]
                is_context = operation.context_mask_id == mask.mask_id
                is_generation = operation.generation_mask_id == mask.mask_id
                is_blend = operation.blend_mask_id == mask.mask_id
                is_extraction_matte = any(self.extractions[item].extraction_matte_id == mask.mask_id
                                          for item in operation.extraction_ids)
                if not (is_context or is_generation or is_blend or is_extraction_matte):
                    raise SchemaValidationError("mask owner BB operation does not reference the mask")
            elif mask.owner_id in self.operations and mask.mask_id not in self.operations[mask.owner_id].produced_ids:
                raise SchemaValidationError("mask owner operation does not list the mask as a produced record")

        for collection in record_collections.values():
            for record in collection.values():
                _validate_json_value(record.to_dict(), "record")
        _validate_json_value(self.visual_profile.to_dict(), "visualProfile")
        _validate_json_value(self.history, "history")
        _validate_json_value(self.metadata, "metadata")
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
            parse_map(value["layers"], LayerRecord.from_dict), parse_map(value["objects"], ObjectRecord.from_dict), parse_map(value["assets"], AssetRecord.from_dict), parse_map(value["masks"], MaskRecord.from_dict), parse_map(value["selections"], SelectionRecord.from_dict), parse_map(value["variants"], VariantSet.from_dict), parse_map(value["interactionGroups"], InteractionGroup.from_dict), parse_map(value["operations"], OperationRecord.from_dict), parse_map(value["depthComposites"], DepthComposite.from_dict), parse_map(value["bbOperations"], BBOperation.from_dict), parse_map(value["candidates"], CandidateRecord.from_dict), parse_map(value["extractions"], ExtractionDerivative.from_dict), parse_map(value["guides"], GuideRecord.from_dict), parse_map(value["patches"], CorrectivePatch.from_dict), parse_map(value["externalRoundTrips"], ExternalRoundTrip.from_dict), parse_map(value["privateProxies"], PrivateProxy.from_dict), VisualProfile.from_dict(value["visualProfile"]), list(value["history"]), _ids(value["historyRefs"], "historyRefs"), list(value["checkpointRefs"]), _dict(value["metadata"]), value.get("revisionDigest"),
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

    def transition_guide(
        self,
        guide_id: str,
        lifecycle: str,
        revision: int,
        *,
        expected_document_revision: int,
        replacement_id: str | None = None,
        actor_id: str = "system",
        actor_kind: str = "system",
        command_id: str | None = None,
    ) -> GuideRecord:
        if type(expected_document_revision) is not int or expected_document_revision != self.current_revision:
            raise RevisionConflict(expected_document_revision, self.current_revision)
        if revision != self.current_revision + 1:
            raise SchemaValidationError("guide lifecycle transition must bind the next document revision")
        guide_id = _id(guide_id, "guideId")
        candidate = deepcopy(self)
        guide = candidate.guides.get(guide_id)
        if guide is None:
            raise SchemaValidationError("unknown guide")
        before = guide.to_dict()
        guide.transition(lifecycle, revision, replacement_id=replacement_id)
        self._commit_candidate(candidate, actor_id=actor_id, actor_kind=actor_kind, command_id=command_id,
                               kind="edit", affected_ids=[guide_id], before=before, after=guide.to_dict())
        return self.guides[guide_id]

    def invalidate_selections_for_source(
        self,
        source_id: str,
        *,
        expected_document_revision: int,
        reason: str,
        source_revision: int,
        source_content_digest: str | None = None,
        actor_id: str = "system",
        actor_kind: str = "system",
        command_id: str | None = None,
    ) -> list[str]:
        """Atomically commit source-side stale state in the source transaction."""

        source_id = _id(source_id, "sourceId")
        if type(expected_document_revision) is not int or expected_document_revision != self.current_revision:
            raise RevisionConflict(expected_document_revision, self.current_revision)
        candidate = deepcopy(self)
        if source_id not in candidate.layers and source_id not in candidate.depth_composites:
            raise SchemaValidationError("selection invalidation source does not exist")
        if source_revision != self.current_revision + 1:
            raise SchemaValidationError("source invalidation revision must be the next document revision")
        source_before: dict[str, Any]
        if source_id in candidate.layers:
            source_before = candidate.layers[source_id].to_dict()
            candidate.layers[source_id].revision = source_revision
        else:
            source_before = candidate.depth_composites[source_id].to_dict()
            candidate.depth_composites[source_id].source_revision = source_revision
        before = {key: value.to_dict() for key, value in candidate.selections.items() if value.source_id == source_id}
        changed: list[str] = []
        for selection in candidate.selections.values():
            if selection.source_id == source_id:
                selection.mark_source_changed(reason, source_revision, source_content_digest)
                changed.append(selection.selection_id)
        self._commit_candidate(candidate, actor_id=actor_id, actor_kind=actor_kind,
                               command_id=command_id, kind="source-invalidation", affected_ids=[source_id, *changed],
                               before={"source": source_before, "selections": before},
                               after={"source": (candidate.layers[source_id].to_dict() if source_id in candidate.layers
                                                  else candidate.depth_composites[source_id].to_dict()),
                                      "selections": {key: candidate.selections[key].to_dict() for key in changed}})
        return changed

    def create_selection(
        self,
        selection: SelectionRecord,
        *,
        expected_document_revision: int,
        actor_id: str = "system",
        actor_kind: str = "system",
        command_id: str | None = None,
    ) -> SelectionRecord:
        if type(expected_document_revision) is not int or expected_document_revision != self.current_revision:
            raise RevisionConflict(expected_document_revision, self.current_revision)
        candidate = deepcopy(self)
        if selection.selection_id in candidate.selections:
            raise SchemaValidationError("selection identity already exists")
        created = deepcopy(selection)
        if created.selection_revision != 1 or created.state != SelectionState.CURRENT.value:
            raise SchemaValidationError("new selections must begin current at selection revision one")
        created.validate()
        candidate.selections[created.selection_id] = created
        self._commit_candidate(candidate, actor_id=actor_id, actor_kind=actor_kind, command_id=command_id,
                               kind="selection-create", affected_ids=[created.selection_id], before=None,
                               after=created.to_dict())
        return self.selections[created.selection_id]

    def refine_selection(
        self,
        selection_id: str,
        new_mask_id: str,
        *,
        expected_document_revision: int,
        expected_selection_revision: int,
        refinement: Mapping[str, Any] | None = None,
        actor_id: str = "system",
        actor_kind: str = "system",
        command_id: str | None = None,
    ) -> SelectionRecord:
        return self._mutate_selection(
            selection_id, expected_document_revision, expected_selection_revision, "selection-refine",
            actor_id, actor_kind, command_id,
            lambda record: record.refine(new_mask_id, expected_selection_revision=expected_selection_revision,
                                         refinement=refinement),
        )

    def register_selection_derived_layer(
        self,
        selection_id: str,
        layer_id: str,
        *,
        expected_document_revision: int,
        expected_selection_revision: int,
        actor_id: str = "system",
        actor_kind: str = "system",
        command_id: str | None = None,
    ) -> SelectionRecord:
        return self._mutate_selection(
            selection_id, expected_document_revision, expected_selection_revision, "selection-duplicate",
            actor_id, actor_kind, command_id,
            lambda record: record.register_derived_layer(layer_id, expected_selection_revision=expected_selection_revision),
        )

    def rebase_selection(
        self,
        selection_id: str,
        target_source_revision: int,
        *,
        expected_document_revision: int,
        expected_selection_revision: int,
        new_mask_id: str | None = None,
        source_content_digest: str | None = None,
        geometry_only: bool = False,
        actor_id: str = "system",
        actor_kind: str = "system",
        command_id: str | None = None,
    ) -> SelectionRecord:
        return self._mutate_selection(
            selection_id, expected_document_revision, expected_selection_revision, "selection-rebase",
            actor_id, actor_kind, command_id,
            lambda record: record.explicit_rebase(
                target_source_revision, expected_selection_revision=expected_selection_revision,
                new_mask_id=new_mask_id, source_content_digest=source_content_digest, geometry_only=geometry_only,
            ),
        )

    def _mutate_selection(
        self,
        selection_id: str,
        expected_document_revision: int,
        expected_selection_revision: int,
        kind: str,
        actor_id: str,
        actor_kind: str,
        command_id: str | None,
        mutation: Any,
    ) -> SelectionRecord:
        if type(expected_document_revision) is not int or expected_document_revision != self.current_revision:
            raise RevisionConflict(expected_document_revision, self.current_revision)
        selection_id = _id(selection_id, "selectionId")
        candidate = deepcopy(self)
        if selection_id not in candidate.selections:
            raise SchemaValidationError("unknown selection")
        selection = candidate.selections[selection_id]
        before = selection.to_dict()
        mutation(selection)
        self._commit_candidate(candidate, actor_id=actor_id, actor_kind=actor_kind,
                               command_id=command_id, kind=kind, affected_ids=[selection_id],
                               before=before, after=selection.to_dict())
        return self.selections[selection_id]

    def _commit_candidate(
        self,
        candidate: "Document",
        *,
        actor_id: str,
        actor_kind: str,
        command_id: str | None,
        kind: str,
        affected_ids: list[str],
        before: Any,
        after: Any,
    ) -> None:
        previous_revision = self.current_revision
        candidate.current_revision = previous_revision + 1
        command = command_id or make_id("cmd")
        transaction = TransactionRecord(
            transaction_id=make_id("txn"), group_id=make_id("grp"), actor_id=actor_id,
            actor_kind=actor_kind, command_ids=[command], previous_revision=previous_revision,
            resulting_revision=candidate.current_revision, affected_ids=list(affected_ids),
            before=before, after=after, delta={"recordType": "selection", "ids": list(affected_ids)}, kind=kind,
        )
        candidate.history.append(transaction.to_dict())
        candidate.history_refs = [item["transactionId"] for item in candidate.history]
        candidate.revision_digest = None
        candidate.validate()
        candidate.refresh_digest()
        self.__dict__.update(candidate.__dict__)

    def reconcile_handoffs(self, *, expected_document_revision: int) -> list[str]:
        """Explicitly orphan notes whose bound transaction/targets no longer resolve."""

        if type(expected_document_revision) is not int or expected_document_revision != self.current_revision:
            raise RevisionConflict(expected_document_revision, self.current_revision)
        transactions = {record.transaction_id: record for record in
                        (TransactionRecord.from_dict(value) for value in self.history)}
        targets = {**self.layers, **self.objects}
        candidate = deepcopy(self)
        changed: list[str] = []
        before: dict[str, list[dict[str, Any]]] = {}
        after: dict[str, list[dict[str, Any]]] = {}
        for layer in candidate.layers.values():
            for note in layer.collaboration.handoff_notes:
                transaction = transactions.get(note.transaction_id)
                reasons: list[str] = []
                if transaction is None or transaction.resulting_revision != note.created_at_revision:
                    reasons.append("bound transaction is no longer retained")
                for target_id, revision in note.target_revisions.items():
                    target = targets.get(target_id)
                    if target is None:
                        reasons.append(f"target {target_id} no longer exists")
                    elif target.revision != revision:
                        reasons.append(f"target {target_id} no longer has revision {revision}")
                if reasons and note.state != "orphaned":
                    before.setdefault(layer.layer_id, []).append(note.to_dict())
                    note.state = "orphaned"
                    note.orphan_reason = "; ".join(reasons)
                    after.setdefault(layer.layer_id, []).append(note.to_dict())
                    changed.append(note.note_id)
        if changed:
            self._commit_candidate(candidate, actor_id="system", actor_kind="system", command_id=None,
                                   kind="edit", affected_ids=changed, before={"handoffNotes": before},
                                   after={"handoffNotes": after})
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
