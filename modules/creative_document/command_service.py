"""Transport-neutral v1 semantic commands over the W02/W03 document core."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .agent_projection import (
    AgentProjectionError,
    build_agent_projection,
    render_agent_safe_preview,
)
from .editor_actions import (
    EditorActionError,
    PreparedAction,
    prepare_action,
    prepare_raster_import,
    prepare_undo_redo,
    publish_pending_assets,
)
from .history import HistoryValidationError, TransactionRecord
from .ids import canonical_json, make_id, validate_id
from .schema import Document, SchemaValidationError


SCHEMA_VERSION = 1
MAX_ENVELOPE_BYTES = 1 * 1024 * 1024
MAX_BATCH_ACTIONS = 100
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 100_000


class CommandServiceError(ValueError):
    def __init__(self, code: str, message: str = "Command was refused.") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ActorContext:
    """Actor identity and scopes supplied by an authenticated adapter."""

    actor_kind: str
    actor_id: str
    scopes: frozenset[str]

    def validate(self) -> None:
        if self.actor_kind not in {"human", "director", "agent", "system"}:
            raise CommandServiceError("INVALID_ACTOR")
        try:
            validate_id(self.actor_id, field="actorId")
        except (TypeError, ValueError) as exc:
            raise CommandServiceError("INVALID_ACTOR") from exc
        if not self.scopes.issubset({"inspect", "propose", "mutate"}):
            raise CommandServiceError("INVALID_SCOPE")


class CommandHandle(Protocol):
    document: Document
    store: Any
    committed_revision: int
    lock: Any


@dataclass(frozen=True)
class CommandEnvelope:
    schema_version: int
    command_id: str
    document_id: str
    intent: str
    expected_revision: int | None
    command_type: str
    target_ids: tuple[str, ...]
    coordinate_space: str
    payload: Mapping[str, Any]
    transaction: Mapping[str, Any] | None
    raw: Mapping[str, Any]


_ENVELOPE_FIELDS = {
    "schemaVersion", "commandId", "documentId", "intent", "expectedRevision",
    "commandType", "targetIds", "coordinateSpace", "payload", "transaction",
}
_DURABLE_COORDINATE_SPACES = {"document", "source-crop", "native-bb", "model"}
_INSPECT_COMMANDS = {
    "inspect_document", "inspect_layers", "inspect_objects", "inspect_collaboration",
    "inspect_selections", "inspect_masks", "inspect_context", "inspect_guides",
    "inspect_transactions", "inspect_assets", "render_preview",
}
_MUTATION_COMMANDS = {
    "add_layer", "rename_layer", "set_layer_visibility", "set_layer_lock", "set_layer_opacity",
    "reorder_layer", "reparent_layer", "duplicate_layer", "delete_layer", "transform",
    "create_object", "create_selection", "refine_selection", "rebase_selection",
    "duplicate_selection_to_layer", "create_context_mask", "set_relational_context",
    "create_guide", "transition_guide", "undo", "redo", "save_scene", "batch",
    "import_image",
}
_UNSUPPORTED_COMMANDS = {
    "generate", "generation", "generate_candidate", "accept_candidate", "accept_for_extraction",
    "accept_layer_candidate", "candidate_acceptance", "extract", "extract_candidate_subject",
    "stitch", "stitch_candidate", "bb_execute", "create_bb_context", "run_inference",
    "provider_call", "invoke_provider", "conversation", "chat", "execute_tool", "filesystem",
}

_ACTION_DATA_FIELDS: dict[str, set[str]] = {
    "add_layer": {"name", "kind", "parentId", "role"},
    "rename_layer": {"layerId", "name"},
    "set_layer_visibility": {"layerId", "visible"},
    "set_layer_lock": {"layerId", "locked"},
    "set_layer_opacity": {"layerId", "opacity"},
    "reorder_layer": {"layerId", "direction", "index"},
    "reparent_layer": {"layerId", "parentId"},
    "duplicate_layer": {"layerId", "name"},
    "delete_layer": {"layerId", "confirmed"},
    "transform": {"transform"},
    "create_object": {"layerId", "kind", "geometry", "style"},
    "create_selection": {"sourceLayerId", "seeds", "semanticHint"},
    "refine_selection": {"selectionId", "expectedSelectionRevision", "operation", "radius", "seed", "details"},
    "rebase_selection": {"selectionId", "expectedSelectionRevision", "geometryOnly", "seeds"},
    "duplicate_selection_to_layer": {"selectionId", "expectedSelectionRevision", "name", "parentId"},
    "create_context_mask": {"sourceLayerId", "seeds"},
    "set_relational_context": {"contextMaskId", "referenceIds", "dilationPx", "editMaskId", "editTargetIds"},
    "create_guide": {"name", "semanticRole", "kind", "geometry", "style"},
    "transition_guide": {"guideId", "lifecycle", "replacementId"},
    "import_image": {"filename", "asGuide", "semanticRole", "parentId", "name"},
}
_ACTION_FIELDS = {"actionType", "data", "targetIds"}
_REFUSAL_MESSAGES = {
    "AUTHORIZATION_REQUIRED": "The local driver grant is unavailable or expired.",
    "SCOPE_REQUIRED": "The local driver grant does not include this command scope.",
    "UNSUPPORTED_COMMAND": "This command is outside the supported creative-document surface.",
    "STALE_DOCUMENT_REVISION": "The document changed. Refresh it and submit a new command explicitly.",
    "COMMAND_ID_REUSE": "This command ID was already used with different content.",
    "INVALID_ENVELOPE": "The command envelope is malformed.",
    "INVALID_COORDINATE_SPACE": "The command coordinate space is unsupported.",
    "TARGET_MISMATCH": "The declared command targets do not match the validated action targets.",
    "REQUEST_TOO_LARGE": "The command exceeds the supported size limit.",
    "REQUEST_STRUCTURE_TOO_LARGE": "The command structure exceeds the supported limits.",
    "VALIDATION_FAILED": "The command did not pass document validation.",
    "ASSET_PUBLICATION_FAILED": "The command could not publish its immutable assets.",
    "SAFE_VIEW_UNAVAILABLE": "The requested Agent-safe view is unavailable.",
    "SAFE_PREVIEW_UNAVAILABLE": "The Agent-safe preview is unavailable.",
}


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CommandServiceError("DUPLICATE_JSON_FIELD")
        value[key] = item
    return value


def strict_json_loads(raw: bytes | str, *, max_bytes: int = MAX_ENVELOPE_BYTES) -> Any:
    """Parse finite UTF-8 JSON with duplicate-key and size rejection."""

    if isinstance(raw, bytes):
        if len(raw) > max_bytes:
            raise CommandServiceError("REQUEST_TOO_LARGE")
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise CommandServiceError("INVALID_JSON") from exc
    elif isinstance(raw, str):
        if len(raw.encode("utf-8")) > max_bytes:
            raise CommandServiceError("REQUEST_TOO_LARGE")
        text = raw
    else:
        raise CommandServiceError("INVALID_JSON")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=lambda _: (_ for _ in ()).throw(CommandServiceError("NON_FINITE_JSON")),
        )
    except CommandServiceError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise CommandServiceError("INVALID_JSON") from exc
    _validate_json_tree(value)
    return value


def _validate_json_tree(value: Any) -> None:
    count = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal count
        count += 1
        if count > MAX_JSON_NODES or depth > MAX_JSON_DEPTH:
            raise CommandServiceError("REQUEST_STRUCTURE_TOO_LARGE")
        if item is None or type(item) in {str, bool, int}:
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise CommandServiceError("NON_FINITE_JSON")
            return
        if isinstance(item, list):
            for child in item:
                visit(child, depth + 1)
            return
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise CommandServiceError("INVALID_JSON")
                visit(child, depth + 1)
            return
        raise CommandServiceError("INVALID_JSON")

    visit(value, 0)


def _validate_envelope_limits(value: Any) -> None:
    """Enforce service-owned finite JSON, canonical byte, depth, and node caps."""

    _validate_json_tree(value)
    try:
        encoded = canonical_json(value)
    except (TypeError, ValueError, RecursionError) as exc:
        raise CommandServiceError("INVALID_JSON") from exc
    if len(encoded) > MAX_ENVELOPE_BYTES:
        raise CommandServiceError("REQUEST_TOO_LARGE")


def _parse_transaction(value: Any, *, required: bool) -> Mapping[str, Any] | None:
    if value is None and not required:
        return None
    if not isinstance(value, dict) or set(value) != {"transactionId", "groupId", "phase"}:
        raise CommandServiceError("INVALID_TRANSACTION")
    if value["phase"] != "commit":
        raise CommandServiceError("INVALID_TRANSACTION")
    try:
        validate_id(value["transactionId"], field="transactionId")
        validate_id(value["groupId"], field="groupId")
    except (TypeError, ValueError) as exc:
        raise CommandServiceError("INVALID_TRANSACTION") from exc
    return value


def _validate_action_payload(command_type: str, payload: Any, *, in_batch: bool = False) -> None:
    if command_type == "batch":
        if not isinstance(payload, dict) or set(payload) != {"actions"}:
            raise CommandServiceError("INVALID_PAYLOAD")
        actions = payload["actions"]
        if not isinstance(actions, list) or not 1 <= len(actions) <= MAX_BATCH_ACTIONS:
            raise CommandServiceError("INVALID_BATCH")
        for action in actions:
            if not isinstance(action, dict) or set(action) - _ACTION_FIELDS or not {"actionType", "data"}.issubset(action):
                raise CommandServiceError("INVALID_BATCH")
            kind = action["actionType"]
            if not isinstance(kind, str) or kind not in _ACTION_DATA_FIELDS or kind in {"import_image"}:
                raise CommandServiceError("UNSUPPORTED_COMMAND")
            if "targetIds" in action:
                _validate_target_ids(action["targetIds"])
            if kind == "transform" and ("targetIds" not in action or not action["targetIds"]):
                raise CommandServiceError("INVALID_TARGETS")
            _validate_action_data(kind, action["data"])
        return

    if command_type == "save_scene":
        if payload not in ({}, None):
            raise CommandServiceError("INVALID_PAYLOAD")
        return
    if command_type in {"undo", "redo"}:
        if payload not in ({}, None):
            raise CommandServiceError("INVALID_PAYLOAD")
        return
    if command_type not in _ACTION_DATA_FIELDS:
        raise CommandServiceError("UNSUPPORTED_COMMAND")
    if command_type == "import_image" and in_batch:
        raise CommandServiceError("UNSUPPORTED_COMMAND")
    if not isinstance(payload, dict) or set(payload) != {"data"}:
        raise CommandServiceError("INVALID_PAYLOAD")
    _validate_action_data(command_type, payload["data"])


def _validate_action_data(command_type: str, data: Any) -> None:
    if not isinstance(data, dict):
        raise CommandServiceError("INVALID_PAYLOAD")
    allowed = _ACTION_DATA_FIELDS[command_type]
    if set(data) - allowed:
        raise CommandServiceError("UNKNOWN_FIELD")
    try:
        canonical_json(data)
    except (TypeError, ValueError) as exc:
        raise CommandServiceError("INVALID_PAYLOAD") from exc
    id_fields = {
        "rename_layer": ("layerId",), "set_layer_visibility": ("layerId",), "set_layer_lock": ("layerId",),
        "set_layer_opacity": ("layerId",), "reorder_layer": ("layerId",),
        "reparent_layer": ("layerId",), "duplicate_layer": ("layerId",), "delete_layer": ("layerId",),
        "create_object": ("layerId",), "create_selection": ("sourceLayerId",),
        "refine_selection": ("selectionId",), "rebase_selection": ("selectionId",),
        "duplicate_selection_to_layer": ("selectionId",), "create_context_mask": ("sourceLayerId",),
        "set_relational_context": ("contextMaskId",), "transition_guide": ("guideId",),
    }.get(command_type, ())
    for field in id_fields:
        if field in data and not (field == "sourceLayerId" and data[field] is None):
            _validate_id_value(data[field])
    for field in ("parentId", "editMaskId", "replacementId"):
        if field in data and data[field] is not None:
            _validate_id_value(data[field])
    string_fields = {
        "add_layer": ("name", "kind", "role"),
        "rename_layer": ("name",), "duplicate_layer": ("name",),
        "duplicate_selection_to_layer": ("name",),
        "create_selection": ("semanticHint",),
        "transition_guide": ("lifecycle",), "refine_selection": ("operation",),
        "import_image": ("filename", "semanticRole", "name"),
        "create_guide": ("name", "semanticRole", "kind"),
    }.get(command_type, ())
    for field in string_fields:
        if field in data and data[field] is not None and (not isinstance(data[field], str) or len(data[field]) > 1024):
            raise CommandServiceError("INVALID_PAYLOAD")
    if "operation" in data and command_type == "refine_selection" and data["operation"] not in {
        "add", "subtract", "intersect", "paint", "invert", "grow", "shrink", "feather", "clean",
    }:
        raise CommandServiceError("INVALID_PAYLOAD")
    if "lifecycle" in data and command_type == "transition_guide" and data["lifecycle"] not in {
        "proposed", "active", "consumed", "replacement-pending", "superseded", "safe-to-remove",
    }:
        raise CommandServiceError("INVALID_PAYLOAD")
    boolean_fields = {
        "set_layer_visibility": ("visible",), "set_layer_lock": ("locked",),
        "delete_layer": ("confirmed",), "rebase_selection": ("geometryOnly",),
        "import_image": ("asGuide",),
    }.get(command_type, ())
    if any(field in data and type(data[field]) is not bool for field in boolean_fields):
        raise CommandServiceError("INVALID_PAYLOAD")
    numeric_fields = {
        "refine_selection": ("radius",),
        "set_relational_context": ("dilationPx",),
    }.get(command_type, ())
    for field in numeric_fields:
        if field in data and (type(data[field]) is not int or data[field] < 0):
            raise CommandServiceError("INVALID_PAYLOAD")
    if (command_type == "set_layer_opacity" and "opacity" in data
            and (type(data["opacity"]) not in {int, float} or not math.isfinite(float(data["opacity"]))
                 or not 0 <= data["opacity"] <= 1)):
        raise CommandServiceError("INVALID_PAYLOAD")
    if command_type == "reorder_layer":
        if "direction" in data and data["direction"] not in {"up", "down"}:
            raise CommandServiceError("INVALID_PAYLOAD")
        if "index" in data and (type(data["index"]) is not int or data["index"] < 0):
            raise CommandServiceError("INVALID_PAYLOAD")
    for field in ("expectedSelectionRevision",):
        if field in data and (type(data[field]) is not int or data[field] < 0):
            raise CommandServiceError("INVALID_REVISION")
    if command_type == "set_relational_context":
        for field in ("referenceIds", "editTargetIds"):
            if field in data:
                _validate_target_ids(data[field])
    for key in ("geometry",):
        if key in data:
            _validate_geometry_mapping(data[key], allow_shape=True, allow_path=True)
    if "style" in data:
        _validate_style_mapping(data["style"])
    if "transform" in data:
        value = data["transform"]
        if (not isinstance(value, list) or len(value) != 9
                or any(type(item) not in {int, float} or not math.isfinite(float(item)) for item in value)):
            raise CommandServiceError("INVALID_TRANSFORM")
    if "seeds" in data:
        seeds = data["seeds"]
        if not isinstance(seeds, list) or not 1 <= len(seeds) <= 2048:
            raise CommandServiceError("INVALID_SEEDS")
        for seed in seeds:
            _validate_seed(seed)
    if "seed" in data:
        _validate_seed(data["seed"])
    if "details" in data:
        # The accepted editor contract currently has no free-form refinement
        # detail fields. Preserve its object shape while rejecting extensions.
        _validate_nested_mapping(data["details"], set())


def _validate_nested_mapping(value: Any, allowed: set[str]) -> None:
    if not isinstance(value, dict) or set(value) - allowed:
        raise CommandServiceError("UNKNOWN_FIELD")


def _validate_geometry_mapping(value: Any, *, allow_shape: bool, allow_path: bool) -> None:
    allowed = {"x", "y", "width", "height", "points"}
    if allow_path:
        allowed.update({"closed", "size", "radius"})
    if allow_shape:
        allowed.add("shape")
    _validate_nested_mapping(value, allowed)
    if "shape" in value and (not isinstance(value["shape"], str)
                             or value["shape"] not in {"line", "rectangle", "ellipse", "polygon"}):
        raise CommandServiceError("INVALID_GEOMETRY")
    if "closed" in value and type(value["closed"]) is not bool:
        raise CommandServiceError("INVALID_GEOMETRY")
    for field in ("x", "y", "width", "height", "size", "radius"):
        if field in value and (type(value[field]) not in {int, float} or not math.isfinite(float(value[field]))):
            raise CommandServiceError("INVALID_GEOMETRY")
    if "points" in value:
        points = value["points"]
        if not isinstance(points, list) or len(points) > 20_000:
            raise CommandServiceError("INVALID_GEOMETRY")
        for point in points:
            if (not isinstance(point, (list, tuple)) or len(point) != 2
                    or any(type(item) not in {int, float} or not math.isfinite(float(item)) for item in point)):
                raise CommandServiceError("INVALID_GEOMETRY")


def _validate_style_mapping(value: Any) -> None:
    allowed = {"fill", "stroke", "color", "width", "strokeWidth", "opacity", "mode", "hardness"}
    _validate_nested_mapping(value, allowed)
    for field, item in value.items():
        if field in {"fill", "stroke", "color", "mode"}:
            if item is not None and not isinstance(item, str):
                raise CommandServiceError("INVALID_STYLE")
            if field == "mode" and item not in {None, "paint", "erase"}:
                raise CommandServiceError("INVALID_STYLE")
        elif type(item) not in {int, float} or not math.isfinite(float(item)):
            raise CommandServiceError("INVALID_STYLE")


def _validate_seed(seed: Any) -> None:
    fields = {"kind", "geometry", "x", "y", "tolerance", "mode", "maskId", "points", "size", "radius"}
    if not isinstance(seed, dict) or set(seed) - fields:
        raise CommandServiceError("INVALID_SEED")
    kind = seed.get("kind")
    if not isinstance(kind, str) or kind not in {"point", "box", "rectangle", "polygon", "paint", "brush", "alpha", "mask"}:
        raise CommandServiceError("INVALID_SEED")
    if "geometry" in seed:
        _validate_geometry_mapping(seed["geometry"], allow_shape=False, allow_path=True)
    if "mode" in seed and (not isinstance(seed["mode"], str)
                            or seed["mode"] not in {"replace", "add", "subtract", "intersect"}):
        raise CommandServiceError("INVALID_SEED")
    if "maskId" in seed:
        _validate_id_value(seed["maskId"])
    for field in ("x", "y"):
        if field in seed and (type(seed[field]) not in {int, float} or not math.isfinite(float(seed[field]))):
            raise CommandServiceError("INVALID_SEED")
    if "tolerance" in seed and (type(seed["tolerance"]) is not int or not 0 <= seed["tolerance"] <= 96):
        raise CommandServiceError("INVALID_SEED")
    for field in ("size", "radius"):
        if field in seed and (type(seed[field]) not in {int, float} or not math.isfinite(float(seed[field]))):
            raise CommandServiceError("INVALID_SEED")
    for key in ("points",):
        if key in seed:
            points = seed[key]
            if not isinstance(points, list) or len(points) > 20_000:
                raise CommandServiceError("INVALID_SEED")
            for point in points:
                if (not isinstance(point, (list, tuple)) or len(point) != 2
                        or any(type(value) not in {int, float} or not math.isfinite(float(value)) for value in point)):
                    raise CommandServiceError("INVALID_SEED")


def _validate_id_value(value: Any) -> None:
    try:
        validate_id(value, field="referenceId")
    except (TypeError, ValueError) as exc:
        raise CommandServiceError("INVALID_TARGETS") from exc


def _validate_target_ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > MAX_BATCH_ACTIONS:
        raise CommandServiceError("INVALID_TARGETS")
    result: list[str] = []
    for item in value:
        try:
            validate_id(item, field="targetId")
        except (TypeError, ValueError) as exc:
            raise CommandServiceError("INVALID_TARGETS") from exc
        result.append(item)
    if len(result) != len(set(result)):
        raise CommandServiceError("DUPLICATE_TARGET_ID")
    return tuple(result)


_ACTION_SELECTOR_FIELDS = {
    "rename_layer": "layerId",
    "set_layer_visibility": "layerId",
    "set_layer_lock": "layerId",
    "set_layer_opacity": "layerId",
    "reorder_layer": "layerId",
    "reparent_layer": "layerId",
    "duplicate_layer": "layerId",
    "delete_layer": "layerId",
    "refine_selection": "selectionId",
    "rebase_selection": "selectionId",
    "duplicate_selection_to_layer": "selectionId",
    "set_relational_context": "contextMaskId",
    "transition_guide": "guideId",
}


def _action_selector_targets(
    command_type: str,
    data: Mapping[str, Any],
    *,
    transform_targets: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return direct command subjects; placement and read-only references stay in payload."""

    if command_type == "transform":
        if not transform_targets:
            raise CommandServiceError("INVALID_TARGETS")
        return transform_targets
    selector = _ACTION_SELECTOR_FIELDS.get(command_type)
    identity = data.get(selector) if selector is not None else None
    return (identity,) if isinstance(identity, str) else ()


def _canonical_target_ids(
    command_type: str,
    payload: Mapping[str, Any],
    declared_targets: tuple[str, ...],
) -> tuple[str, ...]:
    """Derive receipt/execution targets from one validated command model.

    Create destinations, parents, replacement IDs, context references, and
    selection CAS revisions remain typed payload references. They are not
    command subjects. A batch's target list is the ordered union of its direct
    subcommand subjects.
    """

    if command_type == "batch":
        targets: list[str] = []
        for action in payload.get("actions", []):
            kind = action["actionType"]
            data = action["data"]
            nested = _validate_target_ids(action.get("targetIds", []))
            derived = _action_selector_targets(kind, data, transform_targets=nested if kind == "transform" else ())
            if kind != "transform" and "targetIds" in action and nested != derived:
                raise CommandServiceError("TARGET_MISMATCH")
            for target in derived:
                if target not in targets:
                    targets.append(target)
        return _validate_target_ids(targets)
    if command_type == "transform":
        return _action_selector_targets(command_type, payload.get("data", {}),
                                        transform_targets=declared_targets)
    if command_type in _ACTION_DATA_FIELDS:
        return _action_selector_targets(command_type, payload.get("data", {}))
    # Inspect, save, undo, and redo have no object/layer subject.
    return ()


def parse_envelope(value: Any) -> CommandEnvelope:
    _validate_envelope_limits(value)
    if not isinstance(value, dict) or set(value) != _ENVELOPE_FIELDS:
        raise CommandServiceError("INVALID_ENVELOPE")
    if type(value["schemaVersion"]) is not int or value["schemaVersion"] != SCHEMA_VERSION:
        raise CommandServiceError("UNSUPPORTED_SCHEMA_VERSION")
    for field in ("commandId", "documentId"):
        try:
            validate_id(value[field], field=field)
        except (TypeError, ValueError) as exc:
            raise CommandServiceError("INVALID_ENVELOPE") from exc
    intent = value["intent"]
    command_type = value["commandType"]
    if not isinstance(intent, str) or intent not in {"inspect", "propose", "mutate"} or not isinstance(command_type, str):
        raise CommandServiceError("INVALID_ENVELOPE")
    expected = value["expectedRevision"]
    if intent == "inspect":
        if expected is not None and (type(expected) is not int or expected < 0):
            raise CommandServiceError("INVALID_REVISION")
        if command_type not in _INSPECT_COMMANDS:
            raise CommandServiceError("UNSUPPORTED_COMMAND")
        if value["payload"] != {}:
            raise CommandServiceError("UNKNOWN_FIELD")
        transaction = _parse_transaction(value["transaction"], required=False)
    else:
        if type(expected) is not int or expected < 0:
            raise CommandServiceError("INVALID_REVISION")
        if command_type in _UNSUPPORTED_COMMANDS:
            raise CommandServiceError("UNSUPPORTED_COMMAND")
        if command_type not in _MUTATION_COMMANDS:
            raise CommandServiceError("UNSUPPORTED_COMMAND")
        _validate_action_payload(command_type, value["payload"])
        transaction = _parse_transaction(value["transaction"], required=True)
    coordinate_space = value["coordinateSpace"]
    if not isinstance(coordinate_space, str) or coordinate_space not in _DURABLE_COORDINATE_SPACES:
        raise CommandServiceError("INVALID_COORDINATE_SPACE")
    if intent != "inspect" and coordinate_space != "document":
        raise CommandServiceError("INVALID_COORDINATE_SPACE")
    target_ids = _validate_target_ids(value["targetIds"])
    if not isinstance(value["payload"], dict):
        raise CommandServiceError("INVALID_PAYLOAD")
    try:
        canonical_json(value["payload"])
    except (TypeError, ValueError) as exc:
        raise CommandServiceError("INVALID_PAYLOAD") from exc
    expected_targets = _canonical_target_ids(command_type, value["payload"], target_ids)
    if target_ids != expected_targets:
        raise CommandServiceError("TARGET_MISMATCH")
    return CommandEnvelope(
        value["schemaVersion"], value["commandId"], value["documentId"], intent, expected,
        command_type, target_ids, coordinate_space, value["payload"], transaction, value,
    )


def canonical_receipt(
    *,
    status: str,
    command_id: str | None,
    document_id: str | None,
    actor_kind: str | None,
    actor_id: str | None = None,
    intent: str | None,
    previous_revision: int | None,
    new_revision: int | None,
    current_revision: int | None,
    observed_revision: int | None,
    transaction_id: str | None,
    target_ids: list[str] | tuple[str, ...] = (),
    affected_ids: list[str] | tuple[str, ...] = (),
    created_ids: list[str] | tuple[str, ...] = (),
    invalidated: list[dict[str, Any]] | None = None,
    conflicts: list[str] | None = None,
    writes: int = 0,
    result: Any = None,
    error: dict[str, str] | None = None,
    hint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "status": status,
        "commandId": command_id,
        "documentId": document_id,
        "actorKind": actor_kind,
        "actorId": actor_id,
        "intent": intent,
        "previousRevision": previous_revision,
        "newRevision": new_revision,
        "currentRevision": current_revision,
        "observedRevision": observed_revision,
        "transactionId": transaction_id,
        "targetIds": list(target_ids),
        "affectedTargetIds": list(affected_ids),
        "createdIds": list(created_ids),
        "invalidated": list(invalidated or []),
        "conflicts": list(conflicts or []),
        "writes": writes,
        "result": result,
        "error": error,
        "hint": hint,
    }


def refusal_receipt(
    *,
    code: str,
    envelope: CommandEnvelope | None = None,
    actor_kind: str | None = None,
    actor_id: str | None = None,
    current_revision: int | None = None,
    document_id: str | None = None,
) -> dict[str, Any]:
    stale = code in {"STALE_DOCUMENT_REVISION", "STALE_SELECTION_REVISION", "REVISION_CONFLICT"}
    transaction_id = None if envelope is None or envelope.transaction is None else envelope.transaction["transactionId"]
    return canonical_receipt(
        status="conflict" if stale else "error",
        command_id=None if envelope is None else envelope.command_id,
        document_id=(envelope.document_id if envelope is not None else document_id),
        actor_kind=actor_kind,
        actor_id=actor_id,
        intent=None if envelope is None else envelope.intent,
        previous_revision=(envelope.expected_revision if stale and envelope is not None else current_revision),
        new_revision=None,
        current_revision=current_revision,
        observed_revision=current_revision,
        transaction_id=transaction_id,
        target_ids=() if envelope is None else envelope.target_ids,
        conflicts=[code] if stale else [],
        writes=0,
        error={"code": code, "message": _REFUSAL_MESSAGES.get(code, "Command was refused.")},
        hint={"action": "refresh-and-resubmit", "expectedRevision": current_revision} if stale else None,
    )


def _required_scope(envelope: CommandEnvelope) -> str:
    if envelope.intent == "inspect":
        return "inspect"
    return envelope.intent


def _request_digest(envelope: CommandEnvelope, actor: ActorContext, internal_digest: str | None) -> str:
    value = {
        "envelope": envelope.raw,
        "trustedActor": {"kind": actor.actor_kind, "id": actor.actor_id},
        "internalContentDigest": internal_digest,
    }
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _transaction_for_command(document: Document, command_id: str) -> Mapping[str, Any] | None:
    for raw in document.history:
        if command_id in raw.get("commandIds", []):
            return raw
    return None


def _action_payload(envelope: CommandEnvelope, actor: ActorContext, internal_data: Mapping[str, Any] | None) -> dict[str, Any]:
    transaction = envelope.transaction or {}
    payload: dict[str, Any] = {
        "documentId": envelope.document_id,
        "expectedRevision": envelope.expected_revision,
        "actorKind": actor.actor_kind,
        "transactionId": transaction.get("transactionId"),
        "groupId": transaction.get("groupId"),
        "actionType": envelope.command_type,
    }
    if envelope.command_type == "batch":
        actions = deepcopy(envelope.payload["actions"])
        payload["actions"] = actions
    elif envelope.command_type not in {"save_scene", "undo", "redo"}:
        data = dict(envelope.payload["data"])
        if envelope.command_type == "transform":
            payload["targetIds"] = list(envelope.target_ids)
        if envelope.command_type == "import_image" and internal_data:
            data.update(internal_data)
        payload["data"] = data
    return payload


def _invalidated_records(document: Document, ids: list[str]) -> list[dict[str, Any]]:
    result = []
    for identity in ids:
        selection = document.selections.get(identity)
        reason = selection.stale_reason if selection is not None else "derived-content-changed"
        source_revision = selection.source_revision if selection is not None else document.current_revision
        result.append({"id": identity, "reason": reason or "derived-content-changed", "sourceRevision": source_revision})
    return result


class CommandService:
    """Execute commands against document history and ProjectStore authority.

    Mutations are witnessed in their durable transaction metadata. Save
    commands use a ProjectStore intent paired with an exact checkpoint
    fingerprint. No process-local receipt cache can overrule either authority.
    """

    def execute(
        self,
        handle: CommandHandle,
        raw_envelope: Any,
        actor: ActorContext,
        *,
        internal_import: Mapping[str, Any] | None = None,
        read_projector: Any = None,
    ) -> dict[str, Any]:
        try:
            actor.validate()
        except CommandServiceError as exc:
            return refusal_receipt(code=exc.code)
        try:
            envelope = parse_envelope(raw_envelope)
        except CommandServiceError as exc:
            return refusal_receipt(code=exc.code, actor_kind=actor.actor_kind, actor_id=actor.actor_id)
        if handle.document.document_id != envelope.document_id:
            return refusal_receipt(code="UNKNOWN_DOCUMENT", envelope=envelope,
                                   actor_kind=actor.actor_kind, actor_id=actor.actor_id)
        required_scope = _required_scope(envelope)
        if required_scope not in actor.scopes:
            return refusal_receipt(code="SCOPE_REQUIRED", envelope=envelope,
                                   actor_kind=actor.actor_kind, actor_id=actor.actor_id,
                                   current_revision=handle.document.current_revision)

        lock = handle.lock if hasattr(handle, "lock") else threading.RLock()
        with lock:
            document = handle.document
            current_revision = document.current_revision
            internal_digest = None
            if internal_import is not None:
                raw_bytes = internal_import.get("fileBytes")
                if not isinstance(raw_bytes, bytes):
                    return refusal_receipt(code="INVALID_PAYLOAD", envelope=envelope,
                                           actor_kind=actor.actor_kind, actor_id=actor.actor_id,
                                           current_revision=current_revision)
                internal_digest = hashlib.sha256(raw_bytes).hexdigest()
            digest = _request_digest(envelope, actor, internal_digest)
            if envelope.intent == "inspect":
                return self._inspect(handle, envelope, actor, current_revision, read_projector)

            previous = _transaction_for_command(document, envelope.command_id)
            if previous is not None:
                semantic = previous.get("metadata", {}).get("semanticCommand")
                if (isinstance(semantic, dict) and semantic.get("requestDigest") == digest
                        and semantic.get("actorKind") == actor.actor_kind
                        and semantic.get("actorId") == actor.actor_id
                        and isinstance(semantic.get("receipt"), dict)
                        and semantic["receipt"].get("actorId") == actor.actor_id):
                    return deepcopy(semantic["receipt"])
                return refusal_receipt(code="COMMAND_ID_REUSE", envelope=envelope,
                                       actor_kind=actor.actor_kind, actor_id=actor.actor_id,
                                       current_revision=current_revision)

            try:
                save_record = handle.store.read_semantic_save_record(envelope.command_id)
            except Exception:
                return refusal_receipt(code="INTERNAL_ERROR", envelope=envelope,
                                       actor_kind=actor.actor_kind, actor_id=actor.actor_id,
                                       current_revision=current_revision)
            if save_record is not None:
                if (envelope.command_type != "save_scene"
                        or save_record.get("requestDigest") != digest
                        or save_record.get("actorKind") != actor.actor_kind
                        or save_record.get("actorId") != actor.actor_id):
                    return refusal_receipt(code="COMMAND_ID_REUSE", envelope=envelope,
                                           actor_kind=actor.actor_kind, actor_id=actor.actor_id,
                                           current_revision=current_revision)
                try:
                    saved_authority = handle.store.checkpoint_matches(
                        save_record["revision"], save_record["revisionDigest"]
                    )
                except Exception:
                    saved_authority = False
                if saved_authority:
                    receipt = self._saved_receipt(
                        envelope, actor, save_record["revision"],
                    )
                    if save_record.get("status") != "saved":
                        completed_record = {**save_record, "status": "saved", "receipt": deepcopy(receipt)}
                        try:
                            handle.store.write_semantic_save_receipt(completed_record)
                        except Exception:
                            # The durable intent plus matching checkpoint is
                            # sufficient to reconstruct this same receipt.
                            pass
                    return receipt
                if save_record.get("status") == "saved":
                    return refusal_receipt(code="SAVE_FAILED", envelope=envelope,
                                           actor_kind=actor.actor_kind, actor_id=actor.actor_id,
                                           current_revision=current_revision)

            if envelope.expected_revision != current_revision:
                return refusal_receipt(code="STALE_DOCUMENT_REVISION", envelope=envelope,
                                       actor_kind=actor.actor_kind, actor_id=actor.actor_id,
                                       current_revision=current_revision)

            if envelope.intent == "propose":
                return self._propose(handle, envelope, actor)

            if envelope.command_type == "save_scene":
                return self._save(handle, envelope, actor, digest)

            action = _action_payload(envelope, actor, internal_import)
            try:
                if envelope.command_type == "undo":
                    prepared = prepare_undo_redo(document, action, actor_id=actor.actor_id, actor_kind=actor.actor_kind,
                                                 command_id=envelope.command_id, transaction_id=envelope.transaction["transactionId"],
                                                 group_id=envelope.transaction["groupId"], redo=False)
                elif envelope.command_type == "redo":
                    prepared = prepare_undo_redo(document, action, actor_id=actor.actor_id, actor_kind=actor.actor_kind,
                                                 command_id=envelope.command_id, transaction_id=envelope.transaction["transactionId"],
                                                 group_id=envelope.transaction["groupId"], redo=True)
                elif envelope.command_type == "import_image":
                    data = action.get("data", {})
                    raw_bytes = data.pop("fileBytes", None)
                    prepared = prepare_raster_import(
                        document, handle.store.assets, action, raw_bytes,
                        actor_id=actor.actor_id, actor_kind=actor.actor_kind,
                        command_id=envelope.command_id,
                        filename=str(data.get("filename", "import.png")),
                    )
                else:
                    prepared = prepare_action(
                        document, handle.store.assets, action, actor_id=actor.actor_id,
                        actor_kind=actor.actor_kind, command_id=envelope.command_id,
                        transaction_id=envelope.transaction["transactionId"], group_id=envelope.transaction["groupId"],
                    )
            except EditorActionError as exc:
                return refusal_receipt(code=exc.code, envelope=envelope,
                                       actor_kind=actor.actor_kind, actor_id=actor.actor_id,
                                       current_revision=exc.current_revision or current_revision)
            except (SchemaValidationError, HistoryValidationError, TypeError, ValueError):
                return refusal_receipt(code="VALIDATION_FAILED", envelope=envelope,
                                       actor_kind=actor.actor_kind, actor_id=actor.actor_id,
                                       current_revision=current_revision)
            except Exception:
                return refusal_receipt(code="INTERNAL_ERROR", envelope=envelope,
                                       actor_kind=actor.actor_kind, actor_id=actor.actor_id,
                                       current_revision=current_revision)

            receipt = self._mutation_receipt(envelope, actor, prepared)
            try:
                tx = prepared.document.history[-1]
                metadata = dict(tx.get("metadata", {}))
                metadata["semanticCommand"] = {
                    "schemaVersion": SCHEMA_VERSION,
                    "commandId": envelope.command_id,
                    "requestDigest": digest,
                    "actorKind": actor.actor_kind,
                    "actorId": actor.actor_id,
                    "receipt": deepcopy(receipt),
                }
                tx["metadata"] = metadata
                prepared.document.revision_digest = None
                prepared.document.validate()
                prepared.document.refresh_digest()
                publish_pending_assets(handle.store.assets, prepared)
            except EditorActionError as exc:
                return refusal_receipt(code=exc.code, envelope=envelope,
                                       actor_kind=actor.actor_kind, actor_id=actor.actor_id,
                                       current_revision=current_revision)
            except Exception:
                return refusal_receipt(code="ASSET_PUBLICATION_FAILED", envelope=envelope,
                                       actor_kind=actor.actor_kind, actor_id=actor.actor_id,
                                       current_revision=current_revision)

            handle.document = prepared.document
            if hasattr(handle, "recovery_notice"):
                handle.recovery_notice = None
            return receipt

    def _inspect(
        self, handle: CommandHandle, envelope: CommandEnvelope, actor: ActorContext, current_revision: int,
        read_projector: Any = None,
    ) -> dict[str, Any]:
        if envelope.expected_revision is not None and envelope.expected_revision != current_revision:
            return refusal_receipt(code="STALE_DOCUMENT_REVISION", envelope=envelope,
                                   actor_kind=actor.actor_kind, actor_id=actor.actor_id,
                                   current_revision=current_revision)
        try:
            if actor.actor_kind in {"human", "director"} and read_projector is not None:
                view = read_projector(handle)
                snapshot_token = f"{handle.document.document_id}@{current_revision}"
                result = {"view": view}
            else:
                projection = build_agent_projection(handle.document)
                snapshot_token = projection["snapshotToken"]
                if envelope.command_type == "inspect_layers":
                    result = {"rootLayerIds": projection["rootLayerIds"], "layers": projection["layers"], "objects": projection["objects"]}
                elif envelope.command_type == "inspect_objects":
                    result = {"objects": projection["objects"]}
                elif envelope.command_type == "inspect_collaboration":
                    result = {"layers": projection["layers"]}
                elif envelope.command_type == "inspect_selections":
                    result = {"selections": projection["selections"]}
                elif envelope.command_type == "inspect_masks":
                    result = {"masks": projection["masks"]}
                elif envelope.command_type == "inspect_context":
                    result = {"relationalContext": projection["relationalContext"]}
                elif envelope.command_type == "inspect_guides":
                    result = {"guides": projection["guides"]}
                elif envelope.command_type == "inspect_transactions":
                    result = {"recentTransactions": projection["recentTransactions"]}
                elif envelope.command_type == "inspect_assets":
                    result = {"assets": projection["assets"], "privateProxies": projection["privateProxies"]}
                else:
                    result = projection
        except AgentProjectionError as exc:
            return refusal_receipt(code=exc.code, envelope=envelope,
                                   actor_kind=actor.actor_kind, actor_id=actor.actor_id,
                                   current_revision=current_revision)
        except Exception:
            return refusal_receipt(code="SAFE_VIEW_UNAVAILABLE", envelope=envelope,
                                   actor_kind=actor.actor_kind, actor_id=actor.actor_id,
                                   current_revision=current_revision)
        return canonical_receipt(
            status="ok", command_id=envelope.command_id, document_id=envelope.document_id,
            actor_kind=actor.actor_kind, actor_id=actor.actor_id, intent="inspect", previous_revision=current_revision,
            new_revision=None, current_revision=current_revision, observed_revision=current_revision,
            transaction_id=None, target_ids=envelope.target_ids, writes=0,
            result={"snapshotToken": snapshot_token, "observedRevision": current_revision, **result},
        )

    def preview(self, handle: CommandHandle, *, expected_revision: int | None = None) -> tuple[bytes, int, str]:
        """Capture, allowlist, render, and return a safe preview under one lock."""

        with handle.lock:
            document = handle.document
            if expected_revision is not None and expected_revision != document.current_revision:
                raise CommandServiceError("STALE_DOCUMENT_REVISION")
            try:
                content, revision = render_agent_safe_preview(
                    document, lambda asset: handle.store.assets.read_bytes(asset),
                )
            except AgentProjectionError as exc:
                raise CommandServiceError(exc.code) from exc
            return content, revision, document.revision_digest or ""

    def _propose(self, handle: CommandHandle, envelope: CommandEnvelope, actor: ActorContext) -> dict[str, Any]:
        if envelope.command_type in {"save_scene", "undo", "redo", "import_image"}:
            return refusal_receipt(code="UNSUPPORTED_COMMAND", envelope=envelope,
                                   actor_kind=actor.actor_kind, actor_id=actor.actor_id,
                                   current_revision=handle.document.current_revision)
        action = _action_payload(envelope, actor, None)
        try:
            if envelope.command_type == "batch":
                prepared = prepare_action(
                    handle.document, handle.store.assets, action, actor_id=actor.actor_id,
                    actor_kind=actor.actor_kind, command_id=envelope.command_id,
                    transaction_id=envelope.transaction["transactionId"], group_id=envelope.transaction["groupId"],
                )
            else:
                prepared = prepare_action(
                    handle.document, handle.store.assets, action, actor_id=actor.actor_id,
                    actor_kind=actor.actor_kind, command_id=envelope.command_id,
                    transaction_id=envelope.transaction["transactionId"], group_id=envelope.transaction["groupId"],
                )
        except EditorActionError as exc:
            return refusal_receipt(code=exc.code, envelope=envelope,
                                   actor_kind=actor.actor_kind, actor_id=actor.actor_id,
                                   current_revision=exc.current_revision or handle.document.current_revision)
        except (SchemaValidationError, HistoryValidationError, TypeError, ValueError):
            return refusal_receipt(code="VALIDATION_FAILED", envelope=envelope,
                                   actor_kind=actor.actor_kind, actor_id=actor.actor_id,
                                   current_revision=handle.document.current_revision)
        _ = prepared
        return canonical_receipt(
            status="proposed", command_id=envelope.command_id, document_id=envelope.document_id,
            actor_kind=actor.actor_kind, actor_id=actor.actor_id, intent="propose", previous_revision=handle.document.current_revision,
            new_revision=None, current_revision=handle.document.current_revision,
            observed_revision=handle.document.current_revision,
            transaction_id=envelope.transaction["transactionId"], target_ids=envelope.target_ids,
            affected_ids=list(envelope.target_ids), created_ids=[], invalidated=[], conflicts=[], writes=0,
            result={
                "commandType": envelope.command_type,
                "normalizedIntent": deepcopy(dict(envelope.payload)),
                "predictedCategories": _predicted_categories(envelope.command_type),
            },
        )

    def _mutation_receipt(self, envelope: CommandEnvelope, actor: ActorContext, prepared: PreparedAction) -> dict[str, Any]:
        old = prepared.receipt
        previous_revision = old.get("previousRevision")
        new_revision = old.get("newRevision")
        created = list(old.get("createdIds", []))
        affected = list(old.get("affectedIds", []))
        invalidated = _invalidated_records(prepared.document, list(old.get("invalidatedDerivedIds", [])))
        receipt = canonical_receipt(
            status="committed", command_id=envelope.command_id, document_id=envelope.document_id,
            actor_kind=actor.actor_kind, actor_id=actor.actor_id, intent="mutate", previous_revision=previous_revision,
            new_revision=new_revision, current_revision=new_revision, observed_revision=previous_revision,
            transaction_id=envelope.transaction["transactionId"], target_ids=envelope.target_ids,
            affected_ids=affected, created_ids=created, invalidated=invalidated, conflicts=[], writes=1,
            result=old,
        )
        # W03 browser payload/receipt aliases remain available through this thin adapter.
        receipt.update({
            "actionId": old.get("actionId"),
            "groupId": old.get("groupId"),
            "affectedIds": affected,
            "invalidatedDerivedIds": list(old.get("invalidatedDerivedIds", [])),
            "code": None,
        })
        return receipt

    def _saved_receipt(self, envelope: CommandEnvelope, actor: ActorContext, revision: int) -> dict[str, Any]:
        receipt = canonical_receipt(
            status="saved", command_id=envelope.command_id, document_id=envelope.document_id,
            actor_kind=actor.actor_kind, actor_id=actor.actor_id, intent="mutate",
            previous_revision=revision, new_revision=revision, current_revision=revision,
            observed_revision=revision, transaction_id=envelope.transaction["transactionId"],
            target_ids=[], affected_ids=[], created_ids=[], invalidated=[], conflicts=[], writes=1,
            result={"savedRevision": revision, "checkpoint": True},
        )
        receipt.update({"revision": revision, "committedRevision": revision, "dirty": False,
                        "checkpoint": True, "code": None})
        return receipt

    def _save(
        self, handle: CommandHandle, envelope: CommandEnvelope, actor: ActorContext, digest: str,
    ) -> dict[str, Any]:
        current_revision = handle.document.current_revision
        try:
            revision, revision_digest = handle.store.save_fingerprint(handle.document)
            intent = {
                "recordVersion": 1,
                "status": "pending",
                "commandId": envelope.command_id,
                "requestDigest": digest,
                "documentId": envelope.document_id,
                "actorKind": actor.actor_kind,
                "actorId": actor.actor_id,
                "transactionId": envelope.transaction["transactionId"],
                "revision": revision,
                "revisionDigest": revision_digest,
                "receipt": None,
            }
            # The immutable intent survives service replacement. It is
            # completed by the exact checkpoint fingerprint, including if a
            # process stops after publishing the checkpoint.
            handle.store.write_semantic_save_pending(intent)
        except Exception:
            return refusal_receipt(code="SAVE_FAILED", envelope=envelope,
                                   actor_kind=actor.actor_kind, actor_id=actor.actor_id,
                                   current_revision=current_revision)

        def failed() -> dict[str, Any]:
            try:
                handle.store.remove_semantic_save_pending(envelope.command_id, digest)
            except Exception:
                pass
            return refusal_receipt(code="SAVE_FAILED", envelope=envelope,
                                   actor_kind=actor.actor_kind, actor_id=actor.actor_id,
                                   current_revision=handle.document.current_revision)

        def saved() -> dict[str, Any]:
            receipt = self._saved_receipt(envelope, actor, revision)
            try:
                handle.store.write_semantic_save_receipt({**intent, "status": "saved", "receipt": deepcopy(receipt)})
            except Exception:
                # Pending intent plus a matching immutable checkpoint still
                # reconstructs this exact receipt on the next attempt.
                pass
            if handle.document.current_revision == revision:
                handle.committed_revision = revision
            return receipt

        try:
            if not handle.store.checkpoint_matches(revision, revision_digest):
                handle.store.save(handle.document, checkpoint=True)
            if not handle.store.checkpoint_matches(revision, revision_digest):
                return failed()
        except Exception:
            # A failure after immutable checkpoint publication is still a
            # committed save. The intent and checkpoint are its witness.
            try:
                if handle.store.checkpoint_matches(revision, revision_digest):
                    return saved()
            except Exception:
                pass
            return failed()
        return saved()


def _predicted_categories(command_type: str) -> list[str]:
    if command_type == "batch":
        return ["document-state", "history-entry"]
    if command_type in {"add_layer", "duplicate_layer"}:
        return ["layer", "history-entry"]
    if command_type in {"create_object", "create_guide", "duplicate_selection_to_layer"}:
        return ["editable-object", "layer", "history-entry"]
    if command_type in {"create_selection", "refine_selection", "rebase_selection"}:
        return ["selection", "mask-asset", "history-entry"]
    if command_type == "create_context_mask":
        return ["context-mask", "mask-asset", "history-entry"]
    if command_type in {"undo", "redo"}:
        return ["document-state", "history-entry"]
    return ["document-state", "history-entry"]


def human_envelope_from_w03(payload: Mapping[str, Any], *, document_id: str | None = None) -> dict[str, Any]:
    """Normalize the accepted browser action body into the v1 envelope."""

    if not isinstance(payload, Mapping):
        raise CommandServiceError("INVALID_ENVELOPE")
    _validate_envelope_limits(dict(payload))
    allowed = {"documentId", "expectedRevision", "actorKind", "transactionId", "groupId",
               "commandId", "actionType", "targetIds", "data", "actions"}
    if set(payload) - allowed:
        raise CommandServiceError("UNKNOWN_FIELD")
    legacy_actor = payload.get("actorKind", "director")
    if not isinstance(legacy_actor, str) or legacy_actor not in {"human", "director"}:
        raise CommandServiceError("INVALID_ACTOR")
    command_type = payload.get("actionType")
    if not isinstance(command_type, str):
        raise CommandServiceError("INVALID_ENVELOPE")
    if document_id is not None and payload.get("documentId", document_id) != document_id:
        raise CommandServiceError("UNKNOWN_DOCUMENT")
    doc_id = document_id or payload.get("documentId")
    try:
        validate_id(doc_id, field="documentId")
    except (TypeError, ValueError) as exc:
        raise CommandServiceError("INVALID_ENVELOPE") from exc
    expected = payload.get("expectedRevision")
    if type(expected) is not int or expected < 0:
        raise CommandServiceError("INVALID_REVISION")
    raw_targets = _validate_target_ids(payload.get("targetIds", []))
    if command_type == "batch":
        actions = payload.get("actions")
        if not isinstance(actions, list):
            raise CommandServiceError("INVALID_BATCH")
        normalized_actions: list[dict[str, Any]] = []
        for action in actions:
            if isinstance(action, Mapping):
                normalized = deepcopy(dict(action))
                action_type = normalized.get("actionType")
                data = action.get("data", {})
                if isinstance(data, Mapping):
                    if action_type == "transform" and "targetIds" in data:
                        data_copy = deepcopy(dict(data))
                        data_targets = _validate_target_ids(data_copy.pop("targetIds"))
                        action_targets = _validate_target_ids(normalized.get("targetIds", []))
                        if action_targets and action_targets != data_targets:
                            raise CommandServiceError("TARGET_MISMATCH")
                        normalized["targetIds"] = list(action_targets or data_targets)
                        normalized["data"] = data_copy
                    elif action_type == "transform":
                        normalized["targetIds"] = list(_validate_target_ids(normalized.get("targetIds", [])))
                normalized_actions.append(normalized)
            else:
                normalized_actions.append(deepcopy(action))
        command_payload: dict[str, Any] = {"actions": normalized_actions}
    elif command_type in {"undo", "redo"}:
        command_payload = {}
    else:
        data = payload.get("data", {})
        if not isinstance(data, Mapping):
            raise CommandServiceError("INVALID_PAYLOAD")
        normalized_data = deepcopy(dict(data))
        if command_type == "transform" and "targetIds" in normalized_data:
            data_targets = _validate_target_ids(normalized_data.pop("targetIds"))
            if raw_targets and raw_targets != data_targets:
                raise CommandServiceError("TARGET_MISMATCH")
            raw_targets = raw_targets or data_targets
        command_payload = {"data": normalized_data}
    if command_type == "batch":
        derived_targets = _canonical_target_ids(command_type, command_payload, ())
        if raw_targets and raw_targets != derived_targets:
            raise CommandServiceError("TARGET_MISMATCH")
    elif command_type == "transform":
        derived_targets = _canonical_target_ids(command_type, command_payload, raw_targets)
    else:
        derived_targets = _canonical_target_ids(command_type, command_payload, ())
        if raw_targets and raw_targets != derived_targets:
            raise CommandServiceError("TARGET_MISMATCH")
    transaction_id = payload.get("transactionId") or make_id("txn")
    group_id = payload.get("groupId") or make_id("grp")
    command_id = payload.get("commandId") or make_id("cmd")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "commandId": command_id,
        "documentId": doc_id,
        "intent": "mutate",
        "expectedRevision": expected,
        "commandType": command_type,
        "targetIds": list(derived_targets),
        "coordinateSpace": "document",
        "payload": command_payload,
        "transaction": {"transactionId": transaction_id, "groupId": group_id, "phase": "commit"},
    }


def _action_references(command_type: str, data: Mapping[str, Any]) -> list[str]:
    fields = {
        "rename_layer": ("layerId",), "set_layer_visibility": ("layerId",), "set_layer_lock": ("layerId",),
        "set_layer_opacity": ("layerId",), "reorder_layer": ("layerId",),
        "reparent_layer": ("layerId", "parentId"), "duplicate_layer": ("layerId",),
        "delete_layer": ("layerId",), "create_object": ("layerId",),
        "create_selection": ("sourceLayerId",), "refine_selection": ("selectionId",),
        "rebase_selection": ("selectionId",), "duplicate_selection_to_layer": ("selectionId", "parentId"),
        "create_context_mask": ("sourceLayerId",), "set_relational_context": ("contextMaskId",),
        "transition_guide": ("guideId", "replacementId"),
    }.get(command_type, ())
    values = [data[field] for field in fields if isinstance(data.get(field), str)]
    if command_type == "set_relational_context":
        for field in ("referenceIds", "editTargetIds"):
            if isinstance(data.get(field), list):
                values.extend(value for value in data[field] if isinstance(value, str))
    return values


command_service = CommandService()
