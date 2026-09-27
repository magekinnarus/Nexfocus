"""Explicit, ordered, non-destructive schema migration registry."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .ids import content_digest
from .schema import FORMAT_IDENTIFIER, SCHEMA_VERSION


class MigrationError(ValueError):
    """Base migration failure."""


class UnsupportedSchemaVersion(MigrationError):
    """Raised for a future major/schema version with no safe migration."""


Migration = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class MigrationStep:
    from_version: int
    to_version: int
    migrate: Migration


class MigrationRegistry:
    def __init__(self, current_version: int = SCHEMA_VERSION) -> None:
        self.current_version = current_version
        self._steps: dict[int, MigrationStep] = {}

    def register(self, from_version: int, to_version: int, migrate: Migration) -> None:
        if to_version <= from_version:
            raise MigrationError("migrations must move forward")
        if from_version in self._steps:
            raise MigrationError(f"duplicate migration from {from_version}")
        self._steps[from_version] = MigrationStep(from_version, to_version, migrate)

    def migrate(self, source: Mapping[str, Any], target_version: int | None = None) -> dict[str, Any]:
        if target_version is not None and type(target_version) is not int:
            raise UnsupportedSchemaVersion("target schema version must be an integer")
        target = self.current_version if target_version is None else target_version
        result = deepcopy(dict(source))
        version = result.get("schemaVersion", 0)
        if type(version) is not int or version < 0:
            raise UnsupportedSchemaVersion("source schema version must be a non-negative integer")
        if version > target:
            raise UnsupportedSchemaVersion(f"schema {version} is newer than supported {target}")
        while version < target:
            step = self._steps.get(version)
            if step is None or step.to_version > target:
                raise UnsupportedSchemaVersion(f"no migration path from schema {version} to {target}")
            candidate = deepcopy(result)
            try:
                migrated = step.migrate(candidate)
            except Exception as exc:
                raise MigrationError(f"migration {version}->{step.to_version} failed: {exc}") from exc
            if not isinstance(migrated, dict):
                raise MigrationError("migration must return an object")
            migrated["schemaVersion"] = step.to_version
            result = migrated
            version = step.to_version
        return result


def _v0_to_v1(source: dict[str, Any]) -> dict[str, Any]:
    """Small executable synthetic fixture migration.

    Version zero used snake_case for its small identity header and omitted the
    empty typed collections. It is intentionally narrow: ambiguous legacy
    meaning still fails instead of being guessed.
    """

    if source.get("formatId") not in {None, FORMAT_IDENTIFIER, "nexfocus.scene"}:
        raise MigrationError("unknown legacy format")
    if "documentId" not in source and "document_id" in source:
        source["documentId"] = source.pop("document_id")
    source["formatId"] = FORMAT_IDENTIFIER
    source.setdefault("colorSpaceIntent", source.pop("color_space_intent", "sRGB"))
    source.setdefault("currentRevision", source.pop("current_revision", 0))
    source.setdefault("rootLayerIds", source.pop("root_layer_ids", []))
    collections = {
        "layers": {}, "objects": {}, "assets": {}, "masks": {}, "selections": {}, "variants": {},
        "interactionGroups": {}, "operations": {}, "depthComposites": {}, "bbOperations": {},
        "candidates": {}, "extractions": {}, "guides": {}, "patches": {}, "externalRoundTrips": {},
        "privateProxies": {}, "visualProfile": {"palette": {}, "lighting": {}, "style": {}, "prompts": [], "modelChoices": {}, "inferenceSettings": {}},
        "history": [], "historyRefs": [], "checkpointRefs": [], "metadata": {},
    }
    for key, default in collections.items():
        source.setdefault(key, default)
    source.setdefault("width", 1)
    source.setdefault("height", 1)
    return source


def _v1_to_v2(source: dict[str, Any]) -> dict[str, Any]:
    """Add explicit selection invalidation/rebase/duplicate provenance."""

    selections = source.get("selections")
    if not isinstance(selections, dict):
        raise MigrationError("schema v1 selections must be an object")
    for selection in selections.values():
        if not isinstance(selection, dict):
            raise MigrationError("schema v1 selection record must be an object")
        if "duplicateHistory" not in selection:
            selection["duplicateHistory"] = _derive_v1_duplicate_history(selection)
        selection.setdefault("invalidationHistory", [])
        selection.setdefault("rebaseHistory", [])
    source.pop("revisionDigest", None)
    source["schemaVersion"] = 2
    source["revisionDigest"] = content_digest(source)
    return source


def _derive_v1_duplicate_history(selection: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct only duplicate transitions uniquely determined by v1 data."""

    selection_revision = selection.get("selectionRevision")
    derived_layer_ids = selection.get("derivedLayerIds", [])
    if type(selection_revision) is not int or selection_revision < 1:
        raise MigrationError("schema v1 selection revision is invalid")
    if (not isinstance(derived_layer_ids, list)
            or not all(isinstance(layer_id, str) for layer_id in derived_layer_ids)
            or len(set(derived_layer_ids)) != len(derived_layer_ids)):
        raise MigrationError("schema v1 derived layer links are malformed or duplicated")

    occupied: set[int] = set()
    for history_name in ("refinementHistory", "rebaseHistory"):
        history = selection.get(history_name, [])
        if not isinstance(history, list):
            raise MigrationError(f"schema v1 {history_name} must be an array")
        for entry in history:
            if not isinstance(entry, dict):
                raise MigrationError(f"schema v1 {history_name} contains a malformed transition")
            previous = entry.get("previousSelectionRevision")
            next_revision = entry.get("selectionRevision")
            if (type(previous) is not int or type(next_revision) is not int
                    or previous < 1 or next_revision != previous + 1
                    or next_revision > selection_revision or previous in occupied):
                raise MigrationError(f"schema v1 {history_name} does not identify an unambiguous revision transition")
            occupied.add(previous)

    transition_slots = list(range(1, selection_revision))
    missing_slots = [revision for revision in transition_slots if revision not in occupied]
    if len(missing_slots) != len(derived_layer_ids):
        raise MigrationError(
            "schema v1 selection history cannot unambiguously reconcile revision-advancing derived links"
        )
    return [
        {"previousSelectionRevision": previous, "selectionRevision": previous + 1, "layerId": layer_id}
        for previous, layer_id in zip(missing_slots, derived_layer_ids)
    ]


DEFAULT_MIGRATIONS = MigrationRegistry()
DEFAULT_MIGRATIONS.register(0, 1, _v0_to_v1)
DEFAULT_MIGRATIONS.register(1, 2, _v1_to_v2)


def migrate_manifest(source: Mapping[str, Any], target_version: int = SCHEMA_VERSION) -> dict[str, Any]:
    return DEFAULT_MIGRATIONS.migrate(source, target_version)


def migration_fixture_v0() -> dict[str, Any]:
    return {"schemaVersion": 0, "formatId": "nexfocus.scene", "document_id": "doc-fixture-0", "width": 8, "height": 8}
