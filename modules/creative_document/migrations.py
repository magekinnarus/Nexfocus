"""Explicit, ordered, non-destructive schema migration registry."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping

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
        target = self.current_version if target_version is None else int(target_version)
        result = deepcopy(dict(source))
        version = int(result.get("schemaVersion", 0))
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


DEFAULT_MIGRATIONS = MigrationRegistry()
DEFAULT_MIGRATIONS.register(0, 1, _v0_to_v1)


def migrate_manifest(source: Mapping[str, Any], target_version: int = SCHEMA_VERSION) -> dict[str, Any]:
    return DEFAULT_MIGRATIONS.migrate(source, target_version)


def migration_fixture_v0() -> dict[str, Any]:
    return {"schemaVersion": 0, "formatId": "nexfocus.scene", "document_id": "doc-fixture-0", "width": 8, "height": 8}
