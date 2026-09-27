from __future__ import annotations

import copy
from modules.creative_document.ids import content_digest

import pytest

from modules.creative_document import Document, LayerRecord, MaskRecord, SCHEMA_VERSION, SelectionRecord, UnsupportedSchemaVersion, migrate_manifest
from modules.creative_document.migrations import MigrationError, migration_fixture_v0


def test_synthetic_migration_is_ordered_and_idempotent() -> None:
    migrated = migrate_manifest(migration_fixture_v0())
    assert migrated["schemaVersion"] == SCHEMA_VERSION
    assert migrated["formatId"] == "nexfocus.nexscene"
    assert migrate_manifest(migrated) == migrated
    assert Document.from_dict(migrated).document_id == "doc-fixture-0"


def test_v1_selection_migration_adds_provenance_and_recomputes_canonical_digest() -> None:
    document = Document("doc-v1", 8, 8, root_layer_ids=["layer"],
                        layers={"layer": LayerRecord("layer", "Layer", "raster")},
                        masks={"mask": MaskRecord("mask", "editing-selection", "document", 0, editable_source={"shape": []})},
                        selections={"selection": SelectionRecord("selection", "layer", 0, "a" * 64,
                                                                  1, "current", "mask")})
    old = document.to_dict()
    old["schemaVersion"] = 1
    for key in ("invalidationHistory", "rebaseHistory", "duplicateHistory"):
        old["selections"]["selection"].pop(key)
    old.pop("revisionDigest", None)
    old["revisionDigest"] = content_digest(old)
    migrated = migrate_manifest(old)
    assert migrated["schemaVersion"] == SCHEMA_VERSION
    assert Document.from_dict(migrated).canonical_digest() == migrated["revisionDigest"]
    assert old["schemaVersion"] == 1


def test_future_schema_and_failed_migration_preserve_source() -> None:
    future = {"schemaVersion": 99}
    with pytest.raises(UnsupportedSchemaVersion):
        migrate_manifest(future)
    assert future == {"schemaVersion": 99}

    bad = copy.deepcopy(migration_fixture_v0())
    bad["formatId"] = "unknown-format"
    with pytest.raises(MigrationError):
        migrate_manifest(bad)
    assert bad["formatId"] == "unknown-format"
