from __future__ import annotations

import copy

import pytest

from modules.creative_document import Document, UnsupportedSchemaVersion, migrate_manifest
from modules.creative_document.migrations import MigrationError, migration_fixture_v0


def test_synthetic_migration_is_ordered_and_idempotent() -> None:
    migrated = migrate_manifest(migration_fixture_v0())
    assert migrated["schemaVersion"] == 1
    assert migrated["formatId"] == "nexfocus.nexscene"
    assert migrate_manifest(migrated) == migrated
    assert Document.from_dict(migrated).document_id == "doc-fixture-0"


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
