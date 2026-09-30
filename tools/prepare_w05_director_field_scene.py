"""Install the fixed W05 validation fixture in a Director-owned blank scene.

Create and save the blank scene in the Director's normal W03 browser first.
Stop the app before running this offline, test-only fixture initializer. It
preserves the existing W03 owner descriptor byte-for-byte and never reads an
owner key or writes a principal to the receipt.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


OWNER_PRINCIPAL_RE = re.compile(r"^[0-9a-f]{64}$")
BEARER_PATTERNS = (
    re.compile(rb"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    re.compile(rb'(?i)"(?:token|accessToken|capabilityToken)"\s*:\s*"[A-Za-z0-9._~+/=-]{16,}"'),
)


class FieldScenePreparationError(RuntimeError):
    """A bounded, redacted failure while preparing a Director field scene."""


def _load_w05_fixture_builder(repo: Path) -> Callable[..., tuple[str, dict[str, Any]]]:
    browser_dir = repo / "tests" / "browser"
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(browser_dir))
    from run_creative_document_field_validation import _fixture_scene

    return _fixture_scene


def _load_w03_helpers(repo: Path) -> Any:
    browser_dir = repo / "tests" / "browser"
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(browser_dir))
    import run_creative_document_editor_audit as w03

    return w03


def _read_owner_descriptor(owner_path: Path, document_id: str) -> bytes:
    if owner_path.is_symlink() or not owner_path.is_file() or owner_path.stat().st_size > 1024:
        raise FieldScenePreparationError("a saved W03 owner descriptor is required")
    try:
        raw = owner_path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise FieldScenePreparationError("a saved W03 owner descriptor is required") from exc
    if (not isinstance(value, dict) or set(value) != {"version", "documentId", "principalId"}
            or value.get("version") != 1 or value.get("documentId") != document_id
            or not isinstance(value.get("principalId"), str)
            or not OWNER_PRINCIPAL_RE.fullmatch(value["principalId"])):
        raise FieldScenePreparationError("the W03 owner descriptor is malformed")
    return raw


def _require_blank_scene(project_path: Path, document_id: str) -> None:
    from modules.creative_document import ProjectStore

    try:
        document = ProjectStore(project_path).open()
    except Exception as exc:
        raise FieldScenePreparationError("the saved blank W03 scene could not be opened") from exc
    roots = [document.layers.get(identity) for identity in document.root_layer_ids]
    empty_records = (
        document.objects, document.assets, document.masks, document.selections, document.variants,
        document.interaction_groups, document.operations, document.depth_composites, document.bb_operations,
        document.candidates, document.extractions, document.guides, document.patches,
        document.external_round_trips, document.private_proxies, document.history,
    )
    if (document.document_id != document_id or document.current_revision != 0
            or len(document.layers) != 1 or len(roots) != 1 or roots[0] is None or roots[0].kind != "paint"
            or roots[0].object_ids or roots[0].mask_ids
            or any(records for records in empty_records)
            or document.metadata):
        raise FieldScenePreparationError("only a saved, empty revision-0 W03 scene can be initialized")


def _scan_scene(project_path: Path, asset_root: Path) -> int:
    sentinels = {
        str(asset_root).lower().encode("utf-8"),
        str(asset_root).replace("\\", "/").lower().encode("utf-8"),
        b"assets/images/reference",
    }
    checked = 0
    for path in project_path.rglob("*"):
        if not path.is_file():
            continue
        checked += 1
        raw = path.read_bytes()
        lowered = raw.lower()
        if any(sentinel and sentinel in lowered for sentinel in sentinels):
            raise FieldScenePreparationError("the W05 scene contains a private source path")
        if any(pattern.search(raw) for pattern in BEARER_PATTERNS):
            raise FieldScenePreparationError("the W05 scene contains bearer-like data")
    return checked


def prepare_director_scene(
    *,
    document_id: str,
    output_root: Path,
    asset_root: Path,
    fixture_builder: Callable[..., tuple[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Populate a saved blank W03 scene while preserving its owner sidecar."""
    from modules.creative_document import validate_id

    try:
        validate_id(document_id, field="documentId")
    except ValueError as exc:
        raise FieldScenePreparationError("the Director scene ID is malformed") from exc

    output_root = output_root.resolve()
    projects_root = (output_root / "creative_documents").resolve()
    projects_root.mkdir(parents=True, exist_ok=True)
    project_candidate = projects_root / f"{document_id}.nexscene"
    project_path = project_candidate.resolve()
    owner_path = projects_root / f"{document_id}.owner.json"
    if (project_candidate.is_symlink() or project_candidate.parent.resolve() != projects_root
            or project_path.parent != projects_root or project_path.name != f"{document_id}.nexscene"):
        raise FieldScenePreparationError("the scene escaped the configured project root")
    if not project_path.is_dir():
        raise FieldScenePreparationError("the Director must create and save the blank scene in W03 first")

    owner_before = _read_owner_descriptor(owner_path, document_id)
    _require_blank_scene(project_path, document_id)
    if fixture_builder is None:
        fixture_builder = _load_w05_fixture_builder(Path(__file__).resolve().parents[1])

    staging_root = Path(tempfile.mkdtemp(prefix=f".w05-field-prep-{document_id}-", dir=projects_root)).resolve()
    staged_project: Path | None = None
    backup_project = staging_root / "blank-scene-backup.nexscene"
    installed = False
    try:
        built_id, fixture_info = fixture_builder(
            output_root, asset_root.resolve(), None, document_id=document_id,
            projects_root_override=staging_root,
        )
        if built_id != document_id:
            raise FieldScenePreparationError("the fixture builder returned a mismatched scene ID")
        staged_project = (staging_root / f"{document_id}.nexscene").resolve()
        if staged_project.parent != staging_root or not staged_project.is_dir():
            raise FieldScenePreparationError("the fixture builder escaped its staging directory")
        from modules.creative_document import ProjectStore
        prepared = ProjectStore(staged_project).open()
        if prepared.document_id != document_id or prepared.metadata.get("validationFixture") != "P6-M01-W05":
            raise FieldScenePreparationError("the staged project is not the explicit W05 fixture")
        from tools.w05_field_state import build_w05_field_state, validate_w05_field_state

        field_state = build_w05_field_state(staged_project, fixture_info, repo=Path(__file__).resolve().parents[1])
        prepared = ProjectStore(staged_project).open()
        verified_field_state = validate_w05_field_state(
            prepared, fixture_info, actor_id=field_state["preparation"]["actorId"],
        )
        field_state_keys = ("state", "documentId", "revision", "historyTransactions", "preparation",
                            "ids", "guides", "context", "depth", "history")
        if any(verified_field_state[key] != field_state[key] for key in field_state_keys):
            raise FieldScenePreparationError("the staged W05 field state failed its replay verification")
        project_files_scanned = _scan_scene(staged_project, asset_root)

        os.replace(project_path, backup_project)
        try:
            os.replace(staged_project, project_path)
            installed = True
            installed_document = ProjectStore(project_path).open()
            if (installed_document.document_id != document_id
                    or installed_document.metadata.get("validationFixture") != "P6-M01-W05"
                    or _read_owner_descriptor(owner_path, document_id) != owner_before
                    or _scan_scene(project_path, asset_root) != project_files_scanned):
                raise FieldScenePreparationError("the installed W05 scene or owner descriptor failed verification")
        except Exception:
            if installed and project_path.exists():
                os.replace(project_path, staged_project)
                installed = False
            if backup_project.exists():
                os.replace(backup_project, project_path)
            raise

        shutil.rmtree(backup_project)
        project_relative_path = project_path.relative_to(output_root).as_posix()
        owner_relative_path = owner_path.relative_to(output_root).as_posix()
        return {
            "documentId": document_id,
            "revision": installed_document.current_revision,
            "ids": fixture_info["ids"],
            "fixtures": fixture_info["fixtures"],
            "fixtureConstruction": fixture_info["construction"],
            "fieldState": field_state,
            "projectRelativePath": project_relative_path,
            "ownerDescriptorPreservedByteForByte": True,
            "ownerPrincipalRecorded": False,
            "privateSourcePathAbsent": True,
            "bearerPatternAbsent": True,
            "projectFilesScanned": project_files_scanned,
            "retainedForDirector": True,
            "boundedCleanup": f"Remove only {project_path} recursively and {owner_path}.",
            "ownerDescriptorRelativePath": owner_relative_path,
        }
    finally:
        if staging_root.exists():
            resolved_staging = staging_root.resolve()
            if resolved_staging.parent != projects_root or not staging_root.name.startswith(f".w05-field-prep-{document_id}-"):
                raise FieldScenePreparationError("staging cleanup escaped its exact W05 directory")
            shutil.rmtree(staging_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--document-id", required=True,
                        help="opaque ID of a blank scene created and saved in the Director's W03 browser")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--private-asset-root", type=Path,
                        help="exact directory containing the five named W05 validation files")
    parser.add_argument("--app-stopped", action="store_true", required=True,
                        help="confirm the local app is stopped before offline fixture installation")
    args = parser.parse_args()

    repo = args.repo.resolve()
    receipt_path = args.receipt.resolve()
    asset_root = (args.private_asset_root or (repo / "assets" / "images" / "reference")).resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "workOrder": "P6-M01-W05",
        "evidence": "Director-owned saved W03 scene; offline test-only fixture initialization",
        "startedAtUtc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "sourceDirectoryRecorded": False,
        "sourceBytesRecorded": False,
        "ownerCredentialRecorded": False,
        "ownerPrincipalRecorded": False,
    }
    try:
        if not args.app_stopped:
            raise FieldScenePreparationError("stop the app before fixture initialization")
        w03 = _load_w03_helpers(repo)
        output_root = w03.read_server_output_root(repo)
        result = prepare_director_scene(
            document_id=args.document_id,
            output_root=output_root,
            asset_root=asset_root,
        )
        receipt["scene"] = result
        receipt["status"] = "passed"
        print(f"Prepared Director-owned W05 scene {args.document_id} at revision {result['revision']}.")
        print(f"Scene: {result['projectRelativePath']}")
        print(f"Bounded cleanup: {result['boundedCleanup']}")
        return_code = 0
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["failure"] = {"type": type(exc).__name__, "message": "See local runner output for the bounded preparation error."}
        return_code = 1
        detail = str(exc) if isinstance(exc, FieldScenePreparationError) else "See the local report for the failed check."
        print(f"W05 Director scene preparation failed: {type(exc).__name__}: {detail}", file=sys.stderr)

    receipt["finishedAtUtc"] = datetime.now(timezone.utc).isoformat()
    try:
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        return_code = 1
    print(f"W05 Director scene preparation receipt: {receipt_path}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
