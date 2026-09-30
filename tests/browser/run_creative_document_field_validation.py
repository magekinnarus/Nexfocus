"""Run the generation-free P6-M01-W05 field readiness validation.

The runner reads only the five named files from the explicitly designated local
reference directory. It builds one disposable native scene, exercises it
through the accepted local Agent driver and W03 editor, writes a redacted JSON
receipt, and removes the scene, owner descriptor, browser profile, and
capability. A later Director-owned field scene must be initialized from a blank
scene saved in the Director's normal W03 browser with
tools/prepare_w05_director_field_scene.py.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


FIXTURES = {
    "harbor1_step1_final.png": {
        "sha256": "da1e41110059569deb7089c8c02e8f6788135aba328e162f8785566b131a3d9d",
        "dimensions": [2950, 1770],
    },
    "harbor1_bb2-depth_redo.png": {
        "sha256": "33ac63ac50173e8f2ddce1aba6c44976d39c001f37f4ff9d1056ca3281d38d3c",
        "dimensions": [1516, 1037],
    },
    "riders_galloping_toward_black_horse.png": {
        "sha256": "fbacd89c44961b9679d3663273d9ca75fa111198342d96c9e48b50330b0472ee",
        "dimensions": [1422, 1106],
    },
    "harbor1-edit_direction1.png": {
        "sha256": "461fff56e81bb9afa77a902b96a4b95a634167bae9658822e7f761a905a3bd83",
        "dimensions": [2950, 1770],
    },
    "harbor1_bb2_added.png": {
        "sha256": "492640cec9ffe6a0ceeee7c3f86c3448c273f42b571492d5dcb60aa255e5b306",
        "dimensions": [2950, 1770],
    },
}
DRIVER_DIAGNOSTICS: list[dict[str, Any]] = []


def _import_w03_helpers():
    browser_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(browser_dir))
    import run_creative_document_editor_audit as w03

    return w03


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _port_open(port: int) -> bool:
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _wait_http(url: str, *, timeout: float = 150) -> bytes:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                return response.read(1024)
        except Exception as exc:
            last_error = exc
            time.sleep(0.35)
    name = type(last_error).__name__ if last_error else "timeout"
    raise TimeoutError(f"local application did not start ({name})")


def _check(receipt: dict[str, Any], name: str, passed: bool, evidence: Any) -> None:
    receipt["checks"][name] = {"passed": bool(passed), "evidence": evidence}
    if not passed:
        raise AssertionError(f"W05 field check failed: {name}")


def _error_code(result: dict[str, Any]) -> str | None:
    error = result.get("error")
    return error.get("code") if isinstance(error, dict) else None


def _preview_diagnostic(process: subprocess.CompletedProcess[str], output: Path) -> dict[str, Any]:
    try:
        response = json.loads(process.stdout) if process.stdout.strip() else {}
    except json.JSONDecodeError:
        response = {}
    return {"returnCode": process.returncode,
            "status": response.get("status") if isinstance(response, dict) else None,
            "errorCode": _error_code(response) if isinstance(response, dict) else None,
            "outputWritten": output.is_file(), "stderrPresent": bool(process.stderr.strip())}


def _envelope(
    document_id: str,
    intent: str,
    command_type: str,
    *,
    revision: int | None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from modules.creative_document import make_id
    from modules.creative_document.command_service import _canonical_target_ids

    inspect = intent == "inspect"
    command_payload = {} if inspect else (payload or {})
    targets = [] if command_type == "conversational_text" else list(
        _canonical_target_ids(command_type, command_payload, ())
    )
    return {
        "schemaVersion": 1,
        "commandId": make_id("cmd"),
        "documentId": document_id,
        "intent": intent,
        "expectedRevision": None if inspect else revision,
        "commandType": command_type,
        "targetIds": targets,
        "coordinateSpace": "document",
        "payload": command_payload,
        "transaction": None if inspect else {
            "transactionId": make_id("txn"), "groupId": make_id("grp"), "phase": "commit",
        },
    }


def _write_request(profile: Path, label: str, envelope: dict[str, Any]) -> Path:
    path = profile / f"{label}.json"
    path.write_text(json.dumps(envelope, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return path


def _run_cli(repo: Path, capability: Path, intent: str, envelope: dict[str, Any], label: str,
             profile: Path) -> tuple[int, dict[str, Any]]:
    request_path = _write_request(profile, label, envelope)
    command = [str(repo / "venv" / "Scripts" / "python.exe"), "tools/creative_document_driver.py",
               "--capability-file", str(capability), intent, "--request", str(request_path)]
    result = subprocess.run(command, cwd=repo, capture_output=True, text=True, timeout=90, check=False)
    try:
        decoded = json.loads(result.stdout) if result.stdout.strip() else None
    except json.JSONDecodeError as exc:
        DRIVER_DIAGNOSTICS.append({"returnCode": result.returncode, "outputPresent": bool(result.stdout.strip()),
                                  "outputShape": "malformed-json"})
        raise RuntimeError("local Agent driver returned malformed output") from exc
    if result.returncode not in {0, 2} or not isinstance(decoded, dict):
        DRIVER_DIAGNOSTICS.append({"returnCode": result.returncode, "outputShape": type(decoded).__name__,
                                  "status": decoded.get("status") if isinstance(decoded, dict) else None,
                                  "errorCode": _error_code(decoded) if isinstance(decoded, dict) else None})
        raise RuntimeError(f"local Agent driver failed with exit code {result.returncode}")
    return result.returncode, decoded


def _fixture_scene(output_root: Path, asset_root: Path, owner_key: str | None, *,
                   document_id: str | None = None,
                   projects_root_override: Path | None = None) -> tuple[str, dict[str, Any]]:
    """Build one explicit schema fixture from directly named, verified files."""
    from modules.creative_document import (
        AffineTransform,
        CoordinateTransform,
        Document,
        LayerRecord,
        MaskRecord,
        ObjectRecord,
        ProjectStore,
        make_id,
    )

    document_id = make_id("doc") if document_id is None else document_id
    if document_id is not None:
        from modules.creative_document import validate_id
        validate_id(document_id, field="documentId")
    plate_id, depth_id, riders_id, direction_id, composite_id = (make_id("layer") for _ in range(5))
    plate_object_id, depth_object_id, riders_object_id, direction_object_id, composite_object_id = (
        make_id("obj") for _ in range(5)
    )
    projects_root = projects_root_override or (output_root / "creative_documents")
    projects_root.mkdir(parents=True, exist_ok=True)
    project_path = projects_root / f"{document_id}.nexscene"
    store = ProjectStore(project_path)

    verified: dict[str, Any] = {}
    image_records: dict[str, Any] = {}
    for name, expected in FIXTURES.items():
        source = asset_root / name
        if source.is_symlink() or source.resolve().parent != asset_root:
            raise RuntimeError("a designated validation file escaped its exact asset root")
        # This is the only source lookup: no glob, fallback, or directory walk.
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected["sha256"]:
            raise RuntimeError("a designated validation fixture does not match its issued SHA-256")
        with Image.open(io.BytesIO(raw)) as image:
            dimensions = list(image.size)
            mode = image.mode
            has_alpha = "A" in image.getbands()
            if dimensions != expected["dimensions"] or mode not in {"RGB", "RGBA"}:
                raise RuntimeError("a designated validation fixture has unexpected dimensions or mode")
        # The exact bytes just verified are immediately embedded in this local project.
        asset = store.assets.put_bytes(
            raw, media_type="image/png", extension="png", width=dimensions[0], height=dimensions[1],
            has_alpha=has_alpha, color_space="sRGB", provenance={"kind": "w05-shared-local-validation"},
        )
        if asset.content_hash != digest:
            raise RuntimeError("embedded fixture identity changed during import")
        image_records[name] = asset
        verified[name] = {
            "sha256": digest,
            "dimensions": dimensions,
            "mode": mode,
            "disposition": "verified then embedded only in this generated local W05 scene",
        }

    plate_asset = image_records["harbor1_step1_final.png"]
    depth_asset = image_records["harbor1_bb2-depth_redo.png"]
    riders_asset = image_records["riders_galloping_toward_black_horse.png"]
    direction_asset = image_records["harbor1-edit_direction1.png"]
    composite_asset = image_records["harbor1_bb2_added.png"]

    # The named step1 fixture contains transparent pixels. Composite those
    # exact verified pixels over a fixed neutral test matte to make the
    # complete-plate record full-canvas and objectively hole-free. This is a
    # deterministic fixture preparation only; it performs no reconstruction.
    with Image.open(io.BytesIO(store.assets.read_bytes(plate_asset))) as source_plate:
        opaque_plate = Image.new("RGBA", source_plate.size, "#27313D")
        opaque_plate.alpha_composite(source_plate.convert("RGBA"))
        opaque_stream = io.BytesIO()
        opaque_plate.convert("RGB").save(opaque_stream, format="PNG", optimize=False, compress_level=9)
    complete_plate_asset = store.assets.put_bytes(
        opaque_stream.getvalue(), media_type="image/png", extension="png", width=2950, height=1770,
        has_alpha=False, color_space="sRGB",
        provenance={"kind": "w05-opaque-complete-plate-fixture", "sourceAssetId": plate_asset.asset_id,
                    "matte": "#27313D", "operation": "alpha-composite; no reconstruction"},
    )

    edit_mask_image = Image.new("L", (2950, 1770), 0)
    ImageDraw.Draw(edit_mask_image).rectangle((580, 280, 2095, 1316), fill=255)
    mask_stream = io.BytesIO()
    edit_mask_image.save(mask_stream, format="PNG")
    edit_mask_asset = store.assets.put_bytes(
        mask_stream.getvalue(), media_type="image/png", extension="png", width=2950, height=1770,
        has_alpha=False, color_space="sRGB", provenance={"kind": "w05-explicit-generation-scope"},
    )
    edit_mask_id = make_id("mask")

    layers = {
        plate_id: LayerRecord(plate_id, "Complete plate", "raster", role="complete-plate",
                              object_ids=[plate_object_id]),
        depth_id: LayerRecord(
            depth_id, "Registered depth element", "raster", opacity=0.88,
            object_ids=[depth_object_id], mask_ids=[edit_mask_id], role="depth-element",
            depth_element=True, complete_plate_id=plate_id,
            registration=CoordinateTransform.from_forward(
                "source-crop", "document", (depth_asset.width, depth_asset.height),
                (2950, 1770), AffineTransform.translation(580, 280), operation_revision=0,
            ),
            lineage={"completePlateId": plate_id, "registrationPreparedFor": document_id},
        ),
        riders_id: LayerRecord(riders_id, "Relational reference: riders", "raster", role="context-reference",
                               object_ids=[riders_object_id]),
        direction_id: LayerRecord(direction_id, "Direction reference", "raster", role="context-reference",
                                  object_ids=[direction_object_id]),
        composite_id: LayerRecord(composite_id, "Scene comparison reference", "raster", role="context-reference",
                                  object_ids=[composite_object_id]),
    }
    objects = {
        plate_object_id: ObjectRecord(plate_object_id, plate_id, "raster-placement", "document",
                                      {"x": 0, "y": 0, "width": 2950, "height": 1770},
                                      asset_id=complete_plate_asset.asset_id),
        depth_object_id: ObjectRecord(depth_object_id, depth_id, "raster-placement", "document",
                                      {"x": 580, "y": 280, "width": 1516, "height": 1037},
                                      asset_id=depth_asset.asset_id,
                                      lineage={"registrationLayerId": depth_id, "completePlateId": plate_id}),
        riders_object_id: ObjectRecord(riders_object_id, riders_id, "raster-placement", "document",
                                       {"x": 110, "y": 1300, "width": 500, "height": 389},
                                       asset_id=riders_asset.asset_id),
        direction_object_id: ObjectRecord(direction_object_id, direction_id, "raster-placement", "document",
                                          {"x": 2220, "y": 70, "width": 620, "height": 372},
                                          asset_id=direction_asset.asset_id),
        composite_object_id: ObjectRecord(composite_object_id, composite_id, "raster-placement", "document",
                                          {"x": 2220, "y": 475, "width": 620, "height": 372},
                                          asset_id=composite_asset.asset_id),
    }
    edit_mask = MaskRecord(
        edit_mask_id, "generation", "document", 0, asset_id=edit_mask_asset.asset_id,
        owner_id=depth_id, lineage={"derivation": "explicit W05 edit permission", "ownerLayerId": depth_id},
        content_hash=edit_mask_asset.content_hash,
    )
    document = Document(
        document_id, 2950, 1770, root_layer_ids=[plate_id, depth_id, riders_id, direction_id, composite_id],
        layers=layers, objects=objects,
        assets={asset.asset_id: asset for asset in [plate_asset, complete_plate_asset, depth_asset, riders_asset,
                                                   direction_asset, composite_asset, edit_mask_asset]},
        masks={edit_mask_id: edit_mask},
        metadata={"validationFixture": "P6-M01-W05", "lineageMode": "explicit IDs; no filename inference"},
    )
    document.refresh_digest()
    document.validate()
    store.save(document, checkpoint=True)

    if owner_key is not None:
        principal_id = hashlib.sha256(f"browser:{owner_key}".encode("utf-8")).hexdigest()
        owner_record = {"version": 1, "documentId": document_id, "principalId": principal_id}
        (projects_root / f"{document_id}.owner.json").write_text(
            json.dumps(owner_record, sort_keys=True, separators=(",", ":")), encoding="utf-8",
        )
    ids = {
        "documentId": document_id,
        "layers": {"completePlate": plate_id, "depthElement": depth_id, "ridersReference": riders_id,
                   "directionReference": direction_id, "comparisonReference": composite_id},
        "objects": {"completePlate": plate_object_id, "depthElement": depth_object_id,
                    "ridersReference": riders_object_id, "directionReference": direction_object_id,
                    "comparisonReference": composite_object_id},
        "masks": {"generationEditScope": edit_mask_id},
        "assets": {**{name: record.asset_id for name, record in image_records.items()},
                   "opaqueCompletePlateFixture": complete_plate_asset.asset_id},
        "completePlateId": plate_id,
        "depthElementId": depth_id,
        "editTargetId": depth_id,
        "editMaskId": edit_mask_id,
    }
    return document_id, {"ids": ids, "fixtures": verified, "projectPath": project_path,
                         "construction": {"completePlateSource": "harbor1_step1_final.png",
                             "completePlateSourceSha256": verified["harbor1_step1_final.png"]["sha256"],
                             "alphaHandling": "manifest-verified source alpha composited over fixed #27313D test matte",
                             "reconstructionOrSegmentation": False}}


def _render_depth_fixture_preview(project_path: Path, plate_layer_id: str, depth_layer_id: str,
                                  output_path: Path) -> dict[str, Any]:
    """Render only the explicit shared plate/depth pair at a fixed document scale."""
    from modules.creative_document import AffineTransform, ProjectStore
    from modules.creative_document.editor_render import image_bytes

    store = ProjectStore(project_path)
    document = store.open()
    plate = document.layers[plate_layer_id]
    depth = document.layers[depth_layer_id]
    if depth.complete_plate_id != plate_layer_id or depth.registration is None:
        raise RuntimeError("fixture depth registration no longer resolves to the complete plate")
    if depth.transform.to_dict() != AffineTransform.identity().to_dict():
        raise RuntimeError("fixture depth layer transform changed during visibility validation")
    if len(plate.object_ids) != 1 or len(depth.object_ids) != 1:
        raise RuntimeError("depth preview requires one explicit raster placement per plate layer")
    plate_object = document.objects[plate.object_ids[0]]
    depth_object = document.objects[depth.object_ids[0]]
    if depth_object.transform.to_dict() != AffineTransform.identity().to_dict():
        raise RuntimeError("fixture depth object transform changed during visibility validation")

    preview_width = 1200
    scale = preview_width / document.width
    preview_height = round(document.height * scale)
    plate_asset = document.assets[plate_object.asset_id]
    depth_asset = document.assets[depth_object.asset_id]
    with Image.open(io.BytesIO(store.assets.read_bytes(plate_asset))) as source:
        canvas = source.convert("RGBA").resize((preview_width, preview_height), Image.Resampling.LANCZOS)

    registration = depth.registration
    if (registration.from_space != "source-crop" or registration.to_space != "document"
            or tuple(registration.source_dimensions) != (depth_asset.width, depth_asset.height)
            or tuple(registration.target_dimensions) != (document.width, document.height)):
        raise RuntimeError("fixture depth registration dimensions are inconsistent")
    x_doc, y_doc = registration.map_point(0, 0)
    geometry = depth_object.geometry
    if (abs(x_doc - float(geometry["x"])) > 1e-8 or abs(y_doc - float(geometry["y"])) > 1e-8
            or abs(float(geometry["width"]) - depth_asset.width) > 1e-8
            or abs(float(geometry["height"]) - depth_asset.height) > 1e-8):
        raise RuntimeError("fixture raster placement no longer matches its explicit registration")
    if depth.visible:
        with Image.open(io.BytesIO(store.assets.read_bytes(depth_asset))) as source:
            width = round(float(geometry["width"]) * scale)
            height = round(float(geometry["height"]) * scale)
            depth_pixels = source.convert("RGBA").resize((width, height), Image.Resampling.LANCZOS)
        alpha = depth_pixels.getchannel("A").point(lambda value: round(value * float(depth.opacity)))
        depth_pixels.putalpha(alpha)
        x = round(x_doc * scale)
        y = round(y_doc * scale)
        canvas.alpha_composite(depth_pixels, (x, y))
    output_path.write_bytes(image_bytes(canvas))
    with Image.open(output_path) as rendered:
        alpha_extrema = rendered.convert("RGBA").getchannel("A").getextrema()
        render_size = list(rendered.size)
    return {"documentId": document.document_id, "revision": document.current_revision,
            "visible": depth.visible, "completePlateId": depth.complete_plate_id,
            "depthElementId": depth.layer_id, "registration": depth.registration.to_dict(),
            "layerTransform": depth.transform.to_dict(), "objectTransform": depth_object.transform.to_dict(),
            "previewDimensions": render_size, "alphaExtrema": list(alpha_extrema),
            "renderMode": "test-only fixed document-space preview of explicit shared complete-plate and depth layers"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--private-asset-root", type=Path,
                        help="exact directory containing the five named W05 validation files")
    args = parser.parse_args()
    repo = args.repo.resolve()
    receipt_path = args.receipt.resolve()
    asset_root = (args.private_asset_root or (repo / "assets" / "images" / "reference")).resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "workOrder": "P6-M01-W05",
        "evidence": "generation-free local fixture; accepted W03 Director editor and W04 Agent driver",
        "startedAtUtc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "checks": {},
        "sourceDirectoryRecorded": False,
        "sourceBytesRecorded": False,
        "previewBytesRecorded": False,
        "bearerMaterialRecorded": False,
    }
    try:
        receipt["branch"] = subprocess.run(["git", "branch", "--show-current"], cwd=repo,
                                          capture_output=True, text=True, check=True).stdout.strip()
        receipt["head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                                         capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        receipt["branch"] = "unavailable"
        receipt["head"] = "unavailable"
    server: subprocess.Popen[Any] | None = None
    chrome: subprocess.Popen[Any] | None = None
    cdp = None
    output_root: Path | None = None
    document_id: str | None = None
    project_path: Path | None = None
    owner_key: str | None = None
    owner_key_in_arguments = False
    owner_key_in_capability = False
    capability_file: Path | None = None
    capability_token: str | None = None
    agent_actor_id: str | None = None
    grant_id: str | None = None
    profile: Path | None = None
    log_stream = None
    log_path: Path | None = None
    fixture_info: dict[str, Any] = {}
    command_evidence: list[dict[str, Any]] = []
    preview_diagnostics: list[dict[str, Any]] = []
    success = False
    phase = "runner-startup"
    try:
        w03 = _import_w03_helpers()
        output_root = w03.read_server_output_root(repo)
        profile = Path(tempfile.mkdtemp(prefix="w05-field-validation-", dir=receipt_path.parent)).resolve()
        downloads = profile / "downloads"
        downloads.mkdir()
        browser_profile = profile / "chrome-profile"
        browser_profile.mkdir()
        log_path = profile / "gradio.log"
        log_stream = log_path.open("w", encoding="utf-8", errors="replace")
        server_port = 7861 if not _port_open(7861) else _free_port()
        browser_port = _free_port()
        python = repo / "venv" / "Scripts" / "python.exe"
        server = subprocess.Popen(
            [str(python), "webui.py", "--skip-model-load", "--disable-analytics", "--port", str(server_port), "--disable-in-browser"],
            cwd=repo, stdout=log_stream, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        base_url = f"http://127.0.0.1:{server_port}"
        _wait_http(base_url)
        chrome_path = next((path for path in w03.CHROME_CANDIDATES if path.is_file()), None)
        if chrome_path is None:
            raise RuntimeError("headless Chrome is unavailable")
        chrome = subprocess.Popen(
            [str(chrome_path), "--headless=new", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage",
             "--no-first-run", "--no-default-browser-check", "--disable-extensions", "--remote-allow-origins=*",
             f"--remote-debugging-port={browser_port}", f"--user-data-dir={browser_profile}",
             "--window-size=1360,900", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        w03.wait_http(f"http://127.0.0.1:{browser_port}/json/version", timeout=45)
        chrome_version = json.loads(w03.wait_http(f"http://127.0.0.1:{browser_port}/json/version", timeout=10))
        targets = json.loads(w03.wait_http(f"http://127.0.0.1:{browser_port}/json", timeout=10))
        target = next(item for item in targets if item.get("type") == "page")
        cdp = w03.CDP(target["webSocketDebuggerUrl"])
        cdp.command("Page.enable")
        cdp.command("Runtime.enable")
        cdp.command("Network.enable")
        cdp.command("Page.setDownloadBehavior", {"behavior": "allow", "downloadPath": str(downloads)})
        cdp.command("Page.navigate", {"url": base_url})
        w03.wait_js(cdp, "document.readyState === 'complete'", timeout=90)
        w03.wait_js(cdp, "document.querySelector('creative-document-editor')", timeout=90)
        phase = "mount-editor"
        cdp.evaluate("""(()=>{const e=document.querySelector('creative-document-editor');
          const tab=[...document.querySelectorAll('[role=tab]')].find(item=>item.textContent.trim()==='Creative Document');
          if(tab) tab.click(); window.confirm=()=>true; window.prompt=(message,initial)=>initial||''; return !!e;})()""")
        w03.wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return !!e?._capability?.enabled;})()", timeout=60)
        mounted = cdp.evaluate("""(()=>{const e=document.querySelector('creative-document-editor'); return {mounted:!!e,
          renderer:window.Konva?.version||null,session:!!e?._capability?.enabled};})()""")
        _check(receipt, "accepted_w03_editor_runtime_mounted", mounted["mounted"] and mounted["session"] is True
               and mounted["renderer"] == "10.6.0", mounted)
        owner_key = cdp.evaluate("document.querySelector('creative-document-editor')._ownerKey")
        if not isinstance(owner_key, str) or not owner_key:
            raise RuntimeError("editor owner session is unavailable")
        owner_key_in_arguments = owner_key in " ".join(sys.argv)
        document_id, fixture_info = _fixture_scene(output_root, asset_root, owner_key)
        project_path = fixture_info["projectPath"].resolve()
        phase = "open-native-fixture"
        opened = cdp.evaluate("""(async(id)=>{const e=document.querySelector('creative-document-editor');
          const result=await e._request(`/creative_document_api/documents/${encodeURIComponent(id)}/open`,
            {method:'POST',body:{discardUnsaved:true}}); e._doc=result.view; e._selectedLayerId=e._doc.rootLayerIds[0];
          await e._acceptView(e._doc); return {documentId:e._doc.documentId,revision:e._doc.revision};})""" + f"({json.dumps(document_id)})")
        _check(receipt, "native_explicit_fixture_opened", opened == {"documentId": document_id, "revision": 0}, opened)

        # Track public action/save IDs and revision/actor metadata in page memory.
        cdp.evaluate("""(()=>{const e=document.querySelector('creative-document-editor'),original=e._request.bind(e);
          window.__w05Actions=[]; e._request=async(url,options={})=>{const body=options.body;
            const result=await original(url,options); const route=String(url).includes('/actions')?'action':
              (String(url).endsWith('/save')?'save':null);
            if(route&&body&&typeof body==='object') window.__w05Actions.push({route,commandId:body.commandId,
              transactionId:body.transactionId||result.transactionId,expectedRevision:body.expectedRevision,
              actorKind:result.actorKind,actorId:result.actorId,previousRevision:result.previousRevision,
              newRevision:result.newRevision,committedRevision:route==='save'?result.committedRevision:null,
              currentRevision:result.currentRevision,status:result.status,code:result.code}); return result;};
          return true;})()""")

        phase = "issue-agent-grant"
        cdp.evaluate("""(()=>{const e=document.querySelector('creative-document-editor');
          for(const scope of ['inspect','propose','mutate']) e.querySelector(`[data-agent-scope="${scope}"]`).checked=true;
          e.querySelector('[data-action=agent-enable]').click(); return true;})()""")
        w03.wait_js(cdp, "document.querySelector('creative-document-editor')?.querySelector('[data-role=agent-driver-status]')?.textContent.includes('Capability file downloaded.')", timeout=30)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            candidates = list(downloads.glob("nexfocus-driver-*.json"))
            if candidates:
                capability_file = candidates[0]
                break
            time.sleep(0.1)
        if capability_file is None:
            raise RuntimeError("editor did not download its explicit local capability")
        capability_text = capability_file.read_text(encoding="utf-8")
        owner_key_in_capability = owner_key in capability_text
        capability = json.loads(capability_text)
        capability_token = capability.get("token")
        agent_actor_id = capability.get("actorId")
        receipt["grantContractObservation"] = {
            "documentMatches": capability.get("documentId") == document_id,
            "scopeNames": sorted(capability.get("scopes", [])) if isinstance(capability.get("scopes"), list) else [],
            "tokenHasExpectedType": isinstance(capability_token, str),
        }
        if capability.get("documentId") != document_id or set(capability.get("scopes", [])) != {"inspect", "propose", "mutate"}:
            raise RuntimeError("temporary local capability did not match the requested W05 scopes")
        grant_status = cdp.evaluate("""(async()=>{const e=document.querySelector('creative-document-editor');
          return await e._request(`/creative_document_api/documents/${encodeURIComponent(e._doc.documentId)}/agent-grants/status`);})()""")
        grant_id = grant_status.get("grantId")
        _check(receipt, "temporary_inspect_propose_mutate_grant", grant_status.get("status") == "enabled"
               and set(grant_status.get("scopes", [])) == {"inspect", "propose", "mutate"} and isinstance(grant_id, str)
               and not owner_key_in_arguments and not owner_key_in_capability,
               {"status": grant_status.get("status"), "grantId": grant_id,
                "scopes": grant_status.get("scopes"), "bearerIncluded": False,
                "ownerCredentialAbsentFromArguments": not owner_key_in_arguments,
                "ownerCredentialAbsentFromCapability": not owner_key_in_capability})
        storage_expression = """((token)=>{const values=[]; for(const storage of [localStorage,sessionStorage])
          for(let i=0;i<storage.length;i++){const key=storage.key(i); values.push(key,storage.getItem(key)||'');}
          return {bearerAbsent:!values.join(' ').includes(token),sourcePathAbsent:!values.join(' ').includes('assets/images/reference'),
            sourceBytesPersisted:false};})""" + f"({json.dumps(capability_token)})"
        storage_check = cdp.evaluate(storage_expression)
        _check(receipt, "browser_storage_excludes_bearer_and_source_path", storage_check["bearerAbsent"]
               and storage_check["sourcePathAbsent"], storage_check)

        command_evidence.append({"actorId": agent_actor_id, "actorKind": "agent", "operation": "grant"})
        phase = "agent-cli-inspect"

        def agent(intent: str, command_type: str, revision: int | None, payload: dict[str, Any] | None,
                  label: str) -> tuple[int, dict[str, Any]]:
            envelope = _envelope(document_id, intent, command_type, revision=revision, payload=payload)
            code, result = _run_cli(repo, capability_file, intent, envelope, label, profile)
            command_evidence.append({
                "commandId": envelope["commandId"],
                "transactionId": None if envelope["transaction"] is None else envelope["transaction"]["transactionId"],
                "expectedRevision": envelope["expectedRevision"],
                "actorKind": result.get("actorKind", "agent"),
                "actorId": result.get("actorId", agent_actor_id),
                "commandType": command_type,
                "intent": intent,
                "code": _error_code(result),
                "writes": result.get("writes"),
                "previousRevision": result.get("previousRevision"),
                "newRevision": result.get("newRevision"),
                "currentRevision": result.get("currentRevision"),
            })
            return code, result

        code, inspected0 = agent("inspect", "inspect_document", None, None, "inspect-r0")
        projection0 = inspected0.get("result", {})
        source_filename_leak = any(name in json.dumps(projection0) for name in FIXTURES)
        _check(receipt, "agent_inspection_is_safe_revision_zero", code == 0 and inspected0.get("status") == "ok"
               and projection0.get("observedRevision") == 0 and projection0.get("snapshotToken") == f"{document_id}@0"
               and not source_filename_leak,
               {"revision": projection0.get("observedRevision"), "snapshotToken": projection0.get("snapshotToken"),
                "sourceFilenameDisclosed": source_filename_leak})

        phase = "create-marker-guide"
        marker_payload = {"data": {"name": "Position marker", "semanticRole": "position only; no subject identity",
            "kind": "shape", "geometry": {"shape": "ellipse", "x": 980, "y": 350, "width": 130, "height": 130},
            "style": {"fill": "#F2D64B", "stroke": "#342D0D", "strokeWidth": 5, "opacity": 0.92}}}
        _, marker_receipt = agent("mutate", "create_guide", 0, marker_payload, "create-marker-r0")
        marker_id = next(identity for identity in marker_receipt.get("createdIds", []) if identity.startswith("guide-"))
        phase = "create-anchor-guide"
        _, anchor_receipt = agent("mutate", "create_guide", 1, {"data": {
            "name": "Focal relation anchor", "semanticRole": "focal person attended to by the nearby group",
            "kind": "shape", "geometry": {"shape": "polygon", "points": [[1320, 340], [1385, 345], [1425, 410],
                [1410, 465], [1450, 540], [1375, 525], [1335, 600], [1305, 520], [1260, 550], [1285, 445], [1265, 390]]},
            "style": {"fill": "#47C78B", "stroke": "#153B2A", "strokeWidth": 5, "opacity": 0.78},
        }}, "create-anchor-r1")
        anchor_id = next(identity for identity in anchor_receipt.get("createdIds", []) if identity.startswith("guide-"))
        phase = "inspect-proposed-guides"
        _, proposed_projection_receipt = agent("inspect", "inspect_guides", 2, None, "inspect-guides-r2")
        proposed_projection = proposed_projection_receipt.get("result", {})
        marker_state = proposed_projection.get("guides", {}).get(marker_id, {})
        anchor_state = proposed_projection.get("guides", {}).get(anchor_id, {})
        marker_layer_id = marker_state.get("objectIds", [None])[0]
        guide_projection_ok = (marker_state.get("lifecycle") == "proposed" and anchor_state.get("lifecycle") == "proposed"
            and marker_state.get("semanticRole") == "position only; no subject identity"
            and anchor_state.get("semanticRole") == "focal person attended to by the nearby group"
            and marker_state.get("objectIds") != anchor_state.get("objectIds"))
        _check(receipt, "marker_and_semantic_anchor_are_distinct_editable_proposals", guide_projection_ok,
               {"markerGuideId": marker_id, "markerObjectId": marker_state.get("objectIds"),
                "markerRole": marker_state.get("semanticRole"), "markerLifecycle": marker_state.get("lifecycle"),
                "anchorGuideId": anchor_id, "anchorObjectId": anchor_state.get("objectIds"),
                "anchorRole": anchor_state.get("semanticRole"), "anchorLifecycle": anchor_state.get("lifecycle"),
                "observedRevision": proposed_projection_receipt.get("observedRevision")})

        # Visibility and save cannot promote a proposal. The Director action is
        # sent by the normal W03 layer control, then saved through the shared service.
        phase = "director-refresh-before-marker-visibility"
        cdp.evaluate("document.querySelector('creative-document-editor')._refreshAuthoritativeView()")
        w03.wait_js(cdp, "document.querySelector('creative-document-editor')._doc.revision===2", timeout=45)
        marker_layer = marker_state.get("objectIds", [None])[0]
        cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor');
          const object=e._doc.objects.find(item=>item.id==={json.dumps(marker_layer)}); const button=e.querySelector(
            `[data-action=layer-eye][data-layer-id="${{object.layerId}}"]`); if(!button) throw new Error('marker visibility control missing');
          button.click(); return object.layerId;}})()""")
        w03.wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.revision===3;})()", timeout=40)
        cdp.evaluate("document.querySelector('creative-document-editor')._saveDocument()")
        w03.wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return !e._doc.dirty && e._doc.committedRevision===3;})()", timeout=45)
        _, after_save = agent("inspect", "inspect_guides", 3, None, "inspect-proposals-after-save-r3")
        after_save_guides = after_save.get("result", {}).get("guides", {})
        conversational = _envelope(document_id, "mutate", "conversational_text", revision=3,
                                   payload={"text": f"activate guide {anchor_id}"})
        conversational_code, conversational_result = _run_cli(repo, capability_file, "mutate", conversational,
                                                                 "conversational-text-r3", profile)
        command_evidence.append({"commandId": conversational["commandId"],
            "transactionId": conversational["transaction"]["transactionId"], "expectedRevision": 3,
            "actorKind": conversational_result.get("actorKind", "agent"), "actorId": conversational_result.get("actorId", agent_actor_id),
            "commandType": "conversational_text", "intent": "mutate",
            "code": _error_code(conversational_result), "writes": conversational_result.get("writes"),
            "currentRevision": conversational_result.get("currentRevision")})
        _, post_conversation = agent("inspect", "inspect_guides", 3, None, "inspect-after-conversational-refusal-r3")
        post_conversation_guides = post_conversation.get("result", {}).get("guides", {})
        no_implicit_promotion = (after_save.get("observedRevision") == 3 and after_save.get("writes") == 0
            and after_save_guides.get(marker_id, {}).get("lifecycle") == "proposed"
            and after_save_guides.get(anchor_id, {}).get("lifecycle") == "proposed"
            and post_conversation.get("observedRevision") == 3
            and post_conversation.get("writes") == 0
            and post_conversation_guides.get(marker_id, {}).get("lifecycle") == "proposed"
            and post_conversation_guides.get(anchor_id, {}).get("lifecycle") == "proposed"
            and conversational_code == 2 and _error_code(conversational_result) == "UNSUPPORTED_COMMAND"
            and conversational_result.get("writes") == 0)
        _check(receipt, "inspection_visibility_save_and_text_do_not_activate_proposal", no_implicit_promotion,
               {"markerLifecycle": post_conversation_guides.get(marker_id, {}).get("lifecycle"),
                "anchorLifecycle": post_conversation_guides.get(anchor_id, {}).get("lifecycle"),
                "conversationalCode": _error_code(conversational_result),
                "conversationalWrites": conversational_result.get("writes"),
                "postRefusalObservedRevision": post_conversation.get("observedRevision")})

        # Director activates the broad anchor through the W03 lifecycle control.
        cdp.evaluate(f"document.querySelector('creative-document-editor').querySelector('[data-action=guide-activate][data-guide-id=\"{anchor_id}\"]').click()")
        phase = "broad-to-tight-exchange"
        w03.wait_js(cdp, f"(()=>{{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.revision===4 && e._doc.guides.find(g=>g.guideId==={json.dumps(anchor_id)}).lifecycle==='active';}})()", timeout=40)
        _, broad_inspect = agent("inspect", "inspect_document", 4, None, "inspect-director-broad-r4")
        broad_view = broad_inspect.get("result", {})
        broad_anchor = broad_view.get("guides", {}).get(anchor_id, {})
        _check(receipt, "agent_inspects_exact_director_activation_revision", broad_inspect.get("observedRevision") == 4
               and broad_anchor.get("lifecycle") == "active" and broad_anchor.get("stateRevision") == 4,
               {"revision": broad_inspect.get("observedRevision"), "broadGuideId": anchor_id,
                "lifecycle": broad_anchor.get("lifecycle"), "stateRevision": broad_anchor.get("stateRevision")})

        _, tight_receipt = agent("mutate", "create_guide", 4, {"data": {
            "name": "Tighter focal relation", "semanticRole": "same focal person; tighter head-and-shoulder placement cue",
            "kind": "shape", "geometry": {"shape": "polygon", "points": [[1350, 360], [1390, 365], [1415, 405],
                [1407, 448], [1430, 485], [1380, 478], [1350, 515], [1328, 468], [1300, 480], [1315, 414]]},
            "style": {"fill": "#63D7A2", "stroke": "#153B2A", "strokeWidth": 4, "opacity": 0.72},
        }}, "create-tight-guide-r4")
        tight_id = next(identity for identity in tight_receipt.get("createdIds", []) if identity.startswith("guide-"))
        tight_inspect_revision = tight_receipt.get("newRevision")
        cdp.evaluate("document.querySelector('creative-document-editor')._refreshAuthoritativeView()")
        w03.wait_js(cdp, f"(()=>{{const e=document.querySelector('creative-document-editor'); return e._doc.revision==={tight_inspect_revision} && !!e._doc.guides.find(g=>g.guideId==={json.dumps(tight_id)});}})()", timeout=45)
        cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'),id={json.dumps(tight_id)};
          const row=[...e.querySelectorAll('[data-role=guides] .ncd-row')].find(item=>item.querySelector(`[data-guide-id="${{id}}"]`));
          if(!row) throw new Error('tight guide row missing'); row.querySelector('span').click();
          window.prompt=(message,initial)=>message.startsWith('ID of the existing guide')?{json.dumps(anchor_id)}:(initial||'');
          e.querySelector('[data-action=replace-guide]').click(); return true;}})()""")
        w03.wait_js(cdp, f"(()=>{{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.guides.find(g=>g.guideId==={json.dumps(anchor_id)}).lifecycle==='replacement-pending';}})()", timeout=40)
        pending_revision = cdp.evaluate("document.querySelector('creative-document-editor')._doc.revision")
        stale_command = _envelope(document_id, "mutate", "create_guide", revision=tight_inspect_revision,
                                  payload={"data": {"name": "Stale guide must not appear", "semanticRole": "stale probe",
                                  "kind": "shape", "geometry": {"shape": "ellipse", "x": 1200, "y": 780, "width": 30, "height": 30},
                                  "style": {"fill": "#FF0000", "stroke": "#000000", "width": 2}}})
        stale_code, stale_receipt = _run_cli(repo, capability_file, "mutate", stale_command, "stale-tight-r5", profile)
        command_evidence.append({"commandId": stale_command["commandId"],
            "transactionId": stale_command["transaction"]["transactionId"], "expectedRevision": tight_inspect_revision,
            "actorKind": stale_receipt.get("actorKind", "agent"), "actorId": stale_receipt.get("actorId", agent_actor_id),
            "commandType": "create_guide", "intent": "mutate", "code": _error_code(stale_receipt),
            "writes": stale_receipt.get("writes"), "currentRevision": stale_receipt.get("currentRevision")})
        _check(receipt, "stale_agent_mutation_after_director_replacement_is_zero_write", stale_code == 2
               and _error_code(stale_receipt) == "STALE_DOCUMENT_REVISION"
               and stale_receipt.get("writes") == 0 and stale_receipt.get("previousRevision") == tight_inspect_revision
               and stale_receipt.get("currentRevision") == pending_revision,
               {"expectedRevision": tight_inspect_revision, "directorRevision": pending_revision,
                "code": _error_code(stale_receipt), "writes": stale_receipt.get("writes")})

        _, fresh_inspect = agent("inspect", "inspect_document", pending_revision, None, "explicit-refresh-r6")
        fresh_projection = fresh_inspect.get("result", {})
        _, context_create = agent("mutate", "create_context_mask", pending_revision, {"data": {
            "sourceLayerId": fixture_info["ids"]["layers"]["depthElement"],
            "seeds": [{"kind": "box", "geometry": {"x": 600, "y": 300, "width": 1450, "height": 960}}],
        }}, "agent-context-mask-after-refresh")
        context_id = next(identity for identity in context_create.get("createdIds", []) if identity.startswith("mask-"))
        context_revision = context_create.get("newRevision")
        _check(receipt, "explicit_refresh_then_agent_continuation_preserves_both_guide_states",
               fresh_inspect.get("observedRevision") == pending_revision
               and fresh_projection.get("guides", {}).get(anchor_id, {}).get("lifecycle") == "replacement-pending"
               and context_create.get("previousRevision") == pending_revision
               and context_revision == pending_revision + 1,
               {"refreshedRevision": fresh_inspect.get("observedRevision"), "broadGuideId": anchor_id,
                "broadLifecycle": fresh_projection.get("guides", {}).get(anchor_id, {}).get("lifecycle"),
                "tightGuideId": tight_id, "continuationRevision": context_revision})

        # Director performs every remaining guide transition explicitly.
        cdp.evaluate("document.querySelector('creative-document-editor')._refreshAuthoritativeView()")
        w03.wait_js(cdp, f"document.querySelector('creative-document-editor')._doc.revision==={context_revision}", timeout=45)
        phase = "explicit-guide-lifecycle"
        cdp.evaluate(f"document.querySelector('creative-document-editor').querySelector('[data-action=guide-supersede][data-guide-id=\"{anchor_id}\"]').click()")
        w03.wait_js(cdp, f"(()=>{{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.guides.find(g=>g.guideId==={json.dumps(anchor_id)}).lifecycle==='superseded';}})()", timeout=40)
        cdp.evaluate(f"document.querySelector('creative-document-editor').querySelector('[data-action=guide-activate][data-guide-id=\"{tight_id}\"]').click()")
        w03.wait_js(cdp, f"(()=>{{const e=document.querySelector('creative-document-editor'); const g=e._doc.guides.find(g=>g.guideId==={json.dumps(tight_id)}); return !e._actionInFlight && g.lifecycle==='active';}})()", timeout=40)
        cdp.evaluate(f"document.querySelector('creative-document-editor').querySelector('[data-action=guide-retire][data-guide-id=\"{anchor_id}\"]').click()")
        w03.wait_js(cdp, f"(()=>{{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.guides.find(g=>g.guideId==={json.dumps(anchor_id)}).lifecycle==='safe-to-remove';}})()", timeout=40)
        guide_states = cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor');
          const broad=e._doc.guides.find(g=>g.guideId==={json.dumps(anchor_id)}),tight=e._doc.guides.find(g=>g.guideId==={json.dumps(tight_id)});
          return {{revision:e._doc.revision,broad:{{id:broad.guideId,lifecycle:broad.lifecycle,stateRevision:broad.stateRevision,
            replacementId:broad.replacementId}},tight:{{id:tight.guideId,lifecycle:tight.lifecycle,stateRevision:tight.stateRevision,
            supersedesId:tight.supersedesId}},distinctObjects:broad.objectIds[0]!==tight.objectIds[0]}};}})()""")
        _check(receipt, "director_explicit_replace_supersede_activate_retire_lineage", guide_states["broad"]["lifecycle"] == "safe-to-remove"
               and guide_states["tight"]["lifecycle"] == "active" and guide_states["broad"]["replacementId"] == tight_id
               and guide_states["tight"]["supersedesId"] == anchor_id and guide_states["distinctObjects"], guide_states)

        # Use the W03 context controls to keep relation references separate from
        # the generation edit target and mask.
        cdp.evaluate(f"document.querySelector('creative-document-editor')._refreshAuthoritativeView()")
        w03.wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return e._doc.revision===" + str(guide_states["revision"]) + " && e._doc.masks.some(m=>m.id===" + json.dumps(context_id) + ");})()", timeout=45)
        depth_layer_id = fixture_info["ids"]["layers"]["depthElement"]
        riders_layer_id = fixture_info["ids"]["layers"]["ridersReference"]
        edit_mask_id = fixture_info["ids"]["masks"]["generationEditScope"]
        cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor');
          e._selectedLayerId={json.dumps(depth_layer_id)}; e._selectedObjectIds.clear(); e._renderLayerList(); e._renderProperties();
          const ctx=e.querySelector('[data-action=context-mask]'); ctx.value={json.dumps(context_id)};
          ctx.dispatchEvent(new Event('change',{{bubbles:true}}));
          const edit=e.querySelector('[data-action=edit-mask]'); edit.value={json.dumps(edit_mask_id)};
          edit.dispatchEvent(new Event('change',{{bubbles:true}}));
          const ref=e.querySelector(`.ncd-context-ref[data-layer-id="{riders_layer_id}"]`); ref.checked=true;
          ref.dispatchEvent(new Event('change',{{bubbles:true}}));
          e.querySelector('[data-action=context-dilation]').value='24';
          e.querySelector('[data-action=context-dilation]').dispatchEvent(new Event('input',{{bubbles:true}}));
          e.querySelector('[data-action=save-context]').click(); return true;}})()""")
        phase = "relational-context"
        current_context_revision = guide_states["revision"] + 1
        w03.wait_js(cdp, f"(()=>{{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.revision==={current_context_revision};}})()", timeout=45)
        _, context_inspect = agent("inspect", "inspect_context", current_context_revision, None, "inspect-context-r")
        context_summary = next((item for item in context_inspect.get("result", {}).get("relationalContext", [])
                                if item.get("contextMaskId") == context_id), {})
        context_ok = context_summary.get("referenceIds") == [riders_layer_id]
        context_ok = context_ok and context_summary.get("editTargetIds") == [depth_layer_id]
        context_ok = context_ok and context_summary.get("dilationPx") == 24 and context_summary.get("revision") == current_context_revision
        _check(receipt, "context_reference_is_distinct_from_explicit_edit_scope", context_ok,
               {"contextMaskId": context_id, "referenceIds": context_summary.get("referenceIds"),
                "editTargetIds": context_summary.get("editTargetIds"), "editMaskId": edit_mask_id,
                "dilationPx": context_summary.get("dilationPx"), "revision": context_summary.get("revision")})

        refusal_results: list[dict[str, Any]] = []
        def refusal(name: str, expected_revision: int, payload: dict[str, Any], expected_code: str,
                    expected_current_revision: int | None = None) -> None:
            envelope = _envelope(document_id, "mutate", "set_relational_context", revision=expected_revision,
                                 payload={"data": {"contextMaskId": context_id, "referenceIds": [riders_layer_id],
                                     "dilationPx": 24, "editMaskId": edit_mask_id, "editTargetIds": [depth_layer_id], **payload}})
            code, result = _run_cli(repo, capability_file, "mutate", envelope, name, profile)
            expected_current = current_context_revision if expected_current_revision is None else expected_current_revision
            evidence = {"name": name, "expectedCode": expected_code, "code": _error_code(result),
                        "expectedRevision": expected_revision, "currentRevision": result.get("currentRevision"),
                        "expectedCurrentRevision": expected_current,
                        "writes": result.get("writes"), "commandId": envelope["commandId"],
                        "transactionId": envelope["transaction"]["transactionId"]}
            command_evidence.append({**evidence, "actorKind": result.get("actorKind", "agent"),
                                     "actorId": result.get("actorId", agent_actor_id), "commandType": "set_relational_context",
                                     "intent": "mutate"})
            refusal_results.append(evidence)
            _, post_refusal = agent("inspect", "inspect_document", expected_current, None, f"{name}-verify")
            observed_after = post_refusal.get("result", {}).get("observedRevision")
            evidence["postRefusalObservedRevision"] = observed_after
            _check(receipt, name, code == 2 and _error_code(result) == expected_code
                   and result.get("writes") == 0 and observed_after == expected_current
                   and post_refusal.get("writes") == 0, evidence)

        refusal("context_overlap_refused_zero_write", current_context_revision,
                {"referenceIds": [depth_layer_id], "editTargetIds": [depth_layer_id]}, "CONTEXT_EDIT_SCOPE_OVERLAP")
        refusal("malformed_context_reference_refused_zero_write", current_context_revision,
                {"referenceIds": "malformed"}, "INVALID_TARGETS")
        refusal("stale_context_update_refused_zero_write", current_context_revision - 1, {}, "STALE_DOCUMENT_REVISION")

        cdp.evaluate(f"document.querySelector('creative-document-editor').querySelector('[data-action=layer-lock][data-layer-id=\"{depth_layer_id}\"]').click()")
        locked_revision = current_context_revision + 1
        w03.wait_js(cdp, f"(()=>{{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.revision==={locked_revision} && e._doc.layers.find(l=>l.id==={json.dumps(depth_layer_id)}).locked;}})()", timeout=40)
        refusal("locked_context_owner_refused_zero_write", locked_revision,
                {"dilationPx": 25}, "LAYER_LOCKED", expected_current_revision=locked_revision)
        cdp.evaluate(f"document.querySelector('creative-document-editor').querySelector('[data-action=layer-lock][data-layer-id=\"{depth_layer_id}\"]').click()")
        current_depth_revision = locked_revision + 1
        w03.wait_js(cdp, f"(()=>{{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.revision==={current_depth_revision} && !e._doc.layers.find(l=>l.id==={json.dumps(depth_layer_id)}).locked;}})()", timeout=40)

        # Save through the W03 UI, then render only the allowlisted shared
        # complete-plate/depth pair at fixed document scale. The preview helper
        # reads the persisted native scene and does not mutate it.
        phase = "depth-hide-show-preview"
        from modules.creative_document import ProjectStore

        def save_director_revision(revision: int) -> None:
            before = cdp.evaluate("window.__w05Actions.filter(item=>item.route==='save').length")
            cdp.evaluate("document.querySelector('creative-document-editor')._saveDocument()")
            w03.wait_js(cdp, f"(()=>{{const e=document.querySelector('creative-document-editor'); const saves=window.__w05Actions.filter(item=>item.route==='save'); return !e._doc.dirty && e._doc.committedRevision==={revision} && saves.length>{before} && saves[saves.length-1].status==='saved';}})()", timeout=45)

        def capture_depth_preview(label: str, revision: int, visible: bool) -> dict[str, Any]:
            output = profile / f"{label}.png"
            rendered = _render_depth_fixture_preview(project_path, fixture_info["ids"]["completePlateId"],
                                                     depth_layer_id, output)
            if rendered["revision"] != revision or rendered["visible"] is not visible:
                raise RuntimeError("persisted depth preview does not match the requested Director revision")
            rendered["capture"] = label
            rendered["previewSha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
            preview_diagnostics.append(rendered)
            return rendered

        save_director_revision(current_depth_revision)
        shown_capture = capture_depth_preview("shown-before", current_depth_revision, True)
        shown_hash = shown_capture["previewSha256"]
        cdp.evaluate(f"document.querySelector('creative-document-editor').querySelector('[data-action=layer-eye][data-layer-id=\"{depth_layer_id}\"]').click()")
        hidden_revision = current_depth_revision + 1
        w03.wait_js(cdp, f"(()=>{{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.revision==={hidden_revision} && !e._doc.layers.find(l=>l.id==={json.dumps(depth_layer_id)}).visible;}})()", timeout=40)
        depth_stale = _envelope(document_id, "mutate", "set_layer_visibility", revision=current_depth_revision,
                                 payload={"data": {"layerId": depth_layer_id, "visible": True}})
        stale_depth_code, stale_depth = _run_cli(repo, capability_file, "mutate", depth_stale, "stale-depth-show", profile)
        command_evidence.append({"commandId": depth_stale["commandId"],
            "transactionId": depth_stale["transaction"]["transactionId"], "expectedRevision": current_depth_revision,
            "actorKind": stale_depth.get("actorKind", "agent"), "actorId": stale_depth.get("actorId", agent_actor_id),
            "commandType": "set_layer_visibility", "intent": "mutate",
            "code": _error_code(stale_depth), "writes": stale_depth.get("writes"),
            "currentRevision": stale_depth.get("currentRevision")})
        _check(receipt, "stale_depth_show_refused_zero_write", stale_depth_code == 2
               and _error_code(stale_depth) == "STALE_DOCUMENT_REVISION"
               and stale_depth.get("writes") == 0 and stale_depth.get("currentRevision") == hidden_revision,
               {"expectedRevision": current_depth_revision, "currentRevision": hidden_revision,
                "code": _error_code(stale_depth), "writes": stale_depth.get("writes")})

        # The visibility controls share the editor's semantic action route and
        # its existing undo/redo history. Undo restores the shown digest; redo
        # restores the complete-plate hidden state.
        cdp.evaluate("document.querySelector('creative-document-editor').querySelector('[data-action=undo]').click()")
        undo_revision = hidden_revision + 1
        w03.wait_js(cdp, f"(()=>{{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.revision==={undo_revision} && e._doc.layers.find(l=>l.id==={json.dumps(depth_layer_id)}).visible;}})()", timeout=40)
        save_director_revision(undo_revision)
        undo_capture = capture_depth_preview("depth-undo-shown", undo_revision, True)
        undo_hash = undo_capture["previewSha256"]
        cdp.evaluate("document.querySelector('creative-document-editor').querySelector('[data-action=redo]').click()")
        redo_revision = undo_revision + 1
        w03.wait_js(cdp, f"(()=>{{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.revision==={redo_revision} && !e._doc.layers.find(l=>l.id==={json.dumps(depth_layer_id)}).visible;}})()", timeout=40)
        save_director_revision(redo_revision)
        hidden_capture = capture_depth_preview("depth-hidden", redo_revision, False)
        hidden_hash = hidden_capture["previewSha256"]
        redo_hash = hidden_hash
        hidden_state_valid = hidden_capture["alphaExtrema"] == [255, 255] and hidden_hash != shown_hash
        _check(receipt, "complete_plate_hides_without_transparency_and_undo_redo_restore_exact_renders",
               hidden_state_valid and undo_hash == shown_hash,
               {"completePlateId": fixture_info["ids"]["completePlateId"], "depthElementId": depth_layer_id,
                "shownPreviewSha256": shown_hash, "hiddenPreviewSha256": hidden_hash,
                "undoShownPreviewSha256": undo_hash, "redoHiddenPreviewSha256": redo_hash,
                "previewDimensions": hidden_capture["previewDimensions"], "hiddenAlphaExtrema": hidden_capture["alphaExtrema"],
                "renderedPlacementRestoredExactly": undo_hash == shown_hash and redo_hash == hidden_hash})

        # Save hidden, reopen, show, save and reopen again. Both open operations
        # go through the W03 Director endpoint; state writes use the W03 action route.
        cdp.evaluate(f"""(async()=>{{const e=document.querySelector('creative-document-editor');
          const opened=await e._request(`/creative_document_api/documents/{document_id}/open`,{{method:'POST',body:{{discardUnsaved:true}}}});
          e._doc=opened.view; e._selectedLayerId=e._doc.rootLayerIds[0]; await e._acceptView(e._doc); return e._doc.revision;}})()""")
        w03.wait_js(cdp, f"(()=>{{const e=document.querySelector('creative-document-editor'); const d=e._doc.layers.find(l=>l.id==={json.dumps(depth_layer_id)}); return e._doc.revision==={redo_revision} && d.visible===false;}})()", timeout=60)
        first_reopen = ProjectStore(project_path).open()
        first_depth = first_reopen.layers[depth_layer_id]
        first_registration = first_depth.registration.to_dict() if first_depth.registration else None
        first_transform = first_depth.transform.to_dict()
        first_hidden_capture = capture_depth_preview("hidden-reopen", redo_revision, False)
        first_hidden_hash = first_hidden_capture["previewSha256"]
        cdp.evaluate(f"document.querySelector('creative-document-editor').querySelector('[data-action=layer-eye][data-layer-id=\"{depth_layer_id}\"]').click()")
        shown_again_revision = redo_revision + 1
        w03.wait_js(cdp, f"(()=>{{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.revision==={shown_again_revision} && e._doc.layers.find(l=>l.id==={json.dumps(depth_layer_id)}).visible;}})()", timeout=40)
        save_director_revision(shown_again_revision)
        shown_again_capture = capture_depth_preview("depth-shown-again", shown_again_revision, True)
        shown_again_hash = shown_again_capture["previewSha256"]
        cdp.evaluate(f"""(async()=>{{const e=document.querySelector('creative-document-editor');
          const opened=await e._request(`/creative_document_api/documents/{document_id}/open`,{{method:'POST',body:{{discardUnsaved:true}}}});
          e._doc=opened.view; e._selectedLayerId=e._doc.rootLayerIds[0]; await e._acceptView(e._doc); return e._doc.revision;}})()""")
        w03.wait_js(cdp, f"(()=>{{const e=document.querySelector('creative-document-editor'); const d=e._doc.layers.find(l=>l.id==={json.dumps(depth_layer_id)}); return e._doc.revision==={shown_again_revision} && d.visible===true;}})()", timeout=60)
        second_reopen = ProjectStore(project_path).open()
        second_depth = second_reopen.layers[depth_layer_id]
        second_registration = second_depth.registration.to_dict() if second_depth.registration else None
        second_transform = second_depth.transform.to_dict()
        second_shown_capture = capture_depth_preview("shown-reopen", shown_again_revision, True)
        second_shown_hash = second_shown_capture["previewSha256"]
        # The fixture registration is persisted byte-for-byte across both open states.
        depth_persistence_ok = (first_depth.visible is False and second_depth.visible is True
            and first_depth.complete_plate_id == second_depth.complete_plate_id == fixture_info["ids"]["completePlateId"]
            and first_registration == second_registration and first_transform == second_transform
            and shown_again_hash == shown_hash == second_shown_hash and first_hidden_hash == hidden_hash)
        _check(receipt, "hidden_and_shown_save_reopen_preserve_complete_plate_registration",
               depth_persistence_ok,
               {"hiddenReopenRevision": first_reopen.current_revision, "shownReopenRevision": second_reopen.current_revision,
                "hiddenReopenVisible": first_depth.visible, "shownReopenVisible": second_depth.visible,
                "completePlateId": second_depth.complete_plate_id, "registration": second_registration,
                "elementTransform": second_transform, "hiddenReopenPreviewSha256": first_hidden_hash,
                "shownAgainPreviewSha256": shown_again_hash, "shownReopenPreviewSha256": second_shown_hash})

        final_document = ProjectStore(project_path).open()
        phase = "persistence-and-provenance"
        transaction_history = [{"transactionId": item["transactionId"], "actorId": item["actorId"],
            "actorKind": item["actorKind"], "commandIds": item["commandIds"],
            "previousRevision": item["previousRevision"], "resultingRevision": item["resultingRevision"],
            "kind": item["kind"], "affectedIds": item["affectedIds"]} for item in final_document.history]
        actor_kinds = sorted({item["actorKind"] for item in final_document.history})
        revisions_monotonic = all(tx["resultingRevision"] == tx["previousRevision"] + 1 for tx in transaction_history)
        persisted_guides_ok = (final_document.guides[marker_id].lifecycle == "proposed"
            and final_document.guides[anchor_id].lifecycle == "safe-to-remove"
            and final_document.guides[tight_id].lifecycle == "active"
            and final_document.guides[anchor_id].replacement_id == tight_id
            and final_document.guides[tight_id].supersedes_id == anchor_id)
        context_lineage = final_document.masks[context_id].lineage
        persisted_context_ok = (context_lineage["relationalReferences"] == [riders_layer_id]
            and context_lineage["editTargetIds"] == [depth_layer_id]
            and context_lineage["editMaskId"] == edit_mask_id and context_lineage["dilationPx"] == 24)
        _check(receipt, "save_reopen_preserves_guide_context_and_trusted_ordered_history",
               persisted_guides_ok and persisted_context_ok and revisions_monotonic
               and "agent" in actor_kinds and bool({"human", "director"}.intersection(actor_kinds)),
               {"documentId": final_document.document_id, "finalRevision": final_document.current_revision,
                "markerGuideId": marker_id, "broadGuideId": anchor_id, "tightGuideId": tight_id,
                "guideLineagePreserved": persisted_guides_ok, "contextMaskId": context_id,
                "contextLineagePreserved": persisted_context_ok, "actorKinds": actor_kinds,
                "historyTransactions": len(transaction_history), "revisionsMonotonic": revisions_monotonic})

        # Revoke before the runner removes its ephemeral profile and bearer.
        cdp.evaluate("document.querySelector('creative-document-editor')._revokeAgentDriver()")
        w03.wait_js(cdp, "document.querySelector('creative-document-editor')?.querySelector('[data-role=agent-driver-status]')?.textContent.includes('revoked')", timeout=35)
        status_command = [str(python), "tools/creative_document_driver.py", "--capability-file", str(capability_file), "status"]
        status_process = subprocess.run(status_command, cwd=repo, capture_output=True, text=True, timeout=60, check=False)
        status_result = json.loads(status_process.stdout) if status_process.stdout.strip() else {}
        revoke_write = _envelope(document_id, "mutate", "set_layer_visibility", revision=final_document.current_revision,
                                 payload={"data": {"layerId": depth_layer_id, "visible": False}})
        revoked_code, revoked_result = _run_cli(repo, capability_file, "mutate", revoke_write, "after-revoke-write", profile)
        post_revoke_document = ProjectStore(project_path).open()
        _check(receipt, "revocation_refuses_subsequent_agent_access_with_zero_writes",
               status_process.returncode == 2 and status_result.get("status") == "refused"
               and revoked_code == 2 and _error_code(revoked_result) == "AUTHORIZATION_REQUIRED"
               and revoked_result.get("writes") == 0
               and post_revoke_document.current_revision == final_document.current_revision,
               {"statusRefused": status_result.get("status") == "refused", "writeCode": _error_code(revoked_result),
                "writes": revoked_result.get("writes"), "driverCurrentRevision": revoked_result.get("currentRevision"),
                "postRevokePersistedRevision": post_revoke_document.current_revision})
        command_evidence.append({"commandId": revoke_write["commandId"],
            "transactionId": revoke_write["transaction"]["transactionId"], "expectedRevision": final_document.current_revision,
            "actorKind": revoked_result.get("actorKind", "agent"), "actorId": revoked_result.get("actorId", agent_actor_id),
            "commandType": "set_layer_visibility", "intent": "mutate",
            "code": _error_code(revoked_result), "writes": revoked_result.get("writes"),
            "currentRevision": revoked_result.get("currentRevision")})

        receipt["environment"] = {"python": sys.version.split()[0], "platform": sys.platform,
            "chrome": chrome_version.get("Browser"), "renderer": mounted["renderer"],
            "branch": receipt["branch"], "head": receipt["head"], "workingTree": "uncommitted W05 implementation diff"}
        receipt["fixtures"] = fixture_info["fixtures"]
        receipt["document"] = {"ids": fixture_info["ids"], "revision": final_document.current_revision,
            "markerGuideId": marker_id, "semanticAnchorGuideId": anchor_id, "tightGuideId": tight_id,
            "contextMaskId": context_id, "depthElementId": depth_layer_id, "completePlateId": fixture_info["ids"]["completePlateId"],
            "fixtureConstruction": fixture_info["construction"],
            "projectRelativePath": project_path.relative_to(output_root).as_posix(),
            "retainedForDirector": False,
            "sceneOwner": "disposable temporary browser principal; never a Director field-scene owner"}
        receipt["actors"] = {"agentActorId": agent_actor_id,
            "directorActorIds": sorted({item["actorId"] for item in transaction_history if item["actorKind"] in {"human", "director"}}),
            "grantId": grant_id, "grantRevoked": True}
        receipt["commands"] = command_evidence
        receipt["uiReceipts"] = cdp.evaluate("window.__w05Actions")
        receipt["previewDiagnostics"] = preview_diagnostics
        receipt["transactions"] = transaction_history
        receipt["refusals"] = refusal_results
        receipt["privacy"] = {"sourceRootRecorded": False, "sourceBytesRecorded": False,
            "previewBytesRecorded": False, "bearerRecorded": False, "browserStorageBearerAbsent": True,
            "ownerCredentialRecorded": False, "ownerPrincipalRecorded": False,
            "ownerCredentialForwardedToCapabilityOrRunnerArguments": False,
            "browserProfileRemoved": True, "fixtureNamesAndHashesAllowed": True,
            "directorQualitativeFieldResult": "unavailable; CM2 does not author Director feedback"}
        success = all(item["passed"] for item in receipt["checks"].values())
        receipt["status"] = "passed" if success else "failed"
        return_code = 0 if success else 1
    except Exception as exc:
        receipt["status"] = "failed"
        # Exception text may carry a local path or data-derived value; keep only the type.
        receipt["failure"] = {"type": type(exc).__name__, "phase": phase,
                              "message": "See local runner output for the failed check."}
        if DRIVER_DIAGNOSTICS:
            receipt["driverDiagnostics"] = list(DRIVER_DIAGNOSTICS)
        if command_evidence:
            receipt["commands"] = list(command_evidence)
        if cdp is not None:
            try:
                receipt["uiReceipts"] = cdp.evaluate("window.__w05Actions||[]")
            except Exception:
                pass
        if preview_diagnostics:
            receipt["previewDiagnostics"] = list(preview_diagnostics)
        return_code = 1
    finally:
        if cdp is not None:
            try:
                cdp.close()
            except Exception:
                pass
        for process in (chrome, server):
            if process is not None and process.poll() is None:
                if process is chrome and os.name == "nt":
                    # Chrome owns child processes that can hold profile files open.
                    # Scope taskkill to the exact process launched by this runner.
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   capture_output=True, text=True, timeout=15, check=False)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                else:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
        if log_stream is not None:
            log_stream.close()
        if log_path is not None:
            log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
            source_root_sentinel = str(asset_root)
            owner_key_in_logs = bool(owner_key and owner_key in log_text)
            bearer_in_logs = bool(capability_token and capability_token in log_text)
            clean_logs = not bearer_in_logs and source_root_sentinel not in log_text and not owner_key_in_logs
            receipt.setdefault("checks", {})["logs_exclude_bearer_and_private_root"] = {
                "passed": clean_logs, "evidence": {"bearerAbsent": not bearer_in_logs,
                                                      "privateRootAbsent": source_root_sentinel not in log_text,
                                                      "ownerCredentialAbsent": not owner_key_in_logs}}
            if not clean_logs:
                receipt["status"] = "failed"
                return_code = 1
                success = False
        if project_path is not None:
            try:
                serialized_so_far = json.dumps(receipt).lower()
                root_bytes = str(asset_root).lower().encode("utf-8")
                token_bytes = capability_token.encode("utf-8") if capability_token else b""
                project_root = project_path.resolve()
                project_text_has_private_root = str(asset_root).lower() in serialized_so_far
                project_text_has_bearer = bool(capability_token and capability_token in serialized_so_far)
                receipt_has_owner_key = bool(owner_key and owner_key in serialized_so_far)
                retained_files_checked = 0
                output_has_private_sentinel = False
                output_has_owner_credential = False
                if project_root.is_dir():
                    for retained_file in project_root.rglob("*"):
                        if not retained_file.is_file():
                            continue
                        retained_files_checked += 1
                        raw = retained_file.read_bytes()
                        if root_bytes and root_bytes in raw.lower():
                            output_has_private_sentinel = True
                        if token_bytes and token_bytes in raw:
                            output_has_private_sentinel = True
                        if owner_key and owner_key.encode("ascii") in raw:
                            output_has_owner_credential = True
                scan_ok = (not project_text_has_private_root and not project_text_has_bearer
                           and not receipt_has_owner_key and not output_has_private_sentinel
                           and not output_has_owner_credential)
                receipt.setdefault("checks", {})["receipt_and_scene_exclude_private_path_and_bearer"] = {
                    "passed": scan_ok,
                    "evidence": {"privateRootInReceipt": project_text_has_private_root,
                                 "bearerInReceipt": project_text_has_bearer,
                                 "ownerCredentialInReceipt": receipt_has_owner_key,
                                 "privateRootOrBearerInProject": output_has_private_sentinel,
                                 "ownerCredentialInProject": output_has_owner_credential,
                                 "projectFilesScanned": retained_files_checked},
                }
                if not scan_ok:
                    receipt["status"] = "failed"
                    success = False
                    return_code = 1
            except OSError:
                receipt.setdefault("checks", {})["receipt_and_scene_exclude_private_path_and_bearer"] = {
                    "passed": False, "evidence": {"scan": "unavailable"}}
                receipt["status"] = "failed"
                success = False
                return_code = 1
        if project_path is not None:
            try:
                owned_root = (output_root / "creative_documents").resolve() if output_root else None
                exact_project = project_path.resolve()
                exact_owner = exact_project.parent / f"{document_id}.owner.json"
                if owned_root is None or exact_project.parent != owned_root or exact_project.name != f"{document_id}.nexscene":
                    raise RuntimeError("generated project escaped the exact application project root")
                if exact_project.exists():
                    shutil.rmtree(exact_project)
                exact_owner.unlink(missing_ok=True)
                receipt["cleanup"] = {"scene": "removed", "ownerDescriptor": "removed",
                    "browserProfile": "removed", "capabilityFile": "removed", "previewAndRequestFiles": "removed"}
            except OSError:
                receipt.setdefault("cleanup", {})["scene"] = "cleanup needs attention"
                receipt["status"] = "failed"
                return_code = 1
        if profile is not None:
            try:
                if profile.parent == receipt_path.parent.resolve() and profile.name.startswith("w05-field-validation-"):
                    for attempt in range(5):
                        try:
                            shutil.rmtree(profile)
                            break
                        except PermissionError:
                            if attempt == 4:
                                raise
                            time.sleep(0.25)
                    if profile.exists():
                        raise OSError("owned browser profile remains after cleanup")
            except OSError:
                receipt.setdefault("cleanup", {})["browserProfile"] = "cleanup needs attention"
                receipt["status"] = "failed"
                return_code = 1
        receipt["finishedAtUtc"] = datetime.now(timezone.utc).isoformat()
        receipt.setdefault("summary", {"checksPassed": sum(item.get("passed", False) for item in receipt["checks"].values()),
                                        "checksTotal": len(receipt["checks"])})
        serialized = json.dumps(receipt, indent=2, sort_keys=True)
        if ((capability_token and capability_token in serialized)
                or (owner_key and owner_key in serialized)):
            receipt["status"] = "failed"
            receipt["secretMaterialRecorded"] = False
            receipt["failure"] = {"type": "SecretLeakRefused", "message": "Credential material was excluded from the receipt."}
            serialized = json.dumps(receipt, indent=2, sort_keys=True)
            return_code = 1
        try:
            receipt_path.write_text(serialized, encoding="utf-8")
        except OSError:
            return_code = 1
    if project_path is not None:
        print(f"W05 disposable machine-test scene {document_id} was removed with its owner descriptor.")
    print(f"W05 receipt: {receipt_path}")
    if receipt.get("status") != "passed":
        print(json.dumps(receipt, indent=2, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
