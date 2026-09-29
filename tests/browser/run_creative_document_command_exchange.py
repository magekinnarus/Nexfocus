r"""Live W04 human/Agent exact-revision exchange using Gradio, Chrome, and the local CLI.

Run from the repository root with:

  venv\Scripts\python.exe tests\browser\run_creative_document_command_exchange.py \
    --receipt D:\AI\Nexfocus_lab\.agent\temp\P6-M01-W04_exchange.json

The script creates one generated scene, grants a temporary local capability,
performs the W04 CLI/UI exchange, revokes the grant, and removes only the
generated scene, owner descriptor, browser profile, and temporary capability.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def _wait_http(url: str, *, timeout: float = 150) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                response.read(1024)
                return
        except Exception as exc:
            last_error = exc
            time.sleep(0.35)
    raise TimeoutError(f"local application did not start: {type(last_error).__name__ if last_error else 'timeout'}")


def _check(receipt: dict[str, Any], name: str, passed: bool, evidence: Any) -> None:
    receipt["checks"][name] = {"passed": bool(passed), "evidence": evidence}
    if not passed:
        raise AssertionError(f"live exchange check failed: {name}")


def _envelope(
    document_id: str,
    intent: str,
    command_type: str,
    *,
    revision: int | None,
    payload: dict[str, Any] | None = None,
    target_ids: list[str] | None = None,
) -> dict[str, Any]:
    from modules.creative_document import make_id
    from modules.creative_document.command_service import _canonical_target_ids

    inspect = intent == "inspect"
    command_payload = {} if inspect else (payload or {})
    canonical_targets = _canonical_target_ids(command_type, command_payload, tuple(target_ids or ()))
    return {
        "schemaVersion": 1,
        "commandId": make_id("cmd"),
        "documentId": document_id,
        "intent": intent,
        "expectedRevision": revision,
        "commandType": command_type,
        "targetIds": list(canonical_targets),
        "coordinateSpace": "document",
        "payload": command_payload,
        "transaction": None if inspect else {
            "transactionId": make_id("txn"), "groupId": make_id("grp"), "phase": "commit",
        },
    }


def _run_cli(repo: Path, capability: Path, *arguments: str) -> tuple[int, Any]:
    command = [str(repo / "venv" / "Scripts" / "python.exe"), "tools/creative_document_driver.py",
               "--capability-file", str(capability), *arguments]
    result = subprocess.run(command, cwd=repo, capture_output=True, text=True, timeout=60, check=False)
    try:
        decoded = json.loads(result.stdout) if result.stdout.strip() else None
    except json.JSONDecodeError as exc:
        raise RuntimeError("local CLI returned malformed output") from exc
    if result.returncode not in {0, 2}:
        # Do not copy arbitrary process text into the retained security receipt.
        raise RuntimeError(f"local CLI failed with exit code {result.returncode}")
    return result.returncode, decoded


def _write_request(profile: Path, label: str, envelope: dict[str, Any]) -> Path:
    path = profile / f"{label}.json"
    path.write_text(json.dumps(envelope, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    receipt_path = args.receipt.resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "workOrder": "P6-M01-W04",
        "evidence": "generated-fixture live Gradio/Chrome/CLI exact-revision exchange",
        "startedAtUtc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "checks": {},
    }
    server: subprocess.Popen[Any] | None = None
    chrome: subprocess.Popen[Any] | None = None
    cdp = None
    output_root: Path | None = None
    document_id: str | None = None
    capability_file: Path | None = None
    grant_id: str | None = None
    capability_token: str | None = None
    profile: Path | None = None
    log_stream = None
    success = False
    scene_saved = False
    try:
        w03 = _import_w03_helpers()
        output_root = w03.read_server_output_root(repo)
        temp_root = receipt_path.parent
        profile = Path(tempfile.mkdtemp(prefix="w04-command-exchange-", dir=temp_root)).resolve()
        downloads = profile / "downloads"
        downloads.mkdir()
        log_path = profile / "gradio.log"
        log_stream = log_path.open("w", encoding="utf-8", errors="replace")
        server_port = 7861 if not _port_open(7861) else _free_port()
        browser_port = _free_port()
        browser_profile = profile / "chrome-profile"
        browser_profile.mkdir()
        python = repo / "venv" / "Scripts" / "python.exe"
        server = subprocess.Popen(
            [str(python), "webui.py", "--skip-model-load", "--disable-analytics", "--port", str(server_port), "--disable-in-browser"],
            cwd=repo, stdout=log_stream, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        base_url = f"http://127.0.0.1:{server_port}"
        _wait_http(base_url)
        _check(receipt, "loopback_gradio_startup", True, {"status": "ready", "loopback": True})

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
        cdp.evaluate("""(()=>{const e=document.querySelector('creative-document-editor');
          const tab=[...document.querySelectorAll('[role=tab]')].find(item=>item.textContent.trim()==='Creative Document');
          if(tab) tab.click(); window.confirm=()=>true; window.prompt=(_text,initial)=>initial||''; return !!e;})()""")
        w03.wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return !!e?._capability?.enabled;})()", timeout=60)
        mounted = cdp.evaluate("(()=>{const e=document.querySelector('creative-document-editor'); return {mounted:!!e,renderer:window.Konva?.version||null,session:!!e?._capability?.enabled};})()")
        _check(receipt, "w03_editor_mount", mounted["mounted"] and mounted["session"] is True and mounted["renderer"] == "10.6.0", mounted)

        created = cdp.evaluate("""(async()=>{const e=document.querySelector('creative-document-editor');
          const result=await e._request('/creative_document_api/documents',{method:'POST',body:{width:96,height:72,name:'W04 generated exchange'}});
          e._doc=result.view; e._selectedLayerId=e._doc.rootLayerIds[0]; await e._acceptView(e._doc);
          return {documentId:e._doc.documentId,revision:e._doc.revision,rootLayerId:e._doc.rootLayerIds[0]};})()""")
        document_id = created["documentId"]
        _check(receipt, "director_created_live_document", created["revision"] == 0,
               {"documentId": document_id, "revision": created["revision"]})

        enabled = cdp.evaluate("""(async()=>{const e=document.querySelector('creative-document-editor');
          e.querySelector('[data-agent-scope=inspect]').checked=true;
          e.querySelector('[data-agent-scope=propose]').checked=false;
          e.querySelector('[data-agent-scope=mutate]').checked=true;
          e.querySelector('[data-action=agent-enable]').click();
          return true;})()""")
        w03.wait_js(cdp, "document.querySelector('creative-document-editor')?.querySelector('[data-role=agent-driver-status]')?.textContent.includes('Capability file downloaded.')", timeout=30)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            candidates = list(downloads.glob("nexfocus-driver-*.json"))
            if candidates:
                capability_file = candidates[0]
                break
            time.sleep(0.1)
        if capability_file is None:
            raise RuntimeError("browser did not produce the explicit capability download")
        capability = json.loads(capability_file.read_text(encoding="utf-8"))
        capability_token = capability.get("token")
        if (capability.get("documentId") != document_id or capability.get("scopes") != ["inspect", "mutate"]
                or not isinstance(capability_token, str)):
            raise RuntimeError("downloaded capability contract did not match the requested grant")
        # Read the grant ID and public status through the authenticated human API;
        # the capability token is intentionally excluded from the retained receipt.
        grant_status = cdp.evaluate("""(async()=>{const e=document.querySelector('creative-document-editor');
          return await e._request(`/creative_document_api/documents/${encodeURIComponent(e._doc.documentId)}/agent-grants/status`);})()""")
        grant_id = grant_status["grantId"]
        _check(receipt, "explicit_grant_issue", grant_status["status"] == "enabled" and grant_status["scopes"] == ["inspect", "mutate"],
               {"status": grant_status["status"], "grantId": grant_id, "scopes": grant_status["scopes"],
                "expiresAt": grant_status["expiresAt"], "tokenIncluded": False})

        inspect0 = _envelope(document_id, "inspect", "inspect_document", revision=None)
        inspect0_path = _write_request(profile, "inspect-r0", inspect0)
        code, inspect0_receipt = _run_cli(repo, capability_file, "inspect", "--request", str(inspect0_path))
        projection0 = inspect0_receipt["result"]
        _check(receipt, "cli_safe_inspect_revision_n", code == 0 and inspect0_receipt["status"] == "ok"
               and projection0["observedRevision"] == 0 and projection0["snapshotToken"] == f"{document_id}@0",
               {"revision": projection0.get("observedRevision"), "snapshotToken": projection0.get("snapshotToken")})

        preview_path = profile / "safe-preview-r0.png"
        preview_code, preview_receipt = _run_cli(repo, capability_file, "preview", "--expected-revision", "0",
                                                 "--output", str(preview_path))
        _check(receipt, "cli_safe_preview_revision_n", preview_code == 0 and preview_receipt["revision"] == 0
               and preview_path.is_file() and preview_path.stat().st_size > 0,
               {"revision": preview_receipt.get("revision"), "bytes": preview_path.stat().st_size if preview_path.exists() else 0})

        agent_batch = _envelope(document_id, "mutate", "batch", revision=0, payload={"actions": [
            {"actionType": "add_layer", "data": {"kind": "vector", "name": "Agent composition"}},
            {"actionType": "create_object", "data": {
                "layerId": created["rootLayerId"], "kind": "shape",
                "geometry": {"shape": "rectangle", "x": 18, "y": 16, "width": 30, "height": 22},
                "style": {"fill": "#54a8d8", "stroke": "#54a8d8", "width": 1, "opacity": 1},
            }},
        ]})
        batch_path = _write_request(profile, "agent-r0-batch", agent_batch)
        agent_code, agent_receipt = _run_cli(repo, capability_file, "mutate", "--request", str(batch_path))
        if agent_code != 0 or agent_receipt.get("currentRevision") != 1:
            raise RuntimeError("Agent's revision-zero batch did not commit exactly once")
        inspect1 = _envelope(document_id, "inspect", "inspect_document", revision=1)
        inspect1_path = _write_request(profile, "inspect-r1", inspect1)
        _, inspect1_receipt = _run_cli(repo, capability_file, "inspect", "--request", str(inspect1_path))
        projection1 = inspect1_receipt["result"]
        objects1 = projection1["objects"]
        rough_object = next(identity for identity in agent_receipt["createdIds"] if identity in objects1)
        agent_layer = next(layer for layer in projection1["layers"].values() if layer["name"] == "Agent composition")
        _check(receipt, "agent_revision_n_to_n_plus_one", agent_receipt["previousRevision"] == 0
               and agent_receipt["newRevision"] == 1 and agent_receipt["actorKind"] == "agent"
               and rough_object in objects1 and agent_layer["kind"] == "vector",
               {"from": 0, "to": 1, "layerId": agent_layer["layerId"], "objectId": rough_object,
                "commandId": agent_receipt["commandId"], "transactionId": agent_receipt["transactionId"]})

        # Refresh the Director's editor view, then use the accepted W03
        # _sendAction path to transform the same object identity.
        cdp.evaluate("""(async()=>{const e=document.querySelector('creative-document-editor');
          await e._refreshAuthoritativeView(); e._selectedObjectIds=new Set([""" + json.dumps(rough_object) + """]);
          await e._sendAction('transform',{targetIds:[""" + json.dumps(rough_object) + """],
            transform:[1,0,9,0,1,6,0,0,1]}); return true;})()""")
        w03.wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.revision===2;})()", timeout=45)
        human_evidence = cdp.evaluate("""(()=>{const e=document.querySelector('creative-document-editor');
          const object=e._doc.objects.find(item=>item.id===""" + json.dumps(rough_object) + """);
          return {revision:e._doc.revision,objectId:object?.id,transform:object?.transform,actorKind:'human-adapter'};})()""")
        _check(receipt, "director_edits_same_stable_object_n_plus_one_to_n_plus_two",
               human_evidence["revision"] == 2 and human_evidence["objectId"] == rough_object
               and human_evidence["transform"][2] == 9 and human_evidence["transform"][5] == 6,
               human_evidence)

        inspect2 = _envelope(document_id, "inspect", "inspect_document", revision=2)
        inspect2_path = _write_request(profile, "inspect-r2", inspect2)
        _, inspect2_receipt = _run_cli(repo, capability_file, "inspect", "--request", str(inspect2_path))
        seen2 = inspect2_receipt["result"]
        _check(receipt, "agent_observes_director_revision_n_plus_two", seen2["observedRevision"] == 2
               and seen2["objects"][rough_object]["transform"][2] == 9,
               {"revision": seen2["observedRevision"], "objectId": rough_object,
                "directorTranslation": seen2["objects"][rough_object]["transform"][2:6]})

        continue_command = _envelope(document_id, "mutate", "create_object", revision=2, payload={"data": {
            "layerId": agent_layer["layerId"], "kind": "shape",
            "geometry": {"shape": "ellipse", "x": 54, "y": 20, "width": 18, "height": 18},
            "style": {"fill": "#e8aa52", "stroke": "#e8aa52", "width": 1, "opacity": 1},
        }})
        continue_path = _write_request(profile, "agent-r2-continue", continue_command)
        continue_code, continue_receipt = _run_cli(repo, capability_file, "mutate", "--request", str(continue_path))
        _check(receipt, "agent_continues_n_plus_two_to_n_plus_three", continue_code == 0
               and continue_receipt["previousRevision"] == 2 and continue_receipt["newRevision"] == 3
               and continue_receipt["actorKind"] == "agent",
               {"from": 2, "to": 3, "createdIds": continue_receipt["createdIds"],
                "commandId": continue_receipt["commandId"], "transactionId": continue_receipt["transactionId"]})

        inspect3 = _envelope(document_id, "inspect", "inspect_document", revision=3)
        inspect3_path = _write_request(profile, "inspect-r3", inspect3)
        _, inspect3_receipt = _run_cli(repo, capability_file, "inspect", "--request", str(inspect3_path))
        _check(receipt, "agent_inspects_before_stale_mutation", inspect3_receipt["result"]["observedRevision"] == 3,
               {"revision": inspect3_receipt["result"]["observedRevision"]})

        # The Director changes the draft layer after the Agent captured r3.
        cdp.evaluate("""(async()=>{const e=document.querySelector('creative-document-editor');
          await e._refreshAuthoritativeView();
          await e._sendAction('rename_layer',{layerId:""" + json.dumps(agent_layer["layerId"]) + """,name:'Director composition'});
          return true;})()""")
        w03.wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.revision===4;})()", timeout=45)
        stale_command = _envelope(document_id, "mutate", "add_layer", revision=3,
                                  payload={"data": {"kind": "vector", "name": "Must not appear"}})
        stale_path = _write_request(profile, "agent-stale-r3", stale_command)
        stale_code, stale_receipt = _run_cli(repo, capability_file, "mutate", "--request", str(stale_path))
        _check(receipt, "stale_agent_mutation_refused_zero_writes", stale_code == 2
               and stale_receipt["error"]["code"] == "STALE_DOCUMENT_REVISION"
               and stale_receipt["previousRevision"] == 3 and stale_receipt["currentRevision"] == 4
               and stale_receipt["writes"] == 0,
               {"expectedRevision": 3, "currentRevision": 4, "writes": stale_receipt["writes"],
                "code": stale_receipt["error"]["code"]})

        refresh = _envelope(document_id, "inspect", "inspect_document", revision=4)
        refresh_path = _write_request(profile, "agent-refresh-r4", refresh)
        _, refresh_receipt = _run_cli(repo, capability_file, "inspect", "--request", str(refresh_path))
        refreshed = refresh_receipt["result"]
        refreshed_layer = refreshed["layers"][agent_layer["layerId"]]
        fresh_edit = _envelope(document_id, "mutate", "set_layer_opacity", revision=4, payload={"data": {
            "layerId": agent_layer["layerId"], "opacity": 0.8,
        }})
        fresh_path = _write_request(profile, "agent-fresh-r4", fresh_edit)
        fresh_code, fresh_receipt = _run_cli(repo, capability_file, "mutate", "--request", str(fresh_path))
        _check(receipt, "agent_explicit_refresh_then_continues", refreshed["observedRevision"] == 4
               and refreshed_layer["name"] == "Director composition" and fresh_code == 0
               and fresh_receipt["previousRevision"] == 4 and fresh_receipt["newRevision"] == 5,
               {"refreshedRevision": refreshed["observedRevision"], "layerName": refreshed_layer["name"],
                "continuedTo": fresh_receipt["newRevision"]})

        cdp.evaluate("""(async()=>{const e=document.querySelector('creative-document-editor');
          await e._refreshAuthoritativeView(); await e._saveDocument(); return true;})()""")
        w03.wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return e._doc && !e._doc.dirty && e._doc.committedRevision===e._doc.revision;})()", timeout=45)
        scene_saved = True
        scene_root = (output_root / "creative_documents" / f"{document_id}.nexscene").resolve()
        owned_projects = (output_root / "creative_documents").resolve()
        if scene_root.parent != owned_projects:
            raise RuntimeError("generated scene escaped the application project root")
        scene_saved = True
        before_reopen = cdp.evaluate("""(()=>{const e=document.querySelector('creative-document-editor');
          return {documentId:e._doc.documentId,revision:e._doc.revision,
            objectIds:e._doc.objects.map(item=>item.id),layerIds:e._doc.layers.map(item=>item.id)};})()""")
        cdp.evaluate("document.querySelector('creative-document-editor')._openDocument()")
        w03.wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return e._doc && e._doc.recoveryNotice && !e._doc.dirty;})()", timeout=45)
        after_reopen = cdp.evaluate("""(()=>{const e=document.querySelector('creative-document-editor');
          return {documentId:e._doc.documentId,revision:e._doc.revision,
            objectIds:e._doc.objects.map(item=>item.id),layerIds:e._doc.layers.map(item=>item.id)};})()""")
        from modules.creative_document import ProjectStore

        persisted = ProjectStore(scene_root).open()
        agent_transactions = [entry for entry in persisted.history if entry["actorKind"] == "agent"]
        director_transactions = [entry for entry in persisted.history
                                 if entry["actorKind"] == "human" and rough_object in entry["affectedIds"]]
        receipts_present = all(isinstance(entry.get("metadata", {}).get("semanticCommand", {}).get("receipt"), dict)
                               for entry in agent_transactions)
        _check(receipt, "save_reopen_preserves_ids_actor_provenance_and_receipts",
               before_reopen["documentId"] == after_reopen["documentId"] == document_id
               and before_reopen["revision"] == after_reopen["revision"] == 5
               and set(before_reopen["objectIds"]) == set(after_reopen["objectIds"])
               and set(before_reopen["layerIds"]) == set(after_reopen["layerIds"])
               and len(agent_transactions) >= 2 and len(director_transactions) >= 1 and receipts_present,
               {"documentId": after_reopen["documentId"], "revision": after_reopen["revision"],
                "stableObjectIds": rough_object in after_reopen["objectIds"],
                "agentTransactions": len(agent_transactions), "directorObjectEditTransactions": len(director_transactions),
                "semanticReceiptsRetained": receipts_present})

        cdp.evaluate("document.querySelector('creative-document-editor')._revokeAgentDriver()")
        w03.wait_js(cdp, "document.querySelector('creative-document-editor')?.querySelector('[data-role=agent-driver-status]')?.textContent.includes('revoked')", timeout=30)
        status_code, revoked_status = _run_cli(repo, capability_file, "status")
        after_revoke_command = _envelope(document_id, "mutate", "add_layer", revision=5,
                                         payload={"data": {"kind": "vector", "name": "Revoked write"}})
        after_revoke_path = _write_request(profile, "after-revoke.json", after_revoke_command)
        mutation_code, revoked_mutation = _run_cli(repo, capability_file, "mutate", "--request", str(after_revoke_path))
        final_revision = cdp.evaluate("document.querySelector('creative-document-editor')._doc.revision")
        _check(receipt, "revoked_grant_blocks_following_reads_and_writes",
               status_code == 2 and revoked_status.get("status") == "refused"
               and mutation_code == 2 and revoked_mutation["error"]["code"] == "AUTHORIZATION_REQUIRED"
               and revoked_mutation["writes"] == 0 and final_revision == 5,
               {"statusRefused": status_code == 2, "mutationRefused": mutation_code == 2,
                "mutationCode": revoked_mutation["error"]["code"], "writes": revoked_mutation["writes"],
                "revision": final_revision})

        receipt["grant"] = {"grantId": grant_id, "scopes": ["inspect", "mutate"],
                            "expiresAt": grant_status["expiresAt"], "revoked": True, "bearerIncluded": False}
        receipt["document"] = {"documentId": document_id, "finalRevision": final_revision,
                               "stableObjectId": rough_object}
        success = all(item["passed"] for item in receipt["checks"].values())
        receipt["status"] = "passed" if success else "failed"
        return_code = 0 if success else 1
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        return_code = 1
    finally:
        if cdp is not None:
            try:
                cdp.close()
            except Exception:
                pass
        for process in (chrome, server):
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        if log_stream is not None:
            log_stream.close()
        if output_root is not None and document_id:
            owned_projects = (output_root / "creative_documents").resolve()
            scene_root = (owned_projects / f"{document_id}.nexscene").resolve()
            owner_file = owned_projects / f"{document_id}.owner.json"
            try:
                if scene_root.parent == owned_projects and scene_root.exists():
                    shutil.rmtree(scene_root)
                if owner_file.parent.resolve() == owned_projects:
                    owner_file.unlink(missing_ok=True)
            except OSError:
                receipt.setdefault("cleanup", []).append("generated project cleanup needs attention")
        if profile is not None:
            try:
                if profile.parent == receipt_path.parent.resolve() and profile.name.startswith("w04-command-exchange-"):
                    shutil.rmtree(profile)
            except OSError:
                receipt.setdefault("cleanup", []).append("generated browser/capability profile cleanup needs attention")
        receipt["finishedAtUtc"] = datetime.now(timezone.utc).isoformat()
        receipt.setdefault("summary", {"checksPassed": sum(item.get("passed", False) for item in receipt["checks"].values()),
                                        "checksTotal": len(receipt["checks"])})
        try:
            serialized = json.dumps(receipt, indent=2, sort_keys=True)
            if capability_token and capability_token in serialized:
                receipt["status"] = "failed"
                receipt["failure"] = {"type": "SecretLeakRefused", "message": "Bearer material was excluded from the receipt."}
                serialized = json.dumps(receipt, indent=2, sort_keys=True)
                return_code = 1
            receipt_path.write_text(serialized, encoding="utf-8")
        except OSError:
            return_code = 1
    if not success:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
