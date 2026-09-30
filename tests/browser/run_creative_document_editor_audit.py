"""Reproducible headless browser audit for the W03 creative-document editor.

Run from the repository root with:

  venv\\Scripts\\python.exe tests\\browser\\run_creative_document_editor_audit.py \\
    --receipt D:\\AI\\Nexfocus_lab\\.agent\\temp\\P6-M01-W03_Audit01_corrective_browser.json

The script starts a loopback Gradio instance and headless Chrome, performs real
DOM/pointer/CDP interactions, writes a machine-readable receipt, and removes
only the scene and probe files it created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import websocket


CHROME_CANDIDATES = (
    Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "Google/Chrome/Application/chrome.exe",
    Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Google/Chrome/Application/chrome.exe",
)


class CDP:
    def __init__(self, address: str) -> None:
        self.connection = websocket.create_connection(address, timeout=30, suppress_origin=True)
        self.connection.settimeout(30)
        self.sequence = 0

    def command(self, method: str, params: dict[str, Any] | None = None, *, timeout: float = 30) -> dict[str, Any]:
        self.sequence += 1
        identity = self.sequence
        self.connection.send(json.dumps({"id": identity, "method": method, "params": params or {}}))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            message = json.loads(self.connection.recv())
            if message.get("id") != identity:
                continue
            if "error" in message:
                raise RuntimeError(f"Chrome DevTools {method}: {message['error']}")
            return message.get("result", {})
        raise TimeoutError(f"Chrome DevTools command timed out: {method}")

    def evaluate(self, expression: str, *, timeout: float = 30) -> Any:
        result = self.command("Runtime.evaluate", {
            "expression": expression,
            "awaitPromise": True,
            "returnByValue": True,
            "userGesture": True,
        }, timeout=timeout)
        if "exceptionDetails" in result:
            details = result["exceptionDetails"]
            raise RuntimeError(details.get("exception", {}).get("description") or details.get("text") or "browser expression failed")
        remote = result.get("result", {})
        if remote.get("subtype") == "error":
            raise RuntimeError(remote.get("description", "browser expression failed"))
        return remote.get("value")

    def close(self) -> None:
        self.connection.close()


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def port_is_open(port: int) -> bool:
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) == 0


def wait_http(url: str, *, timeout: float = 120) -> bytes:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                return response.read()
        except Exception as exc:
            last_error = exc
            time.sleep(0.35)
    raise TimeoutError(f"service did not answer {url}: {last_error}")


def wait_js(cdp: CDP, expression: str, *, timeout: float = 30) -> Any:
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        last = cdp.evaluate(expression, timeout=10)
        if last:
            return last
        time.sleep(0.1)
    raise TimeoutError(f"browser condition timed out: {expression}; last={last!r}")


def check(receipt: dict[str, Any], name: str, passed: bool, evidence: Any) -> None:
    receipt["checks"][name] = {"passed": bool(passed), "evidence": evidence}
    if not passed:
        raise AssertionError(f"browser audit check failed: {name}: {evidence!r}")


def mouse(cdp: CDP, kind: str, x: float, y: float, *, button: str = "left", buttons: int = 0) -> None:
    cdp.command("Input.dispatchMouseEvent", {
        "type": kind,
        "x": float(x),
        "y": float(y),
        "button": button,
        "buttons": buttons,
        "pointerType": "mouse",
    })


def drag(cdp: CDP, start: dict[str, float], end: dict[str, float], *, button: str = "left") -> None:
    button_mask = {"left": 1, "middle": 4, "right": 2}[button]
    mouse(cdp, "mouseMoved", start["x"], start["y"])
    mouse(cdp, "mousePressed", start["x"], start["y"], button=button, buttons=button_mask)
    steps = 8
    for index in range(1, steps + 1):
        ratio = index / steps
        mouse(cdp, "mouseMoved", start["x"] + (end["x"] - start["x"]) * ratio,
              start["y"] + (end["y"] - start["y"]) * ratio, button=button, buttons=button_mask)
        time.sleep(0.015)
    mouse(cdp, "mouseReleased", end["x"], end["y"], button=button)


def read_server_output_root(repo: Path) -> Path:
    sys.path.insert(0, str(repo))
    original_argv = sys.argv
    try:
        # The product config parses its launch flags at import time.
        sys.argv = [str(repo / "webui.py")]
        import modules.config  # type: ignore[import-not-found]

        return Path(modules.config.path_outputs).resolve()
    finally:
        sys.argv = original_argv


def create_renderer_issue_fixture(output_root: Path, owner_key: str) -> str:
    """Write a valid native project whose server view reports unsupported semantics."""
    from modules.creative_document import Document, LayerRecord, ObjectRecord, ProjectStore, make_id

    document_id = make_id("doc")
    blend_id, clipping_id, text_id, shape_id = (make_id("layer") for _ in range(4))
    text_object_id, shape_object_id = make_id("obj"), make_id("obj")
    layers = {
        blend_id: LayerRecord(blend_id, "Multiply blend fixture", "vector", blend_mode="multiply"),
        clipping_id: LayerRecord(clipping_id, "Clipping fixture", "vector", clipping_refs=[blend_id]),
        text_id: LayerRecord(text_id, "Unsupported text fixture", "text", object_ids=[text_object_id]),
        shape_id: LayerRecord(shape_id, "Unsupported shape fixture", "vector", object_ids=[shape_object_id]),
    }
    objects = {
        text_object_id: ObjectRecord(text_object_id, text_id, "text-placement"),
        shape_object_id: ObjectRecord(shape_object_id, shape_id, "shape", geometry={
            "shape": "star", "x": 4, "y": 5, "width": 30, "height": 24,
        }),
    }
    document = Document(document_id, 80, 64, root_layer_ids=[blend_id, clipping_id, text_id, shape_id],
                        layers=layers, objects=objects)
    projects_root = output_root / "creative_documents"
    projects_root.mkdir(parents=True, exist_ok=True)
    ProjectStore(projects_root / f"{document_id}.nexscene").save(document, checkpoint=True)
    principal_id = hashlib.sha256(f"browser:{owner_key}".encode("utf-8")).hexdigest()
    owner_record = {"version": 1, "documentId": document_id, "principalId": principal_id}
    (projects_root / f"{document_id}.owner.json").write_text(
        json.dumps(owner_record, sort_keys=True, separators=(",", ":")), encoding="utf-8",
    )
    return document_id


def generic_file_probe(
    cdp: CDP,
    base_url: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    preserve_escapes: bool = False,
) -> dict[str, Any]:
    escaped = urllib.parse.quote(path, safe="%" if preserve_escapes else "")
    route = f"/gradio_api/file={escaped}"
    expression = "(async()=>{const r=await fetch(" + json.dumps(route) + ",{" + (
        "headers:" + json.dumps(headers) + "," if headers else ""
    ) + "credentials:'same-origin'}); const t=await r.text(); return {status:r.status,type:r.headers.get('content-type'),length:t.length};})()"
    return cdp.evaluate(expression, timeout=15)


def create_directory_alias(target: Path, alias: Path) -> str:
    try:
        os.symlink(target, alias, target_is_directory=True)
        return "symlink"
    except (OSError, NotImplementedError):
        junction = subprocess.run(
            ["cmd.exe", "/c", "mklink", "/J", str(alias), str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        if junction.returncode != 0:
            raise OSError("Windows symlink and junction creation were both unavailable.")
        return "junction"


def remove_directory_alias(alias: Path | None) -> None:
    if alias is None:
        return
    if alias.is_symlink():
        alias.unlink(missing_ok=True)
    elif getattr(os.path, "isjunction", lambda _path: False)(alias):
        os.rmdir(alias)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    repo = args.repo.resolve()
    receipt_path = args.receipt.resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "workOrder": "P6-M01-W03",
        "auditRound": "Audit 02 corrective browser evidence",
        "startedAtUtc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "checks": {},
    }
    server: subprocess.Popen[Any] | None = None
    chrome: subprocess.Popen[Any] | None = None
    cdp: CDP | None = None
    temporary_root = repo.parent / "Nexfocus_lab" / ".agent" / "temp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    profile = Path(tempfile.mkdtemp(prefix="w03-browser-", dir=temporary_root))
    log_path = profile / "gradio.log"
    log_stream = log_path.open("w", encoding="utf-8", errors="replace")
    created_document_id: str | None = None
    renderer_fixture_id: str | None = None
    ordinary_output: Path | None = None
    symlink_alias: Path | None = None
    cache_alias: Path | None = None
    output_root: Path | None = None
    server_port = args.port
    if port_is_open(server_port):
        server_port = free_port()
    browser_port = free_port()
    browser_profile = profile / "chrome-profile"
    browser_profile.mkdir()
    success = False
    try:
        output_root = read_server_output_root(repo)
        python = repo / "venv" / "Scripts" / "python.exe"
        server = subprocess.Popen(
            [str(python), "webui.py", "--skip-model-load", "--disable-analytics", "--port", str(server_port), "--disable-in-browser"],
            cwd=repo,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        base_url = f"http://127.0.0.1:{server_port}"
        wait_http(base_url, timeout=150)
        check(receipt, "gradio_startup", True, {"status": "loopback server ready", "port": server_port})

        chrome_path = next((path for path in CHROME_CANDIDATES if path.is_file()), None)
        if chrome_path is None:
            raise FileNotFoundError("Chrome was not found in the standard Windows application paths.")
        chrome = subprocess.Popen(
            [str(chrome_path), "--headless=new", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage",
             "--no-first-run", "--no-default-browser-check", "--disable-extensions", "--remote-allow-origins=*",
             f"--remote-debugging-port={browser_port}", f"--user-data-dir={browser_profile}",
             "--window-size=1440,1000", "about:blank"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        version = json.loads(wait_http(f"http://127.0.0.1:{browser_port}/json/version", timeout=45))
        targets = json.loads(wait_http(f"http://127.0.0.1:{browser_port}/json", timeout=10))
        target = next(item for item in targets if item.get("type") == "page")
        cdp = CDP(target["webSocketDebuggerUrl"])
        cdp.command("Page.enable")
        cdp.command("Runtime.enable")
        cdp.command("Network.enable")
        cdp.command("Page.navigate", {"url": base_url})
        wait_js(cdp, "document.readyState === 'complete'", timeout=90)
        wait_js(cdp, "document.querySelector('creative-document-editor')", timeout=90)
        main_tab = cdp.evaluate("""(()=>{
          const editor=document.querySelector('creative-document-editor');
          const matches=[...document.querySelectorAll('[role=tab]')].filter(tab=>{
            const panelId=tab.getAttribute('aria-controls');
            return tab.textContent.trim()==='Creative Document' && panelId && document.getElementById(panelId)?.contains(editor);
          });
          if(matches.length!==1) throw new Error(`Creative Document tab association is ambiguous: ${matches.length}`);
          matches[0].click(); window.confirm=()=>false; window.prompt=()=>null;
          return {tab:matches[0].textContent.trim(),panelId:matches[0].getAttribute('aria-controls')};
        })()""")
        check(receipt, "creative_document_tab_selected_by_panel", main_tab["tab"] == "Creative Document", main_tab)
        wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return e && e._capability && e._capability.enabled;})()", timeout=45)
        browser_version = version.get("Browser", "unknown")
        check(receipt, "headless_browser_mount", True, {
            "browser": browser_version,
            "editorElement": bool(cdp.evaluate("!!document.querySelector('creative-document-editor')")),
            "konvaVersion": cdp.evaluate("window.Konva && window.Konva.version"),
        })
        open_controls = cdp.evaluate("""(()=>{
          const e=document.querySelector('creative-document-editor');
          const documentId=e.querySelector('[data-action=document-id]');
          const open=e.querySelector('[data-action=open]');
          const documentControl=e.querySelector('[data-action=color]');
          return {documentOpen:!!e._doc,documentIdDisabled:documentId?.disabled,
            openDisabled:open?.disabled,documentControlDisabled:documentControl?.disabled};
        })()""")
        check(receipt, "document_id_open_control_available_without_open_scene",
              open_controls["documentOpen"] is False and open_controls["documentIdDisabled"] is False and
              open_controls["openDisabled"] is False and open_controls["documentControlDisabled"] is True,
              open_controls)

        setup = cdp.evaluate("""(async()=>{
          const e=document.querySelector('creative-document-editor');
          const originalFetch=window.fetch.bind(window);
          window.__auditActions=[]; window.__auditFailNextAction=false; window.__auditScheduleCalls=[];
          const originalSchedule=e._scheduleTransformCommit.bind(e);
          e._scheduleTransformCommit=(id)=>{window.__auditScheduleCalls.push({id,stack:new Error().stack.split('\\n').slice(1,4)}); return originalSchedule(id);};
          window.fetch=async(input,init={})=>{
            const url=String(input);
            if(url.includes('/creative_document_api/documents/') && url.endsWith('/actions')){
              const body=typeof init.body==='string'?JSON.parse(init.body):{};
              const entry={actionType:body.actionType,expectedRevision:body.expectedRevision,targetCount:body.actionType==='batch'?(body.actions||[]).reduce((n,a)=>n+(a.targetIds||[]).length,0):(body.targetIds||[]).length,targetIds:body.actionType==='batch'?(body.actions||[]).map(a=>a.targetIds):body.targetIds};
              window.__auditActions.push(entry);
              if(window.__auditFailNextAction){window.__auditFailNextAction=false;entry.injectedFailure=true;return new Response(JSON.stringify({detail:{code:'STALE_DOCUMENT_REVISION',message:'Injected stale revision.',currentRevision:e._doc.revision,refreshHint:'reload-document-view'}}),{status:409,headers:{'content-type':'application/json'}});}
              const response=await originalFetch(input,init); entry.status=response.status;
              const receipt=await response.clone().json().catch(()=>null); if(receipt) entry.currentRevision=receipt.currentRevision;
              return response;
            }
            return originalFetch(input,init);
          };
          const created=await e._request('/creative_document_api/documents',{method:'POST',body:{width:2400,height:1792,name:'W03 audit scene'}});
          e._doc=created.view; e._selectedLayerId=e._doc.rootLayerIds[0]; await e._acceptView(e._doc);
          const canvas=document.createElement('canvas'); canvas.width=2400; canvas.height=1792;
          const ctx=canvas.getContext('2d'); ctx.fillStyle='#2b68bb'; ctx.fillRect(0,0,1200,1792); ctx.fillStyle='#c65436'; ctx.fillRect(1200,0,1200,1792);
          const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/png'));
          window.__auditLargeBlob=blob;
          const form=new FormData(); form.append('file',new File([blob],'audit-source.png',{type:'image/png'}),'audit-source.png');
          form.append('expected_revision',String(e._doc.revision)); form.append('actor_kind','director');
          const imported=await e._request(`/creative_document_api/documents/${encodeURIComponent(e._doc.documentId)}/assets/import`,{method:'POST',body:form});
          await e._refreshView(imported);
          const raster=e._doc.layers.find(layer=>layer.kind==='raster');
          const asset=e._doc.assets.find(item=>item.id===e._doc.objects.find(object=>object.layerId===raster.id)?.assetId);
          await e._loadImageAsset(asset,e._controller.signal);
          const layerReceipt=await e._sendAction('add_layer',{kind:'vector',name:'Ink layer'});
          const ink=e._doc.layers.find(layer=>layer.name==='Ink layer');
          await e._sendAction('add_layer',{kind:'vector',name:'Depth Finish'});
          const depth=e._doc.layers.find(layer=>layer.name==='Depth Finish');
          await e._sendAction('create_object',{layerId:ink.id,kind:'shape',geometry:{shape:'rectangle',x:480,y:390,width:240,height:180},style:{fill:'#e7b45a',stroke:'#e7b45a',width:1}});
          const first=e._doc.objects[e._doc.objects.length-1].id;
          await e._sendAction('create_object',{layerId:ink.id,kind:'shape',geometry:{shape:'rectangle',x:920,y:430,width:240,height:180},style:{fill:'#4d91e8',stroke:'#4d91e8',width:1}});
          const second=e._doc.objects[e._doc.objects.length-1].id;
          e._selectedLayerId=ink.id; e._selectedObjectIds=new Set([first]); e._tool='select'; e._updateTransformer(); e._refreshToolButtons();
          return {documentId:e._doc.documentId,revision:e._doc.revision,rasterLayerId:raster.id,assetId:asset.id,inkLayerId:ink.id,depthLayerId:depth.id,firstObjectId:first,secondObjectId:second,layerReceipt:!!layerReceipt};
        })()""")
        created_document_id = setup["documentId"]
        wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.revision >= 4;})()", timeout=90)
        check(receipt, "large_raster_and_supported_scene_mount", True, {
            "dimensions": [2400, 1792], "loadedAsset": setup["assetId"],
            "rendererLayers": cdp.evaluate("document.querySelector('creative-document-editor')._stage.getLayers().length"),
            "canvasCount": cdp.evaluate("document.querySelector('creative-document-editor')._stageContainer.querySelectorAll('canvas').length"),
        })

        image_input_control = cdp.evaluate("""(()=>{const inputs=[...document.querySelectorAll('.advanced_check_row input[type=checkbox]')];
          if(inputs.length!==1) throw new Error(`Input Image checkbox association is ambiguous: ${inputs.length}`);
          const input=inputs[0];
          input?.scrollIntoView({block:'center'});
          const rect=input?.getBoundingClientRect();
          return {checked:!!input?.checked,disabled:!!input?.disabled,rect:rect?{x:rect.x,y:rect.y,width:rect.width,height:rect.height}:null};})()""")
        if not image_input_control["checked"]:
            rect = image_input_control["rect"]
            if not rect or rect["width"] <= 0 or rect["height"] <= 0:
                raise AssertionError(f"Input Image checkbox is not clickable: {image_input_control!r}")
            x = rect["x"] + rect["width"] / 2
            y = rect["y"] + rect["height"] / 2
            mouse(cdp, "mouseMoved", x, y)
            mouse(cdp, "mousePressed", x, y, buttons=1)
            mouse(cdp, "mouseReleased", x, y)
            toggled = cdp.evaluate("document.querySelector('.advanced_check_row input[type=checkbox]')?.checked === true")
            if not toggled:
                cdp.evaluate("""(()=>{const input=document.querySelector('.advanced_check_row input[type=checkbox]');
                  const setter=Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'checked')?.set;
                  if(input&&setter){setter.call(input,true);input.dispatchEvent(new Event('input',{bubbles:true}));input.dispatchEvent(new Event('change',{bubbles:true}));}
                })()""")
                wait_js(cdp, "document.querySelector('.advanced_check_row input[type=checkbox]')?.checked === true", timeout=5)
        wait_js(cdp, "[...document.querySelectorAll('[role=tab]')].some(item=>item.textContent.trim()==='Upscale')", timeout=15)
        slot_tabs = cdp.evaluate("""(()=>{
          const candidates=[...document.querySelectorAll('[role=tablist]')].filter(list=>{
            const labels=[...list.querySelectorAll('[role=tab]')].map(item=>item.textContent.trim()),rect=list.getBoundingClientRect();
            return labels.includes('Inpaint') && labels.includes('Upscale') && rect.width>10 && rect.height>10 && !list.classList.contains('visually-hidden');
          });
          if(candidates.length!==1) throw new Error(`Expected one visible Inpaint/Upscale nested tablist, found ${candidates.length}`);
          return {count:candidates.length,labels:[...candidates[0].querySelectorAll('[role=tab]')].map(item=>item.textContent.trim()),class:candidates[0].className};
        })()""")
        check(receipt, "input_image_nested_tablist_is_visible_and_unambiguous", slot_tabs["count"] == 1 and
              "Inpaint" in slot_tabs["labels"] and "Upscale" in slot_tabs["labels"], slot_tabs)

        populated_slots = cdp.evaluate("""(async()=>{
          const waitFor=async(selector)=>{for(let attempt=0;attempt<100;attempt++){
            const found=document.querySelector(selector); if(found) return found; await new Promise(resolve=>setTimeout(resolve,50));}
            throw new Error(`timed out waiting for ${selector}`);};
          const candidates=[...document.querySelectorAll('[role=tablist]')].filter(list=>{
            const labels=[...list.querySelectorAll('[role=tab]')].map(item=>item.textContent.trim()),rect=list.getBoundingClientRect();
            return labels.includes('Inpaint') && labels.includes('Upscale') && rect.width>10 && rect.height>10 && !list.classList.contains('visually-hidden');
          });
          if(candidates.length!==1) throw new Error(`visible nested image tablist count=${candidates.length}`);
          const clickSlotTab=async(label,slot)=>{
            const tabs=[...candidates[0].querySelectorAll('[role=tab]')].filter(item=>item.textContent.trim()===label);
            if(tabs.length!==1) throw new Error(`tab ${label} is ambiguous within the nested image tablist: ${tabs.length}`);
            const tab=tabs[0];
            tab.click();
            await waitFor(slot);
            if(tab.getAttribute('aria-selected')!=='true') throw new Error(`${label} tab did not become selected`);
          };
          await clickSlotTab('Inpaint','#inpaint_mask_canvas');
          await clickSlotTab('Upscale','#uov_input_slot');
          const upload=async(id,name,blob)=>{
            const slot=document.getElementById(id), input=slot?.querySelector('.nex-slot__input');
            if(!slot || !input) throw new Error(`missing image slot ${id}`);
            const original=slot.handleFile; let resolveDone;
            const done=new Promise(resolve=>{resolveDone=resolve;});
            slot.handleFile=async function(file){try{await original.call(this,file);}finally{this.handleFile=original;resolveDone();}};
            const transfer=new DataTransfer(); transfer.items.add(new File([blob],name,{type:blob.type||'image/png'}));
            input.files=transfer.files; input.dispatchEvent(new Event('change',{bubbles:true})); await done;
          };
          await Promise.all([
            upload('uov_input_slot','audit-uov-input.png',window.__auditLargeBlob),
          ]);
          await clickSlotTab('Inpaint','#inpaint_bb_canvas');
          await upload('inpaint_bb_canvas','audit-inpaint-base.png',window.__auditLargeBlob);
          await Promise.all(['uov_input_slot','inpaint_bb_canvas'].map(id=>{
            const image=document.getElementById(id)?.querySelector('.nex-slot__img');
            return image&&image.complete&&image.naturalWidth?Promise.resolve():new Promise(resolve=>image?.addEventListener('load',resolve,{once:true}));
          }));
          const read=(id,pathId,workspaceId)=>{
            const slot=document.getElementById(id), image=slot?.querySelector('.nex-slot__img');
            return {id,ready:slot?.dataset.uploading==='false' && slot?.dropZone?.classList.contains('has-image') && image?.naturalWidth>0,
              dimensions:[image?.naturalWidth||0,image?.naturalHeight||0],path:!!slot?.getFieldValue(pathId),workspace:!!slot?.getFieldValue(workspaceId)};
          };
          return {uov:read('uov_input_slot','uov_input_image_path','uov_input_workspace_id'),
            inpaintBase:read('inpaint_bb_canvas','inpaint_bb_image_path','inpaint_bb_workspace_id')};
        })()""")
        check(receipt, "uov_and_inpaint_image_slots_populated_and_displayed",
              populated_slots["uov"]["ready"] and populated_slots["uov"]["path"] and populated_slots["uov"]["workspace"] and
              populated_slots["inpaintBase"]["ready"] and populated_slots["inpaintBase"]["path"] and populated_slots["inpaintBase"]["workspace"],
              populated_slots)

        main_tab = cdp.evaluate("""(()=>{
          const editor=document.querySelector('creative-document-editor');
          const panel=editor?.closest('[role=tabpanel]');
          const matches=[...document.querySelectorAll('[role=tab]')].filter(tab=>
            tab.textContent.trim()==='Creative Document' && panel && tab.getAttribute('aria-controls')===panel.id);
          if(matches.length!==1) throw new Error(`Creative Document panel cannot be uniquely restored: ${matches.length}`);
          matches[0].click();
          editor._stageContainer.scrollIntoView({block:'center',inline:'nearest'});
          return {tab:matches[0].textContent.trim(),panelId:panel.id};
        })()""")
        wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); const p=e?.closest('[role=tabpanel]'); const t=[...document.querySelectorAll('[role=tab]')].find(x=>p&&x.getAttribute('aria-controls')===p.id); const r=e?._stageContainer?.getBoundingClientRect(); const s=e?._stage; return !!e && !!p && t?.getAttribute('aria-selected')==='true' && p.getAttribute('aria-hidden')!=='true' && !!r && r.width>0 && r.height>0 && !!s && s.width()>0 && s.height()>0;})()", timeout=30)
        stage_ready = cdp.evaluate("""(()=>{const e=document.querySelector('creative-document-editor'); const p=e.closest('[role=tabpanel]'); const r=e._stageContainer.getBoundingClientRect(); return {tabSelected:[...document.querySelectorAll('[role=tab]')].some(x=>x.getAttribute('aria-controls')===p.id&&x.getAttribute('aria-selected')==='true'),panel:{id:p.id,hidden:p.getAttribute('aria-hidden'),width:p.getBoundingClientRect().width,height:p.getBoundingClientRect().height},stage:{x:r.x,y:r.y,width:r.width,height:r.height,konva:[e._stage.width(),e._stage.height()]}};})()""")
        check(receipt, "creative_document_stage_restored_before_pointer_replay",
              stage_ready["tabSelected"] and stage_ready["panel"]["hidden"] != "true" and
              stage_ready["stage"]["width"] > 0 and stage_ready["stage"]["height"] > 0 and
              all(value > 0 for value in stage_ready["stage"]["konva"]), stage_ready)
        cdp.evaluate("document.querySelector('creative-document-editor')._fitToView()")

        def document_point(x: float, y: float) -> dict[str, float]:
            return cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'); const r=e._stageContainer.getBoundingClientRect(); const s=e._view.fitScale*e._view.zoom; return {{x:r.left+e._view.offsetX+e._view.panX+{x}*s,y:r.top+e._view.offsetY+e._view.panY+{y}*s}};}})()""")

        def visible_stage_point(offset_x: float = 18, offset_y: float = 18) -> dict[str, float]:
            return cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'),r=e._stageContainer.getBoundingClientRect(); return {{x:r.left+Math.min(r.width-2,{offset_x}),y:Math.max(1,Math.min(window.innerHeight-2,r.top+{offset_y}))}};}})()""")

        # Draw a shape with the actual toolbar tool and pointer path.
        cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'); e.querySelector(`[data-layer-id="{setup['inkLayerId']}"]`).click(); e.querySelector('[data-tool=rectangle]').click(); window.__auditActions=[];}})()""")
        draw_revision = cdp.evaluate("document.querySelector('creative-document-editor')._doc.revision")
        drag(cdp, document_point(130, 255), document_point(315, 395))
        wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && window.__auditActions.length===1 && e._doc.revision>0;})()", timeout=30)
        drawn = cdp.evaluate("""(()=>{const e=document.querySelector('creative-document-editor'); const o=e._doc.objects.at(-1); return {revision:e._doc.revision,action:window.__auditActions[0],object:o&&{id:o.id,kind:o.kind,geometry:o.geometry,layerId:o.layerId}};})()""")
        check(receipt, "drawing_tool_pointer_gesture_creates_shape", drawn["revision"] == draw_revision + 1 and
              drawn["action"]["actionType"] == "create_object" and drawn["object"]["kind"] == "shape" and
              drawn["object"]["geometry"].get("shape") == "rectangle" and drawn["object"]["layerId"] == setup["inkLayerId"], drawn)

        # Exercise the layer tree and its real controls, then restore the layer state.
        initial_order = cdp.evaluate("document.querySelector('creative-document-editor')._doc.rootLayerIds.slice()")
        initial_ink_index = initial_order.index(setup["inkLayerId"])
        reorder_direction = "up" if initial_ink_index < len(initial_order) - 1 else "down"
        restore_direction = "down" if reorder_direction == "up" else "up"
        reordered_index = initial_ink_index + (1 if reorder_direction == "up" else -1)
        cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'); window.__auditActions=[]; e.querySelector(`[data-layer-id="{setup['inkLayerId']}"]`).click(); e.querySelector('[data-action=move-{reorder_direction}]').click();}})()""")
        try:
            wait_js(cdp, f"!document.querySelector('creative-document-editor')._actionInFlight && document.querySelector('creative-document-editor')._doc.rootLayerIds.indexOf({json.dumps(setup['inkLayerId'])})==={reordered_index}", timeout=30)
        except TimeoutError:
            diagnostic = cdp.evaluate("(()=>{const e=document.querySelector('creative-document-editor'); return {order:e._doc.rootLayerIds,selectedLayerId:e._selectedLayerId,actionInFlight:e._actionInFlight,actions:window.__auditActions,status:e.querySelector('[data-role=status]')?.innerText};})()")
            receipt["checks"]["layer_reorder_up_diagnostic"] = {"passed": False, "evidence": diagnostic}
            raise
        reordered = cdp.evaluate("document.querySelector('creative-document-editor')._doc.rootLayerIds.slice()")
        cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'); e.querySelector(`[data-layer-id="{setup['inkLayerId']}"]`).click(); e.querySelector('[data-action=move-{restore_direction}]').click();}})()""")
        try:
            wait_js(cdp, f"!document.querySelector('creative-document-editor')._actionInFlight && document.querySelector('creative-document-editor')._doc.rootLayerIds.indexOf({json.dumps(setup['inkLayerId'])})==={initial_ink_index}", timeout=30)
        except TimeoutError:
            diagnostic = cdp.evaluate("(()=>{const e=document.querySelector('creative-document-editor'); return {order:e._doc.rootLayerIds,selectedLayerId:e._selectedLayerId,actionInFlight:e._actionInFlight,actions:window.__auditActions,status:e.querySelector('[data-role=status]')?.innerText};})()")
            receipt["checks"]["layer_reorder_restore_diagnostic"] = {"passed": False, "evidence": diagnostic}
            raise

        cdp.evaluate(f"document.querySelector('creative-document-editor').querySelector(`[data-layer-id=\"{setup['depthLayerId']}\"] [data-action=layer-eye]`).click()")
        wait_js(cdp, f"!document.querySelector('creative-document-editor')._actionInFlight && document.querySelector('creative-document-editor')._doc.layers.find(x=>x.id==={json.dumps(setup['depthLayerId'])}).visible===false", timeout=30)
        cdp.evaluate(f"document.querySelector('creative-document-editor').querySelector(`[data-layer-id=\"{setup['depthLayerId']}\"] [data-action=layer-eye]`).click()")
        wait_js(cdp, f"!document.querySelector('creative-document-editor')._actionInFlight && document.querySelector('creative-document-editor')._doc.layers.find(x=>x.id==={json.dumps(setup['depthLayerId'])}).visible===true", timeout=30)
        cdp.evaluate(f"document.querySelector('creative-document-editor').querySelector(`[data-layer-id=\"{setup['depthLayerId']}\"] [data-action=layer-lock]`).click()")
        wait_js(cdp, f"!document.querySelector('creative-document-editor')._actionInFlight && document.querySelector('creative-document-editor')._doc.layers.find(x=>x.id==={json.dumps(setup['depthLayerId'])}).locked===true", timeout=30)
        cdp.evaluate(f"document.querySelector('creative-document-editor').querySelector(`[data-layer-id=\"{setup['depthLayerId']}\"] [data-action=layer-lock]`).click()")
        wait_js(cdp, f"!document.querySelector('creative-document-editor')._actionInFlight && document.querySelector('creative-document-editor')._doc.layers.find(x=>x.id==={json.dumps(setup['depthLayerId'])}).locked===false", timeout=30)
        layer_controls = cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'); const layer=e._doc.layers.find(x=>x.id==={json.dumps(setup['depthLayerId'])}); return {{order:e._doc.rootLayerIds.slice(),original:{json.dumps(initial_order)},reordered:{json.dumps(reordered)},visible:layer.visible,locked:layer.locked,revision:e._doc.revision}};}})()""")
        check(receipt, "layer_reorder_visibility_and_lock_controls", layer_controls["reordered"].index(setup["inkLayerId"]) != initial_ink_index and
              layer_controls["order"] == layer_controls["original"] and layer_controls["visible"] and
              not layer_controls["locked"], layer_controls)

        # The layer toolbar's add action supplies a bounded undo/redo history entry.
        before_undo_layers = cdp.evaluate("document.querySelector('creative-document-editor')._doc.rootLayerIds.length")
        cdp.evaluate("(()=>{const e=document.querySelector('creative-document-editor'); window.__auditActions=[]; e.querySelector('[data-action=add-vector]').click();})()")
        wait_js(cdp, f"(()=>{{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.rootLayerIds.length==={before_undo_layers + 1};}})()", timeout=30)
        cdp.evaluate("document.querySelector('creative-document-editor').querySelector('[data-action=undo]').click()")
        wait_js(cdp, f"!document.querySelector('creative-document-editor')._actionInFlight && document.querySelector('creative-document-editor')._doc.rootLayerIds.length==={before_undo_layers}", timeout=30)
        cdp.evaluate("document.querySelector('creative-document-editor').querySelector('[data-action=redo]').click()")
        wait_js(cdp, f"!document.querySelector('creative-document-editor')._actionInFlight && document.querySelector('creative-document-editor')._doc.rootLayerIds.length==={before_undo_layers + 1}", timeout=30)
        undo_redo = cdp.evaluate("({actions:window.__auditActions.slice(-3).map(x=>x.actionType),revision:document.querySelector('creative-document-editor')._doc.revision,layers:document.querySelector('creative-document-editor')._doc.rootLayerIds.length})")
        check(receipt, "undo_redo_controls_restore_layer_snapshot", undo_redo["actions"] == ["add_layer", "undo", "redo"] and
              undo_redo["layers"] == before_undo_layers + 1, {"beforeLayers": before_undo_layers, **undo_redo})

        mixed = cdp.evaluate("""(async()=>{
          const e=document.querySelector('creative-document-editor');
          const canvas=document.createElement('canvas'); canvas.width=1200; canvas.height=900;
          const ctx=canvas.getContext('2d'), pixels=ctx.createImageData(canvas.width,canvas.height); let state=0x13579bdf;
          for(let i=0;i<pixels.data.length;i+=4){state=(Math.imul(state,1664525)+1013904223)>>>0; const value=state>>>24;
            pixels.data[i]=value; pixels.data[i+1]=value; pixels.data[i+2]=value; pixels.data[i+3]=255;}
          ctx.putImageData(pixels,0,0);
          const maskBlob=await new Promise(resolve=>canvas.toBlob(resolve,'image/png'));
          const slot=document.getElementById('inpaint_mask_canvas'), input=slot?.querySelector('.nex-slot__input');
          if(!slot || !input) throw new Error('missing Inpaint BB Mask slot');
          const originalHandle=slot.handleFile; let resolveDone;
          const uploadDone=new Promise(resolve=>{resolveDone=resolve;});
          slot.handleFile=async function(file){try{await originalHandle.call(this,file);}finally{this.handleFile=originalHandle;resolveDone();}};
          const net={active:0,maxActive:0,events:[]}; window.__auditMixedNet=net;
          const priorFetch=window.fetch.bind(window);
          window.fetch=async(inputArg,init={})=>{
            const url=new URL(typeof inputArg==='string'?inputArg:inputArg.url,location.href), path=url.pathname;
            const kind=path==='/image_api/upload'?'image_slot_upload':
              (path.includes('/creative_document_api/documents/') && path.includes('/assets/') && !path.endsWith('/assets/import')?'editor_asset_load':'');
            if(!kind) return priorFetch(inputArg,init);
            const event={kind,start:performance.now(),activeAtStart:net.active}; net.events.push(event); net.active++;
            net.maxActive=Math.max(net.maxActive,net.active);
            try{const response=await priorFetch(inputArg,init); event.status=response.status; return response;}
            catch(error){event.status=0;event.error=String(error);throw error;}
            finally{event.end=performance.now();net.active--;}
          };
          const rasterLayer=e._doc.layers.find(layer=>layer.kind==='raster');
          const rasterObject=rasterLayer&&e._doc.objects.find(object=>object.layerId===rasterLayer.id);
          const asset=rasterObject&&e._doc.assets.find(item=>item.id===rasterObject.assetId);
          if(!asset) throw new Error('large raster asset missing from editor scene');
          e._assetCache.delete(asset.contentHash); e._scheduler.peakActive=0;
          const transfer=new DataTransfer(); transfer.items.add(new File([maskBlob],'audit-inpaint-mask.png',{type:'image/png'}));
          input.files=transfer.files; input.dispatchEvent(new Event('change',{bubbles:true}));
          const loadPromise=e._loadImageAsset({...asset,url:asset.url+(asset.url.includes('?')?'&':'?')+'mixedLoad='+Date.now()},e._controller.signal);
          const loaded=await Promise.all([uploadDone,loadPromise]);
          const image=slot.querySelector('.nex-slot__img');
          await new Promise(resolve=>{if(image.complete&&image.naturalWidth)resolve();else image.addEventListener('load',resolve,{once:true});});
          const uploadEvent=net.events.find(item=>item.kind==='image_slot_upload');
          const loadEvent=net.events.find(item=>item.kind==='editor_asset_load');
          const overlapMs=uploadEvent&&loadEvent?Math.max(0,Math.min(uploadEvent.end,loadEvent.end)-Math.max(uploadEvent.start,loadEvent.start)):0;
          return {assetLoaded:!!loaded[1],assetDimensions:[loaded[1]?.naturalWidth||0,loaded[1]?.naturalHeight||0],
            mask:{ready:slot.dataset.uploading==='false'&&slot.dropZone.classList.contains('has-image')&&image.naturalWidth===1200,
              dimensions:[image.naturalWidth,image.naturalHeight],path:!!slot.getFieldValue('inpaint_mask_image_path'),
              workspace:!!slot.getFieldValue('inpaint_mask_workspace_id')},
            schedulerPeak:e._scheduler.peakActive,network:{maxConcurrent:net.maxActive,overlapMs,
              events:net.events.map(item=>({kind:item.kind,status:item.status,durationMs:Math.round(item.end-item.start)}))}};
        })()""")
        check(receipt, "inpaint_mask_upload_and_large_editor_asset_load_overlap",
              mixed["assetLoaded"] and mixed["assetDimensions"] == [2400, 1792] and
              mixed["mask"]["ready"] and mixed["mask"]["path"] and mixed["mask"]["workspace"] and
              mixed["schedulerPeak"] <= 3 and mixed["network"]["maxConcurrent"] >= 2 and mixed["network"]["overlapMs"] > 0 and
              all(item["status"] == 200 for item in mixed["network"]["events"]), mixed)
        cdp.evaluate("document.querySelector('creative-document-editor')._stageContainer.scrollIntoView({block:'center',inline:'nearest'})")
        wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'),p=e?.closest('[role=tabpanel]'),r=e?._stageContainer?.getBoundingClientRect(); const t=[...document.querySelectorAll('[role=tab]')].find(x=>p&&x.getAttribute('aria-controls')===p.id); return !!r && r.width>0 && r.height>0 && r.bottom>0 && r.top<window.innerHeight && t?.getAttribute('aria-selected')==='true';})()")
        post_slot_stage = cdp.evaluate("(()=>{const e=document.querySelector('creative-document-editor'),p=e.closest('[role=tabpanel]'),r=e._stageContainer.getBoundingClientRect(); const t=[...document.querySelectorAll('[role=tab]')].find(x=>p&&x.getAttribute('aria-controls')===p.id); return {tabSelected:t?.getAttribute('aria-selected')==='true',panelHidden:p.getAttribute('aria-hidden'),stage:{x:r.x,y:r.y,width:r.width,height:r.height,viewportHeight:window.innerHeight,konva:[e._stage.width(),e._stage.height()]}};})()")
        check(receipt, "creative_document_stage_visible_before_pointer_gestures", post_slot_stage["tabSelected"] and
              post_slot_stage["panelHidden"] != "true" and post_slot_stage["stage"]["width"] > 0 and
              post_slot_stage["stage"]["height"] > 0 and all(value > 0 for value in post_slot_stage["stage"]["konva"]), post_slot_stage)

        cdp.evaluate("document.querySelector('creative-document-editor').querySelector('[data-tool=select]').click()")
        cdp.evaluate("(()=>{const e=document.querySelector('creative-document-editor'); window.__auditMiddle=[]; e._stage.on('pointerdown.audit-middle',event=>window.__auditMiddle.push({button:event.evt.button,buttons:event.evt.buttons,tool:e._tool,actionInFlight:e._actionInFlight})); e._stage.on('dragstart.audit-middle dragmove.audit-middle dragend.audit-middle',event=>window.__auditMiddle.push({type:event.type,position:e._stage.position()}));})()")
        middle_start = visible_stage_point(0, 18)
        middle_start["x"] = cdp.evaluate("(()=>{const r=document.querySelector('creative-document-editor')._stageContainer.getBoundingClientRect(); return r.left+r.width/2;})()")
        drag(cdp, middle_start, {"x": middle_start["x"] + 26, "y": middle_start["y"] + 17}, button="middle")
        middle = cdp.evaluate("(()=>{const e=document.querySelector('creative-document-editor'); const p=e._stage.position(),r=e._stageContainer.getBoundingClientRect(); return {pan:[e._view.panX,e._view.panY],stage:[p.x,p.y],draggable:e._stage.draggable(),active:e._panPointerActive,tool:e._tool,container:{x:r.x,y:r.y,width:r.width,height:r.height},events:window.__auditMiddle};})()")
        check(receipt, "middle_button_pan_exits_cleanly", middle["pan"] != [0, 0] and middle["stage"] == [0, 0] and
              not middle["draggable"] and not middle["active"], middle)

        # One real pointer drag must create one mutation and one revision.
        start_revision = cdp.evaluate("document.querySelector('creative-document-editor')._doc.revision")
        cdp.evaluate("window.__auditActions=[]")
        point = cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'); const r=e._activeNodeRecords.get({json.dumps(setup['firstObjectId'])}); const s=e._stageContainer.getBoundingClientRect(); const b=r.node.findOne('Rect').getClientRect({{relativeTo:e._stage}}); return {{x:s.left+b.x+b.width*0.3,y:s.top+b.y+b.height*0.3,rect:{{x:b.x,y:b.y,width:b.width,height:b.height}},wrapperListening:r.node.listening(),recordLocked:r.locked,layer:e._doc.layers.find(l=>l.id===r.layerId)}};}})()""")
        drag(cdp, point, {"x": point["x"] + 8, "y": point["y"] + 6})
        try:
            wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && window.__auditActions.length===1;})()", timeout=8)
        except TimeoutError:
            diagnosis = cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'),r=e._activeNodeRecords.get({json.dumps(setup['firstObjectId'])}),b=r.node.getClientRect({{relativeTo:e._stage}}),s=e._stageContainer.getBoundingClientRect(),hit=e._stage.getIntersection({{x:b.x+b.width*0.3+8,y:b.y+b.height*0.3+6}}); return {{actions:window.__auditActions,actionInFlight:e._actionInFlight,status:e.querySelector('[data-role="status"]')?.innerText,revision:e._doc.revision,selected:[...e._selectedObjectIds],stageSize:[e._stage.width(),e._stage.height()],container:{{x:s.x,y:s.y,width:s.width,height:s.height}},rect:{{x:b.x,y:b.y,width:b.width,height:b.height}},nodePosition:r.node.position(),draggable:r.node.draggable(),listening:r.node.listening(),recordLocked:r.locked,layer:e._doc.layers.find(l=>l.id===r.layerId),pointerTarget:hit?.name(),pointerTargetClass:hit?.getClassName()}};}})()""")
            receipt["checks"]["single_drag_diagnostic"] = {"passed": False, "evidence": diagnosis}
            raise
        single_drag = cdp.evaluate("({revision:document.querySelector('creative-document-editor')._doc.revision,actions:window.__auditActions.slice()})")
        check(receipt, "single_object_drag_is_one_transaction", single_drag["revision"] == start_revision + 1 and len(single_drag["actions"]) == 1,
              {"previousRevision": start_revision, **single_drag})

        # Two selected wrappers move in document space from one pointer gesture.
        cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'); e._selectedObjectIds=new Set([{json.dumps(setup['firstObjectId'])},{json.dumps(setup['secondObjectId'])}]); e._updateTransformer(); e._refreshToolButtons(); window.__auditActions=[];}})()""")
        multi_start = cdp.evaluate("document.querySelector('creative-document-editor')._doc.revision")
        point = cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'); const r=e._activeNodeRecords.get({json.dumps(setup['firstObjectId'])}); const s=e._stageContainer.getBoundingClientRect(); const b=r.node.findOne('Rect').getClientRect({{relativeTo:e._stage}}); return {{x:s.left+b.x+b.width/2,y:s.top+b.y+b.height/2}};}})()""")
        drag(cdp, point, {"x": point["x"] + 8, "y": point["y"] + 6})
        wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && window.__auditActions.length===1;})()", timeout=30)
        multi_move = cdp.evaluate("({revision:document.querySelector('creative-document-editor')._doc.revision,actions:window.__auditActions.slice()})")
        moved_transforms = cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'); return [{json.dumps(setup['firstObjectId'])},{json.dumps(setup['secondObjectId'])}].map(id=>e._doc.objects.find(o=>o.id===id).transform.slice(2,6));}})()""")
        check(receipt, "multi_object_move_is_single_atomic_batch", multi_move["revision"] == multi_start + 1 and
              len(multi_move["actions"]) == 1 and multi_move["actions"][0]["actionType"] == "batch" and
              multi_move["actions"][0]["targetCount"] == 2 and moved_transforms[0] != [0, 0, 0, 0] and moved_transforms[1] != [0, 0, 0, 0],
              {"previousRevision": multi_start, **multi_move, "objectTranslations": moved_transforms})

        # A real Transformer anchor drag exercises the exclusive transformend owner.
        cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'); e._selectedObjectIds=new Set([{json.dumps(setup['firstObjectId'])}]); e._updateTransformer(); window.__auditActions=[];}})()""")
        scale_start = cdp.evaluate("document.querySelector('creative-document-editor')._doc.revision")
        anchor = cdp.evaluate("""(()=>{const e=document.querySelector('creative-document-editor'); const a=e._transformer.findOne('.bottom-right'); if(!a) return null; const p=a.absolutePosition(),r=e._stageContainer.getBoundingClientRect(); return {x:r.left+p.x,y:r.top+p.y};})()""")
        if anchor:
            drag(cdp, anchor, {"x": anchor["x"] + 24, "y": anchor["y"] + 18})
            wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && window.__auditActions.length===1;})()", timeout=30)
            scale_result = cdp.evaluate("({revision:document.querySelector('creative-document-editor')._doc.revision,actions:window.__auditActions.slice()})")
            check(receipt, "transformer_scale_is_one_transaction", scale_result["revision"] == scale_start + 1 and len(scale_result["actions"]) == 1,
                  {"previousRevision": scale_start, **scale_result})
        else:
            check(receipt, "transformer_scale_is_one_transaction", False, "Konva Transformer bottom-right anchor was not present")

        for caption, action in (("rotate", "rotate"), ("flip", "flip")):
            cdp.evaluate(f"window.__auditActions=[]; document.querySelector('creative-document-editor')._selectedObjectIds=new Set([{json.dumps(setup['firstObjectId'])},{json.dumps(setup['secondObjectId'])}]); document.querySelector('creative-document-editor')._updateTransformer()")
            before = cdp.evaluate("document.querySelector('creative-document-editor')._doc.revision")
            cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'); const b=[...e.querySelectorAll('[data-action]')].find(x=>x.dataset.action==={json.dumps(action)}); b.click();}})()""")
            wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && window.__auditActions.length===1;})()", timeout=30)
            result = cdp.evaluate("({revision:document.querySelector('creative-document-editor')._doc.revision,actions:window.__auditActions.slice()})")
            check(receipt, f"multi_object_{caption}_is_one_transaction", result["revision"] == before + 1 and
                  len(result["actions"]) == 1 and result["actions"][0]["targetCount"] == 2,
                  {"previousRevision": before, **result})

        # An injected conflict must restore the authoritative node geometry.
        before_failure = cdp.evaluate("document.querySelector('creative-document-editor')._doc.revision")
        cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'); const n=e._activeNodeRecords.get({json.dumps(setup['firstObjectId'])}).node; n.x(n.x()+30); window.__auditActions=[]; window.__auditFailNextAction=true; e._scheduleTransformCommit({json.dumps(setup['firstObjectId'])});}})()""")
        wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && window.__auditActions.length===1 && e._doc.revision>=0;})()", timeout=30)
        failure_restore = cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'); const o=e._doc.objects.find(x=>x.id==={json.dumps(setup['firstObjectId'])}); const n=e._activeNodeRecords.get(o.id).node; return {{revision:e._doc.revision,position:[n.x(),n.y()],matrix:o.transform,action:window.__auditActions[0],status:e.querySelector('[data-role="status"]').textContent}};}})()""")
        check(receipt, "stale_transform_refusal_restores_authority_without_revision", failure_restore["revision"] == before_failure and
              failure_restore["action"].get("injectedFailure") is True and "STALE_DOCUMENT_REVISION" in failure_restore["status"],
              {"previousRevision": before_failure, **failure_restore})

        # Fit/zoom/pan are measured against actual Konva Stage and group transforms.
        cdp.evaluate("document.querySelector('creative-document-editor')._fitToView()")
        camera_before = cdp.evaluate("document.querySelector('creative-document-editor')._doc.revision")
        stage_center = visible_stage_point()
        cdp.evaluate("document.querySelector('creative-document-editor').querySelector('[data-tool=pan]').click()")
        drag(cdp, stage_center, {"x": stage_center["x"] + 74, "y": stage_center["y"] - 39})
        pan_tool = cdp.evaluate("(()=>{const e=document.querySelector('creative-document-editor'); const p=e._stage.position(),g=e._mainGroup.position(); return {pan:[e._view.panX,e._view.panY],stage:[p.x,p.y],group:[g.x,g.y],offset:[e._view.offsetX,e._view.offsetY],draggable:e._stage.draggable(),revision:e._doc.revision};})()")
        check(receipt, "stage_pan_normalizes_to_content_transform", pan_tool["pan"] != [0, 0] and pan_tool["stage"] == [0, 0] and
              abs(pan_tool["group"][0] - pan_tool["offset"][0] - pan_tool["pan"][0]) < 0.01 and
              abs(pan_tool["group"][1] - pan_tool["offset"][1] - pan_tool["pan"][1]) < 0.01 and not pan_tool["draggable"] and
              pan_tool["revision"] == camera_before, {"cameraRevision": camera_before, **pan_tool})

        cdp.evaluate("document.querySelector('creative-document-editor').querySelector('[data-tool=select]').click(); document.querySelector('.ncd-stage-wrap').focus()")
        space_start = visible_stage_point()
        cdp.command("Input.dispatchKeyEvent", {"type": "keyDown", "key": " ", "code": "Space", "windowsVirtualKeyCode": 32, "nativeVirtualKeyCode": 32})
        mouse(cdp, "mousePressed", space_start["x"], space_start["y"], buttons=1)
        mouse(cdp, "mouseMoved", space_start["x"] + 21, space_start["y"] + 13, buttons=1)
        mouse(cdp, "mouseReleased", space_start["x"] + 21, space_start["y"] + 13)
        cdp.command("Input.dispatchKeyEvent", {"type": "keyUp", "key": " ", "code": "Space", "windowsVirtualKeyCode": 32, "nativeVirtualKeyCode": 32})
        time.sleep(0.15)
        space = cdp.evaluate("(()=>{const e=document.querySelector('creative-document-editor'); const p=e._stage.position(); return {pan:[e._view.panX,e._view.panY],stage:[p.x,p.y],draggable:e._stage.draggable(),space:e._spacePanning,revision:e._doc.revision};})()")
        check(receipt, "space_pan_exits_cleanly", space["stage"] == [0, 0] and not space["draggable"] and not space["space"] and
              space["revision"] == camera_before, {"cameraRevision": camera_before, **space})

        cdp.command("Emulation.setDeviceMetricsOverride", {"width": 1365, "height": 960, "deviceScaleFactor": 2.5, "mobile": False})
        wait_js(cdp, "window.devicePixelRatio===2.5", timeout=10)
        coordinate = cdp.evaluate("(()=>{const e=document.querySelector('creative-document-editor'); e._setZoom(1.7); const p={x:913.25,y:622.75}; const s=e._view.fitScale*e._view.zoom; const x=e._view.offsetX+e._view.panX+p.x*s; const y=e._view.offsetY+e._view.panY+p.y*s; const q=e._previewToDocument(x,y); return {dpr:window.devicePixelRatio,input:[p.x,p.y],roundTrip:[q.x,q.y],stage:[e._stage.x(),e._stage.y()],revision:e._doc.revision};})()")
        check(receipt, "dpr_zoom_resize_coordinate_round_trip", coordinate["dpr"] == 2.5 and
              abs(coordinate["roundTrip"][0] - coordinate["input"][0]) < 1e-6 and
              abs(coordinate["roundTrip"][1] - coordinate["input"][1]) < 1e-6 and coordinate["stage"] == [0, 0] and
              coordinate["revision"] == camera_before, coordinate)
        cdp.command("Emulation.setDeviceMetricsOverride", {"width": 1440, "height": 1000, "deviceScaleFactor": 1, "mobile": False})
        time.sleep(0.25)
        cdp.evaluate("document.querySelector('creative-document-editor')._fitToView()")

        # Selection seeds/refinement use the human controls and document-space pointer path.
        selection_source = setup["rasterLayerId"]
        cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'); e._selectedLayerId={json.dumps(selection_source)}; e._selectedObjectIds.clear(); e._renderProperties(); e._renderLayerList(); const hint=e.querySelector('[data-action="selection-hint"]'); hint.value='painted subject'; e.querySelector('[data-selection-mode="box"]').click();}})()""")
        selection_revision = cdp.evaluate("document.querySelector('creative-document-editor')._doc.revision")
        drag(cdp, document_point(260, 260), document_point(620, 640))
        wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.revision>" + str(selection_revision) + " && e._doc.selections.length===1;})()", timeout=30)
        box_ui = cdp.evaluate("""(()=>{const e=document.querySelector('creative-document-editor'); return {selection:e._doc.selections[0],text:e.querySelector('[data-role="selections"]').innerText,maskOptions:[...e.querySelector('[data-action="selection-mask"]').options].map(x=>x.textContent),buttons:[...e.querySelectorAll('[data-selection-mode]')].map(x=>x.dataset.selectionMode)};})()""")
        check(receipt, "persistent_selection_box_metadata_and_allowlisted_controls", "painted subject" == box_ui["selection"]["semanticHint"] and
              box_ui["selection"]["bounds"] and box_ui["selection"]["sourceContentDigest"] and
              box_ui["selection"]["sourceId"] == selection_source and all(mode in box_ui["buttons"] for mode in ("box", "polygon", "paint", "point")) and
              "context" not in " ".join(box_ui["maskOptions"]),
              {"bounds": box_ui["selection"]["bounds"], "selectionRevision": box_ui["selection"]["selectionRevision"],
               "metadataVisible": all(value in box_ui["text"] for value in (selection_source, box_ui["selection"]["sourceContentDigest"])),
               "seedModes": box_ui["buttons"], "maskOptions": box_ui["maskOptions"]})

        # Paint seed and manual paint refinement are driven by real pointer strokes.
        cdp.evaluate("document.querySelector('creative-document-editor').querySelector('[data-selection-mode=paint]').click()")
        paint_before = cdp.evaluate("document.querySelector('creative-document-editor')._doc.revision")
        drag(cdp, document_point(700, 520), document_point(860, 640))
        wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.revision>" + str(paint_before) + " && e._doc.selections.length===2;})()", timeout=30)
        paint_selection = cdp.evaluate("document.querySelector('creative-document-editor')._doc.selections[1]")
        check(receipt, "paint_seed_selection", paint_selection["seedGeometry"][0]["kind"] == "paint" and paint_selection["bounds"] is not None,
              {"selectionRevision": paint_selection["selectionRevision"], "bounds": paint_selection["bounds"]})

        cdp.evaluate("""(()=>{const e=document.querySelector('creative-document-editor'); const op=e.querySelector('[data-action=refine-operation]'); op.value='subtract'; op.dispatchEvent(new Event('change',{bubbles:true})); e.querySelector('[data-action=draw-paint-refinement]').click();})()""")
        refine_before = cdp.evaluate("document.querySelector('creative-document-editor')._doc.revision")
        drag(cdp, document_point(740, 580), document_point(780, 580))
        wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.revision>" + str(refine_before) + " && e._doc.selections[1].selectionRevision===2;})()", timeout=30)
        refine_state = cdp.evaluate("""(()=>{const e=document.querySelector('creative-document-editor'); const s=e._doc.selections[1]; return {selectionRevision:s.selectionRevision,history:s.refinementHistory,text:e.querySelector('[data-role="selections"]').innerText};})()""")
        check(receipt, "manual_paint_refinement_and_provenance", refine_state["selectionRevision"] == 2 and
              refine_state["history"][-1]["details"]["operation"] == "subtract" and "subtract" in refine_state["text"], refine_state)

        cdp.evaluate("""(()=>{const e=document.querySelector('creative-document-editor'); e.querySelector('[data-action=selection-hint]').value='alpha channel'; e.querySelector('[data-action=selection-alpha]').click();})()""")
        wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.selections.length===3;})()", timeout=30)
        alpha = cdp.evaluate("document.querySelector('creative-document-editor')._doc.selections[2]")
        check(receipt, "source_alpha_seed", alpha["semanticHint"] == "alpha channel" and alpha["seedGeometry"][0]["kind"] == "alpha",
              {"semanticHint": alpha["semanticHint"], "seedKind": alpha["seedGeometry"][0]["kind"]})

        cdp.evaluate("""(()=>{const e=document.querySelector('creative-document-editor'); const select=e.querySelector('[data-action=selection-mask]'); const option=[...select.options].find(x=>x.value && x.textContent.includes('editing-selection')); if(!option) throw new Error('allowlisted selection mask option is missing'); select.value=option.value; e.querySelector('[data-action=selection-mask-create]').click();})()""")
        wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.selections.length===4;})()", timeout=30)
        mask_seed = cdp.evaluate("document.querySelector('creative-document-editor')._doc.selections[3].seedGeometry[0]")
        check(receipt, "allowlisted_existing_mask_seed", mask_seed["kind"] == "mask" and mask_seed.get("maskId"), mask_seed)

        # Rebase first preserves the mask for transforms, then derives a new one after source content changes.
        source_transform_before = cdp.evaluate("document.querySelector('creative-document-editor')._doc.revision")
        cdp.evaluate(f"""document.querySelector('creative-document-editor')._sendAction('transform',{{targetIds:[{json.dumps(selection_source)}],transform:[1,0,4,0,1,3,0,0,1]}})""")
        wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.revision===" + str(source_transform_before + 1) + " && e._doc.selections.every(s=>s.state==='needs-rebase');})()", timeout=30)
        cdp.evaluate("document.querySelector('creative-document-editor').querySelector('[data-action=rebase-selection]').click()")
        wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.selections[3].state==='current';})()", timeout=30)
        geometric = cdp.evaluate("(()=>{const e=document.querySelector('creative-document-editor'); const s=e._doc.selections[3]; return {entry:s.rebaseHistory.at(-1),text:e.querySelector('[data-role=selections]').innerText};})()")
        check(receipt, "geometry_only_rebase_preserves_mask", geometric["entry"]["geometryOnly"] is True and
              "geometry-only rebase; mask preserved" in geometric["text"], geometric)

        cdp.evaluate(f"""document.querySelector('creative-document-editor')._sendAction('set_layer_opacity',{{layerId:{json.dumps(selection_source)},opacity:0.85}})""")
        try:
            wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.selections[3].state==='stale-source';})()", timeout=12)
        except TimeoutError:
            diagnosis = cdp.evaluate("(()=>{const e=document.querySelector('creative-document-editor'); return {actionInFlight:e._actionInFlight,revision:e._doc.revision,selection:e._doc.selections.map(s=>({id:s.id,state:s.state,staleReason:s.staleReason,sourceRevision:s.sourceRevision})),opacity:e._doc.layers.find(l=>l.id===e._selectedLayerId)?.opacity,status:e.querySelector('[data-role=status]')?.innerText,actions:window.__auditActions.slice()};})()")
            receipt["checks"]["content_change_diagnostic"] = {"passed": False, "evidence": diagnosis}
            raise
        cdp.evaluate("""(()=>{const e=document.querySelector('creative-document-editor'); const t=e.querySelector('[data-action=mask-seed-target]'); t.value='rebase'; t.dispatchEvent(new Event('change',{bubbles:true})); e.querySelector('[data-action=draw-mask-seed]').click();})()""")
        for xy in ((250, 250), (720, 260), (690, 720)):
            q = document_point(*xy)
            mouse(cdp, "mouseMoved", q["x"], q["y"])
            mouse(cdp, "mousePressed", q["x"], q["y"])
            mouse(cdp, "mouseReleased", q["x"], q["y"])
        cdp.evaluate("document.querySelector('creative-document-editor').querySelector('[data-action=finish-selection]').click()")
        cdp.evaluate("document.querySelector('creative-document-editor').querySelector('[data-action=rebase-selection]').click()")
        wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.selections[3].state==='current' && e._doc.selections[3].rebaseHistory.length===2;})()", timeout=45)
        content_rebase = cdp.evaluate("(()=>{const e=document.querySelector('creative-document-editor'); const s=e._doc.selections[3]; return {entry:s.rebaseHistory.at(-1),text:e.querySelector('[data-role=selections]').innerText};})()")
        check(receipt, "content_rebase_rederives_mask_and_shows_provenance", content_rebase["entry"]["geometryOnly"] is False and
              "content rebase; mask rederived" in content_rebase["text"], content_rebase)

        # Create, activate, replace, supersede, and retire semantic guides with the guide controls.
        cdp.evaluate("document.querySelector('creative-document-editor')._stageContainer.scrollIntoView({block:'center',inline:'nearest'})")
        cdp.evaluate("(()=>{const e=document.querySelector('creative-document-editor'); const tool=e.querySelector('[data-tool=select]'); if(!tool) throw new Error('select tool control is missing'); tool.click();})()")
        drawn_object = cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'); const record=e._activeNodeRecords.get({json.dumps(drawn['object']['id'])}); const box=record.node.getClientRect({{relativeTo:e._stage}}); const r=e._stageContainer.getBoundingClientRect(); return {{x:r.left+box.x+box.width/2,y:r.top+box.y+box.height/2}};}})()""")
        mouse(cdp, "mouseMoved", drawn_object["x"], drawn_object["y"])
        mouse(cdp, "mousePressed", drawn_object["x"], drawn_object["y"], buttons=1)
        mouse(cdp, "mouseReleased", drawn_object["x"], drawn_object["y"])
        wait_js(cdp, f"[...document.querySelector('creative-document-editor')._selectedObjectIds].includes({json.dumps(drawn['object']['id'])})")
        guide_create_control = cdp.evaluate("(()=>{const e=document.querySelector('creative-document-editor'),b=e.querySelector('[data-action=create-guide]'); return {present:!!b,disabled:b?.disabled,selection:[...e._selectedObjectIds],layerId:e._selectedLayerId,tool:e._tool};})()")
        check(receipt, "guide_creation_control_available_for_selected_object", guide_create_control["present"] and
              guide_create_control["disabled"] is False, guide_create_control)
        cdp.evaluate("(()=>{const e=document.querySelector('creative-document-editor'),b=e.querySelector('[data-action=create-guide]'); if(!b) throw new Error('New proposed guide control is missing'); b.click();})()")
        wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.guides.length===1 && e._doc.guides[0].lifecycle==='proposed';})()", timeout=30)
        old_guide = cdp.evaluate("document.querySelector('creative-document-editor')._doc.guides[0].guideId")
        proposed_controls = cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'),id={json.dumps(old_guide)}; return {{activate:!!e.querySelector('[data-action=guide-activate][data-guide-id=\"'+id+'\"]'),consume:!!e.querySelector('[data-action=guide-consume][data-guide-id=\"'+id+'\"]'),lifecycle:e._doc.guides.find(g=>g.guideId===id).lifecycle}};}})()""")
        check(receipt, "proposed_guide_does_not_offer_consumed_transition", proposed_controls["lifecycle"] == "proposed" and
              proposed_controls["activate"] is True and proposed_controls["consume"] is False, proposed_controls)
        cdp.evaluate(f"document.querySelector('creative-document-editor').querySelector(`[data-action=guide-activate][data-guide-id=\"{old_guide}\"]`).click()")
        wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.guides[0].lifecycle==='active';})()", timeout=30)
        cdp.evaluate("document.querySelector('creative-document-editor').querySelector('[data-action=create-guide]').click()")
        wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.guides.length===2;})()", timeout=30)
        replacement_id = cdp.evaluate("document.querySelector('creative-document-editor')._doc.guides.find(item=>item.guideId!==" + json.dumps(old_guide) + ").guideId")
        cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'); const row=[...e.querySelectorAll('[data-role=guides] .ncd-row')].find(item=>item.querySelector(`[data-guide-id=\"{replacement_id}\"]`)); if(!row) throw new Error('replacement guide row is missing'); row.querySelector('span').click(); window.prompt=(message,initial)=>message.startsWith('ID of the existing guide')?{json.dumps(old_guide)}:initial; e.querySelector('[data-action=replace-guide]').click();}})()""")
        wait_js(cdp, f"(()=>{{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.guides.find(x=>x.guideId==={json.dumps(old_guide)}).lifecycle==='replacement-pending';}})()", timeout=30)
        cdp.evaluate(f"document.querySelector('creative-document-editor').querySelector(`[data-action=guide-supersede][data-guide-id=\"{old_guide}\"]`).click()")
        wait_js(cdp, f"(()=>{{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.guides.find(x=>x.guideId==={json.dumps(old_guide)}).lifecycle==='superseded';}})()", timeout=30)
        cdp.evaluate(f"document.querySelector('creative-document-editor').querySelector(`[data-action=guide-retire][data-guide-id=\"{old_guide}\"]`).click()")
        wait_js(cdp, f"(()=>{{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.guides.find(x=>x.guideId==={json.dumps(old_guide)}).lifecycle==='safe-to-remove';}})()", timeout=30)
        guide_lifecycle = cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'); const old=e._doc.guides.find(x=>x.guideId==={json.dumps(old_guide)}); const replacement=e._doc.guides.find(x=>x.guideId==={json.dumps(replacement_id)}); return {{old:old.lifecycle,replacement:replacement.lifecycle,replacementId:old.replacementId,supersedesId:replacement.supersedesId,action:window.__auditActions.at(-1)}};}})()""")
        check(receipt, "guide_propose_activate_replace_supersede_retire_controls", guide_lifecycle["old"] == "safe-to-remove" and
              guide_lifecycle["replacement"] == "proposed" and guide_lifecycle["replacementId"] == replacement_id and
              guide_lifecycle["supersedesId"] == old_guide and guide_lifecycle["action"]["actionType"] == "transition_guide", guide_lifecycle)

        replacement_activate = cdp.evaluate(f"""(()=>{{const button=document.querySelector('creative-document-editor').querySelector('[data-action=guide-activate][data-guide-id=\"{replacement_id}\"]'); return !!button && !button.disabled;}})()""")
        check(receipt, "replacement_proposal_can_be_activated", replacement_activate is True, {"replacementId": replacement_id})
        cdp.evaluate(f"""document.querySelector('creative-document-editor').querySelector('[data-action=guide-activate][data-guide-id=\"{replacement_id}\"]').click()""")
        wait_js(cdp, f"(()=>{{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.guides.find(g=>g.guideId==={json.dumps(replacement_id)}).lifecycle==='active';}})()", timeout=30)
        active_controls = cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'),id={json.dumps(replacement_id)},guide=e._doc.guides.find(g=>g.guideId===id); return {{lifecycle:guide.lifecycle,stateRevision:guide.stateRevision,consume:!!e.querySelector('[data-action=guide-consume][data-guide-id=\"'+id+'\"]')}};}})()""")
        check(receipt, "active_guide_offers_consumed_transition", active_controls["lifecycle"] == "active" and
              active_controls["consume"] is True, active_controls)
        consume_start_revision = cdp.evaluate("document.querySelector('creative-document-editor')._doc.revision")
        cdp.evaluate(f"""(()=>{{window.__auditActions=[]; document.querySelector('creative-document-editor').querySelector('[data-action=guide-consume][data-guide-id=\"{replacement_id}\"]').click();}})()""")
        wait_js(cdp, f"(()=>{{const e=document.querySelector('creative-document-editor'),g=e._doc.guides.find(item=>item.guideId==={json.dumps(replacement_id)}); return !e._actionInFlight && g.lifecycle==='consumed' && e._doc.revision==={consume_start_revision + 1};}})()", timeout=30)
        consumed_guide = cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'),g=e._doc.guides.find(item=>item.guideId==={json.dumps(replacement_id)}); const row=[...e.querySelectorAll('[data-role=guides] .ncd-row')].find(item=>item.querySelector('[data-guide-id=\"'+g.guideId+'\"]')); return {{documentId:e._doc.documentId,revision:e._doc.revision,lifecycle:g.lifecycle,stateRevision:g.stateRevision,rowText:row?.innerText||'',action:window.__auditActions.at(-1)}};}})()""")
        check(receipt, "active_guide_consumed_transition_updates_state_and_revision", consumed_guide["lifecycle"] == "consumed" and
              consumed_guide["revision"] == consume_start_revision + 1 and consumed_guide["stateRevision"] == consumed_guide["revision"] and
              "consumed" in consumed_guide["rowText"] and consumed_guide["action"]["actionType"] == "transition_guide", consumed_guide)
        cdp.evaluate("document.querySelector('creative-document-editor')._saveDocument()")
        wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return e._doc && e._doc.dirty===false && e._doc.committedRevision===e._doc.revision;})()", timeout=45)
        cdp.evaluate("document.querySelector('creative-document-editor')._openDocument()")
        wait_js(cdp, f"(()=>{{const e=document.querySelector('creative-document-editor'),g=e._doc?.guides.find(item=>item.guideId==={json.dumps(replacement_id)}); return !!g && g.lifecycle==='consumed' && g.stateRevision==={consumed_guide['stateRevision']} && e._doc.revision==={consumed_guide['revision']} && !e._doc.dirty && e._doc.recoveryNotice;}})()", timeout=45)
        reopened_consumed = cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'),g=e._doc.guides.find(item=>item.guideId==={json.dumps(replacement_id)}); return {{documentId:e._doc.documentId,revision:e._doc.revision,lifecycle:g.lifecycle,stateRevision:g.stateRevision,dirty:e._doc.dirty}};}})()""")
        check(receipt, "consumed_guide_state_revision_survives_save_and_reopen", reopened_consumed["documentId"] == consumed_guide["documentId"] and
              reopened_consumed["revision"] == consumed_guide["revision"] and reopened_consumed["lifecycle"] == "consumed" and
              reopened_consumed["stateRevision"] == consumed_guide["stateRevision"] and reopened_consumed["dirty"] is False, reopened_consumed)

        # Use the context-box tool and relational controls to persist a visible reference.
        cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'); e.querySelector(`[data-layer-id=\"{setup['rasterLayerId']}\"]`).click(); e.querySelector('[data-tool=context-box]').click(); window.__auditActions=[];}})()""")
        drag(cdp, document_point(1580, 1080), document_point(1930, 1380))
        wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.masks.some(mask=>mask.purpose==='context');})()", timeout=30)
        context_created = cdp.evaluate("document.querySelector('creative-document-editor')._doc.masks.filter(mask=>mask.purpose==='context').at(-1).id")
        cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'); const select=e.querySelector('[data-action=context-mask]'); select.value={json.dumps(context_created)}; select.dispatchEvent(new Event('change',{{bubbles:true}})); const dilation=e.querySelector('[data-action=context-dilation]'); dilation.value='12'; dilation.dispatchEvent(new Event('input',{{bubbles:true}})); e.querySelector(`[data-layer-id=\"{setup['depthLayerId']}\"] input.ncd-context-ref`).click(); e.querySelector('[data-action=save-context]').click();}})()""")
        wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); const m=e._doc.masks.find(item=>item.purpose==='context' && item.lineage.dilationPx===12); return !e._actionInFlight && !!m && m.lineage.relationalReferences.length===1;})()", timeout=30)
        relational_context = cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'); const mask=e._doc.masks.find(item=>item.id==={json.dumps(context_created)}); return {{ownerId:mask.ownerId,lineage:mask.lineage,expectedReference:{json.dumps(setup['depthLayerId'])},action:window.__auditActions.at(-1)}};}})()""")
        check(receipt, "relational_context_controls_save_reference_and_edit_scope", relational_context["ownerId"] is None and
              relational_context["lineage"]["relationalReferences"] == [setup["depthLayerId"]] and
              relational_context["lineage"]["dilationPx"] == 12 and
              relational_context["lineage"]["editTargetIds"] == [setup["rasterLayerId"]] and
              relational_context["action"]["actionType"] == "set_relational_context", relational_context)

        # Duplicate a freshly created, current selection after pointer-driven UI checks.
        cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'); e.querySelector(`[data-layer-id=\"{selection_source}\"]`).click(); e.querySelector('[data-tool=select]').click(); const hint=e.querySelector('[data-action=selection-hint]'); hint.value='duplicate source'; e.querySelector('[data-selection-mode=box]').click();}})()""")
        duplicate_selection_revision = cdp.evaluate("document.querySelector('creative-document-editor')._doc.revision")
        drag(cdp, document_point(330, 370), document_point(570, 610))
        wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.revision>" + str(duplicate_selection_revision) + " && e._doc.selections.length===5;})()", timeout=30)
        duplicate_selection = cdp.evaluate("""(()=>{const e=document.querySelector('creative-document-editor'); const selection=e._doc.selections.at(-1);
          const button=[...e.querySelectorAll('[data-action=selection-activate]')].find(item=>item.dataset.selectionId===selection.id);
          if(!button) throw new Error('current selection Use control is missing'); button.click(); window.__auditActions=[];
          const duplicate=e.querySelector('[data-action=duplicate-selection]'),active=e._activeSelection(); duplicate.click();
          return {id:selection.id,revision:selection.selectionRevision,state:active?.state,activeId:e._activeSelectionId,buttonDisabled:duplicate.disabled};})()""")
        try:
            wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.layers.some(layer=>layer.name==='Selection copy');})()", timeout=90)
        except TimeoutError:
            diagnostic = cdp.evaluate("(()=>{const e=document.querySelector('creative-document-editor'); return {revision:e._doc.revision,actionInFlight:e._actionInFlight,actions:window.__auditActions,layers:e._doc.layers.map(layer=>({id:layer.id,name:layer.name})),selection:e._activeSelection(),status:e.querySelector('[data-role=status]')?.innerText};})()")
            receipt["checks"]["duplicate_selection_diagnostic"] = {"passed": False, "evidence": {"before": duplicate_selection, "after": diagnostic}}
            raise
        duplicate_result = cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'); const selection=e._doc.selections.find(item=>item.id==={json.dumps(duplicate_selection['id'])}); const layer=e._doc.layers.find(item=>item.name==='Selection copy'); return {{selectionRevision:selection.selectionRevision,derivedLayerIds:selection.derivedLayerIds,layerId:layer.id,objects:layer.objectIds.map(id=>e._doc.objects.find(item=>item.id===id)?.kind),action:window.__auditActions.at(-1)}};}})()""")
        check(receipt, "duplicate_selection_control_registers_derived_layer", duplicate_result["layerId"] in duplicate_result["derivedLayerIds"] and
              duplicate_result["action"]["actionType"] == "duplicate_selection_to_layer" and bool(duplicate_result["objects"]), duplicate_result)

        # Layer identity/status and named relationship projection are visible in the tree.
        cdp.evaluate(f"""document.querySelector('creative-document-editor')._sendAction('rename_layer',{{layerId:{json.dumps(selection_source)},name:'Base Artwork'}})""")
        wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && [...e.querySelectorAll('.ncd-layer-row')].some(x=>x.innerText.includes('Base Artwork'));})()", timeout=30)
        layer_ui = cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'); const l=e._doc.layers.find(x=>x.id==={json.dumps(selection_source)}); l.depthElement=true; l.completePlateId={json.dumps(selection_source)}; e._doc.interactionGroups=[{{groupId:'group-probe',memberLayerIds:[{json.dumps(selection_source)},{json.dumps(setup['inkLayerId'])},{json.dumps(setup['depthLayerId'])}],relationOrder:[{{layerId:{json.dumps(selection_source)},relation:'below'}},{{layerId:{json.dumps(setup['inkLayerId'])},relation:'subject'}},{{layerId:{json.dumps(setup['depthLayerId'])},relation:'above'}}],overlapNotes:'browser relation probe'}}]; e._doc.variants=[{{variantSetId:'variant-probe',semanticRole:'pose',memberIds:[{json.dumps(setup['inkLayerId'])},{json.dumps(selection_source)}],activeMemberId:{json.dumps(setup['inkLayerId'])},sharedAnchor:{{}}}}]; e._renderLayerList(); e._renderRelations(); return {{row:e.querySelector(`[data-layer-id="{selection_source}"]`).innerText,relations:e.querySelector('[data-role=relations]').innerText}};}})()""")
        check(receipt, "named_layer_state_interaction_and_variant_visibility", all(text in layer_ui["row"] for text in ("Base Artwork", "r", "untouched", "depth element", "complete plate")) and
              all(text in layer_ui["relations"] for text in ("Base Artwork", "Ink layer", "Depth Finish", "below", "subject", "above", "active", "inactive")), layer_ui)

        # Open a valid W02 project through the owner capability. Its server view
        # carries unsupported layer/object/shape/blend/clipping semantics.
        cdp.evaluate("document.querySelector('creative-document-editor')._saveDocument()")
        wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return e._doc && e._doc.dirty===false && e._doc.committedRevision===e._doc.revision;})()", timeout=45)
        owner_key = cdp.evaluate("document.querySelector('creative-document-editor')._ownerKey")
        renderer_fixture_id = create_renderer_issue_fixture(output_root, owner_key)
        fixture_view = cdp.evaluate(f"""(async()=>{{const e=document.querySelector('creative-document-editor'); const opened=await e._request(`/creative_document_api/documents/{renderer_fixture_id}/open`,{{method:'POST',body:{{discardUnsaved:true}}}}); e._doc=opened.view; e._selectedLayerId=e._doc.rootLayerIds[0]; await e._acceptView(e._doc); return {{documentId:e._doc.documentId,revision:e._doc.revision,issues:e._doc.rendererIssues.map(item=>item.code)}};}})()""")
        renderer_ui = cdp.evaluate("(()=>{const e=document.querySelector('creative-document-editor'); return {banner:e.querySelector('[data-role=fidelity-issues]')?.innerText||'',codes:[...e.querySelectorAll('[data-role=fidelity-issues] li')].map(item=>item.dataset.code),disabled:e.querySelector('[data-action=add-vector]').disabled,eyeAvailable:!e.querySelector('[data-layer-id] [data-action=layer-eye]')?.disabled,issues:e._doc.rendererIssues.map(item=>item.code)};})()")
        expected_renderer_codes = {"UNSUPPORTED_LAYER_KIND", "UNSUPPORTED_LAYER_BLEND_MODE", "UNSUPPORTED_LAYER_CLIPPING",
                                   "UNSUPPORTED_OBJECT_KIND", "UNSUPPORTED_OBJECT_SHAPE"}
        check(receipt, "authoritative_view_unsupported_semantics_visible_and_fail_closed",
              fixture_view["documentId"] == renderer_fixture_id and expected_renderer_codes.issubset(set(renderer_ui["issues"])) and
              "Renderer fidelity blocked editing" in renderer_ui["banner"] and renderer_ui["disabled"] is True and
              renderer_ui["eyeAvailable"] is True and "PAINT_HARDNESS_PREVIEW_APPROXIMATION" in renderer_ui["codes"],
              {"fixture": fixture_view, **renderer_ui})
        original_view = cdp.evaluate(f"""(async()=>{{const e=document.querySelector('creative-document-editor'); const opened=await e._request(`/creative_document_api/documents/{created_document_id}/open`,{{method:'POST',body:{{discardUnsaved:true}}}}); e._doc=opened.view; e._selectedLayerId=e._doc.rootLayerIds[0]; await e._acceptView(e._doc); return {{documentId:e._doc.documentId,revision:e._doc.revision,dirty:e._doc.dirty}};}})()""")
        check(receipt, "authoritative_fixture_close_restores_saved_scene", original_view["documentId"] == created_document_id and
              original_view["dirty"] is False, original_view)

        # Save/reopen/export with the layer renderer and thumbnail scheduler active.
        cdp.evaluate("""(()=>{const e=document.querySelector('creative-document-editor'); e._fitToView(); return e._saveDocument();})()""")
        wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return e._doc && e._doc.dirty===false && e._doc.committedRevision===e._doc.revision;})()", timeout=45)
        before_reload = cdp.evaluate("""(async()=>{const e=document.querySelector('creative-document-editor'); const blob=await e._apiBlob(`/creative_document_api/documents/${encodeURIComponent(e._doc.documentId)}/export`); const bytes=new Uint8Array(await blob.arrayBuffer()); const hash=await crypto.subtle.digest('SHA-256',bytes); return {revision:e._doc.revision,size:bytes.length,digest:[...new Uint8Array(hash)].map(x=>x.toString(16).padStart(2,'0')).join('')};})()""")
        cdp.evaluate("document.querySelector('creative-document-editor')._openDocument()")
        wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return e._doc && !e._dirty && e._doc.recoveryNotice;})()", timeout=45)
        export = cdp.evaluate("""(async()=>{const e=document.querySelector('creative-document-editor'); const blob=await e._apiBlob(`/creative_document_api/documents/${encodeURIComponent(e._doc.documentId)}/export`); const bytes=new Uint8Array(await blob.arrayBuffer()); const hash=await crypto.subtle.digest('SHA-256',bytes); return {type:blob.type,size:bytes.length,revision:e._doc.revision,digest:[...new Uint8Array(hash)].map(x=>x.toString(16).padStart(2,'0')).join(''),schedulerPeak:e._scheduler.peakActive,surfaces:e._stage.getLayers().length};})()""")
        check(receipt, "save_reload_composite_bytes_match_and_export", export["type"] == "image/png" and export["size"] > 1000 and
              export["revision"] == before_reload["revision"] and export["digest"] == before_reload["digest"] and
              export["schedulerPeak"] <= 3 and export["surfaces"] == 2, {"beforeReload": before_reload, **export})

        # Save creates the files an attacker might try through Gradio's global /file route.
        scene_root = output_root / "creative_documents" / f"{created_document_id}.nexscene"
        if not scene_root.is_dir():
            raise FileNotFoundError("saved scene directory was not found below creative_documents")
        files = [path for path in scene_root.rglob("*") if path.is_file()]
        manifest = next(path for path in files if path.name == "manifest.json" and "revisions" in path.parts)
        project_pointer = scene_root / "manifest.json"
        asset_file = next(path for path in files if any(
            path.parts[index:index + 2] == ("assets", "sha256") for index in range(len(path.parts) - 1)
        ))
        checkpoint = next(path for path in files if "checkpoints" in path.parts and path.name == "manifest.json")
        history = next(path for path in files if "history" in path.parts and path.is_file())
        ordinary_output = output_root / f"ncd-audit-output-{uuid.uuid4().hex}.png"
        from PIL import Image

        Image.new("RGB", (3, 3), (25, 85, 150)).save(ordinary_output, format="PNG")
        private_paths = {"project_pointer": project_pointer, "manifest": manifest, "history": history,
                         "checkpoint": checkpoint, "asset": asset_file}
        generic_results: dict[str, Any] = {}
        owner_headers = cdp.evaluate("(()=>{const e=document.querySelector('creative-document-editor'); return {Authorization:`Bearer ${e._capability.token}`,'X-Gradio-Session':e._capability.session,'X-Editor-Owner-Key':e._ownerKey};})()")
        for label, path in private_paths.items():
            owner_result = generic_file_probe(cdp, base_url, str(path), headers=owner_headers)
            anonymous_result = generic_file_probe(cdp, base_url, str(path))
            generic_results[label] = {"owner": owner_result, "anonymous": anonymous_result}
        blocked_ok = all(result[side]["status"] != 200 or result[side]["length"] == 0
                         for result in generic_results.values() for side in ("owner", "anonymous"))
        check(receipt, "generic_gradio_file_route_blocks_scene_authority", blocked_ok, generic_results)
        authorized_asset = cdp.evaluate("""(async()=>{const e=document.querySelector('creative-document-editor'); const asset=e._doc.assets.find(x=>x.id===e._doc.objects.find(o=>o.assetId)?.assetId); const blob=await e._apiBlob(asset.url); return {type:blob.type,length:blob.size};})()""")
        check(receipt, "owner_capability_asset_route_remains_available", authorized_asset["type"] == "image/png" and authorized_asset["length"] > 0,
              authorized_asset)
        output_result = generic_file_probe(cdp, base_url, str(ordinary_output))
        check(receipt, "ordinary_output_image_remains_served", output_result["status"] == 200 and output_result["length"] > 0, output_result)

        private_variants = {
            "relative": os.path.relpath(manifest, repo),
            "alternate_separator": str(manifest).replace("/", "\\"),
        }
        variant_results = {name: generic_file_probe(cdp, base_url, value) for name, value in private_variants.items()}
        variant_results["encoded_separators"] = generic_file_probe(
            cdp, base_url, urllib.parse.quote(str(manifest), safe=""), preserve_escapes=True,
        )
        check(receipt, "relative_encoded_and_alternate_separator_paths_blocked",
              all(item["status"] != 200 or item["length"] == 0 for item in variant_results.values()), variant_results)

        symlink_alias = output_root / f"ncd-audit-alias-{uuid.uuid4().hex}"
        alias_kind = create_directory_alias(scene_root, symlink_alias)
        try:
            symlink_path = symlink_alias / manifest.relative_to(scene_root)
            symlink_result = generic_file_probe(cdp, base_url, str(symlink_path))
            check(receipt, "symlink_alias_cannot_read_scene_manifest", symlink_result["status"] != 200 or symlink_result["length"] == 0,
                  {"available": True, "kind": alias_kind, "status": symlink_result["status"], "length": symlink_result["length"]})
        except Exception as exc:
            receipt["checks"]["symlink_alias_cannot_read_scene_manifest"] = {
                "passed": False,
                "evidence": {"available": symlink_alias.exists(), "reason": type(exc).__name__},
            }
            raise

        from gradio.utils import get_cache_folder

        cache_root = Path(get_cache_folder()).resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_alias = cache_root / f"ncd-audit-cache-alias-{uuid.uuid4().hex}"
        cache_alias_kind = create_directory_alias(scene_root, cache_alias)
        cache_result = generic_file_probe(cdp, base_url, str(cache_alias / manifest.relative_to(scene_root)))
        check(receipt, "cache_symlink_alias_cannot_read_scene_manifest", cache_result["status"] != 200 or cache_result["length"] == 0,
              {"available": True, "kind": cache_alias_kind, "status": cache_result["status"], "length": cache_result["length"]})

        # A fresh browser mount keeps the local owner key and the saved owner descriptor.
        cdp.evaluate("""(()=>{const e=document.querySelector('creative-document-editor'); const p=e.parentNode; e.remove(); p.appendChild(e); return true;})()""")
        wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return e && e._stage && e._capability && e._capability.enabled && e._doc.documentId;})()", timeout=30)
        reconnect = cdp.evaluate("(()=>{const e=document.querySelector('creative-document-editor'); return {documentId:e._doc.documentId,revision:e._doc.revision,ownerKeyLength:e._ownerKey.length,konva:window.Konva.version};})()")
        check(receipt, "editor_remount_preserves_scene_and_owner_key", reconnect["documentId"] == created_document_id and
              reconnect["ownerKeyLength"] >= 32 and reconnect["konva"] == "10.6.0", reconnect)

        # Delete is an explicit soft-delete: the row remains as recoverable history and Undo restores it.
        delete_probe = cdp.evaluate("""(()=>{const e=document.querySelector('creative-document-editor');
          window.__deleteProbeLayerIds=e._doc.layers.map(layer=>layer.id); window.confirm=()=>true;
          const startRevision=e._doc.revision; e.querySelector('[data-action=add-vector]').click();
          return {startRevision};})()""")
        wait_js(cdp, f"(()=>{{const e=document.querySelector('creative-document-editor'); return !e._actionInFlight && e._doc.revision==={delete_probe['startRevision'] + 1};}})()", timeout=30)
        delete_probe_id = cdp.evaluate("(()=>{const e=document.querySelector('creative-document-editor'); return e._doc.layers.find(layer=>!window.__deleteProbeLayerIds.includes(layer.id)).id;})()")
        cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor');
          const row=e.querySelector(`[data-layer-id="{delete_probe_id}"]`); if(!row) throw new Error('delete probe layer row is missing');
          row.click(); e.querySelector('[data-action=delete-layer]').click();}})()""")
        wait_js(cdp, f"(()=>{{const e=document.querySelector('creative-document-editor'),layer=e._doc.layers.find(item=>item.id==={json.dumps(delete_probe_id)}); return !e._actionInFlight && e._doc.revision==={delete_probe['startRevision'] + 2} && layer?.deleted;}})()", timeout=30)
        deleted_layer = cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'),layer=e._doc.layers.find(item=>item.id==={json.dumps(delete_probe_id)}),row=e.querySelector(`[data-layer-id="{delete_probe_id}"]`); return {{revision:e._doc.revision,selectedLayerId:e._selectedLayerId,deleted:layer.deleted,visible:layer.visible,locked:layer.locked,rowText:row?.innerText||''}};}})()""")
        cdp.evaluate("document.querySelector('creative-document-editor').querySelector('[data-action=undo]').click()")
        wait_js(cdp, f"(()=>{{const e=document.querySelector('creative-document-editor'),layer=e._doc.layers.find(item=>item.id==={json.dumps(delete_probe_id)}); return !e._actionInFlight && e._doc.revision==={delete_probe['startRevision'] + 3} && layer && !layer.deleted;}})()", timeout=30)
        restored_layer = cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'),layer=e._doc.layers.find(item=>item.id==={json.dumps(delete_probe_id)}); return {{revision:e._doc.revision,deleted:layer.deleted,visible:layer.visible,locked:layer.locked}};}})()""")
        check(receipt, "delete_layer_control_soft_deletes_selected_layer_and_undo_restores",
              deleted_layer["selectedLayerId"] == delete_probe_id and deleted_layer["deleted"] is True and
              deleted_layer["visible"] is False and deleted_layer["locked"] is True and "deleted" in deleted_layer["rowText"] and
              restored_layer["deleted"] is False and restored_layer["visible"] is True and restored_layer["locked"] is False,
              {"layerId": delete_probe_id, "deleted": deleted_layer, "restored": restored_layer})

        # Final corruption probe forces the browser loader down its missing-asset path.
        asset_record = json.loads(manifest.read_text(encoding="utf-8"))["assets"]
        asset_entry = asset_record[setup["assetId"]]
        asset_storage_uri = asset_entry["storageUri"]
        stored_asset = scene_root / Path(*asset_storage_uri.replace("\\", "/").split("/"))
        stored_asset.write_bytes(b"audit-induced content mismatch")
        cdp.evaluate(f"""(()=>{{const e=document.querySelector('creative-document-editor'); const asset=e._doc.assets.find(x=>x.id==={json.dumps(setup['assetId'])}); asset.url+=`?auditProbe=${{Date.now()}}`; e._assetCache.clear(); e._renderDocument();}})()""")
        wait_js(cdp, "(()=>{const e=document.querySelector('creative-document-editor'); return e._rendererIssues.some(x=>x.severity==='error' && x.code==='ASSET_UNAVAILABLE');})()", timeout=30)
        missing_asset = cdp.evaluate("(()=>{const e=document.querySelector('creative-document-editor'); return {issue:e._rendererIssues.find(x=>x.code==='ASSET_UNAVAILABLE'),banner:e.querySelector('[data-role=fidelity-issues]')?.innerText,disabled:e.querySelector('[data-action=add-vector]').disabled};})()")
        check(receipt, "missing_mismatched_asset_becomes_typed_visible_error", missing_asset["disabled"] is True and
              "ASSET_UNAVAILABLE" in missing_asset["banner"], missing_asset)

        success = True
        receipt["status"] = "passed"
        receipt["finishedAtUtc"] = datetime.now(timezone.utc).isoformat()
        receipt["summary"] = {"browser": browser_version, "checksPassed": sum(item["passed"] for item in receipt["checks"].values()),
                              "checksTotal": len(receipt["checks"]), "largeRaster": [2400, 1792],
                              "reproCommand": "venv\\Scripts\\python.exe tests\\browser\\run_creative_document_editor_audit.py --receipt <path>"}
        return 0
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["finishedAtUtc"] = datetime.now(timezone.utc).isoformat()
        receipt["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        return_code = 1
        return return_code
    finally:
        if cdp is not None:
            try:
                cdp.evaluate("""(async()=>{for(const id of ['uov_input_slot','inpaint_bb_canvas','inpaint_mask_canvas']){
                  const slot=document.getElementById(id); if(slot&&slot.uploadMode==='api'){
                    try{await slot.clearApiState();slot.clearPreview();}catch(_){}}
                }})()""")
            except Exception:
                pass
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
        log_stream.close()
        if output_root is not None:
            for alias in (symlink_alias, cache_alias):
                try:
                    remove_directory_alias(alias)
                except OSError:
                    receipt.setdefault("cleanup", []).append("generated alias cleanup needs attention")
            if ordinary_output is not None:
                ordinary_output.unlink(missing_ok=True)
            if created_document_id:
                owned_root = (output_root / "creative_documents").resolve()
                for generated_id in (created_document_id, renderer_fixture_id):
                    if not generated_id:
                        continue
                    scene_root = owned_root / f"{generated_id}.nexscene"
                    owner_file = owned_root / f"{generated_id}.owner.json"
                    try:
                        if scene_root.resolve().parent == owned_root and scene_root.exists():
                            shutil.rmtree(scene_root)
                        if owner_file.parent.resolve() == owned_root:
                            owner_file.unlink(missing_ok=True)
                    except OSError:
                        receipt.setdefault("cleanup", []).append("generated project cleanup needs attention")
        receipt.setdefault("summary", {"checksPassed": sum(item.get("passed", False) for item in receipt["checks"].values()),
                                        "checksTotal": len(receipt["checks"])})
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        if not success:
            print(json.dumps(receipt, indent=2, sort_keys=True))
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
