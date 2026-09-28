from __future__ import annotations

import hashlib
import json
import io
import secrets
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image
from starlette.requests import Request

import modules.config
import modules.creative_document_editor_api as editor_api
import args_manager
from modules.creative_document_editor_api import CreativeDocumentRuntime, EditorSessionRegistry
from modules.ui_components.creative_document_panel import _editor_capability
from modules.ui_gradio_extensions import javascript_html


REPO = Path(__file__).resolve().parents[1]


def _request(token: str, session: str, *, origin: str = "http://testserver", include_origin: bool = True,
             include_fetch_site: bool = True) -> Request:
    headers = [
        (b"authorization", f"Bearer {token}".encode()),
        (b"x-gradio-session", session.encode()),
        (b"host", b"testserver"),
    ]
    if include_fetch_site:
        headers.append((b"sec-fetch-site", b"same-origin"))
    if include_origin:
        headers.append((b"origin", origin.encode()))
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "GET", "scheme": "http", "path": "/", "raw_path": b"/",
        "query_string": b"", "headers": headers, "server": ("testserver", 80),
        "client": ("testclient", 50000),
    }
    return Request(scope)


def test_session_registry_rejects_expired_cross_session_and_cross_origin_credentials() -> None:
    registry = EditorSessionRegistry(ttl_seconds=60)
    capability = registry.issue("session-a", "director", remote_exposure=True)
    assert capability["enabled"] is True
    registry.authorize(_request(capability["token"], "session-a"))

    with pytest.raises(HTTPException) as error:
        registry.authorize(_request(capability["token"], "session-b"))
    assert error.value.status_code == 403
    assert error.value.detail["code"] == "EDITOR_CROSS_SESSION"

    with pytest.raises(HTTPException) as error:
        registry.authorize(_request(capability["token"], "session-a", origin="https://attacker.invalid"))
    assert error.value.status_code == 403
    assert error.value.detail["code"] == "EDITOR_CROSS_ORIGIN"

    assert registry.authorize(_request(capability["token"], "session-a", include_origin=False)).session_hash == "session-a"
    with pytest.raises(HTTPException) as error:
        registry.authorize(_request(capability["token"], "session-a", include_origin=False, include_fetch_site=False))
    assert error.value.status_code == 403

    local_only = registry.issue("anonymous", None, remote_exposure=True)
    assert local_only["enabled"] is False
    assert "token" not in local_only

    expired_registry = EditorSessionRegistry(ttl_seconds=0)
    expired = expired_registry.issue("session-expired", "director", remote_exposure=False)
    with pytest.raises(HTTPException) as error:
        expired_registry.authorize(_request(expired["token"], "session-expired"))
    assert error.value.status_code == 401
    assert error.value.detail["code"] == "EDITOR_SESSION_EXPIRED"


def test_loopback_gradio_listener_is_local_but_wildcard_bind_requires_login(monkeypatch) -> None:
    monkeypatch.setattr(args_manager.args, "share", False, raising=False)
    monkeypatch.setattr(args_manager.args, "listen", "127.0.0.1", raising=False)
    local = json.loads(_editor_capability(type("GradioRequest", (), {"session_hash": "local-session", "username": None})()))
    assert local["enabled"] is True

    monkeypatch.setattr(args_manager.args, "listen", "0.0.0.0", raising=False)
    remote = json.loads(_editor_capability(type("GradioRequest", (), {"session_hash": "remote-session", "username": None})()))
    assert remote["enabled"] is False
    assert "authenticated" in remote["reason"]


def test_api_keeps_new_scene_in_memory_until_explicit_save_and_reopens(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(modules.config, "path_outputs", str(tmp_path))
    runtime = CreativeDocumentRuntime()
    monkeypatch.setattr(editor_api, "creative_document_runtime", runtime)
    app = FastAPI()
    app.include_router(editor_api.creative_document_router)
    client = TestClient(app)
    capability = runtime.sessions.issue("session-a", "director", remote_exposure=False)
    headers = {
        "Authorization": f"Bearer {capability['token']}",
        "X-Gradio-Session": "session-a",
        "Origin": "http://testserver",
        "Sec-Fetch-Site": "same-origin",
    }

    created = client.post("/creative_document_api/documents", headers=headers, json={"width": 32, "height": 24, "name": "Unit scene"})
    assert created.status_code == 200, created.text
    document_id = created.json()["documentId"]
    assert created.json()["view"]["dirty"] is True
    project_path = runtime.project_path(document_id)
    assert not (project_path / "manifest.json").exists()

    cross_session = dict(headers, **{"X-Gradio-Session": "session-b"})
    refused = client.get(f"/creative_document_api/documents/{document_id}/view", headers=cross_session)
    assert refused.status_code == 403
    assert refused.json()["detail"]["code"] == "EDITOR_CROSS_SESSION"

    action = client.post(f"/creative_document_api/documents/{document_id}/actions", headers=headers, json={
        "documentId": document_id,
        "expectedRevision": 0,
        "actorKind": "director",
        "actionType": "add_layer",
        "data": {"kind": "vector", "name": "Ink"},
    })
    assert action.status_code == 200, action.text
    assert action.json()["currentRevision"] == 1
    assert action.json()["dirty"] is True

    saved = client.post(f"/creative_document_api/documents/{document_id}/save", headers=headers, json={"expectedRevision": 1})
    assert saved.status_code == 200, saved.text
    assert saved.json()["dirty"] is False
    assert (project_path / "manifest.json").is_file()

    reopened = client.post(f"/creative_document_api/documents/{document_id}/open", headers=headers, json={"discardUnsaved": False})
    assert reopened.status_code == 200, reopened.text
    view = client.get(f"/creative_document_api/documents/{document_id}/view", headers=headers)
    assert view.status_code == 200
    assert view.json()["revision"] == 1
    assert any(layer["name"] == "Ink" for layer in view.json()["layers"])
    response_material = json.dumps(view.json())
    assert str(tmp_path.resolve()) not in response_material
    assert "storageUri" not in response_material

    unknown = client.get("/creative_document_api/documents/doc-unknown/view", headers=headers)
    assert unknown.status_code == 404
    assert str(tmp_path.resolve()) not in unknown.text
    bad_asset = client.get(f"/creative_document_api/documents/{document_id}/assets/asset-unknown/content", headers=headers)
    assert bad_asset.status_code == 404
    assert str(tmp_path.resolve()) not in bad_asset.text

    with pytest.raises(HTTPException) as error:
        runtime.project_path("doc-../../outside")
    assert error.value.status_code == 404


def test_saved_scene_supports_later_edit_undo_redo_and_save(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(modules.config, "path_outputs", str(tmp_path))
    runtime = CreativeDocumentRuntime()
    monkeypatch.setattr(editor_api, "creative_document_runtime", runtime)
    app = FastAPI()
    app.include_router(editor_api.creative_document_router)
    client = TestClient(app)
    capability = runtime.sessions.issue("session-history", "director", remote_exposure=False)
    headers = {
        "Authorization": f"Bearer {capability['token']}",
        "X-Gradio-Session": "session-history",
        "Origin": "http://testserver",
        "Sec-Fetch-Site": "same-origin",
    }
    created = client.post("/creative_document_api/documents", headers=headers, json={"width": 32, "height": 24})
    document_id = created.json()["documentId"]

    first = client.post(f"/creative_document_api/documents/{document_id}/actions", headers=headers, json={
        "expectedRevision": 0, "actorKind": "director", "actionType": "add_layer",
        "data": {"kind": "vector", "name": "First edit"},
    })
    assert first.status_code == 200, first.text
    saved = client.post(f"/creative_document_api/documents/{document_id}/save", headers=headers,
                        json={"expectedRevision": 1})
    assert saved.status_code == 200, saved.text
    assert runtime._handles[document_id].document.checkpoint_refs == ["checkpoints/1/manifest.json"]

    second = client.post(f"/creative_document_api/documents/{document_id}/actions", headers=headers, json={
        "expectedRevision": 1, "actorKind": "director", "actionType": "add_layer",
        "data": {"kind": "paint", "name": "Second edit"},
    })
    assert second.status_code == 200, second.text
    assert runtime._handles[document_id].document.checkpoint_refs == []

    undone = client.post(f"/creative_document_api/documents/{document_id}/actions", headers=headers, json={
        "expectedRevision": 2, "actorKind": "director", "actionType": "undo",
    })
    assert undone.status_code == 200, undone.text
    after_undo = client.get(f"/creative_document_api/documents/{document_id}/view", headers=headers).json()
    assert after_undo["revision"] == 3
    assert not any(layer["name"] == "Second edit" for layer in after_undo["layers"])

    redone = client.post(f"/creative_document_api/documents/{document_id}/actions", headers=headers, json={
        "expectedRevision": 3, "actorKind": "director", "actionType": "redo",
    })
    assert redone.status_code == 200, redone.text
    redone_view = client.get(f"/creative_document_api/documents/{document_id}/view", headers=headers).json()
    assert redone_view["revision"] == 4
    assert any(layer["name"] == "Second edit" for layer in redone_view["layers"])
    saved_again = client.post(f"/creative_document_api/documents/{document_id}/save", headers=headers,
                              json={"expectedRevision": 4})
    assert saved_again.status_code == 200, saved_again.text
    reopened = client.post(f"/creative_document_api/documents/{document_id}/open", headers=headers,
                           json={"discardUnsaved": False})
    assert reopened.status_code == 200, reopened.text
    reopened_view = client.get(f"/creative_document_api/documents/{document_id}/view", headers=headers).json()
    assert reopened_view["revision"] == 4
    assert any(layer["name"] == "Second edit" for layer in reopened_view["layers"])


def _headers(capability, session: str, *, owner_key: str | None = None) -> dict[str, str]:
    result = {
        "Authorization": f"Bearer {capability['token']}",
        "X-Gradio-Session": session,
        "Origin": "http://testserver",
        "Sec-Fetch-Site": "same-origin",
    }
    if owner_key is not None:
        result["X-Editor-Owner-Key"] = owner_key
    return result


def test_projects_are_bound_to_principal_across_every_document_route_and_reconnect(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(modules.config, "path_outputs", str(tmp_path))
    runtime = CreativeDocumentRuntime()
    monkeypatch.setattr(editor_api, "creative_document_runtime", runtime)
    app = FastAPI()
    app.include_router(editor_api.creative_document_router)
    client = TestClient(app)
    alice = runtime.sessions.issue("alice-session", "Alice", remote_exposure=True)
    bob = runtime.sessions.issue("bob-session", "Bob", remote_exposure=True)
    alice_headers = _headers(alice, "alice-session")
    bob_headers = _headers(bob, "bob-session")

    created = client.post(
        "/creative_document_api/documents", headers=alice_headers,
        json={"width": 24, "height": 16, "name": "Private scene"},
    )
    assert created.status_code == 200, created.text
    document_id = created.json()["documentId"]
    output = io.BytesIO()
    Image.new("RGBA", (24, 16), (35, 80, 140, 255)).save(output, format="PNG")
    raw = output.getvalue()
    imported = client.post(
        f"/creative_document_api/documents/{document_id}/assets/import",
        headers=alice_headers,
        data={"expected_revision": "0", "actor_kind": "director"},
        files={"file": ("private.png", raw, "image/png")},
    )
    assert imported.status_code == 200, imported.text
    asset_id = next(identity for identity in imported.json()["createdIds"] if identity.startswith("asset-"))
    saved = client.post(
        f"/creative_document_api/documents/{document_id}/save",
        headers=alice_headers, json={"expectedRevision": 1},
    )
    assert saved.status_code == 200, saved.text

    denied_requests = [
        client.get(f"/creative_document_api/documents/{document_id}/view", headers=bob_headers),
        client.get(f"/creative_document_api/documents/{document_id}/assets/{asset_id}/content", headers=bob_headers),
        client.get(f"/creative_document_api/documents/{document_id}/assets/{asset_id}/thumbnail", headers=bob_headers),
        client.get(f"/creative_document_api/documents/{document_id}/preview", headers=bob_headers),
        client.get(f"/creative_document_api/documents/{document_id}/export", headers=bob_headers),
        client.post(
            f"/creative_document_api/documents/{document_id}/open",
            headers=bob_headers, json={"discardUnsaved": True},
        ),
        client.post(
            f"/creative_document_api/documents/{document_id}/save",
            headers=bob_headers, json={"expectedRevision": 1},
        ),
        client.post(
            f"/creative_document_api/documents/{document_id}/actions",
            headers=bob_headers,
            json={"expectedRevision": 1, "actionType": "add_layer", "data": {"name": "Intrusion"}},
        ),
    ]
    assert all(response.status_code == 404 for response in denied_requests)
    assert all(response.json()["detail"]["code"] == "UNKNOWN_DOCUMENT" for response in denied_requests)
    assert all(document_id not in response.text and str(tmp_path.resolve()) not in response.text
               for response in denied_requests)

    # A new authenticated session for the same principal can reconnect after
    # the runtime handle cache is lost; the persisted owner sidecar is checked.
    restarted_runtime = CreativeDocumentRuntime()
    monkeypatch.setattr(editor_api, "creative_document_runtime", restarted_runtime)
    alice_reconnected = restarted_runtime.sessions.issue("alice-session-2", "alice", remote_exposure=True)
    reopened = client.post(
        f"/creative_document_api/documents/{document_id}/open",
        headers=_headers(alice_reconnected, "alice-session-2"),
        json={"discardUnsaved": False},
    )
    assert reopened.status_code == 200, reopened.text
    assert reopened.json()["documentId"] == document_id


def test_local_owner_key_allows_same_browser_reconnect_without_id_authority(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(modules.config, "path_outputs", str(tmp_path))
    runtime = CreativeDocumentRuntime()
    monkeypatch.setattr(editor_api, "creative_document_runtime", runtime)
    app = FastAPI()
    app.include_router(editor_api.creative_document_router)
    client = TestClient(app)
    owner_key = secrets.token_urlsafe(32)
    first = runtime.sessions.issue("local-tab-1", None, remote_exposure=False)
    created = client.post(
        "/creative_document_api/documents",
        headers=_headers(first, "local-tab-1", owner_key=owner_key),
        json={"width": 16, "height": 16},
    )
    assert created.status_code == 200, created.text
    document_id = created.json()["documentId"]
    saved = client.post(
        f"/creative_document_api/documents/{document_id}/save",
        headers=_headers(first, "local-tab-1", owner_key=owner_key),
        json={"expectedRevision": 0},
    )
    assert saved.status_code == 200, saved.text

    restarted_runtime = CreativeDocumentRuntime()
    monkeypatch.setattr(editor_api, "creative_document_runtime", restarted_runtime)
    same_browser = restarted_runtime.sessions.issue("local-tab-2", None, remote_exposure=False)
    reopened = client.post(
        f"/creative_document_api/documents/{document_id}/open",
        headers=_headers(same_browser, "local-tab-2", owner_key=owner_key),
        json={"discardUnsaved": False},
    )
    assert reopened.status_code == 200, reopened.text
    other_browser = restarted_runtime.sessions.issue("other-tab", None, remote_exposure=False)
    denied = client.get(
        f"/creative_document_api/documents/{document_id}/view",
        headers=_headers(other_browser, "other-tab", owner_key=secrets.token_urlsafe(32)),
    )
    assert denied.status_code == 404
    assert denied.json()["detail"]["code"] == "UNKNOWN_DOCUMENT"


def test_import_asset_routes_verify_immutable_bytes_and_hide_storage_errors(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(modules.config, "path_outputs", str(tmp_path))
    runtime = CreativeDocumentRuntime()
    monkeypatch.setattr(editor_api, "creative_document_runtime", runtime)
    app = FastAPI()
    app.include_router(editor_api.creative_document_router)
    client = TestClient(app)
    capability = runtime.sessions.issue("session-assets", "director", remote_exposure=False)
    headers = {
        "Authorization": f"Bearer {capability['token']}",
        "X-Gradio-Session": "session-assets",
        "Origin": "http://testserver",
        "Sec-Fetch-Site": "same-origin",
    }
    created = client.post("/creative_document_api/documents", headers=headers, json={"width": 24, "height": 16})
    document_id = created.json()["documentId"]
    output = io.BytesIO()
    Image.new("RGBA", (24, 16), (32, 90, 180, 255)).save(output, format="PNG")
    raw = output.getvalue()
    imported = client.post(
        f"/creative_document_api/documents/{document_id}/assets/import",
        headers=headers,
        data={"expected_revision": "0", "actor_kind": "director"},
        files={"file": ("source.png", raw, "image/png")},
    )
    assert imported.status_code == 200, imported.text
    asset_id = next(identity for identity in imported.json()["createdIds"] if identity.startswith("asset-"))
    content = client.get(f"/creative_document_api/documents/{document_id}/assets/{asset_id}/content", headers=headers)
    assert content.status_code == 200
    assert content.content == raw
    assert content.headers["etag"].strip('"') == hashlib.sha256(raw).hexdigest()
    thumbnail = client.get(f"/creative_document_api/documents/{document_id}/assets/{asset_id}/thumbnail", headers=headers)
    assert thumbnail.status_code == 200
    assert thumbnail.headers["content-type"].startswith("image/png")

    record = runtime._handles[document_id].document.assets[asset_id]
    stored_path = runtime.project_path(document_id) / record.storage_uri
    stored_path.write_bytes(b"tampered bytes")
    refused = client.get(f"/creative_document_api/documents/{document_id}/assets/{asset_id}/content", headers=headers)
    assert refused.status_code == 404
    assert refused.json()["detail"]["code"] == "ASSET_UNAVAILABLE"
    assert str(tmp_path.resolve()) not in refused.text


@pytest.mark.parametrize("route", ["preview", "export"])
def test_preview_and_export_metadata_stays_with_render_snapshot(route, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(modules.config, "path_outputs", str(tmp_path))
    runtime = CreativeDocumentRuntime()
    monkeypatch.setattr(editor_api, "creative_document_runtime", runtime)
    app = FastAPI()
    app.include_router(editor_api.creative_document_router)
    client = TestClient(app)
    capability = runtime.sessions.issue("snapshot-session", "director", remote_exposure=False)
    headers = {
        "Authorization": f"Bearer {capability['token']}",
        "X-Gradio-Session": "snapshot-session",
        "Origin": "http://testserver",
        "Sec-Fetch-Site": "same-origin",
    }
    created = client.post("/creative_document_api/documents", headers=headers,
                          json={"width": 32, "height": 24, "name": "Snapshot scene"})
    assert created.status_code == 200, created.text
    document_id = created.json()["documentId"]
    added = client.post(f"/creative_document_api/documents/{document_id}/actions", headers=headers, json={
        "expectedRevision": 0, "actorKind": "director", "actionType": "add_layer",
        "data": {"kind": "vector", "name": "Bound snapshot"},
    })
    assert added.status_code == 200, added.text
    handle = runtime._handles[document_id]
    with handle.lock:
        snapshot = deepcopy(handle.document)
        expected_id = snapshot.document_id
        expected_revision = snapshot.current_revision
        expected_digest = snapshot.revision_digest
        load_asset = lambda asset_id: handle.store.assets.read_bytes(snapshot.assets[asset_id])
        if route == "preview":
            image = editor_api.render_document(snapshot, load_asset)
            image.thumbnail((1600, 1200), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            image.save(output, format="PNG", optimize=False, compress_level=9)
            expected_content = output.getvalue()
        else:
            expected_content = editor_api.render_document_png(snapshot, load_asset)

    interleaving: dict[str, Any] = {}

    class AdvanceDocumentOnLockExit:
        """Advance the handle after the route releases its render snapshot lock."""

        def __init__(self, lock):
            self.lock = lock
            self.advanced = False

        def __enter__(self):
            self.lock.acquire()
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.lock.release()
            if exc_type is None and not self.advanced:
                self.advanced = True
                with self.lock:
                    current = handle.document
                    changed = editor_api.prepare_action(current, handle.store.assets, {
                        "documentId": document_id,
                        "expectedRevision": current.current_revision,
                        "actorKind": "director",
                        "actionType": "add_layer",
                        "data": {"kind": "vector", "name": "Concurrent later edit"},
                    }, actor_id="snapshot-race")
                    handle.document = changed.document
                    interleaving.update({
                        "revision": handle.document.current_revision,
                        "digest": handle.document.revision_digest,
                    })
            return False

    lock_interleaving = AdvanceDocumentOnLockExit(handle.lock)
    monkeypatch.setattr(handle, "lock", lock_interleaving)
    response = client.get(f"/creative_document_api/documents/{document_id}/{route}", headers=headers)
    assert response.status_code == 200, response.text
    assert lock_interleaving.advanced
    assert interleaving["revision"] == expected_revision + 1
    assert interleaving["digest"] != expected_digest
    assert response.content == expected_content
    assert response.headers["x-document-id"] == expected_id
    assert response.headers["x-document-revision"] == str(expected_revision)
    assert response.headers["x-document-revision-digest"] == expected_digest
    if route == "export":
        assert response.headers["content-disposition"] == (
            f'attachment; filename="creative-document-{expected_id}-r{expected_revision}.png"'
        )


def test_pinned_vendor_hash_loading_order_and_connection_budget_contract() -> None:
    vendor_root = REPO / "javascript" / "vendor" / "konva" / "10.6.0"
    vendor = (vendor_root / "konva.min.js").read_bytes()
    license_bytes = (vendor_root / "LICENSE").read_bytes()
    provenance = json.loads((vendor_root / "PROVENANCE.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(vendor).hexdigest().upper() == provenance["files"]["konva.min.js"]["sha256"]
    assert hashlib.sha256(license_bytes).hexdigest().upper() == provenance["files"]["LICENSE"]["sha256"]
    assert provenance["version"] == "10.6.0"
    assert provenance["runtimeDependencies"] == []

    head = javascript_html()
    vendor_position = head.find("/creative_document_api/vendor/konva-10.6.0.js")
    editor_position = head.find("customElements.define('creative-document-editor'")
    assert vendor_position >= 0 and editor_position > vendor_position
    assert f"sha256-{__import__('base64').b64encode(bytes.fromhex(provenance['files']['konva.min.js']['sha256'])).decode()}" in head

    editor_js = (REPO / "javascript" / "modules" / "55_creative_document_editor.js").read_text(encoding="utf-8")
    assert "const MAX_ASSET_FETCHES = 3" in editor_js
    assert "Math.min(3, Math.max(1, limit || 3))" in editor_js
    assert "thumbnailPromise" in editor_js and "fullPromise" in editor_js
    assert "headers: this.headers()" in editor_js
    assert "Authorization: `Bearer ${this._capability.token}`" in editor_js
    assert "if (this._viewSnapshots.has(key)) return this._viewSnapshots.get(key)" in editor_js
    assert "EventSource" not in editor_js and "WebSocket" not in editor_js
    assert "setInterval" not in editor_js and "setTimeout" not in editor_js
    assert editor_js.count("new window.Konva.Layer") == 2

    # W03 mounts alongside the established specialized mask and image-slot seams.
    assert "nex-image-slot" in (REPO / "javascript" / "modules" / "40_nex_image_slot.js").read_text(encoding="utf-8")
    assert "staging_router" in (REPO / "modules" / "staging_api.py").read_text(encoding="utf-8")
    assert "creative_document_router" in (REPO / "webui.py").read_text(encoding="utf-8")
    mask_js = (REPO / "javascript" / "modules" / "10_inpaint_mask.js").read_text(encoding="utf-8")
    assert "inpaint_bb_canvas" in mask_js and "outpaint_bb_canvas" in mask_js
    assert "creative_document" not in mask_js
