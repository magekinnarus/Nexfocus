"""Session-bound human UI bridge for the W03 creative-document editor.

This module deliberately exposes no Agent identity, safe-view protocol, or
general semantic command transport. Every route resolves opaque document and
asset IDs below a configured server-owned root.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import APIRouter, Body, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from PIL import Image

import modules.config
from modules.creative_document import (
    AssetStore,
    Document,
    HistoryValidationError,
    LayerRecord,
    ProjectStore,
    SchemaValidationError,
    make_id,
    validate_id,
)
from modules.creative_document.editor_actions import (
    EditorActionError,
    actor_for_session,
    prepare_action,
    prepare_raster_import,
    prepare_undo_redo,
    publish_pending_assets,
)
from modules.creative_document.editor_render import RenderError, render_document, render_document_png
from modules.creative_document.editor_view import document_to_view
from modules.creative_document.ids import sha256_bytes


creative_document_router = APIRouter()
_SESSION_TTL_SECONDS = 8 * 60 * 60
_MAX_PROJECT_PIXELS = 80_000_000
_OWNER_DESCRIPTOR_VERSION = 1


def _principal_id(grant: "SessionGrant", owner_key: str | None) -> str:
    """Return a stable, non-disclosing project principal.

    Authenticated users reconnect by Gradio username. Local unauthenticated
    sessions present a random browser-profile owner key so a fresh Gradio
    session in the same browser profile can reopen its own saved projects.
    """

    if grant.username:
        identity = "user:" + grant.username.strip().casefold()
    else:
        if not isinstance(owner_key, str) or not 32 <= len(owner_key) <= 128:
            raise HTTPException(status_code=401, detail={
                "code": "EDITOR_OWNER_REQUIRED",
                "message": "Editor project ownership credentials are required.",
            })
        if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in owner_key):
            raise HTTPException(status_code=401, detail={
                "code": "EDITOR_OWNER_REQUIRED",
                "message": "Editor project ownership credentials are required.",
            })
        identity = "browser:" + owner_key
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


@dataclass
class SessionGrant:
    session_hash: str
    username: str | None
    token_hash: str
    expires_at: float


class EditorSessionRegistry:
    """Ephemeral capabilities issued only by a Gradio load callback."""

    def __init__(self, *, ttl_seconds: int = _SESSION_TTL_SECONDS) -> None:
        self.ttl_seconds = ttl_seconds
        self._grants: dict[str, SessionGrant] = {}
        self._lock = threading.RLock()

    def issue(self, session_hash: str | None, username: str | None, *, remote_exposure: bool) -> dict[str, Any]:
        if not isinstance(session_hash, str) or not session_hash or len(session_hash) > 256:
            return {"enabled": False, "reason": "Gradio session is unavailable."}
        if remote_exposure and not username:
            return {"enabled": False, "reason": "Creative documents require an authenticated Gradio session when remotely exposed."}
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode("ascii")).hexdigest()
        grant = SessionGrant(session_hash, username, token_hash, time.time() + self.ttl_seconds)
        with self._lock:
            self._prune()
            self._grants[token_hash] = grant
        return {"enabled": True, "token": token, "session": session_hash, "expiresAt": int(grant.expires_at)}

    def _prune(self) -> None:
        now = time.time()
        for key in [key for key, value in self._grants.items() if value.expires_at <= now]:
            self._grants.pop(key, None)

    def authorize(self, request: Request) -> SessionGrant:
        authorization = request.headers.get("authorization", "")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(status_code=401, detail={"code": "EDITOR_SESSION_REQUIRED", "message": "Editor session is required."})
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        session_hash = request.headers.get("x-gradio-session")
        origin = request.headers.get("origin")
        host = request.headers.get("host")
        fetch_site = request.headers.get("sec-fetch-site")
        if fetch_site and fetch_site != "same-origin":
            raise HTTPException(status_code=403, detail={"code": "EDITOR_CROSS_ORIGIN", "message": "Cross-origin editor request was refused."})
        if origin:
            if not host:
                raise HTTPException(status_code=403, detail={"code": "EDITOR_ORIGIN_REQUIRED", "message": "Same-origin editor request is required."})
            origin_parts = urlsplit(origin)
            expected_origin = f"{request.url.scheme}://{host}"
            if origin_parts.path or origin_parts.query or origin_parts.fragment or origin.rstrip("/") != expected_origin:
                raise HTTPException(status_code=403, detail={"code": "EDITOR_CROSS_ORIGIN", "message": "Cross-origin editor request was refused."})
        elif fetch_site != "same-origin":
            # Browsers omit Origin on some same-origin GET fetches. Fetch
            # Metadata plus the unguessable session capability covers that
            # case; clients without either proof remain refused.
            raise HTTPException(status_code=403, detail={"code": "EDITOR_ORIGIN_REQUIRED", "message": "Same-origin editor request is required."})
        with self._lock:
            self._prune()
            grant = self._grants.get(token_hash)
        if grant is None:
            raise HTTPException(status_code=401, detail={"code": "EDITOR_SESSION_EXPIRED", "message": "Editor session expired. Reload the workspace."})
        if not session_hash or session_hash != grant.session_hash:
            raise HTTPException(status_code=403, detail={"code": "EDITOR_CROSS_SESSION", "message": "Editor session credentials do not match."})
        return grant


@dataclass
class ProjectHandle:
    document: Document
    store: ProjectStore
    project_path: Path
    committed_revision: int
    owner_principal: str
    lock: threading.RLock = field(default_factory=threading.RLock)
    recovery_notice: str | None = None


class CreativeDocumentRuntime:
    def __init__(self) -> None:
        self.sessions = EditorSessionRegistry()
        self._handles: dict[str, ProjectHandle] = {}
        self._handles_lock = threading.RLock()
        self._view_cache: dict[tuple[str, int], dict[str, Any]] = {}
        self._thumbnail_cache: dict[str, bytes] = {}
        self._cache_lock = threading.RLock()

    def projects_root(self) -> Path:
        configured = Path(modules.config.path_outputs).resolve()
        root = configured / "creative_documents"
        root.mkdir(parents=True, exist_ok=True)
        resolved = root.resolve()
        if os.path.commonpath([str(configured), str(resolved)]) != str(configured):
            raise HTTPException(status_code=503, detail={"code": "PROJECT_ROOT_UNAVAILABLE", "message": "Creative document storage is unavailable."})
        return resolved

    def project_path(self, document_id: str) -> Path:
        try:
            identity = validate_id(document_id, field="documentId")
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=404, detail={"code": "UNKNOWN_DOCUMENT", "message": "Document does not exist."}) from exc
        root = self.projects_root()
        path = root / f"{identity}.nexscene"
        if path.parent.resolve() != root or path.is_symlink():
            raise HTTPException(status_code=404, detail={"code": "UNKNOWN_DOCUMENT", "message": "Document does not exist."})
        return path

    def _owner_path(self, document_id: str) -> Path:
        root = self.projects_root()
        path = root / f"{document_id}.owner.json"
        if path.parent.resolve() != root or path.is_symlink():
            raise HTTPException(status_code=404, detail={"code": "UNKNOWN_DOCUMENT", "message": "Document does not exist."})
        return path

    def _read_owner(self, document_id: str) -> str:
        path = self._owner_path(document_id)
        try:
            if not path.is_file() or path.stat().st_size > 512:
                raise ValueError("owner record missing")
            value = json.loads(path.read_text(encoding="utf-8"))
            if (not isinstance(value, dict) or set(value) != {"version", "documentId", "principalId"}
                    or value["version"] != _OWNER_DESCRIPTOR_VERSION
                    or value["documentId"] != document_id
                    or not isinstance(value["principalId"], str)
                    or len(value["principalId"]) != 64
                    or any(character not in "0123456789abcdef" for character in value["principalId"])):
                raise ValueError("owner record invalid")
            return value["principalId"]
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=404, detail={"code": "UNKNOWN_DOCUMENT", "message": "Document does not exist."}) from exc

    def persist_owner(self, handle: ProjectHandle) -> None:
        """Persist ownership before publishing the first scene checkpoint."""

        path = self._owner_path(handle.document.document_id)
        if path.exists():
            if self._read_owner(handle.document.document_id) != handle.owner_principal:
                raise HTTPException(status_code=404, detail={"code": "UNKNOWN_DOCUMENT", "message": "Document does not exist."})
            return
        payload = json.dumps({
            "version": _OWNER_DESCRIPTOR_VERSION,
            "documentId": handle.document.document_id,
            "principalId": handle.owner_principal,
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")
        temporary = path.with_name(f".{path.name}.{secrets.token_hex(16)}.tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if self._read_owner(handle.document.document_id) != handle.owner_principal:
                    raise HTTPException(status_code=404, detail={"code": "UNKNOWN_DOCUMENT", "message": "Document does not exist."})
        except HTTPException:
            raise
        except OSError as exc:
            raise HTTPException(status_code=500, detail={"code": "OWNER_RECORD_FAILED", "message": "Document access could not be saved."}) from exc
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def _new_handle(self, document: Document, path: Path, store: ProjectStore, *, owner_principal: str,
                    committed_revision: int | None = None,
                    recovery_notice: str | None = None) -> ProjectHandle:
        committed = document.current_revision if committed_revision is None else committed_revision
        handle = ProjectHandle(document, store, path, committed, owner_principal, recovery_notice=recovery_notice)
        with self._handles_lock:
            self._handles[document.document_id] = handle
        return handle

    def create(self, width: int, height: int, owner_principal: str, name: str = "Creative Document") -> ProjectHandle:
        if type(width) is not int or type(height) is not int or width <= 0 or height <= 0 or width * height > _MAX_PROJECT_PIXELS:
            raise HTTPException(status_code=422, detail={"code": "INVALID_DOCUMENT_SIZE", "message": "Document dimensions exceed supported limits."})
        if not isinstance(name, str) or not name.strip():
            raise HTTPException(status_code=422, detail={"code": "INVALID_DOCUMENT_NAME", "message": "Document name must contain text."})
        document_id = make_id("doc")
        layer_id = make_id("layer")
        layer = LayerRecord(layer_id, name.strip()[:120], "paint")
        document = Document(document_id, width, height, root_layer_ids=[layer_id], layers={layer_id: layer})
        document.validate()
        path = self.project_path(document_id)
        if path.exists():
            raise HTTPException(status_code=409, detail={"code": "DOCUMENT_ID_COLLISION", "message": "Could not create document."})
        store = ProjectStore(path)
        # The empty scene remains an in-memory revision until explicit Save.
        return self._new_handle(document, path, store, owner_principal=owner_principal, committed_revision=-1)

    def get(self, document_id: str, owner_principal: str) -> ProjectHandle:
        path = self.project_path(document_id)
        with self._handles_lock:
            handle = self._handles.get(document_id)
            if handle is not None:
                if not secrets.compare_digest(handle.owner_principal, owner_principal):
                    raise HTTPException(status_code=404, detail={"code": "UNKNOWN_DOCUMENT", "message": "Document does not exist."})
                return handle
            if not path.is_dir():
                raise HTTPException(status_code=404, detail={"code": "UNKNOWN_DOCUMENT", "message": "Document does not exist."})
            actual_owner = self._read_owner(document_id)
            if not secrets.compare_digest(actual_owner, owner_principal):
                raise HTTPException(status_code=404, detail={"code": "UNKNOWN_DOCUMENT", "message": "Document does not exist."})
            store = ProjectStore(path)
            try:
                document = store.open()
            except Exception as exc:
                raise HTTPException(status_code=404, detail={"code": "DOCUMENT_UNAVAILABLE", "message": "Document could not be opened."}) from exc
            if document.document_id != document_id:
                raise HTTPException(status_code=404, detail={"code": "UNKNOWN_DOCUMENT", "message": "Document does not exist."})
            notice = "Recovered the latest fully committed revision." if document.current_revision > 0 else None
            return self._new_handle(document, path, store, owner_principal=actual_owner, recovery_notice=notice)

    def force_open(self, document_id: str, *, owner_principal: str, discard_unsaved: bool) -> ProjectHandle:
        handle = self.get(document_id, owner_principal)
        with handle.lock:
            if handle.document.current_revision != handle.committed_revision and not discard_unsaved:
                raise HTTPException(status_code=409, detail={"code": "UNSAVED_CHANGES", "message": "Save or explicitly discard unsaved changes before reopening."})
            try:
                document = handle.store.open()
            except Exception as exc:
                raise HTTPException(status_code=404, detail={"code": "DOCUMENT_UNAVAILABLE", "message": "Document could not be opened."}) from exc
            if document.document_id != document_id:
                raise HTTPException(status_code=404, detail={"code": "UNKNOWN_DOCUMENT", "message": "Document does not exist."})
            handle.document = document
            handle.committed_revision = document.current_revision
            handle.recovery_notice = "Reopened the latest committed revision."
            with self._cache_lock:
                for key in [key for key in self._view_cache if key[0] == document_id]:
                    self._view_cache.pop(key, None)
            return handle

    def view(self, handle: ProjectHandle) -> dict[str, Any]:
        document = handle.document
        key = (document.document_id, document.current_revision)
        with self._cache_lock:
            view = self._view_cache.get(key)
            if view is not None:
                return dict(view)
        view = document_to_view(document)
        for asset in document.assets.values():
            if asset.external_uri is not None and asset.external_status != "embedded":
                continue
            try:
                verified = handle.store.assets.verify(asset)
            except Exception:
                verified = False
            if not verified:
                layer_id = next((layer.layer_id for layer in document.layers.values()
                                 if asset.asset_id in layer.asset_ids
                                 or any(document.objects[object_id].asset_id == asset.asset_id
                                        for object_id in layer.object_ids)), None)
                issue = {
                    "code": "ASSET_MISSING_OR_MISMATCH",
                    "severity": "error",
                    "layerId": layer_id,
                    "assetId": asset.asset_id,
                    "message": "An embedded image is missing or does not match its recorded content hash.",
                }
                view["rendererIssues"].append(issue)
                layer_view = next((item for item in view["layers"] if item["id"] == layer_id), None)
                if layer_view is not None:
                    layer_view["rendererIssues"].append(issue)
                object_view = next((item for item in view["objects"] if item.get("assetId") == asset.asset_id), None)
                if object_view is not None:
                    object_view["rendererIssues"].append(issue)
        view["committedRevision"] = handle.committed_revision
        view["dirty"] = document.current_revision != handle.committed_revision
        view["recoveryNotice"] = handle.recovery_notice
        with self._cache_lock:
            self._view_cache[key] = view
            # Keep only the latest eight immutable snapshots per document.
            revisions = sorted(revision for doc_id, revision in self._view_cache if doc_id == document.document_id)
            for revision in revisions[:-8]:
                self._view_cache.pop((document.document_id, revision), None)
        return dict(view)

    def asset_store(self, handle: ProjectHandle) -> AssetStore:
        return handle.store.assets

    def cache_preview(self, handle: ProjectHandle, encoded: bytes) -> None:
        # A returned render is derived data only; this cache is never written
        # into the scene manifest or used as project authority.
        key = f"{handle.document.document_id}:{handle.document.current_revision}:{sha256_bytes(encoded)}"
        with self._cache_lock:
            self._thumbnail_cache[key] = encoded


creative_document_runtime = CreativeDocumentRuntime()


def issue_editor_session(request: Any, *, remote_exposure: bool) -> str:
    """Gradio load callback; the returned token is scoped to this Gradio session."""

    session_hash = getattr(request, "session_hash", None)
    username = getattr(request, "username", None)
    capability = creative_document_runtime.sessions.issue(session_hash, username, remote_exposure=remote_exposure)
    return json.dumps(capability, separators=(",", ":"))


def _authorized(request: Request) -> SessionGrant:
    return creative_document_runtime.sessions.authorize(request)


def _handle(document_id: str, request: Request, grant: SessionGrant | None = None) -> ProjectHandle:
    session = grant or _authorized(request)
    principal = _principal_id(session, request.headers.get("x-editor-owner-key"))
    return creative_document_runtime.get(document_id, principal)


def _action_error(exc: EditorActionError) -> HTTPException:
    if exc.code == "STALE_DOCUMENT_REVISION":
        return HTTPException(status_code=409, detail={
            "status": "conflict", "code": exc.code, "message": str(exc),
            "currentRevision": exc.current_revision, "refreshHint": "reload-document-view",
        })
    conflict = exc.code in {"STALE_SELECTION_REVISION"}
    return HTTPException(status_code=409 if conflict else 422, detail={
        "status": "refused", "code": exc.code, "message": str(exc),
        "currentRevision": exc.current_revision, "refreshHint": "reload-document-view" if conflict else None,
    })


@creative_document_router.get("/creative_document_api/vendor/konva-10.6.0.js")
async def get_pinned_konva(request: Request) -> Response:
    # The public vendor artifact contains no user/project data. Its route is
    # same-origin and its exact identity is enforced in the response headers.
    path = Path(__file__).resolve().parent.parent / "javascript" / "vendor" / "konva" / "10.6.0" / "konva.min.js"
    try:
        contents = path.read_bytes()
    except OSError as exc:
        raise HTTPException(status_code=503, detail="Editor renderer is unavailable.") from exc
    expected = "C03625663B3F4B79C64AECD5671F1A83A37AA2D1E276005B186E42E7D8DBA5A1"
    if sha256_bytes(contents).upper() != expected:
        raise HTTPException(status_code=503, detail="Editor renderer integrity check failed.")
    digest_b64 = __import__("base64").b64encode(bytes.fromhex(expected)).decode("ascii")
    return Response(contents, media_type="text/javascript", headers={
        "Cache-Control": "public, max-age=31536000, immutable",
        "X-Content-Type-Options": "nosniff",
        "ETag": f'"sha256-{expected}"',
        "X-Konva-Version": "10.6.0",
        "X-Konva-SRI": f"sha256-{digest_b64}",
    })


@creative_document_router.post("/creative_document_api/documents")
async def create_document(request: Request, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    grant = _authorized(request)
    principal = _principal_id(grant, request.headers.get("x-editor-owner-key"))
    handle = creative_document_runtime.create(
        payload.get("width"), payload.get("height"), principal, payload.get("name", "Creative Document")
    )
    return {"status": "created", "documentId": handle.document.document_id,
            "revision": handle.document.current_revision, "view": creative_document_runtime.view(handle)}


@creative_document_router.post("/creative_document_api/documents/{document_id}/open")
async def open_document(document_id: str, request: Request, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    grant = _authorized(request)
    principal = _principal_id(grant, request.headers.get("x-editor-owner-key"))
    handle = creative_document_runtime.force_open(
        document_id, owner_principal=principal, discard_unsaved=payload.get("discardUnsaved") is True
    )
    return {"status": "opened", "documentId": document_id,
            "revision": handle.document.current_revision, "view": creative_document_runtime.view(handle)}


@creative_document_router.get("/creative_document_api/documents/{document_id}/view")
async def get_document_view(document_id: str, request: Request) -> dict[str, Any]:
    handle = _handle(document_id, request)
    with handle.lock:
        return creative_document_runtime.view(handle)


@creative_document_router.post("/creative_document_api/documents/{document_id}/actions")
async def apply_document_action(document_id: str, request: Request, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    grant = _authorized(request)
    handle = _handle(document_id, request, grant)
    with handle.lock:
        payload = dict(payload)
        payload.setdefault("documentId", document_id)
        if payload.get("documentId") != document_id:
            raise HTTPException(status_code=404, detail={"code": "UNKNOWN_DOCUMENT", "message": "Document does not exist."})
        actor_id = actor_for_session(grant.username)
        try:
            if payload.get("actionType") == "undo":
                prepared = prepare_undo_redo(handle.document, payload, actor_id=actor_id, redo=False)
            elif payload.get("actionType") == "redo":
                prepared = prepare_undo_redo(handle.document, payload, actor_id=actor_id, redo=True)
            else:
                prepared = prepare_action(handle.document, handle.store.assets, payload, actor_id=actor_id)
            publish_pending_assets(handle.store.assets, prepared)
        except EditorActionError as exc:
            raise _action_error(exc) from exc
        except (SchemaValidationError, HistoryValidationError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail={"status": "refused", "code": "VALIDATION_FAILED", "message": str(exc)}) from exc
        handle.document = prepared.document
        handle.recovery_notice = None
        with creative_document_runtime._cache_lock:
            creative_document_runtime._view_cache.pop((document_id, handle.document.current_revision), None)
        receipt = dict(prepared.receipt)
        receipt["dirty"] = handle.document.current_revision != handle.committed_revision
        receipt["committedRevision"] = handle.committed_revision
        return receipt


@creative_document_router.post("/creative_document_api/documents/{document_id}/assets/import")
async def import_image(
    document_id: str,
    request: Request,
    file: UploadFile = File(...),
    expected_revision: int = Form(...),
    actor_kind: str = Form("director"),
    as_guide: bool = Form(False),
    semantic_role: str | None = Form(None),
) -> dict[str, Any]:
    grant = _authorized(request)
    handle = _handle(document_id, request, grant)
    raw = await file.read(100 * 1024 * 1024 + 1)
    payload = {"documentId": document_id, "expectedRevision": expected_revision,
               "actorKind": actor_kind, "actionType": "import_image",
               "data": {"filename": file.filename or "import.png", "asGuide": as_guide,
                        "semanticRole": semantic_role}}
    with handle.lock:
        try:
            prepared = prepare_raster_import(handle.document, handle.store.assets, payload, raw,
                                             actor_id=actor_for_session(grant.username), filename=file.filename or "import.png")
            publish_pending_assets(handle.store.assets, prepared)
        except EditorActionError as exc:
            raise _action_error(exc) from exc
        handle.document = prepared.document
        handle.recovery_notice = None
        receipt = dict(prepared.receipt)
        receipt["dirty"] = handle.document.current_revision != handle.committed_revision
        receipt["committedRevision"] = handle.committed_revision
        return receipt


@creative_document_router.post("/creative_document_api/documents/{document_id}/save")
async def save_document(document_id: str, request: Request, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    handle = _handle(document_id, request)
    with handle.lock:
        expected = payload.get("expectedRevision")
        if type(expected) is not int or expected != handle.document.current_revision:
            raise HTTPException(status_code=409, detail={"status": "conflict", "code": "STALE_DOCUMENT_REVISION",
                                                         "currentRevision": handle.document.current_revision,
                                                         "refreshHint": "reload-document-view"})
        try:
            creative_document_runtime.persist_owner(handle)
            if handle.document.current_revision != handle.committed_revision:
                handle.store.save(handle.document, checkpoint=True)
                handle.committed_revision = handle.document.current_revision
            else:
                handle.store.checkpoint(handle.document)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail={"status": "refused", "code": "SAVE_FAILED",
                                                         "message": "The document could not be saved."}) from exc
        handle.recovery_notice = "Explicit save checkpoint is available."
        with creative_document_runtime._cache_lock:
            creative_document_runtime._view_cache.pop((document_id, handle.document.current_revision), None)
        return {"status": "saved", "documentId": document_id,
                "revision": handle.document.current_revision,
                "committedRevision": handle.committed_revision,
                "dirty": False, "checkpoint": True, "code": None}


@creative_document_router.get("/creative_document_api/documents/{document_id}/assets/{asset_id}/content")
async def get_asset(document_id: str, asset_id: str, request: Request) -> Response:
    handle = _handle(document_id, request)
    with handle.lock:
        record = handle.document.assets.get(asset_id)
        if record is None:
            raise HTTPException(status_code=404, detail={"code": "UNKNOWN_ASSET", "message": "Asset does not exist."})
        if record.external_uri is not None:
            raise HTTPException(status_code=409, detail={"code": "EXTERNAL_ASSET_UNAVAILABLE", "message": "External asset must be embedded before browser access."})
        try:
            content = handle.store.assets.read_bytes(record)
        except Exception as exc:
            raise HTTPException(status_code=404, detail={"code": "ASSET_UNAVAILABLE", "message": "Asset content is unavailable."}) from exc
    return Response(content, media_type=record.media_type,
                    headers={"ETag": f'"{record.content_hash}"', "X-Content-Type-Options": "nosniff",
                             "Cache-Control": "private, max-age=31536000, immutable"})


@creative_document_router.get("/creative_document_api/documents/{document_id}/assets/{asset_id}/thumbnail")
async def get_asset_thumbnail(document_id: str, asset_id: str, request: Request) -> Response:
    handle = _handle(document_id, request)
    with handle.lock:
        record = handle.document.assets.get(asset_id)
        if record is None:
            raise HTTPException(status_code=404, detail={"code": "UNKNOWN_ASSET", "message": "Asset does not exist."})
        key = f"thumb:{record.content_hash}:384"
        with creative_document_runtime._cache_lock:
            cached = creative_document_runtime._thumbnail_cache.get(key)
        if cached is None:
            if record.external_uri is not None:
                raise HTTPException(status_code=409, detail={"code": "EXTERNAL_ASSET_UNAVAILABLE", "message": "External asset must be embedded before browser access."})
            try:
                raw = handle.store.assets.read_bytes(record)
                with Image.open(io.BytesIO(raw)) as opened:
                    image = opened.convert("RGBA")
                    image.thumbnail((384, 384), Image.Resampling.LANCZOS)
                    output = io.BytesIO()
                    image.save(output, format="PNG", optimize=False, compress_level=9)
                    cached = output.getvalue()
            except Exception as exc:
                raise HTTPException(status_code=404, detail={"code": "THUMBNAIL_UNAVAILABLE", "message": "Asset thumbnail is unavailable."}) from exc
            with creative_document_runtime._cache_lock:
                creative_document_runtime._thumbnail_cache[key] = cached
    return Response(cached, media_type="image/png", headers={
        "ETag": f'"{record.content_hash}-thumb-384"', "X-Content-Type-Options": "nosniff",
        "Cache-Control": "private, max-age=31536000, immutable",
    })


@creative_document_router.get("/creative_document_api/documents/{document_id}/preview")
async def get_preview(document_id: str, request: Request) -> Response:
    handle = _handle(document_id, request)
    with handle.lock:
        document = handle.document
        snapshot_document_id = document.document_id
        revision = document.current_revision
        revision_digest = document.revision_digest or ""
        try:
            image = render_document(document, lambda asset_id: handle.store.assets.read_bytes(document.assets[asset_id]))
            image.thumbnail((1600, 1200), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            image.save(output, format="PNG", optimize=False, compress_level=9)
            content = output.getvalue()
        except RenderError as exc:
            raise HTTPException(status_code=409, detail={"code": exc.code, "message": str(exc)}) from exc
        except Exception as exc:
            raise HTTPException(status_code=409, detail={"code": "PREVIEW_UNAVAILABLE", "message": "Preview could not be rendered."}) from exc
    return Response(content, media_type="image/png", headers={
        "Cache-Control": "no-store",
        "X-Document-ID": snapshot_document_id,
        "X-Document-Revision": str(revision),
        "X-Document-Revision-Digest": revision_digest,
    })


@creative_document_router.get("/creative_document_api/documents/{document_id}/export")
async def export_composite(document_id: str, request: Request) -> Response:
    handle = _handle(document_id, request)
    with handle.lock:
        document = handle.document
        snapshot_document_id = document.document_id
        revision = document.current_revision
        revision_digest = document.revision_digest or ""
        try:
            content = render_document_png(document, lambda asset_id: handle.store.assets.read_bytes(document.assets[asset_id]))
        except RenderError as exc:
            raise HTTPException(status_code=409, detail={"code": exc.code, "message": str(exc)}) from exc
        except Exception as exc:
            raise HTTPException(status_code=409, detail={"code": "EXPORT_UNAVAILABLE", "message": "Composite export could not be rendered."}) from exc
    return Response(content, media_type="image/png", headers={
        "Content-Disposition": f'attachment; filename="creative-document-{snapshot_document_id}-r{revision}.png"',
        "X-Document-ID": snapshot_document_id,
        "X-Document-Revision": str(revision),
        "X-Document-Revision-Digest": revision_digest,
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
    })
