import os
import io
import json
import zipfile
import urllib.request
import urllib.parse
import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import JSONResponse, FileResponse, Response
import modules.config
import modules.util
from PIL import Image

staging_router = APIRouter()


def get_staging_dir():
    staging_dir = os.path.join(modules.config.path_outputs, "staging")
    os.makedirs(staging_dir, exist_ok=True)
    return staging_dir


def _gimp_target_file():
    return os.path.join(get_staging_dir(), ".gimp_target.txt")


def _gimp_queue_file():
    return os.path.join(get_staging_dir(), ".gimp_queue.json")


def _markers_manifest_file():
    return os.path.join(get_staging_dir(), ".markers.json")


def _read_gimp_target_name():
    target_file = _gimp_target_file()
    if not os.path.exists(target_file):
        return None
    try:
        with open(target_file, "r", encoding="utf-8") as f:
            name = f.read().strip()
        return name or None
    except Exception:
        return None


def _read_gimp_queue_names():
    queue_file = _gimp_queue_file()
    names = []

    if os.path.exists(queue_file):
        try:
            with open(queue_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for name in data:
                    if not isinstance(name, str):
                        continue
                    clean = name.strip()
                    if clean and clean not in names:
                        names.append(clean)
        except Exception:
            names = []

    if names:
        return names

    legacy_target = _read_gimp_target_name()
    return [legacy_target] if legacy_target else []


def _write_gimp_queue_names(names):
    queue_file = _gimp_queue_file()
    cleaned = []
    for name in names:
        if not isinstance(name, str):
            continue
        clean = name.strip()
        if clean and clean not in cleaned:
            cleaned.append(clean)

    if cleaned:
        with open(queue_file, "w", encoding="utf-8") as f:
            json.dump(cleaned, f)
    elif os.path.exists(queue_file):
        os.remove(queue_file)

    legacy_target = _gimp_target_file()
    if os.path.exists(legacy_target):
        os.remove(legacy_target)


_STAGING_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
_FORMAT_TO_EXTENSION = {
    "PNG": ".png",
    "JPEG": ".jpg",
    "WEBP": ".webp",
}
_MARKER_ICON_OPTIONS = {"star", "flag", "circle", "triangle", "pin", "bookmark"}
_MARKER_COLOR_OPTIONS = {"red", "amber", "green", "blue", "violet", "gray"}


def _read_staging_markers():
    manifest_path = _markers_manifest_file()
    if not os.path.exists(manifest_path):
        return {}

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    sanitized = {}
    for name, marker in data.items():
        if not isinstance(name, str):
            continue
        clean_name = name.strip()
        clean_marker = _sanitize_marker_payload(marker)
        if clean_name and clean_marker is not None:
            sanitized[clean_name] = clean_marker
    return sanitized


def _write_staging_markers(markers):
    manifest_path = _markers_manifest_file()
    sanitized = {}
    for name, marker in (markers or {}).items():
        if not isinstance(name, str):
            continue
        clean_name = name.strip()
        clean_marker = _sanitize_marker_payload(marker)
        if clean_name and clean_marker is not None:
            sanitized[clean_name] = clean_marker

    if sanitized:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(sanitized, f, ensure_ascii=True, indent=2, sort_keys=True)
    elif os.path.exists(manifest_path):
        os.remove(manifest_path)


def _sanitize_marker_payload(marker):
    if marker is None:
        return None
    if not isinstance(marker, dict):
        return None

    icon = str(marker.get("icon", "")).strip().lower()
    color = str(marker.get("color", "")).strip().lower()
    label = str(marker.get("label", "")).strip()

    if icon not in _MARKER_ICON_OPTIONS:
        return None
    if color not in _MARKER_COLOR_OPTIONS:
        return None

    if len(label) > 48:
        label = label[:48].rstrip()

    return {
        "icon": icon,
        "color": color,
        "label": label,
    }


def _guess_extension(source_name: str | None, detected_format: str | None) -> str:
    suffix = Path(source_name or "").suffix.lower()
    if suffix in _STAGING_EXTENSIONS:
        return suffix
    return _FORMAT_TO_EXTENSION.get(str(detected_format or "").upper(), ".png")


def _stage_bytes(staging_dir: str, contents: bytes, source_name: str | None = None) -> tuple[str, str]:
    try:
        with Image.open(io.BytesIO(contents)) as img:
            detected_format = img.format
            img.verify()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image payload: {exc}") from exc

    import datetime

    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S_%f")
    extension = _guess_extension(source_name, detected_format)
    filename = f"staged_{time_str}{extension}"
    filepath = os.path.join(staging_dir, filename)
    with open(filepath, "wb") as f:
        f.write(contents)
    return filename, filepath


def _safe_rooted_file_path(root_dir: str, relative_name: str) -> str:
    filepath = os.path.abspath(os.path.join(root_dir, str(relative_name or "")))
    root_dir = os.path.abspath(root_dir)
    if filepath != root_dir and not filepath.startswith(root_dir + os.sep):
        raise HTTPException(status_code=403, detail="Forbidden")
    return filepath


def _read_local_image_file(filepath: str, *, missing_detail: str) -> bytes:
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=400, detail=missing_detail)
    with open(filepath, "rb") as f:
        return f.read()


@staging_router.get("/staging_api/images")
async def list_staging_images():
    """Returns a list of image URLs currently in the staging directory."""
    staging_dir = get_staging_dir()
    files = []
    markers = _read_staging_markers()

    try:
        entries = sorted(
            [e for e in os.scandir(staging_dir) if e.is_file() and e.name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))],
            key=lambda e: e.stat().st_mtime,
            reverse=True,
        )
        for entry in entries:
            files.append({
                "name": entry.name,
                "url": f"/staging_api/image/{entry.name}",
                "marker": markers.get(entry.name),
            })
    except Exception as e:
        print(f"Error reading staging dir: {e}")

    return JSONResponse(content={"images": files, "gimp_queue": _read_gimp_queue_names()})


@staging_router.post("/staging_api/upload")
async def upload_staging_image(
    file: UploadFile = File(None),
    url: str = Form(None)
):
    """Accepts either a File upload or an existing URL/Base64 to save to staging."""
    staging_dir = get_staging_dir()

    try:
        contents = None
        source_name = None
        if file is not None:
            contents = await file.read()
            source_name = file.filename

        elif url is not None:
            parsed_url = urllib.parse.urlparse(url)
            parsed_path = parsed_url.path or ""
            if url.startswith("data:image"):
                header, encoded = url.split(",", 1)
                import base64
                contents = base64.b64decode(encoded)
                mime = header.split(";", 1)[0].split(":", 1)[-1].strip().lower()
                extension = {
                    "image/png": ".png",
                    "image/jpeg": ".jpg",
                    "image/webp": ".webp",
                }.get(mime, ".png")
                source_name = f"staged_upload{extension}"
            elif parsed_path.startswith("/file=") or "/file=" in url:
                filepath = url.split("/file=", 1)[1].split("?")[0]
                filepath = urllib.parse.unquote(filepath)
                if os.path.exists(filepath):
                    with open(filepath, "rb") as f:
                        contents = f.read()
                    source_name = os.path.basename(filepath)
                else:
                    raise HTTPException(status_code=400, detail="Local file not found")
            elif parsed_path.startswith("/image_api/image/"):
                parts = parsed_path[len("/image_api/image/"):].split("/")
                if len(parts) != 2:
                    raise HTTPException(status_code=400, detail="Invalid workspace image URL")
                workspace_id = urllib.parse.unquote(parts[0])
                filename = urllib.parse.unquote(parts[1])
                from modules.image_api import get_workspace_dir
                workspace_dir = get_workspace_dir(workspace_id)
                filepath = _safe_rooted_file_path(workspace_dir, filename)
                contents = _read_local_image_file(filepath, missing_detail="Workspace file not found")
                source_name = os.path.basename(filepath)
            elif parsed_path.startswith("/staging_api/image/"):
                parts = parsed_path[len("/staging_api/image/"):].split("/")
                if len(parts) != 1:
                    raise HTTPException(status_code=400, detail="Invalid staging image URL")
                filename = urllib.parse.unquote(parts[0])
                filepath = _safe_rooted_file_path(get_staging_dir(), filename)
                contents = _read_local_image_file(filepath, missing_detail="Staged file not found")
                source_name = os.path.basename(filepath)
            elif parsed_path == "/runtime_surface_api/preview_image":
                from modules import runtime_surface_state
                filepath = runtime_surface_state.get_preview_image_path()
                if filepath and os.path.isfile(filepath):
                    with open(filepath, "rb") as f:
                        contents = f.read()
                    source_name = os.path.basename(filepath)
                else:
                    preview_bytes, _, _ = runtime_surface_state.get_preview_image_bytes()
                    if preview_bytes:
                        contents = preview_bytes
                        source_name = "preview.png"
                    else:
                        raise HTTPException(status_code=400, detail="Preview image not found")
            elif parsed_path.startswith("/runtime_surface_api/completed_image/"):
                parts = parsed_path[len("/runtime_surface_api/completed_image/"):].split("/")
                if len(parts) != 2:
                    raise HTTPException(status_code=400, detail="Invalid completed image URL")
                task_id = urllib.parse.unquote(parts[0])
                image_index = int(urllib.parse.unquote(parts[1]))
                from modules import runtime_surface_state
                filepath = runtime_surface_state.get_completed_image_path(task_id, image_index)
                if filepath and os.path.isfile(filepath):
                    with open(filepath, "rb") as f:
                        contents = f.read()
                    source_name = os.path.basename(filepath)
                else:
                    raise HTTPException(status_code=400, detail="Completed image not found")
            elif url.startswith("file://"):
                filepath = url.replace("file:///", "").replace("file://", "")
                filepath = urllib.parse.unquote(filepath)
                if os.path.exists(filepath):
                    with open(filepath, "rb") as f:
                        contents = f.read()
                    source_name = os.path.basename(filepath)
                else:
                    raise HTTPException(status_code=400, detail="Local file not found")
            elif url.startswith("http"):
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as response:
                    contents = response.read()
                parsed_url = urllib.parse.urlparse(url)
                source_name = os.path.basename(parsed_url.path) or "staged_remote.png"
            else:
                raise HTTPException(status_code=400, detail="Invalid URL format")
        else:
            raise HTTPException(status_code=400, detail="Must provide file or url")

        if contents:
            filename, filepath = _stage_bytes(staging_dir, contents, source_name)
            return JSONResponse(content={
                "status": "success",
                "file": filename,
                "filepath": filepath,
                "url": f"/staging_api/image/{filename}",
            })

        return JSONResponse(content={"status": "error", "message": "Failed to process image"})

    except HTTPException:
        raise
    except Exception as e:
        print(f"Staging upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@staging_router.delete("/staging_api/delete")
async def delete_staging_image(name: str):
    """Deletes a specific image from the staging directory."""
    staging_dir = get_staging_dir()
    filepath = os.path.join(staging_dir, name)

    if not os.path.abspath(filepath).startswith(os.path.abspath(staging_dir)):
        raise HTTPException(status_code=403, detail="Forbidden")

    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            remaining_queue = [queued for queued in _read_gimp_queue_names() if queued != name]
            _write_gimp_queue_names(remaining_queue)
            markers = _read_staging_markers()
            if name in markers:
                del markers[name]
                _write_staging_markers(markers)
            return JSONResponse(content={"status": "success"})
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        print(f"Staging delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@staging_router.post("/staging_api/clear")
async def clear_staging_images():
    """Clears all images from the staging directory."""
    staging_dir = get_staging_dir()
    try:
        import shutil
        for filename in os.listdir(staging_dir):
            file_path = os.path.join(staging_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
        return JSONResponse(content={"status": "success"})
    except Exception as e:
        print(f"Staging clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@staging_router.post("/staging_api/gimp_target")
async def set_gimp_target(name: str):
    """Toggles an image in the queued GIMP import set."""
    staging_dir = get_staging_dir()
    filepath = os.path.join(staging_dir, name)

    if not os.path.abspath(filepath).startswith(os.path.abspath(staging_dir)):
        raise HTTPException(status_code=403, detail="Forbidden")

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        queue = _read_gimp_queue_names()
        if name in queue:
            queue = [queued for queued in queue if queued != name]
            queued = False
        else:
            queue.append(name)
            queued = True

        _write_gimp_queue_names(queue)
        return JSONResponse(content={"status": "success", "queue": queue, "queued": queued})
    except Exception as e:
        print(f"Staging GIMP queue error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@staging_router.post("/staging_api/marker")
async def set_staging_marker(payload: dict = Body(...)):
    name = payload.get("name") if isinstance(payload, dict) else None
    marker = payload.get("marker") if isinstance(payload, dict) else None

    if not isinstance(name, str) or not name.strip():
        raise HTTPException(status_code=400, detail="Missing staged image name")

    name = name.strip()
    _, filepath = _safe_staging_path(name)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    sanitized_marker = _sanitize_marker_payload(marker)
    if marker is not None and sanitized_marker is None:
        raise HTTPException(status_code=400, detail="Invalid marker payload")

    try:
        markers = _read_staging_markers()
        if sanitized_marker is None:
            markers.pop(name, None)
        else:
            markers[name] = sanitized_marker
        _write_staging_markers(markers)
        return JSONResponse(content={"status": "success", "marker": sanitized_marker})
    except Exception as e:
        print(f"Staging marker error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@staging_router.get("/staging_api/gimp_target")
async def get_gimp_target():
    """Returns all queued GIMP images as a ZIP bundle and clears the queue."""
    staging_dir = get_staging_dir()
    queue = _read_gimp_queue_names()
    if not queue:
        raise HTTPException(status_code=404, detail="No queued GIMP images")

    bundle = io.BytesIO()
    included_names = []

    try:
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
            for index, name in enumerate(queue, start=1):
                filepath = os.path.join(staging_dir, name)
                if not os.path.abspath(filepath).startswith(os.path.abspath(staging_dir)):
                    continue
                if not os.path.exists(filepath):
                    continue

                arcname = f"{index:02d}_{os.path.basename(name)}"
                zip_file.write(filepath, arcname=arcname)
                included_names.append(name)

        if not included_names:
            raise HTTPException(status_code=404, detail="Queued GIMP images no longer exist")

        _write_gimp_queue_names([])
        headers = {
            "Content-Disposition": 'attachment; filename="fooocus_nex_gimp_queue.zip"',
            "X-Fooocus-Queue-Count": str(len(included_names)),
        }
        return Response(content=bundle.getvalue(), media_type="application/zip", headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Staging GIMP retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _safe_staging_path(name: str):
    staging_dir = get_staging_dir()
    filepath = os.path.join(staging_dir, name)
    if not os.path.abspath(filepath).startswith(os.path.abspath(staging_dir)):
        raise HTTPException(status_code=403, detail="Forbidden")
    return staging_dir, filepath


@staging_router.get("/staging_api/image/{name}")
async def get_staging_image(name: str):
    """Serves a specific image from the staging directory."""
    staging_dir, filepath = _safe_staging_path(name)

    if os.path.exists(filepath):
        return FileResponse(filepath)

    raise HTTPException(status_code=404, detail="Image not found")
