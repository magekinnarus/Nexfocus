import io
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from modules.staging_api import staging_router, get_staging_dir


app = FastAPI()
app.include_router(staging_router)
client = TestClient(app)


def _cleanup_staged_file(path: str) -> None:
    if path and os.path.exists(path):
        os.remove(path)


def _cleanup_marker_manifest() -> None:
    manifest = os.path.join(get_staging_dir(), ".markers.json")
    if os.path.exists(manifest):
        os.remove(manifest)


def test_staging_upload_preserves_png_metadata_and_rgba():
    img = Image.new("RGBA", (32, 32), color=(10, 20, 30, 128))
    info = PngInfo()
    info.add_text("fooocus_scheme", "fooocus_nex")
    info.add_text("parameters", '{"prompt": "hello"}')

    image_bytes = io.BytesIO()
    img.save(image_bytes, format="PNG", pnginfo=info)

    response = client.post(
        "/staging_api/upload",
        files={"file": ("gimp_source.png", image_bytes.getvalue(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["file"].endswith(".png")

    saved_path = payload["filepath"]
    assert Path(saved_path).parent == Path(get_staging_dir())
    assert os.path.exists(saved_path)

    try:
        with Image.open(saved_path) as restored:
            assert restored.mode == "RGBA"
            assert restored.info.get("fooocus_scheme") == "fooocus_nex"
            assert restored.info.get("parameters") == '{"prompt": "hello"}'
    finally:
        _cleanup_staged_file(saved_path)


def test_staging_marker_roundtrip_and_delete_cleanup():
    _cleanup_marker_manifest()
    img = Image.new("RGBA", (16, 16), color=(200, 100, 50, 255))
    image_bytes = io.BytesIO()
    img.save(image_bytes, format="PNG")

    upload_response = client.post(
        "/staging_api/upload",
        files={"file": ("marker_source.png", image_bytes.getvalue(), "image/png")},
    )

    assert upload_response.status_code == 200
    upload_payload = upload_response.json()
    saved_name = upload_payload["file"]
    saved_path = upload_payload["filepath"]

    try:
        marker_payload = {
            "name": saved_name,
            "marker": {
                "icon": "flag",
                "color": "blue",
                "label": "best hand",
            },
        }
        marker_response = client.post("/staging_api/marker", json=marker_payload)
        assert marker_response.status_code == 200
        assert marker_response.json()["marker"] == marker_payload["marker"]

        list_response = client.get("/staging_api/images")
        assert list_response.status_code == 200
        images = list_response.json()["images"]
        staged_entry = next(item for item in images if item["name"] == saved_name)
        assert staged_entry["marker"] == marker_payload["marker"]

        delete_response = client.delete(f"/staging_api/delete?name={saved_name}")
        assert delete_response.status_code == 200

        post_delete_images = client.get("/staging_api/images").json()["images"]
        assert all(item["name"] != saved_name for item in post_delete_images)

        manifest = os.path.join(get_staging_dir(), ".markers.json")
        if os.path.exists(manifest):
            with open(manifest, "r", encoding="utf-8") as f:
                assert saved_name not in f.read()
    finally:
        _cleanup_staged_file(saved_path)
        _cleanup_marker_manifest()


def test_staging_upload_local_endpoints_direct_resolution(monkeypatch):
    import shutil
    from modules.image_api import get_workspace_dir
    from modules import runtime_surface_state

    # Setup directories
    ws_id = "test_local_ws"
    ws_dir = get_workspace_dir(ws_id)
    shutil.rmtree(ws_dir, ignore_errors=True)
    ws_dir = get_workspace_dir(ws_id)

    # 1. Test /image_api/image/
    ws_img_path = os.path.join(ws_dir, "base.png")
    img = Image.new("RGBA", (10, 10), color=(0, 255, 0, 255))
    img.save(ws_img_path)

    response = client.post(
        "/staging_api/upload",
        data={"url": f"http://localhost:7865/image_api/image/{ws_id}/base.png"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    uploaded_name = payload["file"]
    uploaded_path = payload["filepath"]
    assert os.path.exists(uploaded_path)
    _cleanup_staged_file(uploaded_path)
    shutil.rmtree(ws_dir, ignore_errors=True)

    # 2. Test /staging_api/image/
    staged_source_path = os.path.join(get_staging_dir(), "dummy_staged.png")
    img = Image.new("RGBA", (10, 10), color=(0, 0, 255, 255))
    img.save(staged_source_path)

    response = client.post(
        "/staging_api/upload",
        data={"url": "http://localhost:7865/staging_api/image/dummy_staged.png"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    uploaded_path = payload["filepath"]
    assert os.path.exists(uploaded_path)
    _cleanup_staged_file(uploaded_path)
    _cleanup_staged_file(staged_source_path)

    # 3. Test /runtime_surface_api/preview_image (string/filepath variant)
    preview_file_path = os.path.join(get_staging_dir(), "temp_preview.png")
    img = Image.new("RGBA", (10, 10), color=(255, 255, 0, 255))
    img.save(preview_file_path)

    monkeypatch.setattr(runtime_surface_state, "get_preview_image_path", lambda: preview_file_path)
    # mock _last_preview_value as string
    monkeypatch.setattr(runtime_surface_state, "_last_preview_value", preview_file_path)

    response = client.post(
        "/staging_api/upload",
        data={"url": "http://localhost:7865/runtime_surface_api/preview_image?revision=1"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    uploaded_path = payload["filepath"]
    assert os.path.exists(uploaded_path)
    _cleanup_staged_file(uploaded_path)
    _cleanup_staged_file(preview_file_path)

    # 4. Test /runtime_surface_api/completed_image/
    task_id = "test_task"
    completed_img_path = os.path.join(get_staging_dir(), "completed.png")
    img = Image.new("RGBA", (10, 10), color=(255, 0, 255, 255))
    img.save(completed_img_path)

    monkeypatch.setattr(
        runtime_surface_state,
        "get_completed_image_path",
        lambda t_id, idx: completed_img_path if t_id == task_id and idx == 0 else None
    )

    response = client.post(
        "/staging_api/upload",
        data={"url": f"http://localhost:7865/runtime_surface_api/completed_image/{task_id}/0"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    uploaded_path = payload["filepath"]
    assert os.path.exists(uploaded_path)
    _cleanup_staged_file(uploaded_path)
    _cleanup_staged_file(completed_img_path)


def test_staging_upload_local_endpoint_blocks_path_traversal():
    response = client.post(
        "/staging_api/upload",
        data={"url": "http://localhost:7865/staging_api/image/..%2Foutside.png"}
    )
    assert response.status_code == 403

    response = client.post(
        "/staging_api/upload",
        data={"url": "http://localhost:7865/image_api/image/test_local_ws/..%2Foutside.png"}
    )
    assert response.status_code == 403
