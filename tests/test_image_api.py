import os
import io
import base64
import sys
from unittest.mock import MagicMock
import pytest
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from PIL import ImageDraw

# Ensure we can import modules from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

@pytest.fixture(scope="module")
def client():
    # Back up original modules to prevent polluting other tests
    modules_to_backup = ["args_manager", "gradio", "gradio.routes"]
    original_modules = {name: sys.modules.get(name) for name in modules_to_backup}

    import argparse
    mock_args = argparse.Namespace()
    mock_args.output_path = None
    mock_args.listen = "127.0.0.1"
    mock_args.port = 7860
    mock_args.share = False
    mock_args.in_browser = False
    mock_args.temp_path = None
    mock_args.preset = "default"

    sys.modules["args_manager"] = MagicMock()
    sys.modules["args_manager"].args = mock_args

    sys.modules["gradio"] = MagicMock()
    sys.modules["gradio.routes"] = MagicMock()

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from modules.image_api import image_router

    app = FastAPI()
    app.include_router(image_router)
    test_client = TestClient(app)

    yield test_client

    # Restore original modules
    for name, orig in original_modules.items():
        if orig is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = orig


def test_image_api_workspace_isolation(client):
    ws1 = "test_ws_1"
    ws2 = "test_ws_2"

    # Clean any previous test state
    import shutil
    from modules.image_api import get_workspaces_root
    root = get_workspaces_root()
    if os.path.exists(os.path.join(root, ws1)):
        shutil.rmtree(os.path.join(root, ws1))
    if os.path.exists(os.path.join(root, ws2)):
        shutil.rmtree(os.path.join(root, ws2))

    # 1. Create a dummy image
    img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 255))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    # 2. Upload to WS1
    response = client.post(
        "/image_api/upload",
        data={"workspace_id": ws1},
        files={"file": ("test.png", img_byte_arr, "image/png")}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["workspace_id"] == ws1
    assert response.json()["path"].endswith(os.path.join(ws1, "base.png"))

    # 3. Check WS1 image exists
    response = client.get(f"/image_api/image/{ws1}/base.png")
    assert response.status_code == 200

    # 4. Check WS2 image DOES NOT exist
    response = client.get(f"/image_api/image/{ws2}/base.png")
    assert response.status_code == 404

    # 5. Cleanup WS1
    response = client.delete(f"/image_api/workspace/{ws1}")
    assert response.status_code == 200

    # 6. Verify WS1 deleted
    response = client.get(f"/image_api/image/{ws1}/base.png")
    assert response.status_code == 404


def test_compute_context(client):
    ws = "test_compute_ws"

    # Mock mask_processing dependencies if necessary
    # (Since we are testing the router, we want to see it calling mask_proc)

    # 1. Upload base image
    img = Image.new("RGBA", (256, 256), color=(100, 100, 100, 255))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    client.post(
        "/image_api/upload",
        data={"workspace_id": ws},
        files={"file": ("base.png", img_byte_arr, "image/png")}
    )

    # 2. Create a mock mask
    mask = Image.new("L", (256, 256), color=0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([50, 50, 150, 150], fill=255)

    mask_byte_arr = io.BytesIO()
    mask.save(mask_byte_arr, format="PNG")
    mask_b64 = base64.b64encode(mask_byte_arr.getvalue()).decode("utf-8")
    mask_data_url = f"data:image/png;base64,{mask_b64}"

    # 3. Mock the core computation to avoid loading actual models
    import modules.mask_processing as mask_proc
    mask_proc.core_compute_inpaint_step1_context = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.bb_image = np.zeros((128, 128, 3), dtype=np.uint8)
    mask_proc.core_compute_inpaint_step1_context.return_value = mock_ctx

    # 4. Compute Context
    response = client.post(
        "/image_api/compute_context",
        data={
            "workspace_id": ws,
            "mask_base64": mask_data_url
        }
    )

    assert response.status_code == 200
    assert "context_mask_url" in response.json()
    assert "bb_patch_url" in response.json()

    # Verify files exist in workspace
    assert client.get(f"/image_api/image/{ws}/context_mask.png").status_code == 200
    assert client.get(f"/image_api/image/{ws}/bb_patch.png").status_code == 200

    # Cleanup
    client.delete(f"/image_api/workspace/{ws}")


def test_image_api_preserve_metadata_upload(client):
    ws = "test_meta_ws"
    from modules.image_api import get_workspaces_root
    root = get_workspaces_root()
    import shutil
    if os.path.exists(os.path.join(root, ws)):
        shutil.rmtree(os.path.join(root, ws))

    img = Image.new("RGBA", (32, 32), color=(10, 20, 30, 255))
    info = PngInfo()
    info.add_text("fooocus_scheme", "fooocus_nex")
    info.add_text("parameters", '{"prompt": "hello"}')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG", pnginfo=info)
    img_byte_arr = img_byte_arr.getvalue()

    response = client.post(
        "/image_api/upload",
        data={"workspace_id": ws, "preserve_metadata": "true"},
        files={"file": ("meta_source.png", img_byte_arr, "image/png")}
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["workspace_id"] == ws
    assert payload["filename"] == "meta_source.png"

    saved_path = payload["path"]
    assert os.path.exists(saved_path)

    with Image.open(saved_path) as restored:
        assert restored.info.get("fooocus_scheme") == "fooocus_nex"
        assert restored.info.get("parameters") == '{"prompt": "hello"}'

    client.delete(f"/image_api/workspace/{ws}")


def test_prepare_outpaint_step1_reuses_workspace_ids(client, monkeypatch):
    import modules.mask_processing as mask_proc
    import modules.pipeline.outpaint as outpaint_module

    monkeypatch.setattr(mask_proc.gr, "update", lambda **kwargs: kwargs)
    monkeypatch.setattr(mask_proc, "resolve_workspace_image_path", lambda *args, **kwargs: "D:/fake/base.png")
    monkeypatch.setattr(mask_proc, "unpack_gradio_data", lambda *args, **kwargs: np.zeros((32, 32, 3), dtype=np.uint8))

    class FakeOutpaintPipeline:
        def prepare_outpaint_canvas_only(self, image, direction, expansion_size=384, pixelate=False):
            _ = direction
            _ = expansion_size
            _ = pixelate
            return image, np.zeros((32, 32), dtype=np.uint8)

        def prepare(self, image, mask, outpaint_direction=None, extend_factor=1.2, generate_blend_mask=True):
            _ = image
            _ = mask
            _ = outpaint_direction
            _ = extend_factor
            _ = generate_blend_mask
            return MagicMock(bb_image=np.zeros((16, 16, 3), dtype=np.uint8))

    monkeypatch.setattr(outpaint_module, "OutpaintPipeline", FakeOutpaintPipeline)

    save_calls = []

    def fake_save_to_workspace_png(numpy_img, workspace_id=None, filename="base.png", prefix="mask_slot"):
        save_calls.append(
            {
                "workspace_id": workspace_id,
                "filename": filename,
                "prefix": prefix,
                "shape": tuple(numpy_img.shape),
            }
        )
        resolved_workspace_id = workspace_id or f"{prefix}_generated"
        return f"D:/fake/{filename}", resolved_workspace_id

    monkeypatch.setattr(mask_proc, "save_to_workspace_png", fake_save_to_workspace_png)

    result = mask_proc.prepare_outpaint_step1_assets(
        "D:/fake/base.png",
        "base_ws",
        "bb_ws",
        "mask_ws",
        ["right"],
        384,
    )

    assert save_calls[0]["workspace_id"] == "base_ws"
    assert save_calls[1]["workspace_id"] == "bb_ws"
    assert result[1]["value"] == "base_ws"
    assert result[3]["value"] == "bb_ws"
    assert result[5]["value"] == "mask_ws"


def test_save_to_workspace_png_keeps_prior_versions_for_queued_tasks(client):
    import shutil
    import modules.mask_processing as mask_proc

    workspace_id = "queue_snapshot_ws"
    _, workspace_dir = mask_proc.ensure_workspace_dir(workspace_id, prefix="inpaint_bb")
    if os.path.exists(workspace_dir):
        shutil.rmtree(workspace_dir)
    _, workspace_dir = mask_proc.ensure_workspace_dir(workspace_id, prefix="inpaint_bb")

    first_path, _ = mask_proc.save_to_workspace_png(
        np.zeros((8, 8, 3), dtype=np.uint8),
        workspace_id=workspace_id,
        filename="bb_image_first.png",
        prefix="inpaint_bb",
    )
    second_path, _ = mask_proc.save_to_workspace_png(
        np.ones((8, 8, 3), dtype=np.uint8),
        workspace_id=workspace_id,
        filename="bb_image_second.png",
        prefix="inpaint_bb",
    )

    assert os.path.exists(first_path)
    assert os.path.exists(second_path)

    shutil.rmtree(workspace_dir)


def test_metadata_preview_empty_path_is_safe():
    import modules.meta_parser as meta_parser

    assert meta_parser.read_info_from_image("") == (None, None)

