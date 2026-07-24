import os
import sys
import types
import numpy as np
import torch

mock_args = types.SimpleNamespace(
    colab=False,
    preset=None,
    output_path=None,
    temp_path=None,
    skip_model_load=False,
    disable_preset_selection=False,
    disable_image_log=False,
)

# Mock args_manager to prevent argparse parse_args() from parsing pytest CLI flags
sys.modules["args_manager"] = types.ModuleType("args_manager")
sys.modules["args_manager"].args = mock_args

from backend import preview as preview_mod
from modules.pipeline import inference


def test_sdxl_preview_transform_decodes_tensor_preview(monkeypatch):
    calls = {}

    class FakeLatent2RGBPreviewer:
        def __init__(self, factors):
            calls["factors"] = factors

    def fake_decode_latent_preview(previewer, latent_format, x0):
        calls["previewer_type"] = type(previewer).__name__
        calls["latent_format"] = latent_format
        calls["preview_input"] = x0.clone()
        return np.full((1, 1, 3), 42, dtype=np.uint8)

    monkeypatch.setattr(preview_mod, "Latent2RGBPreviewer", FakeLatent2RGBPreviewer)
    monkeypatch.setattr(preview_mod, "resolve_taesd_previewer", lambda *args, **kwargs: None)
    monkeypatch.setattr(preview_mod, "decode_latent_preview", fake_decode_latent_preview)

    latent_format = types.SimpleNamespace(
        latent_rgb_factors=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        process_out=lambda tensor: tensor + 1.0,
    )
    
    # Mock runtime to match what _build_sdxl_preview_transform queries
    mock_runtime = types.SimpleNamespace(
        unet=types.SimpleNamespace(
            load_device=torch.device("cpu"),
            model=types.SimpleNamespace(latent_format=latent_format),
        ),
        vae=None,
    )

    task_state = types.SimpleNamespace(disable_preview=False)
    transform = inference._build_sdxl_preview_transform(task_state, mock_runtime)
    preview = transform(torch.zeros((1, 4, 1, 1), dtype=torch.float32))

    assert preview.tolist() == [[[42, 42, 42]]]
    assert calls["factors"] == latent_format.latent_rgb_factors
    assert calls["previewer_type"] == "FakeLatent2RGBPreviewer"
    assert calls["latent_format"] is latent_format
    assert torch.equal(calls["preview_input"], torch.zeros((1, 4, 1, 1), dtype=torch.float32))


def test_sdxl_preview_transform_prefers_taesd_when_available(monkeypatch):
    calls = {"latent2rgb_used": False}

    class FakeTAESDPreviewer:
        pass

    def fake_latent2rgb_previewer(_factors):
        calls["latent2rgb_used"] = True
        return object()

    def fake_decode_latent_preview(previewer, latent_format, x0):
        calls["previewer"] = previewer
        calls["latent_format"] = latent_format
        calls["preview_input"] = x0.clone()
        return np.full((1, 1, 3), 7, dtype=np.uint8)

    monkeypatch.setattr(preview_mod, "Latent2RGBPreviewer", fake_latent2rgb_previewer)
    monkeypatch.setattr(preview_mod, "resolve_taesd_previewer", lambda *args, **kwargs: FakeTAESDPreviewer())
    monkeypatch.setattr(preview_mod, "decode_latent_preview", fake_decode_latent_preview)

    latent_format = types.SimpleNamespace(
        latent_rgb_factors=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        process_out=lambda tensor: tensor + 2.0,
    )

    mock_runtime = types.SimpleNamespace(
        unet=types.SimpleNamespace(
            load_device=torch.device("cpu"),
            model=types.SimpleNamespace(latent_format=latent_format),
        ),
        vae=None,
    )

    task_state = types.SimpleNamespace(disable_preview=False)
    transform = inference._build_sdxl_preview_transform(task_state, mock_runtime)
    preview = transform(torch.zeros((1, 4, 1, 1), dtype=torch.float32))

    assert preview.tolist() == [[[7, 7, 7]]]
    assert calls["latent2rgb_used"] is False
    assert type(calls["previewer"]).__name__ == "FakeTAESDPreviewer"
    assert calls["latent_format"] is latent_format
    assert torch.equal(calls["preview_input"], torch.zeros((1, 4, 1, 1), dtype=torch.float32))


def test_sdxl_preview_transform_uses_preview_tensor_device_for_previewer_resolution(monkeypatch):
    calls = {}

    def fake_resolve_best_available_previewer(device, latent_format, vae_approx_path=None):
        calls["device"] = str(device)
        calls["latent_format"] = latent_format
        return object()

    def fake_decode_preview_payload(previewer, latent_format, preview_payload):
        calls["previewer"] = previewer
        calls["preview_input_device"] = str(preview_payload.device)
        return np.full((1, 1, 3), 11, dtype=np.uint8)

    monkeypatch.setattr(preview_mod, "resolve_best_available_previewer", fake_resolve_best_available_previewer)
    monkeypatch.setattr(preview_mod, "decode_preview_payload", fake_decode_preview_payload)

    latent_format = types.SimpleNamespace(process_out=lambda tensor: tensor)
    mock_runtime = types.SimpleNamespace(
        unet=types.SimpleNamespace(
            load_device=torch.device("cuda"),
            model=types.SimpleNamespace(latent_format=latent_format),
        ),
        vae=None,
    )

    task_state = types.SimpleNamespace(disable_preview=False)
    transform = inference._build_sdxl_preview_transform(task_state, mock_runtime)
    preview = transform(torch.zeros((1, 4, 1, 1), dtype=torch.float32))

    assert preview.tolist() == [[[11, 11, 11]]]
    assert calls["device"] == "cpu"
    assert calls["latent_format"] is latent_format
    assert calls["preview_input_device"] == "cpu"
