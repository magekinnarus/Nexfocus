import os
import sys
import types

import numpy as np
import torch

import ldm_patched.utils.path_utils as path_utils
from backend import preview as preview_mod

mock_args = types.SimpleNamespace(
    colab=False,
    preset=None,
    output_path=None,
    temp_path=None,
    skip_model_load=False,
    disable_preset_selection=False,
    disable_image_log=False,
)

sys.modules["args_manager"] = types.ModuleType("args_manager")
sys.modules["args_manager"].args = mock_args

from modules.pipeline import routes


def test_preview_module_exports():
    """Verify that backend/preview.py exports the necessary production preview helpers."""
    from backend.preview import (
        Latent2RGBPreviewer,
        LatentPreviewer,
        TAESDPreviewerImpl,
        decode_latent_preview,
        resolve_taesd_previewer,
    )

    assert Latent2RGBPreviewer is not None
    assert decode_latent_preview is not None
    assert resolve_taesd_previewer is not None
    assert TAESDPreviewerImpl is not None
    assert LatentPreviewer is not None


def test_precision_module_exports():
    """Verify that backend/precision.py exports the necessary precision helpers."""
    from backend.precision import (
        pick_weight_dtype,
        text_encoder_dtype,
        unet_dtype,
        unet_manual_cast,
    )

    assert unet_dtype is not None
    assert unet_manual_cast is not None
    assert text_encoder_dtype is not None
    assert pick_weight_dtype is not None


def test_latent2rgb_previewer_decodes_to_numpy_rgb():
    previewer = preview_mod.Latent2RGBPreviewer(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    latent = torch.tensor([[[[1.0]], [[0.0]], [[-1.0]], [[0.0]]]], dtype=torch.float32)

    preview = preview_mod.decode_latent_preview(previewer, latent_format=None, x0=latent)

    assert isinstance(preview, np.ndarray)
    assert preview.shape == (1, 1, 3)
    assert preview.dtype == np.uint8
    assert preview.tolist() == [[[255, 127, 0]]]


def test_resolve_taesd_previewer_prefers_direct_vae_approx_path(monkeypatch):
    expected_path = os.path.join("D:\\vae_approx", "taesd_decoder.pth")
    calls = {}

    class FakeTAESD:
        def __init__(self, _unused, checkpoint_path):
            calls["checkpoint_path"] = checkpoint_path

        def to(self, device):
            calls["device"] = str(device)
            return self

    def fail_directory_walk(_category):
        raise AssertionError("legacy directory walk should not run when direct lookup succeeds")

    monkeypatch.setattr(preview_mod, "TAESD", FakeTAESD)
    monkeypatch.setattr(preview_mod.os.path, "isfile", lambda candidate: candidate == expected_path)
    monkeypatch.setattr(path_utils, "get_filename_list", fail_directory_walk)

    latent_format = types.SimpleNamespace(taesd_decoder_name="taesd_decoder")
    previewer = preview_mod.resolve_taesd_previewer(
        torch.device("cpu"),
        latent_format,
        vae_approx_path="D:\\vae_approx",
    )

    assert isinstance(previewer, preview_mod.TAESDPreviewerImpl)
    assert calls == {
        "checkpoint_path": expected_path,
        "device": "cpu",
    }


def test_resolve_taesd_previewer_falls_back_to_legacy_directory_walk(monkeypatch):
    expected_path = "D:\\legacy\\taesd_decoder.pt"
    calls = {"filename_list": 0, "full_path": 0}

    class FakeTAESD:
        def __init__(self, _unused, checkpoint_path):
            calls["checkpoint_path"] = checkpoint_path

        def to(self, device):
            calls["device"] = str(device)
            return self

    def fake_get_filename_list(category):
        assert category == "vae_approx"
        calls["filename_list"] += 1
        return ["taesd_decoder.pt"]

    def fake_get_full_path(category, filename):
        assert category == "vae_approx"
        assert filename == "taesd_decoder.pt"
        calls["full_path"] += 1
        return expected_path

    monkeypatch.setattr(preview_mod, "TAESD", FakeTAESD)
    monkeypatch.setattr(preview_mod.os.path, "isfile", lambda _candidate: False)
    monkeypatch.setattr(path_utils, "get_filename_list", fake_get_filename_list)
    monkeypatch.setattr(path_utils, "get_full_path", fake_get_full_path)

    latent_format = types.SimpleNamespace(taesd_decoder_name="taesd_decoder")
    previewer = preview_mod.resolve_taesd_previewer(
        torch.device("cpu"),
        latent_format,
        vae_approx_path="D:\\vae_approx",
    )

    assert isinstance(previewer, preview_mod.TAESDPreviewerImpl)
    assert calls == {
        "filename_list": 1,
        "full_path": 1,
        "checkpoint_path": expected_path,
        "device": "cpu",
    }


def test_routes_no_longer_expose_flux_session_preview_transform():
    assert not hasattr(routes, "_build_flux_preview_transform")


def test_import_cleanliness():
    """Verify that rewired production files do not import legacy preview or model management helpers directly."""
    files_to_check = [
        "backend/preview.py",
        "backend/precision.py",
        "modules/pipeline/routes.py",
        "modules/core.py",
        "modules/monitor_api.py",
        "modules/upscaler.py",
        "backend/controlnet_compat.py",
        "backend/ip_adapter.py",
        "backend/pulid_runtime.py",

    ]

    for relative_path in files_to_check:
        full_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), relative_path)
        assert os.path.exists(full_path), f"{relative_path} does not exist!"

        with open(full_path, "r", encoding="utf-8") as handle:
            content = handle.read()

        assert "ldm_patched.utils.latent_visualization" not in content, (
            f"Found direct latent_visualization import in {relative_path}"
        )
        assert "latent_visualization" not in content, (
            f"Found latent_visualization mention in {relative_path}"
        )

        assert "ldm_patched.modules.model_management" not in content, (
            f"Found direct model_management import in {relative_path}"
        )
        assert "import model_management" not in content, (
            f"Found generic model_management import in {relative_path}"
        )
        assert "backend.resources as model_management" not in content, (
            f"Found misleading backend.resources alias in {relative_path}"
        )
        if relative_path == "modules/pipeline/routes.py":
            assert "_build_flux_preview_transform" not in content, (
                "Found archived Flux preview transform helper in modules/pipeline/routes.py"
            )
            assert "get_active_flux_fill_session" not in content, (
                "Found archived Flux session preview hook in modules/pipeline/routes.py"
            )


def test_repackaged_model_management_helpers():
    """Verify that backend/resources.py correctly implements the repackaged model_management helpers."""
    from backend.resources import (
        load_model_gpu,
        unet_manual_cast,
        text_encoder_dtype,
        pick_weight_dtype,
        supports_cast,
    )

    # Test load_model_gpu calls load_models_gpu
    calls = []
    import backend.resources as resources
    original_load_models_gpu = resources.load_models_gpu
    try:
        resources.load_models_gpu = lambda models, **kwargs: calls.append(models)
        load_model_gpu("fake-model")
        assert calls == [["fake-model"]]
    finally:
        resources.load_models_gpu = original_load_models_gpu

    # Test unet_manual_cast
    assert unet_manual_cast(torch.float32, torch.device("cpu")) is None
    # If fp16 is supported, it should return None or fp16 depending on support
    res = unet_manual_cast(torch.float16, torch.device("cpu"), [torch.float16])
    assert res in (None, torch.float16, torch.float32)

    # Test text_encoder_dtype
    te_dtype = text_encoder_dtype(torch.device("cpu"))
    assert te_dtype in (torch.float16, torch.float32)

    # Test supports_cast and pick_weight_dtype
    assert supports_cast(torch.device("cpu"), torch.float32) is True
    assert pick_weight_dtype(None, torch.float32, torch.device("cpu")) == torch.float32


def test_latent_visualization_deletion_proof():
    """Verify that ldm_patched/utils/latent_visualization.py is deleted and has no workspace consumers."""
    legacy_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "ldm_patched/utils/latent_visualization.py"
    )
    assert not os.path.exists(legacy_file), "latent_visualization.py was not deleted!"
