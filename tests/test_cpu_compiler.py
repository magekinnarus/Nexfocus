import os
import tempfile
from unittest.mock import MagicMock

import pytest
import safetensors.torch
import torch

import backend.cpu_compiler as cpu_compiler_mod
from backend.gpu_compiler import GpuArtifactCompiler
from backend.cpu_compiler import CpuArtifactCompiler, LoRAPatchDef, SafeOpenHeaderOnly, LazyWeight
from backend.weight_ops import calculate_weight
import ldm_patched.modules.weight_adapter as weight_adapter


@pytest.fixture
def dummy_lora_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as tmp:
        # Create a dummy LoRA with 2 layers: one Linear (2D), one Conv2d (4D)
        # linear: out=16, in=16, rank=4
        mat1_lin = torch.randn(16, 4, dtype=torch.float16)
        mat2_lin = torch.randn(4, 16, dtype=torch.float16)
        alpha_lin = torch.tensor(4.0, dtype=torch.float32)

        # conv: out=8, in=8, kh=3, kw=3, rank=2
        mat1_conv = torch.randn(8, 2, dtype=torch.float16)
        mat2_conv = torch.randn(2, 8, 3, 3, dtype=torch.float16)
        alpha_conv = torch.tensor(2.0, dtype=torch.float32)

        tensors = {
            "lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight": mat1_lin,
            "lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight": mat2_lin,
            "lora_unet_down_blocks_0_attentions_0_proj_in.alpha": alpha_lin,
            "lora_unet_down_blocks_0_attentions_0_conv_in.lora_up.weight": mat1_conv,
            "lora_unet_down_blocks_0_attentions_0_conv_in.lora_down.weight": mat2_conv,
            "lora_unet_down_blocks_0_attentions_0_conv_in.alpha": alpha_conv,
        }
        safetensors.torch.save_file(tensors, tmp.name)
        tmp_path = tmp.name

    yield tmp_path
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def test_lazy_weight_repr_and_loading(dummy_lora_file):
    header = SafeOpenHeaderOnly(dummy_lora_file)
    assert "lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight" in header

    lazy_w = header["lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight"]
    assert isinstance(lazy_w, LazyWeight)
    assert lazy_w.shape == [16, 4]
    assert repr(lazy_w) == f"LazyWeight({dummy_lora_file!r}, 'lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight', [16, 4], 'torch.float16')"

    loaded = lazy_w.load()
    assert isinstance(loaded, torch.Tensor)
    assert loaded.shape == (16, 4)
    assert loaded.dtype == torch.float16


def test_cpu_compiler_standalone_parity(dummy_lora_file):
    base_sd = {
        "diffusion_model.input_blocks.1.1.proj_in.weight": torch.randn(16, 16, dtype=torch.float16),
        "diffusion_model.input_blocks.1.1.conv_in.weight": torch.randn(8, 8, 3, 3, dtype=torch.float16),
    }

    key_map = {
        "lora_unet_down_blocks_0_attentions_0_proj_in": "diffusion_model.input_blocks.1.1.proj_in.weight",
        "lora_unet_down_blocks_0_attentions_0_conv_in": "diffusion_model.input_blocks.1.1.conv_in.weight",
    }

    # Keep a pristine copy for sequential calculation
    base_sd_copy = {k: v.clone() for k, v in base_sd.items()}

    # Run CpuArtifactCompiler
    patches = [LoRAPatchDef(lora_path=dummy_lora_file, strength=0.8)]
    result = CpuArtifactCompiler.compile_unet(base_sd, patches, key_map=key_map, pin_unet_host=False, num_workers=2)
    assert result["status"] == "compiled"
    assert result["patch_count"] == 2

    # Run sequential calculate_weight for parity check
    with safetensors.torch.safe_open(dummy_lora_file, framework="pt", device="cpu") as f:
        mat1_lin = f.get_tensor("lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight")
        mat2_lin = f.get_tensor("lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight")
        alpha_lin = f.get_tensor("lora_unet_down_blocks_0_attentions_0_proj_in.alpha").item()

        mat1_conv = f.get_tensor("lora_unet_down_blocks_0_attentions_0_conv_in.lora_up.weight")
        mat2_conv = f.get_tensor("lora_unet_down_blocks_0_attentions_0_conv_in.lora_down.weight")
        alpha_conv = f.get_tensor("lora_unet_down_blocks_0_attentions_0_conv_in.alpha").item()

    lin_adapter = weight_adapter.LoRAAdapter(set(), (mat1_lin, mat2_lin, alpha_lin, None, None, None))
    conv_adapter = weight_adapter.LoRAAdapter(set(), (mat1_conv, mat2_conv, alpha_conv, None, None, None))

    expected_lin = calculate_weight(
        [(0.8, lin_adapter, 1.0, None, lambda x: x)],
        base_sd_copy["diffusion_model.input_blocks.1.1.proj_in.weight"].clone(),
        "diffusion_model.input_blocks.1.1.proj_in.weight",
        intermediate_dtype=torch.float16,
    )

    expected_conv = calculate_weight(
        [(0.8, conv_adapter, 1.0, None, lambda x: x)],
        base_sd_copy["diffusion_model.input_blocks.1.1.conv_in.weight"].clone(),
        "diffusion_model.input_blocks.1.1.conv_in.weight",
        intermediate_dtype=torch.float16,
    )

    assert torch.allclose(base_sd["diffusion_model.input_blocks.1.1.proj_in.weight"], expected_lin, atol=1e-2, rtol=1e-2)
    assert torch.allclose(base_sd["diffusion_model.input_blocks.1.1.conv_in.weight"], expected_conv, atol=1e-2, rtol=1e-2)


def test_cpu_compiler_patcher_parity(dummy_lora_file):
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight1 = torch.nn.Parameter(torch.randn(16, 16, dtype=torch.float16))

    dummy_model = DummyModel()
    patcher = MagicMock()
    patcher.model = dummy_model

    # Create dummy LoRA adapter
    header = SafeOpenHeaderOnly(dummy_lora_file)
    lazy_up = header["lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight"]
    lazy_down = header["lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight"]
    alpha = header["lora_unet_down_blocks_0_attentions_0_proj_in.alpha"].item()

    adapter = weight_adapter.LoRAAdapter(set(), (lazy_up, lazy_down, alpha, None, None, None))
    patcher.patches = {
        "weight1": [(0.5, adapter, 1.0, None, lambda x: x)]
    }

    orig_weight = dummy_model.weight1.clone()

    # Compile patcher
    res = CpuArtifactCompiler.compile_patcher(patcher, pin_unet_host=False, num_workers=1)
    assert res["status"] == "compiled"
    assert len(patcher.patches) == 0

    # Verify weight mutated
    assert not torch.allclose(dummy_model.weight1, orig_weight)


def test_generic_patcher_compile_does_not_inherit_streaming_unet_pinning(monkeypatch):
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.float16))

    patcher = MagicMock()
    patcher.model = DummyModel()
    patcher.patches = {}

    pinned_models: list[torch.nn.Module] = []
    monkeypatch.setattr(
        cpu_compiler_mod,
        "_pin_module_tensors",
        lambda model: pinned_models.append(model) or 0,
    )

    CpuArtifactCompiler.compile_patcher(patcher)
    assert pinned_models == []

    CpuArtifactCompiler.compile_streaming_unet_patcher(
        patcher,
        pin_unet_host=True,
    )
    assert pinned_models == [patcher.model]


def test_cpu_compiler_patcher_pins_patched_outputs_during_compile(dummy_lora_file, monkeypatch):
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight1 = torch.nn.Parameter(torch.randn(16, 16, dtype=torch.float16))

    dummy_model = DummyModel()
    patcher = MagicMock()
    patcher.model = dummy_model
    patcher.weight_wrapper_patches = {}
    patcher.backup = {}
    patcher.object_patches_backup = {}
    patcher.model_size = lambda: 0

    header = SafeOpenHeaderOnly(dummy_lora_file)
    lazy_up = header["lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight"]
    lazy_down = header["lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight"]
    alpha = header["lora_unet_down_blocks_0_attentions_0_proj_in.alpha"].item()

    adapter = weight_adapter.LoRAAdapter(set(), (lazy_up, lazy_down, alpha, None, None, None))
    patcher.patches = {
        "weight1": [(0.5, adapter, 1.0, None, lambda x: x)]
    }
    dummy_model.current_weight_patches_uuid = "patched"
    dummy_model.model_loaded_weight_memory = 0
    dummy_model.model_lowvram = True
    dummy_model.lowvram_patch_counter = 1
    dummy_model.device = torch.device("cpu")

    pinned_shapes: list[tuple[int, ...]] = []

    monkeypatch.setattr(cpu_compiler_mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cpu_compiler_mod, "_pin_module_tensors", lambda module: 0)

    def fake_pin_tensor(tensor):
        pinned_shapes.append(tuple(tensor.shape))
        return tensor.clone(), tensor.numel() * tensor.element_size()

    monkeypatch.setattr(cpu_compiler_mod, "_pin_tensor", fake_pin_tensor)

    res = CpuArtifactCompiler.compile_patcher(patcher, pin_unet_host=True, num_workers=1)

    assert res["status"] == "compiled"
    assert pinned_shapes == [(16, 16)]


def test_cpu_compiler_progressive_materialization_replaces_untouched_params_too(dummy_lora_file, monkeypatch):
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight1 = torch.nn.Parameter(torch.randn(16, 16, dtype=torch.float16))
            self.weight2 = torch.nn.Parameter(torch.randn(8, 8, dtype=torch.float16))

    dummy_model = DummyModel()
    patcher = MagicMock()
    patcher.model = dummy_model
    patcher.weight_wrapper_patches = {}
    patcher.backup = {}
    patcher.object_patches_backup = {}
    patcher.model_size = lambda: 0

    header = SafeOpenHeaderOnly(dummy_lora_file)
    lazy_up = header["lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight"]
    lazy_down = header["lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight"]
    alpha = header["lora_unet_down_blocks_0_attentions_0_proj_in.alpha"].item()

    adapter = weight_adapter.LoRAAdapter(set(), (lazy_up, lazy_down, alpha, None, None, None))
    patcher.patches = {
        "weight1": [(0.5, adapter, 1.0, None, lambda x: x)]
    }
    dummy_model.current_weight_patches_uuid = "patched"
    dummy_model.model_loaded_weight_memory = 0
    dummy_model.model_lowvram = True
    dummy_model.lowvram_patch_counter = 1
    dummy_model.device = torch.device("cpu")

    pinned_shapes: list[tuple[int, ...]] = []

    monkeypatch.setattr(cpu_compiler_mod.torch.cuda, "is_available", lambda: True)

    def fake_pin_tensor(tensor):
        pinned_shapes.append(tuple(tensor.shape))
        return tensor.clone(), tensor.numel() * tensor.element_size()

    monkeypatch.setattr(cpu_compiler_mod, "_pin_tensor", fake_pin_tensor)

    res = CpuArtifactCompiler.compile_patcher(patcher, pin_unet_host=True, num_workers=1)
    assert res["status"] == "compiled"
    assert (16, 16) in pinned_shapes


def test_safe_open_header_only_fallback(tmp_path):
    # Create a dummy state dict and save it as a legacy PyTorch (.pt) file
    legacy_file = tmp_path / "legacy_model.pt"
    tensors = {
        "layer1.weight": torch.randn(4, 4, dtype=torch.float32),
        "layer2.weight": torch.randn(2, 2, dtype=torch.float16),
        "some_metadata": "some_value"
    }
    torch.save(tensors, legacy_file)

    # Load it via SafeOpenHeaderOnly
    header = SafeOpenHeaderOnly(str(legacy_file))

    # Verify the keys are loaded correctly
    assert "layer1.weight" in header
    assert "layer2.weight" in header
    assert "some_metadata" in header
    assert header["some_metadata"] == "some_value"

    # Verify LazyWeight metadata extraction
    lazy_w1 = header["layer1.weight"]
    assert isinstance(lazy_w1, LazyWeight)
    assert lazy_w1.shape == [4, 4]
    assert lazy_w1.dtype == "torch.float32"

    lazy_w2 = header["layer2.weight"]
    assert isinstance(lazy_w2, LazyWeight)
    assert lazy_w2.shape == [2, 2]
    assert lazy_w2.dtype == "torch.float16"

    # Test loading tensors via load() method
    t1 = lazy_w1.load()
    assert torch.equal(t1, tensors["layer1.weight"])
    t2 = lazy_w2.load()
    assert torch.equal(t2, tensors["layer2.weight"])


def test_legacy_lazy_weight_can_reload_after_clear(tmp_path):
    legacy_file = tmp_path / "legacy_model.pt"
    tensor = torch.randn(4, 4, dtype=torch.float32)
    torch.save({"layer.weight": tensor}, legacy_file)

    header = SafeOpenHeaderOnly(str(legacy_file))
    lazy_weight = header["layer.weight"]

    assert isinstance(lazy_weight, LazyWeight)
    assert torch.equal(lazy_weight.load(), tensor)

    lazy_weight.clear_materialized_tensor()
    assert lazy_weight._tensor is None
    assert torch.equal(lazy_weight.load(), tensor)


def test_cpu_compiler_rejects_gpu_target(dummy_lora_file):
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight1 = torch.nn.Parameter(torch.randn(16, 16, dtype=torch.float16))

    dummy_model = DummyModel()
    patcher = MagicMock()
    patcher.model = dummy_model
    patcher.patches = {
        "weight1": [(0.5, object(), 1.0, None, lambda x: x)]
    }

    mock_tensor = MagicMock()
    mock_tensor.device = torch.device("cuda")
    dummy_model.parameters = MagicMock(return_value=[mock_tensor])
    dummy_model.buffers = MagicMock(return_value=[])

    with pytest.raises(AssertionError) as excinfo:
        CpuArtifactCompiler.compile_patcher(patcher, pin_unet_host=False, num_workers=1)
    assert "CpuArtifactCompiler cannot compile for GPU target device" in str(excinfo.value)


def test_gpu_artifact_compiler_parity(dummy_lora_file):
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight1 = torch.nn.Parameter(torch.randn(16, 16, dtype=torch.float16))

    dummy_model = DummyModel()
    patcher = MagicMock()
    patcher.model = dummy_model
    patcher.weight_wrapper_patches = {}
    patcher.backup = {}
    patcher.object_patches_backup = {}
    patcher.model_size = lambda: 0

    header = SafeOpenHeaderOnly(dummy_lora_file)
    lazy_up = header["lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight"]
    lazy_down = header["lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight"]
    alpha = header["lora_unet_down_blocks_0_attentions_0_proj_in.alpha"].item()

    adapter = weight_adapter.LoRAAdapter(set(), (lazy_up, lazy_down, alpha, None, None, None))
    patcher.patches = {
        "weight1": [(0.5, adapter, 1.0, None, lambda x: x)]
    }

    clean_source = {
        "weight1": dummy_model.weight1.clone()
    }

    res = GpuArtifactCompiler.compile_patcher(
        patcher,
        clean_source=clean_source,
        target_device=torch.device("cpu"),
        intermediate_dtype=torch.float16,
    )
    assert res["status"] == "compiled"
    assert len(patcher.patches) == 0
    assert not torch.allclose(dummy_model.weight1, clean_source["weight1"])
