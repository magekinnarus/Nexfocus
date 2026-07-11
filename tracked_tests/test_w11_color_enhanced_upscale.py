"""Tracked W11c color-enhanced-upscale coverage."""

import os
import sys
import pytest
import numpy as np
import torch
from types import SimpleNamespace

# Setup sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import modules.model_taxonomy as model_taxonomy
from backend.sdxl_assembly.contracts import ColorExtractionSpec, SDXLAssemblyRequest, ResolvedFileIdentity
from backend.sdxl_assembly.color_extraction_worker import ColorExtractionWorker
from backend.sdxl_assembly.request_builder import build_assembly_request
from backend.sdxl_assembly.wavelet_color import wavelet_decomposition, wavelet_reconstruction
from modules.route_intent import resolve_route_intent
import modules.flags as flags
from modules.pipeline.routes import select_nearest_sdxl_bucket


def test_color_extraction_worker_correction_formula() -> None:
    """Verify that ColorExtractionWorker applies the x_center correction formula correctly."""
    spec = ColorExtractionSpec(enabled=True, restore_cfg=4.0, restore_cfg_s_tmin=0.0)
    worker = ColorExtractionWorker(spec)

    # 1. Base case: active correction
    denoised = torch.ones((1, 4, 16, 16), dtype=torch.float32) * 2.0
    x_center = torch.ones((1, 4, 16, 16), dtype=torch.float32) * 0.5

    sigma = 2.5
    sigma_max = 5.0

    worker.prepare(x_center, sigma_max)
    corrected = worker.correct_denoised(denoised, sigma)

    # Mathematical calculation:
    # d_center = 2.0 - 0.5 = 1.5
    # factor = (2.5 / 5.0) ** 4.0 = 0.5 ** 4.0 = 0.0625
    # expected = 2.0 - 1.5 * 0.0625 = 2.0 - 0.09375 = 1.90625
    expected = 1.90625
    assert torch.allclose(corrected, torch.tensor(expected)), f"Expected {expected}, got {corrected[0,0,0,0]}"

    # 2. Disabled behavior when spec enabled = False
    spec_disabled = ColorExtractionSpec(enabled=False)
    worker_disabled = ColorExtractionWorker(spec_disabled)
    worker_disabled.prepare(x_center, sigma_max)
    assert torch.allclose(worker_disabled.correct_denoised(denoised, sigma), denoised)

    # 3. Disabled behavior when sigma <= restore_cfg_s_tmin
    spec_tmin = ColorExtractionSpec(enabled=True, restore_cfg=4.0, restore_cfg_s_tmin=1.0)
    worker_tmin = ColorExtractionWorker(spec_tmin)
    worker_tmin.prepare(x_center, sigma_max)
    assert torch.allclose(worker_tmin.correct_denoised(denoised, 0.5), denoised)

    # 4. Safe cleanup on close
    worker.close()
    assert worker.x_center is None
    assert worker.sigma_max == 0.0

    # Stable fp32 interpolation must not overflow the fp16 subtraction
    # intermediate when finite endpoints have opposite large values.
    stress_worker = ColorExtractionWorker(ColorExtractionSpec(enabled=True, restore_cfg=1.0))
    stress_worker.prepare(torch.full((1, 1, 2, 2), -60000.0, dtype=torch.float16), 1.0)
    stress = stress_worker.correct_denoised(
        torch.full((1, 1, 2, 2), 60000.0, dtype=torch.float16),
        0.5,
    )
    assert torch.isfinite(stress).all()
    assert torch.equal(stress, torch.zeros_like(stress))
    stress_worker.close()


def test_color_extraction_worker_broadcasts_batch_sigma_and_rejects_invalid_sigma_max() -> None:
    spec = ColorExtractionSpec(enabled=True, restore_cfg=1.0)
    worker = ColorExtractionWorker(spec)
    x_center = torch.zeros((2, 1, 2, 2), dtype=torch.float32)
    denoised = torch.ones_like(x_center) * 2.0

    worker.prepare(x_center, 4.0)
    corrected = worker.correct_denoised(denoised, torch.tensor([1.0, 2.0]))
    assert torch.allclose(corrected[0], torch.full_like(corrected[0], 1.5))
    assert torch.allclose(corrected[1], torch.full_like(corrected[1], 1.0))

    worker.prepare(x_center, float('nan'))
    assert torch.equal(worker.correct_denoised(denoised, 2.0), denoised)
    worker.close()


def test_transient_vae_worker_uses_preloaded_encode_seam() -> None:
    from backend.sdxl_assembly.vae_encode_worker import _encode_attached_vae

    class TinyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

        def encode(self, pixels):
            return torch.zeros(
                (pixels.shape[0], 4, pixels.shape[-2] // 8, pixels.shape[-1] // 8),
                device=pixels.device,
                dtype=pixels.dtype,
            )

    class IdentityLatentFormat:
        @staticmethod
        def process_in(latent):
            return latent

    class AttachedVae:
        first_stage_model = TinyEncoder()
        latent_format = IdentityLatentFormat()

        def encode(self, _pixels):
            raise AssertionError("public VAE encode would re-enter residency admission")

    result = _encode_attached_vae(
        AttachedVae(),
        torch.zeros((1, 16, 16, 3), dtype=torch.float32),
    )
    assert result["samples"].shape == (1, 4, 2, 2)


def test_vae_encode_worker_attaches_vae_in_fp32(monkeypatch) -> None:
    from backend.sdxl_assembly.vae_encode_worker import VaeEncodeWorker

    class DummyFirstStage:
        def __init__(self) -> None:
            self.param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float16), requires_grad=False)
            self.last_to_dtype = None

        def parameters(self):
            return iter([self.param])

        def to(self, *, device=None, dtype=None):
            if dtype is not None:
                self.param = torch.nn.Parameter(self.param.detach().to(dtype=dtype), requires_grad=False)
                self.last_to_dtype = dtype
            return self

    class DummyPatcher:
        def patch_model(self, *, device_to, lowvram_model_memory=0):
            _ = device_to
            _ = lowvram_model_memory

    dummy_vae = SimpleNamespace(
        first_stage_model=DummyFirstStage(),
        patcher=DummyPatcher(),
    )

    monkeypatch.setattr(
        "backend.sdxl_assembly.vae_encode_worker.acquire_vae_component",
        lambda _request: dummy_vae,
    )
    monkeypatch.setattr(
        "backend.sdxl_assembly.vae_encode_worker._encode_attached_vae",
        lambda _vae, pixels: {
            "samples": torch.zeros(
                (pixels.shape[0], 4, pixels.shape[1] // 8, pixels.shape[2] // 8),
                dtype=torch.float32,
            )
        },
    )
    monkeypatch.setattr("backend.resources.eject_model", lambda _patcher: None)

    VaeEncodeWorker._ENCODE_CACHE.clear()
    request = SimpleNamespace(
        route_id="color_enhancement",
        device="cpu",
        vae=SimpleNamespace(sha256="vae_sha"),
        checkpoint=SimpleNamespace(sha256="ckpt_sha"),
    )
    pixels = torch.zeros((1, 16, 16, 3), dtype=torch.float32)
    prepared = SimpleNamespace(
        mode="image",
        original_pixels=pixels,
        original_mask=None,
        bb_pixels=pixels,
        bb_mask=None,
        blend_mask=None,
        bbox=(0, 16, 0, 16),
        bbox_area_ratio=1.0,
        mask_coverage=0.0,
        image_fingerprint="img_fp",
        mask_fingerprint=None,
        bb_pixels_fingerprint="bb_fp",
        bb_mask_fingerprint=None,
        get_cache_key=lambda _vae_identity: "w11c_fp32_encode_attach",
    )

    try:
        worker = VaeEncodeWorker(request)
        result = worker.encode(prepared)
        assert dummy_vae.first_stage_model.last_to_dtype == torch.float32
        assert result.route_latent.dtype == torch.float32
    finally:
        VaeEncodeWorker._ENCODE_CACHE.clear()


def test_vae_encode_worker_does_not_reenter_shared_residency(monkeypatch) -> None:
    from backend.sdxl_assembly.vae_encode_worker import VaeEncodeWorker

    class DummyFirstStage(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=False)

        def to(self, *, device=None, dtype=None):
            if dtype is not None:
                self.anchor = torch.nn.Parameter(self.anchor.detach().to(dtype=dtype), requires_grad=False)
            return self

        def encode(self, batch):
            return torch.zeros(
                (batch.shape[0], 4, batch.shape[-2] // 8, batch.shape[-1] // 8),
                dtype=batch.dtype,
                device=batch.device,
            )

    class DummyLatentFormat:
        @staticmethod
        def process_in(latent):
            return latent

    class DummyPatcher:
        def patch_model(self, *, device_to, lowvram_model_memory=0):
            _ = device_to
            _ = lowvram_model_memory

    dummy_vae = SimpleNamespace(
        first_stage_model=DummyFirstStage(),
        patcher=DummyPatcher(),
        latent_format=DummyLatentFormat(),
        runtime_policy=SimpleNamespace(vae_encode_mode="transient_gpu", prefer_gpu_vae_encode=True),
    )

    monkeypatch.setattr(
        "backend.sdxl_assembly.vae_encode_worker.acquire_vae_component",
        lambda _request: dummy_vae,
    )
    monkeypatch.setattr(
        "backend.encode.resources.prepare_models_for_stage",
        lambda *_args, **_kwargs: pytest.fail("worker encode must not re-enter shared residency admission"),
    )
    monkeypatch.setattr("backend.encode.resources.get_free_memory", lambda _device: 1024 * 1024 * 1024)
    monkeypatch.setattr("backend.resources.eject_model", lambda _patcher: None)

    VaeEncodeWorker._ENCODE_CACHE.clear()
    request = SimpleNamespace(
        route_id="color_enhancement",
        device="cpu",
        vae=SimpleNamespace(sha256="vae_sha"),
        checkpoint=SimpleNamespace(sha256="ckpt_sha"),
    )
    pixels = torch.zeros((1, 16, 16, 3), dtype=torch.float32)
    prepared = SimpleNamespace(
        mode="image",
        original_pixels=pixels,
        original_mask=None,
        bb_pixels=pixels,
        bb_mask=None,
        blend_mask=None,
        bbox=(0, 16, 0, 16),
        bbox_area_ratio=1.0,
        mask_coverage=0.0,
        image_fingerprint="img_fp",
        mask_fingerprint=None,
        bb_pixels_fingerprint="bb_fp",
        bb_mask_fingerprint=None,
        get_cache_key=lambda _vae_identity: "w11c_no_residency_encode",
    )

    try:
        worker = VaeEncodeWorker(request)
        result = worker.encode(prepared)
        assert result.route_latent.shape == (1, 4, 2, 2)
    finally:
        VaeEncodeWorker._ENCODE_CACHE.clear()


def test_vae_decode_rejects_nonfinite_latent_before_model_acquisition(monkeypatch) -> None:
    from backend.sdxl_assembly.vae_decode_worker import TransientVaeDecodeWorker

    acquired = []
    monkeypatch.setattr(
        "backend.sdxl_assembly.vae_decode_worker.acquire_vae_component",
        lambda _request: acquired.append(True),
    )
    worker = TransientVaeDecodeWorker(SimpleNamespace(tiled=False, route_id="color_enhancement"))
    latent = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    latent[..., 0, 0] = float("nan")

    with pytest.raises(RuntimeError, match="rejected a non-finite"):
        worker.decode(latent, torch.device("cpu"))
    assert acquired == []


def test_vae_decode_worker_attaches_vae_in_fp32(monkeypatch) -> None:
    from backend.sdxl_assembly.vae_decode_worker import TransientVaeDecodeWorker

    class DummyFirstStage:
        def __init__(self) -> None:
            self.param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float16), requires_grad=False)
            self.last_to_dtype = None

        def parameters(self):
            return iter([self.param])

        def to(self, *, device=None, dtype=None):
            if dtype is not None:
                self.param = torch.nn.Parameter(self.param.detach().to(dtype=dtype), requires_grad=False)
                self.last_to_dtype = dtype
            return self

    class DummyPatcher:
        def patch_model(self, *, device_to, lowvram_model_memory=0):
            _ = device_to
            _ = lowvram_model_memory

    dummy_vae = SimpleNamespace(
        first_stage_model=DummyFirstStage(),
        patcher=DummyPatcher(),
    )

    monkeypatch.setattr(
        "backend.sdxl_assembly.vae_decode_worker.acquire_vae_component",
        lambda _request: dummy_vae,
    )
    monkeypatch.setattr(
        "backend.decode.decode_preloaded_vae",
        lambda _vae, latent, tiled=False: torch.zeros(
            (latent.shape[0], latent.shape[2], latent.shape[3], 3),
            dtype=torch.float32,
        ),
    )
    monkeypatch.setattr(
        "modules.core.pytorch_to_numpy",
        lambda tensor: [tensor[0].detach().cpu().numpy()],
    )
    monkeypatch.setattr("backend.resources.eject_model", lambda _patcher: None)

    worker = TransientVaeDecodeWorker(SimpleNamespace(tiled=False, route_id="color_enhancement"))
    output, _attach_time, _decode_time = worker.decode(
        torch.zeros((1, 4, 8, 8), dtype=torch.float32),
        torch.device("cpu"),
    )

    assert dummy_vae.first_stage_model.last_to_dtype == torch.float32
    assert output.shape == (8, 8, 3)


def test_vae_decode_worker_does_not_reenter_shared_residency(monkeypatch) -> None:
    from backend.sdxl_assembly.vae_decode_worker import TransientVaeDecodeWorker

    class DummyFirstStage(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=False)

        def to(self, *, device=None, dtype=None):
            if dtype is not None:
                self.anchor = torch.nn.Parameter(self.anchor.detach().to(dtype=dtype), requires_grad=False)
            return self

        def decode(self, batch):
            return torch.zeros(
                (batch.shape[0], 3, batch.shape[-2], batch.shape[-1]),
                dtype=torch.float32,
                device=batch.device,
            )

    class DummyLatentFormat:
        @staticmethod
        def process_out(latent):
            return latent

    class DummyPatcher:
        def patch_model(self, *, device_to, lowvram_model_memory=0):
            _ = device_to
            _ = lowvram_model_memory

        def current_loaded_device(self):
            return torch.device("cpu")

        load_device = torch.device("cpu")

    dummy_vae = SimpleNamespace(
        first_stage_model=DummyFirstStage(),
        patcher=DummyPatcher(),
        latent_format=DummyLatentFormat(),
        runtime_policy=SimpleNamespace(vae_encode_mode="transient_gpu", prefer_gpu_vae_encode=True),
    )

    monkeypatch.setattr(
        "backend.sdxl_assembly.vae_decode_worker.acquire_vae_component",
        lambda _request: dummy_vae,
    )
    monkeypatch.setattr(
        "backend.decode.resources.prepare_models_for_stage",
        lambda *_args, **_kwargs: pytest.fail("worker decode must not re-enter shared residency admission"),
    )
    monkeypatch.setattr("backend.decode.resources.get_free_memory", lambda _device: 1024 * 1024 * 1024)
    monkeypatch.setattr(
        "modules.core.pytorch_to_numpy",
        lambda tensor: [tensor[0].detach().cpu().numpy()],
    )
    monkeypatch.setattr("backend.resources.eject_model", lambda _patcher: None)

    worker = TransientVaeDecodeWorker(SimpleNamespace(tiled=False, route_id="color_enhancement"))
    output, _attach_time, _decode_time = worker.decode(
        torch.zeros((1, 4, 8, 8), dtype=torch.float32),
        torch.device("cpu"),
    )

    assert output.shape == (8, 8, 3)


def test_streaming_spine_color_state_close_releases_latent() -> None:
    from backend.sdxl_assembly.streaming_unet import StreamingUnetSpine

    spine = StreamingUnetSpine.__new__(StreamingUnetSpine)
    spine.x_center = torch.ones((1, 1, 2, 2))
    closed = []

    class Worker:
        def close(self):
            closed.append(True)

    spine._active_color_worker = Worker()
    spine._close_color_extraction_state()
    assert closed == [True]
    assert spine.x_center is None
    assert spine._active_color_worker is None


def test_standard_scheduler_changes_reuse_warm_unet_key_but_lora_changes_do_not() -> None:
    from backend.sdxl_assembly.runtime_state import SDXLStreamingRuntimeState

    checkpoint = SimpleNamespace(sha256="checkpoint")
    common = dict(
        checkpoint=checkpoint,
        device="cpu",
        prefetch_depth=1,
        prefetch_chunk_mb=64,
        lora_stack_hash="same-lora",
    )
    beta = SimpleNamespace(**common, scheduler="beta")
    karras = SimpleNamespace(**common, scheduler="karras")
    changed_lora = SimpleNamespace(**{**common, "lora_stack_hash": "changed-lora"}, scheduler="beta")

    assert SDXLStreamingRuntimeState._build_key(beta) == SDXLStreamingRuntimeState._build_key(karras)
    assert SDXLStreamingRuntimeState._build_key(beta) != SDXLStreamingRuntimeState._build_key(changed_lora)


def test_wavelet_donor_direction_and_reconstruction() -> None:
    """Verify that wavelet reconstruction takes colors from color_ref and details from content donor."""
    # Create a bounded detail donor and a uniform color donor.  The uniform
    # color must remain uniform at every undecimated blur scale.
    y = torch.linspace(0.0, 1.0, 64).view(1, 1, 64, 1)
    x = torch.linspace(0.0, 1.0, 64).view(1, 1, 1, 64)
    content = 0.5 + 0.08 * torch.sin(11.0 * x) * torch.sin(9.0 * y)
    content = content.expand(1, 3, 64, 64).clone()

    # Create uniform color reference (color donor) - low frequency colors
    color_ref = torch.ones((1, 3, 64, 64), dtype=torch.float32) * 0.5

    reconstructed = wavelet_reconstruction(content, color_ref, levels=3)

    # Shape, dtype, and device must be preserved
    assert reconstructed.shape == content.shape
    assert reconstructed.dtype == content.dtype
    assert reconstructed.device == content.device

    content_high, content_low = wavelet_decomposition(content, levels=3)
    color_high, color_low = wavelet_decomposition(color_ref, levels=3)

    assert content_high.shape == content.shape
    assert content_low.shape == content.shape
    assert torch.allclose(content_high + content_low, content, atol=1e-5)
    assert color_high.shape == color_ref.shape
    assert torch.allclose(color_high + color_low, color_ref, atol=1e-5)

    # The transplant is exactly GAN high-frequency residual plus SDXL low
    # frequency field.  In particular, it cannot accidentally reverse donors.
    assert torch.allclose(reconstructed, content_high + color_low, atol=1e-5)
    assert not torch.allclose(color_low, content_low, atol=1e-4)

    # The final output values must be clamped to [0.0, 1.0]
    assert reconstructed.min() >= 0.0
    assert reconstructed.max() <= 1.0

    odd = torch.rand((1, 3, 65, 67), dtype=torch.float32)
    odd_high, odd_low = wavelet_decomposition(odd, levels=5)
    assert odd_high.shape == odd.shape
    assert odd_low.shape == odd.shape
    assert torch.allclose(odd_high + odd_low, odd, atol=1e-5)


def test_wavelet_transplant_is_shift_stable_without_decimation_blocks() -> None:
    """A one-pixel translation must not change a critically sampled phase grid."""
    height, width = 160, 176
    y = torch.linspace(0.0, 1.0, height).view(1, 1, height, 1)
    x = torch.linspace(0.0, 1.0, width).view(1, 1, 1, width)
    content = 0.5 + 0.04 * torch.sin(17.0 * x + 3.0 * y)
    content = content.expand(1, 3, height, width).clone()
    color_ref = 0.25 + 0.45 * x + 0.1 * y
    color_ref = color_ref.expand(1, 3, height, width).clone()

    shifted_content = torch.roll(content, shifts=(1, 1), dims=(-2, -1))
    shifted_color = torch.roll(color_ref, shifts=(1, 1), dims=(-2, -1))
    output = wavelet_reconstruction(content, color_ref, levels=5)
    shifted_output = wavelet_reconstruction(shifted_content, shifted_color, levels=5)
    unshifted = torch.roll(shifted_output, shifts=(-1, -1), dims=(-2, -1))

    # Ignore the finite-support boundary where replicate padding and roll
    # intentionally interact.  A decimated five-level Haar transplant fails
    # this interior phase-invariance check with its 32-pixel block lattice.
    assert torch.allclose(output[..., 40:-40, 40:-40], unshifted[..., 40:-40, 40:-40], atol=2e-5)


def test_nearest_bucket_selection() -> None:
    """Verify aspect-ratio bucket matching logic matches the stable inventory tie-break policy."""
    # 1024x1024 source aspect ratio 1.0 -> maps to 1024x1024
    assert select_nearest_sdxl_bucket(1024, 1024) == (1024, 1024)

    # Landscape aspect ratio 2.0 (e.g., 2000x1000) -> maps to the closest inventory ratio
    assert select_nearest_sdxl_bucket(2000, 1000) == (1408, 704)

    # Portrait aspect ratio 0.5 (e.g., 1000x2000) -> maps to tallest portrait bucket
    assert select_nearest_sdxl_bucket(1000, 2000) == (704, 1408)


def test_color_route_has_no_gan_upscaler_resource() -> None:
    from modules.pipeline.routes import ColorEnhancedUpscaleStage, PipelineRouteContext

    state = SimpleNamespace(upscale_gan_output_image=np.zeros((64, 64, 3), dtype=np.uint8))
    context = PipelineRouteContext(
        async_task=SimpleNamespace(),
        task_state=state,
        route_id="color_enhanced_upscale",
        route_family="upscale",
        prompt_tasks=[],
        progressbar_callback=None,
        yield_result_callback=None,
    )

    resource_ids = {item.resource_id for item in ColorEnhancedUpscaleStage().describe_resources(context)}
    assert "upscaler_model" not in resource_ids
    assert "sdxl_assembly" in resource_ids


def test_ui_and_frozen_route_intent() -> None:
    """Verify route intent mapping for Color Enhancement."""
    task_state = SimpleNamespace(
        current_tab="uov",
        input_image_checkbox=True,
        uov_input_image=np.zeros((128, 128, 3), dtype=np.uint8),
        uov_method="Color Enhancement",
        cn_tasks={},
        goals=["upscale"],
    )

    intent = resolve_route_intent(task_state)
    assert intent.wants_upscale is True
    assert intent.route_id == "color_enhanced_upscale"
    assert intent.route_family == "upscale"


def test_color_method_exposes_prompt_and_existing_gan_input() -> None:
    from modules.ui_logic import uov_method_change

    color_updates = uov_method_change("Color Enhancement")
    normal_updates = uov_method_change("Upscale")

    assert len(color_updates) == 6
    assert color_updates[3]["visible"] is True
    assert color_updates[4]["visible"] is True
    assert color_updates[1]["visible"] is False
    assert color_updates[2]["visible"] is False
    assert color_updates[5]["visible"] is False
    assert normal_updates[3]["visible"] is False
    assert normal_updates[4]["visible"] is False


def test_color_enhancement_request_freezes_workflow_contract(monkeypatch, tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.safetensors"
    checkpoint_path.write_bytes(b"checkpoint")

    monkeypatch.setattr(
        "backend.sdxl_assembly.request_builder.get_file_from_folder_list",
        lambda _name, _folders: str(checkpoint_path),
    )
    monkeypatch.setattr(
        "backend.sdxl_assembly.request_builder.config.resolve_model_taxonomy",
        lambda _path: SimpleNamespace(architecture=model_taxonomy.ARCHITECTURE_SDXL),
    )

    source_pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    state = SimpleNamespace(
        last_stop=False,
        base_model_name="checkpoint.safetensors",
        vae_name="Default (model)",
        goals=["upscale"],
        tiled=False,
        prepared_structural_cn_tasks={},
        prepared_contextual_cn_tasks={},
        initial_latent=None,
        prompt="unused main prompt",
        negative_prompt="negative",
        width=64,
        height=64,
        steps=23,
        cfg_scale=1.5,
        sampler_name="dpmpp_2m",
        scheduler_name="lcm",
        clip_skip=1,
        style_selections=[],
        sdxl_execution_policy=None,
        sharpness=2.0,
        adaptive_cfg=7.0,
        adm_scaler_positive=1.5,
        adm_scaler_negative=0.8,
        adm_scaler_end=0.3,
        prefetch_depth=1,
        prefetch_chunk_mb=64,
        use_expansion=False,
        disable_intermediate_results=True,
        input_image_checkbox=True,
        current_tab="uov",
        uov_method="Color Enhancement",
        uov_input_image=source_pixels,
        source_pixels=source_pixels,
    )

    request = build_assembly_request(
        task_state=state,
        task_dict={
            "task_seed": 123,
            "task_prompt": "unused main prompt",
            "task_negative_prompt": "negative",
            "positive": ["unused main prompt"],
            "negative": ["negative"],
        },
        current_task_id=0,
        total_count=1,
        all_steps=23,
        preparation_steps=0,
        denoising_strength=0.35,
        final_scheduler_name="sgm_uniform",
        loras=[],
        image_input_result={},
        force_eligible=True,
    )

    assert request.route_id == "color_enhancement"
    assert request.color_extraction is not None
    assert request.color_extraction.enabled is True
    assert request.spatial_context is not None
    assert request.spatial_context.mode == "image"
    assert request.metadata["workflow_contract"] == {
        "workflow_id": "color_enhanced_upscale",
        "workflow_name": "Color Enhancement",
        "workflow_family": "upscale",
        "assembly_route_id": "color_enhancement",
        "assembly_variant": "sdxl_color_enhancement",
        "source_policy": "strict_original",
        "donor_policy": "provided_gan_detail",
        "sampler_policy": "forced_dpmpp_2m",
        "scheduler_policy": "inherit_user_selection",
        "steps_policy": "inherit_user_selection",
        "cfg_policy": "fixed_1_5",
    }


def test_color_enhanced_upscale_stage_execution(monkeypatch) -> None:
    """Verify strict original-source SDXL execution and donor-only GAN use."""
    from modules.pipeline.routes import ColorEnhancedUpscaleStage, PipelineRouteContext, PipelineStageResult

    execution_order = []
    captured_color_request = {}
    saved_outputs = []

    # 2. Mock SDXL Assembly Task Execution
    def fake_run_sdxl_assembly_task(
        task_state,
        task_dict,
        current_task_id,
        total_count,
        all_steps,
        preparation_steps,
        denoising_strength,
        final_scheduler_name,
        **kwargs
    ):
        assert getattr(task_state, "upscale_gan_output_image", None) is not None
        assert not any(event.startswith("gan_") for event in execution_order)
        execution_order.append("sdxl_color_pass")
        captured_color_request.update({
            "prompt": task_state.prompt,
            "negative_prompt": task_state.negative_prompt,
            "loras": list(task_state.loras),
            "sampler": task_state.sampler_name,
            "scheduler": task_state.scheduler_name,
            "all_steps": all_steps,
            "final_scheduler": final_scheduler_name,
            "denoising_strength": denoising_strength,
            "source_shape": tuple(task_state.source_pixels.shape),
            "source_mean": int(np.rint(task_state.source_pixels.mean())),
        })

        # Return dummy SDXL output matching the input shape of the color pass source
        h, w = task_state.source_pixels.shape[:2]
        return np.ones((h, w, 3), dtype=np.uint8) * 180

    monkeypatch.setattr("backend.sdxl_assembly.gateway.run_sdxl_assembly_task", fake_run_sdxl_assembly_task)

    # Mock output save and log
    def fake_save_and_log(_task_state, _height, _width, images, task_dict, *_args):
        saved_outputs.append((task_dict.get("description"), list(images)))
        return [f"/mock/{task_dict.get('description')}.png"]

    monkeypatch.setattr("modules.pipeline.output.save_and_log", fake_save_and_log)

    # 3. A small original is still the exclusive SDXL source; the required GAN
    # image is used only as the final high-frequency donor.
    task_state_small = SimpleNamespace(
        goals=["upscale"],
        current_progress=0,
        uov_method="Color Enhancement",
        uov_input_image=np.ones((100, 100, 3), dtype=np.uint8) * 50,
        upscale_model="4xNomos2_otf_esrgan.pth",
        upscale_scale_override=0,
        upscale_prompt="vivid natural color",
        upscale_gan_output_image=np.ones((400, 400, 3), dtype=np.uint8) * 128,
        seed=123,
        prompt="must not be used by the color pass",
        negative_prompt="low quality",
        scheduler_name="karras",
        steps=24,
        cfg_scale=7.0,
        style_selections=[],
        use_expansion=False,
        loras=["frozen-lora"],
    )

    context_small = PipelineRouteContext(
        async_task=SimpleNamespace(),
        task_state=task_state_small,
        route_id="color_enhanced_upscale",
        route_family="upscale",
        prompt_tasks=[],
        progressbar_callback=lambda *args, **kwargs: None,
        yield_result_callback=lambda *args, **kwargs: None,
    )

    stage = ColorEnhancedUpscaleStage()
    execution_order.clear()

    result = stage.execute(context_small)

    assert result.route_complete is True
    assert result.notes['source_branch'] == "original"
    assert not any(event.startswith("gan_") for event in execution_order)
    assert "sdxl_color_pass" in execution_order
    assert captured_color_request == {
        "prompt": "vivid natural color",
        "negative_prompt": "low quality",
        "loras": ["frozen-lora"],
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
        "all_steps": 24,
        "final_scheduler": "karras",
        "denoising_strength": 0.35,
        "source_shape": (1024, 1024, 3),
        "source_mean": 50,
    }

    # Output must have GAN dimensions (400x400) and shape contract
    assert task_state_small.uov_input_image.shape == (400, 400, 3)
    assert task_state_small.uov_input_image.dtype == np.uint8
    assert task_state_small.uov_input_image.flags.c_contiguous
    assert [description for description, _images in saved_outputs] == ["Color Enhancement"]
    assert saved_outputs[0][1][0].shape == (400, 400, 3)

    # 4. Test branching behavior B: large image (>= 1,000,000 pixels)
    # Original: 1000x1000 (1,000,000 pixels). GAN output: 4000x4000.
    task_state_large = SimpleNamespace(
        goals=["upscale"],
        current_progress=0,
        uov_method="Color Enhancement",
        uov_input_image=np.ones((1000, 1000, 3), dtype=np.uint8) * 50,
        upscale_model="4xNomos2_otf_esrgan.pth",
        upscale_scale_override=0,
        upscale_gan_output_image=np.ones((4000, 4000, 3), dtype=np.uint8) * 128,
        seed=123,
        prompt="scenic",
        negative_prompt="",
        scheduler_name="karras",
        steps=24,
        cfg_scale=7.0,
        style_selections=[],
        use_expansion=False,
        loras=[],
    )

    context_large = PipelineRouteContext(
        async_task=SimpleNamespace(),
        task_state=task_state_large,
        route_id="color_enhanced_upscale",
        route_family="upscale",
        prompt_tasks=[],
        progressbar_callback=lambda *args, **kwargs: None,
        yield_result_callback=lambda *args, **kwargs: None,
    )

    execution_order.clear()

    result_large = stage.execute(context_large)

    assert result_large.route_complete is True
    assert result_large.notes['source_branch'] == "original"
    assert task_state_large.uov_input_image.shape == (4000, 4000, 3)

    # 5. The required GAN result remains the content donor without worker admission.
    task_state_bypass = SimpleNamespace(
        goals=["upscale"],
        current_progress=0,
        uov_method="Color Enhancement",
        uov_input_image=np.ones((50, 50, 3), dtype=np.uint8) * 50,
        upscale_gan_output_image=np.ones((100, 100, 3), dtype=np.uint8) * 120,
        upscale_model="4xNomos2_otf_esrgan.pth",
        upscale_scale_override=0,
        upscale_prompt="",
        seed=123,
        prompt="unused main prompt",
        negative_prompt="",
        scheduler_name="karras",
        steps=24,
        cfg_scale=7.0,
        style_selections=[],
        use_expansion=False,
        loras=[],
    )
    context_bypass = PipelineRouteContext(
        async_task=SimpleNamespace(),
        task_state=task_state_bypass,
        route_id="color_enhanced_upscale",
        route_family="upscale",
        prompt_tasks=[],
        progressbar_callback=lambda *args, **kwargs: None,
        yield_result_callback=lambda *args, **kwargs: None,
    )

    execution_order.clear()
    saved_outputs.clear()
    result_bypass = stage.execute(context_bypass)

    assert result_bypass.route_complete is True
    assert result_bypass.notes["gan_source"] == "provided"
    assert result_bypass.notes["workflow_id"] == "color_enhanced_upscale"
    assert result_bypass.notes["assembly_route_id"] == "color_enhancement"
    assert result_bypass.notes["assembly_variant"] == "sdxl_color_enhancement"
    assert result_bypass.notes["source_policy"] == "strict_original"
    assert result_bypass.notes["donor_policy"] == "provided_gan_detail"
    assert result_bypass.notes["sampler"] == "dpmpp_2m"
    assert result_bypass.notes["scheduler"] == "karras"
    assert result_bypass.notes["final_scheduler"] == "karras"
    assert result_bypass.notes["steps"] == 24
    assert not any(event.startswith("gan_") for event in execution_order)
    assert task_state_bypass.uov_input_image.shape == (100, 100, 3)
    assert [description for description, _images in saved_outputs] == ["Color Enhancement"]
    resource_ids = {item.resource_id for item in stage.describe_resources(context_bypass)}
    assert "upscaler_model" not in resource_ids
    assert "sdxl_assembly" in resource_ids


def test_color_route_does_not_run_generic_prompt_stage() -> None:
    from modules.pipeline.routes import build_generation_route, describe_route

    state = SimpleNamespace(
        current_tab="uov",
        input_image_checkbox=True,
        uov_input_image=np.zeros((32, 32, 3), dtype=np.uint8),
        uov_method="Color Enhancement",
        cn_tasks={},
        goals=[],
    )
    route = build_generation_route(state)
    assert route.route_id == "color_enhanced_upscale"
    assert describe_route(route) == ["image_input_prepare", "color_enhanced_upscale"]


def test_color_route_rejects_smaller_provided_gan_donor() -> None:
    from modules.pipeline.routes import ColorEnhancedUpscaleStage, PipelineRouteContext

    state = SimpleNamespace(
        goals=["upscale"],
        current_progress=0,
        uov_input_image=np.zeros((64, 64, 3), dtype=np.uint8),
        upscale_gan_output_image=np.zeros((32, 32, 3), dtype=np.uint8),
    )
    context = PipelineRouteContext(
        async_task=SimpleNamespace(),
        task_state=state,
        route_id="color_enhanced_upscale",
        route_family="upscale",
        prompt_tasks=[],
        progressbar_callback=None,
        yield_result_callback=None,
    )

    with pytest.raises(ValueError, match="must not be smaller"):
        ColorEnhancedUpscaleStage().execute(context)


def test_color_route_requires_existing_gan_donor() -> None:
    from modules.pipeline.routes import ColorEnhancedUpscaleStage, PipelineRouteContext

    state = SimpleNamespace(
        goals=["upscale"],
        current_progress=0,
        uov_input_image=np.zeros((64, 64, 3), dtype=np.uint8),
        upscale_gan_output_image=None,
    )
    context = PipelineRouteContext(
        async_task=SimpleNamespace(),
        task_state=state,
        route_id="color_enhanced_upscale",
        route_family="upscale",
        prompt_tasks=[],
        progressbar_callback=None,
        yield_result_callback=None,
    )

    with pytest.raises(ValueError, match="requires a color enhancement target"):
        ColorEnhancedUpscaleStage().execute(context)
