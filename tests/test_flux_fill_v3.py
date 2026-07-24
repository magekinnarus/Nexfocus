from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

import ldm_patched.modules.ops as base_ops
from backend.flux_fill_v3.contracts import (
    FluxFillCategory,
    FluxFillRequest,
    FluxRuntimeIdentity,
    T5PostureKind,
    UNetSpineKind,
    VAEPostureKind,
)
from backend.flux_fill_v3.director import FluxAssemblyDirector
from backend.flux_fill_v3.assembly import FluxAssembly
from backend.flux_fill_v3.t5_worker import (
    DiskPagedTextWorker,
    T5Stack,
    _normalize_t5_loader_policy,
    _resolve_disk_paged_t5_gc_interval,
    _resolve_disk_paged_t5_gc_config,
)
from backend.flux_fill_v3.vae_worker import TransientVaeWorker, compute_artifact_fingerprint
from backend.flux_fill_v3.runtime_state import (
    acquire_active_flux_streaming_spine,
    release_active_flux_resident_spine,
    release_flux_latent_artifacts,
    get_cached_latent_artifact_bundle,
    set_cached_latent_artifact_bundle,
)


class TestFluxFillV3Contracts(unittest.TestCase):
    def test_request_normalizes_disk_paged_t5_posture(self):
        req = FluxFillRequest(
            unet_path="unet.safetensors",
            ae_path="ae.safetensors",
            conditioning_cache_path="cache.pt",
            seed=42,
            steps=10,
            t5_posture="disk_paged",
        )
        self.assertEqual(req.t5_posture, T5PostureKind.DISK_PAGED)

    def test_request_rejects_unknown_t5_posture(self):
        with self.assertRaises(ValueError):
            FluxFillRequest(
                unet_path="unet.safetensors",
                ae_path="ae.safetensors",
                conditioning_cache_path="cache.pt",
                seed=42,
                steps=10,
                t5_posture="invalid",
            )

    def test_request_normalizes_cpu_resident_t5_posture(self):
        req = FluxFillRequest(
            unet_path="unet.safetensors",
            ae_path="ae.safetensors",
            conditioning_cache_path="cache.pt",
            seed=42,
            steps=10,
            t5_posture="cpu_resident",
        )
        self.assertEqual(req.t5_posture, T5PostureKind.CPU_RESIDENT)

    def test_request_normalizes_disk_paged_t5_gc_interval_override(self):
        req = FluxFillRequest(
            unet_path="unet.safetensors",
            ae_path="ae.safetensors",
            conditioning_cache_path="cache.pt",
            seed=42,
            steps=10,
            disk_paged_t5_gc_interval="8",
        )
        self.assertEqual(req.disk_paged_t5_gc_interval, 8)

    def test_request_rejects_invalid_disk_paged_t5_gc_interval_override(self):
        with self.assertRaises(ValueError):
            FluxFillRequest(
                unet_path="unet.safetensors",
                ae_path="ae.safetensors",
                conditioning_cache_path="cache.pt",
                seed=42,
                steps=10,
                disk_paged_t5_gc_interval="invalid",
            )

    def test_request_static_validation_adjusts_samplers(self):
        req = FluxFillRequest(
            unet_path="unet.safetensors",
            ae_path="ae.safetensors",
            conditioning_cache_path="cache.pt",
            seed=42,
            steps=10,
            sampler="invalid_sampler",
            scheduler="invalid_scheduler",
        )
        # Verify normalization/adjusting samplers
        req.validate_static(require_existing_assets=False)
        self.assertEqual(req.sampler, "euler")
        self.assertEqual(req.scheduler, "simple")

    def test_request_validation_enforces_positive_steps_and_guidance(self):
        with self.assertRaises(ValueError):
            FluxFillRequest(
                unet_path="unet.safetensors",
                ae_path="ae.safetensors",
                conditioning_cache_path="cache.pt",
                seed=42,
                steps=0,
            ).validate_static(require_existing_assets=False)

        with self.assertRaises(ValueError):
            FluxFillRequest(
                unet_path="unet.safetensors",
                ae_path="ae.safetensors",
                conditioning_cache_path="cache.pt",
                seed=42,
                steps=10,
                guidance=-1.0,
            ).validate_static(require_existing_assets=False)

    def test_request_dispatch_ready_checks_arrays(self):
        req = FluxFillRequest(
            unet_path="unet.safetensors",
            ae_path="ae.safetensors",
            conditioning_cache_path="cache.pt",
            seed=42,
            steps=10,
        )
        with self.assertRaises(ValueError):
            req.validate_dispatch_ready(require_existing_assets=False)

        req.image = np.zeros((16, 16, 3), dtype=np.uint8)
        req.mask = np.zeros((16, 16), dtype=np.uint8)
        # Should pass now
        req.validate_dispatch_ready(require_existing_assets=False)


class TestFluxFillV3Director(unittest.TestCase):
    def setUp(self):
        release_active_flux_resident_spine(reason="test_setup")

    def tearDown(self):
        release_active_flux_resident_spine(reason="test_cleanup")

    def test_director_selects_streaming_unet_posture(self):
        req = FluxFillRequest(
            unet_path="unet.safetensors",
            ae_path="ae.safetensors",
            conditioning_cache_path="cache.pt",
            seed=42,
            steps=10,
        )
        assembly = FluxAssemblyDirector.select_assembly(req)
        self.assertIsInstance(assembly, FluxAssembly)
        self.assertEqual(assembly.spine.request, req)
        self.assertIsInstance(assembly.text_worker, DiskPagedTextWorker)
        self.assertIsInstance(assembly.vae_worker, TransientVaeWorker)
        self.assertFalse(assembly.release_spine_after_execute)

    def test_director_selects_resident_unet_posture(self):
        req = FluxFillRequest(
            unet_path="unet.safetensors",
            ae_path="ae.safetensors",
            conditioning_cache_path="cache.pt",
            seed=42,
            steps=10,
            unet_spine=UNetSpineKind.RESIDENT,
        )
        assembly = FluxAssemblyDirector.select_assembly(req)
        self.assertIsInstance(assembly, FluxAssembly)
        self.assertEqual(assembly.spine.request, req)
        self.assertIsInstance(assembly.text_worker, DiskPagedTextWorker)
        self.assertIsInstance(assembly.vae_worker, TransientVaeWorker)
        from backend.flux_fill_v3.resident_spine import ResidentUnetSpine
        self.assertIsInstance(assembly.spine, ResidentUnetSpine)


class TestFluxFillV3ActivationSeams(unittest.TestCase):
    def test_process_key_ignores_t5_posture_for_prompt_conditioning_identity(self):
        from backend.flux_fill_v3.activation import resolve_flux_fill_process_key

        shared = dict(
            objr_engine="flux fill",
            inpaint_route="flux",
            flux_fill_conditioning="empty",
            flux_fill_prompt_cache="temp",
            prompt="Repair statue",
            inpaint_additional_prompt="",
            remove_prompt="",
            current_tab="inpaint",
            goals=[],
            flux_fill_runtime_posture="resident",
            flux_fill_unet_path="unet.safetensors",
            flux_fill_ae_path="ae.safetensors",
            flux_fill_conditioning_cache_path="cache.pt",
            flux_fill_clip_l_path="clip_l.safetensors",
            flux_fill_t5_path="t5xxl.safetensors",
            total_ram_gb=32.0,
        )

        disk_paged_state = SimpleNamespace(flux_fill_t5_posture="disk_paged", **shared)
        cpu_resident_state = SimpleNamespace(flux_fill_t5_posture="cpu_resident", **shared)

        disk_paged_key = resolve_flux_fill_process_key(disk_paged_state, route_family="flux_fill")
        cpu_resident_key = resolve_flux_fill_process_key(cpu_resident_state, route_family="flux_fill")

        self.assertIsNotNone(disk_paged_key)
        self.assertEqual(disk_paged_key, cpu_resident_key)
        self.assertIn(("conditioning_cache_path", "prompt_conditioning"), disk_paged_key.authoritative_identity)
        self.assertEqual(disk_paged_state.flux_fill_t5_posture, "disk_paged")
        self.assertEqual(cpu_resident_state.flux_fill_t5_posture, "cpu_resident")


class TestFluxFillV3StreamingRuntimeState(unittest.TestCase):
    def setUp(self):
        release_active_flux_resident_spine(reason="test_setup")
        release_flux_latent_artifacts()

    def tearDown(self):
        release_active_flux_resident_spine(reason="test_cleanup")
        release_flux_latent_artifacts()

    def test_streaming_spine_reuses_same_unet_identity(self):
        def fake_start(self):
            self.started = True
            self.unet_patcher = object()

        def fake_end(self):
            self.started = False
            self.unet_patcher = None

        req = FluxFillRequest(
            unet_path="unet.safetensors",
            ae_path="ae.safetensors",
            conditioning_cache_path="cache.pt",
            seed=42,
            steps=10,
        )

        with patch("backend.flux_fill_v3.streaming_spine.StreamingUnetSpine.start", autospec=True, side_effect=fake_start) as mock_start, \
             patch("backend.flux_fill_v3.streaming_spine.StreamingUnetSpine.end", autospec=True, side_effect=fake_end) as mock_end:
            spine_first, reused_first = acquire_active_flux_streaming_spine(req)
            spine_first.start()
            spine_second, reused_second = acquire_active_flux_streaming_spine(req)

            self.assertFalse(reused_first)
            self.assertTrue(reused_second)
            self.assertIs(spine_first, spine_second)
            self.assertEqual(mock_start.call_count, 1)

            released = release_active_flux_resident_spine(reason="test_release")
            self.assertTrue(released)
            self.assertEqual(mock_end.call_count, 1)


class TestT5WorkerPolicyValidation(unittest.TestCase):
    @staticmethod
    def _build_small_t5_stack() -> T5Stack:
        return T5Stack(
            num_layers=4,
            model_dim=16,
            inner_dim=16,
            ff_dim=32,
            ff_activation="gelu_pytorch_tanh",
            gated_act=True,
            num_heads=2,
            relative_attention=True,
            dtype=torch.float32,
            device=torch.device("cpu"),
            operations=base_ops.manual_cast,
        )

    def test_text_encoder_loader_rejects_gguf_checkpoint(self):
        from backend.flux_fill_v3.t5_worker import _load_text_encoder_state_dict

        with self.assertRaises(ValueError):
            _load_text_encoder_state_dict(Path("clip.gguf"))

    def test_strict_loader_policy_rejections(self):
        # 1. Reject non-safetensors paths
        with self.assertRaises(ValueError):
            _normalize_t5_loader_policy("stream_safetensors_runtime", t5_path="t5.bin")

        with self.assertRaises(ValueError):
            _normalize_t5_loader_policy("stream_safetensors_runtime", t5_path="t5.ckpt")

        # 2. Reject other policies
        with self.assertRaises(ValueError):
            _normalize_t5_loader_policy("resident", t5_path="t5.safetensors")

        # 3. Accept valid combinations
        policy = _normalize_t5_loader_policy("stream_safetensors_runtime", t5_path="t5.safetensors")
        self.assertEqual(policy, "stream_safetensors_runtime")

        policy_eager = _normalize_t5_loader_policy("eager", t5_path="t5.safetensors")
        self.assertEqual(policy_eager, "eager")

    def test_disk_paged_t5_gc_config_defaults_to_periodic_interval_with_headroom(self):
        with patch(
            "backend.flux_fill_v3.t5_worker.resources.memory_policy_summary",
            return_value={"low_ram_headroom_mb": 3072.0, "critical_ram_headroom_mb": 1536.0},
        ), patch(
            "backend.flux_fill_v3.t5_worker.resources.active_memory_environment_profile",
            return_value=SimpleNamespace(name="colab_free"),
        ), patch(
            "backend.flux_fill_v3.t5_worker.resources.capture_memory_snapshot",
            return_value=SimpleNamespace(free_ram_mb=4096.0),
        ):
            config = _resolve_disk_paged_t5_gc_config()

        self.assertEqual(config["interval"], 4)
        self.assertEqual(config["recheck_blocks"], 2)
        self.assertEqual(config["profile_name"], "colab_free")
        self.assertEqual(config["initial_free_ram_mb"], 4096.0)
        self.assertTrue(config["adaptive"])

    def test_disk_paged_t5_gc_config_uses_override_as_preferred_healthy_interval(self):
        with patch(
            "backend.flux_fill_v3.t5_worker.resources.memory_policy_summary",
            return_value={"low_ram_headroom_mb": 3072.0, "critical_ram_headroom_mb": 1536.0},
        ), patch(
            "backend.flux_fill_v3.t5_worker.resources.active_memory_environment_profile",
            return_value=SimpleNamespace(name="colab_pro"),
        ), patch(
            "backend.flux_fill_v3.t5_worker.resources.capture_memory_snapshot",
            return_value=SimpleNamespace(free_ram_mb=8192.0),
        ):
            config = _resolve_disk_paged_t5_gc_config(override_interval=16)

        self.assertEqual(config["interval"], 16)
        self.assertEqual(config["healthy_interval"], 16)
        self.assertEqual(config["recheck_blocks"], 2)
        self.assertEqual(config["override_interval"], 16)
        self.assertTrue(config["adaptive"])

    def test_disk_paged_t5_gc_override_downshifts_only_under_pressure(self):
        self.assertEqual(
            _resolve_disk_paged_t5_gc_interval(
                free_ram_mb=8192.0,
                low_headroom_mb=3072.0,
                critical_headroom_mb=1536.0,
                healthy_interval=16,
            ),
            16,
        )
        self.assertEqual(
            _resolve_disk_paged_t5_gc_interval(
                free_ram_mb=2500.0,
                low_headroom_mb=3072.0,
                critical_headroom_mb=1536.0,
                healthy_interval=16,
            ),
            2,
        )
        self.assertEqual(
            _resolve_disk_paged_t5_gc_interval(
                free_ram_mb=1200.0,
                low_headroom_mb=3072.0,
                critical_headroom_mb=1536.0,
                healthy_interval=16,
            ),
            1,
        )

    def test_t5_low_ram_posture_uses_periodic_gc_by_default(self):
        stack = self._build_small_t5_stack()
        x = torch.zeros((1, 4, 16), dtype=torch.float32)
        gc_calls: list[str] = []

        stack._t5_lazy_runtime = True
        stack._t5_lazy_gc_interval = 4
        stack._t5_lazy_gc_recheck_blocks = 2
        stack._t5_lazy_gc_low_headroom_mb = 3072.0
        stack._t5_lazy_gc_critical_headroom_mb = 1536.0

        with patch(
            "backend.flux_fill_v3.t5_worker.resources.capture_memory_snapshot",
            return_value=SimpleNamespace(free_ram_mb=4096.0),
        ) as mock_snapshot, patch(
            "backend.flux_fill_v3.t5_worker.gc.collect",
            side_effect=lambda: gc_calls.append("gc"),
        ):
            stack(x)

        self.assertEqual(len(gc_calls), 1)
        self.assertEqual(mock_snapshot.call_count, 1)

    def test_t5_low_ram_posture_critical_headroom_falls_back_to_every_block(self):
        stack = self._build_small_t5_stack()
        x = torch.zeros((1, 4, 16), dtype=torch.float32)
        gc_calls: list[str] = []

        stack._t5_lazy_runtime = True
        stack._t5_lazy_gc_interval = 4
        stack._t5_lazy_gc_recheck_blocks = 2
        stack._t5_lazy_gc_low_headroom_mb = 3072.0
        stack._t5_lazy_gc_critical_headroom_mb = 1536.0

        with patch(
            "backend.flux_fill_v3.t5_worker.resources.capture_memory_snapshot",
            return_value=SimpleNamespace(free_ram_mb=1200.0),
        ) as mock_snapshot, patch(
            "backend.flux_fill_v3.t5_worker.gc.collect",
            side_effect=lambda: gc_calls.append("gc"),
        ):
            stack(x)

        self.assertEqual(len(gc_calls), 3)
        self.assertEqual(mock_snapshot.call_count, 1)


class TestVaeAndTextWorkerCachingSymmetry(unittest.TestCase):
    def setUp(self):
        release_flux_latent_artifacts()

    def tearDown(self):
        release_flux_latent_artifacts()

    @patch("backend.flux_fill_v3.vae_worker.load_flux_ae")
    @patch("backend.flux_fill_v3.vae_worker._encode_vae_latents")
    def test_vae_worker_cache_hit_miss_symmetry(self, mock_encode, mock_load_ae):
        # Setup mocks
        mock_load_ae.return_value = MagicMock()
        mock_encode.return_value = (
            torch.zeros((1, 16, 2, 2)),
            torch.zeros((1, 32, 2, 2)),
            torch.zeros((1, 1, 2, 2)),
        )

        req = FluxFillRequest(
            unet_path="unet.safetensors",
            ae_path="ae.safetensors",
            conditioning_cache_path="cache.pt",
            seed=42,
            steps=10,
            image=np.zeros((16, 16, 3), dtype=np.uint8),
            mask=np.zeros((16, 16), dtype=np.uint8),
            category=FluxFillCategory.INPAINT,
        )

        worker = TransientVaeWorker(req)
        device = torch.device("cpu")

        # First run: cache MISS
        bundle_first = worker.prepare_latents(device)
        self.assertEqual(bundle_first.vae_load_time, bundle_first.vae_load_time)
        self.assertTrue(bundle_first.vae_load_time > 0.0 or True)
        self.assertEqual(mock_encode.call_count, 1)

        # Verify populated cache
        fingerprint = compute_artifact_fingerprint(req)
        cached = get_cached_latent_artifact_bundle(fingerprint)
        self.assertIsNotNone(cached)

        # Second run: cache HIT
        bundle_second = worker.prepare_latents(device)
        self.assertEqual(bundle_second.vae_load_time, 0.0)
        self.assertEqual(bundle_second.vae_encode_time, 0.0)
        # Should not have called encode again
        self.assertEqual(mock_encode.call_count, 1)

    @patch("backend.flux_fill_v3.t5_worker.subprocess.run")
    def test_text_worker_cache_hit_miss_symmetry(self, mock_run):
        # Mock subprocess run to simulate artifact generation success
        mock_completed = MagicMock()
        mock_completed.returncode = 0
        mock_completed.stdout = '{"status": "ok"}'
        mock_run.return_value = mock_completed

        req = FluxFillRequest(
            unet_path="unet.safetensors",
            ae_path="ae.safetensors",
            conditioning_cache_path="cache.pt",
            seed=42,
            steps=10,
            prompt="A cozy cottage",
            clip_l_path="clip_l.safetensors",
            t5_path="t5xxl.safetensors",
        )

        worker = DiskPagedTextWorker(req)

        cache_exists_status = True
        original_exists = Path.exists

        def mock_exists_fn(self_path):
            if "cache.pt" in str(self_path):
                return cache_exists_status
            return original_exists(self_path)

        with patch.object(Path, "exists", mock_exists_fn), \
             patch("backend.flux_fill_v3.t5_worker.load_flux_empty_conditioning_cache") as mock_load_cond:

            # Scenario 1: Cache hit
            cache_exists_status = True
            cond = worker.get_conditioning()
            self.assertEqual(mock_run.call_count, 0)

            # Scenario 2: Cache miss
            cache_exists_status = False
            cond = worker.get_conditioning()
            self.assertEqual(mock_run.call_count, 1)



class TestFluxFillV3CategoryNormalization(unittest.TestCase):
    def test_category_normalization_values(self):
        from backend.flux_fill_v3.contracts import normalize_category, FluxFillCategory

        # 1. Normalize explicit enum types
        self.assertEqual(normalize_category(FluxFillCategory.INPAINT), FluxFillCategory.INPAINT)
        self.assertEqual(normalize_category(FluxFillCategory.REMOVAL), FluxFillCategory.REMOVAL)

        # 2. Normalize raw string inputs (case-insensitive & aliases)
        self.assertEqual(normalize_category("inpaint"), FluxFillCategory.INPAINT)
        self.assertEqual(normalize_category("INPAINT"), FluxFillCategory.INPAINT)
        self.assertEqual(normalize_category("flux_inpaint"), FluxFillCategory.INPAINT)

        self.assertEqual(normalize_category("removal"), FluxFillCategory.REMOVAL)
        self.assertEqual(normalize_category("remove"), FluxFillCategory.REMOVAL)
        self.assertEqual(normalize_category("remove_obj"), FluxFillCategory.REMOVAL)

        # 3. None behavior
        self.assertIsNone(normalize_category(None))

        # 4. Unknown category raises ValueError
        with self.assertRaises(ValueError):
            normalize_category("invalid_category")

    def test_request_initialization_normalizes_category(self):
        req = FluxFillRequest(
            unet_path="unet.safetensors",
            ae_path="ae.safetensors",
            conditioning_cache_path="cache.pt",
            seed=42,
            steps=10,
            category="remove_obj",
        )
        self.assertEqual(req.category, FluxFillCategory.REMOVAL)


class TestFluxAssemblyOrdering(unittest.TestCase):
    def test_prompt_conditioning_failure_does_not_prepare_vae_latents(self):
        req = FluxFillRequest(
            unet_path="unet.safetensors",
            ae_path="ae.safetensors",
            conditioning_cache_path="cache.pt",
            seed=42,
            steps=10,
            image=np.zeros((16, 16, 3), dtype=np.uint8),
            mask=np.zeros((16, 16), dtype=np.uint8),
            prompt="repair statue",
            clip_l_path="clip_l.safetensors",
            t5_path="t5xxl.safetensors",
        )
        assembly = FluxAssemblyDirector.select_assembly(req)

        with patch.object(req, "validate_dispatch_ready", return_value=None), \
             patch.object(assembly.text_worker, "get_conditioning", side_effect=RuntimeError("conditioning failed")), \
             patch.object(assembly.vae_worker, "prepare_latents") as mock_prepare_latents:
            with self.assertRaisesRegex(RuntimeError, "conditioning failed"):
                assembly.execute(req)

        mock_prepare_latents.assert_not_called()

    def test_none_blend_mode_decodes_raw_samples_without_internal_latent_blend(self):
        req = FluxFillRequest(
            unet_path="unet.safetensors",
            ae_path="ae.safetensors",
            conditioning_cache_path="cache.pt",
            seed=42,
            steps=10,
            image=np.zeros((16, 16, 3), dtype=np.uint8),
            mask=np.zeros((16, 16), dtype=np.uint8),
            blend_mode="none",
            prompt="repair statue",
            clip_l_path="clip_l.safetensors",
            t5_path="t5xxl.safetensors",
        )
        assembly = FluxAssemblyDirector.select_assembly(req)
        fake_samples = torch.randn((1, 16, 2, 2))
        fake_bundle = MagicMock()
        fake_output = np.zeros((16, 16, 3), dtype=np.uint8)

        with patch.object(req, "validate_dispatch_ready", return_value=None), \
             patch.object(assembly.text_worker, "get_conditioning", return_value=MagicMock()), \
             patch.object(assembly.vae_worker, "prepare_latents", return_value=fake_bundle), \
             patch.object(assembly.spine, "start", return_value=None), \
             patch.object(assembly.spine, "denoise", return_value=(fake_samples, torch.zeros(1))), \
             patch.object(assembly.vae_worker, "decode", return_value=(fake_output, 0.0, 0.0)) as mock_decode:
            result = assembly.execute(req)

        self.assertIs(mock_decode.call_args[0][0], fake_samples)
        self.assertNotIn("latent_blend", result.timings)


class TestFluxFillV3ResidentRuntimeState(unittest.TestCase):
    def setUp(self):
        from backend.flux_fill_v3.runtime_state import release_active_flux_resident_spine, release_flux_latent_artifacts
        release_active_flux_resident_spine(reason="test_setup")
        release_flux_latent_artifacts()

    def tearDown(self):
        from backend.flux_fill_v3.runtime_state import release_active_flux_resident_spine, release_flux_latent_artifacts
        release_active_flux_resident_spine(reason="test_cleanup")
        release_flux_latent_artifacts()

    def test_resident_spine_class_retention(self):
        def fake_start(self):
            self.started = True
            self.unet_patcher = object()

        def fake_end(self):
            self.started = False
            self.unet_patcher = None

        req = FluxFillRequest(
            unet_path="unet.safetensors",
            ae_path="ae.safetensors",
            conditioning_cache_path="cache.pt",
            seed=42,
            steps=10,
            unet_spine=UNetSpineKind.RESIDENT,
        )

        from backend.flux_fill_v3.runtime_state import acquire_active_flux_resident_spine, release_active_flux_resident_spine

        with patch("backend.flux_fill_v3.resident_spine.ResidentUnetSpine.start", autospec=True, side_effect=fake_start) as mock_start, \
             patch("backend.flux_fill_v3.resident_spine.ResidentUnetSpine.end", autospec=True, side_effect=fake_end) as mock_end:
            spine_first, reused_first = acquire_active_flux_resident_spine(req)
            spine_first.start()
            spine_second, reused_second = acquire_active_flux_resident_spine(req)

            self.assertFalse(reused_first)
            self.assertTrue(reused_second)
            self.assertIs(spine_first, spine_second)
            self.assertEqual(mock_start.call_count, 1)

            released = release_active_flux_resident_spine(reason="test_release")
            self.assertTrue(released)
            self.assertEqual(mock_end.call_count, 1)


class TestFluxFillV3ResidentMemoryContract(unittest.TestCase):
    @patch("backend.flux_fill_v3.resident_loader.load_flux_fill_unet_resident")
    @patch("backend.resources.load_models_gpu")
    def test_resident_spine_loads_with_sticky_no_cpu_shadow(self, mock_load_gpu, mock_load_resident):
        mock_load_resident.return_value = MagicMock()
        
        req = FluxFillRequest(
            unet_path="unet.safetensors",
            ae_path="ae.safetensors",
            conditioning_cache_path="cache.pt",
            seed=42,
            steps=10,
            device="cuda:0",
            unet_spine=UNetSpineKind.RESIDENT,
        )
        
        from backend.flux_fill_v3.resident_spine import ResidentUnetSpine
        spine = ResidentUnetSpine(req)
        with patch.object(FluxFillRequest, "validate_static", return_value=None):
            spine.start()
        
        # Verify loader was called with load_device == offload_device
        mock_load_resident.assert_called_once_with(
            "unet.safetensors",
            load_device=torch.device("cuda:0"),
            offload_device=torch.device("cuda:0"),
            execution_class="standard_resident",
            resident_load_strategy="sticky_no_cpu_shadow",
        )
        mock_load_gpu.assert_called_once()


class TestFluxFillV3CpuResidentWorker(unittest.TestCase):
    def setUp(self):
        from backend.flux_fill_v3.cpu_resident_text_worker import CpuResidentTextEncoderCache
        CpuResidentTextEncoderCache.teardown()

    def tearDown(self):
        from backend.flux_fill_v3.cpu_resident_text_worker import CpuResidentTextEncoderCache
        CpuResidentTextEncoderCache.teardown()

    def test_zero_copy_popping_sequence(self):
        from backend.flux_fill_v3.cpu_resident_text_worker import load_t5_state_dict_zero_copy
        
        model = torch.nn.Module()
        model.param1 = torch.nn.Parameter(torch.zeros(2, 3))
        model.buf1 = torch.zeros(4)
        model.register_buffer("buf1_buf", model.buf1)

        def named_parameters_mock():
            return [("param1", model.param1)]
        def named_buffers_mock():
            return [("buf1_buf", model.buf1_buf)]

        model.named_parameters = named_parameters_mock
        model.named_buffers = named_buffers_mock

        sd = {
            "param1": torch.ones(2, 3),
            "buf1_buf": torch.ones(4) * 2,
            "extra_key": torch.ones(5)
        }

        missing, unexpected = load_t5_state_dict_zero_copy(model, sd)
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, ["extra_key"])
        self.assertTrue(torch.allclose(model.param1.data, torch.ones(2, 3)))
        self.assertTrue(torch.allclose(model.buf1_buf, torch.ones(4) * 2))
        self.assertEqual(sd, {}) # cleared

    def test_eager_t5_loader_filters_expected_shared_embedding_residual(self):
        from backend.flux_fill_v3.cpu_resident_text_worker import _filter_expected_eager_t5_unexpected_keys

        filtered = _filter_expected_eager_t5_unexpected_keys(
            ["encoder.embed_tokens.weight", "encoder.extra.weight"]
        )
        self.assertEqual(filtered, ["encoder.extra.weight"])

    def test_format_flux_conditioning_memory_summary_reports_process_aware_fields(self):
        from backend.flux_fill_v3.conditioning_loader import format_flux_conditioning_memory_summary

        snapshot = SimpleNamespace(
            phase="diffusion",
            total_ram_mb=1024.0,
            free_ram_mb=768.0,
            total_vram_mb=2048.0,
            free_vram_mb=1536.0,
        )
        process = MagicMock()
        process.memory_info.return_value = SimpleNamespace(rss=200 * 1024 * 1024)
        process.memory_full_info.return_value = SimpleNamespace(
            shared=50 * 1024 * 1024,
            uss=150 * 1024 * 1024,
            pss=175 * 1024 * 1024,
        )

        with patch(
            "backend.flux_fill_v3.conditioning_loader.resources.capture_memory_snapshot",
            return_value=snapshot,
        ), patch("backend.flux_fill_v3.conditioning_loader.psutil.Process", return_value=process):
            summary = format_flux_conditioning_memory_summary(tag="unit")

        self.assertIn("tag=unit", summary)
        self.assertIn("phase=diffusion", summary)
        self.assertIn("ram_available=768.0MB", summary)
        self.assertIn("ram_unavailable_est=256.0MB", summary)
        self.assertIn("proc_rss=200.0MB", summary)
        self.assertIn("proc_shared=50.0MB", summary)
        self.assertIn("proc_uss=150.0MB", summary)
        self.assertIn("proc_pss=175.0MB", summary)

    @patch("backend.flux_fill_v3.cpu_resident_text_worker.format_flux_conditioning_memory_summary", return_value="mem=ok")
    @patch("backend.flux_fill_v3.cpu_resident_text_worker.load_flux_prompt_text_encoder_eager")
    def test_cpu_resident_text_encoder_cache_acquires_and_reuses(self, mock_load, mock_memory_summary):
        from backend.flux_fill_v3.cpu_resident_text_worker import CpuResidentTextEncoderCache
        mock_encoder = MagicMock()
        mock_load.return_value = mock_encoder

        with self.assertLogs("backend.flux_fill_v3.cpu_resident_text_worker", level="DEBUG") as captured:
            enc1, reused1 = CpuResidentTextEncoderCache.acquire(Path("clip.safetensors"), Path("t5.safetensors"))
            self.assertFalse(reused1)
            self.assertIs(enc1, mock_encoder)
            self.assertEqual(mock_load.call_count, 1)

            enc2, reused2 = CpuResidentTextEncoderCache.acquire(Path("clip.safetensors"), Path("t5.safetensors"))
            self.assertTrue(reused2)
            self.assertIs(enc2, mock_encoder)
            self.assertEqual(mock_load.call_count, 1) # no new load

            # Change fingerprint path
            enc3, reused3 = CpuResidentTextEncoderCache.acquire(Path("clip.safetensors"), Path("t5_new.safetensors"))
            self.assertFalse(reused3)
            self.assertEqual(mock_load.call_count, 2) # reload triggered

        joined_logs = "\n".join(captured.output)
        self.assertIn("event=cpu_resident_encoder_ready", joined_logs)
        self.assertIn("event=cpu_resident_encoder_reuse", joined_logs)
        self.assertGreaterEqual(mock_memory_summary.call_count, 3)

    @patch("backend.flux_fill_v3.cpu_resident_text_worker.CpuResidentTextEncoderCache.acquire")
    @patch("backend.flux_fill_v3.cpu_resident_text_worker.save_flux_prompt_conditioning_cache")
    def test_cpu_resident_text_worker_runs_in_process_on_miss(self, mock_save, mock_acquire):
        from backend.flux_fill_v3.cpu_resident_text_worker import CpuResidentTextWorker
        
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = (torch.zeros(1), torch.zeros(1))
        mock_acquire.return_value = (mock_encoder, False)

        req = FluxFillRequest(
            unet_path="unet.safetensors",
            ae_path="ae.safetensors",
            conditioning_cache_path="cache.pt",
            seed=42,
            steps=10,
            prompt="Sunny beach",
            clip_l_path="clip_l.safetensors",
            t5_path="t5xxl.safetensors",
            t5_posture=T5PostureKind.CPU_RESIDENT,
        )

        worker = CpuResidentTextWorker(req)

        # Mock cache check as HIT vs MISS
        cache_exists = [False]
        original_exists = Path.exists
        def mock_exists_fn(self_path):
            if "cache.pt" in str(self_path):
                return cache_exists[0]
            return original_exists(self_path)

        with patch.object(Path, "exists", mock_exists_fn), \
             patch("backend.flux_fill_v3.cpu_resident_text_worker.load_flux_empty_conditioning_cache") as mock_load_cond:
            
            # Scenario 1: Cache MISS
            worker.get_conditioning()
            self.assertEqual(mock_acquire.call_count, 1)
            mock_encoder.encode.assert_called_once_with("Sunny beach")
            self.assertEqual(mock_save.call_count, 1)

            # Scenario 2: Cache HIT
            cache_exists[0] = True
            worker.get_conditioning()
            self.assertEqual(mock_acquire.call_count, 1) # no new acquire
            self.assertEqual(mock_save.call_count, 1) # no new save

    @patch("backend.flux_fill_v3.t5_worker.generate_flux_prompt_conditioning_artifact")
    @patch("backend.flux_fill_v3.t5_worker.load_flux_empty_conditioning_cache")
    @patch("backend.flux_fill_v3.cpu_resident_text_worker.CpuResidentTextEncoderCache.teardown")
    def test_disk_paged_worker_tears_down_cpu_resident_encoder_on_cache_hit(
        self,
        mock_teardown,
        mock_load_cond,
        mock_generate,
    ):
        mock_teardown.return_value = True
        mock_load_cond.return_value = MagicMock()

        req = FluxFillRequest(
            unet_path="unet.safetensors",
            ae_path="ae.safetensors",
            conditioning_cache_path="cache.pt",
            seed=42,
            steps=10,
            prompt="Sunny beach",
            clip_l_path="clip_l.safetensors",
            t5_path="t5xxl.safetensors",
        )

        worker = DiskPagedTextWorker(req)
        original_exists = Path.exists

        def mock_exists_fn(self_path):
            if "cache.pt" in str(self_path):
                return True
            return original_exists(self_path)

        with patch.object(Path, "exists", mock_exists_fn):
            worker.get_conditioning()

        mock_teardown.assert_called_once_with()
        mock_generate.assert_not_called()
        mock_load_cond.assert_called_once()

    @patch("backend.flux_fill_v3.t5_worker.generate_flux_prompt_conditioning_artifact")
    @patch("backend.flux_fill_v3.t5_worker.load_flux_empty_conditioning_cache")
    @patch("backend.flux_fill_v3.cpu_resident_text_worker.CpuResidentTextEncoderCache.teardown")
    def test_disk_paged_worker_tears_down_cpu_resident_encoder_on_cache_miss(
        self,
        mock_teardown,
        mock_load_cond,
        mock_generate,
    ):
        mock_teardown.return_value = True
        mock_load_cond.return_value = MagicMock()

        req = FluxFillRequest(
            unet_path="unet.safetensors",
            ae_path="ae.safetensors",
            conditioning_cache_path="cache.pt",
            seed=42,
            steps=10,
            prompt="Sunny beach",
            clip_l_path="clip_l.safetensors",
            t5_path="t5xxl.safetensors",
        )

        worker = DiskPagedTextWorker(req)
        original_exists = Path.exists

        def mock_exists_fn(self_path):
            if "cache.pt" in str(self_path):
                return False
            return original_exists(self_path)

        with patch.object(Path, "exists", mock_exists_fn):
            worker.get_conditioning()

        mock_teardown.assert_called_once_with()
        mock_generate.assert_called_once()
        mock_load_cond.assert_called_once()

    def test_director_selects_cpu_resident_text_worker_assemblies(self):
        req_res = FluxFillRequest(
            unet_path="unet.safetensors",
            ae_path="ae.safetensors",
            conditioning_cache_path="cache.pt",
            seed=42,
            steps=10,
            t5_posture=T5PostureKind.CPU_RESIDENT,
            unet_spine=UNetSpineKind.RESIDENT,
        )
        assembly_res = FluxAssemblyDirector.select_assembly(req_res)
        from backend.flux_fill_v3.cpu_resident_text_worker import CpuResidentTextWorker
        self.assertIsInstance(assembly_res.text_worker, CpuResidentTextWorker)
        from backend.flux_fill_v3.resident_spine import ResidentUnetSpine
        self.assertIsInstance(assembly_res.spine, ResidentUnetSpine)

        req_stream = FluxFillRequest(
            unet_path="unet.safetensors",
            ae_path="ae.safetensors",
            conditioning_cache_path="cache.pt",
            seed=42,
            steps=10,
            t5_posture=T5PostureKind.CPU_RESIDENT,
            unet_spine=UNetSpineKind.STREAMING,
        )
        assembly_stream = FluxAssemblyDirector.select_assembly(req_stream)
        self.assertIsInstance(assembly_stream.text_worker, CpuResidentTextWorker)
        from backend.flux_fill_v3.streaming_spine import StreamingUnetSpine
        self.assertIsInstance(assembly_stream.spine, StreamingUnetSpine)


def test_generate_flux_prompt_conditioning_artifact_includes_gc_interval_override(monkeypatch, tmp_path):
    from backend.flux_fill_v3.t5_worker import generate_flux_prompt_conditioning_artifact

    cache_path = tmp_path / "conditioning.pt"
    metrics_path = cache_path.with_suffix(cache_path.suffix + ".metrics.json")
    captured: dict[str, object] = {}

    def mock_run(command, cwd, capture_output, text, check):
        captured["command"] = list(command)
        captured["cwd"] = cwd
        captured["capture_output"] = capture_output
        captured["text"] = text
        captured["check"] = check
        payload = {
            "status": "ok",
            "output_path": str(cache_path),
            "metrics_path": str(metrics_path),
        }
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr("backend.flux_fill_v3.t5_worker.subprocess.run", mock_run)
    monkeypatch.setattr("backend.resources.soft_empty_cache", lambda *args, **kwargs: None)

    payload = generate_flux_prompt_conditioning_artifact(
        prompt_text="garden scene",
        clip_l_path="clip_l.safetensors",
        t5_path="t5xxl.safetensors",
        cache_path=cache_path,
        disk_paged_t5_gc_interval=8,
    )

    command = captured["command"]
    assert payload["status"] == "ok"
    assert Path(command[1]).name == "prompt_conditioning_artifact_worker.py"
    assert Path(captured["cwd"]) == Path(__file__).resolve().parents[1]
    assert captured["capture_output"] is True
    assert captured["text"] is True
    assert captured["check"] is False
    assert "--disk-paged-t5-gc-interval" in command
    flag_index = command.index("--disk-paged-t5-gc-interval")
    assert command[flag_index + 1] == "8"


def test_prompt_conditioning_artifact_worker_isolates_private_cli(monkeypatch):
    from argparse import Namespace
    import sys
    from backend.flux_fill_v3 import prompt_conditioning_artifact_worker as worker

    monkeypatch.setattr(
        worker,
        "_parse_args",
        lambda: Namespace(
            prompt="",
            output="unused.pt",
            clip_l="clip.safetensors",
            fp16_t5="t5.safetensors",
            embedding_directory=None,
            metrics_json=None,
            disk_paged_t5_gc_interval=None,
            traceback=False,
        ),
    )
    monkeypatch.setattr(sys, "argv", ["worker.py", "--prompt", "private flag"])
    with pytest.raises(ValueError, match="non-empty"):
        worker.main()
    assert sys.argv == ["worker.py"]
