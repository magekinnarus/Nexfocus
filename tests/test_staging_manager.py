import unittest

from backend.staging_manager import (
    ExecutionClass,
    FLUX_FILL_STREAMING_PROFILE_OPEN_C128_D1_S1,
    FLUX_FILL_STREAMING_PROFILE_OPEN_C64_D1_S1,
    FLUX_RESIDENT_LOAD_STANDARD,
    FLUX_RESIDENT_LOAD_STICKY_NO_CPU_SHADOW,
    FLUX_RUNTIME_FAMILY_NATIVE_FP8,
    FLUX_RUNTIME_POSTURE_RESIDENT,
    FLUX_RUNTIME_POSTURE_STREAMING,
    HardwareTier,
    PlacementSolver,
    ResidencyMode,
    ResourceLedger,
)


class TestStagingManager(unittest.TestCase):
    def test_hardware_tier_classification(self):
        self.assertEqual(PlacementSolver.get_hardware_tier(3000), HardwareTier.LOW_VRAM)
        self.assertEqual(PlacementSolver.get_hardware_tier(6000), HardwareTier.LOW_VRAM)
        self.assertEqual(PlacementSolver.get_hardware_tier(8000), HardwareTier.NORMAL_VRAM)
        self.assertEqual(PlacementSolver.get_hardware_tier(12000), HardwareTier.NORMAL_VRAM)
        self.assertEqual(PlacementSolver.get_hardware_tier(15360), HardwareTier.HIGH_VRAM)
        self.assertEqual(PlacementSolver.get_hardware_tier(16000, 12000), HardwareTier.HIGH_VRAM)
        self.assertEqual(PlacementSolver.get_hardware_tier(16000, 16384), HardwareTier.HIGH_VRAM)
        self.assertEqual(PlacementSolver.get_hardware_tier(24000), HardwareTier.HIGH_VRAM)

    def test_sdxl_streaming_t1_uses_pinned_cpu_unet(self):
        plan = PlacementSolver.solve(4096, 16384, "sdxl")

        self.assertEqual(plan.execution_class, ExecutionClass.SDXL_STREAMING_T1)
        self.assertEqual(plan.tier, HardwareTier.LOW_VRAM)
        self.assertEqual(plan.model_variant, "sdxl_fp16")
        self.assertEqual(plan.unet.residency_mode, ResidencyMode.CPU_PINNED_STREAMING.value)
        self.assertEqual(plan.unet.mode, ResidencyMode.CPU_RESIDENT)
        self.assertEqual(plan.unet.device.type, "cpu")
        self.assertEqual(plan.unet.load_device, "pinned_cpu")
        self.assertGreater(plan.unet.pinned_cpu_mb, 0.0)
        self.assertGreater(plan.unet.transient_gpu_mb, 0.0)
        self.assertIn("unet", plan.phase_plans["diffusion"].required_components)

    def test_sdxl_odd_7gb_biases_to_streaming_unified_fp16(self):
        plan = PlacementSolver.solve(7168, 16384, "sdxl")

        self.assertEqual(plan.execution_class, ExecutionClass.SDXL_STREAMING_T1)
        self.assertEqual(plan.tier, HardwareTier.NORMAL_VRAM)
        self.assertEqual(plan.model_variant, "sdxl_fp16")
        self.assertEqual(plan.unet.residency_mode, ResidencyMode.CPU_PINNED_STREAMING.value)
        self.assertEqual(plan.vae.residency_mode, ResidencyMode.TRANSIENT_GPU.value)

    def test_sdxl_resident_t2_keeps_unet_on_gpu(self):
        plan = PlacementSolver.solve(8192, 16384, "sdxl")

        self.assertEqual(plan.execution_class, ExecutionClass.SDXL_RESIDENT_T2)
        self.assertEqual(plan.tier, HardwareTier.NORMAL_VRAM)
        self.assertEqual(plan.model_variant, "sdxl_fp16")
        self.assertEqual(plan.unet.residency_mode, ResidencyMode.GPU_RESIDENT.value)
        self.assertEqual(plan.unet.mode, ResidencyMode.GPU_RESIDENT)
        self.assertEqual(plan.unet.device.type, "cuda")
        self.assertEqual(plan.vae.residency_mode, ResidencyMode.TRANSIENT_GPU.value)
        self.assertEqual(plan.vae.load_device, "cpu")
        self.assertEqual(plan.vae.compute_device, "cuda")

    def test_sdxl_high_vram_reuses_resident_t2_contract(self):
        plan = PlacementSolver.solve(16384, 16384, "sdxl")

        self.assertEqual(plan.execution_class, ExecutionClass.SDXL_RESIDENT_T2)
        self.assertEqual(plan.tier, HardwareTier.HIGH_VRAM)
        self.assertEqual(plan.model_variant, "sdxl_fp16")
        self.assertEqual(plan.unet.residency_mode, ResidencyMode.GPU_RESIDENT.value)
        self.assertEqual(plan.phase_plans["prompt_encode"].preferred_gpu_components, ())
        self.assertEqual(plan.clip.preferred_gpu_mb, 0.0)
        self.assertEqual(plan.vae.residency_mode, ResidencyMode.TRANSIENT_GPU.value)
        self.assertEqual(plan.vae.load_device, "cpu")
        self.assertEqual(plan.vae.compute_device, "cuda")
        self.assertEqual(plan.vae.preferred_gpu_mb, 0.0)

    def test_concrete_variant_ids_do_not_override_execution_class_policy(self):
        sdxl_plan = PlacementSolver.solve(16384, 16384, "sdxl_q8")
        flux_t4_plan = PlacementSolver.solve(12288, 16384, "flux_fill_fp8")
        flux_t6_plan = PlacementSolver.solve(24576, 32768, "flux_fill_fp8")

        self.assertEqual(sdxl_plan.execution_class, ExecutionClass.SDXL_RESIDENT_T2)
        self.assertEqual(sdxl_plan.model_variant, "sdxl_fp16")

        self.assertEqual(flux_t4_plan.execution_class, ExecutionClass.FLUX_STREAMING_T3)
        self.assertEqual(flux_t4_plan.model_variant, "flux_fill_fp8")
        self.assertEqual(flux_t4_plan.runtime_posture, FLUX_RUNTIME_POSTURE_STREAMING)

        self.assertEqual(flux_t6_plan.execution_class, ExecutionClass.FLUX_RESIDENT_T6)
        self.assertEqual(flux_t6_plan.model_variant, "flux_fill_fp8")
        self.assertEqual(flux_t6_plan.runtime_posture, FLUX_RUNTIME_POSTURE_RESIDENT)

    def test_flux_streaming_t3_cpu_pins_unet(self):
        plan = PlacementSolver.solve(8192, 16384, "flux_fill")

        self.assertEqual(plan.execution_class, ExecutionClass.FLUX_STREAMING_T3)
        self.assertEqual(plan.tier, HardwareTier.NORMAL_VRAM)
        self.assertEqual(plan.model_variant, "flux_fill_fp8")
        self.assertEqual(plan.unet.residency_mode, ResidencyMode.CPU_PINNED_STREAMING.value)
        self.assertEqual(plan.unet.device.type, "cpu")
        self.assertEqual(plan.unet.load_device, "pinned_cpu")
        self.assertEqual(plan.t5_mode, "disk_paged_fp16")
        self.assertEqual(plan.t5.device.type, "cpu")
        self.assertEqual(plan.t5.mode, ResidencyMode.DISK_PAGED)
        self.assertEqual(plan.runtime_family, FLUX_RUNTIME_FAMILY_NATIVE_FP8)
        self.assertEqual(plan.runtime_posture, FLUX_RUNTIME_POSTURE_STREAMING)
        self.assertEqual(plan.streaming_profile, FLUX_FILL_STREAMING_PROFILE_OPEN_C64_D1_S1)
        self.assertEqual(plan.vae.residency_mode, ResidencyMode.GPU_RESIDENT.value)
        self.assertEqual(plan.vae.load_device, "cuda")
        self.assertEqual(plan.vae.compute_device, "cuda")
        self.assertEqual(plan.fallback_model_variant, None)

    def test_flux_6gb_streaming_uses_64mb_profile_with_transient_vae(self):
        plan = PlacementSolver.solve(6144, 16384, "flux_fill")

        self.assertEqual(plan.execution_class, ExecutionClass.FLUX_STREAMING_T3)
        self.assertEqual(plan.tier, HardwareTier.LOW_VRAM)
        self.assertEqual(plan.model_variant, "flux_fill_fp8")
        self.assertEqual(plan.unet.residency_mode, ResidencyMode.CPU_PINNED_STREAMING.value)
        self.assertEqual(plan.t5_mode, "disk_paged_fp16")
        self.assertEqual(plan.runtime_family, FLUX_RUNTIME_FAMILY_NATIVE_FP8)
        self.assertEqual(plan.runtime_posture, FLUX_RUNTIME_POSTURE_STREAMING)
        self.assertEqual(plan.streaming_profile, FLUX_FILL_STREAMING_PROFILE_OPEN_C64_D1_S1)
        self.assertEqual(plan.vae.residency_mode, ResidencyMode.TRANSIENT_GPU.value)
        self.assertEqual(plan.vae.load_device, "cpu")
        self.assertEqual(plan.vae.compute_device, "cuda")
        self.assertEqual(plan.fallback_model_variant, None)

    def test_flux_10gb_streaming_uses_128mb_profile_with_transient_vae(self):
        plan = PlacementSolver.solve(10240, 16384, "flux_fill")

        self.assertEqual(plan.execution_class, ExecutionClass.FLUX_STREAMING_T3)
        self.assertEqual(plan.tier, HardwareTier.NORMAL_VRAM)
        self.assertEqual(plan.model_variant, "flux_fill_fp8")
        self.assertEqual(plan.unet.residency_mode, ResidencyMode.CPU_PINNED_STREAMING.value)
        self.assertEqual(plan.t5_mode, "disk_paged_fp16")
        self.assertEqual(plan.runtime_family, FLUX_RUNTIME_FAMILY_NATIVE_FP8)
        self.assertEqual(plan.runtime_posture, FLUX_RUNTIME_POSTURE_STREAMING)
        self.assertEqual(plan.streaming_profile, FLUX_FILL_STREAMING_PROFILE_OPEN_C128_D1_S1)
        self.assertEqual(plan.vae.residency_mode, ResidencyMode.GPU_RESIDENT.value)
        self.assertEqual(plan.vae.load_device, "cuda")
        self.assertEqual(plan.vae.compute_device, "cuda")
        self.assertEqual(plan.fallback_model_variant, None)

    def test_flux_t4_uses_fp8_streaming_headroom_profile(self):
        plan = PlacementSolver.solve(12288, 16384, "flux_fill")

        self.assertEqual(plan.execution_class, ExecutionClass.FLUX_STREAMING_T3)
        self.assertEqual(plan.tier, HardwareTier.NORMAL_VRAM)
        self.assertEqual(plan.model_variant, "flux_fill_fp8")
        self.assertEqual(plan.unet.residency_mode, ResidencyMode.CPU_PINNED_STREAMING.value)
        self.assertEqual(plan.unet.device.type, "cpu")
        self.assertEqual(plan.t5_mode, "disk_paged_fp16")
        self.assertEqual(plan.runtime_family, FLUX_RUNTIME_FAMILY_NATIVE_FP8)
        self.assertEqual(plan.runtime_posture, FLUX_RUNTIME_POSTURE_STREAMING)
        self.assertEqual(plan.streaming_profile, FLUX_FILL_STREAMING_PROFILE_OPEN_C128_D1_S1)
        self.assertEqual(plan.vae.residency_mode, ResidencyMode.GPU_RESIDENT.value)
        self.assertEqual(plan.vae.load_device, "cuda")
        self.assertEqual(plan.vae.compute_device, "cuda")
        self.assertEqual(plan.fallback_model_variant, None)

    def test_flux_15gb_t4_prefers_resident_posture(self):
        plan = PlacementSolver.solve(15360, 16384, "flux_fill")

        self.assertEqual(plan.execution_class, ExecutionClass.FLUX_RESIDENT_T6)
        self.assertEqual(plan.tier, HardwareTier.HIGH_VRAM)
        self.assertEqual(plan.model_variant, "flux_fill_fp8")
        self.assertEqual(plan.runtime_posture, FLUX_RUNTIME_POSTURE_RESIDENT)
        self.assertEqual(plan.t5_mode, "disk_paged_fp16")
        self.assertEqual(plan.resident_load_strategy, FLUX_RESIDENT_LOAD_STICKY_NO_CPU_SHADOW)

    def test_flux_resident_t5_uses_fp8_unet(self):
        plan = PlacementSolver.solve(16384, 12000, "flux_fill")

        self.assertEqual(plan.execution_class, ExecutionClass.FLUX_RESIDENT_T6)
        self.assertEqual(plan.tier, HardwareTier.HIGH_VRAM)
        self.assertEqual(plan.model_variant, "flux_fill_fp8")
        self.assertEqual(plan.unet.residency_mode, ResidencyMode.GPU_RESIDENT.value)
        self.assertEqual(plan.t5_mode, "disk_paged_fp16")
        self.assertEqual(plan.t5.device.type, "cpu")
        self.assertEqual(plan.t5.mode, ResidencyMode.DISK_PAGED)
        self.assertEqual(plan.runtime_posture, FLUX_RUNTIME_POSTURE_RESIDENT)
        self.assertEqual(plan.resident_load_strategy, FLUX_RESIDENT_LOAD_STICKY_NO_CPU_SHADOW)
        self.assertEqual(plan.fallback_model_variant, None)

    def test_flux_resident_32gb_ram_keeps_disk_paged_t5(self):
        plan = PlacementSolver.solve(16384, 32768, "flux_fill")

        self.assertEqual(plan.execution_class, ExecutionClass.FLUX_RESIDENT_T6)
        self.assertEqual(plan.tier, HardwareTier.HIGH_VRAM)
        self.assertEqual(plan.t5_mode, "disk_paged_fp16")
        self.assertEqual(plan.t5.device.type, "cpu")
        self.assertEqual(plan.t5.mode, ResidencyMode.DISK_PAGED)
        self.assertEqual(plan.runtime_posture, FLUX_RUNTIME_POSTURE_RESIDENT)
        self.assertEqual(plan.resident_load_strategy, FLUX_RESIDENT_LOAD_STICKY_NO_CPU_SHADOW)

    def test_flux_resident_31gb_ram_keeps_disk_paged_t5(self):
        plan = PlacementSolver.solve(16384, 31744, "flux_fill")

        self.assertEqual(plan.execution_class, ExecutionClass.FLUX_RESIDENT_T6)
        self.assertEqual(plan.tier, HardwareTier.HIGH_VRAM)
        self.assertEqual(plan.t5_mode, "disk_paged_fp16")
        self.assertEqual(plan.t5.device.type, "cpu")
        self.assertEqual(plan.t5.mode, ResidencyMode.DISK_PAGED)
        self.assertEqual(plan.runtime_posture, FLUX_RUNTIME_POSTURE_RESIDENT)
        self.assertEqual(plan.resident_load_strategy, FLUX_RESIDENT_LOAD_STICKY_NO_CPU_SHADOW)

    def test_flux_resident_t6_uses_fp8_unet_and_sticky_no_cpu_shadow(self):
        plan = PlacementSolver.solve(24576, 49152, "flux_fill")

        self.assertEqual(plan.execution_class, ExecutionClass.FLUX_RESIDENT_T6)
        self.assertEqual(plan.tier, HardwareTier.HIGH_VRAM)
        self.assertEqual(plan.model_variant, "flux_fill_fp8")
        self.assertEqual(plan.unet.residency_mode, ResidencyMode.GPU_RESIDENT.value)
        self.assertEqual(plan.t5_mode, "disk_paged_fp16")
        self.assertEqual(plan.t5.device.type, "cpu")
        self.assertNotEqual(plan.t5.mode, ResidencyMode.GPU_RESIDENT)
        self.assertEqual(plan.runtime_posture, FLUX_RUNTIME_POSTURE_RESIDENT)
        self.assertEqual(plan.resident_load_strategy, FLUX_RESIDENT_LOAD_STICKY_NO_CPU_SHADOW)
        self.assertEqual(plan.fallback_model_variant, None)

    def test_t5_mode_is_stable_across_ram_bands(self):
        low_ram_plan = PlacementSolver.solve(12288, 12288, "flux_fill")
        mid_ram_plan = PlacementSolver.solve(12288, 16384, "flux_fill")
        high_ram_plan = PlacementSolver.solve(12288, 32768, "flux_fill")
        roomy_plan = PlacementSolver.solve(12288, 49152, "flux_fill")
        resident_31gb_plan = PlacementSolver.solve(16384, 31744, "flux_fill")

        self.assertEqual(low_ram_plan.t5_mode, "disk_paged_fp16")
        self.assertEqual(low_ram_plan.t5.mode, ResidencyMode.DISK_PAGED)
        self.assertEqual(low_ram_plan.t5.device.type, "cpu")
        self.assertEqual(low_ram_plan.component("t5").host_ram_mb, 9334.0)
        self.assertEqual(low_ram_plan.component("t5").pinned_cpu_mb, 0.0)

        self.assertEqual(mid_ram_plan.t5_mode, "disk_paged_fp16")
        self.assertEqual(mid_ram_plan.t5.mode, ResidencyMode.DISK_PAGED)
        self.assertEqual(mid_ram_plan.t5.device.type, "cpu")
        self.assertEqual(mid_ram_plan.component("t5").host_ram_mb, 9334.0)

        self.assertEqual(high_ram_plan.t5_mode, "disk_paged_fp16")
        self.assertEqual(high_ram_plan.t5.mode, ResidencyMode.DISK_PAGED)
        self.assertEqual(high_ram_plan.t5.device.type, "cpu")
        self.assertEqual(high_ram_plan.component("t5").host_ram_mb, 9334.0)

        self.assertEqual(roomy_plan.t5_mode, "disk_paged_fp16")
        self.assertEqual(roomy_plan.t5.mode, ResidencyMode.DISK_PAGED)
        self.assertEqual(roomy_plan.t5.device.type, "cpu")
        self.assertEqual(roomy_plan.component("t5").host_ram_mb, 9334.0)
        self.assertEqual(roomy_plan.component("t5").pinned_cpu_mb, 0.0)

        self.assertEqual(resident_31gb_plan.t5_mode, "disk_paged_fp16")
        self.assertEqual(resident_31gb_plan.t5.mode, ResidencyMode.DISK_PAGED)
        self.assertEqual(resident_31gb_plan.t5.device.type, "cpu")

    def test_current_ledger_reduces_required_headroom(self):
        ledger = ResourceLedger()
        ledger.register_load(
            "unet",
            current_device="cuda",
            gpu_mb=5135.0,
            family="sdxl",
            variant="sdxl_fp16",
            residency_mode=ResidencyMode.GPU_RESIDENT.value,
            fingerprint="warm-unet",
        )

        plan = PlacementSolver.solve(8192, 16384, "sdxl", current_ledger=ledger)

        self.assertIn("unet", plan.reusable_components)
        self.assertEqual(plan.phase_plans["diffusion"].required_headroom_mb, 0.0)

    def test_flux_t5_mode_depends_on_total_ram_not_current_host_usage(self):
        ledger = ResourceLedger()
        ledger.register_load(
            "host-cache",
            current_device="cpu",
            pinned_cpu_mb=28000.0,
            host_ram_mb=28000.0,
            family="misc",
            variant="misc",
            residency_mode=ResidencyMode.CPU_RESIDENT.value,
            fingerprint="host-pressure",
        )

        resident_plan = PlacementSolver.solve(16384, 32768, "flux_fill", current_ledger=ledger)
        streaming_plan = PlacementSolver.solve(12288, 49152, "flux_fill", current_ledger=ledger)

        self.assertEqual(resident_plan.t5_mode, "disk_paged_fp16")
        self.assertEqual(resident_plan.t5.mode, ResidencyMode.DISK_PAGED)
        self.assertEqual(streaming_plan.t5_mode, "disk_paged_fp16")
        self.assertEqual(streaming_plan.t5.mode, ResidencyMode.DISK_PAGED)

    def test_model_refresh_identifies_gpu_blockers_for_eviction(self):
        ledger = ResourceLedger()
        ledger.register_load(
            "other",
            current_device="cuda",
            gpu_mb=11000.0,
            family="misc",
            variant="misc",
            residency_mode=ResidencyMode.GPU_RESIDENT.value,
            fingerprint="blocker",
        )

        plan = PlacementSolver.solve(16384, 16384, "sdxl", current_ledger=ledger)

        self.assertEqual(plan.execution_class, ExecutionClass.SDXL_RESIDENT_T2)
        self.assertIn("other", plan.phase_plans["model_refresh"].evict_ledger_entries)
        self.assertEqual(plan.clip.preferred_gpu_mb, 0.0)


if __name__ == "__main__":
    unittest.main()
