import os
import sys
import types
import unittest
from unittest.mock import MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock args_manager to avoid argparse system exits
mock_args = types.SimpleNamespace(
    colab=False,
    preset=None,
    output_path=None,
    temp_path=None,
    skip_model_load=False,
    disable_preset_selection=False,
    disable_image_log=False,
)
sys.modules['args_manager'] = types.ModuleType('args_manager')
sys.modules['args_manager'].args = mock_args

from backend.sdxl_runtime_policy import resolve_sdxl_execution_policy, SDXLExecutionPolicy
from backend.staging_manager import PlacementSolver, HardwareTier, ExecutionClass
from backend import process_transition

class TestW11PolicySimplification(unittest.TestCase):
    def setUp(self):
        process_transition.clear_active_process_key()

    def tearDown(self):
        process_transition.clear_active_process_key()

    def test_hardware_tier_reclassification_and_colab_free(self):
        # 6GB VRAM -> LOW_VRAM
        tier_low = PlacementSolver.get_hardware_tier(6144.0, 16384.0)
        self.assertEqual(tier_low, HardwareTier.LOW_VRAM)

        # 8GB VRAM -> NORMAL_VRAM
        tier_mid = PlacementSolver.get_hardware_tier(8192.0, 16384.0)
        self.assertEqual(tier_mid, HardwareTier.NORMAL_VRAM)

        # 12GB VRAM -> NORMAL_VRAM
        tier_norm = PlacementSolver.get_hardware_tier(12288.0, 16384.0)
        self.assertEqual(tier_norm, HardwareTier.NORMAL_VRAM)

        # 16GB VRAM, Low RAM (Colab Free) -> HIGH_VRAM
        tier_colab = PlacementSolver.get_hardware_tier(16384.0, 12000.0)
        self.assertEqual(tier_colab, HardwareTier.HIGH_VRAM)

        # 16GB VRAM, High RAM -> HIGH_VRAM
        tier_high = PlacementSolver.get_hardware_tier(16384.0, 32768.0)
        self.assertEqual(tier_high, HardwareTier.HIGH_VRAM)

    def test_resolve_sdxl_execution_policy_disables_clean_shadow(self):
        # 1. Standard Resident with 32GB system RAM -> clean-shadow policy stays disabled
        profile_high_ram = types.SimpleNamespace(total_vram_mb=16384.0, total_ram_mb=32768.0)
        policy_high = resolve_sdxl_execution_policy(
            architecture="sdxl",
            base_model_name="model.safetensors",
            profile=profile_high_ram
        )
        self.assertFalse(policy_high.allow_cpu_shadow)
        self.assertEqual(policy_high.runtime_family, "unified_sdxl")
        self.assertEqual(policy_high.execution_mode, "resident")
        self.assertEqual(policy_high.execution_class, ExecutionClass.SDXL_RESIDENT_T2)

        # 2. Standard Resident with 16GB system RAM -> same resident class, no clean-shadow divergence
        profile_low_ram = types.SimpleNamespace(total_vram_mb=16384.0, total_ram_mb=16384.0)
        policy_low = resolve_sdxl_execution_policy(
            architecture="sdxl",
            base_model_name="model.safetensors",
            profile=profile_low_ram
        )
        self.assertFalse(policy_low.allow_cpu_shadow)
        self.assertEqual(policy_low.execution_class, ExecutionClass.SDXL_RESIDENT_T2)

        # 3. Standard Streaming with 32GB system RAM -> allow_cpu_shadow = False
        profile_streaming = types.SimpleNamespace(total_vram_mb=4096.0, total_ram_mb=32768.0)
        policy_stream = resolve_sdxl_execution_policy(
            architecture="sdxl",
            base_model_name="model.safetensors",
            profile=profile_streaming
        )
        self.assertFalse(policy_stream.allow_cpu_shadow)

    def test_colab_free_clip_gpu(self):
        profile_colab = types.SimpleNamespace(total_vram_mb=16384.0, total_ram_mb=12000.0)
        policy_colab = resolve_sdxl_execution_policy(
            architecture="sdxl",
            base_model_name="model.safetensors",
            profile=profile_colab
        )
        # CLIP stays CPU-side unconditionally in resident mode under simplified W14c contract
        self.assertFalse(policy_colab.prefer_clip_gpu)

    def test_high_vram_policy_keeps_resident_t2_label(self):
        profile_high_vram = types.SimpleNamespace(total_vram_mb=24576.0, total_ram_mb=57344.0)
        policy = resolve_sdxl_execution_policy(
            architecture="sdxl",
            base_model_name="model.safetensors",
            profile=profile_high_vram
        )

        self.assertEqual(policy.execution_mode, "resident")
        self.assertEqual(policy.hardware_tier, HardwareTier.HIGH_VRAM.name)
        self.assertEqual(policy.execution_class, ExecutionClass.SDXL_RESIDENT_T2)

    def test_policy_backward_compatibility_properties(self):
        policy = SDXLExecutionPolicy(
            enabled=True,
            architecture="sdxl",
            runtime_family="unified_sdxl",
            execution_mode="resident",
            hardware_tier="HIGH_VRAM",
            allow_cpu_shadow=True,
            prefer_clip_gpu=True,
            prefer_gpu_vae_encode=True
        )

        self.assertEqual(policy.execution_family, "standard_sdxl")
        self.assertEqual(policy.residency_class, "full_resident")
        self.assertEqual(policy.clip_residency_mode, "gpu_resident")
        self.assertEqual(policy.vae_encode_mode, "transient_gpu")
        self.assertTrue(policy.keep_clip_loaded)
        self.assertEqual(policy.execution_class, ExecutionClass.SDXL_RESIDENT_T2)

    def test_process_transition_evaluation(self):
        # 1. Base case: transition to standard SDXL from None
        key_sdxl = process_transition.build_process_key(
            family="sdxl",
            process_class="standard_sdxl",
            authoritative_identity="sdxl_id"
        )
        dec1 = process_transition.evaluate_process_transition(key_sdxl)
        self.assertEqual(dec1.action, "start")

        # Set active process
        process_transition.set_active_process_key(key_sdxl)

        # 2. Same process -> reuse
        dec_reuse = process_transition.evaluate_process_transition(key_sdxl)
        self.assertEqual(dec_reuse.action, "reuse")
        self.assertFalse(dec_reuse.reset_required)

        # 3. Family change -> reset (family_change)
        key_flux = process_transition.build_process_key(
            family="flux_fill",
            process_class="flux_fill",
            authoritative_identity="flux_id"
        )
        dec_fam = process_transition.evaluate_process_transition(key_flux)
        self.assertEqual(dec_fam.action, "reset")
        self.assertEqual(dec_fam.reason, "family_change")

        # 4. Process class change -> reset (process_class_change)
        key_alternate = process_transition.build_process_key(
            family="sdxl",
            process_class="alternate_sdxl",
            authoritative_identity="sdxl_id"
        )
        dec_class = process_transition.evaluate_process_transition(key_alternate)
        self.assertEqual(dec_class.action, "reset")
        self.assertEqual(dec_class.reason, "process_class_change")

        # Clean registry
        process_transition.clear_active_process_key()

if __name__ == "__main__":
    unittest.main()
