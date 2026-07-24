import unittest
from unittest.mock import MagicMock, patch
import types

from backend.sdxl_assembly.runtime_state import (
    LifecycleDomain,
    release_domain,
    release_model_prompt_caches,
    release_prompt_conditioning_caches,
    release_spatial_vae_caches,
    clear_all_caches,
    _TEXT_ENCODER_COMPONENT_CACHE,
    _PROMPT_CONDITIONING_CACHE,
    _STREAMING_RUNTIME_STATE,
)
import backend.sdxl_assembly.runtime_state as runtime_state
from backend.sdxl_assembly.lifecycle_coordinator import (
    LifecycleChange,
    plan_release_for_changes,
    release_for_changes,
)
from backend.sdxl_assembly.vae_encode_worker import VaeEncodeWorker
from backend.sdxl_assembly.stream_st_preprocess_worker import StreamingStructuralPreprocessWorker
from backend.sdxl_assembly.stream_st_cn_worker import StreamingStructuralControlWorker
from backend.sdxl_assembly.stream_ctx_cn_worker import StreamingContextualControlWorker
from backend.sdxl_assembly.cpu_lora_worker import _PARSED_LORA_CACHE
from backend.sdxl_assembly.assembly import SDXLAssembly


class TestSDXLAssemblyW10b(unittest.TestCase):
    def setUp(self):
        # Populate all caches with dummy entries
        _TEXT_ENCODER_COMPONENT_CACHE.clear()
        _TEXT_ENCODER_COMPONENT_CACHE["dummy_clip"] = MagicMock()
        runtime_state._PATCHED_TEXT_ENCODER_COMPONENT_SLOT = MagicMock()
        runtime_state._PATCHED_TEXT_ENCODER_COMPONENT_SLOT_KEY = ("dummy_checkpoint", "cpu_pinned", (("dummy_lora", 1.0),))

        _PROMPT_CONDITIONING_CACHE.clear()
        _PROMPT_CONDITIONING_CACHE[("dummy_key",)] = "dummy_cond"

        _STREAMING_RUNTIME_STATE._spine = MagicMock()
        _STREAMING_RUNTIME_STATE._key = "dummy_spine_key"

        _PARSED_LORA_CACHE.clear()
        _PARSED_LORA_CACHE[("lora_path", "unet", "model")] = ("hash", {})

        VaeEncodeWorker._ENCODE_CACHE.clear()
        VaeEncodeWorker._ENCODE_CACHE["dummy_vae"] = {}

        StreamingStructuralPreprocessWorker._PREPROCESS_CACHE.clear()
        StreamingStructuralPreprocessWorker._PREPROCESS_CACHE["dummy_prep"] = MagicMock()

        StreamingStructuralControlWorker._SUPPORT_MODEL_CACHE.clear()
        StreamingStructuralControlWorker._SUPPORT_MODEL_CACHE["dummy_support"] = MagicMock()

        StreamingContextualControlWorker._PAYLOAD_CACHE.clear()
        StreamingContextualControlWorker._PAYLOAD_CACHE["dummy_payload"] = MagicMock()

        StreamingContextualControlWorker._CONTEXTUAL_MODELS.clear()
        StreamingContextualControlWorker._CONTEXTUAL_MODELS["dummy_model"] = MagicMock()

        StreamingContextualControlWorker._CLIP_VISION_MODELS.clear()
        StreamingContextualControlWorker._CLIP_VISION_MODELS["dummy_clip"] = MagicMock()

        StreamingContextualControlWorker._IP_NEGATIVES.clear()
        StreamingContextualControlWorker._IP_NEGATIVES["dummy_neg"] = MagicMock()

        StreamingContextualControlWorker._EVA_CLIP_MODELS.clear()
        StreamingContextualControlWorker._EVA_CLIP_MODELS["dummy_eva"] = MagicMock()

        StreamingContextualControlWorker._FACE_PARSERS.clear()
        StreamingContextualControlWorker._FACE_PARSERS["dummy_parser"] = MagicMock()

        StreamingContextualControlWorker._INSIGHTFACE_APPS.clear()
        StreamingContextualControlWorker._INSIGHTFACE_APPS["dummy_insight"] = MagicMock()

    def tearDown(self):
        # Clean up all caches after each test
        clear_all_caches()

    def _assert_model_prompt_cleared(self):
        self.assertEqual(len(_TEXT_ENCODER_COMPONENT_CACHE), 0)
        self.assertIsNone(runtime_state._PATCHED_TEXT_ENCODER_COMPONENT_SLOT)
        self.assertIsNone(runtime_state._PATCHED_TEXT_ENCODER_COMPONENT_SLOT_KEY)
        self.assertEqual(len(_PROMPT_CONDITIONING_CACHE), 0)
        self.assertIsNone(_STREAMING_RUNTIME_STATE._spine)
        self.assertIsNone(_STREAMING_RUNTIME_STATE._key)
        self.assertEqual(len(_PARSED_LORA_CACHE), 0)

    def _assert_model_prompt_preserved(self):
        self.assertGreater(len(_TEXT_ENCODER_COMPONENT_CACHE), 0)
        self.assertIsNotNone(runtime_state._PATCHED_TEXT_ENCODER_COMPONENT_SLOT)
        self.assertIsNotNone(runtime_state._PATCHED_TEXT_ENCODER_COMPONENT_SLOT_KEY)
        self.assertGreater(len(_PROMPT_CONDITIONING_CACHE), 0)
        self.assertIsNotNone(_STREAMING_RUNTIME_STATE._spine)
        self.assertIsNotNone(_STREAMING_RUNTIME_STATE._key)
        self.assertGreater(len(_PARSED_LORA_CACHE), 0)

    def _assert_prompt_conditioning_cleared(self):
        self.assertEqual(len(_PROMPT_CONDITIONING_CACHE), 0)

    def _assert_prompt_conditioning_preserved(self):
        self.assertGreater(len(_PROMPT_CONDITIONING_CACHE), 0)

    def _assert_spatial_vae_cleared(self):
        self.assertEqual(len(VaeEncodeWorker._ENCODE_CACHE), 0)

    def _assert_spatial_vae_preserved(self):
        self.assertGreater(len(VaeEncodeWorker._ENCODE_CACHE), 0)

    def _assert_structural_cn_cleared(self):
        self.assertEqual(len(StreamingStructuralPreprocessWorker._PREPROCESS_CACHE), 0)
        self.assertEqual(len(StreamingStructuralControlWorker._SUPPORT_MODEL_CACHE), 0)

    def _assert_structural_cn_preserved(self):
        self.assertGreater(len(StreamingStructuralPreprocessWorker._PREPROCESS_CACHE), 0)
        self.assertGreater(len(StreamingStructuralControlWorker._SUPPORT_MODEL_CACHE), 0)

    def _assert_contextual_cn_cleared(self):
        self.assertEqual(len(StreamingContextualControlWorker._PAYLOAD_CACHE), 0)
        self.assertEqual(len(StreamingContextualControlWorker._CONTEXTUAL_MODELS), 0)
        self.assertEqual(len(StreamingContextualControlWorker._CLIP_VISION_MODELS), 0)
        self.assertEqual(len(StreamingContextualControlWorker._IP_NEGATIVES), 0)
        self.assertEqual(len(StreamingContextualControlWorker._EVA_CLIP_MODELS), 0)
        self.assertEqual(len(StreamingContextualControlWorker._FACE_PARSERS), 0)
        self.assertEqual(len(StreamingContextualControlWorker._INSIGHTFACE_APPS), 0)

    def _assert_contextual_cn_preserved(self):
        self.assertGreater(len(StreamingContextualControlWorker._PAYLOAD_CACHE), 0)
        self.assertGreater(len(StreamingContextualControlWorker._CONTEXTUAL_MODELS), 0)
        self.assertGreater(len(StreamingContextualControlWorker._CLIP_VISION_MODELS), 0)
        self.assertGreater(len(StreamingContextualControlWorker._IP_NEGATIVES), 0)
        self.assertGreater(len(StreamingContextualControlWorker._EVA_CLIP_MODELS), 0)
        self.assertGreater(len(StreamingContextualControlWorker._FACE_PARSERS), 0)
        self.assertGreater(len(StreamingContextualControlWorker._INSIGHTFACE_APPS), 0)

    def test_model_prompt_release_isolation(self):
        release_domain(LifecycleDomain.MODEL_PROMPT, reason="test")
        self._assert_model_prompt_cleared()
        self._assert_spatial_vae_preserved()
        self._assert_structural_cn_preserved()
        self._assert_contextual_cn_preserved()

    def test_prompt_conditioning_release_isolation(self):
        release_domain(LifecycleDomain.PROMPT_CONDITIONING, reason="test")
        self._assert_prompt_conditioning_cleared()
        self.assertGreater(len(_TEXT_ENCODER_COMPONENT_CACHE), 0)
        self.assertIsNotNone(runtime_state._PATCHED_TEXT_ENCODER_COMPONENT_SLOT)
        self.assertIsNotNone(runtime_state._PATCHED_TEXT_ENCODER_COMPONENT_SLOT_KEY)
        self.assertIsNotNone(_STREAMING_RUNTIME_STATE._spine)
        self.assertIsNotNone(_STREAMING_RUNTIME_STATE._key)
        self.assertGreater(len(_PARSED_LORA_CACHE), 0)
        self._assert_spatial_vae_preserved()
        self._assert_structural_cn_preserved()
        self._assert_contextual_cn_preserved()

    def test_spatial_vae_release_isolation(self):
        release_domain(LifecycleDomain.SPATIAL_VAE, reason="test")
        self._assert_model_prompt_preserved()
        self._assert_spatial_vae_cleared()
        self._assert_structural_cn_preserved()
        self._assert_contextual_cn_preserved()

    def test_structural_cn_release_isolation(self):
        release_domain(LifecycleDomain.STRUCTURAL_CN, reason="test")
        self._assert_model_prompt_preserved()
        self._assert_spatial_vae_preserved()
        self._assert_structural_cn_cleared()
        self._assert_contextual_cn_preserved()

    def test_contextual_cn_release_isolation(self):
        release_domain(LifecycleDomain.CONTEXTUAL_CN, reason="test")
        self._assert_model_prompt_preserved()
        self._assert_spatial_vae_preserved()
        self._assert_structural_cn_preserved()
        self._assert_contextual_cn_cleared()

    def test_full_teardown_clears_all(self):
        release_domain(LifecycleDomain.FULL_TEARDOWN, reason="test")
        self._assert_model_prompt_cleared()
        self._assert_spatial_vae_cleared()
        self._assert_structural_cn_cleared()
        self._assert_contextual_cn_cleared()

    def test_full_teardown_continues_after_spine_release_failure(self):
        _STREAMING_RUNTIME_STATE._spine.release_owned_resources.side_effect = RuntimeError("spine boom")

        result = release_domain(LifecycleDomain.FULL_TEARDOWN, reason="test_failure_isolation")

        self.assertFalse(result.ok)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0].domain, LifecycleDomain.MODEL_PROMPT)
        self.assertEqual(result.errors[0].step, "streaming_spine")
        self._assert_model_prompt_cleared()
        self._assert_spatial_vae_cleared()
        self._assert_structural_cn_cleared()
        self._assert_contextual_cn_cleared()

    def test_run_bound_release_closes_assembly_without_clearing_warm_domains(self):
        assembly = MagicMock()

        result = release_domain(LifecycleDomain.RUN_BOUND, reason="request_end", assembly=assembly)

        self.assertTrue(result.ok)
        assembly.close.assert_called_once()
        self._assert_model_prompt_preserved()
        self._assert_spatial_vae_preserved()
        self._assert_structural_cn_preserved()
        self._assert_contextual_cn_preserved()

    def test_lifecycle_change_planner_maps_changes_to_domains(self):
        plan = plan_release_for_changes(
            [
                LifecycleChange.PROMPT_CHANGE,
            ],
            reason="prompt_only_refresh",
        )
        self.assertEqual(plan.domains, (LifecycleDomain.PROMPT_CONDITIONING,))

        plan = plan_release_for_changes(
            [
                LifecycleChange.LORA_STACK_CHANGE,
                LifecycleChange.MODEL_CHANGE,
                LifecycleChange.SPINE_POSTURE_CHANGE,
            ],
            reason="same_family_model_prompt",
        )
        self.assertEqual(plan.domains, (LifecycleDomain.MODEL_PROMPT,))

        plan = plan_release_for_changes(
            [
                LifecycleChange.SPATIAL_VAE_CHANGE,
                LifecycleChange.STRUCTURAL_CN_CHANGE,
                LifecycleChange.CONTEXTUAL_CN_CHANGE,
            ],
            reason="artifact_refresh",
        )
        self.assertEqual(
            plan.domains,
            (
                LifecycleDomain.SPATIAL_VAE,
                LifecycleDomain.STRUCTURAL_CN,
                LifecycleDomain.CONTEXTUAL_CN,
            ),
        )

        plan = plan_release_for_changes(
            [LifecycleChange.MODEL_TYPE_CHANGE],
            reason="family_transition",
        )
        self.assertEqual(plan.domains, (LifecycleDomain.FULL_TEARDOWN,))

    def test_release_for_changes_uses_planned_domains(self):
        release_for_changes([LifecycleChange.PROMPT_CHANGE], reason="prompt_change")
        self._assert_prompt_conditioning_cleared()
        self.assertGreater(len(_TEXT_ENCODER_COMPONENT_CACHE), 0)
        self.assertIsNotNone(runtime_state._PATCHED_TEXT_ENCODER_COMPONENT_SLOT)
        self.assertIsNotNone(runtime_state._PATCHED_TEXT_ENCODER_COMPONENT_SLOT_KEY)
        self.assertIsNotNone(_STREAMING_RUNTIME_STATE._spine)
        self.assertIsNotNone(_STREAMING_RUNTIME_STATE._key)
        self.assertGreater(len(_PARSED_LORA_CACHE), 0)
        self._assert_spatial_vae_preserved()
        self._assert_structural_cn_preserved()
        self._assert_contextual_cn_preserved()

        self.setUp()
        release_for_changes([LifecycleChange.LORA_STACK_CHANGE], reason="lora_change")
        self._assert_model_prompt_cleared()
        self._assert_spatial_vae_preserved()
        self._assert_structural_cn_preserved()
        self._assert_contextual_cn_preserved()

    def test_legacy_helpers_compatibility(self):
        # 1. release_model_prompt_caches
        self.setUp()
        release_model_prompt_caches(reason="legacy_test")
        self._assert_model_prompt_cleared()
        self._assert_spatial_vae_preserved()
        self._assert_structural_cn_preserved()
        self._assert_contextual_cn_preserved()

        # 2. release_prompt_conditioning_caches
        self.setUp()
        release_prompt_conditioning_caches(reason="legacy_test")
        self._assert_prompt_conditioning_cleared()
        self.assertGreater(len(_TEXT_ENCODER_COMPONENT_CACHE), 0)
        self.assertIsNotNone(runtime_state._PATCHED_TEXT_ENCODER_COMPONENT_SLOT)
        self.assertIsNotNone(runtime_state._PATCHED_TEXT_ENCODER_COMPONENT_SLOT_KEY)
        self.assertIsNotNone(_STREAMING_RUNTIME_STATE._spine)
        self.assertIsNotNone(_STREAMING_RUNTIME_STATE._key)
        self.assertGreater(len(_PARSED_LORA_CACHE), 0)
        self._assert_spatial_vae_preserved()
        self._assert_structural_cn_preserved()
        self._assert_contextual_cn_preserved()

        # 3. release_spatial_vae_caches
        self.setUp()
        release_spatial_vae_caches(reason="legacy_test")
        self._assert_model_prompt_preserved()
        self._assert_spatial_vae_cleared()
        self._assert_structural_cn_preserved()
        self._assert_contextual_cn_preserved()

        # 4. clear_all_caches
        self.setUp()
        clear_all_caches(reason="legacy_test")
        self._assert_model_prompt_cleared()
        self._assert_spatial_vae_cleared()
        self._assert_structural_cn_cleared()
        self._assert_contextual_cn_cleared()

    def test_assembly_close_calls_end_on_cn_workers(self):
        st_worker = MagicMock()
        ctx_worker = MagicMock()
        vae_decode = MagicMock()
        vae_encode = MagicMock()
        spatial = MagicMock()
        lora = MagicMock()
        text = MagicMock()
        unet = MagicMock()

        assembly = SDXLAssembly(
            unet_spine=unet,
            text_encode_worker=text,
            vae_decode_worker=vae_decode,
            lora_worker=lora,
            spatial_context_worker=spatial,
            vae_encode_worker=vae_encode,
            st_preprocess_worker=None,
            st_control_worker=st_worker,
            ctx_control_worker=ctx_worker,
        )

        assembly.close()

        # Assert .end() called instead of .release_owned_resources()
        st_worker.end.assert_called_once()
        ctx_worker.end.assert_called_once()
        st_worker.release_owned_resources.assert_not_called()
        ctx_worker.release_owned_resources.assert_not_called()

        # Assert teardown_assembly_order called on other workers
        vae_decode.teardown_assembly_order.assert_called_once()
        vae_encode.teardown_assembly_order.assert_called_once()
        spatial.teardown_assembly_order.assert_called_once()
        lora.teardown_assembly_order.assert_called_once()
        text.teardown_assembly_order.assert_called_once()
        unet.teardown_assembly_order.assert_called_once()

    def test_assembly_close_hardened_against_worker_failure(self):
        st_worker = MagicMock()
        # Make one step raise an exception
        st_worker.end.side_effect = RuntimeError("Mock failure during close")
        ctx_worker = MagicMock()
        vae_decode = MagicMock()

        assembly = SDXLAssembly(
            unet_spine=MagicMock(),
            text_encode_worker=MagicMock(),
            vae_decode_worker=vae_decode,
            lora_worker=MagicMock(),
            st_control_worker=st_worker,
            ctx_control_worker=ctx_worker,
        )

        # Calling close should not raise runtime error
        try:
            assembly.close()
        except Exception as e:
            self.fail(f"close() raised exception: {e}")

        # Assert subsequent cleanups were still executed
        ctx_worker.end.assert_called_once()
        vae_decode.teardown_assembly_order.assert_called_once()


if __name__ == "__main__":
    unittest.main()
