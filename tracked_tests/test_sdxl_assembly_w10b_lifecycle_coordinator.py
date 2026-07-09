from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.sdxl_assembly.cpu_lora_worker import _PARSED_LORA_CACHE
from backend.sdxl_assembly.lifecycle_coordinator import (
    LifecycleChange,
    plan_release_for_changes,
)
from backend.sdxl_assembly.runtime_state import (
    LifecycleDomain,
    _PROMPT_CONDITIONING_CACHE,
    _STREAMING_RUNTIME_STATE,
    _TEXT_ENCODER_COMPONENT_CACHE,
    clear_all_caches,
    release_domain,
)
import backend.sdxl_assembly.runtime_state as runtime_state
from backend.sdxl_assembly.stream_ctx_cn_worker import StreamingContextualControlWorker
from backend.sdxl_assembly.stream_st_cn_worker import StreamingStructuralControlWorker
from backend.sdxl_assembly.stream_st_preprocess_worker import StreamingStructuralPreprocessWorker
from backend.sdxl_assembly.vae_encode_worker import VaeEncodeWorker


@pytest.fixture(autouse=True)
def _seed_warm_domains():
    _TEXT_ENCODER_COMPONENT_CACHE.clear()
    _TEXT_ENCODER_COMPONENT_CACHE["clip"] = MagicMock()
    runtime_state._PATCHED_TEXT_ENCODER_COMPONENT_SLOT = MagicMock()
    runtime_state._PATCHED_TEXT_ENCODER_COMPONENT_SLOT_KEY = ("checkpoint", "cpu_pinned", (("lora", 1.0),))
    _PROMPT_CONDITIONING_CACHE.clear()
    _PROMPT_CONDITIONING_CACHE[("prompt",)] = object()
    _STREAMING_RUNTIME_STATE._spine = MagicMock()
    _STREAMING_RUNTIME_STATE._key = "spine_key"
    _PARSED_LORA_CACHE.clear()
    _PARSED_LORA_CACHE[("lora", "unet", "model")] = ("hash", {})

    VaeEncodeWorker._ENCODE_CACHE.clear()
    VaeEncodeWorker._ENCODE_CACHE["vae"] = {}
    StreamingStructuralPreprocessWorker._PREPROCESS_CACHE.clear()
    StreamingStructuralPreprocessWorker._PREPROCESS_CACHE["hint"] = MagicMock()
    StreamingStructuralControlWorker._SUPPORT_MODEL_CACHE.clear()
    StreamingStructuralControlWorker._SUPPORT_MODEL_CACHE["support"] = MagicMock()
    StreamingContextualControlWorker._PAYLOAD_CACHE.clear()
    StreamingContextualControlWorker._PAYLOAD_CACHE["payload"] = MagicMock()
    StreamingContextualControlWorker._CONTEXTUAL_MODELS.clear()
    StreamingContextualControlWorker._CONTEXTUAL_MODELS["context"] = MagicMock()

    yield

    clear_all_caches(reason="tracked_w10b_cleanup")


def _assert_all_domains_cleared() -> None:
    assert _STREAMING_RUNTIME_STATE._spine is None
    assert _STREAMING_RUNTIME_STATE._key is None
    assert len(_TEXT_ENCODER_COMPONENT_CACHE) == 0
    assert runtime_state._PATCHED_TEXT_ENCODER_COMPONENT_SLOT is None
    assert runtime_state._PATCHED_TEXT_ENCODER_COMPONENT_SLOT_KEY is None
    assert len(_PROMPT_CONDITIONING_CACHE) == 0
    assert len(_PARSED_LORA_CACHE) == 0
    assert len(VaeEncodeWorker._ENCODE_CACHE) == 0
    assert len(StreamingStructuralPreprocessWorker._PREPROCESS_CACHE) == 0
    assert len(StreamingStructuralControlWorker._SUPPORT_MODEL_CACHE) == 0
    assert len(StreamingContextualControlWorker._PAYLOAD_CACHE) == 0
    assert len(StreamingContextualControlWorker._CONTEXTUAL_MODELS) == 0


def test_lifecycle_planner_maps_change_events_to_release_domains():
    plan = plan_release_for_changes(
        [
            LifecycleChange.PROMPT_CHANGE,
        ],
        reason="prompt_only_refresh",
    )
    assert plan.domains == (LifecycleDomain.PROMPT_CONDITIONING,)

    plan = plan_release_for_changes(
        [
            LifecycleChange.LORA_STACK_CHANGE,
            LifecycleChange.CHECKPOINT_CHANGE,
            LifecycleChange.SPINE_POSTURE_CHANGE,
        ],
        reason="model_prompt_refresh",
    )
    assert plan.domains == (LifecycleDomain.MODEL_PROMPT,)

    plan = plan_release_for_changes(
        [
            LifecycleChange.SPATIAL_VAE_CHANGE,
            LifecycleChange.STRUCTURAL_CN_CHANGE,
            LifecycleChange.CONTEXTUAL_CN_CHANGE,
        ],
        reason="worker_artifact_refresh",
    )
    assert plan.domains == (
        LifecycleDomain.SPATIAL_VAE,
        LifecycleDomain.STRUCTURAL_CN,
        LifecycleDomain.CONTEXTUAL_CN,
    )

    plan = plan_release_for_changes([LifecycleChange.MODEL_TYPE_CHANGE])
    assert plan.domains == (LifecycleDomain.FULL_TEARDOWN,)


def test_run_bound_release_closes_assembly_without_clearing_warm_domains():
    assembly = MagicMock()

    result = release_domain(LifecycleDomain.RUN_BOUND, reason="request_end", assembly=assembly)

    assert result.ok
    assembly.close.assert_called_once()
    assert len(_TEXT_ENCODER_COMPONENT_CACHE) == 1
    assert runtime_state._PATCHED_TEXT_ENCODER_COMPONENT_SLOT is not None
    assert runtime_state._PATCHED_TEXT_ENCODER_COMPONENT_SLOT_KEY is not None
    assert len(VaeEncodeWorker._ENCODE_CACHE) == 1
    assert len(StreamingStructuralControlWorker._SUPPORT_MODEL_CACHE) == 1
    assert len(StreamingContextualControlWorker._PAYLOAD_CACHE) == 1


def test_full_teardown_continues_when_spine_release_fails():
    _STREAMING_RUNTIME_STATE._spine.release_owned_resources.side_effect = RuntimeError("spine boom")

    result = release_domain(LifecycleDomain.FULL_TEARDOWN, reason="failure_isolation")

    assert not result.ok
    assert len(result.errors) == 1
    assert result.errors[0].domain == LifecycleDomain.MODEL_PROMPT
    assert result.errors[0].step == "streaming_spine"
    _assert_all_domains_cleared()
