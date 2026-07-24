from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch

import ldm_patched.modules.ops as base_ops
from backend.flux_fill_v3.t5_worker import T5Stack, _resolve_disk_paged_t5_gc_config


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


def test_disk_paged_t5_gc_config_defaults_to_periodic_interval_with_headroom():
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

    assert config["interval"] == 4
    assert config["recheck_blocks"] == 2
    assert config["profile_name"] == "colab_free"
    assert config["initial_free_ram_mb"] == 4096.0


def test_disk_paged_t5_runtime_uses_periodic_gc_by_default():
    stack = _build_small_t5_stack()
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

    assert len(gc_calls) == 1
    assert mock_snapshot.call_count == 1


def test_disk_paged_t5_runtime_falls_back_to_every_block_when_critical():
    stack = _build_small_t5_stack()
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

    assert len(gc_calls) == 3
    assert mock_snapshot.call_count == 1
