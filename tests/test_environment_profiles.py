import os
import sys
from types import SimpleNamespace

sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.environment_profile import (
    FLUX_ACCELERATION_CLASS_TENSOR_CORE,
    PROFILE_COLAB_FREE,
    PROFILE_COLAB_PRO,
    PROFILE_CUSTOM,
    PROFILE_LOCAL_LOW_VRAM,
    PROFILE_LOCAL_NORMAL,
    resolve_environment_profile,
    should_skip_eager_model_preload,
)


def test_auto_detect_colab_profiles():
    profile = resolve_environment_profile(total_ram_mb=53248, total_vram_mb=24576, is_colab=True)
    assert profile.name == PROFILE_COLAB_PRO
    assert profile.source == 'auto'

    profile = resolve_environment_profile(total_ram_mb=13000, total_vram_mb=15360, is_colab=True)
    assert profile.name == PROFILE_COLAB_FREE

    profile = resolve_environment_profile(total_ram_mb=57344, total_vram_mb=12288, is_colab=True)
    assert profile.name == PROFILE_COLAB_PRO


def test_auto_detect_local_profiles():
    profile = resolve_environment_profile(total_ram_mb=32768, total_vram_mb=4096, is_colab=False)
    assert profile.name == PROFILE_LOCAL_LOW_VRAM

    profile = resolve_environment_profile(total_ram_mb=32768, total_vram_mb=12288, is_colab=False)
    assert profile.name == PROFILE_LOCAL_NORMAL


def test_custom_override_uses_custom_policy_values():
    profile = resolve_environment_profile(
        override=PROFILE_CUSTOM,
        custom_name='Director Custom',
        total_ram_mb=65536,
        total_vram_mb=8192,
        is_colab=False,
        custom_policy_overrides={
            'low_ram_headroom_mb': 9000.0,
            'checkpoint_switch_ram_headroom_mb': 12000.0,
        },
    )

    assert profile.name == PROFILE_CUSTOM
    assert profile.display_name == 'Director Custom'
    assert profile.policy_overrides['low_ram_headroom_mb'] == 9000.0
    assert profile.policy_overrides['checkpoint_switch_ram_headroom_mb'] == 12000.0


def test_skip_eager_model_preload_only_for_colab_free():
    colab_free = resolve_environment_profile(total_ram_mb=13000, total_vram_mb=15360, is_colab=True)
    colab_pro = resolve_environment_profile(total_ram_mb=53248, total_vram_mb=24576, is_colab=True)
    local_normal = resolve_environment_profile(total_ram_mb=32768, total_vram_mb=12288, is_colab=False)

    assert should_skip_eager_model_preload(colab_free) is True
    assert should_skip_eager_model_preload(colab_pro) is False
    assert should_skip_eager_model_preload(local_normal) is False


def test_debug_tab_exposes_streaming_only_flux_fill_runtime_override():
    import gradio as gr
    from modules.ui_components import advanced_panel

    with gr.Blocks():
        controls = advanced_panel.build_debug_tab()

    assert "flux_fill_t5_low_ram" not in controls
    assert "flux_fill_runtime_posture" in controls
    assert controls["flux_fill_runtime_posture"].value == "auto"
    assert "flux_fill_disk_paged_t5_gc_interval" in controls
    gc_cadence = controls["flux_fill_disk_paged_t5_gc_interval"]
    assert gc_cadence.label == "T5 Host-RAM Cleanup Cadence"
    assert gc_cadence.choices == [("auto", "auto"), ("8", "8"), ("16", "16")]
    assert gc_cadence.value == "auto"
    assert "garbage collection" in gc_cadence.info


def test_debug_tab_defaults_sdxl_gpu_text_only_on_colab_free(monkeypatch):
    import gradio as gr
    from modules.ui_components import advanced_panel

    monkeypatch.setattr(
        advanced_panel.modules.config,
        "resolved_memory_environment_profile",
        SimpleNamespace(name=PROFILE_COLAB_FREE),
    )
    with gr.Blocks():
        controls = advanced_panel.build_debug_tab()
    assert controls["sdxl_assembly_posture"].value == "gpu_text"

    monkeypatch.setattr(
        advanced_panel.modules.config,
        "resolved_memory_environment_profile",
        SimpleNamespace(name=PROFILE_COLAB_PRO),
    )
    with gr.Blocks():
        controls = advanced_panel.build_debug_tab()
    assert controls["sdxl_assembly_posture"].value == "auto"


def test_debug_tab_exposes_t5_posture_exposure_gate():
    import gradio as gr
    from modules.ui_components import advanced_panel
    from unittest.mock import patch

    # Below threshold (e.g. 16 GB)
    with patch("backend.flux_fill_v3.activation.resolve_flux_fill_total_ram_gb", return_value=16.0):
        with gr.Blocks():
            controls = advanced_panel.build_debug_tab()
        assert "flux_fill_t5_posture" in controls
        assert controls["flux_fill_t5_posture"].visible is False

    # Above threshold (e.g. 32 GB)
    with patch("backend.flux_fill_v3.activation.resolve_flux_fill_total_ram_gb", return_value=32.0):
        with gr.Blocks():
            controls = advanced_panel.build_debug_tab()
        assert "flux_fill_t5_posture" in controls
        assert controls["flux_fill_t5_posture"].visible is True


def test_resolve_environment_profile_includes_flux_acceleration_notes(monkeypatch):
    monkeypatch.setattr(
        'backend.environment_profile.detect_primary_gpu_notes',
        lambda: {
            'gpu_name': 'Tesla T4',
            'cuda_capability': '7.5',
            'flux_acceleration_class': FLUX_ACCELERATION_CLASS_TENSOR_CORE,
            'tensor_core_accelerated': True,
        },
    )

    profile = resolve_environment_profile(total_ram_mb=32768, total_vram_mb=15360, is_colab=False)

    assert profile.notes['gpu_name'] == 'Tesla T4'
    assert profile.notes['cuda_capability'] == '7.5'
    assert profile.notes['flux_acceleration_class'] == FLUX_ACCELERATION_CLASS_TENSOR_CORE
    assert profile.notes['tensor_core_accelerated'] is True
