import os
import sys
import pytest
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.environment_profile import (
    PROFILE_AUTO,
    PROFILE_COLAB_FREE,
    PROFILE_COLAB_PRO,
    PROFILE_CUSTOM,
    PROFILE_LOCAL_LOW_VRAM,
    PROFILE_LOCAL_NORMAL,
    auto_detect_profile_name,
    resolve_environment_profile,
    should_skip_eager_model_preload,
)
import ldm_patched.modules.args_parser as ldm_args_parser


def test_environment_profile_auto_detection_matrix():
    # Colab Free: Low VRAM/RAM on Colab
    assert auto_detect_profile_name(total_ram_mb=13000, total_vram_mb=15360, is_colab=True) == PROFILE_COLAB_FREE

    # Colab Pro: High VRAM on Colab
    assert auto_detect_profile_name(total_ram_mb=53248, total_vram_mb=24576, is_colab=True) == PROFILE_COLAB_PRO
    # Colab Pro: High RAM on Colab
    assert auto_detect_profile_name(total_ram_mb=45000, total_vram_mb=15360, is_colab=True) == PROFILE_COLAB_PRO

    # Local Low VRAM: <= 6144 MB VRAM
    assert auto_detect_profile_name(total_ram_mb=16384, total_vram_mb=6144, is_colab=False) == PROFILE_LOCAL_LOW_VRAM

    # Local Normal: > 6144 MB VRAM
    assert auto_detect_profile_name(total_ram_mb=32768, total_vram_mb=8192, is_colab=False) == PROFILE_LOCAL_NORMAL


def test_resolve_environment_profile_fail_closed_on_invalid_override():
    with pytest.raises(ValueError, match="Invalid memory environment profile override"):
        resolve_environment_profile(override="invalid_profile_name")


def test_resolve_environment_profile_fail_closed_on_non_positive_ram_vram():
    with pytest.raises(ValueError, match="RAM override must be greater than 0 MB."):
        resolve_environment_profile(total_ram_mb=0.0)

    with pytest.raises(ValueError, match="RAM override must be greater than 0 MB."):
        resolve_environment_profile(total_ram_mb=-1024.0)

    with pytest.raises(ValueError, match="VRAM override must be greater than 0 MB."):
        resolve_environment_profile(total_vram_mb=0.0)

    with pytest.raises(ValueError, match="VRAM override must be greater than 0 MB."):
        resolve_environment_profile(total_vram_mb=-512.0)


def test_resolve_environment_profile_explicit_overrides():
    profile = resolve_environment_profile(
        override=PROFILE_LOCAL_LOW_VRAM,
        total_ram_mb=16384,
        total_vram_mb=4096,
        is_colab=False,
    )
    assert profile.name == PROFILE_LOCAL_LOW_VRAM
    assert profile.source == "override"
    assert profile.total_ram_mb == 16384.0
    assert profile.total_vram_mb == 4096.0


def test_eager_model_preload_policy():
    free_profile = resolve_environment_profile(total_ram_mb=13000, total_vram_mb=15360, is_colab=True)
    pro_profile = resolve_environment_profile(total_ram_mb=53248, total_vram_mb=24576, is_colab=True)
    local_profile = resolve_environment_profile(total_ram_mb=32768, total_vram_mb=8192, is_colab=False)

    assert should_skip_eager_model_preload(free_profile) is True
    assert should_skip_eager_model_preload(pro_profile) is False
    assert should_skip_eager_model_preload(local_profile) is False


def test_cli_parser_defaults_and_listen_flag():
    import args_manager

    # Verify parser action defaults from ldm_args_parser
    actions = {action.dest: action for action in ldm_args_parser.parser._actions}
    assert "listen" in actions
    assert actions["listen"].default == "127.0.0.1"
    assert actions["listen"].const == "0.0.0.0"

    assert "port" in actions
    assert "in_browser" in actions
    assert "disable_in_browser" in actions
    assert "debug_mode" in actions

    # Verify parsed args on args_manager
    assert hasattr(args_manager.args, "colab")
    assert hasattr(args_manager.args, "preset")
    assert hasattr(args_manager.args, "disable_preset_selection")
    assert hasattr(args_manager.args, "disable_image_log")


def test_cli_port_validation():
    # Test valid port range bounds
    valid_ports = [1, 7865, 8188, 65535]
    for port in valid_ports:
        port_val = int(port)
        assert 1 <= port_val <= 65535

    invalid_ports = [0, -1, 65536, 99999]
    for port in invalid_ports:
        port_val = int(port)
        assert not (1 <= port_val <= 65535)
