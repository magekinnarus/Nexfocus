import os
import tempfile
import pytest
from modules.util import get_file_from_folder_list
from modules.config import get_preferred_asset_root_path, _persistent_asset_filenames, asset_root_path_groups


def test_get_file_from_folder_list_slash_normalization():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a nested file
        nested_dir = os.path.join(tmp_dir, "sdxl")
        os.makedirs(nested_dir, exist_ok=True)
        test_file = os.path.join(nested_dir, "test_model.safetensors")
        with open(test_file, "w") as f:
            f.write("dummy content")

        # Test forward slash resolution
        resolved_fw = get_file_from_folder_list("sdxl/test_model.safetensors", tmp_dir)
        assert os.path.isfile(resolved_fw)
        assert os.path.basename(resolved_fw) == "test_model.safetensors"

        # Test backward slash resolution
        resolved_bw = get_file_from_folder_list("sdxl\\test_model.safetensors", tmp_dir)
        assert os.path.isfile(resolved_bw)
        assert os.path.basename(resolved_bw) == "test_model.safetensors"


def test_get_file_from_folder_list_sdxl_vae_fallback():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create vae in sdxl/ subdirectory
        nested_dir = os.path.join(tmp_dir, "sdxl")
        os.makedirs(nested_dir, exist_ok=True)
        vae_file = os.path.join(nested_dir, "sdxl_vae.safetensors")
        with open(vae_file, "w") as f:
            f.write("dummy vae content")

        # Resolving "sdxl_vae.safetensors" should fall back to sdxl/sdxl_vae.safetensors
        resolved = get_file_from_folder_list("sdxl_vae.safetensors", tmp_dir)
        assert os.path.isfile(resolved)
        assert os.path.basename(resolved) == "sdxl_vae.safetensors"
        assert "sdxl" in os.path.dirname(resolved)


def test_get_preferred_asset_root_path_basename(monkeypatch):
    # Temporarily set asset_root_path_groups for vae to have two dummy paths
    dummy_roots = ["/mock/root1", "/mock/root2"]
    monkeypatch.setitem(asset_root_path_groups, 'vae', dummy_roots)

    # VAE list should contain 'sdxl_vae.safetensors'
    assert 'sdxl_vae.safetensors' in _persistent_asset_filenames['vae']

    # With a simple file name
    root_simple = get_preferred_asset_root_path('vae', file_name='sdxl_vae.safetensors')
    assert root_simple == "/mock/root1"

    # With nested path structure in file_name (it should extract basename and still match persistent names)
    root_nested = get_preferred_asset_root_path('vae', file_name='sdxl/sdxl_vae.safetensors')
    assert root_nested == "/mock/root1"

    # With non-persistent VAE name, it should fall back to root2 (roots[1])
    root_fallback = get_preferred_asset_root_path('vae', file_name='some_custom_vae.safetensors')
    assert root_fallback == "/mock/root2"
