import sys
import os
import torch
import logging

# Mock necessary paths
sys.path.append(os.getcwd())

from backend import loader
from ldm_patched.modules import latent_formats

def test_vae_defaults():
    print("Testing VAE defaults...")
    # Mocking model parameters since we don't want to load a real model
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.ones(1))
    
    mock_model = MockModel()
    
    # Case 1: Default (No latent_format provided)
    vae = loader.VAE(mock_model, "cpu", "cpu")
    print(f"Default VAE latent_format: {type(vae.latent_format).__name__}")
    assert isinstance(vae.latent_format, latent_formats.SD15)
    
    # Case 2: SDXL format provided
    vae_xl = loader.VAE(mock_model, "cpu", "cpu", latent_format=latent_formats.SDXL())
    print(f"Explicit SDXL VAE latent_format: {type(vae_xl.latent_format).__name__}")
    assert isinstance(vae_xl.latent_format, latent_formats.SDXL)

def test_vae_inference():
    print("\nTesting VAE architecture inference...")
    
    original_resolve = loader.resolve_source
    original_inspect = getattr(loader, "_inspect_safetensors_vae_metadata", None)
    original_load_prefixed = getattr(loader, "_load_prefixed_safetensors_into_module", None)
    
    loader.resolve_source = lambda x, *args, **kwargs: {} # Return empty dict as mock state dict
    
    def mock_inspect(source_path, *, prefixes=None):
        return {
            "key_count": 10,
            "decoder_conv_in_shape": (128, 4, 3, 3),
            "post_quant_conv_shape": (128, 4),
            "has_downsample": True,
            "has_upsample": True,
        }
    loader._inspect_safetensors_vae_metadata = mock_inspect
    loader._load_prefixed_safetensors_into_module = lambda *args, **kwargs: ([], [])
    
    try:
        # Test SDXL inference from filename
        vae_inferred_xl = loader.load_vae("path/to/sdxl_vae.safetensors", load_device="cpu", offload_device="cpu")
        print(f"Inferred VAE format (sdxl in name): {type(vae_inferred_xl.latent_format).__name__}")
        assert isinstance(vae_inferred_xl.latent_format, latent_formats.SDXL)
        
        # Test SD15 inference from filename
        vae_inferred_15 = loader.load_vae("path/to/v1-5-vae.safetensors", load_device="cpu", offload_device="cpu")
        print(f"Inferred VAE format (v1-5 in name): {type(vae_inferred_15.latent_format).__name__}")
        assert isinstance(vae_inferred_15.latent_format, latent_formats.SD15)
        
    finally:
        loader.resolve_source = original_resolve
        if original_inspect is not None:
            loader._inspect_safetensors_vae_metadata = original_inspect
        if original_load_prefixed is not None:
            loader._load_prefixed_safetensors_into_module = original_load_prefixed


def test_flux_ae_path_inference():
    print("\nTesting Flux AE path inference...")
    original_resolve = loader.resolve_source
    original_inspect = getattr(loader, "_inspect_safetensors_vae_metadata", None)
    original_load_prefixed = getattr(loader, "_load_prefixed_safetensors_into_module", None)
    
    loader.resolve_source = lambda x, *args, **kwargs: {}
    
    def mock_inspect(source_path, *, prefixes=None):
        return {
            "key_count": 10,
            "decoder_conv_in_shape": (128, 16, 3, 3),
            "post_quant_conv_shape": None,
            "has_downsample": False,
            "has_upsample": False,
        }
    loader._inspect_safetensors_vae_metadata = mock_inspect
    loader._load_prefixed_safetensors_into_module = lambda *args, **kwargs: ([], [])

    try:
        vae_flux = loader.load_vae(
            "path/to/flux_fill/ae.safetensors",
            load_device="cpu",
            offload_device="cpu",
        )
        print(f"Inferred VAE format (flux_fill/ae): {type(vae_flux.latent_format).__name__}")
        assert isinstance(vae_flux.latent_format, latent_formats.Flux)
    finally:
        loader.resolve_source = original_resolve
        if original_inspect is not None:
            loader._inspect_safetensors_vae_metadata = original_inspect
        if original_load_prefixed is not None:
            loader._load_prefixed_safetensors_into_module = original_load_prefixed

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        test_vae_defaults()
        test_vae_inference()
        test_flux_ae_path_inference()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
