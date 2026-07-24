import pytest

from backend import loader
from modules import config, core


def test_gguf_is_hidden_from_sdxl_model_selection():
    choices = config.filter_supported_sdxl_base_model_choices(
        ["sdxl/base.safetensors", "sdxl/legacy.gguf", "flux/fill.safetensors"]
    )

    assert choices == ["sdxl/base.safetensors"]
    assert config.is_deprecated_sdxl_base_model_selector("sdxl/legacy.gguf") is True


def test_gguf_is_rejected_at_loader_and_core_boundaries():
    with pytest.raises(ValueError, match="GGUF.*not supported"):
        loader.load_sdxl_unet("legacy.gguf")

    with pytest.raises(ValueError, match="GGUF.*not supported"):
        core.load_model("legacy.gguf")
