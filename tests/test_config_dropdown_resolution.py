import argparse
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

mock_args = argparse.Namespace()
mock_args.colab = False
mock_args.preset = None
mock_args.output_path = None
mock_args.temp_path = None
mock_args.skip_model_load = True

sys.modules['args_manager'] = MagicMock()
sys.modules['args_manager'].args = mock_args

import modules.config as config


def test_resolve_dropdown_choice_matches_absolute_path_to_relative_choice():
    choices = ['sdxl/illustrious/beretMixReal_v100_clips.safetensors']
    absolute_path = r'D:\AI\Imagine\models\clip\sdxl\illustrious\beretMixReal_v100_clips.safetensors'

    resolved = config.resolve_dropdown_choice(
        absolute_path,
        choices,
        folder_paths=[r'D:\AI\Imagine\models\clip'],
        root_keys=('clip',),
    )

    assert resolved == 'sdxl/illustrious/beretMixReal_v100_clips.safetensors'


def test_resolve_dropdown_choice_matches_catalog_selector(monkeypatch):
    choices = ['sdxl/illustrious/beretMixReal_v100_Q8.gguf']
    entry = SimpleNamespace(
        relative_path='sdxl/illustrious/beretMixReal_v100_Q8.gguf',
        name='beretMixReal_v100_Q8.gguf',
        alias='beretmixreal-q8',
        id='catalog.beretmixreal.q8',
    )
    monkeypatch.setattr(config, 'resolve_model_catalog_entry', lambda *args, **kwargs: entry)

    resolved = config.resolve_dropdown_choice(
        'catalog.beretmixreal.q8',
        choices,
        folder_paths=[r'D:\AI\Imagine\models\unet'],
        root_keys=('checkpoints', 'unet'),
    )

    assert resolved == 'sdxl/illustrious/beretMixReal_v100_Q8.gguf'


def test_filter_supported_sdxl_base_model_choices_excludes_legacy_gguf_and_flux_entries():
    filtered = config.filter_supported_sdxl_base_model_choices(
        [
            'sdxl/base/model_a.safetensors',
            'sdxl/illustrious/beretMixReal_v100_Q8.gguf',
            'flux/flux-fill-q8.gguf',
        ]
    )

    assert filtered == ['sdxl/base/model_a.safetensors']
