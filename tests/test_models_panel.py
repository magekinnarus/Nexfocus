import importlib
import os
import sys
import types

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

mock_args = types.SimpleNamespace(
    colab=False,
    preset=None,
    output_path=None,
    temp_path=None,
    skip_model_load=False,
    disable_preset_selection=False,
    disable_image_log=False,
    disable_metadata=False,
)

sys.modules['args_manager'] = types.ModuleType('args_manager')
sys.modules['args_manager'].args = mock_args

import gradio as gr

sys.modules.pop('modules.ui_components.models_panel', None)
sys.modules.pop('modules.ui_components', None)
sys.modules.pop('modules.ui_components.styles_panel', None)

import modules.config as config

models_panel = importlib.import_module('modules.ui_components.models_panel')


def test_build_models_tab_filters_sd15_vae_and_omits_force_clip(monkeypatch):
    monkeypatch.setattr(models_panel.modules.config, 'model_filenames', ['sdxl/unet/test.safetensors'])
    monkeypatch.setattr(models_panel.modules.config, 'default_base_model_name', 'sdxl/unet/test.safetensors')
    monkeypatch.setattr(
        models_panel.modules.config,
        'resolve_dropdown_choice',
        lambda value, choices, **kwargs: value if value in choices else None,
    )
    monkeypatch.setattr(
        models_panel.modules.config,
        'resolve_model_catalog_entry',
        lambda *args, **kwargs: types.SimpleNamespace(root_key='unet'),
    )
    monkeypatch.setattr(
        models_panel.modules.config,
        'get_compatible_vae_choices_for_model',
        lambda base_model_name: ['sdxl/base/vae.safetensors'],
    )
    monkeypatch.setattr(models_panel.styles_panel, 'build_styles_tab', lambda: {}, raising=False)
    monkeypatch.setattr(models_panel.modules.config, 'default_loras', [(True, 'None', 1.0)])
    monkeypatch.setattr(models_panel.modules.config, 'lora_filenames', [])

    with gr.Blocks():
        results = models_panel.build_models_tab()

    vae_model = results['vae_model']

    assert vae_model.choices == [
        (models_panel.flags.default_vae, models_panel.flags.default_vae),
        ('sdxl/base/vae.safetensors', 'sdxl/base/vae.safetensors'),
    ]
    assert 'clip_model' not in results
