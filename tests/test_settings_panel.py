import argparse
import importlib
import os
import sys
from unittest.mock import MagicMock

import gradio as gr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

mock_args = argparse.Namespace()
mock_args.colab = False
mock_args.preset = None
mock_args.output_path = None
mock_args.temp_path = None
mock_args.skip_model_load = False
mock_args.disable_preset_selection = True
mock_args.disable_image_log = False
mock_args.disable_metadata = False

sys.modules['args_manager'] = MagicMock()
sys.modules['args_manager'].args = mock_args
sys.modules.pop('modules.ui_components.settings_panel', None)
sys.modules.pop('modules.ui_components', None)

import modules.config as config

settings_panel = importlib.import_module('modules.ui_components.settings_panel')


def test_build_settings_tab_uses_active_base_model_for_startup_aspect_ratios(monkeypatch):
    monkeypatch.setattr(settings_panel.modules.config, 'default_base_model_name', 'sd15/base/legacy_model.safetensors')
    monkeypatch.setattr(settings_panel.modules.config, 'default_aspect_ratio', '768x512 (3:2)')
    monkeypatch.setattr(
        settings_panel.modules.config,
        'coerce_active_base_model_selection',
        lambda base_model_name: 'sdxl/base/base_model.safetensors',
    )
    monkeypatch.setattr(
        settings_panel.modules.config,
        'get_aspect_ratio_labels_for_model',
        lambda base_model_name: ['1024x1024 (1:1)'] if base_model_name == 'sdxl/base/base_model.safetensors' else ['512x768 (2:3)'],
    )
    monkeypatch.setattr(
        settings_panel.modules.config,
        'get_default_aspect_ratio_label_for_model',
        lambda base_model_name: '1024x1024 (1:1)',
    )

    with gr.Blocks():
        results = settings_panel.build_settings_tab()

    radio = results['aspect_ratios_selection']
    assert radio.choices == [('1024x1024 (1:1)', '1024x1024 (1:1)')]
    assert radio.value == '1024x1024 (1:1)'
    assert 'image_number' not in results
