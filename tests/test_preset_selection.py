import sys
import os
import json
import argparse
from unittest.mock import MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup clean argparse Namespace before importing config
mock_args = argparse.Namespace()
mock_args.colab = False
mock_args.preset = None
mock_args.output_path = None
mock_args.temp_path = None
mock_args.skip_model_load = False
mock_args.disable_metadata = False
mock_args.disable_preset_selection = False

sys.modules['args_manager'] = MagicMock()
sys.modules['args_manager'].args = mock_args

# pop cached modules to ensure clean import with our mocked args
sys.modules.pop('modules.config', None)
sys.modules.pop('modules.ui_logic', None)

import modules.config as config
import modules.ui_components.metadata_ui as metadata_ui
import modules.ui_logic as ui_logic


def test_preset_selection_change_shifts_custom_lora_without_extra_callback_args(monkeypatch):
    monkeypatch.setattr(ui_logic, 'download_preset_models', lambda default_model, checkpoint_downloads, *args: (default_model, checkpoint_downloads, False))
    
    captured_payloads = []

    def mock_load_click(raw_metadata, is_generating):
        captured_payloads.append(json.loads(raw_metadata))
        return []

    monkeypatch.setattr(metadata_ui, 'load_parameter_button_click', mock_load_click)

    preset_data = {
        "default_overwrite_step": 8,
        "default_sampler": "euler",
        "default_scheduler": "sgm_uniform",
        "default_loras": [[True, "preset_lora.safetensors", 1.0]]
    }
    monkeypatch.setattr(config, 'try_get_preset_content', lambda preset: preset_data)

    args = [
        True, "my_custom_style_lora.safetensors", 0.8,
        True, "my_existing_style_lora.safetensors", 0.5,
        True, "None", 1.0,
        True, "None", 1.0,
        True, "None", 1.0,
    ]

    ui_logic.preset_selection_change("custom", False, *args)

    assert len(captured_payloads) == 1
    result_dict = captured_payloads[0]

    assert "base_model" not in result_dict
    assert result_dict["steps"] == 8
    assert result_dict["sampler"] == "euler"
    assert result_dict["scheduler"] == "sgm_uniform"
    assert result_dict["lora_combined_1"] == "True : preset_lora.safetensors : 1.0"
    assert result_dict["lora_combined_2"] == "True : my_existing_style_lora.safetensors : 0.5"
    assert result_dict["lora_combined_3"] == "True : my_custom_style_lora.safetensors : 0.8"


def test_preset_selection_change_preserves_custom_slots(monkeypatch):
    monkeypatch.setattr(ui_logic, 'download_preset_models', lambda default_model, checkpoint_downloads, *args: (default_model, checkpoint_downloads, False))
    
    captured_payloads = []

    def mock_load_click(raw_metadata, is_generating):
        captured_payloads.append(json.loads(raw_metadata))
        return []

    monkeypatch.setattr(metadata_ui, 'load_parameter_button_click', mock_load_click)

    preset_data = {
        "default_overwrite_step": 4,
        "default_sampler": "euler",
        "default_scheduler": "sgm_uniform",
        "default_loras": [[True, "preset_lora.safetensors", 1.0]]
    }
    monkeypatch.setattr(config, 'try_get_preset_content', lambda preset: preset_data)

    args = [
        True, "old_preset_lora.safetensors", 1.0,
        True, "my_style.safetensors", 0.5,
        True, "another_lora.safetensors", 1.0,
        True, "None", 1.0,
        True, "None", 1.0,
    ]

    ui_logic.preset_selection_change("custom", False, *args)

    assert len(captured_payloads) == 1
    result_dict = captured_payloads[0]

    assert result_dict["lora_combined_1"] == "True : preset_lora.safetensors : 1.0"
    assert result_dict["lora_combined_2"] == "True : my_style.safetensors : 0.5"
    assert result_dict["lora_combined_3"] == "True : another_lora.safetensors : 1.0"


def test_preset_selection_change_refreshes_model_indexes_after_new_download(monkeypatch):
    monkeypatch.setattr(
        ui_logic,
        'download_preset_models',
        lambda default_model, checkpoint_downloads, *args: (default_model, checkpoint_downloads, True),
    )

    captured_payloads = []

    def mock_load_click(raw_metadata, is_generating):
        captured_payloads.append(json.loads(raw_metadata))
        return []

    monkeypatch.setattr(metadata_ui, 'load_parameter_button_click', mock_load_click)

    preset_data = {
        "default_overwrite_step": 8,
        "default_sampler": "euler",
        "default_scheduler": "sgm_uniform",
        "default_loras": [[True, "preset_lora.safetensors", 1.0]],
    }
    monkeypatch.setattr(config, 'try_get_preset_content', lambda preset: preset_data)

    refresh_calls = []
    monkeypatch.setattr(ui_logic.modules.config, 'update_files', lambda: refresh_calls.append('update_files'))

    class RefreshTrackingManager:
        def refresh_catalog_index(self, *, force_refresh=False):
            refresh_calls.append(('catalog', force_refresh))

        def refresh_installed_index(self):
            refresh_calls.append('installed')

    monkeypatch.setattr(ui_logic, 'default_model_manager', RefreshTrackingManager())

    args = [
        True, "None", 1.0,
        True, "None", 1.0,
        True, "None", 1.0,
        True, "None", 1.0,
        True, "None", 1.0,
    ]

    ui_logic.preset_selection_change("custom", False, *args)

    assert len(captured_payloads) == 1
    assert refresh_calls == [('catalog', True), 'installed', 'update_files']
