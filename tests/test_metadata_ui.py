import argparse
import importlib
import os
import sys
from unittest.mock import MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

mock_args = argparse.Namespace()
mock_args.colab = False
mock_args.preset = None
mock_args.output_path = None
mock_args.temp_path = None

sys.modules['args_manager'] = MagicMock()
sys.modules['args_manager'].args = mock_args
sys.modules.pop('modules.ui_components.metadata_ui', None)
sys.modules.pop('modules.ui_components', None)

import modules.config as config

metadata_ui = importlib.import_module('modules.ui_components.metadata_ui')


def test_load_parameter_button_click_accepts_json_string():
    results = metadata_ui.load_parameter_button_click('{}', False)

    assert isinstance(results, list)
    assert results[metadata_ui.METADATA_OUTPUT_INDEX['prompt']] == {'__type__': 'update'}
    assert len(results) == metadata_ui.METADATA_OUTPUT_INDEX['loras_start'] + (
        metadata_ui.modules.config.default_max_lora_number * 3
    )


def test_load_parameter_button_click_uses_base_model_specific_resolution_labels(monkeypatch):
    monkeypatch.setattr(
        metadata_ui.modules.config,
        'coerce_active_base_model_selection',
        lambda base_model_name: 'sdxl/base/base_model.safetensors',
    )
    monkeypatch.setattr(
        metadata_ui.modules.config,
        'get_aspect_ratio_labels_for_model',
        lambda base_model_name: ['1024x1024 (1:1)'] if base_model_name == 'sdxl/base/base_model.safetensors' else ['512x768 (2:3)'],
    )

    results = metadata_ui.load_parameter_button_click(
        {
            'base_model': 'sd15_model.safetensors',
            'resolution': '(1024, 1024)',
        },
        False,
    )

    assert results[metadata_ui.METADATA_OUTPUT_INDEX['resolution']] == '1024x1024 (1:1)'
    assert results[metadata_ui.METADATA_OUTPUT_INDEX['base_model']] == 'sdxl/base/base_model.safetensors'


def test_load_parameter_button_click_does_not_emit_oversized_overwrite_resolution(monkeypatch):
    monkeypatch.setattr(metadata_ui.modules.config, 'get_aspect_ratio_labels_for_model', lambda base_model_name: ['832x1216 (13:19)'])

    results = metadata_ui.load_parameter_button_click(
        {
            'base_model': 'sdxl_model.safetensors',
            'resolution': '(1664, 2432)',
        },
        False,
    )

    assert results[metadata_ui.METADATA_OUTPUT_INDEX['resolution']] == {'__type__': 'update'}


def test_load_parameter_button_click_preserves_base_model_when_missing():
    results = metadata_ui.load_parameter_button_click(
        {
            'sampler': 'euler',
        },
        False,
    )

    assert results[metadata_ui.METADATA_OUTPUT_INDEX['base_model']] == {'__type__': 'update'}
    assert results[metadata_ui.METADATA_OUTPUT_INDEX['sampler']] == 'euler'


def test_load_parameter_button_click_routes_tab_local_prompts():
    inpaint_results = metadata_ui.load_parameter_button_click(
        {
            'metadata_version': 2,
            'workflow': 'inpaint_sdxl',
            'prompt': 'main scene',
            'inpaint_prompt': 'repair the hair',
            'inpaint_route': 'sdxl',
        },
        False,
    )
    assert inpaint_results[metadata_ui.METADATA_OUTPUT_INDEX['prompt']] == 'main scene'
    assert inpaint_results[metadata_ui.METADATA_OUTPUT_INDEX['inpaint_prompt']] == 'repair the hair'
    assert inpaint_results[metadata_ui.METADATA_OUTPUT_INDEX['outpaint_prompt']] == {'__type__': 'update'}

    outpaint_results = metadata_ui.load_parameter_button_click(
        {
            'metadata_version': 2,
            'workflow': 'outpaint_sdxl',
            'prompt': 'main scene',
            'outpaint_prompt': 'extend the mountains',
        },
        False,
    )
    assert outpaint_results[metadata_ui.METADATA_OUTPUT_INDEX['outpaint_prompt']] == 'extend the mountains'

    remove_results = metadata_ui.load_parameter_button_click(
        {
            'metadata_version': 2,
            'workflow': 'flux_fill_remove',
            'prompt_description': 'replace with beach',
        },
        False,
    )
    assert remove_results[metadata_ui.METADATA_OUTPUT_INDEX['remove_prompt']] == 'replace with beach'

    color_results = metadata_ui.load_parameter_button_click(
        {
            'metadata_version': 2,
            'workflow': 'color_enhance',
            'prompt_description': 'warm cinematic colors',
        },
        False,
    )
    assert color_results[metadata_ui.METADATA_OUTPUT_INDEX['color_enhance_prompt']] == 'warm cinematic colors'


def test_non_generative_metadata_applies_no_parameters():
    results = metadata_ui.load_parameter_button_click(
        {'metadata_version': 2, 'workflow': 'bgr_subject', 'resolution': '512x512'},
        False,
    )
    for index in range(25):
        assert results[index] == {'__type__': 'update'}


def test_metadata_list_and_resolution_parsing_never_executes_expressions(tmp_path):
    marker = tmp_path / 'executed.txt'
    expression = f"__import__('pathlib').Path({str(marker)!r}).write_text('bad')"

    list_results = []
    metadata_ui.get_list('styles', None, {'styles': expression}, list_results)
    resolution_results = []
    metadata_ui.get_resolution('resolution', None, {'resolution': expression}, resolution_results)

    assert not marker.exists()
    assert list_results == [{'__type__': 'update'}]
    assert resolution_results == [{'__type__': 'update'}]


def test_get_inpaint_route_does_not_import_objr_engine():
    original_objr_engine = sys.modules.pop('modules.objr_engine', None)
    try:
        results = []
        value = metadata_ui.get_inpaint_route(
            'inpaint_route',
            'Inpaint Route',
            {'inpaint_route': 'flux_fill'},
            results,
        )

        assert value == 'flux'
        assert results == ['flux']
        assert 'modules.objr_engine' not in sys.modules
    finally:
        if original_objr_engine is not None:
            sys.modules['modules.objr_engine'] = original_objr_engine


def test_parse_meta_from_preset_initial_loads_all_defaults():
    results = metadata_ui.parse_meta_from_preset({})
    assert isinstance(results, dict)
    # Reverting to initial preset should populate all possible keys from config
    assert "steps" in results
    assert "sampler" in results
    assert "scheduler" in results
    assert "lora_combined_1" in results
    assert "lora_combined_5" in results


def test_parse_meta_from_preset_custom_omits_unspecified_keys():
    preset_data = {
        "default_overwrite_step": 8,
        "default_sampler": "euler",
        "default_scheduler": "sgm_uniform",
        "default_loras": [[True, "sdxl_lora.safetensors", 1.0]]
    }
    results = metadata_ui.parse_meta_from_preset(preset_data)
    assert isinstance(results, dict)
    
    # Specified keys should be present
    assert results["steps"] == 8
    assert results["sampler"] == "euler"
    assert results["scheduler"] == "sgm_uniform"
    assert results["lora_combined_1"] == "True : sdxl_lora.safetensors : 1.0"
    
    # Omitted keys must NOT be present
    assert "base_model" not in results
    assert "prompt" not in results
    assert "resolution" not in results
    assert "lora_combined_2" not in results
    assert "lora_combined_5" not in results


def test_parse_meta_from_preset_handles_aspect_ratio_formats():
    preset_data_star = {"default_aspect_ratio": "1024*1024"}
    results_star = metadata_ui.parse_meta_from_preset(preset_data_star)
    assert results_star["resolution"] == "('1024', '1024')"

    preset_data_cross = {"default_aspect_ratio": "1152×896 (9:7)"}
    results_cross = metadata_ui.parse_meta_from_preset(preset_data_cross)
    assert results_cross["resolution"] == "('1152', '896')"
