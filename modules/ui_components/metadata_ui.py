import ast
import json
import gradio as gr
from pathlib import Path
import re
import modules.config
import modules.flags
from modules.flux_fill_surface import normalize_flux_fill_inpaint_route
from modules.flags import SAMPLERS
from modules.util import unquote, get_file_from_folder_list
import modules.meta_parser

OVERWRITE_DIMENSION_MAX = 2048

METADATA_OUTPUT_INDEX = {
    'prompt': 0,
    'negative_prompt': 1,
    'styles': 2,
    'steps': 3,
    'resolution': 4,
    'cfg_scale': 5,
    'base_model': 12,
    'sampler': 14,
    'scheduler': 15,
    'seed_random': 16,
    'seed': 17,
    'outpaint_engine': 18,
    'inpaint_engine': 19,
    'inpaint_route': 20,
    'inpaint_prompt': 21,
    'outpaint_prompt': 22,
    'remove_prompt': 23,
    'color_enhance_prompt': 24,
    'generate_button': 25,
    'load_parameter_button': 26,
    'loras_start': 27,
}


def get_str(key: str, fallback: str | None, source_dict: dict, results: list, default=None) -> str | None:
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert isinstance(h, str)
        results.append(h)
        return h
    except Exception:
        results.append(gr.update())
        return None

def get_list(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        if isinstance(h, str):
            parsed = None
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(h)
                except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                    continue
                break
            h = parsed
        assert isinstance(h, list)
        if key == 'styles':
            h = [s for s in h if s != 'Fooocus V2']
        results.append(h)
    except Exception:
        results.append(gr.update())

def get_number(key: str, fallback: str | None, source_dict: dict, results: list, default=None, cast_type=float):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert h is not None
        h = cast_type(h)
        results.append(h)
    except Exception:
        results.append(gr.update())

def get_image_number(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert h is not None
        int(h)
        results.append(1)
    except Exception:
        results.append(1)

def get_steps(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert h is not None
        h = int(h)
        results.append(h)
    except Exception:
        results.append(gr.update())

def get_resolution(key: str, fallback: str | None, source_dict: dict, results: list, default=None, valid_labels=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        if isinstance(h, (tuple, list)):
            width, height = h[0], h[1]
        elif isinstance(h, str):
            match = re.search(r'(\d+)\s*(?:\*|x|X|\u00d7|횞)\s*(\d+)', h)
            if match:
                width, height = match.groups()
            else:
                parsed = ast.literal_eval(h)
                assert isinstance(parsed, (tuple, list)) and len(parsed) >= 2
                width, height = parsed[0], parsed[1]
        else:
            raise ValueError('Unsupported resolution metadata')
        formatted = modules.config.add_ratio(f'{width}*{height}')
        if valid_labels is None:
            valid_labels = modules.config.available_aspect_ratios_labels
        if formatted in valid_labels:
            results.append(formatted)
        else:
            results.append(gr.update())
    except Exception:
        results.append(gr.update())

def get_seed(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert h is not None
        h = int(h)
        results.append(False)
        results.append(h)
    except Exception:
        results.append(gr.update())
        results.append(gr.update())

def get_inpaint_engine_version(key: str, fallback: str | None, source_dict: dict, results: list, default=None) -> str | None:
    try:
        h = source_dict.get(key, source_dict.get(fallback, None))
        if h is None:
            results.append(gr.update())
            return None
        h = modules.flags.normalize_inpaint_engine_version(h, default=modules.flags.INPAINT_ENGINE_NONE)
        assert isinstance(h, str) and h in modules.flags.inpaint_engine_versions
        results.append(h)
        return h
    except Exception:
        results.append(modules.flags.INPAINT_ENGINE_NONE)
        return None

def get_outpaint_engine_version(key: str, fallback: str | None, source_dict: dict, results: list, default=None) -> str | None:
    try:
        h = source_dict.get(key, source_dict.get(fallback, None))
        if h is None:
            results.append(gr.update())
            return None
        h = modules.flags.normalize_inpaint_engine_version(h, default=modules.flags.INPAINT_ENGINE_NONE)
        assert isinstance(h, str) and h in modules.flags.inpaint_engine_versions
        results.append(h)
        return h
    except Exception:
        results.append(modules.flags.INPAINT_ENGINE_NONE)
        return None

def get_inpaint_route(key: str, fallback: str | None, source_dict: dict, results: list, default=None) -> str | None:
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        if h == 'empty':
            h = modules.config.default_inpaint_route
        normalized = normalize_flux_fill_inpaint_route(h)
        results.append(normalized)
        return normalized
    except Exception:
        results.append('sdxl')
        return None

def get_lora(key: str, fallback: str | None, source_dict: dict, results: list):
    try:
        val = source_dict.get(key, source_dict.get(fallback))
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            results.append(True)
            results.append(val[0])
            results.append(float(val[1]))
            return
        split_data = str(val).split(' : ')
        enabled = True
        name = split_data[0]
        weight = split_data[1]
        if len(split_data) == 3:
            enabled = split_data[0] == 'True'
            name = split_data[1]
            weight = split_data[2]
        weight = float(weight)
        results.append(enabled)
        results.append(name)
        results.append(weight)
    except Exception:
        results.append(True)
        results.append('None')
        results.append(1)


def _load_parameter_button_click(raw_metadata: dict | str, is_generating: bool, *, convert_legacy: bool):
    loaded_parameter_dict = raw_metadata
    if isinstance(raw_metadata, str):
        try:
            loaded_parameter_dict = json.loads(raw_metadata)
        except Exception:
            loaded_parameter_dict = {}
    if not isinstance(loaded_parameter_dict, dict):
        loaded_parameter_dict = {}

    if convert_legacy and (
        'metadata_version' not in loaded_parameter_dict
        or loaded_parameter_dict.get('metadata_version') == 1
    ):
        loaded_parameter_dict = modules.meta_parser.convert_v1_to_v2_metadata(loaded_parameter_dict)

    workflow = str(loaded_parameter_dict.get('workflow', 'txt2img'))
    results = []

    # Non-generative / identity-only workflows apply NO parameters on import
    if workflow in ['upscale_gan', 'remove_mat', 'remove_objr', 'bgr_subject', 'bgr_mask']:
        results.extend([gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()])
        results.extend([gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()])
        results.extend([gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()])
        results.extend([gr.update(), gr.update(), gr.update()])
        results.extend([gr.update(), gr.update(), gr.update(), gr.update()])
        if is_generating:
            results.append(gr.update())
        else:
            results.append(gr.update(visible=True))
        results.append(gr.update(visible=False))
        for _ in range(modules.config.default_max_lora_number):
            results.extend([gr.update(), gr.update(), gr.update()])
        return results

    base_model_name = loaded_parameter_dict.get('base_model')
    active_base_model_name = None
    if isinstance(base_model_name, str) and workflow in ['txt2img', 'inpaint_sdxl', 'outpaint_sdxl', 'super_upscale', 'color_enhance']:
        active_base_model_name = modules.config.coerce_active_base_model_selection(base_model_name)
        loaded_parameter_dict['base_model'] = active_base_model_name

    resolution_labels = modules.config.get_aspect_ratio_labels_for_model(
        active_base_model_name or modules.config.default_base_model_name
    )

    # 1. Prompt
    if workflow in ['txt2img', 'inpaint_sdxl', 'outpaint_sdxl', 'super_upscale', 'flux_fill_inpaint']:
        get_str('prompt', 'Prompt', loaded_parameter_dict, results)
    else:
        results.append(gr.update())

    # 2. Negative Prompt
    if workflow in ['txt2img', 'inpaint_sdxl', 'outpaint_sdxl', 'super_upscale']:
        get_str('negative_prompt', 'Negative Prompt', loaded_parameter_dict, results)
    else:
        results.append(gr.update())

    # 3. Styles
    if workflow in ['txt2img', 'inpaint_sdxl', 'outpaint_sdxl', 'super_upscale']:
        get_list('styles', 'Styles', loaded_parameter_dict, results)
    else:
        results.append(gr.update())

    # 4. Steps
    if workflow in ['txt2img', 'inpaint_sdxl', 'outpaint_sdxl', 'super_upscale', 'flux_fill_inpaint', 'flux_fill_remove', 'color_enhance']:
        get_steps('steps', 'Steps', loaded_parameter_dict, results)
    else:
        results.append(-1)

    # 5. Resolution (deployable ONLY for txt2img)
    if workflow == 'txt2img':
        get_resolution('resolution', 'Resolution', loaded_parameter_dict, results, valid_labels=resolution_labels)
    else:
        results.append(gr.update())

    # 6. CFG Scale
    if workflow in ['txt2img', 'inpaint_sdxl', 'outpaint_sdxl', 'super_upscale']:
        get_number('cfg_scale', 'Guidance Scale', loaded_parameter_dict, results)
    else:
        results.append(gr.update())

    # 7-12. Hidden fields skipped
    results.append(gr.update())  # sharpness
    results.extend([gr.update(), gr.update(), gr.update()])  # adm_guidance
    results.append(gr.update())  # adaptive_cfg
    results.append(gr.update())  # clip_skip

    # 13. Base Model
    if workflow in ['txt2img', 'inpaint_sdxl', 'outpaint_sdxl', 'super_upscale', 'color_enhance']:
        get_str('base_model', 'Base Model', loaded_parameter_dict, results)
    else:
        results.append(gr.update())

    # 14. VAE (display-only)
    results.append(gr.update())

    # 15. Sampler
    if workflow in ['txt2img', 'inpaint_sdxl', 'outpaint_sdxl', 'super_upscale', 'flux_fill_inpaint', 'flux_fill_remove']:
        get_str('sampler', 'Sampler', loaded_parameter_dict, results)
    else:
        results.append(gr.update())

    # 16. Scheduler
    if workflow in ['txt2img', 'inpaint_sdxl', 'outpaint_sdxl', 'super_upscale', 'flux_fill_inpaint', 'flux_fill_remove', 'color_enhance']:
        get_str('scheduler', 'Scheduler', loaded_parameter_dict, results)
    else:
        results.append(gr.update())

    # 17-18. Seed
    if workflow in ['txt2img', 'inpaint_sdxl', 'outpaint_sdxl', 'super_upscale', 'flux_fill_inpaint', 'flux_fill_remove', 'color_enhance']:
        get_seed('seed', 'Seed', loaded_parameter_dict, results)
    else:
        results.extend([gr.update(), gr.update()])

    # 19-20. Route-owned inpaint patch engines
    if workflow == 'outpaint_sdxl':
        get_outpaint_engine_version('outpaint_engine', None, loaded_parameter_dict, results)
    else:
        results.append(gr.update())

    if workflow == 'inpaint_sdxl':
        get_inpaint_engine_version('inpaint_engine', None, loaded_parameter_dict, results)
    else:
        results.append(gr.update())

    # 21. Inpaint Route
    if workflow == 'inpaint_sdxl':
        get_inpaint_route('inpaint_route', 'Inpaint Route', loaded_parameter_dict, results)
    else:
        results.append(gr.update())

    # 22-25. Workflow-owned tab-local prompts
    if workflow in ['inpaint_sdxl', 'flux_fill_inpaint']:
        get_str('inpaint_prompt', None, loaded_parameter_dict, results)
    else:
        results.append(gr.update())

    if workflow == 'outpaint_sdxl':
        get_str('outpaint_prompt', None, loaded_parameter_dict, results)
    else:
        results.append(gr.update())

    if workflow == 'flux_fill_remove':
        get_str('prompt_description', None, loaded_parameter_dict, results)
    else:
        results.append(gr.update())

    if workflow == 'color_enhance':
        get_str('prompt_description', None, loaded_parameter_dict, results)
    else:
        results.append(gr.update())

    # 26-27. Buttons
    if is_generating:
        results.append(gr.update())
    else:
        results.append(gr.update(visible=True))

    results.append(gr.update(visible=False))

    # 28+. LoRAs
    if workflow in ['txt2img', 'inpaint_sdxl', 'outpaint_sdxl', 'super_upscale', 'color_enhance']:
        loras_list = loaded_parameter_dict.get('loras', [])
        if isinstance(loras_list, list) and loras_list:
            for i in range(modules.config.default_max_lora_number):
                key = f'lora_combined_{i + 1}'
                if i < len(loras_list):
                    item = loras_list[i]
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        loaded_parameter_dict[key] = f"{item[0]} : {item[1]}"
        for i in range(modules.config.default_max_lora_number):
            get_lora(f'lora_combined_{i + 1}', f'LoRA {i + 1}', loaded_parameter_dict, results)
    else:
        for _ in range(modules.config.default_max_lora_number):
            results.extend([gr.update(), gr.update(), gr.update()])

    return results


def load_parameter_button_click(raw_metadata: dict | str, is_generating: bool):
    return _load_parameter_button_click(raw_metadata, is_generating, convert_legacy=True)


def load_preset_button_click(raw_metadata: dict | str, is_generating: bool):
    return _load_parameter_button_click(raw_metadata, is_generating, convert_legacy=False)


def parse_meta_from_preset(preset_content):
    assert isinstance(preset_content, dict)
    preset_prepared = {}
    items = preset_content
    is_initial = len(items) == 0

    def get_preset_key_fallback(key):
        mapping = {
            "Selected_model": "default_model",
        }
        attr_name = mapping.get(key, key)
        return getattr(modules.config, attr_name, None)

    for settings_key, meta_key in modules.config.possible_preset_keys.items():
        if not is_initial and settings_key not in items:
            continue

        if settings_key == "default_loras":
            loras = get_preset_key_fallback(settings_key)
            if settings_key in items:
                loras = items[settings_key]
            max_loras = modules.config.default_max_lora_number if is_initial else len(loras)
            for index, lora in enumerate(loras[:max_loras]):
                preset_prepared[f'lora_combined_{index + 1}'] = ' : '.join(map(str, lora))
        elif settings_key == "default_aspect_ratio":
            default_aspect_ratio = items.get(settings_key) or get_preset_key_fallback(settings_key)
            if default_aspect_ratio is not None:
                clean_str = default_aspect_ratio
                for sep in ['*', 'x', '×', '횞']:
                    clean_str = clean_str.replace(sep, ' ')
                tokens = clean_str.strip().split()
                if len(tokens) >= 2:
                    width, height = tokens[0], tokens[1]
                    preset_prepared[meta_key] = (width, height)
        else:
            preset_prepared[meta_key] = items[settings_key] if settings_key in items and items[settings_key] is not None else get_preset_key_fallback(settings_key)

        if settings_key == "default_styles" or settings_key == "default_aspect_ratio":
            if meta_key in preset_prepared:
                preset_prepared[meta_key] = str(preset_prepared[meta_key])

    return preset_prepared


def trigger_metadata_import(file, state_is_generating):
    parameters, metadata_scheme = modules.meta_parser.read_info_from_image(file)
    if parameters is None:
        print('Could not find metadata in the image!')
        parsed_parameters = {}
    else:
        metadata_parser = modules.meta_parser.get_metadata_parser(metadata_scheme)
        parsed_parameters = metadata_parser.to_json(parameters)

    return load_parameter_button_click(parsed_parameters, state_is_generating)
