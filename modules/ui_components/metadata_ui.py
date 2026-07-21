import json
import gradio as gr
from pathlib import Path
import modules.config
import modules.flags
from modules.flux_fill_surface import normalize_flux_fill_inpaint_route
from modules.flags import SAMPLERS
from modules.util import unquote, get_file_from_folder_list
import modules.meta_parser

OVERWRITE_DIMENSION_MAX = 2048

def get_str(key: str, fallback: str | None, source_dict: dict, results: list, default=None) -> str | None:
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert isinstance(h, str)
        results.append(h)
        return h
    except:
        results.append(gr.update())
        return None

def get_list(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        h = eval(h)
        assert isinstance(h, list)

        if key == 'styles':
            h = [s for s in h if s != 'Fooocus V2']

        results.append(h)
    except:
        results.append(gr.update())

def get_number(key: str, fallback: str | None, source_dict: dict, results: list, default=None, cast_type=float):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert h is not None
        h = cast_type(h)
        results.append(h)
    except:
        results.append(gr.update())

def get_image_number(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert h is not None
        int(h)
        # Queueing is now the repetition mechanism, so the UI always loads a
        # single image per Generate click regardless of stored metadata.
        results.append(1)
    except:
        results.append(1)

def get_steps(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert h is not None
        h = int(h)
        results.append(h)
    except:
        results.append(-1)

def get_resolution(key: str, fallback: str | None, source_dict: dict, results: list, default=None, valid_labels=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        width, height = eval(h)
        formatted = modules.config.add_ratio(f'{width}*{height}')
        if valid_labels is None:
            valid_labels = modules.config.available_aspect_ratios_labels
        if formatted in valid_labels:
            results.append(formatted)
            results.append(-1)
            results.append(-1)
        elif int(width) <= OVERWRITE_DIMENSION_MAX and int(height) <= OVERWRITE_DIMENSION_MAX:
            results.append(gr.update())
            results.append(int(width))
            results.append(int(height))
        else:
            results.append(gr.update())
            results.append(-1)
            results.append(-1)
    except:
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())

def get_seed(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert h is not None
        h = int(h)
        results.append(False)
        results.append(h)
    except:
        results.append(gr.update())
        results.append(gr.update())

def get_inpaint_engine_version(key: str, fallback: str | None, source_dict: dict, results: list, default=None) -> str | None:
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        h = modules.flags.normalize_inpaint_engine_version(h, default=modules.config.default_inpaint_engine_version)
        assert isinstance(h, str) and h in modules.flags.inpaint_engine_versions
        results.append(h)
        return h
    except:
        results.append('empty')
        return None

def get_outpaint_engine_version(key: str, fallback: str | None, source_dict: dict, results: list, default=None) -> str | None:
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        h = modules.flags.normalize_inpaint_engine_version(h, default=modules.config.default_outpaint_engine_version)
        assert isinstance(h, str) and h in modules.flags.inpaint_engine_versions
        results.append(h)
        return h
    except:
        results.append('empty')
        return None

def get_inpaint_route(key: str, fallback: str | None, source_dict: dict, results: list, default=None) -> str | None:
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        if h == 'empty':
            h = modules.config.default_inpaint_route

        normalized = normalize_flux_fill_inpaint_route(h)
        results.append(normalized)
        return normalized
    except:
        results.append('sdxl')
        return None

def get_inpaint_method(key: str, fallback: str | None, source_dict: dict, results: list, default=None) -> str | None:
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert isinstance(h, str) and h in modules.flags.inpaint_options
        results.append(h)
        return h
    except:
        results.append(gr.update())
        return None

def get_adm_guidance(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        p, n, e = eval(h)
        results.append(float(p))
        results.append(float(n))
        results.append(float(e))
    except:
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())

def get_lora(key: str, fallback: str | None, source_dict: dict, results: list):
    try:
        split_data = source_dict.get(key, source_dict.get(fallback)).split(' : ')
        enabled = True
        name = split_data[0]
        weight = split_data[1]

        if len(split_data) == 3:
            enabled = split_data[0] == 'True'
            name = split_data[1]
            weight = split_data[2]

        # name validation could be added here if needed

        weight = float(weight)
        results.append(enabled)
        results.append(name)
        results.append(weight)
    except:
        results.append(True)
        results.append('None')
        results.append(1)

def load_parameter_button_click(raw_metadata: dict | str, is_generating: bool):
    loaded_parameter_dict = raw_metadata
    if isinstance(raw_metadata, str):
        loaded_parameter_dict = json.loads(raw_metadata)
    assert isinstance(loaded_parameter_dict, dict)

    results = []
    normalized_parameter_dict = dict(loaded_parameter_dict)
    base_model_name = loaded_parameter_dict.get('base_model', loaded_parameter_dict.get('Base Model'))
    active_base_model_name = None

    if isinstance(base_model_name, str):
        active_base_model_name = modules.config.coerce_active_base_model_selection(base_model_name)
        normalized_parameter_dict['base_model'] = active_base_model_name
        normalized_parameter_dict['Base Model'] = active_base_model_name

    resolution_labels = modules.config.get_aspect_ratio_labels_for_model(
        active_base_model_name or modules.config.default_base_model_name
    )

    get_str('prompt', 'Prompt', normalized_parameter_dict, results)
    get_str('negative_prompt', 'Negative Prompt', normalized_parameter_dict, results)
    get_list('styles', 'Styles', normalized_parameter_dict, results)
    get_steps('steps', 'Steps', normalized_parameter_dict, results)
    get_resolution('resolution', 'Resolution', normalized_parameter_dict, results, valid_labels=resolution_labels)
    get_number('guidance_scale', 'Guidance Scale', normalized_parameter_dict, results)
    get_number('sharpness', 'Sharpness', normalized_parameter_dict, results)
    get_adm_guidance('adm_guidance', 'ADM Guidance', normalized_parameter_dict, results)
    get_number('adaptive_cfg', 'CFG Mimicking from TSNR', normalized_parameter_dict, results)
    get_number('clip_skip', 'CLIP Skip', normalized_parameter_dict, results, cast_type=int)
    get_str('base_model', 'Base Model', normalized_parameter_dict, results)
    get_str('vae', 'VAE', normalized_parameter_dict, results)
    get_str('sampler', 'Sampler', normalized_parameter_dict, results)
    get_str('scheduler', 'Scheduler', normalized_parameter_dict, results)
    get_seed('seed', 'Seed', normalized_parameter_dict, results)
    get_outpaint_engine_version('outpaint_engine_version', 'Outpaint Engine Version', normalized_parameter_dict, results)
    get_inpaint_engine_version('inpaint_engine_version', 'Inpaint Engine Version', normalized_parameter_dict, results)
    get_inpaint_route('inpaint_route', 'Inpaint Route', normalized_parameter_dict, results)

    if is_generating:
        results.append(gr.update())
    else:
        results.append(gr.update(visible=True))

    results.append(gr.update(visible=False))

    for i in range(modules.config.default_max_lora_number):
        get_lora(f'lora_combined_{i + 1}', f'LoRA {i + 1}', normalized_parameter_dict, results)

    return results

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



