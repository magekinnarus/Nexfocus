import os
import json
import math
import numbers
import time

import args_manager
import tempfile
import modules.flags
import modules.sdxl_styles
import modules.model_taxonomy
import modules.model_catalog_index
from backend import environment_profile as memory_environment_profiles

from modules.model_loader import load_file_from_url
from modules.extra_utils import makedirs_with_log, get_files_from_folder, try_eval_env_var
from modules.flags import OutputFormat, MetadataScheme


def get_config_path(key, default_value):
    env = os.getenv(key)
    if env is not None and isinstance(env, str):
        print(f"Environment: {key} = {env}")
        return env

    candidate = default_value
    if getattr(args_manager.args, 'colab', False):
        if key == 'config_path' and default_value == "./config.txt":
            candidate = "./config_colab.txt"
        elif key == 'config_example_path' and default_value == "config_modification_tutorial.txt":
            candidate = "config_colab_modification_tutorial.txt"

    return os.path.abspath(candidate)

config_path = get_config_path('config_path', "./config.txt")
config_example_path = get_config_path('config_example_path', "config_modification_tutorial.txt")
config_dict = {}
always_save_keys = []
visited_keys = []


def try_load_json_file(path):
    try:
        with open(path, "r", encoding="utf-8") as json_file:
            return json.load(json_file)
    except Exception as e:
        print(f'Load json file failed: {path}')
        print(e)
    return {}

runtime_defaults = try_load_json_file(os.path.abspath('./configs/defaults/runtime_default.json'))
resolution_set_sdxl = try_load_json_file(os.path.abspath('./configs/resolution_sets/sdxl.json'))
resolution_set_sd15 = try_load_json_file(os.path.abspath('./configs/resolution_sets/sd15.json'))
resolution_sets = {
    'sdxl': resolution_set_sdxl,
    'sd15': resolution_set_sd15,
}


def get_resolution_set_config(resolution_set_id):
    normalized_id = modules.model_taxonomy.normalize_resolution_set_id(resolution_set_id)
    return resolution_sets.get(normalized_id, {})


def get_available_aspect_ratios_for_architecture(architecture=None, sub_architecture=None):
    resolution_set_id = modules.model_taxonomy.resolve_resolution_set_id(architecture, sub_architecture)
    resolution_set = get_resolution_set_config(resolution_set_id)
    if resolution_set_id == modules.model_taxonomy.ARCHITECTURE_SD15:
        fallback = modules.flags.sd15_aspect_ratios
    else:
        fallback = modules.flags.sdxl_aspect_ratios
    return resolution_set.get('available_aspect_ratios', fallback)


def get_default_aspect_ratio_for_architecture(architecture=None, sub_architecture=None):
    available_ratios = get_available_aspect_ratios_for_architecture(architecture, sub_architecture)
    preferred_ratio = modules.model_taxonomy.get_preferred_aspect_ratio(architecture, sub_architecture)
    if preferred_ratio in available_ratios:
        return preferred_ratio
    return available_ratios[0]

def _normalize_model_selector(value):
    if value is None:
        return None
    normalized = str(value).replace('\\', '/').strip()
    return normalized or None


def _build_model_selector_candidates(name_or_path, folder_paths):
    candidates = []

    def add_candidate(candidate):
        normalized = _normalize_model_selector(candidate)
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    add_candidate(name_or_path)
    if name_or_path is None:
        return candidates

    value = str(name_or_path)
    add_candidate(os.path.basename(value))

    if os.path.isabs(value):
        for folder in folder_paths:
            try:
                relative_path = os.path.relpath(value, folder)
            except ValueError:
                continue
            if relative_path.startswith('..'):
                continue
            add_candidate(relative_path)

    return candidates


def resolve_model_catalog_entry(name_or_path, root_keys=('checkpoints', 'unet'), folder_paths=None):
    if folder_paths is None:
        folder_paths = paths_checkpoints

    index = modules.model_catalog_index.load_runtime_model_catalog_index(get_model_catalog_directories())
    candidates = _build_model_selector_candidates(name_or_path, folder_paths)

    for candidate in candidates:
        record = index.find_by_relative_path(candidate, root_keys=root_keys)
        if record is not None:
            return record.entry

    for candidate in candidates:
        record = index.find_by_name(candidate, root_keys=root_keys)
        if record is not None:
            return record.entry

    for candidate in candidates:
        entry = index.get(candidate)
        if entry is not None and (root_keys is None or entry.root_key in set(root_keys)):
            return entry

    return None


def resolve_dropdown_choice(name_or_path, choices, *, folder_paths=None, root_keys=None):
    normalized_choices = {}
    for choice in list(choices or []):
        normalized = _normalize_model_selector(choice)
        if normalized is not None and normalized not in normalized_choices:
            normalized_choices[normalized] = choice

    if not normalized_choices:
        return None

    candidates = _build_model_selector_candidates(name_or_path, folder_paths or [])

    if root_keys is not None:
        entry = resolve_model_catalog_entry(name_or_path, root_keys=root_keys, folder_paths=folder_paths)
        if entry is not None:
            for value in (
                getattr(entry, 'relative_path', None),
                getattr(entry, 'name', None),
                getattr(entry, 'alias', None),
                getattr(entry, 'id', None),
            ):
                normalized = _normalize_model_selector(value)
                if normalized and normalized not in candidates:
                    candidates.append(normalized)

    for candidate in candidates:
        if candidate in normalized_choices:
            return normalized_choices[candidate]

    return None


def resolve_model_taxonomy(name_or_path, root_keys=('checkpoints', 'unet'), folder_paths=None):
    entry = resolve_model_catalog_entry(name_or_path, root_keys=root_keys, folder_paths=folder_paths)
    if entry is not None:
        return modules.model_taxonomy.build_resolved_model_taxonomy(
            architecture=entry.architecture,
            sub_architecture=entry.sub_architecture,
            compatibility_family=entry.compatibility_family,
            source='catalog',
            catalog_entry_id=entry.id,
        )

    architecture, sub_architecture = modules.model_taxonomy.infer_model_taxonomy_from_filename(name_or_path)
    if architecture is not None:
        return modules.model_taxonomy.build_resolved_model_taxonomy(
            architecture=architecture,
            sub_architecture=sub_architecture,
            source='filename',
        )

    return modules.model_taxonomy.build_resolved_model_taxonomy(source='default')


def get_aspect_ratios_for_model(name_or_path, root_keys=('checkpoints', 'unet'), folder_paths=None):
    taxonomy = resolve_model_taxonomy(name_or_path, root_keys=root_keys, folder_paths=folder_paths)
    return get_available_aspect_ratios_for_architecture(taxonomy.architecture, taxonomy.sub_architecture)


def get_aspect_ratio_labels_for_model(name_or_path, root_keys=('checkpoints', 'unet'), folder_paths=None):
    taxonomy = resolve_model_taxonomy(name_or_path, root_keys=root_keys, folder_paths=folder_paths)
    return get_available_aspect_ratio_labels_for_architecture(taxonomy.architecture, taxonomy.sub_architecture)


def get_default_aspect_ratio_for_model(name_or_path, root_keys=('checkpoints', 'unet'), folder_paths=None):
    taxonomy = resolve_model_taxonomy(name_or_path, root_keys=root_keys, folder_paths=folder_paths)
    return get_default_aspect_ratio_for_architecture(taxonomy.architecture, taxonomy.sub_architecture)


def get_default_aspect_ratio_label_for_model(name_or_path, root_keys=('checkpoints', 'unet'), folder_paths=None):
    return add_ratio(get_default_aspect_ratio_for_model(name_or_path, root_keys=root_keys, folder_paths=folder_paths))


config_dict.update(runtime_defaults)

try:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as json_file:
            config_dict.update(json.load(json_file))
            always_save_keys = list(config_dict.keys())
except Exception as e:
    print(f'Failed to load config file "{config_path}" . The reason is: {str(e)}')
    print('Please make sure that:')
    print(f'1. The file "{config_path}" is a valid text file, and you have access to read it.')
    print('2. Use "\\\\" instead of "\\" when describing paths.')
    print('3. There is no "," before the last "}".')
    print('4. All key/value formats are correct.')


def try_load_deprecated_user_path_config():
    global config_dict

    if not os.path.exists('user_path_config.txt'):
        return

    try:
        deprecated_config_dict = json.load(open('user_path_config.txt', "r", encoding="utf-8"))

        def replace_config(old_key, new_key):
            if old_key in deprecated_config_dict:
                config_dict[new_key] = deprecated_config_dict[old_key]
                del deprecated_config_dict[old_key]

        replace_config('modelfile_path', 'path_checkpoints')
        replace_config('lorafile_path', 'path_loras')
        replace_config('embeddings_path', 'path_embeddings')
        replace_config('vae_approx_path', 'path_vae_approx')
        replace_config('upscale_models_path', 'path_upscale_models')
        replace_config('inpaint_models_path', 'path_inpaint')
        replace_config('controlnet_models_path', 'path_controlnet')
        replace_config('clip_vision_models_path', 'path_clip_vision')

        replace_config('temp_outputs_path', 'path_outputs')

        if deprecated_config_dict.get("default_model", None) == 'juggernautXL_version6Rundiffusion.safetensors':
            os.replace('user_path_config.txt', 'user_path_config-deprecated.txt')
            print('Config updated successfully in silence. '
                  'A backup of previous config is written to "user_path_config-deprecated.txt".')
            return

        if input("Newer models and configs are available. "
                 "Download and update files? [Y/n]:") in ['n', 'N', 'No', 'no', 'NO']:
            config_dict.update(deprecated_config_dict)
            print('Loading using deprecated old models and deprecated old configs.')
            return
        else:
            os.replace('user_path_config.txt', 'user_path_config-deprecated.txt')
            print('Config updated successfully by user. '
                  'A backup of previous config is written to "user_path_config-deprecated.txt".')
            return
    except Exception as e:
        print('Processing deprecated config failed')
        print(e)
    return


try_load_deprecated_user_path_config()

def get_presets():
    preset_folder = 'presets'
    presets = ['initial']
    if not os.path.exists(preset_folder):
        print('No presets found.')
        return presets

    visible_presets = []
    for filename in os.listdir(preset_folder):
        if not filename.endswith('.json'):
            continue
        preset_name = filename[:filename.index('.json')]
        visible_presets.append(preset_name)
    return presets + visible_presets

def update_presets():
    global available_presets
    available_presets = get_presets()

def try_get_preset_content(preset):
    if isinstance(preset, str):
        preset_path = os.path.abspath(f'./presets/{preset}.json')
        try:
            if os.path.exists(preset_path):
                with open(preset_path, "r", encoding="utf-8") as json_file:
                    json_content = json.load(json_file)
                    print(f'Loaded preset: {preset_path}')
                    return json_content
            else:
                raise FileNotFoundError
        except Exception as e:
            print(f'Load preset [{preset_path}] failed')
            print(e)
    return {}

available_presets = get_presets()
preset = args_manager.args.preset
config_dict.update(try_get_preset_content(preset))

def get_path_output() -> str:
    """
    Checking output path argument and overriding default path.
    """
    global config_dict
    path_output = get_dir_or_set_default('path_outputs', '../outputs/', make_directory=True)
    if args_manager.args.output_path:
        print(f'Overriding config value path_outputs with {args_manager.args.output_path}')
        config_dict['path_outputs'] = path_output = args_manager.args.output_path
    return path_output


def get_dir_or_set_default(key, default_value, as_array=False, make_directory=False):
    global config_dict, visited_keys, always_save_keys

    if key not in visited_keys:
        visited_keys.append(key)

    if key not in always_save_keys:
        always_save_keys.append(key)

    v = os.getenv(key)
    if v is not None:
        print(f"Environment: {key} = {v}")
        config_dict[key] = v
    else:
        v = config_dict.get(key, None)

    if isinstance(v, str):
        if make_directory:
            makedirs_with_log(v)
        if os.path.exists(v) and os.path.isdir(v):
            return v if not as_array else [v]
    elif isinstance(v, list):
        if make_directory:
            for d in v:
                makedirs_with_log(d)
        if all([os.path.exists(d) and os.path.isdir(d) for d in v]):
            return v if as_array else v[0]

    if v is not None:
        print(f'Failed to load config key: {json.dumps({key:v})} is invalid or does not exist; will use {json.dumps({key:default_value})} instead.')
    if isinstance(default_value, list):
        dp = []
        for path in default_value:
            abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
            dp.append(abs_path)
            os.makedirs(abs_path, exist_ok=True)
    else:
        dp = os.path.abspath(os.path.join(os.path.dirname(__file__), default_value))
        os.makedirs(dp, exist_ok=True)
        if as_array:
            dp = [dp]
    config_dict[key] = dp
    return dp


paths_checkpoints = get_dir_or_set_default('path_checkpoints', ['../models/checkpoints/'], True, True)
paths_checkpoint_bases = list(paths_checkpoints) if isinstance(paths_checkpoints, list) else [paths_checkpoints]
paths_loras = get_dir_or_set_default('path_loras', ['../models/loras/'], True, True)
path_embeddings = get_dir_or_set_default('path_embeddings', '../models/embeddings/', make_directory=True)
path_vae_approx = get_dir_or_set_default('path_vae_approx', '../models/vae_approx/', make_directory=True)
path_vae = get_dir_or_set_default('path_vae', '../models/vae/', True, True)
path_unet = get_dir_or_set_default('path_unet', '../models/unet/', True, True)
path_clip = get_dir_or_set_default('path_clip', '../models/clip/', True, True)
path_upscale_models = get_dir_or_set_default('path_upscale_models', '../models/upscale_models/', True, True)
path_inpaint = get_dir_or_set_default('path_inpaint', '../models/inpaint/', make_directory=True)
path_controlnet = get_dir_or_set_default('path_controlnet', '../models/controlnet/', True, True)
path_vision_support = get_dir_or_set_default('path_vision_support', '../models/vision_support/', make_directory=True)
path_clip_vision = get_dir_or_set_default('path_clip_vision', '../models/vision_support/clip_vision/', make_directory=True)
path_preprocessors = get_dir_or_set_default('path_preprocessors', '../models/preprocessors/', make_directory=True)
path_insightface = get_dir_or_set_default('path_insightface', '../models/insightface/', make_directory=True)
path_removals = get_dir_or_set_default('path_removals', '../models/removals/', make_directory=True)
path_loras_lcm = get_dir_or_set_default('path_loras_lcm', '../models/performance_loras/lcm/', make_directory=True)
path_loras_lightning = get_dir_or_set_default('path_loras_lightning', '../models/performance_loras/lightning/', make_directory=True)


# Add unet path to checkpoints for base model selection
if not isinstance(paths_checkpoints, list):
    paths_checkpoints = [paths_checkpoints]

if isinstance(path_unet, list):
    for x in path_unet:
        if x not in paths_checkpoints:
            paths_checkpoints.append(x)
else:
    if path_unet not in paths_checkpoints:
        paths_checkpoints.append(path_unet)

paths_clips = []
if isinstance(path_clip, list):
    paths_clips += path_clip
else:
    paths_clips.append(path_clip)

paths_lora_discovery = []
for folder in paths_loras + [path_loras_lcm, path_loras_lightning]:
    if folder not in paths_lora_discovery:
        paths_lora_discovery.append(folder)
paths_lora_lookup = list(paths_lora_discovery)


path_outputs = get_path_output()
path_temp_outputs = os.path.join(path_outputs, 'temp')
os.makedirs(path_temp_outputs, exist_ok=True)
path_download_manifests = get_dir_or_set_default('path_download_manifests', '../configs/download_manifests/')
path_model_catalogs_preset = get_dir_or_set_default('path_model_catalogs_preset', '../configs/model_catalogs/', make_directory=True)
path_model_catalogs_user = get_dir_or_set_default('path_model_catalogs_user', '../configs/model_catalogs/user/', make_directory=True)
path_model_thumbnails = get_dir_or_set_default('path_model_thumbnails', '../thumbnails/', make_directory=True)


def get_model_catalog_directories():
    directories = []
    for path in [path_model_catalogs_preset, path_model_catalogs_user]:
        if path and path not in directories:
            directories.append(path)
    return directories


def get_writable_model_catalog_directory():
    return path_model_catalogs_user


def get_model_thumbnail_directory():
    return path_model_thumbnails


def get_default_thumbnail_relative_path():
    return 'thumbnails/default_0001.png'


def get_model_thumbnail_size():
    return int(get_config_item_or_set_default(
        'model_thumbnail_size',
        400,
        lambda value: isinstance(value, int) and value > 0,
        expected_type=int,
    ))

asset_root_paths = {
    'checkpoints': paths_checkpoints[0],
    'loras': paths_loras[0],
    'loras_lcm': path_loras_lcm,
    'loras_lightning': path_loras_lightning,
    'embeddings': path_embeddings,
    'vae_approx': path_vae_approx,
    'vae': path_vae[0] if isinstance(path_vae, list) else path_vae,
    'unet': path_unet[0] if isinstance(path_unet, list) else path_unet,
    'clip': paths_clips[0],
    'upscale_models': path_upscale_models[0] if isinstance(path_upscale_models, list) else path_upscale_models,
    'inpaint': path_inpaint,
    'controlnet_models': path_controlnet[0],
    'clip_vision': path_clip_vision,
    'vision_support': path_vision_support,
    'preprocessors': path_preprocessors,
    'insightface': path_insightface,
    'removals': path_removals,
    'outputs': path_outputs,
}

asset_root_path_groups = {
    'checkpoints': list(paths_checkpoints),
    'loras': list(paths_loras),
    'loras_lcm': [path_loras_lcm],
    'loras_lightning': [path_loras_lightning],
    'embeddings': [path_embeddings],
    'vae_approx': [path_vae_approx],
    'vae': list(path_vae) if isinstance(path_vae, list) else [path_vae],
    'unet': list(path_unet) if isinstance(path_unet, list) else [path_unet],
    'clip': list(paths_clips),
    'upscale_models': list(path_upscale_models) if isinstance(path_upscale_models, list) else [path_upscale_models],
    'inpaint': [path_inpaint],
    'controlnet_models': list(path_controlnet),
    'clip_vision': [path_clip_vision],
    'vision_support': [path_vision_support],
    'preprocessors': [path_preprocessors],
    'insightface': [path_insightface],
    'removals': [path_removals],
    'outputs': [path_outputs],
}

_persistent_asset_filenames = {
    'upscale_models': {
        '2xnomosuni_span_multijpg_ldl.pth',
        '4xnomos2_otf_esrgan.pth',
    },
    'vae': {
        'sdxl_vae.safetensors',
        'fixfp16errorssdxllowermemoryuse_v10.safetensors',
        'vae-ft-mse-840000-ema-pruned.safetensors',
    },
    'clip': {
        'flux_empty_conditioning.pt',
    },
}


def get_asset_root_path(key):
    if key not in asset_root_paths:
        raise KeyError(f'Unknown asset root path key: {key}')
    return asset_root_paths[key]


def get_asset_root_paths(key):
    if key not in asset_root_path_groups:
        raise KeyError(f'Unknown asset root path key: {key}')
    return list(asset_root_path_groups[key])


def get_preferred_asset_root_path(key, *, file_name=None, relative_path=None):
    roots = get_asset_root_paths(key)
    if not roots:
        raise KeyError(f'No configured filesystem path for asset root key: {key}')
    if len(roots) == 1:
        return roots[0]

    persistent_names = _persistent_asset_filenames.get(key)
    if not persistent_names:
        return roots[0]

    candidate_name = os.path.basename(str(file_name or relative_path or '')).strip()
    normalized_name = str(candidate_name or '').strip().lower()
    if normalized_name and normalized_name in persistent_names:
        return roots[0]

    return roots[1]


def get_download_manifest_root():
    return path_download_manifests


def get_download_manifest_asset_dir():
    return os.path.join(path_download_manifests, 'assets')


def get_config_item_or_set_default(key, default_value, validator, disable_empty_as_none=False, expected_type=None):
    global config_dict, visited_keys

    if key not in visited_keys:
        visited_keys.append(key)
    
    v = os.getenv(key)
    if v is not None:
        v = try_eval_env_var(v, expected_type)
        print(f"Environment: {key} = {v}")
        config_dict[key] = v

    if key not in config_dict:
        config_dict[key] = default_value
        return default_value

    v = config_dict.get(key, None)
    if not disable_empty_as_none:
        if v is None or v == '':
            v = 'None'
    if validator(v):
        return v
    else:
        if v is not None:
            print(f'Failed to load config key: {json.dumps({key:v})} is invalid; will use {json.dumps({key:default_value})} instead.')
        config_dict[key] = default_value
        return default_value


def _get_optional_memory_policy_override(key):
    value = get_config_item_or_set_default(
        key=key,
        default_value=None,
        validator=lambda x: x is None or x == 'None' or isinstance(x, numbers.Number),
        expected_type=numbers.Number
    )
    if value in (None, 'None'):
        return None
    return float(value)


memory_environment_profile_override = get_config_item_or_set_default(
    key='memory_environment_profile',
    default_value=memory_environment_profiles.PROFILE_AUTO,
    validator=lambda x: isinstance(x, str) and x.lower() in memory_environment_profiles.KNOWN_PROFILE_OVERRIDES,
    expected_type=str
)
cli_memory_environment_profile_override = getattr(args_manager.args, 'memory_environment_profile', None)
if isinstance(cli_memory_environment_profile_override, str) and cli_memory_environment_profile_override.strip():
    normalized_cli_memory_profile = cli_memory_environment_profile_override.strip().lower()
    if normalized_cli_memory_profile not in memory_environment_profiles.KNOWN_PROFILE_OVERRIDES:
        raise ValueError(
            '--memory-environment-profile must be one of: '
            + ', '.join(sorted(memory_environment_profiles.KNOWN_PROFILE_OVERRIDES))
        )
    print(
        f'Overriding config value memory_environment_profile with '
        f'{cli_memory_environment_profile_override}'
    )
    memory_environment_profile_override = normalized_cli_memory_profile
    config_dict['memory_environment_profile'] = memory_environment_profile_override
memory_profile_custom_name = get_config_item_or_set_default(
    key='memory_profile_custom_name',
    default_value='Custom Override',
    validator=lambda x: isinstance(x, str) and len(x.strip()) > 0,
    expected_type=str
)
hardware_total_ram_override_mb = getattr(args_manager.args, 'hardware_total_ram_mb', None)
if hardware_total_ram_override_mb is not None:
    hardware_total_ram_override_mb = float(hardware_total_ram_override_mb)
    if hardware_total_ram_override_mb <= 0.0:
        raise ValueError('--hardware-total-ram-mb must be greater than 0.')
    print(f'Overriding detected total RAM with {hardware_total_ram_override_mb:.0f} MB')

hardware_total_vram_override_mb = getattr(args_manager.args, 'hardware_total_vram_mb', None)
if hardware_total_vram_override_mb is not None:
    hardware_total_vram_override_mb = float(hardware_total_vram_override_mb)
    if hardware_total_vram_override_mb <= 0.0:
        raise ValueError('--hardware-total-vram-mb must be greater than 0.')
    print(f'Overriding detected total VRAM with {hardware_total_vram_override_mb:.0f} MB')

memory_low_ram_headroom_override_mb = _get_optional_memory_policy_override('memory_low_ram_headroom_mb')
memory_critical_ram_headroom_override_mb = _get_optional_memory_policy_override('memory_critical_ram_headroom_mb')
memory_checkpoint_switch_headroom_override_mb = _get_optional_memory_policy_override('memory_checkpoint_switch_ram_headroom_mb')
memory_linux_malloc_trim_trigger_override_mb = _get_optional_memory_policy_override('memory_linux_malloc_trim_trigger_mb')
resolved_memory_environment_profile = memory_environment_profiles.resolve_environment_profile(
    override=memory_environment_profile_override,
    custom_name=memory_profile_custom_name,
    total_ram_mb=hardware_total_ram_override_mb,
    total_vram_mb=hardware_total_vram_override_mb,
    custom_policy_overrides={
        'low_ram_headroom_mb': memory_low_ram_headroom_override_mb,
        'critical_ram_headroom_mb': memory_critical_ram_headroom_override_mb,
        'checkpoint_switch_ram_headroom_mb': memory_checkpoint_switch_headroom_override_mb,
        'linux_malloc_trim_trigger_mb': memory_linux_malloc_trim_trigger_override_mb,
    },
)

import backend.memory_governor as memory_governor
memory_governor.configure_environment(resolved_memory_environment_profile)
print(resolved_memory_environment_profile.startup_message())
print(f"[Startup] Memory policy summary: {memory_governor.policy_summary()}")


def init_temp_path(path: str | None, default_path: str) -> str:
    if args_manager.args.temp_path:
        path = args_manager.args.temp_path

    if path != '' and path != default_path:
        try:
            if not os.path.isabs(path):
                path = os.path.abspath(path)
            os.makedirs(path, exist_ok=True)
            print(f'Using temp path {path}')
            return path
        except Exception as e:
            print(f'Could not create temp path {path}. Reason: {e}')
            print(f'Using default temp path {default_path} instead.')

    os.makedirs(default_path, exist_ok=True)
    return default_path


default_temp_path = os.path.join(tempfile.gettempdir(), 'fooocus')
temp_path = init_temp_path(get_config_item_or_set_default(
    key='temp_path',
    default_value=default_temp_path,
    validator=lambda x: isinstance(x, str),
    expected_type=str
), default_temp_path)
temp_path_cleanup_on_launch = get_config_item_or_set_default(
    key='temp_path_cleanup_on_launch',
    default_value=True,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_base_model_name = default_model = get_config_item_or_set_default(
    key='Selected_model',
    default_value='None',
    validator=lambda x: isinstance(x, str),
    expected_type=str
)
default_model_taxonomy = resolve_model_taxonomy(default_base_model_name)

if getattr(args_manager.args, 'skip_model_load', False):
    print("Skipping model load: Forcing Selected_model to None")
    default_base_model_name = default_model = 'None'
    default_model_taxonomy = resolve_model_taxonomy(default_base_model_name)
default_loras_min_weight = get_config_item_or_set_default(
    key='default_loras_min_weight',
    default_value=runtime_defaults.get('default_loras_min_weight', -2),
    validator=lambda x: isinstance(x, numbers.Number) and -10 <= x <= 10,
    expected_type=numbers.Number
)
default_loras_max_weight = get_config_item_or_set_default(
    key='default_loras_max_weight',
    default_value=runtime_defaults.get('default_loras_max_weight', 2),
    validator=lambda x: isinstance(x, numbers.Number) and -10 <= x <= 10,
    expected_type=numbers.Number
)
default_loras = get_config_item_or_set_default(
    key='default_loras',
    default_value=[
        [
            True,
            "None",
            1.0
        ],
        [
            True,
            "None",
            1.0
        ],
        [
            True,
            "None",
            1.0
        ],
        [
            True,
            "None",
            1.0
        ],
        [
            True,
            "None",
            1.0
        ]
    ],
    validator=lambda x: isinstance(x, list) and all(
        len(y) == 3 and isinstance(y[0], bool) and isinstance(y[1], str) and isinstance(y[2], numbers.Number)
        or len(y) == 2 and isinstance(y[0], str) and isinstance(y[1], numbers.Number)
        for y in x),
    expected_type=list
)
if getattr(args_manager.args, 'skip_model_load', False):
    print("Skipping model load: Forcing default_loras to None")
    default_loras = [[y[0], 'None', y[2]] if len(y) == 3 else [True, 'None', y[1]] for y in default_loras]

default_loras = [(y[0], y[1], y[2]) if len(y) == 3 else (True, y[0], y[1]) for y in default_loras]
default_max_lora_number = get_config_item_or_set_default(
    key='default_max_lora_number',
    default_value=runtime_defaults.get(
        'default_max_lora_number',
        len(default_loras) if isinstance(default_loras, list) and len(default_loras) > 0 else 5
    ),
    validator=lambda x: isinstance(x, int) and x >= 1,
    expected_type=int
)
default_cfg_scale = get_config_item_or_set_default(
    key='default_cfg_scale',
    default_value=7.0,
    validator=lambda x: isinstance(x, numbers.Number),
    expected_type=numbers.Number
)
default_sample_sharpness = get_config_item_or_set_default(
    key='default_sample_sharpness',
    default_value=2.0,
    validator=lambda x: isinstance(x, numbers.Number),
    expected_type=numbers.Number
)
default_sampler = get_config_item_or_set_default(
    key='default_sampler',
    default_value='dpmpp_2m_sde_gpu',
    validator=lambda x: x in modules.flags.sampler_list,
    expected_type=str
)
default_scheduler = get_config_item_or_set_default(
    key='default_scheduler',
    default_value='karras',
    validator=lambda x: x in modules.flags.scheduler_list,
    expected_type=str
)
default_vae = get_config_item_or_set_default(
    key='default_vae',
    default_value=modules.flags.default_vae,
    validator=lambda x: isinstance(x, str),
    expected_type=str
)
default_vae = resolve_dropdown_choice(
    default_vae,
    [modules.flags.default_vae],
    folder_paths=path_vae,
    root_keys=('vae',),
) or (
    modules.flags.default_vae
    if str(default_vae or '').strip() in {'', 'Default (model)', 'Default (Same as model)'}
    else default_vae
)
default_styles = get_config_item_or_set_default(
    key='default_styles',
    default_value=[
        "Fooocus Enhance",
        "Fooocus Sharp"
    ],
    validator=lambda x: isinstance(x, list) and all(y in modules.sdxl_styles.legal_style_names for y in x),
    expected_type=list
)
default_prompt_negative = get_config_item_or_set_default(
    key='default_prompt_negative',
    default_value='',
    validator=lambda x: isinstance(x, str),
    disable_empty_as_none=True,
    expected_type=str
)
default_prompt = get_config_item_or_set_default(
    key='default_prompt',
    default_value='',
    validator=lambda x: isinstance(x, str),
    disable_empty_as_none=True,
    expected_type=str
)
default_image_prompt_checkbox = get_config_item_or_set_default(
    key='default_image_prompt_checkbox',
    default_value=False,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_advanced_checkbox = get_config_item_or_set_default(
    key='default_advanced_checkbox',
    default_value=False,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_developer_debug_mode_checkbox = get_config_item_or_set_default(
    key='default_developer_debug_mode_checkbox',
    default_value=False,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_image_prompt_advanced_checkbox = get_config_item_or_set_default(
    key='default_image_prompt_advanced_checkbox',
    default_value=False,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_max_image_number = get_config_item_or_set_default(
    key='default_max_image_number',
    default_value=32,
    validator=lambda x: isinstance(x, int) and x >= 1,
    expected_type=int
)
default_output_format = get_config_item_or_set_default(
    key='default_output_format',
    default_value='png',
    validator=lambda x: x in OutputFormat.list(),
    expected_type=str
)
default_image_number = get_config_item_or_set_default(
    key='default_image_number',
    default_value=2,
    validator=lambda x: isinstance(x, int) and 1 <= x <= default_max_image_number,
    expected_type=int
)
checkpoint_downloads = get_config_item_or_set_default(
    key='checkpoint_downloads',
    default_value={},
    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items()),
    expected_type=dict
)
lora_downloads = get_config_item_or_set_default(
    key='lora_downloads',
    default_value={},
    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items()),
    expected_type=dict
)
embeddings_downloads = get_config_item_or_set_default(
    key='embeddings_downloads',
    default_value={},
    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items()),
    expected_type=dict
)
vae_downloads = get_config_item_or_set_default(
    key='vae_downloads',
    default_value={},
    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items()),
    expected_type=dict
)
upscale_downloads = get_config_item_or_set_default(
    key='upscale_downloads',
    default_value={},
    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items()),
    expected_type=dict
)
available_aspect_ratios = get_config_item_or_set_default(
    key='available_aspect_ratios',
    default_value=get_available_aspect_ratios_for_architecture(
        default_model_taxonomy.architecture,
        default_model_taxonomy.sub_architecture,
    ),
    validator=lambda x: isinstance(x, list) and all('*' in v for v in x) and len(x) > 1,
    expected_type=list
)
default_aspect_ratio = get_config_item_or_set_default(
    key='default_aspect_ratio',
    default_value=get_default_aspect_ratio_for_architecture(
        default_model_taxonomy.architecture,
        default_model_taxonomy.sub_architecture,
    ),
    validator=lambda x: x in available_aspect_ratios,
    expected_type=str
)
default_inpaint_engine_version = get_config_item_or_set_default(
    key='default_inpaint_engine_version',
    default_value=modules.flags.normalize_inpaint_engine_version(
        runtime_defaults.get('default_inpaint_engine_version', modules.flags.INPAINT_ENGINE_NONE),
        default=modules.flags.INPAINT_ENGINE_NONE,
    ),
    validator=lambda x: modules.flags.normalize_inpaint_engine_version(x, default='') in modules.flags.inpaint_engine_versions,
    expected_type=str
)
default_inpaint_route = get_config_item_or_set_default(
    key='default_inpaint_route',
    default_value=runtime_defaults.get('default_inpaint_route', 'sdxl'),
    validator=lambda x: x in {'sdxl', 'flux'},
    expected_type=str
)
default_outpaint_engine_version = get_config_item_or_set_default(
    key='default_outpaint_engine_version',
    default_value=modules.flags.normalize_inpaint_engine_version(
        runtime_defaults.get('default_outpaint_engine_version', modules.flags.INPAINT_ENGINE_V26),
        default=modules.flags.INPAINT_ENGINE_V26,
    ),
    validator=lambda x: modules.flags.normalize_inpaint_engine_version(x, default='') in modules.flags.inpaint_engine_versions,
    expected_type=str
)
default_outpaint_expansion_size = get_config_item_or_set_default(
    key='default_outpaint_expansion_size',
    default_value=384,
    validator=lambda x: isinstance(x, int) and x > 0,
    expected_type=int
)
default_selected_image_input_tab_id = get_config_item_or_set_default(
    key='default_selected_image_input_tab_id',
    default_value=modules.flags.default_input_image_tab,
    validator=lambda x: x in modules.flags.input_image_tab_ids,
    expected_type=str
)
default_uov_method = get_config_item_or_set_default(
    key='default_uov_method',
    default_value=modules.flags.disabled,
    validator=lambda x: x in modules.flags.uov_list,
    expected_type=str
)
default_controlnet_image_count = get_config_item_or_set_default(
    key='default_controlnet_image_count',
    default_value=4,
    validator=lambda x: isinstance(x, int) and x > 0,
    expected_type=int
)
default_ip_images = {}
default_ip_stop_ats = {}
default_ip_weights = {}
default_ip_types = {}

for image_count in range(default_controlnet_image_count):
    image_count += 1
    default_ip_images[image_count] = get_config_item_or_set_default(
        key=f'default_ip_image_{image_count}',
        default_value='None',
        validator=lambda x: x == 'None' or isinstance(x, str) and os.path.exists(x),
        expected_type=str
    )

    if default_ip_images[image_count] == 'None':
        default_ip_images[image_count] = None

    default_ip_type = get_config_item_or_set_default(
        key=f'default_ip_type_{image_count}',
        default_value=modules.flags.default_ip,
        validator=lambda x: modules.flags.normalize_cn_type(x) in modules.flags.cn_all_types,
        expected_type=str
    )
    default_ip_types[image_count] = modules.flags.resolve_cn_type(default_ip_type)

    default_end, default_weight = modules.flags.default_parameters[default_ip_types[image_count]]

    default_ip_stop_ats[image_count] = get_config_item_or_set_default(
        key=f'default_ip_stop_at_{image_count}',
        default_value=default_end,
        validator=lambda x: isinstance(x, float) and 0 <= x <= 1,
        expected_type=float
    )
    default_ip_weights[image_count] = get_config_item_or_set_default(
        key=f'default_ip_weight_{image_count}',
        default_value=default_weight,
        validator=lambda x: isinstance(x, float) and 0 <= x <= 2,
        expected_type=float
    )

default_inpaint_advanced_masking_checkbox = get_config_item_or_set_default(
    key='default_inpaint_advanced_masking_checkbox',
    default_value=False,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_inpaint_method = get_config_item_or_set_default(
    key='default_inpaint_method',
    default_value=modules.flags.inpaint_option_default,
    validator=lambda x: x in modules.flags.inpaint_options,
    expected_type=str
)
default_cfg_tsnr = get_config_item_or_set_default(
    key='default_cfg_tsnr',
    default_value=7.0,
    validator=lambda x: isinstance(x, numbers.Number),
    expected_type=numbers.Number
)
default_clip_skip = get_config_item_or_set_default(
    key='default_clip_skip',
    default_value=runtime_defaults.get('default_clip_skip', 2),
    validator=lambda x: isinstance(x, int) and 1 <= x <= modules.flags.clip_skip_max,
    expected_type=int
)
default_clip = get_config_item_or_set_default(
    key='default_clip',
    default_value='None',
    validator=lambda x: isinstance(x, str),
    expected_type=str
)
default_overwrite_step = get_config_item_or_set_default(
    key='default_overwrite_step',
    default_value=20,
    validator=lambda x: isinstance(x, int),
    expected_type=int
)
default_overwrite_upscale = get_config_item_or_set_default(
    key='default_overwrite_upscale',
    default_value=-1,
    validator=lambda x: isinstance(x, numbers.Number)
)
example_inpaint_prompts = get_config_item_or_set_default(
    key='example_inpaint_prompts',
    default_value=runtime_defaults.get(
        'example_inpaint_prompts',
        ['highly detailed face', 'detailed girl face', 'detailed man face', 'detailed hand', 'beautiful eyes']
    ),
    validator=lambda x: isinstance(x, list) and all(isinstance(v, str) for v in x),
    expected_type=list
)
default_save_metadata_to_images = get_config_item_or_set_default(
    key='default_save_metadata_to_images',
    default_value=True,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_metadata_scheme = get_config_item_or_set_default(
    key='default_metadata_scheme',
    default_value=MetadataScheme.FOOOCUS_NEX.value,
    validator=lambda x: x in [y[1] for y in modules.flags.metadata_scheme if y[1] == x],
    expected_type=str
)
metadata_created_by = get_config_item_or_set_default(
    key='metadata_created_by',
    default_value='',
    validator=lambda x: isinstance(x, str),
    expected_type=str
)

example_inpaint_prompts = [[x] for x in example_inpaint_prompts]



config_dict["default_loras"] = default_loras = default_loras[:default_max_lora_number] + [[True, 'None', 1.0] for _ in range(default_max_lora_number - len(default_loras))]

# mapping config to meta parameter
possible_preset_keys = {
    "Selected_model": "base_model",
    "default_loras_min_weight": "default_loras_min_weight",
    "default_loras_max_weight": "default_loras_max_weight",
    "default_loras": "<processed>",
    "default_cfg_scale": "guidance_scale",
    "default_sample_sharpness": "sharpness",
    "default_cfg_tsnr": "adaptive_cfg",
    "default_clip_skip": "clip_skip",
    "default_sampler": "sampler",
    "default_scheduler": "scheduler",
    "default_overwrite_step": "steps",
    "default_image_number": "image_number",
    "default_prompt": "prompt",
    "default_prompt_negative": "negative_prompt",
    "default_styles": "styles",
    "default_aspect_ratio": "resolution",
    "default_save_metadata_to_images": "default_save_metadata_to_images",
    "checkpoint_downloads": "checkpoint_downloads",
    "embeddings_downloads": "embeddings_downloads",
    "lora_downloads": "lora_downloads",
    "vae_downloads": "vae_downloads",
    "default_vae": "vae",
    "default_clip": "clip_model",
    # "default_inpaint_method": "inpaint_method", # disabled so inpaint mode doesn't refresh after every preset change
    "default_inpaint_engine_version": "inpaint_engine_version",
    "default_inpaint_route": "inpaint_route",
    "default_outpaint_engine_version": "outpaint_engine_version",
}

REWRITE_PRESET = False

if REWRITE_PRESET and isinstance(args_manager.args.preset, str):
    save_path = 'presets/' + args_manager.args.preset + '.json'
    with open(save_path, "w", encoding="utf-8") as json_file:
        json.dump({k: config_dict[k] for k in possible_preset_keys}, json_file, indent=4)
    print(f'Preset saved to {save_path}. Exiting ...')
    exit(0)


def add_ratio(x):
    a, b = x.replace('*', ' ').split(' ')[:2]
    a, b = int(a), int(b)
    g = math.gcd(a, b)
    return f'{a}x{b} ({a // g}:{b // g})'


def get_available_aspect_ratio_labels_for_architecture(architecture=None, sub_architecture=None):
    return [add_ratio(x) for x in get_available_aspect_ratios_for_architecture(architecture, sub_architecture)]


def get_default_aspect_ratio_label_for_architecture(architecture=None, sub_architecture=None):
    return add_ratio(get_default_aspect_ratio_for_architecture(architecture, sub_architecture))


default_aspect_ratio = add_ratio(default_aspect_ratio)
available_aspect_ratios_labels = [add_ratio(x) for x in available_aspect_ratios]


# Only write config in the first launch.
if not os.path.exists(config_path):
    with open(config_path, "w", encoding="utf-8") as json_file:
        json.dump({k: config_dict[k] for k in always_save_keys}, json_file, indent=4)


# Always write tutorials.
with open(config_example_path, "w", encoding="utf-8") as json_file:
    cpa = config_path.replace("\\", "\\\\")
    json_file.write(f'You can modify your "{cpa}" using the below keys, formats, and examples.\n'
                    f'Do not modify this file. Modifications in this file will not take effect.\n'
                    f'This file is a tutorial and example. Please edit "{cpa}" to really change any settings.\n'
                    + 'Remember to split the paths with "\\\\" rather than "\\", '
                      'and there is no "," before the last "}". \n\n\n')
    json.dump({k: config_dict[k] for k in visited_keys}, json_file, indent=4)

model_filenames = []
clip_filenames = []
lora_filenames = []
embedding_filenames = []
vae_filenames = []
embedding_path_lookup = {}


def get_model_filenames(folder_paths, extensions=None, name_filter=None):
    if extensions is None:
        extensions = ['.pth', '.ckpt', '.bin', '.safetensors', '.fooocus.patch', '.gguf', '.sft']
    files = []

    if not isinstance(folder_paths, list):
        folder_paths = [folder_paths]
    for folder in folder_paths:
        files += get_files_from_folder(folder, extensions, name_filter)

    return files


def is_deprecated_sdxl_base_model_selector(name_or_path):
    normalized = _normalize_model_selector(name_or_path)
    if normalized is None:
        return False

    lowered = normalized.lower()
    if lowered.startswith('flux/'):
        return True
    if lowered.endswith('.gguf'):
        return True

    # Exclude SD 1.5 checkpoints from active base model selection
    taxonomy = resolve_model_taxonomy(normalized)
    if taxonomy.architecture == modules.model_taxonomy.ARCHITECTURE_SD15:
        return True

    return False



def filter_supported_sdxl_base_model_choices(candidates):
    filtered = []
    seen = set()

    for candidate in list(candidates or []):
        normalized = _normalize_model_selector(candidate)
        if normalized is None or normalized in seen:
            continue
        if is_deprecated_sdxl_base_model_selector(normalized):
            continue
        seen.add(normalized)
        filtered.append(candidate)

    return filtered


def coerce_active_base_model_selection(name_or_path, choices=None):
    active_choices = list(choices if choices is not None else model_filenames or [])
    active_choices = filter_supported_sdxl_base_model_choices(active_choices)

    if not active_choices:
        return 'None'

    resolved = resolve_dropdown_choice(
        name_or_path,
        active_choices,
        folder_paths=paths_checkpoints,
        root_keys=('checkpoints', 'unet'),
    )
    return resolved or active_choices[0]


def _filter_model_choices_for_architecture(candidates, architecture, *, root_keys, folder_paths=None):
    filtered = []
    seen = set()
    for candidate in list(candidates or []):
        normalized = _normalize_model_selector(candidate)
        if normalized is None or normalized in seen:
            continue
        taxonomy = resolve_model_taxonomy(candidate, root_keys=root_keys, folder_paths=folder_paths)
        if taxonomy.architecture != architecture:
            continue
        seen.add(normalized)
        filtered.append(candidate)
    return filtered


def get_compatible_clip_choices_for_model(base_model_name):
    base_taxonomy = resolve_model_taxonomy(base_model_name, root_keys=('checkpoints', 'unet'), folder_paths=paths_checkpoints)
    return _filter_model_choices_for_architecture(
        clip_filenames,
        base_taxonomy.architecture,
        root_keys=('clip',),
        folder_paths=paths_clips,
    )


def get_compatible_vae_choices_for_model(base_model_name):
    base_taxonomy = resolve_model_taxonomy(base_model_name, root_keys=('checkpoints', 'unet'), folder_paths=paths_checkpoints)
    return _filter_model_choices_for_architecture(
        vae_filenames,
        base_taxonomy.architecture,
        root_keys=('vae',),
        folder_paths=path_vae,
    )


def _register_embedding_alias(lookup, alias, full_path):
    normalized = _normalize_model_selector(alias)
    if normalized and normalized not in lookup:
        lookup[normalized] = full_path


def rebuild_embedding_path_lookup():
    global embedding_path_lookup

    lookup = {}
    embedding_roots = get_asset_root_paths('embeddings')

    for relative_path in embedding_filenames:
        resolved_path = None

        if os.path.isabs(relative_path):
            candidate = os.path.abspath(os.path.realpath(relative_path))
            if os.path.isfile(candidate):
                resolved_path = candidate
        else:
            for root in embedding_roots:
                candidate = os.path.abspath(os.path.realpath(os.path.join(root, relative_path)))
                if os.path.isfile(candidate):
                    resolved_path = candidate
                    break

        if resolved_path is None:
            continue

        basename = os.path.basename(relative_path)
        relative_stem = os.path.splitext(relative_path)[0]
        basename_stem = os.path.splitext(basename)[0]

        for alias in (relative_path, basename, relative_stem, basename_stem):
            _register_embedding_alias(lookup, alias, resolved_path)

    embedding_path_lookup = lookup
    return


def resolve_embedding_path(name_or_path):
    if name_or_path is None:
        return None

    value = str(name_or_path).strip()
    if value == '':
        return None

    if os.path.isabs(value) and os.path.isfile(value):
        return os.path.abspath(os.path.realpath(value))

    candidates = _build_model_selector_candidates(value, get_asset_root_paths('embeddings'))

    for extra_candidate in (os.path.splitext(value)[0], os.path.splitext(os.path.basename(value))[0]):
        normalized = _normalize_model_selector(extra_candidate)
        if normalized and extra_candidate not in candidates:
            candidates.append(extra_candidate)

    for candidate in candidates:
        resolved = embedding_path_lookup.get(_normalize_model_selector(candidate))
        if resolved is not None and os.path.isfile(resolved):
            return resolved

    return None


def update_files():
    global model_filenames, clip_filenames, lora_filenames, embedding_filenames, vae_filenames, available_presets
    global default_base_model_name, default_model, default_model_taxonomy

    start = time.perf_counter()

    model_start = time.perf_counter()
    model_filenames = sorted(list(set(get_model_filenames(paths_checkpoint_bases))))
    model_filenames = filter_supported_sdxl_base_model_choices(model_filenames)
    active_default_base_model_name = coerce_active_base_model_selection(default_base_model_name, model_filenames)
    if active_default_base_model_name != default_base_model_name:
        print(
            f'[Startup] Selected_model [{default_base_model_name}] is no longer an active UI base model. '
            f'Using [{active_default_base_model_name}] instead.'
        )
    default_base_model_name = default_model = active_default_base_model_name
    default_model_taxonomy = resolve_model_taxonomy(default_base_model_name)
    model_elapsed = time.perf_counter() - model_start

    clip_start = time.perf_counter()
    clip_filenames = sorted(list(set(get_model_filenames(paths_clips))))
    clip_filenames = [x for x in clip_filenames if not x.replace('\\', '/').lower().startswith('flux/')]
    clip_elapsed = time.perf_counter() - clip_start

    lora_start = time.perf_counter()
    lora_filenames = sorted(list(set(get_model_filenames(paths_lora_discovery))))
    lora_elapsed = time.perf_counter() - lora_start

    embedding_start = time.perf_counter()
    embedding_filenames = sorted(list(set(get_model_filenames(path_embeddings, extensions=['.safetensors', '.pt', '.bin']))))
    rebuild_embedding_path_lookup()
    embedding_elapsed = time.perf_counter() - embedding_start

    vae_start = time.perf_counter()
    vae_filenames = sorted(list(set(get_model_filenames(path_vae))))
    vae_filenames = [x for x in vae_filenames if not x.replace('\\', '/').lower().startswith('flux/')]
    vae_elapsed = time.perf_counter() - vae_start

    preset_start = time.perf_counter()
    available_presets = get_presets()
    preset_elapsed = time.perf_counter() - preset_start

    print(
        '[Startup] update_files summary: '
        f'models={len(model_filenames)} ({model_elapsed:.2f}s), '
        f'clips={len(clip_filenames)} ({clip_elapsed:.2f}s), '
        f'loras={len(lora_filenames)} ({lora_elapsed:.2f}s), '
        f'embeddings={len(embedding_filenames)} ({embedding_elapsed:.2f}s), '
        f'vaes={len(vae_filenames)} ({vae_elapsed:.2f}s), '
        f'presets={len(available_presets)} ({preset_elapsed:.2f}s), '
        f'total={time.perf_counter() - start:.2f}s'
    )
    return

def downloading_inpaint_models(v):
    normalized = modules.flags.normalize_inpaint_engine_version(v, default=modules.flags.INPAINT_ENGINE_NONE)
    assert normalized in modules.flags.inpaint_engine_versions

    asset_ids = {
        modules.flags.INPAINT_ENGINE_V26: 'inpaint.fooocus_patch.v2_6',
    }
    asset_id = asset_ids.get(normalized)
    if asset_id is None:
        return None

    from modules import model_registry

    return model_registry.ensure_asset(asset_id)

def downloading_sdxl_lcm_lora():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/sdxl_lcm_lora.safetensors',
        model_dir=path_loras_lcm,
        file_name='sdxl_lcm_lora.safetensors'
    )
    return 'sdxl_lcm_lora.safetensors'


def downloading_sdxl_lightning_lora():
    load_file_from_url(
        url='https://huggingface.co/mashb1t/misc/resolve/main/sdxl_lightning_4step_lora.safetensors',
        model_dir=path_loras_lightning,
        file_name='sdxl_lightning_4step_lora.safetensors'
    )
    return 'sdxl_lightning_4step_lora.safetensors'


def downloading_sdxl_hyper_sd_lora():
    load_file_from_url(
        url='https://huggingface.co/mashb1t/misc/resolve/main/sdxl_hyper_sd_4step_lora.safetensors',
        model_dir=paths_loras[0],
        file_name=modules.flags.PerformanceLoRA.HYPER_SD.value
    )
    return modules.flags.PerformanceLoRA.HYPER_SD.value


def downloading_controlnet_canny():
    from modules import model_registry

    return model_registry.ensure_asset('structural.canny.controlnet')


def downloading_controlnet_cpds():
    from modules import model_registry

    return model_registry.ensure_asset('structural.cpds.controlnet')


def downloading_ip_adapters(v):
    assert v in ['ip', 'face']

    results = []

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/clip_vision_vit_h.safetensors',
        model_dir=path_clip_vision,
        file_name='clip_vision_vit_h.safetensors'
    )
    results += [os.path.join(path_clip_vision, 'clip_vision_vit_h.safetensors')]

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_ip_negative.safetensors',
        model_dir=path_controlnet[0],
        file_name='fooocus_ip_negative.safetensors'
    )
    results += [os.path.join(path_controlnet[0], 'fooocus_ip_negative.safetensors')]

    if v == 'ip':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus_sdxl_vit-h.bin',
            model_dir=path_controlnet[0],
            file_name='ip-adapter-plus_sdxl_vit-h.bin'
        )
        results += [os.path.join(path_controlnet[0], 'ip-adapter-plus_sdxl_vit-h.bin')]

    if v == 'face':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus-face_sdxl_vit-h.bin',
            model_dir=path_controlnet[0],
            file_name='ip-adapter-plus-face_sdxl_vit-h.bin'
        )
        results += [os.path.join(path_controlnet[0], 'ip-adapter-plus-face_sdxl_vit-h.bin')]

    return results
