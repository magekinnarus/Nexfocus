"""
Parameter Registry - Single source of truth for UI-to-backend parameter contract.

Each parameter is registered by name with its target TaskState field.
The webui.py ctrls_dict and AsyncTask init both use this registry,
eliminating positional alignment as a failure mode.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional, List
from modules.flux_fill_surface import (
    FLUX_FILL_BLEND_MORPHOLOGICAL,
    FLUX_FILL_INPAINT_ROUTE_SDXL,
    OBJR_ENGINE_MAT,
    normalize_flux_fill_blend_mode,
)
from modules.upscale_tile_policy import normalize_gan_tile_size

@dataclass
class ParamDef:
    """Definition of a single UI parameter."""
    name: str                             # Registry key (matches ctrls_dict key in webui.py)
    task_field: Optional[str]             # Target field on TaskState (None if ignored/deprecated)
    default: Any = None                   # Default if missing from args
    transform: Optional[Callable] = None  # Optional transform (e.g., int, float)


def _normalize_flux_fill_conditioning_value(value: Any) -> str:
    return 'empty'


def _normalize_flux_fill_prompt_cache_value(value: Any) -> str:
    normalized = str(value or 'temp').strip().lower().replace('-', '_').replace(' ', '_')
    if normalized in {'permanent', 'persist', 'persistent'}:
        return 'permanent'
    return 'temp'


def _normalize_objr_blend_mode_value(value: Any) -> str:
    return normalize_flux_fill_blend_mode(value)


def _normalize_flux_fill_runtime_posture_value(value: Any) -> str:
    normalized = str(value or 'auto').strip().lower().replace('-', '_').replace(' ', '_')
    if normalized == 'streaming':
        return 'streaming'
    return 'auto'


def _normalize_sdxl_assembly_posture_value(value: Any) -> str:
    normalized = str(value or 'auto').strip().lower().replace('-', '_').replace(' ', '_')
    if normalized == 'streaming':
        return 'streaming'
    return 'auto'


def _normalize_flux_fill_t5_posture_value(value: Any) -> str:
    normalized = str(value or 'disk_paged').strip().lower().replace('-', '_').replace(' ', '_')
    if normalized == 'cpu_resident':
        return 'cpu_resident'
    return 'disk_paged'


def _normalize_flux_fill_disk_paged_t5_gc_interval_value(value: Any) -> str:
    normalized = str(value or 'auto').strip().lower().replace('-', '_').replace(' ', '_')
    if normalized in {'4', '8', '16'}:
        return normalized
    return 'auto'

# Ordered list - order is for documentation, not for correctness.
# Special dynamic groups (LoRA, ControlNet) are handled explicitly in async_worker
# and are NOT listed as static ParamDefs here.
PARAM_REGISTRY: List[ParamDef] = [
    # --- Generation ---
    ParamDef('generate_image_grid', 'generate_image_grid', False, bool),
    ParamDef('prompt', 'prompt', '', str),
    ParamDef('negative_prompt', 'negative_prompt', '', str),
    ParamDef('style_selections', 'style_selections', [], list),
    ParamDef('aspect_ratios_selection', 'aspect_ratios_selection', '1024x1024', str),
    ParamDef('image_number', 'image_number', 1, int),
    ParamDef('output_format', 'output_format', 'png', str),
    ParamDef('image_seed', 'seed', -1, int),
    ParamDef('sharpness', 'sharpness', 2.0, float),
    ParamDef('guidance_scale', 'cfg_scale', 7.0, float),
    ParamDef('base_model', 'base_model_name', 'None', str),
    ParamDef('vae_model', 'vae_name', 'None', str),
    ParamDef('clip_model', 'clip_model_name', 'None', str),
    # LoRAs intentionally omitted here (15 parameters handled explicitly)

    # --- Mode & Inputs ---
    ParamDef('input_image_checkbox', 'input_image_checkbox', False, bool),
    ParamDef('current_tab', 'current_tab', 'uov', str),
    ParamDef('uov_method', 'uov_method', 'Disabled', str),
    ParamDef('uov_input_image', 'uov_input_image', None),
    ParamDef('upscale_model', 'upscale_model', 'None', str),
    ParamDef('upscale_scale_override', 'upscale_scale_override', 0, float),
    ParamDef('upscale_prompt', 'upscale_prompt', '', str),
    ParamDef('upscale_gan_output_image', 'upscale_gan_output_image', None),
    ParamDef('upscale_gan_tile_size', 'upscale_gan_tile_size', 256, normalize_gan_tile_size),
    ParamDef('upscale_refinement_tile_overlap', 'upscale_refinement_tile_overlap', 128, int),
    ParamDef('upscale_refinement_denoise', 'upscale_refinement_denoise', 0.3, float),

    # --- Remove (BGR/OBJR) ---
    ParamDef('remove_base_image', 'remove_base_image', None),
    ParamDef('remove_prompt', 'remove_prompt', '', str),
    ParamDef('remove_mask_image', 'remove_mask_image', None),
    ParamDef('remove_mask_data', 'remove_mask_data', '', str),
    ParamDef('remove_bg_enabled', 'remove_bg_enabled', False, bool),
    ParamDef('remove_obj_enabled', 'remove_obj_enabled', False, bool),
    ParamDef('objr_engine', 'objr_engine', OBJR_ENGINE_MAT, str),
    ParamDef('flux_fill_conditioning', 'flux_fill_conditioning', 'empty', _normalize_flux_fill_conditioning_value),
    ParamDef('flux_fill_prompt_cache', 'flux_fill_prompt_cache', 'temp', _normalize_flux_fill_prompt_cache_value),
    ParamDef('flux_fill_runtime_posture', 'flux_fill_runtime_posture', 'auto', _normalize_flux_fill_runtime_posture_value),
    ParamDef('flux_fill_t5_posture', 'flux_fill_t5_posture', 'disk_paged', _normalize_flux_fill_t5_posture_value),
    ParamDef('sdxl_assembly_posture', 'sdxl_assembly_posture', 'auto', _normalize_sdxl_assembly_posture_value),
    ParamDef(
        'flux_fill_disk_paged_t5_gc_interval',
        'flux_fill_disk_paged_t5_gc_interval',
        'auto',
        _normalize_flux_fill_disk_paged_t5_gc_interval_value,
    ),
    ParamDef('objr_mask_dilate', 'objr_mask_dilate', 16, int),
    ParamDef('objr_mask_blur', 'objr_mask_blur', 6, int),
    ParamDef('objr_blend_mode', 'objr_blend_mode', FLUX_FILL_BLEND_MORPHOLOGICAL, _normalize_objr_blend_mode_value),
    ParamDef('bgr_threshold', 'bgr_threshold', 0.5, float),
    ParamDef('bgr_jit', 'bgr_jit', True, bool),

    # --- Outpaint ---
    ParamDef('outpaint_selections', 'outpaint_selections', [], list),
    ParamDef('outpaint_input_image', 'outpaint_input_image', None),
    ParamDef('outpaint_mask_image', 'outpaint_mask_image', None),
    ParamDef('outpaint_additional_prompt', 'outpaint_additional_prompt', '', str),
    ParamDef('outpaint_bb_image', 'outpaint_bb_image', None),
    ParamDef('outpaint_bb_mask_data', 'outpaint_bb_mask_data', '', str),
    ParamDef('outpaint_step2_checkbox', 'outpaint_step2_checkbox', False, bool),
    ParamDef('outpaint_engine', 'outpaint_engine', 'None', str),
    ParamDef('outpaint_strength', 'outpaint_strength', 1.0, float),

    # --- Inpaint/Outpaint shared ---
    ParamDef('inpaint_outpaint_expansion_size', 'inpaint_outpaint_expansion_size', 384, int),

    # --- Inpaint ---
    ParamDef('inpaint_input_image', 'inpaint_input_image', None),
    ParamDef('inpaint_context_mask_image', 'inpaint_context_mask_image', None),
    ParamDef('inpaint_additional_prompt', 'inpaint_additional_prompt', '', str),
    ParamDef('inpaint_mask_image', 'inpaint_mask_image', None),
    ParamDef('inpaint_bb_image', 'inpaint_bb_image', None),
    ParamDef('inpaint_route', 'inpaint_route', FLUX_FILL_INPAINT_ROUTE_SDXL, str),
    ParamDef('inpaint_step2_checkbox', 'inpaint_step2_checkbox', False, bool),
    ParamDef('inpaint_engine', 'inpaint_engine', 'None', str),
    ParamDef('inpaint_strength', 'inpaint_strength', 0.5, float),
    ParamDef('inpaint_erode_or_dilate', 'inpaint_erode_or_dilate', 0, int),
    ParamDef('debugging_inpaint_preprocessor', 'debugging_inpaint_preprocessor', False, bool),
    ParamDef('inpaint_disable_initial_latent', 'inpaint_disable_initial_latent', False, bool),

    # --- Settings & Toggles ---
    ParamDef('disable_preview', 'disable_preview', False, bool),
    ParamDef('preview_update_interval', 'preview_update_interval', 1, int),
    ParamDef('disable_intermediate_results', 'disable_intermediate_results', False, bool),
    ParamDef('disable_seed_increment', 'disable_seed_increment', False, bool),
    ParamDef('prefetch_depth', 'prefetch_depth', 1, int),
    ParamDef('prefetch_chunk_mb', 'prefetch_chunk_mb', 64, int),
    # --- Advanced Sampling ---
    ParamDef('adm_scaler_positive', 'adm_scaler_positive', 1.5, float),
    ParamDef('adm_scaler_negative', 'adm_scaler_negative', 0.8, float),
    ParamDef('adm_scaler_end', 'adm_scaler_end', 0.3, float),
    ParamDef('adaptive_cfg', 'adaptive_cfg', 7.0, float),
    ParamDef('clip_skip', 'clip_skip', 2, int),
    ParamDef('sampler_name', 'sampler_name', 'dpmpp_2m_sde_gpu', str),
    ParamDef('scheduler_name', 'scheduler_name', 'karras', str),
    ParamDef('steps', 'steps', 30, int),
    ParamDef('overwrite_width', 'overwrite_width', -1, int),
    ParamDef('overwrite_height', 'overwrite_height', -1, int),
    ParamDef('overwrite_upscale_strength', 'overwrite_upscale_strength', -1.0, float),

    # --- Control / Image Prompts ---
    ParamDef('mixing_image_prompt_and_inpaint', 'mixing_image_prompt_and_inpaint', False, bool),
    ParamDef('mixing_image_prompt_and_outpaint', 'mixing_image_prompt_and_outpaint', False, bool),
    ParamDef('skipping_cn_preprocessor', 'skipping_cn_preprocessor', False, bool),
    ParamDef('canny_low_threshold', 'canny_low_threshold', 64, int),
    ParamDef('canny_high_threshold', 'canny_high_threshold', 128, int),
    ParamDef('controlnet_softness', 'controlnet_softness', 0.25, float),
    # ControlNet arrays injected dynamically

    # --- Metadata (Conditional) ---
    ParamDef('save_metadata_to_images', 'save_metadata_to_images', False, bool),
    ParamDef('metadata_scheme', 'metadata_scheme', 'fooocus_nex', str),
]

def validate_ctrls(ctrls_dict: dict):
    """
    Validate that all registered parameters are present in the controls dict,
    and warn if unexpected parameters are found.
    """
    import modules.config as config
    import args_manager

    registered_keys = {p.name for p in PARAM_REGISTRY}

    # Add dynamically generated keys that we know we expect
    for i in range(config.default_max_lora_number):
        registered_keys.update([f'lora_{i}_enabled', f'lora_{i}_model', f'lora_{i}_weight'])

    for i in range(config.default_controlnet_image_count):
        registered_keys.update([f'cn_{i}_image', f'cn_{i}_stop', f'cn_{i}_weight', f'cn_{i}_type'])

    # Exclude special internal keys
    provided_keys = set(ctrls_dict.keys()) - {'_currentTask'}

    if args_manager.args.disable_metadata:
        # If disabled, we expect them to be missing
        registered_keys -= {'save_metadata_to_images', 'metadata_scheme'}

    missing = registered_keys - provided_keys
    extra = provided_keys - registered_keys

    if missing:
        raise ValueError(f"[Parameter Registry] Missing required parameters in ctrls_dict: {missing}")

    if extra:
        print(f"[Parameter Registry] Warning: Unrecognized extra parameters in ctrls_dict: {extra}")
