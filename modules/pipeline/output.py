import os
import math
import cv2
import numpy as np
import fooocus_version
import modules.config as config
import modules.flags as flags
import modules.meta_parser as meta_parser
from modules.hash_cache import sha256_from_cache
from modules.util import HWC3, get_file_from_folder_list, resize_image


def yield_result(task_state, imgs, progressbar_index, do_not_show_finished_images=False):
    """
    Updates the task results and yields them for the UI.
    """
    if not isinstance(imgs, list):
        imgs = [imgs]

    task_state.results.extend(imgs)

    if do_not_show_finished_images:
        return

    task_state.yields.append(['results', task_state.results])


def build_image_wall(task_state):
    """
    Creates a grid (image wall) of all generated images in the current task.
    """
    results = []

    if len(task_state.results) < 2:
        return

    for img in task_state.results:
        if isinstance(img, str) and os.path.exists(img):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not isinstance(img, np.ndarray):
            return
        if img.ndim != 3:
            return
        results.append(img)

    H, W, C = results[0].shape

    for img in results:
        Hn, Wn, Cn = img.shape
        if H != Hn or W != Wn or C != Cn:
            return

    cols = int(math.ceil(float(len(results)) ** 0.5))
    rows = int(math.ceil(float(len(results)) / float(cols)))

    wall = np.zeros(shape=(H * rows, W * cols, C), dtype=np.uint8)

    for y in range(rows):
        for x in range(cols):
            idx = y * cols + x
            if idx < len(results):
                img = results[idx]
                wall[y * H:y * H + H, x * W:x * W + W, :] = img

    task_state.results.append(wall)


def resolve_workflow_identity(task_state, task_dict, explicit_workflow=None):
    if explicit_workflow:
        return explicit_workflow

    workflow_plan = getattr(task_state, 'workflow_plan', None)
    route_id = str(getattr(workflow_plan, 'route_id', '') or '').strip().lower()
    route_workflows = {
        'txt2img': 'txt2img',
        'inpaint': 'inpaint_sdxl',
        'outpaint': 'outpaint_sdxl',
        'flux_inpaint': 'flux_fill_inpaint',
        'flux_removal': 'flux_fill_remove',
        'upscale': 'upscale_gan',
        'super_upscale': 'super_upscale',
        'color_enhanced_upscale': 'color_enhance',
    }
    if route_id in route_workflows:
        return route_workflows[route_id]

    desc = str(task_dict.get('description', '') or '').strip()
    if desc == 'Color Enhancement':
        return 'color_enhance'
    if desc == 'Flux Fill Inpaint':
        return 'flux_fill_inpaint'
    if desc == 'Flux Fill Remove':
        return 'flux_fill_remove'

    uov_method = str(getattr(task_state, 'uov_method', 'Disabled') or '')
    if uov_method in {'Upscale', 'Fast (Super Resolution)'}:
        return 'upscale_gan'
    if uov_method in {'Super-Upscale', 'Vary (Subtle)', 'Vary (Strong)', 'Upscale (1.5x)', 'Upscale (2x)', 'Upscale (Fast 2x)'}:
        return 'super_upscale'
    if uov_method == 'Color Enhancement':
        return 'color_enhance'

    tab = getattr(task_state, 'current_tab', None)
    goals = getattr(task_state, 'goals', [])
    if tab == 'inpaint' or 'inpaint' in goals:
        return 'inpaint_sdxl'
    if tab == 'outpaint' or 'outpaint' in goals:
        return 'outpaint_sdxl'
    if tab == 'remove' or 'remove' in goals:
        if getattr(task_state, 'remove_obj_enabled', False):
            engine = str(getattr(task_state, 'objr_engine', '') or '').lower()
            return 'flux_fill_remove' if 'flux' in engine else 'remove_mat'
        if getattr(task_state, 'remove_bg_enabled', False):
            return 'bgr_subject'

    return 'txt2img'


_resolve_workflow_identity = resolve_workflow_identity


def _resolve_base_model_hash(base_model_name):
    if not base_model_name:
        return ''
    try:
        model_path = get_file_from_folder_list(base_model_name, config.paths_checkpoints)
        if not model_path or not os.path.isfile(model_path):
            return ''
        return sha256_from_cache(model_path)
    except Exception:
        return ''


def save_and_log(task_state, height, width, images, task_dict, use_expansion, loras, persist_image=True, workflow=None):
    """
    Saves the generated images to disk and logs the generation parameters using v2 schema.
    """
    from datetime import datetime, timezone
    from pathlib import Path
    from modules.private_logger import log

    workflow_id = resolve_workflow_identity(task_state, task_dict, workflow)
    now_iso = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    # Collect active ControlNet types
    cn_types = []
    if getattr(task_state, 'workflow_plan', None) is not None:
        cn_tasks_map = getattr(task_state, 'planned_cn_tasks', {}) or {}
    else:
        cn_tasks_map = getattr(task_state, 'cn_tasks', {}) or {}
    for cn_type, tasks in cn_tasks_map.items():
        if tasks:
            from modules.flags import resolve_cn_type
            norm_type = resolve_cn_type(cn_type, default=None)
            if norm_type and norm_type not in cn_types:
                cn_types.append(norm_type)

    # Build canonical v2 record
    v2_record = {
        'metadata_version': 2,
        'workflow': workflow_id,
        'timestamp': now_iso,
        'version': f'{meta_parser.METADATA_APP_NAME} {fooocus_version.version}'
    }

    if workflow_id in ['txt2img', 'inpaint_sdxl', 'outpaint_sdxl', 'super_upscale', 'color_enhance']:
        v2_record['base_model'] = Path(task_state.base_model_name).stem if task_state.base_model_name else ''
        v2_record['base_model_hash'] = _resolve_base_model_hash(task_state.base_model_name)
        v2_record['seed'] = int(task_dict.get('task_seed', task_state.seed))
        v2_record['scheduler'] = task_state.scheduler_name
        v2_record['steps'] = int(task_state.steps)
        v2_record['resolution'] = f"{width}x{height}"
        if task_state.vae_name and task_state.vae_name != 'None':
            v2_record['vae'] = Path(task_state.vae_name).stem
        if cn_types:
            v2_record['cn'] = [{'type': cn_type} for cn_type in cn_types]

        active_loras = [[Path(n).stem, float(w)] for n, w in loras if n != 'None']
        v2_record['loras'] = active_loras

        if workflow_id != 'color_enhance':
            v2_record['sampler'] = task_state.sampler_name
            v2_record['cfg_scale'] = float(task_state.cfg_scale)
            v2_record['styles'] = list(task_dict.get('styles', []) or [])
            v2_record['negative_prompt'] = str(task_dict.get('log_negative_prompt', '') or '')

        if workflow_id == 'txt2img' or workflow_id == 'super_upscale':
            v2_record['prompt'] = str(getattr(task_state, 'prompt', '') or task_dict.get('log_positive_prompt', '') or '')
        elif workflow_id == 'inpaint_sdxl':
            v2_record['prompt'] = str(getattr(task_state, 'prompt', '') or '')
            v2_record['inpaint_prompt'] = str(getattr(task_state, 'inpaint_additional_prompt', '') or '')
            v2_record['inpaint_route'] = str(getattr(task_state, 'inpaint_route', 'sdxl'))
            v2_record['inpaint_engine'] = flags.normalize_inpaint_engine_version(
                getattr(task_state, 'inpaint_engine', flags.INPAINT_ENGINE_NONE),
                default=flags.INPAINT_ENGINE_NONE,
            )
        elif workflow_id == 'outpaint_sdxl':
            v2_record['prompt'] = str(getattr(task_state, 'prompt', '') or '')
            v2_record['outpaint_prompt'] = str(getattr(task_state, 'outpaint_additional_prompt', '') or '')
            v2_record['outpaint_engine'] = flags.normalize_inpaint_engine_version(
                getattr(task_state, 'outpaint_engine', flags.INPAINT_ENGINE_NONE),
                default=flags.INPAINT_ENGINE_NONE,
            )
        elif workflow_id == 'color_enhance':
            v2_record['prompt_description'] = str(getattr(task_state, 'upscale_prompt', '') or '')

        # Hidden fields (stored in JSON record, omitted from preview and apply)
        v2_record['sharpness'] = float(task_state.sharpness)
        v2_record['clip_skip'] = int(task_state.clip_skip)
        v2_record['adm_guidance'] = (
            float(task_state.adm_scaler_positive),
            float(task_state.adm_scaler_negative),
            float(task_state.adm_scaler_end)
        )
        v2_record['adaptive_cfg'] = float(task_state.adaptive_cfg)

    elif workflow_id in ['flux_fill_inpaint', 'flux_fill_remove']:
        v2_record['seed'] = int(task_dict.get('task_seed', task_state.seed))
        v2_record['steps'] = int(task_state.steps)
        v2_record['sampler'] = task_state.sampler_name
        v2_record['scheduler'] = task_state.scheduler_name
        v2_record['resolution'] = f"{width}x{height}"
        if cn_types:
            v2_record['cn'] = [{'type': cn_type} for cn_type in cn_types]
        if getattr(task_state, 'flux_fill_t5_path', None):
            v2_record['t5'] = Path(task_state.flux_fill_t5_path).stem
        if getattr(task_state, 'flux_fill_clip_l_path', None):
            v2_record['clip_l'] = Path(task_state.flux_fill_clip_l_path).stem
        if getattr(task_state, 'flux_fill_ae_path', None):
            v2_record['ae'] = Path(task_state.flux_fill_ae_path).stem

        if workflow_id == 'flux_fill_inpaint':
            v2_record['prompt'] = str(getattr(task_state, 'prompt', '') or '')
            v2_record['inpaint_prompt'] = str(getattr(task_state, 'inpaint_additional_prompt', '') or '')
        else:
            v2_record['prompt_description'] = str(getattr(task_state, 'remove_prompt', '') or task_dict.get('log_positive_prompt', '') or '')

    elif workflow_id in ['upscale_gan', 'remove_mat', 'remove_objr', 'bgr_subject', 'bgr_mask']:
        v2_record['resolution'] = f"{width}x{height}"

    img_paths = []
    for x in images:
        d = [
            ('Workflow', 'workflow', workflow_id),
            ('Resolution', 'resolution', f"{width}x{height}"),
        ]

        if 'prompt' in v2_record:
            d.insert(0, ('Prompt', 'prompt', v2_record['prompt']))
        if 'inpaint_prompt' in v2_record:
            d.append(('Inpaint Prompt', 'inpaint_prompt', v2_record['inpaint_prompt']))
        if 'outpaint_prompt' in v2_record:
            d.append(('Outpaint Prompt', 'outpaint_prompt', v2_record['outpaint_prompt']))
        if 'prompt_description' in v2_record:
            d.insert(0, ('Description', 'prompt_description', v2_record['prompt_description']))
        if 'negative_prompt' in v2_record and v2_record['negative_prompt']:
            d.append(('Negative Prompt', 'negative_prompt', v2_record['negative_prompt']))
        if 'styles' in v2_record and v2_record['styles']:
            d.append(('Styles', 'styles', str(v2_record['styles'])))
        if 'steps' in v2_record:
            d.append(('Steps', 'steps', v2_record['steps']))
        if 'base_model' in v2_record and v2_record['base_model']:
            d.append(('Base Model', 'base_model', v2_record['base_model']))
        if 'sampler' in v2_record:
            d.append(('Sampler', 'sampler', v2_record['sampler']))
        if 'scheduler' in v2_record:
            d.append(('Scheduler', 'scheduler', v2_record['scheduler']))
        if 'cfg_scale' in v2_record:
            d.append(('Guidance Scale', 'guidance_scale', v2_record['cfg_scale']))
        if 'seed' in v2_record:
            d.append(('Seed', 'seed', str(v2_record['seed'])))
        if 'vae' in v2_record:
            d.append(('VAE', 'vae', v2_record['vae']))

        for li, (n, w) in enumerate(v2_record.get('loras', [])):
            d.append((f'LoRA {li + 1}', f'lora_combined_{li + 1}', f'{n} : {w}'))

        metadata_parser_instance = None
        if task_state.save_metadata_to_images:
            metadata_parser_instance = meta_parser.get_metadata_parser(meta_parser.MetadataScheme.FOOOCUS_NEX)
            metadata_parser_instance.set_v2_record(v2_record)

        d.append(('Metadata Scheme', 'metadata_scheme',
                  meta_parser.MetadataScheme.FOOOCUS_NEX.value if task_state.save_metadata_to_images else task_state.save_metadata_to_images))
        d.append(('Version', 'version', f'{meta_parser.METADATA_APP_NAME} {fooocus_version.version}'))

        img_paths.append(
            log(
                x,
                d,
                metadata_parser_instance,
                task_state.output_format,
                task_dict,
                persist_image,
                clipboard_metadata=v2_record,
            )
        )

    return img_paths
