import math
import random
from collections import OrderedDict
from types import SimpleNamespace
import backend.resources as resources
import modules.config as config
import modules.constants as constants
import modules.core as core
from backend import conditioning
import modules.util as util
from modules.sdxl_styles import apply_style, get_random_style, apply_arrays, random_style_name
from modules.util import safe_str, remove_empty_str, parse_lora_references_from_prompt

# Retained compatibility shim for legacy tests patching preprocessing.pipeline.
pipeline = SimpleNamespace(
    refresh_everything=None,
    clip_encode=None,
)


_PROMPT_TASK_CACHE: OrderedDict[str, dict] = OrderedDict()
_PROMPT_TASK_CACHE_LIMIT = 16





def _resolve_residency_class(task_state, residency_class=None):
    if residency_class is not None:
        return resources.normalize_sdxl_residency_class(residency_class)
    return resources.normalize_sdxl_residency_class(getattr(task_state, 'sdxl_residency_class', None))


def _clone_cond(conds):
    import torch
    results = []
    for c, p in conds:
        p = p["pooled_output"]
        if isinstance(c, torch.Tensor):
            c = c.clone()
        if isinstance(p, torch.Tensor):
            p = p.clone()
        results.append([c, {"pooled_output": p}])
    return results


def _clone_prompt_task(task):
    cloned = dict(task)
    if cloned.get('c') is not None:
        cloned['c'] = _clone_cond(cloned['c'])
    if cloned.get('uc') is not None:
        cloned['uc'] = _clone_cond(cloned['uc'])
    for key in ('positive', 'negative', 'styles'):
        if isinstance(cloned.get(key), list):
            cloned[key] = list(cloned[key])
    return cloned


def _clone_prompt_tasks(tasks):
    return [_clone_prompt_task(task) for task in tasks]


def _freeze_prompt_tasks(tasks):
    return tuple(
        tuple(sorted((key, value) for key, value in task.items() if key not in {'c', 'uc', 'task_seed'}))
        for task in tasks
    )


def _build_prompt_task_fingerprint(task_state, tasks, *, route_family=None, residency_class=None):
    residency = _resolve_residency_class(task_state, residency_class=residency_class)
    prompt_blueprint = _freeze_prompt_tasks(tasks)
    execution_policy = getattr(task_state, 'sdxl_execution_policy', None)
    return conditioning.build_stage_fingerprint(
        'sdxl_prompt_encode',
        residency_class=residency,
        model_identity=str(getattr(task_state, 'base_model_name', None) or ''),
        text_encoder_identity=(
            'unified_runtime_clip',
            -abs(int(getattr(task_state, 'clip_skip', 1) or 1)),
        ),
        clip_patch_uuid=tuple(getattr(task_state, 'loras_processed', ()) or getattr(task_state, 'loras', ()) or ()),
        clip_layer_idx=-abs(int(getattr(task_state, 'clip_skip', 1) or 1)),
        lora_artifacts_state=tuple(getattr(task_state, 'loras_processed', ()) or ()),
        route_family_reconciliation_signature=(
            route_family or getattr(task_state, 'current_tab', None),
            'unified',
        ),
        route_family=route_family or getattr(task_state, 'current_tab', None),
        execution_family=getattr(execution_policy, 'execution_family', None),
        clip_residency_mode='runtime_owned',
        prompt_blueprint=prompt_blueprint,
    )


def _remember_prompt_tasks(fingerprint, tasks):
    cache_key = fingerprint.digest()
    _PROMPT_TASK_CACHE[cache_key] = {
        'fingerprint': fingerprint,
        'tasks': _clone_prompt_tasks(tasks),
    }
    _PROMPT_TASK_CACHE.move_to_end(cache_key)
    while len(_PROMPT_TASK_CACHE) > _PROMPT_TASK_CACHE_LIMIT:
        _PROMPT_TASK_CACHE.popitem(last=False)


def _load_prompt_tasks_from_cache(fingerprint):
    cached = _PROMPT_TASK_CACHE.get(fingerprint.digest())
    if cached is None:
        return None
    _PROMPT_TASK_CACHE.move_to_end(fingerprint.digest())
    return _clone_prompt_tasks(cached['tasks'])


def apply_overrides(task_state):
    """
    Applies user-defined overrides for width and height.
    Steps are now controlled directly by task_state.steps.
    """
    steps = task_state.steps
    width = task_state.width
    height = task_state.height

    if task_state.overwrite_width > 0:
        width = task_state.overwrite_width
    if task_state.overwrite_height > 0:
        height = task_state.overwrite_height
    
    task_state.width = width
    task_state.height = height
    return steps, width, height

def patch_samplers(task_state):
    """
    Returns the scheduler name expected by the sampler layer.

    Scheduler-specific UNet patching now happens inside the unified runtime so
    the production path does not mutate any shared default-pipeline surfaces.
    """
    return task_state.scheduler_name
def process_prompt(task_state, base_model_additional_loras, progressbar_callback=None, *, route_context=None, route_family=None, residency_class=None):
    """
    Gathers prompts, styles, and LoRAs. Encodes prompts via CLIP.
    """
    prompt = task_state.prompt
    negative_prompt = task_state.negative_prompt
    image_number = task_state.image_number
    disable_seed_increment = task_state.disable_seed_increment
    use_expansion = task_state.use_expansion
    use_style = task_state.use_style

    prompts = remove_empty_str([safe_str(p) for p in prompt.splitlines()], default='')
    negative_prompts = remove_empty_str([safe_str(p) for p in negative_prompt.splitlines()], default='')
    prompt = prompts[0]
    negative_prompt = negative_prompts[0]
    
    # Masked-edit additional prompt handling
    edit_additional_prompt = ''
    if 'inpaint' in task_state.goals and task_state.inpaint_additional_prompt != '':
        edit_additional_prompt = task_state.inpaint_additional_prompt
    elif 'outpaint' in task_state.goals and getattr(task_state, 'outpaint_additional_prompt', '') != '':
        edit_additional_prompt = task_state.outpaint_additional_prompt

    if edit_additional_prompt != '':
        if prompt == '':
            prompt = edit_additional_prompt
        else:
            # Concatenate to the beginning so it's prioritized by CLIP
            prompt = edit_additional_prompt + '\n' + prompt
    
    if prompt == '':
        use_expansion = False
    
    extra_positive_prompts = prompts[1:] if len(prompts) > 1 else []
    extra_negative_prompts = negative_prompts[1:] if len(negative_prompts) > 1 else []

    if progressbar_callback:
        task_state.current_progress += 1
        progressbar_callback(task_state, task_state.current_progress, 'Loading models ...')

    loras, prompt = parse_lora_references_from_prompt(prompt, task_state.loras,
                                                       config.default_max_lora_number)
    task_state.loras_processed = loras

    if progressbar_callback:
        task_state.current_progress += 1
        progressbar_callback(task_state, task_state.current_progress, 'Processing prompts ...')

    tasks = []
    task_rng = random.Random(task_state.seed)

    for i in range(image_number):
        if disable_seed_increment:
            task_seed = task_state.seed % (constants.MAX_SEED + 1)
        else:
            task_seed = (task_state.seed + i) % (constants.MAX_SEED + 1)

        task_prompt = apply_arrays(prompt, i)
        task_negative_prompt = negative_prompt
        task_extra_positive_prompts = extra_positive_prompts
        task_extra_negative_prompts = extra_negative_prompts

        positive_basic_workloads = []
        negative_basic_workloads = []

        task_styles = task_state.style_selections.copy()
        if use_style:
            for j, s in enumerate(task_styles):
                if s == random_style_name:
                    s = get_random_style(task_rng)
                    task_styles[j] = s
                p, n, _ = apply_style(s, positive=task_prompt)
                positive_basic_workloads = positive_basic_workloads + p
                negative_basic_workloads = negative_basic_workloads + n

            positive_basic_workloads = [task_prompt] + positive_basic_workloads
            negative_basic_workloads = [task_negative_prompt] + negative_basic_workloads
        else:
            positive_basic_workloads.append(task_prompt)
            negative_basic_workloads.append(task_negative_prompt)

        positive_basic_workloads = positive_basic_workloads + task_extra_positive_prompts
        negative_basic_workloads = negative_basic_workloads + task_extra_negative_prompts

        positive_basic_workloads = remove_empty_str(positive_basic_workloads, default=task_prompt)
        negative_basic_workloads = remove_empty_str(negative_basic_workloads, default=task_negative_prompt)

        tasks.append(dict(
            task_seed=task_seed,
            task_prompt=task_prompt,
            task_negative_prompt=task_negative_prompt,
            positive=positive_basic_workloads,
            negative=negative_basic_workloads,
            expansion='',
            c=None,
            uc=None,
            positive_top_k=len(positive_basic_workloads),
            negative_top_k=len(negative_basic_workloads),
            log_positive_prompt='\n'.join([task_prompt] + task_extra_positive_prompts),
            log_negative_prompt='\n'.join([task_negative_prompt] + task_extra_negative_prompts),
            styles=task_styles
        ))

    prompt_fingerprint = _build_prompt_task_fingerprint(
        task_state,
        tasks,
        route_family=route_family or getattr(route_context, 'route_family', None),
        residency_class=residency_class,
    )
    cached_tasks = _load_prompt_tasks_from_cache(prompt_fingerprint)
    if cached_tasks is not None:
        for i, task in enumerate(cached_tasks):
            if disable_seed_increment:
                task['task_seed'] = task_state.seed % (constants.MAX_SEED + 1)
            else:
                task['task_seed'] = (task_state.seed + i) % (constants.MAX_SEED + 1)
        task_state.use_expansion = use_expansion
        task_state.positive_cond = None
        task_state.negative_cond = None
        if route_context is not None:
            route_context.set_route_artifact('prompt_encode', cached_tasks, fingerprint=prompt_fingerprint)
        return cached_tasks

    task_state.use_expansion = use_expansion
    task_state.positive_cond = None
    task_state.negative_cond = None
    _remember_prompt_tasks(prompt_fingerprint, tasks)
    if route_context is not None:
        route_context.set_route_artifact('prompt_encode', tasks, fingerprint=prompt_fingerprint)
        
    return tasks
