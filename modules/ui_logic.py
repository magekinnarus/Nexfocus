import gradio as gr
import os

gr.set_static_paths(paths=["javascript", "css", f"sdxl_styles{os.sep}samples"])
import random
import os
import json
import html
import time
import numpy as np
import shared
import modules.config
import fooocus_version
import modules.html
import modules.async_worker as worker
import modules.runtime_surface_state as runtime_surface_state
import modules.constants as constants
import modules.flags as flags
import modules.gradio_hijack as grh
import modules.style_sorter as style_sorter
import modules.meta_parser
from modules.flux_fill_surface import (
    OBJR_ENGINE_FLUX_FILL,
    OBJR_ENGINE_MAT,
    is_flux_fill_inpaint_route,
)
import modules.ui_components.metadata_ui as metadata_ui
from modules.ui_components.metadata_preview import format_metadata_preview
import modules.ui_components.settings_panel as settings_panel
import modules.ui_components.styles_panel as styles_panel
import modules.ui_components.models_panel as models_panel
import modules.ui_components.advanced_panel as advanced_panel
import modules.ui_components.control_panel as control_panel
import modules.ui_components.inpaint_panel as inpaint_panel
import modules.ui_components.outpaint_panel as outpaint_panel
import args_manager
import copy
from modules.setup_utils import download_preset_models
from modules.model_manager import default_model_manager

from modules.sdxl_styles import legal_style_names
from modules.private_logger import get_current_html_path
from modules.ui_gradio_extensions import javascript_html, css_html
from modules.auth import auth_enabled, check_auth
from modules.route_intent import normalize_current_tab
from modules.pipeline.workflow_legacy_adapter import capture_workflow_selection
from modules.util import get_enabled_loras, is_json
CompletedTaskRecord = runtime_surface_state.CompletedTaskRecord
completed_tasks_history = runtime_surface_state.completed_tasks_history
_last_rendered_completed_queue_html = None
_last_seen_active_task = None
_last_rendered_progress_state = None

def get_completed_queue_html():
    if not completed_tasks_history:
        return '<p class="empty-queue-msg">No completed tasks.</p>'

    completed_html = '<div class="nex-queue-list">'
    for task in reversed(completed_tasks_history):
        prompt_preview = str(task.prompt)[:40]
        if len(str(task.prompt)) > 40:
            prompt_preview += '...'
        if not prompt_preview.strip():
            prompt_preview = "Image generation"

        model_name = html.escape(str(task.model_name or ''))
        seed = html.escape(str(task.seed or ''))

        thumbs_html = '<div class="task-thumbnails" style="display: flex; gap: 8px; margin-bottom: 8px; flex-wrap: wrap;">'
        for img_path in task.images:
            file_url = html.escape(runtime_surface_state.build_file_url(str(img_path)))
            thumbs_html += f"""
            <div class="task-thumbnail-wrapper" style="width: 64px; height: 64px; flex-shrink: 0; border-radius: 8px; overflow: hidden; border: 1px solid rgba(255,255,255,0.1);">
                <a href="{file_url}" target="_blank" title="Click to view full image">
                    <img src="{file_url}" style="width: 100%; height: 100%; object-fit: cover;" />
                </a>
            </div>
            """
        thumbs_html += '</div>'

        import json
        escaped_images_json = html.escape(json.dumps(task.images))

        completed_html += f"""
        <div class="nex-queue-item completed-task">
            <div class="nex-queue-item-header completed-header">
                <div class="nex-queue-item-summary">
                    <span class="badge completed-badge">Completed</span>
                    <span class="task-id">ID: {task.task_id}</span>
                </div>
            </div>
            <div class="task-details">
                <p class="task-prompt">"{html.escape(prompt_preview)}"</p>
                <p class="task-meta">Model: {model_name} | Seed: {seed}</p>
                {thumbs_html}
                <div class="task-actions" style="display: flex; gap: 8px; margin-top: 8px;">
                    <button onclick='stageAllImages({escaped_images_json}, this)' class="queue-btn btn-stage" style="padding: 4px 10px; font-size: 0.75rem; border-radius: 6px; background: rgba(167, 139, 250, 0.12); color: #c084fc; border: 1px solid rgba(167, 139, 250, 0.25); cursor: pointer;">Stage</button>
                    <button onclick="triggerQueueAction('{task.task_id}', 'delete_completed')" class="queue-btn btn-delete" style="padding: 4px 10px; font-size: 0.75rem; border-radius: 6px; cursor: pointer;">Delete</button>
                </div>
            </div>
        </div>
        """
    completed_html += '</div>'
    return completed_html


def _record_completed_task(task, images):
    if not images:
        return False

    task_id = getattr(task, 'task_id', None)
    if task_id is None or any(record.task_id == task_id for record in completed_tasks_history):
        return False

    record = CompletedTaskRecord(
        task_id=task_id,
        prompt=getattr(getattr(task, 'state', None), 'prompt', ''),
        model_name=getattr(getattr(task, 'state', None), 'base_model_name', ''),
        seed=getattr(getattr(task, 'state', None), 'seed', ''),
        images=list(images),
    )
    completed_tasks_history.append(record)
    if len(completed_tasks_history) > 50:
        completed_tasks_history.pop(0)
    return True


def _drain_task_ui_events(task, gallery_items):
    latest_preview_img = None
    latest_progress_pct = None
    latest_progress_msg = None
    finished_images = None

    while len(task.yields) > 0:
        flag, product = task.yields.pop(0)
        if flag == 'preview':
            pct, msg, img = product
            latest_progress_pct = pct
            latest_progress_msg = msg
            if img is not None:
                latest_preview_img = img
        elif flag == 'results':
            _append_new_gallery_items(task, gallery_items, product)
        elif flag == 'finish':
            _append_new_gallery_items(task, gallery_items, product)
            finished_images = list(product) if isinstance(product, list) else [product]
            _record_completed_task(task, finished_images)

    return latest_preview_img, latest_progress_pct, latest_progress_msg, finished_images


def _get_progress_html_update(*, visible, number=0, text=''):
    global _last_rendered_progress_state
    import gradio as gr

    next_state = (bool(visible), int(number) if visible else None, str(text or '') if visible else '')
    if next_state == _last_rendered_progress_state:
        return gr.skip()

    _last_rendered_progress_state = next_state
    if not visible:
        return gr.update(visible=False)

    return gr.update(visible=True, value=modules.html.make_progress_html(next_state[1], next_state[2]))

def _has_uploaded_value(value):
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(_has_uploaded_value(value.get(key)) for key in ('image', 'mask', 'background'))
    return True


def _join_slot_labels(labels):
    labels = tuple(labels)
    if len(labels) < 2:
        return ''.join(labels)
    if len(labels) == 2:
        return f'{labels[0]} and {labels[1]}'
    return f"{', '.join(labels[:-1])}, and {labels[-1]}"


def _prepared_image_guidance(workflow_name, missing):
    missing_text = _join_slot_labels(missing)
    return (
        f'{workflow_name} is not ready. Missing: {missing_text}. '
        f'Complete {workflow_name} preparation and wait for the prepared image slots '
        'to finish loading, then Generate again.'
    )


def _uploaded_image_guidance(workflow_name, missing):
    missing_text = _join_slot_labels(missing)
    pronoun = 'it' if len(missing) == 1 else 'them'
    return (
        f'{workflow_name} is not ready. Upload {missing_text} and wait for {pronoun} '
        'to finish loading, then Generate again.'
    )


def validate_inpaint_generate_request(named_args, workflow_selection):
    if workflow_selection.source_surface != 'inpaint':
        return ''

    if not named_args.get('inpaint_step2_checkbox', False):
        return (
            'Prepare Inpaint first so the BB image and BB mask are ready, '
            'then Generate again.'
        )

    missing = [
        label
        for key, label in (
            ('inpaint_input_image', 'Base Image'),
            ('inpaint_bbox', 'BB Selection'),
            ('inpaint_bb_image', 'BB Image'),
            ('inpaint_mask_image', 'BB Mask'),
        )
        if not _has_uploaded_value(named_args.get(key))
    ]
    if missing:
        return _prepared_image_guidance('Inpaint', missing)
    return ''


def validate_outpaint_generate_request(named_args, workflow_selection=None):
    selection = workflow_selection or capture_workflow_selection(named_args, queue_capture=True)
    if selection.source_surface != 'outpaint':
        return ''

    if not named_args.get('outpaint_step2_checkbox', False):
        return (
            'Prepare Outpaint first so the expanded canvas, BB image, and BB mask are ready, '
            'then Generate again.'
        )

    missing = [
        label
        for key, label in (
            ('outpaint_input_image', 'Base Image'),
            ('outpaint_bb_image', 'BB Image'),
            ('outpaint_mask_image', 'BB Mask'),
        )
        if not _has_uploaded_value(named_args.get(key))
    ]
    if missing:
        return _prepared_image_guidance('Outpaint', missing)
    return ''


def validate_required_image_slots(named_args, workflow_selection):
    surface = workflow_selection.source_surface
    if surface in {'upscale', 'super_upscale', 'color_enhanced_upscale'}:
        workflow_name = {
            'upscale': 'Upscale',
            'super_upscale': 'Super-Upscale',
            'color_enhanced_upscale': 'Color Enhancement',
        }[surface]
        missing = []
        if not _has_uploaded_value(named_args.get('uov_input_image')):
            missing.append('the source Image')
        if (
            surface in {'super_upscale', 'color_enhanced_upscale'}
            and not _has_uploaded_value(named_args.get('upscale_gan_output_image'))
        ):
            missing.append('the Upscale Target')
        if missing:
            return _uploaded_image_guidance(workflow_name, missing)

    if surface == 'removal':
        missing = []
        removal_requested = workflow_selection.remove_background or workflow_selection.remove_object
        if removal_requested and not _has_uploaded_value(named_args.get('remove_base_image')):
            missing.append('the Base Image')
        if (
            workflow_selection.remove_object
            and not workflow_selection.remove_background
            and not _has_uploaded_value(named_args.get('remove_mask_image'))
        ):
            missing.append('the Mask')
        if missing:
            return _uploaded_image_guidance('Removal', missing)

    return ''


def _has_controlnet_slot_input(named_args):
    return any(
        _has_uploaded_value(named_args.get(f'cn_{index}_image'))
        for index in range(modules.config.default_controlnet_image_count)
    )


def validate_user_correctable_generate_request(named_args, workflow_selection=None):
    """Return guidance for requests the user can correct before queueing."""
    selection = workflow_selection or capture_workflow_selection(named_args, queue_capture=True)
    if (
        selection.source_surface == 'inpaint'
        and is_flux_fill_inpaint_route(selection.inpaint_route)
        and selection.allow_inpaint_controlnet
        and _has_controlnet_slot_input(named_args)
    ):
        return (
            "Flux Fill does not support ControlNet guidance. Select SDXL in Inpaint, "
            "or turn off 'Add ControlNet to Inpaint', then Generate again."
        )

    for validator in (
        validate_inpaint_generate_request,
        validate_outpaint_generate_request,
        validate_required_image_slots,
    ):
        message = validator(named_args, selection)
        if message:
            return message

    return ''


def _invalid_task(validation_message):
    task = worker.AsyncTask(args={})
    task.is_valid = False
    task.validation_message = validation_message
    return task


def get_task(*args):
    global ctrls_keys
    named_args = dict(zip(ctrls_keys, args))
    named_args.pop('_currentTask', None)
    named_args['current_tab'] = normalize_current_tab(named_args.get('current_tab'))
    workflow_selection = capture_workflow_selection(named_args, queue_capture=True)
    validation_message = validate_user_correctable_generate_request(named_args, workflow_selection)
    if validation_message:
        return _invalid_task(validation_message)
    named_args['workflow_selection'] = workflow_selection
    return worker.AsyncTask(args=named_args)

def generate_clicked(task: worker.AsyncTask, disable_preview):
    import backend.resources as resources

    with resources.interrupt_processing_mutex:
        resources.interrupt_processing = False
    # Legacy direct Gradio generation path retained for compatibility reference.
    # The queued runtime-surface UI no longer uses this function for live preview transport.

    if not task.is_valid:
        message = getattr(task, 'validation_message', 'The current request is not ready yet.')
        yield gr.update(visible=True, value=modules.html.make_progress_html(0, message)), \
            gr.update(), \
            gr.update(), \
            gr.update(), \
            gr.update()
        return
    try:
        batch_size = 1
    except Exception:
        batch_size = 1
    preview_enabled = not bool(disable_preview)

    execution_start_time = time.perf_counter()
    finished = False
    has_results = False

    if preview_enabled:
        initial_preview_col = gr.update(visible=True)
        initial_gallery_col = gr.update(visible=False)
    else:
        initial_preview_col = gr.update(visible=False)
        initial_gallery_col = gr.update(visible=True)

    yield gr.update(visible=True, value=modules.html.make_progress_html(1, 'Waiting for task to start ...')), \
        gr.update(visible=True, value=None), \
        gr.update(visible=True, columns=1), \
        initial_preview_col, \
        initial_gallery_col

    worker.async_tasks.append(task)

    while not finished:
        time.sleep(0.01)
        if len(task.yields) > 0:
            flag, product = task.yields.pop(0)
            if flag == 'preview':
                percentage, title, image = product
                # Prioritize image-bearing sampling previews so they do not sit
                # behind text-only progress frames on slower links (e.g. Colab
                # tunnels). We still collapse runs of text-only previews to
                # avoid starving the UI during long samplers.
                if image is None:
                    while len(task.yields) > 0 and task.yields[0][0] == 'preview':
                        next_percentage, next_title, next_image = task.yields.pop(0)[1]
                        percentage, title, image = next_percentage, next_title, next_image
                        if image is not None:
                            break
                else:
                    while len(task.yields) > 0 and task.yields[0][0] == 'preview':
                        next_percentage, next_title, next_image = task.yields[0][1]
                        if next_image is not None:
                            break
                        task.yields.pop(0)
                        percentage, title = next_percentage, next_title
                if preview_enabled:
                    if has_results and batch_size >= 2:
                        preview_col = gr.update(visible=True)
                        gallery_col = gr.update(visible=True)
                    else:
                        preview_col = gr.update(visible=True)
                        gallery_col = gr.update(visible=False)
                else:
                    preview_col = gr.update(visible=False)
                    gallery_col = gr.update(visible=True)

                yield gr.update(visible=True, value=modules.html.make_progress_html(percentage, title)), \
                    gr.update(visible=True, value=image) if image is not None else gr.update(visible=True), \
                    gr.update(visible=True, columns=1), \
                    preview_col, \
                    gallery_col
            if flag == 'results':
                has_results = True
                if preview_enabled and batch_size >= 2:
                    preview_col = gr.update(visible=True)
                    gallery_col = gr.update(visible=True)
                elif preview_enabled and batch_size == 1:
                    preview_col = gr.update(visible=True)
                    gallery_col = gr.update(visible=False)
                else:
                    preview_col = gr.update(visible=False)
                    gallery_col = gr.update(visible=True)

                yield gr.update(visible=True), \
                    gr.update(), \
                    gr.update(visible=True, value=product, columns=1), \
                    preview_col, \
                    gallery_col
            if flag == 'finish':
                cols = max(1, int(np.ceil(np.sqrt(len(product))))) if len(product) > 0 else 1

                yield gr.update(visible=False), \
                    gr.update(visible=True, value=None), \
                    gr.update(visible=True, value=product, columns=cols), \
                    gr.update(visible=False), \
                    gr.update(visible=True)
                finished = True

                # Auto-populate mask if BGR was run
                if task.state.current_tab == 'remove' and 'remove_bg' in task.state.goals and len(product) > 1:
                    # product[0] = character, product[1] = mask
                    task.yields.append(('bgr_mask_update', product[1]))

                # delete Fooocus temp images, only keep gradio temp images
                if args_manager.args.disable_image_log:
                    for filepath in product:
                        if isinstance(filepath, str) and os.path.exists(filepath):
                            os.remove(filepath)

    execution_time = time.perf_counter() - execution_start_time
    print(f'Total time: {execution_time:.2f} seconds')
    return




def inpaint_mode_change(mode, inpaint_engine_version):
    assert mode in modules.flags.inpaint_options

    # inpaint_disable_initial_latent, inpaint_engine,
    # inpaint_strength

    inpaint_engine_version = modules.flags.normalize_inpaint_engine_version(
        inpaint_engine_version,
        default=modules.config.default_inpaint_engine_version,
    )

    if mode == modules.flags.inpaint_option_detail:
        return [
            gr.update(visible=True), gr.update(visible=False, value=[]),
            gr.Dataset.update(visible=True, samples=modules.config.example_inpaint_prompts),
            False, 'None', 0.5
        ]

    if inpaint_engine_version == 'empty':
        inpaint_engine_version = modules.config.default_inpaint_engine_version

    if mode == modules.flags.inpaint_option_modify:
        return [
            gr.update(visible=True), gr.update(visible=False, value=[]),
            gr.Dataset.update(visible=False, samples=modules.config.example_inpaint_prompts),
            True, inpaint_engine_version, 1.0
        ]

    return [
        gr.update(visible=False, value=''), gr.update(visible=True),
        gr.Dataset.update(visible=False, samples=modules.config.example_inpaint_prompts),
        False, inpaint_engine_version, 0.5
    ]


def expand_mask(outpaint_selections, inpaint_mask_image):
    from modules.pipeline.inference import _is_debug_console_logging_enabled

    debug_mode = _is_debug_console_logging_enabled()
    if debug_mode:
        print(f"[Debug] Mask Expansion Requested. Direction: {outpaint_selections}")
    if inpaint_mask_image is None:
        if debug_mode:
            print("[Debug] Mask Image is None. Aborting.")
        return gr.update()

    from modules.mask_processing import combine_image_and_mask, to_binary_mask, expand_mask_direction, extract_mask_from_layers

    # Handle ImageEditor EditorValue
    if isinstance(inpaint_mask_image, dict) and 'background' in inpaint_mask_image:
        merged = combine_image_and_mask(inpaint_mask_image)
    else:
        merged = combine_image_and_mask(inpaint_mask_image)
    if merged is None:
        return gr.update()

    if debug_mode:
        print(f"[Debug Expand Mask] merged shape: {merged.shape}, max: {merged.max()}, min: {merged.min()}, mean: {merged.mean()}")

    new_mask = to_binary_mask(merged)
    if debug_mode:
        print(f"[Debug Expand Mask] binary_mask shape: {new_mask.shape}, sum (white pixels): {new_mask.sum() // 255} out of {new_mask.size}")

    for direction in outpaint_selections:
        new_mask = expand_mask_direction(new_mask, direction, pixels=32)

    from PIL import Image
    import modules.util
    import os

    result_rgb = np.stack([new_mask]*3, axis=-1)
    result_img = Image.fromarray(result_rgb)

    _, temp_path, _ = modules.util.generate_temp_filename(folder=modules.config.path_temp_outputs, extension='png')
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    result_img.save(temp_path)

    return temp_path

def uov_method_change(method):
    if method == 'Super-Upscale':
        # Super-Upscale now refines a provided pre-upscaled target directly.
        return gr.update(visible=True), gr.update(interactive=False, visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    if method in {'Color Enhancement', 'Color-Enhanced-Upscale'}:
        # Color enhancement consumes a required existing GAN donor; it never
        # admits an upscaler model. The main negative prompt remains shared.
        return gr.update(visible=False), gr.update(interactive=False, visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    return gr.update(visible=False), gr.update(interactive=True, visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)

def update_upscale_scale_info(image_path, model_name, scale_override):
    if image_path is None:
        return gr.update(value="<b>Scale:</b> No image uploaded.")

    if model_name == 'None' or model_name is None:
         return gr.update(value="<b>Scale:</b> No model selected.")

    if scale_override > 0:
        return gr.update(value=f"<b>Scale:</b> {scale_override}x (Overridden)")

    import modules.upscaler as upscaler
    try:
        native_scale = upscaler.get_model_scale_for_name(model_name)
        return gr.update(value=f"<b>Scale:</b> {native_scale}x (Model default)")
    except Exception as e:
        return gr.update(value=f"<b>Scale:</b> Error detection: {str(e)}")

def refresh_upscale_models():
    import modules.upscaler as upscaler
    models = upscaler.list_available_models()
    default_model = 'None'
    if '4xNomos2_otf_esrgan.pth' in models:
        default_model = '4xNomos2_otf_esrgan.pth'
    elif len(models) > 0:
        default_model = models[0]

    return gr.update(choices=['None'] + models, value=default_model)

def stop_clicked(currentTask):
    worker.request_interrupt('stop', currentTask)
    return currentTask

def skip_clicked(currentTask):
    worker.request_interrupt('skip', currentTask)
    return currentTask

def outpaint_selection_change(choices):
    if len(choices) <= 1:
        return choices
    return [choices[-1]]

def trigger_metadata_preview(file):
    parameters, metadata_scheme = modules.meta_parser.read_info_from_image(file)
    return format_metadata_preview(parameters, metadata_scheme)

def random_checked(r):
    return gr.update(visible=not r)

def refresh_seed(r, seed_string):
    if r:
        return random.randint(constants.MIN_SEED, constants.MAX_SEED)
    else:
        try:
            seed_value = int(seed_string)
            if constants.MIN_SEED <= seed_value <= constants.MAX_SEED:
                return seed_value
        except ValueError:
            pass
        return random.randint(constants.MIN_SEED, constants.MAX_SEED)

def update_history_link(output_format=None):
    if args_manager.args.disable_image_log:
        return gr.update(value='')

    return gr.update(value=f'<a href="file={get_current_html_path(output_format)}" target="_blank">\U0001F4DA History Log</a>')

def update_aspect_ratio_choices_for_model(base_model_name, current_aspect_ratio):
    labels = modules.config.get_aspect_ratio_labels_for_model(base_model_name)
    if not labels:
        labels = modules.config.available_aspect_ratios_labels

    if current_aspect_ratio in labels:
        value = current_aspect_ratio
    elif modules.config.default_aspect_ratio in labels:
        value = modules.config.default_aspect_ratio
    else:
        value = modules.config.get_default_aspect_ratio_label_for_model(base_model_name)
    return gr.update(choices=labels, value=value)


def get_filtered_lora_choices_for_model(base_model_name):
    try:
        choices = default_model_manager.list_installed_lora_dropdown_choices(base_model_name=base_model_name)
    except Exception as exc:
        print(f'Failed to build filtered LoRA choices for {base_model_name}: {exc}')
        choices = modules.config.lora_filenames
    return ['None'] + choices


def get_active_lora_choices_for_model(base_model_name, *current_lora_models):
    choices = list(get_filtered_lora_choices_for_model(base_model_name))
    try:
        installed_choices_with_presets = set(
            default_model_manager.list_installed_lora_dropdown_choices(
                base_model_name=base_model_name,
                include_preset_managed=True,
            )
        )
    except Exception as exc:
        print(f'Failed to build active LoRA choices for {base_model_name}: {exc}')
        installed_choices_with_presets = set()

    for current_lora_model in current_lora_models:
        if (
            isinstance(current_lora_model, str)
            and current_lora_model not in {'', 'None'}
            and current_lora_model in installed_choices_with_presets
            and current_lora_model not in choices
        ):
            choices.append(current_lora_model)
    return choices


def get_filtered_vae_choices_for_model(base_model_name):
    if _base_model_requires_default_vae(base_model_name):
        return [modules.flags.default_vae]
    return [modules.flags.default_vae] + modules.config.get_compatible_vae_choices_for_model(base_model_name)


def _get_base_model_dropdown_state(current_base_model=None):
    base_choices = list(modules.config.model_filenames or [])
    if not base_choices:
        return ['None'], 'None'

    current_value = modules.config.resolve_dropdown_choice(
        current_base_model,
        base_choices,
        folder_paths=modules.config.paths_checkpoints,
        root_keys=('checkpoints', 'unet'),
    )
    if current_value is not None:
        return base_choices, current_value

    default_value = modules.config.resolve_dropdown_choice(
        modules.config.default_base_model_name,
        base_choices,
        folder_paths=modules.config.paths_checkpoints,
        root_keys=('checkpoints', 'unet'),
    )
    if default_value is not None:
        return base_choices, default_value

    return base_choices, base_choices[0]


def _base_model_requires_default_vae(base_model_name):
    base_entry = default_model_manager.get_entry(base_model_name)
    return getattr(base_entry, 'root_key', None) == 'checkpoints'


def _resolve_vae_value_for_base_model(base_model_name, current_vae_model, vae_choices):
    if _base_model_requires_default_vae(base_model_name):
        return modules.flags.default_vae
    resolved = modules.config.resolve_dropdown_choice(
        current_vae_model,
        vae_choices,
        folder_paths=modules.config.path_vae,
        root_keys=('vae',),
    )
    return resolved or modules.flags.default_vae


def update_model_dependent_choices(base_model_name, current_aspect_ratio, current_vae_model, *current_lora_models):
    aspect_ratio_update = update_aspect_ratio_choices_for_model(base_model_name, current_aspect_ratio)
    vae_choices = get_filtered_vae_choices_for_model(base_model_name)
    vae_value = _resolve_vae_value_for_base_model(base_model_name, current_vae_model, vae_choices)
    lora_choices = get_active_lora_choices_for_model(base_model_name, *current_lora_models)
    lora_updates = []
    for current_lora_model in current_lora_models:
        value = current_lora_model if current_lora_model in lora_choices else 'None'
        lora_updates.append(gr.update(choices=lora_choices, value=value))
    return [aspect_ratio_update, gr.update(choices=vae_choices, value=vae_value)] + lora_updates


def refresh_files_clicked(current_base_model, current_aspect_ratio, current_vae_model, *current_lora_models):
    _refresh_model_file_indexes()

    base_model_choices, base_model_value = _get_base_model_dropdown_state(current_base_model)

    aspect_ratio_update, vae_update, *lora_model_updates = update_model_dependent_choices(
        base_model_value,
        current_aspect_ratio,
        current_vae_model,
        *current_lora_models,
    )

    results = [gr.update(choices=base_model_choices, value=base_model_value)]
    results += [aspect_ratio_update]
    results += [vae_update]
    if not args_manager.args.disable_preset_selection:
        results += [gr.update(choices=modules.config.available_presets)]
    for lora_model_update in lora_model_updates:
        results += [gr.update(interactive=True), lora_model_update, gr.update()]
    return results


def _refresh_model_file_indexes():
    try:
        default_model_manager.refresh_catalog_index(force_refresh=True)
        default_model_manager.refresh_installed_index()
    except Exception as exc:
        print(f'Failed to refresh model index: {exc}')
    modules.config.update_files()

def _resolve_dropdown_choice(candidate_value, available_choices):
    if candidate_value is None:
        return None

    return modules.config.resolve_dropdown_choice(candidate_value, available_choices)


def _get_asset_root_paths(root_keys):
    folder_paths = []
    for root_key in list(root_keys or []):
        try:
            for path in modules.config.get_asset_root_paths(root_key):
                if path not in folder_paths:
                    folder_paths.append(path)
        except KeyError:
            continue
    return folder_paths



def _get_installed_dropdown_value(selector, expected_root_keys, available_choices=None):
    entry = default_model_manager.get_entry(selector)
    if entry is None or entry.root_key not in set(expected_root_keys):
        return None

    inventory_record = default_model_manager.inventory_record(entry)
    if not inventory_record.installed:
        return None

    candidate_value = inventory_record.installed_relative_path or entry.relative_path or entry.name
    if available_choices is None:
        return candidate_value
    for value in (
        candidate_value,
        entry.relative_path,
        entry.name,
        getattr(entry, 'alias', None),
        getattr(entry, 'id', None),
    ):
        resolved = modules.config.resolve_dropdown_choice(
            value,
            available_choices,
            folder_paths=_get_asset_root_paths(expected_root_keys),
            root_keys=tuple(expected_root_keys),
        )
        if resolved is not None:
            return resolved
    return None


def _selector_matches_base_architecture(selector, base_model_name):
    candidate_entry = default_model_manager.get_entry(selector)
    if candidate_entry is None:
        return False

    base_entry = default_model_manager.get_entry(base_model_name)
    candidate_architecture = getattr(candidate_entry, 'architecture', None)
    base_architecture = getattr(base_entry, 'architecture', None) if base_entry is not None else None
    if candidate_architecture and base_architecture and candidate_architecture != base_architecture:
        return False
    return True


def apply_model_browser_drop(apply_data_json, current_base_model, current_vae_model, *current_lora_ctrl_values):
    base_choices, base_value = _get_base_model_dropdown_state(current_base_model)
    vae_choices = get_filtered_vae_choices_for_model(base_value)
    lora_slot_count = modules.config.default_max_lora_number

    current_lora_enabled = []
    current_lora_models = []
    current_lora_weights = []
    for index in range(lora_slot_count):
        offset = index * 3
        current_lora_enabled.append(current_lora_ctrl_values[offset] if offset < len(current_lora_ctrl_values) else False)
        current_lora_models.append(current_lora_ctrl_values[offset + 1] if offset + 1 < len(current_lora_ctrl_values) else 'None')
        current_lora_weights.append(current_lora_ctrl_values[offset + 2] if offset + 2 < len(current_lora_ctrl_values) else 1.0)

    vae_value = _resolve_vae_value_for_base_model(base_value, current_vae_model, vae_choices)
    lora_choices = get_active_lora_choices_for_model(base_value, *current_lora_models)

    drop_selector = ''
    drop_target = ''
    current_aspect_ratio = ''
    if apply_data_json:
        try:
            parsed = json.loads(apply_data_json)
            if isinstance(parsed, dict):
                drop_selector = str(parsed.get('selector', '') or '')
                drop_target = str(parsed.get('target', '') or '')
                current_aspect_ratio = str(parsed.get('aspect_ratio', '') or '')
        except (json.JSONDecodeError, TypeError, ValueError):
            drop_selector = ''
            drop_target = ''
            current_aspect_ratio = ''

    if not current_aspect_ratio:
        current_aspect_ratio = modules.config.default_aspect_ratio
        if current_aspect_ratio not in modules.config.get_aspect_ratio_labels_for_model(base_value):
            current_aspect_ratio = modules.config.get_default_aspect_ratio_label_for_model(base_value)

    if drop_selector and drop_target:
        if drop_target == 'base_model':
            candidate = _get_installed_dropdown_value(drop_selector, {'checkpoints', 'unet'}, base_choices)
            if candidate is None or candidate not in base_choices:
                _refresh_model_file_indexes()
                base_choices, base_value = _get_base_model_dropdown_state(current_base_model)
                vae_choices = get_filtered_vae_choices_for_model(base_value)
                candidate = _get_installed_dropdown_value(drop_selector, {'checkpoints', 'unet'}, base_choices)
            if candidate and candidate in base_choices:
                base_value = candidate
                aspect_ratio_update, vae_update, *lora_model_updates = update_model_dependent_choices(
                    base_value,
                    current_aspect_ratio,
                    current_vae_model,
                    *current_lora_models,
                )
                vae_value = vae_update['value']
                lora_choices = get_active_lora_choices_for_model(base_value, *current_lora_models)
            else:
                aspect_ratio_update = gr.update(choices=modules.config.get_aspect_ratio_labels_for_model(base_value) or modules.config.available_aspect_ratios_labels, value=current_aspect_ratio)
                lora_model_updates = [gr.update(choices=lora_choices, value=(model if model in lora_choices else 'None')) for model in current_lora_models]
        elif drop_target == 'vae_model':
            if not _base_model_requires_default_vae(base_value):
                candidate = _get_installed_dropdown_value(drop_selector, {'vae'}, vae_choices)
                if candidate and candidate in vae_choices and _selector_matches_base_architecture(drop_selector, base_value):
                    vae_value = candidate
            else:
                vae_value = modules.flags.default_vae
            aspect_ratio_update = gr.update(choices=modules.config.get_aspect_ratio_labels_for_model(base_value) or modules.config.available_aspect_ratios_labels, value=current_aspect_ratio)
            lora_model_updates = [gr.update(choices=lora_choices, value=(model if model in lora_choices else 'None')) for model in current_lora_models]
        elif str(drop_target).startswith('lora_model:'):
            try:
                target_index = max(0, int(str(drop_target).split(':', 1)[1]) - 1)
            except Exception:
                target_index = None
            candidate = _get_installed_dropdown_value(drop_selector, {'loras'}, lora_choices)
            if candidate and candidate in lora_choices and target_index is not None and target_index < lora_slot_count:
                current_lora_enabled[target_index] = True
                current_lora_models[target_index] = candidate
            aspect_ratio_update = gr.update(choices=modules.config.get_aspect_ratio_labels_for_model(base_value) or modules.config.available_aspect_ratios_labels, value=current_aspect_ratio)
            lora_model_updates = [gr.update(choices=lora_choices, value=(model if model in lora_choices else 'None')) for model in current_lora_models]
        else:
            aspect_ratio_update = gr.update(choices=modules.config.get_aspect_ratio_labels_for_model(base_value) or modules.config.available_aspect_ratios_labels, value=current_aspect_ratio)
            lora_model_updates = [gr.update(choices=lora_choices, value=(model if model in lora_choices else 'None')) for model in current_lora_models]
    else:
        aspect_ratio_update = gr.update(choices=modules.config.get_aspect_ratio_labels_for_model(base_value) or modules.config.available_aspect_ratios_labels, value=current_aspect_ratio)
        lora_model_updates = [gr.update(choices=lora_choices, value=(model if model in lora_choices else 'None')) for model in current_lora_models]

    results = [gr.update(choices=base_choices, value=base_value)]
    results += [aspect_ratio_update]
    results += [gr.update(choices=vae_choices, value=vae_value)]
    for index in range(lora_slot_count):
        results += [
            gr.update(value=bool(current_lora_enabled[index])),
            lora_model_updates[index],
            gr.update(value=current_lora_weights[index]),
        ]
    return results


def update_style_label(selections):
    if not selections or len(selections) == 0:
        return gr.update(label='Prompt Presets')

    visible_styles = selections[:2]
    label = f"Presets: {', '.join(visible_styles)}"
    if len(selections) > 2:
        label += f" ... (+{len(selections) - 2} more)"

    return gr.update(label=label)

def _parse_lora_metadata(raw_value):
    parts = str(raw_value).split(' : ')
    enabled = True
    name = 'None'
    weight = 1.0

    if len(parts) == 3:
        enabled = parts[0] == 'True'
        name = parts[1]
        weight = float(parts[2])
    elif len(parts) == 2:
        name = parts[0]
        weight = float(parts[1])
    elif len(parts) == 1 and parts[0]:
        name = parts[0]

    return [enabled, name, weight]


def _format_lora_metadata(lora_state):
    enabled, name, weight = lora_state
    return f'{enabled} : {name} : {weight}'


def _merge_preset_lora_state(preset_prepared, lora_args):
    if len(lora_args) < 3:
        return

    slot_count = min(len(lora_args) // 3, modules.config.default_max_lora_number)
    current_loras = []
    for slot_index in range(slot_count):
        offset = slot_index * 3
        current_loras.append([
            bool(lora_args[offset]),
            str(lora_args[offset + 1]),
            float(lora_args[offset + 2]),
        ])

    merged_loras = [slot.copy() for slot in current_loras]
    preset_loras = {}
    for slot_index in range(slot_count):
        key = f'lora_combined_{slot_index + 1}'
        if key in preset_prepared:
            preset_loras[slot_index] = _parse_lora_metadata(preset_prepared[key])

    if 0 in preset_loras:
        slot1_lora = merged_loras[0]
        preset_slot1_lora = preset_loras[0]
        if (
            slot1_lora[1] not in {'', 'None'}
            and slot1_lora != preset_slot1_lora
        ):
            free_slot_index = next(
                (
                    index for index in range(1, slot_count)
                    if merged_loras[index][1] == 'None' and index not in preset_loras
                ),
                None,
            )
            if free_slot_index is not None:
                merged_loras[free_slot_index] = slot1_lora.copy()
                print(
                    f"[Preset] Shifted custom LoRA '{slot1_lora[1]}' "
                    f"from slot 1 to slot {free_slot_index + 1}"
                )
            else:
                print(
                    f"[Preset] Slot 1 custom LoRA '{slot1_lora[1]}' overwritten "
                    f"(no free slots 2-{slot_count})"
                )

    for slot_index, lora_state in preset_loras.items():
        merged_loras[slot_index] = lora_state.copy()

    for slot_index, lora_state in enumerate(merged_loras):
        preset_prepared[f'lora_combined_{slot_index + 1}'] = _format_lora_metadata(lora_state)


def preset_selection_change(preset, is_generating, *args):
    preset_content = modules.config.try_get_preset_content(preset) if preset != 'initial' else {}
    preset_prepared = metadata_ui.parse_meta_from_preset(preset_content)
    if preset != 'initial':
        _merge_preset_lora_state(preset_prepared, args)

    default_model = preset_prepared.get('base_model')
    previous_default_models = preset_prepared.get('previous_default_models', [])
    checkpoint_downloads = preset_prepared.get('checkpoint_downloads', {})
    embeddings_downloads = preset_prepared.get('embeddings_downloads', {})
    lora_downloads = preset_prepared.get('lora_downloads', {})
    vae_downloads = preset_prepared.get('vae_downloads', {})
    upscale_downloads = preset_prepared.get('upscale_downloads', {})

    downloaded_base_model, downloaded_checkpoint_downloads, downloaded_new_assets = download_preset_models(
        default_model, checkpoint_downloads, embeddings_downloads, lora_downloads,
        vae_downloads, upscale_downloads)

    if downloaded_new_assets:
        _refresh_model_file_indexes()

    if downloaded_base_model is not None:
        preset_prepared['base_model'] = downloaded_base_model
    else:
        preset_prepared.pop('base_model', None)

    if downloaded_checkpoint_downloads:
        preset_prepared['checkpoint_downloads'] = downloaded_checkpoint_downloads
    else:
        preset_prepared.pop('checkpoint_downloads', None)

    if 'prompt' in preset_prepared and preset_prepared.get('prompt') == '':
        del preset_prepared['prompt']

    # Presets are already normalized current-schema control dictionaries, not
    # legacy image metadata. Mark them explicitly so the metadata loader does
    # not run the v1 compatibility conversion and synthesize absent fields.
    preset_prepared['metadata_version'] = 2
    preset_prepared['workflow'] = 'txt2img'
    return metadata_ui.load_parameter_button_click(json.dumps(preset_prepared), is_generating)

def inpaint_engine_state_change(inpaint_engine_version):
    inpaint_engine_version = modules.flags.normalize_inpaint_engine_version(
        inpaint_engine_version,
        default=modules.config.default_inpaint_engine_version,
    )
    return gr.update(value=inpaint_engine_version)

def outpaint_engine_state_change(outpaint_engine_version):
    outpaint_engine_version = modules.flags.normalize_inpaint_engine_version(
        outpaint_engine_version,
        default=modules.config.default_outpaint_engine_version,
    )
    return gr.update(value=outpaint_engine_version)

def objr_engine_change(objr_engine_value):
    if str(objr_engine_value or '').strip() in {OBJR_ENGINE_MAT, OBJR_ENGINE_FLUX_FILL}:
        return gr.update(value=16)
    return gr.update(value=16)

def parse_meta(raw_prompt_txt, is_generating):
    loaded_json = None
    if is_json(raw_prompt_txt):
        loaded_json = json.loads(raw_prompt_txt)

    if loaded_json is None:
        if is_generating:
            return gr.update(), gr.update(), gr.update()
        else:
            return gr.update(), gr.update(visible=True), gr.update(visible=False)

    return json.dumps(loaded_json), gr.update(visible=False), gr.update(visible=True)

def trigger_metadata_import(file, state_is_generating):
    parameters, metadata_scheme = modules.meta_parser.read_info_from_image(file)
    if parameters is None:
        print('Could not find metadata in the image!')
        parsed_parameters = {}
    else:
        metadata_parser = modules.meta_parser.get_metadata_parser(metadata_scheme)
        parsed_parameters = metadata_parser.to_json(parameters)

    return metadata_ui.load_parameter_button_click(parsed_parameters, state_is_generating)





# MVC-Light Controller: ui_logic.py

def register_all_events(ctrls_dict, currentTask_component, ui_elements):
    # Unpack components for easy reference
    for name, component in ctrls_dict.items():
        globals()[name] = component

    for name, component in ui_elements.items():
        globals()[name] = component

    global currentTask
    currentTask = currentTask_component

    global ctrls_keys, ctrls
    ctrls_keys = ['_currentTask'] + list(ctrls_dict.keys())
    ctrls = [currentTask_component] + list(ctrls_dict.values())

    # Global/Shared states and components that are needed by logic
    # (These are passed in via ctrls_dict or accessible via shared)

    # Phase 3 UI Bindings
    global toggle_toolbar_js, switch_js, down_js
    toggle_toolbar_js = """
    () => {
        const wrap = document.querySelector('#inpaint_canvas');
        if(wrap){
            wrap.classList.toggle('hide-toolbar');
            if(!document.getElementById('inpaint-toolbar-style')){
                const style = document.createElement('style');
                style.id = 'inpaint-toolbar-style';
                style.innerHTML = `
                    #inpaint_canvas.hide-toolbar button[aria-label="Undo"],
                    #inpaint_canvas.hide-toolbar button[aria-label="Clear"],
                    #inpaint_canvas.hide-toolbar button[aria-label="Remove Image"],
                    #inpaint_canvas.hide-toolbar button[aria-label="Draw"],
                    #inpaint_canvas.hide-toolbar button[aria-label="Erase"],
                    #inpaint_canvas.hide-toolbar .canvas-tooltip-info,
                    #inpaint_canvas.hide-toolbar .toolbar,
                    #inpaint_canvas.hide-toolbar input[type="range"] {
                        display: none !important;
                        opacity: 0 !important;
                        visibility: hidden !important;
                    }
                `;
                document.head.appendChild(style);
            }
        }
    }
    """

    switch_js = "(x) => {if(x){if(window.viewer_to_bottom){viewer_to_bottom(100);viewer_to_bottom(500);}}else{if(window.viewer_to_top){viewer_to_top();}} return x;}"
    down_js = "() => {if(window.viewer_to_bottom){viewer_to_bottom();}}"
    resolve_generate_tab_js = """
    (currentTab) => {
        const resolveVisibleTab = () => {
            const candidates = [
                ['remove', 'remove_tab'],
                ['inpaint', 'inpaint_tab'],
                ['outpaint', 'outpaint_tab'],
                ['ip', 'ip_tab'],
                ['metadata', 'metadata_tab'],
                ['uov', 'uov_tab'],
            ];

            const isVisible = (element) => {
                if (!element) {
                    return false;
                }
                const style = window.getComputedStyle(element);
                if (style.display === 'none' || style.visibility === 'hidden') {
                    return false;
                }
                if (element.hasAttribute('hidden') || element.getAttribute('aria-hidden') === 'true') {
                    return false;
                }
                return element.offsetParent !== null || style.position === 'fixed';
            };

            for (const [routeTab, panelId] of candidates) {
                const panel = document.getElementById(panelId);
                if (isVisible(panel)) {
                    return routeTab;
                }
                const tabButton = document.getElementById(`${panelId}-button`);
                if (tabButton && tabButton.getAttribute('aria-selected') === 'true') {
                    return routeTab;
                }
            }
            return currentTab;
        };

        ['inpaint_additional_prompt', 'outpaint_additional_prompt'].forEach(id => {
            const el = document.querySelector(`#${id} textarea`);
            if (el) {
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
            }
        });

        return resolveVisibleTab() || currentTab;
    }
    """

    # Bindings start here
    inpaint_toggle_toolbar.click(lambda: None, queue=False, show_progress=False, js=toggle_toolbar_js)

    input_image_checkbox.change(lambda x: gr.update(visible=x), inputs=input_image_checkbox,
                                outputs=image_input_panel, queue=False, show_progress=False, js=switch_js)

    outpaint_selections.change(outpaint_selection_change, inputs=outpaint_selections, outputs=outpaint_selections, queue=False, show_progress=False)

    uov_tab.select(lambda: 'uov', outputs=current_tab, queue=False, js=down_js, show_progress=False)
    inpaint_tab.select(lambda: 'inpaint', outputs=current_tab, queue=False, js=down_js, show_progress=False)
    outpaint_tab.select(lambda: 'outpaint', outputs=current_tab, queue=False, js=down_js, show_progress=False)
    remove_tab.select(lambda: 'remove', outputs=current_tab, queue=False, js=down_js, show_progress=False)
    ip_tab.select(lambda: 'ip', outputs=current_tab, queue=False, js=down_js, show_progress=False)
    metadata_tab.select(lambda: 'metadata', outputs=current_tab, queue=False, js=down_js, show_progress=False)

    uov_method.change(uov_method_change, inputs=uov_method, outputs=[upscale_refinement_container, upscale_model, upscale_scale_override, upscale_prompt, upscale_gan_output_container, upscale_scale_info, upscale_gan_tile_size], queue=False, show_progress=False)

    uov_input_image.change(update_upscale_scale_info, inputs=[uov_input_image, upscale_model, upscale_scale_override], outputs=upscale_scale_info, queue=False, show_progress=False)
    upscale_model.change(update_upscale_scale_info, inputs=[uov_input_image, upscale_model, upscale_scale_override], outputs=upscale_scale_info, queue=False, show_progress=False)
    upscale_scale_override.change(update_upscale_scale_info, inputs=[uov_input_image, upscale_model, upscale_scale_override], outputs=upscale_scale_info, queue=False, show_progress=False)

    shared.gradio_root.load(refresh_upscale_models, outputs=upscale_model, queue=False, show_progress=False)

    lora_model_ctrls = [lora_ctrls[i * 3 + 1] for i in range(modules.config.default_max_lora_number)]
    model_choice_inputs = [base_model, aspect_ratios_selection, vae_model] + lora_model_ctrls
    model_choice_outputs = [aspect_ratios_selection, vae_model] + lora_model_ctrls

    base_model.change(update_model_dependent_choices, inputs=model_choice_inputs, outputs=model_choice_outputs, queue=False, show_progress=False)
    shared.gradio_root.load(update_model_dependent_choices, inputs=model_choice_inputs, outputs=model_choice_outputs, queue=False, show_progress=False)
    aspect_ratios_selection.change(lambda x: None, inputs=aspect_ratios_selection, queue=False, show_progress=False, js='(x)=>{refresh_aspect_ratios_label(x);}')
    shared.gradio_root.load(lambda x: None, inputs=aspect_ratios_selection, queue=False, show_progress=False, js='(x)=>{refresh_aspect_ratios_label(x);}')

    seed_random.change(random_checked, inputs=[seed_random], outputs=[image_seed],
                       queue=False, show_progress=False)

    shared.gradio_root.load(update_history_link, outputs=history_link, queue=False, show_progress=False)

    style_selections.change(update_style_label, inputs=style_selections, outputs=style_selections_accordion, queue=False, show_progress=False)

    shared.gradio_root.load(
        lambda: gr.update(
            choices=copy.deepcopy(style_sorter.all_styles),
            value=[x for x in modules.config.default_styles if x in style_sorter.all_styles]
        ),
        outputs=style_selections,
        queue=False,
        show_progress=False
    ).then(update_style_label, inputs=style_selections, outputs=style_selections_accordion, queue=False, show_progress=False).then(lambda: None, js='()=>{refresh_style_localization();}', queue=False, show_progress=False)

    style_search_bar.change(style_sorter.search_styles,
                            inputs=[style_selections, style_search_bar],
                            outputs=style_selections,
                            queue=False,
                            show_progress=False).then(
        lambda: None, js='()=>{refresh_style_localization();}')

    refresh_files_output = [base_model, aspect_ratios_selection, vae_model]
    if not args_manager.args.disable_preset_selection:
        refresh_files_output += [preset_selection]
    refresh_files.click(refresh_files_clicked, [base_model, aspect_ratios_selection, vae_model] + lora_model_ctrls, refresh_files_output + lora_ctrls,
                        queue=False, show_progress=False)
    shared.gradio_root.load(
        refresh_files_clicked,
        inputs=[base_model, aspect_ratios_selection, vae_model] + lora_model_ctrls,
        outputs=refresh_files_output + lora_ctrls,
        queue=False,
        show_progress=False,
    )

    model_browser_drop_outputs = [base_model, aspect_ratios_selection, vae_model] + lora_ctrls
    model_browser_apply_data.change(
        apply_model_browser_drop,
        inputs=[model_browser_apply_data, base_model, vae_model] + lora_ctrls,
        outputs=model_browser_drop_outputs,
        queue=False,
        show_progress=False,
    )

    if not args_manager.args.disable_preset_selection:
        preset_selection.change(preset_selection_change, inputs=[preset_selection, state_is_generating] + lora_ctrls, outputs=load_data_outputs, queue=False, show_progress=True) \
            .then(update_model_dependent_choices, inputs=model_choice_inputs, outputs=model_choice_outputs, queue=False, show_progress=False) \
            .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
            .then(lambda: None, js='()=>{refresh_style_localization();}')

    output_format.change(
        update_history_link,
        inputs=[output_format],
        outputs=[history_link],
        queue=False,
        show_progress=False,
    )

    # load configured default_inpaint_method
    shared.gradio_root.load(inpaint_engine_state_change, inputs=[inpaint_engine_state], outputs=[
        inpaint_engine
    ], show_progress=False, queue=False)

    shared.gradio_root.load(outpaint_engine_state_change, inputs=[outpaint_engine_state], outputs=[
        outpaint_engine
    ], show_progress=False, queue=False)
    objr_engine.change(objr_engine_change, inputs=objr_engine, outputs=objr_mask_dilate, queue=False, show_progress=False)

    prompt.input(parse_meta, inputs=[prompt, state_is_generating], outputs=[prompt, generate_button, load_parameter_button], queue=False, show_progress=False)

    load_parameter_button.click(metadata_ui.load_parameter_button_click, inputs=[prompt, state_is_generating], outputs=load_data_outputs, queue=False, show_progress=False) \
        .then(update_model_dependent_choices, inputs=model_choice_inputs, outputs=model_choice_outputs, queue=False, show_progress=False)

    metadata_import_button.click(trigger_metadata_import, inputs=[metadata_input_image_path, state_is_generating], outputs=load_data_outputs, queue=False, show_progress=True) \
        .then(update_model_dependent_choices, inputs=model_choice_inputs, outputs=model_choice_outputs, queue=False, show_progress=False) \
        .then(style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False)

    import modules.mask_processing as mask_proc
    inpaint_input_image_path.change(
        mask_proc.reset_inpaint_prepared_assets,
        inputs=[],
        outputs=[inpaint_context_mask_image_path, inpaint_context_mask_workspace_id, inpaint_bb_image_path, inpaint_bb_workspace_id, inpaint_mask_image_path, inpaint_mask_workspace_id, inpaint_context_mask_data, inpaint_bb_mask_data, inpaint_bbox, inpaint_step2_checkbox],
        queue=False,
        show_progress=False
    )

    inpaint_context_mask_data.change(
        mask_proc.compute_inpaint_step1_context,
        inputs=[inpaint_input_image_path, inpaint_input_workspace_id, inpaint_context_mask_workspace_id, inpaint_bb_workspace_id, inpaint_mask_workspace_id, inpaint_context_mask_data],
        outputs=[inpaint_context_mask_image_path, inpaint_context_mask_workspace_id, inpaint_bb_image_path, inpaint_bb_workspace_id, inpaint_mask_image_path, inpaint_mask_workspace_id, inpaint_context_mask_data, inpaint_bb_mask_data, inpaint_bbox, inpaint_step2_checkbox],
        queue=False,
        show_progress=False
    )

    inpaint_replace_bb_nonce.change(
        mask_proc.refresh_inpaint_bb_image,
        inputs=[inpaint_input_image_path, inpaint_input_workspace_id, inpaint_context_mask_image_path, inpaint_context_mask_workspace_id, inpaint_bb_workspace_id, inpaint_mask_workspace_id, inpaint_context_mask_data],
        outputs=[inpaint_bb_image_path, inpaint_bb_workspace_id, inpaint_mask_image_path, inpaint_mask_workspace_id, inpaint_bb_mask_data, inpaint_bbox, inpaint_step2_checkbox],
        queue=False,
        show_progress=False
    ).then(
        lambda: None,
        queue=False,
        show_progress=False,
        js="""
        () => {
            const pathFieldIds = ['inpaint_bb_image_path', 'inpaint_mask_image_path'];
            if (typeof window.nexDispatchSlotServerSync === 'function') {
                window.nexDispatchSlotServerSync(pathFieldIds, 'once');
                return;
            }
            window.dispatchEvent(new CustomEvent('nex-slot:server-sync', {
                detail: { pathFieldIds, mode: 'once' },
            }));
        }
        """
    )

    inpaint_bb_mask_data.change(
        mask_proc.compute_inpaint_step2_mask,
        inputs=[inpaint_mask_workspace_id, inpaint_bb_mask_data],
        outputs=[inpaint_mask_image_path, inpaint_mask_workspace_id, inpaint_bb_mask_data],
        queue=False,
        show_progress=False
    )

    outpaint_prepare_button.click(
        mask_proc.prepare_outpaint_step1_assets,
        inputs=[outpaint_input_image, outpaint_input_workspace_id, outpaint_bb_workspace_id, outpaint_mask_workspace_id, outpaint_selections, inpaint_outpaint_expansion_size],
        outputs=[outpaint_input_image, outpaint_input_workspace_id, outpaint_bb_image, outpaint_bb_workspace_id, outpaint_mask_image, outpaint_mask_workspace_id, outpaint_bb_mask_data, outpaint_step2_checkbox, outpaint_prepare_notice],
        queue=False,
        show_progress=True
    ).then(
        lambda: None,
        queue=False,
        show_progress=False,
        js="""
        () => {
            const pathFieldIds = ['outpaint_input_image_path', 'outpaint_bb_image_path', 'outpaint_mask_image_path'];
            if (typeof window.nexDispatchSlotServerSync === 'function') {
                window.nexDispatchSlotServerSync(pathFieldIds, 'once');
                return;
            }
            window.dispatchEvent(new CustomEvent('nex-slot:server-sync', {
                detail: { pathFieldIds, mode: 'once' },
            }));
        }
        """
    )

    outpaint_bb_mask_data.change(
        mask_proc.compute_outpaint_step2_mask,
        inputs=[outpaint_mask_workspace_id, outpaint_bb_mask_data],
        outputs=[outpaint_mask_image, outpaint_mask_workspace_id, outpaint_bb_mask_data],
        queue=False,
        show_progress=False
    )

    remove_mask_data.change(
        mask_proc.compute_remove_mask,
        inputs=[remove_mask_workspace_id, remove_mask_data],
        outputs=[remove_mask_image_path, remove_mask_workspace_id, remove_mask_data],
        queue=False,
        show_progress=False
    )

    generate_button.click(
        fn=prepare_generate_surface,
        inputs=[current_tab],
        outputs=[current_tab, gallery, preview_column, gallery_column],
        js=resolve_generate_tab_js
    ) \
        .then(fn=refresh_seed, inputs=[seed_random, image_seed], outputs=image_seed) \
        .then(fn=get_tasks, inputs=ctrls, outputs=current_tasks_state) \
        .then(
            fn=enqueue_tasks_with_ui_feedback,
            inputs=[current_tasks_state],
            outputs=[currentTask, progress_html, preview_column, gallery_column],
        ) \
        .then(fn=update_history_link, outputs=history_link)

    release_cn_cache_btn.click(
        fn=release_controlnet_cache_clicked,
        outputs=[currentTask]
    )

    def handle_reconnect_click(task):
        worker.request_interrupt('stop', task)

        results = getattr(task, 'results', []) if task else []
        cols = max(1, int(np.ceil(np.sqrt(len(results))))) if len(results) > 0 else 2
        res_val = results[0] if len(results) > 0 else None

        return [
            worker.AsyncTask(args=[]),
            False,
            gr.update(visible=True, interactive=True),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True, value=res_val),
            gr.update(visible=True, value=results, columns=cols),
            gr.update(visible=True),
            gr.update(visible=False)
        ]

    stop_button.click(stop_clicked, inputs=currentTask, outputs=currentTask, queue=False, show_progress=False, js='(x)=>{cancelGenerateForever(); return x;}')
    skip_button.click(skip_clicked, inputs=currentTask, outputs=currentTask, queue=False, show_progress=False)

    example_inpaint_prompts.click(lambda x: x[0], inputs=example_inpaint_prompts, outputs=inpaint_additional_prompt, show_progress=False, queue=False)
    metadata_input_image_path.change(trigger_metadata_preview, inputs=metadata_input_image_path, outputs=metadata_json, queue=False, show_progress=True)
    outpaint_mask_expansion_button.click(expand_mask, inputs=[outpaint_selections, outpaint_mask_image], outputs=[outpaint_mask_image], queue=False, show_progress=False)


def get_tasks(*args):
    global ctrls_keys
    named_args = dict(zip(ctrls_keys, args))
    named_args.pop('_currentTask', None)

    # The queue is now the only repetition model: one Generate click creates
    # one image task, and the next image should be queued with the next click.
    task_args = dict(named_args)
    task_args['image_number'] = 1
    task_args['generate_image_grid'] = False
    task_args['current_tab'] = normalize_current_tab(task_args.get('current_tab'))

    # Freeze the selected UI surface only.  The complete route and ControlNet
    # overlay are compiled after AsyncTask has parsed the raw CN slots; doing
    # route inference here would observe an intentionally empty CN map.
    workflow_selection = capture_workflow_selection(task_args, queue_capture=True)
    task_args['workflow_selection'] = workflow_selection
    task_args['requested_source_surface'] = workflow_selection.source_surface
    validation_message = validate_user_correctable_generate_request(task_args, workflow_selection)
    if validation_message:
        return [_invalid_task(validation_message)]

    frozen_goals = []
    if workflow_selection.source_surface == 'removal':
        if workflow_selection.remove_background:
            frozen_goals.append(flags.remove_bg)
        if workflow_selection.remove_object:
            frozen_goals.append(flags.remove_obj)
    task_args['goals'] = frozen_goals

    task = worker.AsyncTask(args=task_args)
    return [task]


def prepare_generate_surface(current_tab):
    return (
        normalize_current_tab(current_tab),
        gr.skip(),
        gr.update(visible=True),
        gr.update(visible=False),
    )


def release_controlnet_cache_clicked():
    task = worker.AsyncTask(args={})
    task.is_valid = True
    task.is_utility = True
    task.utility_action = "release_controlnet_cache"
    worker.async_tasks.append(task)
    if worker.get_active_task() is None:
        runtime_surface_state.set_progress_state(visible=True, number=1, text='Waiting for task to start ...')
    return task


def enqueue_tasks(tasks, *_legacy_route_inputs):
    import modules.async_worker as worker
    if not isinstance(tasks, list):
        tasks = [tasks]

    first_task = tasks[0] if tasks else None
    if first_task and not first_task.is_valid:
        message = getattr(first_task, 'validation_message', 'The current request is not ready yet.')
        print(f'[Nex] {message}')
        runtime_surface_state.set_idle_notice(message)
        return first_task

    runtime_surface_state.clear_idle_notice()
    for task in tasks:
        worker.async_tasks.append(task)
    if worker.get_active_task() is None:
        runtime_surface_state.set_progress_state(visible=True, number=1, text='Waiting for task to start ...')
    return first_task


def enqueue_tasks_with_ui_feedback(tasks):
    first_task = enqueue_tasks(tasks)
    if first_task and not first_task.is_valid:
        message = getattr(first_task, 'validation_message', 'The current request is not ready yet.')
        return (
            first_task,
            gr.update(visible=True, value=modules.html.make_progress_html(0, message)),
            gr.update(visible=True),
            gr.update(visible=False),
        )

    return (
        first_task,
        gr.update(visible=False),
        gr.skip(),
        gr.skip(),
    )


def _append_new_gallery_items(task, gallery_items, product):
    if not isinstance(product, list):
        product = [product]

    delivered_count = max(0, int(getattr(task, 'ui_delivered_result_count', 0) or 0))
    if delivered_count < len(product):
        gallery_items.extend(product[delivered_count:])
    task.ui_delivered_result_count = max(delivered_count, len(product))


def _queue_prompt_preview(task):
    prompt_preview = getattr(task.state, 'prompt', '')[:40]
    if len(getattr(task.state, 'prompt', '')) > 40:
        prompt_preview += '...'
    if not prompt_preview.strip():
        prompt_preview = "Image generation"
    return prompt_preview


def get_running_task_html(active_task=None):
    if active_task is None:
        import modules.async_worker as worker

        active_task = worker.get_active_task()

    if not active_task:
        return '<p class="empty-queue-msg">No task running.</p>'

    prompt_preview = _queue_prompt_preview(active_task)
    model_name = html.escape(str(getattr(active_task.state, 'base_model_name', '') or ''))
    seed_text = html.escape(str(getattr(active_task.state, 'seed', '') or ''))
    return f"""
    <div class="nex-running-card">
        <div class="nex-running-card__header">
            <span class="badge active-badge">Running</span>
            <span class="task-id">ID: {active_task.task_id}</span>
        </div>
        <p class="task-prompt"><strong>Prompt:</strong> "{html.escape(prompt_preview)}"</p>
        <p class="task-meta">Model: {model_name} | Seed: {seed_text}</p>
    </div>
    """


def get_pending_queue_html(pending_tasks=None, active_task=None):
    if pending_tasks is None:
        import modules.async_worker as worker

        pending_tasks = worker.async_tasks
        if active_task is None:
            active_task = worker.get_active_task()

    if not pending_tasks:
        return '<p class="empty-queue-msg">No queued tasks.</p>'

    pending_html = '<div class="nex-queue-list">'
    for idx, task in enumerate(pending_tasks):
        prompt_preview = _queue_prompt_preview(task)
        pending_html += f"""
        <div class="nex-queue-item pending-task">
            <div class="nex-queue-item-header pending-header">
                <div class="nex-queue-item-summary">
                    <span class="badge pending-badge">Queued #{idx+1}</span>
                    <span class="task-id">ID: {task.task_id}</span>
                </div>
                <button onclick="triggerQueueAction('{task.task_id}', 'cancel')" class="queue-btn btn-cancel pending-inline-action">Cancel</button>
            </div>
            <div class="task-details">
                <p class="task-prompt">"{html.escape(prompt_preview)}"</p>
                <p class="task-meta">Model: {html.escape(str(getattr(task.state, 'base_model_name', '') or ''))} | Seed: {html.escape(str(getattr(task.state, 'seed', '') or ''))}</p>
            </div>
        </div>
        """
    pending_html += '</div>'
    return pending_html


def get_running_status_html(active_task=None):
    if active_task is None:
        import modules.async_worker as worker

        active_task = worker.get_active_task()

    if not active_task:
        return '<p class="nex-running-status empty">Idle.</p>'

    status_text = str(getattr(active_task.state, 'current_status_text', '') or '').strip()
    if not status_text:
        status_text = 'Waiting for task to start ...'
    return f'<p class="nex-running-status">{html.escape(status_text)}</p>'


def get_running_panel_updates(active_task=None):
    if active_task is None:
        import modules.async_worker as worker

        active_task = worker.get_active_task()

    task_html = get_running_task_html(active_task)
    progress_value = max(0, min(int(getattr(getattr(active_task, 'state', None), 'current_progress', 0) or 0), 100)) if active_task else 0
    status_html = get_running_status_html(active_task)
    skip_update = gr.update(interactive=bool(active_task))
    return task_html, progress_value, status_html, skip_update


_last_rendered_running_task_html = None
_last_rendered_running_progress_value = None
_last_rendered_running_status_html = None
_last_rendered_running_skip_interactive = None
_last_rendered_pending_queue_html = None
_last_rendered_queue_len = -1


# Legacy Gradio queue/panel polling remainder kept only as an explicit quarantine.
# W14 removed the active preview transport bridge from the wired UI path.
def poll_active_task_status(session_gallery, last_preview, active_id, disable_preview_val):
    global _last_seen_active_task
    global _last_rendered_running_task_html, _last_rendered_running_progress_value
    global _last_rendered_running_status_html, _last_rendered_running_skip_interactive
    global _last_rendered_pending_queue_html, _last_rendered_queue_len
    global _last_rendered_completed_queue_html
    import modules.async_worker as worker
    import numpy as np
    import gradio as gr

    active_task = worker.get_active_task()
    pending = worker.async_tasks

    # Calculate queue length for tab label update
    queue_len = len(pending)
    if active_task:
        queue_len += 1

    running_task_html, running_progress_value, running_status_html, running_skip_button_update = get_running_panel_updates(active_task)
    pending_queue_html = get_pending_queue_html(pending, active_task=active_task)

    if running_task_html != _last_rendered_running_task_html:
        _last_rendered_running_task_html = running_task_html
        running_task_update = running_task_html
    else:
        running_task_update = gr.skip()

    if running_progress_value != _last_rendered_running_progress_value:
        _last_rendered_running_progress_value = running_progress_value
        running_progress_update = gr.update(value=running_progress_value)
    else:
        running_progress_update = gr.skip()

    if running_status_html != _last_rendered_running_status_html:
        _last_rendered_running_status_html = running_status_html
        running_status_update = running_status_html
    else:
        running_status_update = gr.skip()

    running_skip_interactive = bool(active_task)
    if running_skip_interactive != _last_rendered_running_skip_interactive:
        _last_rendered_running_skip_interactive = running_skip_interactive
        running_skip_update = running_skip_button_update
    else:
        running_skip_update = gr.skip()

    if pending_queue_html != _last_rendered_pending_queue_html:
        _last_rendered_pending_queue_html = pending_queue_html
        pending_queue_update = pending_queue_html
    else:
        pending_queue_update = gr.skip()

    if queue_len != _last_rendered_queue_len:
        _last_rendered_queue_len = queue_len
        if queue_len > 0:
            queue_tab_update = gr.update(label=f"Queue ({queue_len})")
        else:
            queue_tab_update = gr.update(label="Queue")
    else:
        queue_tab_update = gr.skip()

    progress_update = gr.skip()
    preview_update = gr.skip()
    gallery_update = gr.skip()
    preview_column_update = gr.skip()
    gallery_column_update = gr.skip()

    new_active_id = active_id
    new_gallery = list(session_gallery)
    new_preview = last_preview
    previous_task = None

    if (
        active_id is not None
        and _last_seen_active_task is not None
        and getattr(_last_seen_active_task, 'task_id', None) == active_id
        and (active_task is None or getattr(active_task, 'task_id', None) != active_id)
    ):
        previous_task = _last_seen_active_task

    if previous_task is not None:
        _, _, _, previous_finished_images = _drain_task_ui_events(previous_task, new_gallery)
        if previous_finished_images:
            new_preview = previous_finished_images[0]
            preview_update = gr.update(visible=True, value=previous_finished_images[0])

    if active_task:
        new_active_id = active_task.task_id
        latest_preview_img, latest_progress_pct, latest_progress_msg, finished_images = (None, None, None, None)
        task_started = active_id != active_task.task_id

        if task_started:
            preview_column_update = gr.update(visible=True)
            gallery_column_update = gr.update(visible=False)
            progress_update = _get_progress_html_update(visible=True, number=1, text='Waiting for task to start ...')

        latest_preview_img, latest_progress_pct, latest_progress_msg, finished_images = _drain_task_ui_events(active_task, new_gallery)

        if finished_images:
            new_preview = finished_images[0]
            preview_update = gr.update(visible=True, value=finished_images[0])

        if latest_preview_img is not None:
            new_preview = latest_preview_img
            preview_update = gr.update(visible=True, value=latest_preview_img)

        if latest_progress_msg is not None:
            progress_update = _get_progress_html_update(
                visible=True,
                number=latest_progress_pct or 0,
                text=latest_progress_msg,
            )

        if new_gallery != list(session_gallery):
            cols = max(1, int(np.ceil(np.sqrt(len(new_gallery))))) if len(new_gallery) > 0 else 1
            gallery_update = gr.update(value=new_gallery, columns=cols)

    else:
        # Check if active task just finished
        if active_id is not None:
            new_active_id = None
            progress_update = _get_progress_html_update(visible=False)
            # Restore standard layout when done
            preview_column_update = gr.update(visible=True)
            gallery_column_update = gr.update(visible=False)

    if active_task is not None:
        _last_seen_active_task = active_task
    elif active_id is None or previous_task is not None:
        _last_seen_active_task = None

    completed_queue_html = get_completed_queue_html()
    if completed_queue_html != _last_rendered_completed_queue_html:
        _last_rendered_completed_queue_html = completed_queue_html
        completed_queue_update = completed_queue_html
    else:
        completed_queue_update = gr.skip()

    return running_task_update, running_progress_update, running_status_update, running_skip_update, pending_queue_update, progress_update, preview_update, gallery_update, queue_tab_update, new_gallery, new_preview, new_active_id, preview_column_update, gallery_column_update, completed_queue_update


def skip_active_clicked():
    import modules.async_worker as worker

    active_task = worker.get_active_task()
    if active_task is not None:
        worker.request_interrupt('skip', active_task)
    task_html, progress_value, status_html, skip_update = get_running_panel_updates(active_task)
    return task_html, gr.update(value=progress_value), status_html, skip_update, get_pending_queue_html(active_task=active_task)


def handle_queue_action(task_id, action_type, current_task):
    import modules.async_worker as worker
    active_task = worker.get_active_task()
    if action_type == 'stop':
        if active_task is not None and getattr(active_task, 'task_id', None) == task_id:
            worker.request_interrupt('stop', active_task)
        else:
            worker.cancel_task(task_id)
        while len(worker.async_tasks) > 0:
            task = worker.async_tasks.pop(0)
            task.yields.append(['finish', []])
    elif action_type == 'skip':
        if active_task is not None and getattr(active_task, 'task_id', None) == task_id:
            worker.request_interrupt('skip', active_task)
    elif action_type == 'cancel':
        worker.cancel_task(task_id)
    elif action_type == 'delete_completed':
        runtime_surface_state.request_delete_completed_task(task_id)

    task_html, progress_value, status_html, skip_update = get_running_panel_updates()
    return task_html, gr.update(value=progress_value), status_html, skip_update, get_pending_queue_html()


