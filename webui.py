import gradio as gr
import os

gr.set_static_paths(paths=["javascript", "css", f"sdxl_styles{os.sep}samples"])
import random
import os
import json
import time
import numpy as np
import shared
import modules.config
import fooocus_version
import modules.html
import modules.async_worker as worker
import modules.constants as constants
import modules.flags as flags
import modules.gradio_hijack as grh
import modules.style_sorter as style_sorter
import modules.meta_parser
import modules.ui_components.metadata_panel as metadata_panel
import modules.ui_components.settings_panel as settings_panel
import modules.ui_components.styles_panel as styles_panel
import modules.ui_components.models_panel as models_panel
import modules.ui_components.advanced_panel as advanced_panel
import modules.ui_components.control_panel as control_panel
import modules.ui_components.inpaint_panel as inpaint_panel
import modules.ui_components.outpaint_panel as outpaint_panel
import modules.ui_components.staging_panel as staging_panel
from modules.flux_fill_surface import (
    FLUX_FILL_BLEND_MORPHOLOGICAL,
    OBJR_ENGINE_DROPDOWN_CHOICES,
    OBJR_ENGINE_MAT,
)
import args_manager
import copy
from modules.setup_utils import download_models

from modules.sdxl_styles import legal_style_names
from modules.private_logger import get_current_html_path
from modules.ui_gradio_extensions import javascript_html, css_html
from modules.auth import auth_enabled, check_auth
from modules.util import is_json


import modules.ui_logic as ui_logic
from modules.staging_api import staging_router
from modules.runtime_surface_api import runtime_surface_router




# reload_javascript() removed; handled via gr.Blocks(head=...)

title = f'{fooocus_version.app_name} {fooocus_version.version}'

def make_nex_image_slot(slot_id, bridge_id, label, extra_attrs=''):
    attrs = f' {extra_attrs}' if extra_attrs else ''
    return f'<nex-image-slot id="{slot_id}" data-bridge-id="{bridge_id}" data-label="{label}"{attrs}></nex-image-slot>'

if isinstance(args_manager.args.preset, str):
    title += ' ' + args_manager.args.preset

shared.gradio_root = gr.Blocks(title=title, head=javascript_html() + css_html()).queue()

with shared.gradio_root:
    currentTask = gr.State(worker.AsyncTask(args=[]))
    current_tasks_state = gr.State([])
    inpaint_engine_state = gr.State('empty')
    outpaint_engine_state = gr.State('empty')
    remove_mask_state = gr.State(None)
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tabs(selected='preview_workspace'):
                with gr.Tab(label='Preview', id='preview_workspace', elem_id='preview_workspace'):
                    with gr.Row():
                        with gr.Column(scale=5, min_width=420, visible=True) as preview_column:
                            gr.HTML('<div id="nex-runtime-preview-panel" class="nex-runtime-preview-panel main_view"></div>')
                        with gr.Column(scale=6, min_width=500, visible=False) as gallery_column:
                            gallery = gr.Gallery(label='Gallery', show_label=True, object_fit='contain', visible=True, height=768,
                                                 elem_classes=['resizable_area', 'main_view', 'final_gallery', 'image_gallery'],
                                                 elem_id='final_gallery')
                with gr.Tab(label='Model Browser', id='model_browser_workspace'):
                    gr.HTML(
                        """
<div id="nex-model-browser-panel" class="nex-model-browser-panel">
  <nex-model-browser id="nex-model-browser" data-refresh-button-id="refresh_files_button" data-apply-data-id="model_browser_apply_data_bridge"></nex-model-browser>
</div>
                        """
                    )
                    model_browser_apply_data = gr.Textbox(value='', visible=True, elem_id='model_browser_apply_data_bridge', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
            gr.HTML('<div id="nex-runtime-status-panel" class="nex-runtime-status-panel"></div>')
            progress_html = gr.HTML(value=modules.html.make_progress_html(32, 'Progress 32%'), visible=False,
                                    elem_id='progress-bar', elem_classes='progress-bar')
            with gr.Row():
                with gr.Column(scale=17):
                    prompt = gr.Textbox(show_label=False, placeholder="Type prompt here or paste parameters.", elem_id='positive_prompt',
                                        autofocus=True, lines=3)

                    default_prompt = modules.config.default_prompt
                    if isinstance(default_prompt, str) and default_prompt != '':
                        shared.gradio_root.load(lambda: default_prompt, outputs=prompt)

                with gr.Column(scale=3, min_width=0):
                    generate_button = gr.Button(value="Generate", elem_classes='type_row', elem_id='generate_button', visible=True)
                    load_parameter_button = gr.Button(value="Load Parameters", elem_classes='type_row', elem_id='load_parameter_button', visible=False)
                    skip_button = gr.Button(value="Skip", elem_classes='type_row_half', elem_id='skip_button', visible=False)
                    stop_button = gr.Button(value="Stop", elem_classes='type_row_half', elem_id='stop_button', visible=False)



            with gr.Row(elem_classes='advanced_check_row'):
                input_image_checkbox = gr.Checkbox(label='Input Image', value=modules.config.default_image_prompt_checkbox, container=False, elem_classes='min_check')
            with gr.Row(visible=modules.config.default_image_prompt_checkbox) as image_input_panel:
                with gr.Tabs(selected=modules.config.default_selected_image_input_tab_id):
                    with gr.Tab(label='Upscale / Super / Color', id='uov_tab') as uov_tab:
                        with gr.Row():
                            with gr.Column():
                                gr.HTML(make_nex_image_slot('uov_input_slot', 'uov_input_image_bridge', 'Image', 'data-upload-mode="api" data-path-field-id="uov_input_image_path" data-workspace-field-id="uov_input_workspace_id"'))
                                uov_input_image = gr.Image(label='Image', sources='upload', type='filepath', show_label=False, elem_id='uov_input_image_bridge', elem_classes=['nex-image-slot-bridge'])
                                uov_input_image_path = gr.Textbox(value='', visible=True, elem_id='uov_input_image_path', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                uov_input_workspace_id = gr.Textbox(value='', visible=True, elem_id='uov_input_workspace_id', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                with gr.Group(visible=False) as upscale_gan_output_container:
                                    gr.HTML(make_nex_image_slot('upscale_gan_output_slot', 'upscale_gan_output_bridge', 'Upscale Target', 'data-upload-mode="api" data-path-field-id="upscale_gan_output_path" data-workspace-field-id="upscale_gan_output_workspace_id"'))
                                    gr.HTML('<div class="nex-image-slot-guidance">Place your already upscaled image here. Color Enhancement uses it as the donor target, and Super-Upscale uses it as the tiled refinement target.</div>')
                                    upscale_gan_output_image = gr.Image(label='Upscale Target', sources='upload', type='filepath', show_label=False, elem_id='upscale_gan_output_bridge', elem_classes=['nex-image-slot-bridge'])
                                    upscale_gan_output_path = gr.Textbox(value='', visible=True, elem_id='upscale_gan_output_path', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                    upscale_gan_output_workspace_id = gr.Textbox(value='', visible=True, elem_id='upscale_gan_output_workspace_id', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                            with gr.Column():
                                uov_method = gr.Radio(label='Method:', choices=['Upscale', 'Super-Upscale', 'Color Enhancement'], value='Upscale')
                                upscale_model = gr.Dropdown(label='Upscale Model', choices=['None'], value='None')
                                upscale_scale_info = gr.HTML(value="<b>Scale:</b> Auto-detecting...", elem_id='upscale_scale_info')
                                upscale_scale_override = gr.Slider(label='Scale Override', minimum=0.0, maximum=8.0, step=0.1, value=0.0, info='Set to 0.0 to use model default scale.')
                                upscale_gan_tile_size = gr.Slider(label='GAN Tiling Size', minimum=256, maximum=1024, step=64, value=256, info='Explicit tile size for direct GAN upscale. Uses a 256px safe floor and 64px increments.')
                                upscale_prompt = gr.Textbox(label='Color Enhancement Prompt (optional)', placeholder='Optional color/style prompt for Color Enhancement.', visible=False)

                                with gr.Group(visible=False) as upscale_refinement_container:
                                    upscale_refinement_denoise = gr.Slider(label='Refinement Denoise', minimum=0.0, maximum=1.0, step=0.001, value=0.382)
                                    upscale_refinement_tile_overlap = gr.Slider(label='Refinement Tile Overlap', minimum=0, maximum=256, step=1, value=128)

                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/390" target="_blank">\U0001F4D4 Documentation</a>')

                    with gr.Tab(label='Remove', id='remove_tab') as remove_tab:
                        with gr.Row():
                            with gr.Column():
                                gr.HTML(make_nex_image_slot('remove_base_image_slot', 'remove_base_image_bridge', 'Base Image', 'data-upload-mode="api" data-path-field-id="remove_base_image_path" data-workspace-field-id="remove_base_workspace_id" data-tool-group="remove"'))
                                remove_base_image = gr.Image(label='Base Image', sources='upload', type='filepath', height=500, show_label=False, elem_id='remove_base_image_bridge', elem_classes=['nex-image-slot-bridge'])
                                remove_base_image_path = gr.Textbox(value='', visible=True, elem_id='remove_base_image_path', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                remove_base_workspace_id = gr.Textbox(value='', visible=True, elem_id='remove_base_workspace_id', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                with gr.Row():
                                    remove_bg_enabled = gr.Checkbox(label='Background pass', value=False, elem_id='remove_bg_enabled')
                                    remove_obj_enabled = gr.Checkbox(label='Object pass', value=False, elem_id='remove_obj_enabled')

                                objr_engine = gr.Dropdown(label='Removal Pass', choices=OBJR_ENGINE_DROPDOWN_CHOICES, value=OBJR_ENGINE_MAT)
                                remove_prompt = gr.Textbox(placeholder='Optional prompt for the Flux Fill refinement pass. Empty uses the downloaded empty conditioning cache.', elem_id='remove_prompt', label='Remove Prompt', visible=True)
                                flux_fill_conditioning = gr.Textbox(value='empty', visible=False, elem_id='flux_fill_conditioning', show_label=False, container=False)
                                flux_fill_prompt_cache = gr.Textbox(value='temp', visible=False, elem_id='flux_fill_prompt_cache', show_label=False, container=False)
                                objr_blend_mode = gr.Textbox(value=FLUX_FILL_BLEND_MORPHOLOGICAL, visible=False, elem_id='objr_blend_mode', show_label=False, container=False)
                                gr.HTML('* <b>Background pass</b> extracts the person.<br>'
                                        '* <b>Object pass</b> runs MAT512 first, then Flux Fill refinement. Morphological blending is fixed on.')

                            with gr.Column():
                                remove_mask_data = gr.Textbox(value='', visible=True, elem_id='remove_mask_data', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                gr.HTML(make_nex_image_slot('remove_mask_image_slot', 'remove_mask_image_bridge', 'Mask', 'data-upload-mode="api" data-path-field-id="remove_mask_image_path" data-workspace-field-id="remove_mask_workspace_id"'))
                                remove_mask_image = gr.Image(label='Mask', sources='upload', type='filepath', height=500, elem_id='remove_mask_image_bridge', elem_classes=['nex-image-slot-bridge'])
                                remove_mask_image_path = gr.Textbox(value='', visible=True, elem_id='remove_mask_image_path', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                remove_mask_workspace_id = gr.Textbox(value='', visible=True, elem_id='remove_mask_workspace_id', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                gr.HTML("""
<div id="remove-mask-tools" class="mask-workflow-toolbar" style="display:flex; flex-direction:column; gap:14px; margin:8px 0 16px; padding:14px; border:1px solid rgba(128,128,128,0.2); border-radius:12px; background:rgba(128,128,128,0.03);">
  <div style="display:flex; flex-wrap:wrap; gap:12px; align-items:center;">
    <span style="font-size:0.9rem; font-weight:700; color:var(--body-text-color); margin-right:4px;">REMOVE MASK</span>
    <div style="display:flex; gap:8px; padding:2px; background:rgba(0,0,0,0.1); border-radius:8px;">
      <button type="button" class="mask-tool-btn" id="remove-mask-mode-bb" title="Enable Remove Mask">Mask</button>
      <button type="button" class="mask-tool-btn active" id="remove-mask-mode-disable" title="Disable Masking">Disable</button>
    </div>
    <button type="button" class="mask-tool-btn" id="remove-mask-reset" title="Refresh masking controls if tools stop responding" style="border-color:rgba(255,160,64,0.45); color:rgba(255,180,96,0.95); margin-left:auto;">Refresh</button>
  </div>
  <div style="display:flex; flex-wrap:wrap; gap:16px; align-items:center; padding-top:4px; border-top:1px solid rgba(128,128,128,0.1);">
    <label style="display:flex; align-items:center; gap:12px; font-size:0.9rem; font-weight:500; flex-grow:1; min-width:200px;">
      <span style="white-space:nowrap; opacity:0.8;">Brush Size</span>
      <input id="remove-mask-size" type="range" min="8" max="160" step="1" value="36" style="flex-grow:1; accent-color:var(--button-primary-background-fill);">
    </label>
    <span id="remove-mask-status" style="font-size:0.85rem; opacity:0.6; font-style:italic; min-width:120px; text-align:right;">Ready</span>
  </div>
</div>
""")
                                bgr_threshold = gr.Slider(label='BGR Threshold', minimum=0.0, maximum=1.0, step=0.01, value=0.5, info='Higher = tighter cutout; Lower = keep softer edges.')
                                bgr_jit = gr.Checkbox(label='Use JIT (Optimized)', value=True)
                                objr_mask_dilate = gr.Slider(label='Mask Dilate', minimum=0, maximum=128, step=1, value=16, info='Shared default for MAT512 and Flux Fill.')
                                objr_mask_blur = gr.Slider(label='Flux Mask Blur', minimum=0, maximum=64, step=1, value=6, info='Lower keeps refinement sharper; higher softens the edge more.')
                    with gr.Tab(label='Controlnet', id='ip_tab') as ip_tab:
                        ip_images = []
                        cn_image_paths = []
                        ip_types = []
                        ip_stops = []
                        ip_weights = []
                        ip_ad_cols = []

                        guidance_choices_by_channel = {
                            flags.cn_structural: flags.cn_structural_types,
                            flags.cn_contextual: flags.cn_contextual_types,
                        }
                        def resolve_channel_default(image_count):
                            default_type = flags.resolve_cn_type(modules.config.default_ip_types[image_count])
                            default_channel = flags.get_cn_channel(default_type)
                            if default_channel in guidance_choices_by_channel:
                                return default_channel
                            return flags.cn_contextual

                        def resolve_type_default(image_count, channel):
                            choices = guidance_choices_by_channel.get(channel, flags.cn_contextual_types)
                            default_type = flags.resolve_cn_type(modules.config.default_ip_types[image_count])
                            if default_type in choices:
                                return default_type
                            return choices[0]

                        def update_guidance_type_choices(channel):
                            normalized_channel = channel if channel in guidance_choices_by_channel else flags.cn_contextual
                            choices = guidance_choices_by_channel.get(normalized_channel, flags.cn_contextual_types)
                            default_type = flags.get_default_cn_type_for_channel(normalized_channel)
                            default_stop, default_weight = flags.get_default_cn_parameters_for_type(default_type)
                            return (
                                gr.update(choices=choices, value=default_type),
                                gr.update(value=float(default_stop)),
                                gr.update(value=float(default_weight)),
                            )

                        def create_ip_slot(image_count):
                            default_channel = resolve_channel_default(image_count)
                            default_type = resolve_type_default(image_count, default_channel)
                            with gr.Column():
                                gr.HTML(make_nex_image_slot(
                                    f'ip_image_slot_{image_count}',
                                    f'ip_image_bridge_{image_count}',
                                    f'Guidance Image {image_count}',
                                    f'data-upload-mode="api" data-path-field-id="cn_{image_count - 1}_image_path" data-workspace-field-id="cn_{image_count - 1}_workspace_id" data-method-field-id="cn_{image_count - 1}_type"'
                                ))
                                ip_image = gr.Image(
                                    label='Image',
                                    sources='upload',
                                    type='filepath',
                                    show_label=False,
                                    height=300,
                                    value=modules.config.default_ip_images[image_count],
                                    elem_id=f'ip_image_bridge_{image_count}',
                                    elem_classes=['nex-image-slot-bridge']
                                )
                                cn_image_path = gr.Textbox(
                                    value=modules.config.default_ip_images[image_count],
                                    visible=True,
                                    elem_id=f'cn_{image_count - 1}_image_path',
                                    elem_classes=['inpaint-hidden-mask-field'],
                                    show_label=False,
                                    container=False
                                )
                                cn_workspace_id = gr.Textbox(
                                    value='',
                                    visible=True,
                                    elem_id=f'cn_{image_count - 1}_workspace_id',
                                    elem_classes=['inpaint-hidden-mask-field'],
                                    show_label=False,
                                    container=False
                                )
                                ip_images.append(ip_image)
                                cn_image_paths.append(cn_image_path)
                                with gr.Column(visible=True) as ad_col:
                                    with gr.Row():
                                        ip_channel = gr.Radio(
                                            label='Guidance Channel',
                                            choices=[flags.cn_structural, flags.cn_contextual],
                                            value=default_channel,
                                            container=False,
                                            scale=1
                                        )
                                        ip_type = gr.Dropdown(
                                            label='Method',
                                            choices=guidance_choices_by_channel.get(default_channel, flags.cn_contextual_types),
                                            value=default_type,
                                            allow_custom_value=True,
                                            container=False,
                                            scale=1,
                                            elem_id=f'cn_{image_count - 1}_type'
                                        )
                                        ip_types.append(ip_type)

                                    with gr.Row():
                                        ip_stop = gr.Slider(
                                            label='Stop At',
                                            minimum=0.0,
                                            maximum=1.0,
                                            step=0.001,
                                            value=modules.config.default_ip_stop_ats[image_count]
                                        )
                                        ip_stops.append(ip_stop)

                                        ip_weight = gr.Slider(
                                            label='Weight',
                                            minimum=0.0,
                                            maximum=2.0,
                                            step=0.001,
                                            value=modules.config.default_ip_weights[image_count]
                                        )
                                        ip_weights.append(ip_weight)

                                        ip_channel.change(
                                            fn=update_guidance_type_choices,
                                            inputs=ip_channel,
                                            outputs=[ip_type, ip_stop, ip_weight],
                                            queue=False,
                                            show_progress=False
                                        )

                                ip_ad_cols.append(ad_col)

                        with gr.Row():
                            with gr.Column(scale=1):
                                for image_count in range(1, modules.config.default_controlnet_image_count + 1, 2):
                                    create_ip_slot(image_count)

                            with gr.Column(scale=1):
                                for image_count in range(2, modules.config.default_controlnet_image_count + 1, 2):
                                    create_ip_slot(image_count)

                        with gr.Group():
                            gr.HTML('<div style="margin-top:20px; border-top:1px solid rgba(128,128,128,0.2); padding-top:15px; font-weight:bold;">Advanced Control</div>')
                            control_panel_result = control_panel.build_control_tab()
                            skipping_cn_preprocessor = control_panel_result['skipping_cn_preprocessor']
                            mixing_image_prompt_and_inpaint = control_panel_result['mixing_image_prompt_and_inpaint']
                            mixing_image_prompt_and_outpaint = control_panel_result['mixing_image_prompt_and_outpaint']
                            controlnet_softness = control_panel_result['controlnet_softness']
                            canny_low_threshold = control_panel_result['canny_low_threshold']
                            canny_high_threshold = control_panel_result['canny_high_threshold']

                        gr.HTML('* "Controlnet" is powered by Fooocus Image Mixture Engine (v1.0.1). <a href="https://github.com/lllyasviel/Fooocus/discussions/557" target="_blank">Documentation</a>')



                    with gr.Tab(label='Outpaint', id='outpaint_tab') as outpaint_tab:
                        with gr.Row():
                            with gr.Column():
                                gr.HTML(make_nex_image_slot('outpaint_input_slot', 'outpaint_input_image_bridge', 'Base Image', 'data-upload-mode="api" data-path-field-id="outpaint_input_image_path" data-workspace-field-id="outpaint_input_workspace_id"'))
                                outpaint_input_image = gr.Image(label='Base Image', sources='upload', type='filepath', height=500, show_label=False, elem_id='outpaint_input_image_bridge', elem_classes=['nex-image-slot-bridge'])
                                outpaint_input_image_path = gr.Textbox(value='', visible=True, elem_id='outpaint_input_image_path', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                outpaint_input_workspace_id = gr.Textbox(value='', visible=True, elem_id='outpaint_input_workspace_id', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                outpaint_selections = gr.CheckboxGroup(choices=['Left', 'Right', 'Top', 'Bottom'], value=['Left'], label='Outpaint Direction')
                                with gr.Column(elem_classes=["step2-toolbox"]):
                                    outpaint_prepare_button = gr.Button(value='Prepare Outpaint', variant='primary', elem_id='outpaint_prepare_button')
                                    outpaint_step2_checkbox = gr.Checkbox(label='Prepared Outpaint Assets', value=False, visible=False, elem_id='outpaint_step2_checkbox', elem_classes=['step2-status-btn'], container=False)
                                    outpaint_prepare_notice = gr.Markdown(value='')
                                    gr.HTML('<p class="step2-desc">Prepare the expanded canvas and BB assets first, then Generate runs inference with those resolved slots.</p>')

                                outpaint_panel_result = outpaint_panel.build_outpaint_tab()
                                outpaint_engine = outpaint_panel_result['outpaint_engine']
                                outpaint_strength = outpaint_panel_result['outpaint_strength']
                                inpaint_outpaint_expansion_size = outpaint_panel_result['inpaint_outpaint_expansion_size']
                                outpaint_additional_prompt = gr.Textbox(placeholder="Describe what you want to outpaint.", elem_id='outpaint_additional_prompt', label='Outpaint Additional Prompt', visible=True)

                                gr.HTML('* Powered by Fooocus Inpaint Engine <a href="https://github.com/lllyasviel/Fooocus/discussions/414" target="_blank">\U0001F4D4 Documentation</a>')

                            with gr.Column(visible=True) as outpaint_mask_generation_col:
                                gr.HTML(make_nex_image_slot('outpaint_bb_canvas', 'outpaint_bb_image_bridge', 'BB Image', 'data-upload-mode="api" data-path-field-id="outpaint_bb_image_path" data-workspace-field-id="outpaint_bb_workspace_id" data-tool-group="outpaint"'))
                                outpaint_bb_image_path = gr.Textbox(value='', visible=True, elem_id='outpaint_bb_image_path', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                outpaint_bb_workspace_id = gr.Textbox(value='', visible=True, elem_id='outpaint_bb_workspace_id', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                gr.HTML("""
<div id="outpaint-mask-tools" class="mask-workflow-toolbar" style="display:flex; flex-direction:column; gap:14px; margin:8px 0 16px; padding:14px; border:1px solid rgba(128,128,128,0.2); border-radius:12px; background:rgba(128,128,128,0.03);">
  <div style="display:flex; flex-wrap:wrap; gap:12px; align-items:center;">
    <span style="font-size:0.9rem; font-weight:700; color:var(--body-text-color); margin-right:4px;">OUTPAINT MASK</span>
    <div style="display:flex; gap:8px; padding:2px; background:rgba(0,0,0,0.1); border-radius:8px;">
      <button type="button" class="mask-tool-btn" id="outpaint-mask-mode-bb" title="Enable BB Mask">BB Mask</button>
      <button type="button" class="mask-tool-btn active" id="outpaint-mask-mode-disable" title="Disable Masking">Disable</button>
    </div>
    <button type="button" class="mask-tool-btn" id="outpaint-mask-reset" title="Refresh masking controls if tools stop responding" style="border-color:rgba(255,160,64,0.45); color:rgba(255,180,96,0.95); margin-left:auto;">Refresh</button>
  </div>
  <div style="display:flex; flex-wrap:wrap; gap:16px; align-items:center; padding-top:4px; border-top:1px solid rgba(128,128,128,0.1);">
    <label style="display:flex; align-items:center; gap:12px; font-size:0.9rem; font-weight:500; flex-grow:1; min-width:200px;">
      <span style="white-space:nowrap; opacity:0.8;">Brush Size</span>
      <input id="outpaint-mask-size" type="range" min="8" max="160" step="1" value="36" style="flex-grow:1; accent-color:var(--button-primary-background-fill);">
    </label>
    <span id="outpaint-mask-status" style="font-size:0.85rem; opacity:0.6; font-style:italic; min-width:120px; text-align:right;">Ready</span>
  </div>
</div>
""")
                                outpaint_bb_mask_data = gr.Textbox(value="", visible=True, elem_id="outpaint_bb_mask_data", elem_classes=["inpaint-hidden-mask-field"], show_label=False, container=False)
                                gr.HTML(make_nex_image_slot('outpaint_mask_canvas', 'outpaint_mask_image_bridge', 'BB Mask', 'data-upload-mode="api" data-path-field-id="outpaint_mask_image_path" data-workspace-field-id="outpaint_mask_workspace_id"'))
                                outpaint_mask_image_path = gr.Textbox(value='', visible=True, elem_id='outpaint_mask_image_path', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                outpaint_mask_workspace_id = gr.Textbox(value='', visible=True, elem_id='outpaint_mask_workspace_id', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                outpaint_mask_expansion_button = gr.Button(value='Expand Mask (32 pixels)', visible=False)

                    with gr.Tab(label='Inpaint', id='inpaint_tab') as inpaint_tab:
                        with gr.Row():
                            with gr.Column():
                                gr.HTML(make_nex_image_slot('inpaint_canvas', 'inpaint_input_image_bridge', 'Base Image', 'data-upload-mode="api" data-path-field-id="inpaint_input_image_path" data-workspace-field-id="inpaint_input_workspace_id" data-tool-group="inpaint-base"'))
                                inpaint_input_image = gr.Image(label='Base Image', sources='upload', type='filepath', height=500, elem_id='inpaint_input_image_bridge', show_label=False, elem_classes=['nex-image-slot-bridge'])
                                inpaint_input_image_path = gr.Textbox(value='', visible=True, elem_id='inpaint_input_image_path', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                inpaint_input_workspace_id = gr.Textbox(value='', visible=True, elem_id='inpaint_input_workspace_id', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                inpaint_context_mask_data = gr.Textbox(value="", visible=True, elem_id="inpaint_context_mask_data", elem_classes=["inpaint-hidden-mask-field"], show_label=False, container=False)
                                inpaint_replace_bb_nonce = gr.Textbox(value='', visible=True, elem_id='inpaint_replace_bb_nonce', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                gr.HTML("""
<div id="inpaint-mask-tools" style="display:flex; flex-direction:column; gap:14px; margin:8px 0 16px; padding:14px; border:1px solid rgba(128,128,128,0.2); border-radius:12px; background:rgba(128,128,128,0.03);">
  <div style="display:flex; flex-wrap:wrap; gap:12px; align-items:center;">
    <span style="font-size:0.9rem; font-weight:700; color:var(--body-text-color); margin-right:4px;">Inpaint Mask</span>
    <div style="display:flex; gap:8px; padding:2px; background:rgba(0,0,0,0.1); border-radius:8px;">
      <button type="button" class="mask-tool-btn" id="inpaint-mask-mode-context" title="Paint Context Mask">Context Mask</button>
      <button type="button" class="mask-tool-btn" id="inpaint-mask-mode-bb" title="Paint BB Mask">BB Mask</button>
      <button type="button" class="mask-tool-btn active" id="inpaint-mask-mode-disable" title="Disable Masking">Disable</button>
    </div>
    <button type="button" class="mask-tool-btn" id="inpaint-mask-reset" title="Refresh masking controls if tools stop responding" style="border-color:rgba(255,160,64,0.45); color:rgba(255,180,96,0.95); margin-left:auto;">Refresh</button>
  </div>
  <div style="display:flex; flex-wrap:wrap; gap:16px; align-items:center; padding-top:4px; border-top:1px solid rgba(128,128,128,0.1);">
    <label style="display:flex; align-items:center; gap:12px; font-size:0.9rem; font-weight:500; flex-grow:1; min-width:200px;">
      <span style="white-space:nowrap; opacity:0.8;">Brush Size</span>
      <input id="inpaint-mask-size" type="range" min="8" max="160" step="1" value="36" style="flex-grow:1; accent-color:var(--button-primary-background-fill);">
    </label>
    <button type="button" class="mask-tool-btn" id="inpaint-mask-refresh-bb" title="Rebuild BB Image from the current Base Image and Context Mask">Replace BB Image</button>
    <span id="inpaint-mask-status" style="font-size:0.85rem; opacity:0.6; font-style:italic; min-width:120px; text-align:right;">Ready</span>
  </div>
</div>
""")
                                inpaint_toggle_toolbar = gr.Button("Toggle Canvas Toolbar", size="sm", visible=False)
                                inpaint_additional_prompt = gr.Textbox(placeholder="Describe what you want to inpaint.", elem_id='inpaint_additional_prompt', label='Inpaint Additional Prompt', visible=True)
                                example_inpaint_prompts = gr.Dataset(samples=modules.config.example_inpaint_prompts,
                                                                     label='Additional Prompt Quick List',
                                                                     components=[inpaint_additional_prompt],
                                                                     visible=True)
                                with gr.Column(elem_classes=["step2-toolbox"]):
                                    inpaint_step2_checkbox = gr.Checkbox(label='Prepared Inpaint Assets', value=False, visible=False, elem_id='inpaint_step2_checkbox', elem_classes=['step2-status-btn'], container=False)
                                    gr.HTML('<p class="step2-desc step2-desc--inpaint"><span class="step2-desc__title">Prepare Inpaint Assets</span><span class="step2-desc__body">Context Mask, BB Image, and BB Mask are prepared before inference. Generate then runs from the resolved inpaint slots.</span></p>')

                                inpaint_panel_result = inpaint_panel.build_inpaint_tab()
                                debugging_inpaint_preprocessor = inpaint_panel_result['debugging_inpaint_preprocessor']
                                inpaint_disable_initial_latent = inpaint_panel_result['inpaint_disable_initial_latent']
                                inpaint_engine = inpaint_panel_result['inpaint_engine']
                                inpaint_route = inpaint_panel_result['inpaint_route']
                                inpaint_strength = inpaint_panel_result['inpaint_strength']
                                inpaint_erode_or_dilate = inpaint_panel_result['inpaint_erode_or_dilate']

                                gr.HTML('* Powered by Fooocus Inpaint Engine <a href="https://github.com/lllyasviel/Fooocus/discussions/414" target="_blank">Documentation</a>')

                            with gr.Column(visible=True) as inpaint_mask_generation_col:
                                gr.HTML(make_nex_image_slot('inpaint_context_mask_canvas', 'inpaint_context_mask_image_bridge', 'Context Mask', 'data-upload-mode="api" data-path-field-id="inpaint_context_mask_image_path" data-workspace-field-id="inpaint_context_mask_workspace_id"'))
                                inpaint_context_mask_image = gr.Image(label='Context Mask', sources='upload', type='filepath', height=500, elem_id='inpaint_context_mask_image_bridge', elem_classes=['nex-image-slot-bridge'])
                                inpaint_context_mask_image_path = gr.Textbox(value='', visible=True, elem_id='inpaint_context_mask_image_path', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                inpaint_context_mask_workspace_id = gr.Textbox(value='', visible=True, elem_id='inpaint_context_mask_workspace_id', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                gr.HTML(make_nex_image_slot('inpaint_bb_canvas', 'inpaint_bb_image_bridge', 'BB Image', 'data-upload-mode="api" data-path-field-id="inpaint_bb_image_path" data-workspace-field-id="inpaint_bb_workspace_id" data-tool-group="inpaint-bb"'))
                                inpaint_bb_image = gr.Image(label='BB Image', sources='upload', type='filepath', height=500, elem_id='inpaint_bb_image_bridge', elem_classes=['nex-image-slot-bridge'])
                                inpaint_bb_image_path = gr.Textbox(value='', visible=True, elem_id='inpaint_bb_image_path', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                inpaint_bb_workspace_id = gr.Textbox(value='', visible=True, elem_id='inpaint_bb_workspace_id', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                inpaint_bb_mask_data = gr.Textbox(value="", visible=True, elem_id="inpaint_bb_mask_data", elem_classes=["inpaint-hidden-mask-field"], show_label=False, container=False)
                                gr.HTML(make_nex_image_slot('inpaint_mask_canvas', 'inpaint_mask_image_bridge', 'BB Mask', 'data-upload-mode="api" data-path-field-id="inpaint_mask_image_path" data-workspace-field-id="inpaint_mask_workspace_id"'))
                                inpaint_mask_image = gr.Image(label='BB Mask', sources='upload', type='filepath', height=500, elem_id='inpaint_mask_image_bridge', elem_classes=['nex-image-slot-bridge'])
                                inpaint_mask_image_path = gr.Textbox(value='', visible=True, elem_id='inpaint_mask_image_path', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                inpaint_mask_workspace_id = gr.Textbox(value='', visible=True, elem_id='inpaint_mask_workspace_id', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)





                    with gr.Tab(label='Metadata', id='metadata_tab') as metadata_tab:
                        metadata_panel_result = metadata_panel.build_metadata_tab()
                        metadata_input_image = metadata_panel_result['metadata_input_image']
                        metadata_input_image_path = metadata_panel_result['metadata_input_image_path']
                        metadata_input_workspace_id = metadata_panel_result['metadata_input_workspace_id']
                        metadata_json = metadata_panel_result['metadata_json']
                        metadata_import_button = metadata_panel_result['metadata_import_button']

            current_tab = gr.Textbox(value='uov', visible=False)

            # Phase 3 UI Bindings




        with gr.Column(scale=1, visible=True) as advanced_column:
            with gr.Row():
                gr.HTML('<button id="staging-panel-launcher" class="lg secondary gradio-button" style="width:100%; margin-bottom:12px; font-weight:bold;">\U0001F5C2\uFE0F Open Staging Palette</button>')
                gr.HTML('<button id="monitor-panel-launcher" class="lg secondary gradio-button" style="width:100%; margin-bottom:12px; font-weight:bold;">\U0001F4CA Monitor Dashboard</button>')

            with gr.Tab(label='Queue', elem_id='nex-queue-tab-wrapper') as queue_tab:
                gr.HTML('<div id="nex-runtime-queue-panel" class="nex-runtime-queue-panel"></div>')

            with gr.Tab(label='Settings'):
                settings_panel_result = settings_panel.build_settings_tab()
                if not args_manager.args.disable_preset_selection:
                    preset_selection = settings_panel_result['preset_selection']
                aspect_ratios_selection = settings_panel_result['aspect_ratios_selection']
                image_number = settings_panel_result['image_number']
                steps = settings_panel_result['steps']
                sampler_name = settings_panel_result['sampler_name']
                scheduler_name = settings_panel_result['scheduler_name']
                guidance_scale = settings_panel_result['guidance_scale']
                clip_skip = settings_panel_result['clip_skip']
                # output_format moved to debug_panel_result
                negative_prompt = settings_panel_result['negative_prompt']
                seed_random = settings_panel_result['seed_random']
                image_seed = settings_panel_result['image_seed']
                history_link = settings_panel_result['history_link']





            with gr.Tab(label='Models'):
                models_panel_result = models_panel.build_models_tab()
                base_model = models_panel_result['base_model']
                vae_model = models_panel_result['vae_model']
                clip_model = models_panel_result['clip_model']

                style_search_bar = models_panel_result['style_search_bar']
                style_selections = models_panel_result['style_selections']
                style_selections_accordion = models_panel_result['style_selections_accordion']

                lora_ctrls = models_panel_result['lora_ctrls']
                refresh_files = models_panel_result['refresh_files']
            with gr.Tab(label='Advanced'):
                debug_panel_result = advanced_panel.build_debug_tab()
                sharpness = debug_panel_result['sharpness']
                output_format = debug_panel_result['output_format']
                adm_scaler_positive = debug_panel_result['adm_scaler_positive']
                adm_scaler_negative = debug_panel_result['adm_scaler_negative']
                adm_scaler_end = debug_panel_result['adm_scaler_end']
                adaptive_cfg = debug_panel_result['adaptive_cfg']
                generate_image_grid = debug_panel_result['generate_image_grid']
                overwrite_width = debug_panel_result['overwrite_width']
                overwrite_height = debug_panel_result['overwrite_height']
                overwrite_upscale_strength = debug_panel_result['overwrite_upscale_strength']
                disable_preview = debug_panel_result['disable_preview']
                preview_update_interval = debug_panel_result['preview_update_interval']
                disable_intermediate_results = debug_panel_result['disable_intermediate_results']
                disable_seed_increment = debug_panel_result['disable_seed_increment']
                prefetch_depth = debug_panel_result['prefetch_depth']
                prefetch_chunk_mb = debug_panel_result['prefetch_chunk_mb']
                flux_fill_runtime_posture = debug_panel_result['flux_fill_runtime_posture']
                flux_fill_t5_posture = debug_panel_result['flux_fill_t5_posture']
                flux_fill_disk_paged_t5_gc_interval = debug_panel_result['flux_fill_disk_paged_t5_gc_interval']
                sdxl_assembly_posture = debug_panel_result['sdxl_assembly_posture']
                if not args_manager.args.disable_metadata:
                    save_metadata_to_images = debug_panel_result['save_metadata_to_images']
                    metadata_scheme = debug_panel_result['metadata_scheme']

                # Control settings moved to Image Prompt tab
                # (Removed outpaint advanced tab)
                # (Removed inpaint advanced tab)

                outpaint_ctrls = [outpaint_engine, outpaint_strength,
                                  inpaint_outpaint_expansion_size, outpaint_step2_checkbox]
                inpaint_ctrls = [debugging_inpaint_preprocessor, inpaint_disable_initial_latent, inpaint_engine,
                                 inpaint_route, inpaint_strength, inpaint_erode_or_dilate, inpaint_step2_checkbox]





        state_is_generating = gr.State(False)

        load_data_outputs = [image_number, prompt, negative_prompt, style_selections,
                             steps, aspect_ratios_selection,
                             overwrite_width, overwrite_height, guidance_scale, sharpness, adm_scaler_positive,
                             adm_scaler_negative, adm_scaler_end, adaptive_cfg, clip_skip,
                             base_model, vae_model, clip_model, sampler_name, scheduler_name,
                             seed_random, image_seed, outpaint_engine_state, inpaint_engine_state, inpaint_route,
                             generate_button,
                             load_parameter_button] + lora_ctrls

        ctrls_dict = {
            'generate_image_grid': generate_image_grid,
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'style_selections': style_selections,
            'aspect_ratios_selection': aspect_ratios_selection,
            'image_number': image_number,
            'output_format': output_format,
            'image_seed': image_seed,
            'sharpness': sharpness,
            'guidance_scale': guidance_scale,
            'base_model': base_model,
            'vae_model': vae_model,
            'clip_model': clip_model,
        }

        for i in range(modules.config.default_max_lora_number):
            ctrls_dict[f'lora_{i}_enabled'] = lora_ctrls[i * 3]
            ctrls_dict[f'lora_{i}_model'] = lora_ctrls[i * 3 + 1]
            ctrls_dict[f'lora_{i}_weight'] = lora_ctrls[i * 3 + 2]

        ctrls_dict.update({
            'input_image_checkbox': input_image_checkbox,
            'current_tab': current_tab,
            'uov_method': uov_method,
            'uov_input_image': uov_input_image_path,
            'upscale_model': upscale_model,
            'upscale_scale_override': upscale_scale_override,
            'upscale_prompt': upscale_prompt,
            'upscale_gan_output_image': upscale_gan_output_path,
            'upscale_gan_tile_size': upscale_gan_tile_size,
            'upscale_refinement_denoise': upscale_refinement_denoise,
            'upscale_refinement_tile_overlap': upscale_refinement_tile_overlap,
            'outpaint_selections': outpaint_selections,
            'outpaint_input_image': outpaint_input_image_path,
            'outpaint_mask_image': outpaint_mask_image_path,
            'outpaint_additional_prompt': outpaint_additional_prompt,
            'inpaint_input_image': inpaint_input_image_path,
            'inpaint_context_mask_image': inpaint_context_mask_image_path,
            'inpaint_additional_prompt': inpaint_additional_prompt,
            'inpaint_mask_image': inpaint_mask_image_path,
            'inpaint_bb_image': inpaint_bb_image_path,
            'disable_preview': disable_preview,
            'preview_update_interval': preview_update_interval,
            'disable_intermediate_results': disable_intermediate_results,
            'disable_seed_increment': disable_seed_increment,
            'prefetch_depth': prefetch_depth,
            'prefetch_chunk_mb': prefetch_chunk_mb,
            'flux_fill_runtime_posture': flux_fill_runtime_posture,
            'flux_fill_t5_posture': flux_fill_t5_posture,
            'flux_fill_disk_paged_t5_gc_interval': flux_fill_disk_paged_t5_gc_interval,
            'sdxl_assembly_posture': sdxl_assembly_posture,
            'adm_scaler_positive': adm_scaler_positive,
            'adm_scaler_negative': adm_scaler_negative,
            'adm_scaler_end': adm_scaler_end,
            'adaptive_cfg': adaptive_cfg,
            'clip_skip': clip_skip,
            'sampler_name': sampler_name,
            'scheduler_name': scheduler_name,
            'overwrite_width': overwrite_width,
            'overwrite_height': overwrite_height,
            'overwrite_upscale_strength': overwrite_upscale_strength,
            'mixing_image_prompt_and_inpaint': mixing_image_prompt_and_inpaint,
            'mixing_image_prompt_and_outpaint': mixing_image_prompt_and_outpaint,
            'skipping_cn_preprocessor': skipping_cn_preprocessor,
            'canny_low_threshold': canny_low_threshold,
            'canny_high_threshold': canny_high_threshold,
            'controlnet_softness': controlnet_softness,

            # inpaint_ctrls
            'debugging_inpaint_preprocessor': debugging_inpaint_preprocessor,
            'inpaint_disable_initial_latent': inpaint_disable_initial_latent,
            'inpaint_engine': inpaint_engine,
            'inpaint_route': inpaint_route,
            'inpaint_strength': inpaint_strength,
            'inpaint_erode_or_dilate': inpaint_erode_or_dilate,
            'inpaint_step2_checkbox': inpaint_step2_checkbox,
            'steps': steps,

            # outpaint_ctrls
            'outpaint_engine': outpaint_engine,
            'outpaint_strength': outpaint_strength,
            'inpaint_outpaint_expansion_size': inpaint_outpaint_expansion_size,
            'outpaint_step2_checkbox': outpaint_step2_checkbox,
            'outpaint_bb_image': outpaint_bb_image_path,
            'outpaint_bb_mask_data': outpaint_bb_mask_data,
            'remove_base_image': remove_base_image_path,
            'remove_prompt': remove_prompt,
            'remove_mask_image': remove_mask_image_path,
            'remove_mask_data': remove_mask_data,
            'remove_bg_enabled': remove_bg_enabled,
            'remove_obj_enabled': remove_obj_enabled,
            'objr_engine': objr_engine,
            'flux_fill_conditioning': flux_fill_conditioning,
            'flux_fill_prompt_cache': flux_fill_prompt_cache,
            'objr_mask_dilate': objr_mask_dilate,
            'objr_mask_blur': objr_mask_blur,
            'objr_blend_mode': objr_blend_mode,
            'bgr_threshold': bgr_threshold,
            'bgr_jit': bgr_jit,
        })

        if not args_manager.args.disable_metadata:
            ctrls_dict['save_metadata_to_images'] = save_metadata_to_images
            ctrls_dict['metadata_scheme'] = metadata_scheme

        for i in range(modules.config.default_controlnet_image_count):
            ctrls_dict[f'cn_{i}_image'] = cn_image_paths[i]
            ctrls_dict[f'cn_{i}_stop'] = ip_stops[i]
            ctrls_dict[f'cn_{i}_weight'] = ip_weights[i]
            ctrls_dict[f'cn_{i}_type'] = ip_types[i]

        import modules.parameter_registry as parameter_registry
        parameter_registry.validate_ctrls(ctrls_dict)


        ui_elements = {
            'image_input_panel': image_input_panel,
            'uov_tab': uov_tab,
            'inpaint_tab': inpaint_tab,
            'outpaint_tab': outpaint_tab,
            'ip_tab': ip_tab,
            'metadata_tab': metadata_tab,
            'history_link': history_link,
            'style_selections_accordion': style_selections_accordion,
            'state_is_generating': state_is_generating,
            'stop_button': stop_button,
            'skip_button': skip_button,
            'progress_html': progress_html,
            'preview_column': preview_column,
            'gallery_column': gallery_column,
            'inpaint_toggle_toolbar': inpaint_toggle_toolbar,
            'outpaint_mask_expansion_button': outpaint_mask_expansion_button,
            'example_inpaint_prompts': example_inpaint_prompts,
            'metadata_import_button': metadata_import_button,
            'load_data_outputs': load_data_outputs,
            'inpaint_mask_generation_col': inpaint_mask_generation_col,
            'outpaint_mask_generation_col': outpaint_mask_generation_col,
            'inpaint_bb_image': inpaint_bb_image,
            'ip_ad_cols': ip_ad_cols,
            'ip_types': ip_types,
            'ip_stops': ip_stops,
            'ip_weights': ip_weights,
            'lora_ctrls': lora_ctrls,
            'style_search_bar': style_search_bar,
            'refresh_files': refresh_files,
            'inpaint_engine_state': inpaint_engine_state,
            'model_browser_apply_data': model_browser_apply_data,
            'outpaint_engine_state': outpaint_engine_state,
            'generate_button': generate_button,
            'load_parameter_button': load_parameter_button,
            'metadata_input_image': metadata_input_image,
            'metadata_input_image_path': metadata_input_image_path,
            'metadata_input_workspace_id': metadata_input_workspace_id,
            'metadata_json': metadata_json,
            'inpaint_context_mask_data': inpaint_context_mask_data,
            'inpaint_replace_bb_nonce': inpaint_replace_bb_nonce,
            'inpaint_bb_mask_data': inpaint_bb_mask_data,
            'inpaint_input_image_path': inpaint_input_image_path,
            'inpaint_input_workspace_id': inpaint_input_workspace_id,
            'inpaint_context_mask_image_path': inpaint_context_mask_image_path,
            'inpaint_context_mask_workspace_id': inpaint_context_mask_workspace_id,
            'inpaint_bb_image_path': inpaint_bb_image_path,
            'inpaint_bb_workspace_id': inpaint_bb_workspace_id,
            'inpaint_mask_image_path': inpaint_mask_image_path,
            'inpaint_mask_workspace_id': inpaint_mask_workspace_id,
            'outpaint_bb_mask_data': outpaint_bb_mask_data,
            'remove_mask_data': remove_mask_data,
            'remove_mask_image_path': remove_mask_image_path,
            'remove_mask_workspace_id': remove_mask_workspace_id,
            'outpaint_input_workspace_id': outpaint_input_workspace_id,
            'outpaint_mask_image_path': outpaint_mask_image_path,
            'outpaint_mask_workspace_id': outpaint_mask_workspace_id,
            'outpaint_bb_image_path': outpaint_bb_image_path,
            'outpaint_bb_workspace_id': outpaint_bb_workspace_id,
            'outpaint_prepare_button': outpaint_prepare_button,
            'outpaint_prepare_notice': outpaint_prepare_notice,
            'upscale_refinement_container': upscale_refinement_container,
            'upscale_gan_output_container': upscale_gan_output_container,
            'upscale_scale_info': upscale_scale_info,
            'gallery': gallery,
            'seed_random': seed_random,
            'inpaint_tab': inpaint_tab,
            'outpaint_tab': outpaint_tab,
            'remove_tab': remove_tab,
            'remove_bg_enabled': remove_bg_enabled,
            'remove_obj_enabled': remove_obj_enabled,
            'remove_mask_state': remove_mask_state,
            'queue_tab': queue_tab,
            'current_tasks_state': current_tasks_state,
        }

        if not args_manager.args.disable_preset_selection:
            ui_elements['preset_selection'] = preset_selection

        ui_logic.register_all_events(ctrls_dict, currentTask, ui_elements)


def dump_default_english_config():
    from modules.localization import dump_english_config
    dump_english_config(grh.all_components)


# dump_default_english_config()

# Hijack Gradio's app creation to mount our staging router
import gradio.routes
old_create_app = gradio.routes.App.create_app

@staticmethod
def patched_create_app(*args, **kwargs):
    app = old_create_app(*args, **kwargs)
    from modules.staging_api import staging_router
    from modules.runtime_surface_api import runtime_surface_router
    from modules.monitor_api import monitor_router
    from modules.image_api import image_router
    from modules.model_api import model_router
    app.include_router(staging_router)
    app.include_router(runtime_surface_router)
    app.include_router(monitor_router)
    app.include_router(image_router)
    app.include_router(model_router)
    return app

gradio.routes.App.create_app = patched_create_app

shared.gradio_root.launch(
    inbrowser=args_manager.args.in_browser,
    server_name=args_manager.args.listen,
    server_port=args_manager.args.port,
    share=args_manager.args.share,
    auth=check_auth if (args_manager.args.share or args_manager.args.listen) and auth_enabled else None,
    allowed_paths=[
        modules.config.path_outputs,
        os.path.abspath('javascript'),
        os.path.abspath('css'),
        os.path.abspath('sdxl_styles/samples')
    ],
    blocked_paths=[constants.AUTH_FILENAME]
)
