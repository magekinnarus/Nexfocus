import gradio as gr
import args_manager
import modules.config
import modules.flags as flags

def build_settings_tab():
    """
    Builds the Settings tab contents: preset, performance, aspect ratios,
    image number, output format, negative prompt, seed, and history link.
    
    Returns:
        dict: Gradio components mapping name to instance.
    """
    results = {}

    active_base_model_name = modules.config.coerce_active_base_model_selection(modules.config.default_base_model_name)
    default_aspect_ratio_labels = modules.config.get_aspect_ratio_labels_for_model(active_base_model_name)
    default_aspect_ratio_label = modules.config.default_aspect_ratio
    if default_aspect_ratio_label not in default_aspect_ratio_labels:
        default_aspect_ratio_label = modules.config.get_default_aspect_ratio_label_for_model(active_base_model_name)

    if not args_manager.args.disable_preset_selection:
        results['preset_selection'] = gr.Dropdown(
            label='Preset',
            choices=modules.config.available_presets,
            value=args_manager.args.preset if args_manager.args.preset else "initial",
            interactive=True
        )

    with gr.Accordion(label='Aspect Ratios', open=False, elem_id='aspect_ratios_accordion') as aspect_ratios_accordion:
        results['aspect_ratios_accordion'] = aspect_ratios_accordion
        results['aspect_ratios_selection'] = gr.Radio(
            label='Aspect Ratios',
            show_label=False,
            choices=default_aspect_ratio_labels,
            value=default_aspect_ratio_label,
            info='width × height',
            elem_classes='aspect_ratios',
            elem_id='aspect_ratios_selection'
        )

    results['steps'] = gr.Slider(
        label='Sampling Steps',
        minimum=1, maximum=200, step=1,
        value=modules.config.default_overwrite_step,
        info='Number of sampling steps.'
    )

    results['sampler_name'] = gr.Dropdown(
        label='Sampler', choices=flags.sampler_list,
        value=modules.config.default_sampler
    )

    results['scheduler_name'] = gr.Dropdown(
        label='Scheduler', choices=flags.scheduler_list,
        value=modules.config.default_scheduler
    )

    results['guidance_scale'] = gr.Slider(
        label='Guidance Scale', minimum=1.0, maximum=30.0, step=0.01,
        value=modules.config.default_cfg_scale,
        info='Higher value means following prompt more strictly.'
    )

    results['clip_skip'] = gr.Slider(
        label='CLIP Skip', minimum=1, maximum=flags.clip_skip_max, step=1,
        value=modules.config.default_clip_skip,
        info='Bypass CLIP layers to avoid overfitting (2 is recommended).'
    )


    results['negative_prompt'] = gr.Textbox(
        label='Negative Prompt',
        show_label=True,
        placeholder="Type prompt here.",
        info='Describing what you do not want to see.',
        lines=2,
        elem_id='negative_prompt',
        value=modules.config.default_prompt_negative
    )

    results['seed_random'] = gr.Checkbox(label='Random', value=True)
    results['image_seed'] = gr.Textbox(label='Seed', value=0, max_lines=1, visible=False)
    results['history_link'] = gr.HTML()

    return results
