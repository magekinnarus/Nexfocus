import modules.flags as flags
import modules.meta_parser


def _indent_metadata_lines(text, indent):
    lines = str(text).splitlines() or ['']
    return '\n'.join(f'{indent}{line}' if line else indent.rstrip() for line in lines)


def _format_metadata_scalar(value):
    if value is None:
        return 'None'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return str(value)


def _format_metadata_list(values):
    if not values:
        return '  []'

    lines = []
    for index, value in enumerate(values, start=1):
        rendered = _format_metadata_value(value)
        rendered_lines = rendered.splitlines() or ['']
        lines.append(f'  {index}. {rendered_lines[0]}')
        for line in rendered_lines[1:]:
            lines.append(f'     {line}')
    return '\n'.join(lines)


def _format_metadata_mapping(parameters):
    lines = []
    for key, value in parameters.items():
        if isinstance(value, dict):
            lines.append(f'{key}:')
            lines.append(_indent_metadata_lines(_format_metadata_mapping(value), '  '))
        elif isinstance(value, list):
            lines.append(f'{key}:')
            lines.append(_format_metadata_list(value))
        else:
            rendered = _format_metadata_scalar(value)
            if '\n' in rendered:
                lines.append(f'{key}:')
                lines.append(_indent_metadata_lines(rendered, '  '))
            else:
                lines.append(f'{key}: {rendered}')
    return '\n'.join(lines)


def _format_metadata_value(value):
    if isinstance(value, dict):
        return _format_metadata_mapping(value)
    if isinstance(value, list):
        return _format_metadata_list(value)
    return _format_metadata_scalar(value)


def format_metadata_preview(parameters, metadata_scheme=None):
    if parameters is None or not isinstance(parameters, (dict, list)):
        return 'No metadata found.'

    if isinstance(parameters, list):
        # Convert list of tuples to dict
        parameters = {k: v for _, k, v in parameters}

    if not isinstance(parameters, dict) or not parameters:
        return 'No metadata found.'

    # Convert v1 records via compatibility shim
    if 'metadata_version' not in parameters or parameters.get('metadata_version') == 1:
        parameters = modules.meta_parser.convert_v1_to_v2_metadata(parameters)

    workflow = str(parameters.get('workflow', 'txt2img'))
    version = str(parameters.get('version', ''))
    timestamp = str(parameters.get('timestamp', ''))

    lines = []
    lines.append(f"Workflow: {workflow} (v{parameters.get('metadata_version', 2)})")
    if version:
        lines.append(f"Version: {version}")
    if timestamp:
        lines.append(f"Timestamp: {timestamp}")
    lines.append("")

    hidden_keys = {'sharpness', 'clip_skip', 'adm_guidance', 'adaptive_cfg', 'base_model_hash', 'metadata_version', 'workflow', 'version', 'timestamp', 'created_by'}

    if workflow == 'txt2img':
        deployable_keys = ['prompt', 'negative_prompt', 'styles', 'base_model', 'seed', 'resolution', 'sampler', 'scheduler', 'steps', 'cfg_scale', 'loras']
        display_only_keys = ['vae', 'cn']
    elif workflow == 'inpaint_sdxl':
        deployable_keys = ['prompt', 'inpaint_prompt', 'negative_prompt', 'styles', 'base_model', 'seed', 'sampler', 'scheduler', 'steps', 'cfg_scale', 'loras', 'inpaint_route']
        display_only_keys = ['resolution', 'vae', 'cn']
    elif workflow == 'outpaint_sdxl':
        deployable_keys = ['prompt', 'outpaint_prompt', 'negative_prompt', 'styles', 'base_model', 'seed', 'sampler', 'scheduler', 'steps', 'cfg_scale', 'loras']
        display_only_keys = ['resolution', 'vae', 'cn']
    elif workflow == 'flux_fill_inpaint':
        deployable_keys = ['prompt', 'inpaint_prompt', 'seed', 'steps', 'sampler', 'scheduler']
        display_only_keys = ['t5', 'clip_l', 'ae', 'resolution', 'cn']
    elif workflow == 'flux_fill_remove':
        deployable_keys = ['prompt_description', 'seed', 'steps', 'sampler', 'scheduler']
        display_only_keys = ['t5', 'clip_l', 'ae', 'resolution', 'cn']
    elif workflow == 'super_upscale':
        deployable_keys = ['prompt', 'negative_prompt', 'styles', 'base_model', 'seed', 'sampler', 'scheduler', 'steps', 'cfg_scale', 'loras']
        display_only_keys = ['resolution', 'vae', 'cn']
    elif workflow == 'color_enhance':
        deployable_keys = ['prompt_description', 'base_model', 'seed', 'scheduler', 'steps', 'loras']
        display_only_keys = ['resolution', 'vae', 'cn']
    elif workflow in ['upscale_gan', 'remove_mat', 'remove_objr', 'bgr_subject', 'bgr_mask']:
        deployable_keys = []
        display_only_keys = ['resolution']
    else:
        deployable_keys = [k for k in parameters if k not in hidden_keys]
        display_only_keys = []

    deployable_items = {k: parameters[k] for k in deployable_keys if k in parameters and parameters[k] not in [None, '', [], 'None']}
    if deployable_items:
        lines.append("Deployable Parameters:")
        lines.append(_indent_metadata_lines(_format_metadata_mapping(deployable_items), "  "))
        lines.append("")

    display_items = {k: parameters[k] for k in display_only_keys if k in parameters and parameters[k] not in [None, '', [], 'None']}
    if display_items:
        lines.append("Display-Only Reference:")
        lines.append(_indent_metadata_lines(_format_metadata_mapping(display_items), "  "))

    if not deployable_items and not display_items:
        lines.append("Identity-only workflow record (no parameters).")

    return '\n'.join(lines).strip()
