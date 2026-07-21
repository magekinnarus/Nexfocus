import gradio as gr
import modules.config
import modules.flags as flags
import args_manager
from backend.environment_profile import PROFILE_COLAB_FREE


def resolve_default_sdxl_assembly_posture(profile=None):
    """Return the initial SDXL posture value: 'gpu_text' on Colab Free, 'auto' otherwise."""
    if profile is None:
        profile = getattr(modules.config, 'resolved_memory_environment_profile', None)
    profile_name = str(getattr(profile, 'name', '') or '').strip().lower()
    return 'gpu_text' if profile_name == PROFILE_COLAB_FREE else 'auto'

def build_debug_tab():
    """
    Builds the Advanced tab contents.
    """
    results = {}

    # --- Preview Settings ---
    results['preview_section_label'] = gr.Markdown('### Preview Settings')
    results['preview_update_interval'] = gr.Slider(
        label='Preview Update Every N Steps',
        minimum=1,
        maximum=20,
        step=1,
        value=modules.config.preview_update_interval,
        info='Keeps progress text live every step, but only sends preview images every N steps.'
    )
    results['disable_preview'] = gr.Checkbox(
        label='Disable Preview', value=False,
        info='Disable preview during generation.'
    )

    # --- SDXL Settings ---
    results['sdxl_section_label'] = gr.Markdown('### SDXL Settings')
    results['sdxl_assembly_posture'] = gr.Radio(
        label='SDXL Posture Override',
        choices=[('Default', 'auto'), ('CLIP in GPU', 'gpu_text'), ('Streaming', 'streaming')],
        value=resolve_default_sdxl_assembly_posture(),
        info='Colab Free defaults to CLIP in GPU, which pins both resident UNet and CLIP to GPU and requires at least 10GB VRAM. Other profiles default to Default. Use Streaming to force the streaming lane on roomier hardware.'
    )

    # --- Flux Fill Settings ---
    results['flux_section_label'] = gr.Markdown('### Flux Fill Settings')
    results['flux_fill_runtime_posture'] = gr.Radio(
        label='Flux Fill Posture Override',
        choices=[('Default', 'auto'), ('Streaming', 'streaming')],
        value='auto',
        info='Debug/benchmark override. Keep Default for normal use; force Streaming only when benchmarking Flux Fill on high-RAM Colab sessions.'
    )

    total_ram_gb = 0.0
    try:
        from backend.flux_fill_v3.activation import resolve_flux_fill_total_ram_gb
        total_ram_gb = resolve_flux_fill_total_ram_gb()
    except Exception:
        pass

    results['flux_fill_t5_posture'] = gr.Radio(
        label='T5 Posture Override',
        choices=['disk_paged', 'cpu_resident'],
        value='disk_paged',
        visible=(total_ram_gb >= 31.0),
        info='Advanced opt-in. Default is disk_paged (lowest RAM overhead). cpu_resident requires minimum 32 GB RAM (or 45 GB+ RAM if combined with streaming UNet).'
    )

    results['prefetch_depth'] = gr.Radio(
        label='Flux Fill Prefetch Depth', choices=[('1 (default)', 1), ('2', 2)],
        value=1,
        info='Async scheduler prefetch depth for Flux Fill (1 or 2).'
    )

    results['prefetch_chunk_mb'] = gr.Radio(
        label='Flux Fill Prefetch Chunk Size', choices=[('64 (default)', 64), ('128', 128)],
        value=64,
        info='Max size of chunks to prefetch for Flux Fill (64MB or 128MB).'
    )

    results['flux_fill_disk_paged_t5_gc_interval'] = gr.Radio(
        label='T5 Host-RAM Cleanup Cadence',
        choices=['auto', '8', '16'],
        value='auto',
        info='Controls disk-paged T5 prompt encoding garbage collection. Auto collects every 4 blocks while selecting 8 or 16 reduces cleanup overhead while increasing memory pressure.'
    )

    # --- Image & Metadata ---
    results['param_section_label'] = gr.Markdown('### Parameter Settings')
    results['sharpness'] = gr.Slider(
        label='Image Sharpness', minimum=0.0, maximum=30.0, step=0.01,
        value=modules.config.default_sample_sharpness,
        info='Higher value means sharper edges.'
    )

    if not args_manager.args.disable_metadata:
        results['save_metadata_to_images'] = gr.Checkbox(
            label='Save Metadata to Images',
            value=modules.config.default_save_metadata_to_images,
            info='Adds parameters to generated images allowing manual regeneration.'
        )
        filtered_schemes = [c for c in flags.metadata_scheme if c[1] != 'fooocus']
        results['metadata_scheme'] = gr.Radio(
            label='Metadata Scheme',
            choices=filtered_schemes,
            value=modules.config.default_metadata_scheme,
            info='Image Prompt parameters are not included. Use png and a1111 for compatibility with Civitai.',
            visible=modules.config.default_save_metadata_to_images
        )

        results['save_metadata_to_images'].change(
            lambda x: gr.update(visible=x),
            inputs=[results['save_metadata_to_images']],
            outputs=[results['metadata_scheme']],
            queue=False,
            show_progress=False
        )

    # --- Advanced Sampling ---
    results['adm_scaler_positive'] = gr.Slider(
        label='Positive ADM Guidance Scaler', minimum=0.1, maximum=3.0, step=0.001,
        value=1.5,
        info='The scaler multiplied to positive ADM (use 1.0 to disable).'
    )
    results['adm_scaler_negative'] = gr.Slider(
        label='Negative ADM Guidance Scaler', minimum=0.1, maximum=3.0, step=0.001,
        value=0.8,
        info='The scaler multiplied to negative ADM (use 1.0 to disable).'
    )
    results['adm_scaler_end'] = gr.Slider(
        label='ADM Guidance End At Step', minimum=0.0, maximum=1.0, step=0.001,
        value=0.3,
        info='When to end the guidance from positive/negative ADM.'
    )

    results['adaptive_cfg'] = gr.Slider(
        label='CFG Mimicking from TSNR', minimum=1.0, maximum=30.0, step=0.01,
        value=modules.config.default_cfg_tsnr,
        info='Enabling Fooocus\'s implementation of CFG mimicking for TSNR (effective when real CFG > mimicked CFG).'
    )

    results['output_format'] = gr.Radio(
        label='Output Format',
        choices=flags.OutputFormat.list(),
        value=modules.config.default_output_format
    )

    return results

