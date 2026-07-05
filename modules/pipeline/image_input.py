import numpy as np
import hashlib
from collections import OrderedDict
import modules.config as config
import modules.core as core
import modules.flags as flags
import extras.preprocessors as preprocessors
import backend.ip_adapter as contextual_ip_adapter
import backend.preprocessors as structural_preprocessors
import backend.pulid_runtime as pulid_runtime
import backend.resources as resources
from modules.route_intent import resolve_route_intent
from modules.util import (HWC3, resize_image, get_image_shape_ceil, set_image_shape_ceil, 
                          get_shape_ceil, resample_image, erode_or_dilate)
from modules.upscaler import perform_upscale
import modules.mask_processing as mask_proc

_STRUCTURAL_PREPROCESS_CACHE: OrderedDict[tuple, np.ndarray] = OrderedDict()
_STRUCTURAL_PREPROCESS_CACHE_LIMIT = 8


class EarlyReturnException(BaseException):
    def __init__(self, payload=None):
        super().__init__()
        self.payload = payload


def _reset_preprocessor_metrics_once(task_state):
    if getattr(task_state, "_nex_preprocessor_metrics_started", False):
        return
    from backend.sdxl_unified_runtime import clear_preprocessor_metrics

    clear_preprocessor_metrics()
    setattr(task_state, "_nex_preprocessor_metrics_started", True)


def prepare_flux_inpaint_context(task_state, inpaint_image, inpaint_mask):
    """
    Build the prepared inpaint context without running SDXL VAE encode.

    Flux Fill uses the same prepared image/mask inputs but consumes them
    through the Flux session instead of the SDXL inpaint diffusion path.
    """
    denoising_strength = getattr(task_state, 'inpaint_strength', 1.0)

    raw_input_image = getattr(task_state, 'inpaint_input_image', None)
    raw_context_mask = getattr(task_state, 'inpaint_context_mask_image', None)
    raw_bb_image = getattr(task_state, 'inpaint_bb_image', None)
    raw_bb_mask = getattr(task_state, 'inpaint_mask_image', None)

    input_image = mask_proc.unpack_gradio_data(raw_input_image) if raw_input_image is not None else inpaint_image
    prepared_context_mask = getattr(task_state, 'context_mask', None)
    if raw_context_mask is not None:
        context_mask = mask_proc.unpack_gradio_data(raw_context_mask)
    elif prepared_context_mask is not None:
        context_mask = prepared_context_mask
    else:
        context_mask = inpaint_mask
    if context_mask is not None:
        context_mask = mask_proc.to_binary_mask(mask_proc.ensure_numpy(context_mask))

    bb_img_data = mask_proc.unpack_gradio_data(raw_bb_image)
    if bb_img_data is not None:
        bb_img_data = HWC3(bb_img_data)
    bb_mask_2d = mask_proc.combine_masks(
        mask_proc.unpack_gradio_data(raw_bb_mask),
        mask_proc.extract_mask_from_layers(raw_bb_image) if isinstance(raw_bb_image, dict) else None,
    )

    from modules.pipeline.inpaint import InpaintPipeline

    inpaint = InpaintPipeline()
    if input_image is not None and context_mask is not None:
        ctx = inpaint.prepare(image=input_image, mask=context_mask, extend_factor=1.2, generate_blend_mask=False)
    else:
        ctx = inpaint.prepare(image=inpaint_image, mask=inpaint_mask, extend_factor=1.2, generate_blend_mask=False)

    if bb_img_data is not None:
        # Flux Inpaint reuses the SDXL bucket selected by prepare() so BB edits
        # still resolve to the same native inference canvas.
        ctx.bb_image = bb_img_data

    if bb_mask_2d is not None:
        ctx.bb_mask = bb_mask_2d

    if ctx.bb_image is None:
        raise ValueError('Inpaint BB image is required before inference')
    if ctx.bb_mask is None:
        raise ValueError('Inpaint BB mask is required before inference')

    ctx.bb_image = HWC3(mask_proc.ensure_numpy(ctx.bb_image))
    ctx.bb_mask = mask_proc.to_binary_mask(mask_proc.ensure_numpy(ctx.bb_mask))
    ctx.bb_image = resample_image(ctx.bb_image, width=ctx.target_w, height=ctx.target_h)
    ctx.bb_mask = resample_image(ctx.bb_mask, width=ctx.target_w, height=ctx.target_h)

    y1, y2, x1, x2 = ctx.bb
    full_mask = np.zeros_like(ctx.original_image[:, :, 0])
    H, W = full_mask.shape
    patch_mask_resized = resample_image(ctx.bb_mask, width=x2 - x1, height=y2 - y1)

    iy1, iy2 = max(0, y1), min(H, y2)
    ix1, ix2 = max(0, x1), min(W, x2)
    cy1, cy2 = iy1 - y1, iy2 - y1
    cx1, cx2 = ix1 - x1, ix2 - x1

    full_mask[iy1:iy2, ix1:ix2] = patch_mask_resized[cy1:cy2, cx1:cx2]
    ctx.blend_mask = inpaint._morphological_open(full_mask)

    task_state.inpaint_context = ctx
    task_state.width = ctx.target_w
    task_state.height = ctx.target_h
    task_state.initial_latent = None
    task_state.denoising_strength = denoising_strength
    return ctx




def apply_outpaint_inference_setup(task_state, inpaint_image, inpaint_mask, 
                                  progressbar_callback=None, yield_result_callback=None):
    """
    Sets up the outpainting worker, patches the UNet, and encodes the initial latent.
    Exclusively using OutpaintPipeline.
    """
    from modules.pipeline.outpaint import OutpaintPipeline
    outpaint = OutpaintPipeline()
    
    # Use UI outpaint_strength (default 1.0, user usually lowers it for sketching)
    denoising_strength = getattr(task_state, 'outpaint_strength', 1.0)
        
    outpaint_direction = getattr(task_state, 'outpaint_direction', None)
    if isinstance(outpaint_direction, list) and len(outpaint_direction) > 0:
        outpaint_direction = outpaint_direction[0].lower()
        
    ctx = outpaint.prepare(
        image=inpaint_image,
        mask=inpaint_mask,
        outpaint_direction=outpaint_direction,
        extend_factor=1.2,
        generate_blend_mask=False
    )
    
    # --- Resolved BB Image and BB Mask Support ---
    # If the user has provided an edited BB image or a manual BB mask, override the context.
    import modules.mask_processing as mask_proc
    
    raw_bb_image = getattr(task_state, 'outpaint_bb_image', None)
    raw_bb_mask = getattr(task_state, 'outpaint_mask_image', None) # Upload slot
    # Hidden mask field from brush drawing on BB image
    brush_mask_data = getattr(task_state, 'outpaint_bb_mask_data', '')
    
    bb_img_data = mask_proc.unpack_gradio_data(raw_bb_image)
    if bb_img_data is not None:
        ctx.bb_image = bb_img_data
        ctx.target_h, ctx.target_w = bb_img_data.shape[:2]
        
    # Combine uploaded mask with brush-drawn mask
    manual_mask = mask_proc.unpack_gradio_data(raw_bb_mask)
    brush_mask = mask_proc.unpack_gradio_data(brush_mask_data)
    combined_bb_mask = mask_proc.combine_masks(manual_mask, brush_mask)
    if combined_bb_mask is not None:
        # Ensure mask matches BB image resolution
        combined_bb_mask = resample_image(combined_bb_mask, width=ctx.target_w, height=ctx.target_h)
        ctx.bb_mask = combined_bb_mask
        ctx.bb_image = outpaint.pixelate_mask_area(ctx.bb_image, combined_bb_mask)
    # Rebuild the full-image blend mask from the final BB mask so stitch-back
    # follows the user-edited Outpaint mask instead of the earlier auto mask.
    y1, y2, x1, x2 = ctx.bb
    full_mask = np.zeros_like(ctx.original_image[:, :, 0])
    H, W = full_mask.shape
    patch_mask_resized = resample_image(ctx.bb_mask, width=x2-x1, height=y2-y1)

    iy1, iy2 = max(0, y1), min(H, y2)
    ix1, ix2 = max(0, x1), min(W, x2)
    cy1, cy2 = iy1 - y1, iy2 - y1
    cx1, cx2 = ix1 - x1, ix2 - x1

    full_mask[iy1:iy2, ix1:ix2] = patch_mask_resized[cy1:cy2, cx1:cx2]
    ctx.blend_mask = outpaint._morphological_open(full_mask)

    
    task_state.inpaint_context = ctx
    task_state.width = ctx.target_w
    task_state.height = ctx.target_h
    task_state.initial_latent = None
    task_state.denoising_strength = denoising_strength
    
    final_height, final_width = ctx.original_image.shape[:2]
    print(f'Outpaint setup: BB resolution {ctx.target_w}x{ctx.target_h}, Original resolution {final_width}x{final_height}.')


def apply_inpaint(task_state, inpaint_image, inpaint_mask, 
                  progressbar_callback=None, yield_result_callback=None):
    """
    Resolves the required inpaint assets, patches the UNet, and encodes the initial latent.
    Inference always runs from the resolved Full Image, Context Mask, BB Image, and BB Mask set.
    """
    denoising_strength = getattr(task_state, 'inpaint_strength', 1.0)

    raw_input_image = getattr(task_state, 'inpaint_input_image', None)
    raw_context_mask = getattr(task_state, 'inpaint_context_mask_image', None)
    raw_bb_image = getattr(task_state, 'inpaint_bb_image', None)
    raw_bb_mask = getattr(task_state, 'inpaint_mask_image', None)

    input_image = mask_proc.unpack_gradio_data(raw_input_image) if raw_input_image is not None else inpaint_image
    prepared_context_mask = getattr(task_state, 'context_mask', None)
    if raw_context_mask is not None:
        context_mask = mask_proc.unpack_gradio_data(raw_context_mask)
    elif prepared_context_mask is not None:
        context_mask = prepared_context_mask
    else:
        context_mask = inpaint_mask
    if context_mask is not None:
        context_mask = mask_proc.to_binary_mask(mask_proc.ensure_numpy(context_mask))

    bb_img_data = mask_proc.unpack_gradio_data(raw_bb_image)
    if bb_img_data is not None:
        bb_img_data = HWC3(bb_img_data)
    bb_mask_2d = mask_proc.combine_masks(
        mask_proc.unpack_gradio_data(raw_bb_mask),
        mask_proc.extract_mask_from_layers(raw_bb_image) if isinstance(raw_bb_image, dict) else None
    )

    from modules.pipeline.inpaint import InpaintPipeline
    inpaint = InpaintPipeline()

    if input_image is not None and context_mask is not None:
        ctx = inpaint.prepare(
            image=input_image,
            mask=context_mask,
            extend_factor=1.2,
            generate_blend_mask=False
        )
        print(f"[Debug] Context derived from {input_image.shape[1]}x{input_image.shape[0]} image via context mask.")
    else:
        ctx = inpaint.prepare(
            image=inpaint_image,
            mask=inpaint_mask,
            extend_factor=1.2,
            generate_blend_mask=False
        )
        print(f"[Debug] Context derived from standard inpaint inputs.")

    if bb_img_data is not None:
        ctx.bb_image = bb_img_data
        ctx.target_h, ctx.target_w = bb_img_data.shape[:2]
        print(f"[Debug] Using resolved BB image: {ctx.target_w}x{ctx.target_h}")

    if bb_mask_2d is not None:
        ctx.bb_mask = bb_mask_2d
        print(f"[Debug] Using resolved BB mask.")

    if ctx.bb_image is None:
        raise ValueError('Inpaint BB image is required before inference')
    if ctx.bb_mask is None:
        raise ValueError('Inpaint BB mask is required before inference')

    ctx.bb_image = HWC3(mask_proc.ensure_numpy(ctx.bb_image))
    ctx.bb_mask = mask_proc.to_binary_mask(mask_proc.ensure_numpy(ctx.bb_mask))
    ctx.bb_image = resample_image(ctx.bb_image, width=ctx.target_w, height=ctx.target_h)
    ctx.bb_mask = resample_image(ctx.bb_mask, width=ctx.target_w, height=ctx.target_h)

    y1, y2, x1, x2 = ctx.bb
    full_mask = np.zeros_like(ctx.original_image[:, :, 0])
    H, W = full_mask.shape
    patch_mask_resized = resample_image(ctx.bb_mask, width=x2-x1, height=y2-y1)

    iy1, iy2 = max(0, y1), min(H, y2)
    ix1, ix2 = max(0, x1), min(W, x2)
    cy1, cy2 = iy1 - y1, iy2 - y1
    cx1, cx2 = ix1 - x1, ix2 - x1

    full_mask[iy1:iy2, ix1:ix2] = patch_mask_resized[cy1:cy2, cx1:cx2]
    ctx.blend_mask = inpaint._morphological_open(full_mask)

    task_state.width = ctx.target_w
    task_state.height = ctx.target_h

    if getattr(task_state, 'debugging_inpaint_preprocessor', False):
        if yield_result_callback:
            yield_result_callback(task_state, [ctx.bb_image, ctx.bb_mask], 100, do_not_show_finished_images=True)
        raise EarlyReturnException

    task_state.inpaint_context = ctx
    task_state.width = ctx.target_w
    task_state.height = ctx.target_h
    task_state.initial_latent = None
    task_state.denoising_strength = denoising_strength

    final_height, final_width = ctx.original_image.shape[:2]
    print(f'Inpaint setup: BB resolution {ctx.target_w}x{ctx.target_h}, Original resolution {final_width}x{final_height}.')


def apply_upscale(task_state, progressbar_callback=None):
    """
    Performs image upscaling and sets up the latent for the diffusion pass.
    """
    uov_input_image = task_state.uov_input_image
    uov_method = task_state.uov_method.lower()
    uov_input_image = mask_proc.ensure_numpy(uov_input_image)
    H, W, C = uov_input_image.shape
    
    if progressbar_callback:
        task_state.current_progress += 1
        progressbar_callback(task_state, task_state.current_progress, f'Upscaling image from {str((W, H))} ...')
    
    from backend import resources
    from modules.pipeline.tiled_refinement import should_retain_sdxl_warm_state

    # Calculate retention flag to check if the next stage (tiled refinement) can reuse active SDXL models
    retain_warm = should_retain_sdxl_warm_state(task_state)

    # Pre-upscale cleanup: central governor path before bringing the GAN model online.
    resources.cleanup_memory(
        'upscale_preflight',
        unload_models=not retain_warm,
        force_cache=True,
        trim_host=True,
        target_phase=resources.MemoryPhase.UPSCALE,
        notes={'uov_method': uov_method},
    )

    # 1. GAN Upscale with new multi-model engine
    from modules.upscaler import perform_upscale, clear_model_cache

    # Super-Upscale should use the lightest default model (Nomos2) to save memory
    upscale_model_to_use = task_state.upscale_model
    if uov_method == 'super-upscale':
        upscale_model_to_use = '4xNomos2_otf_esrgan.pth'
        print(f'Super-Upscale detected: Forcing light model {upscale_model_to_use} for initial pass.')

    uov_input_image = perform_upscale(
        uov_input_image, 
        model_name=upscale_model_to_use if upscale_model_to_use != "None" else None,
        scale_override=task_state.upscale_scale_override if task_state.upscale_scale_override > 0 else None,
        retain_warm=retain_warm
    )
    print(f'Image upscaled via GAN to {str(uov_input_image.shape[:2])}.')

    # Post-upscale cleanup: Purge GAN model and route cleanup through the governor.
    clear_model_cache()
    resources.cleanup_memory(
        'upscale_postflight',
        unload_models=not retain_warm,
        force_cache=True,
        trim_host=False,
        notes={'uov_method': uov_method},
        target_phase=resources.MemoryPhase.FINALIZE
    )

    # 2. Handle "Upscale" (Light) or "Super-Upscale" (Stage 1)
    if uov_method == 'upscale':
        task_state.uov_input_image = uov_input_image
        task_state.width = uov_input_image.shape[1]
        task_state.height = uov_input_image.shape[0]
        print('Upscale (Light) completed.')
        return True

    if uov_method == 'super-upscale':
        # Prepare for sequential tiled refinement. NO VAE encode here to save VRAM.
        task_state.uov_input_image = uov_input_image
        task_state.width = uov_input_image.shape[1]
        task_state.height = uov_input_image.shape[0]
        print('Super-Upscale Stage 1 (GAN) completed. Passing to Tiled Refinement.')
        return False # False triggers refinement in worker


def prepare_upscale(task_state, progressbar_callback=None):
    """
    Determines if upscale is needed and sets the appropriate goals.
    """
    task_state.uov_input_image = HWC3(mask_proc.ensure_numpy(task_state.uov_input_image))
    uov_method = task_state.uov_method.lower()
    
    skip_prompt_processing = False
    if 'upscale' in uov_method:
        task_state.goals.append('upscale')
        
        # Validate selected model exists (if not "None")
        if task_state.upscale_model != "None":
            from modules.upscaler import list_available_models
            available = list_available_models()
            if task_state.upscale_model not in available:
                print(f"[Warning] Selected upscale model {task_state.upscale_model} not found. Fallback will be used.")
        
        if uov_method == 'upscale':
            skip_prompt_processing = True
            task_state.steps = 0
            # Note: bypass_alignment is implicit since skip_prompt_processing avoids SDXL specific steps
        else: # Super-Upscale
             # Use the current steps from state (user can still tweak them in Settings)
             # But for UoV it usually defaults to something reasonable.
             pass
    
    return skip_prompt_processing


def apply_image_input(task_state: 'TaskState', base_model_additional_loras, progressbar_callback=None):
    """
    Orchestrates the image input stage, handling UoV, Inpaint/Outpaint, and Image Prompt goals.
    """
    inpaint_image = None
    inpaint_mask = None
    outpaint_image = None
    outpaint_mask = None
    inpaint_patch_model_path = None
    controlnet_paths = {}
    structural_preprocessor_paths = {}
    contextual_assets = {
        'clip_vision_path': None,
        'ip_negative_path': None,
        'contextual_model_paths': {},
        'insightface_model_names': [],
        'eva_clip_path': None,
    }
    skip_prompt_processing = False
    intent = resolve_route_intent(task_state)
    use_flux_fill_inpaint = intent.wants_flux_inpaint

    # UoV handling
    if intent.wants_upscale:
        skip_prompt_processing = prepare_upscale(task_state, progressbar_callback)

    # Outpaint UI Parsing & setup
    if intent.wants_outpaint and task_state.outpaint_input_image is not None:
        if isinstance(task_state.outpaint_input_image, dict):
            if 'background' in task_state.outpaint_input_image:
                outpaint_image = HWC3(mask_proc.ensure_numpy(task_state.outpaint_input_image['background']))
                outpaint_mask = mask_proc.extract_mask_from_layers(task_state.outpaint_input_image)
            else:
                outpaint_image = HWC3(mask_proc.ensure_numpy(task_state.outpaint_input_image['image']))
                raw_mask = task_state.outpaint_input_image.get('mask')
                outpaint_mask = mask_proc.to_binary_mask(mask_proc.ensure_numpy(raw_mask))
        else:
            outpaint_image = HWC3(mask_proc.ensure_numpy(task_state.outpaint_input_image))
            outpaint_mask = None

        if outpaint_mask is None:
            outpaint_mask = np.zeros(outpaint_image.shape[:2], dtype=np.uint8)

        merged_upload = mask_proc.combine_image_and_mask(task_state.outpaint_mask_image)
        if merged_upload is not None:
            H, W, C = outpaint_image.shape
            upload_mask = mask_proc.to_binary_mask(resample_image(merged_upload, width=W, height=H))
            outpaint_mask = mask_proc.combine_masks(outpaint_mask, upload_mask)

        if len(task_state.outpaint_selections) > 0:
            task_state.outpaint_direction = task_state.outpaint_selections[0].lower()

        task_state.inpaint_pixelate_primer = False
        task_state.goals.append('outpaint')

    # Inpaint UI Parsing & setup
    elif intent.wants_inpaint and task_state.inpaint_input_image is not None:
        if isinstance(task_state.inpaint_input_image, dict):
            if 'background' in task_state.inpaint_input_image:
                inpaint_image = mask_proc.ensure_numpy(task_state.inpaint_input_image['background'])
            else:
                inpaint_image = mask_proc.ensure_numpy(task_state.inpaint_input_image['image'])
        else:
            inpaint_image = mask_proc.ensure_numpy(task_state.inpaint_input_image)

        inpaint_mask = np.zeros(inpaint_image.shape[:2], dtype=np.uint8)
        context_mask = np.zeros(inpaint_image.shape[:2], dtype=np.uint8)

        task_state.context_mask = context_mask

        if not getattr(task_state, 'inpaint_step2_checkbox', False):
            merged_upload = mask_proc.combine_image_and_mask(task_state.inpaint_mask_image)
            if merged_upload is not None:
                H, W, C = inpaint_image.shape
                merged_upload = resample_image(merged_upload, width=W, height=H)
                upload_mask = mask_proc.to_binary_mask(merged_upload)
                task_state.context_mask = mask_proc.combine_masks(task_state.context_mask, upload_mask)

        if int(task_state.inpaint_erode_or_dilate) != 0:
            inpaint_mask = erode_or_dilate(inpaint_mask, task_state.inpaint_erode_or_dilate)

        inpaint_image = HWC3(inpaint_image)
        task_state.goals.append('inpaint')

    if use_flux_fill_inpaint:
        skip_prompt_processing = True

    if ('inpaint' in task_state.goals or 'outpaint' in task_state.goals) and not skip_prompt_processing:
        working_image = outpaint_image if 'outpaint' in task_state.goals else inpaint_image
        working_mask = outpaint_mask if 'outpaint' in task_state.goals else inpaint_mask

        if isinstance(working_image, np.ndarray) and isinstance(working_mask, np.ndarray):
            if progressbar_callback:
                progressbar_callback(task_state, 1, 'Initializing inpainter ...')

            engine = getattr(task_state, 'outpaint_engine', 'None') if 'outpaint' in task_state.goals \
                else getattr(task_state, 'inpaint_engine', 'None')
            engine = flags.normalize_inpaint_engine_version(engine, default=flags.INPAINT_ENGINE_NONE)

            if engine != flags.INPAINT_ENGINE_NONE:
                if progressbar_callback:
                    progressbar_callback(task_state, 1, 'Downloading inpainter ...')
                inpaint_patch_model_path = config.downloading_inpaint_models(engine)
                if (inpaint_patch_model_path, 1.0) not in base_model_additional_loras:
                    base_model_additional_loras += [(inpaint_patch_model_path, 1.0)]
                print(f'[Inpaint] Current inpaint model is {inpaint_patch_model_path}')
            else:
                inpaint_patch_model_path = None

    # ControlNet handling
    if intent.expects_controlnet:
        task_state.goals.append('cn')
        if progressbar_callback:
            progressbar_callback(task_state, 1, 'Downloading control models ...')

        structural_tasks = task_state.get_cn_tasks_for_channel(flags.cn_structural)
        contextual_tasks = task_state.get_cn_tasks_for_channel(flags.cn_contextual)

        from modules import model_registry

        for cn_type in flags.cn_structural_types:
            if len(structural_tasks.get(cn_type, [])) == 0:
                continue

            controlnet_asset_id = structural_preprocessors.STRUCTURAL_CONTROLNET_ASSETS.get(cn_type)
            if controlnet_asset_id is not None:
                controlnet_paths[cn_type] = model_registry.ensure_asset(controlnet_asset_id)

            if not task_state.skipping_cn_preprocessor:
                preprocessor_asset_id = structural_preprocessors.STRUCTURAL_PREPROCESSOR_ASSETS.get(cn_type)
                if preprocessor_asset_id is not None:
                    structural_preprocessor_paths[cn_type] = model_registry.ensure_asset(preprocessor_asset_id)

        if any(len(contextual_tasks.get(cn_type, [])) > 0 for cn_type in flags.cn_contextual_types):
            if len(contextual_tasks.get(flags.cn_ip, [])) > 0:
                contextual_assets['clip_vision_path'] = model_registry.ensure_asset('contextual.shared.clip_vision')

            if len(contextual_tasks.get(flags.cn_ip, [])) > 0:
                contextual_assets['ip_negative_path'] = model_registry.ensure_asset('contextual.shared.ip_negative')
                contextual_assets['contextual_model_paths'][flags.cn_ip] = model_registry.ensure_asset('contextual.image_prompt.adapter')

            if len(contextual_tasks.get(flags.cn_pulid, [])) > 0:
                contextual_assets['contextual_model_paths'][flags.cn_pulid] = model_registry.ensure_asset('contextual.pulid.model')
                model_registry.ensure_asset('contextual.insightface.antelopev2')
                contextual_assets['eva_clip_path'] = model_registry.ensure_asset('contextual.pulid.eva_clip')
                if 'antelopev2' not in contextual_assets['insightface_model_names']:
                    contextual_assets['insightface_model_names'].append('antelopev2')

    return {
        'base_model_additional_loras': base_model_additional_loras,
        'clip_vision_path': contextual_assets.get('clip_vision_path'),
        'contextual_assets': contextual_assets,
        'controlnet_paths': controlnet_paths,
        'controlnet_canny_path': controlnet_paths.get(flags.cn_canny),
        'controlnet_cpds_path': controlnet_paths.get(flags.cn_cpds),
        'inpaint_image': inpaint_image,
        'inpaint_mask': inpaint_mask,
        'outpaint_image': outpaint_image,
        'outpaint_mask': outpaint_mask,
        'ip_adapter_path': contextual_assets.get('contextual_model_paths', {}).get(flags.cn_ip),
        'ip_negative_path': contextual_assets.get('ip_negative_path'),
        'skip_prompt_processing': skip_prompt_processing,
        'structural_preprocessor_paths': structural_preprocessor_paths
    }

def load_controlnet_support_models(image_input_result=None):
    image_input_result = image_input_result or {}

    controlnet_paths = image_input_result.get('controlnet_paths') or {}
    requested_controlnets = [path for path in dict.fromkeys(controlnet_paths.values()) if path]
    if requested_controlnets:
        from backend import resources

        resources.trigger_refresh_controlnets(requested_controlnets)
    # Leave contextual support lazy. Their preprocess seams are payload-cache aware,
    # so eager warm-up here would turn warm requests back into cold-looking setup work.


def _unpack_cn_image(raw_img, label):
    cn_img = mask_proc.unpack_gradio_data(raw_img)
    if cn_img is None:
        print(f'[ControlNet] Skipping {label} task with empty or invalid image input.')
        return None
    return HWC3(cn_img)


def _save_structural_preprocessor_output(cn_img, cn_type, slot_index):
    prefix = f"{cn_type.lower().replace(' ', '_')}_slot{slot_index}"
    saved_path = mask_proc.save_to_temp_png(cn_img)
    if saved_path is not None:
        print(f'[ControlNet] Saved {cn_type} preprocessor output to temp: {saved_path}')


def preprocess_structural_controlnets(task_state, structural_preprocessor_paths=None):
    _reset_preprocessor_metrics_once(task_state)
    width, height = task_state.width, task_state.height
    structural_tasks = task_state.get_cn_tasks_for_channel(flags.cn_structural)
    structural_preprocessor_paths = structural_preprocessor_paths or {}

    def preprocess_structural_tasks(cn_type, tasks, processor=None):
        from backend.sdxl_unified_runtime import _PREPROCESSOR_METRICS
        valid_tasks = []
        for slot_index, task in enumerate(tasks, start=1):
            raw_img, cn_stop, cn_weight = task[:3]
            cn_img = _unpack_cn_image(raw_img, cn_type)
            if cn_img is None:
                continue
            cn_img = resize_image(cn_img, width=width, height=height)
            if not task_state.skipping_cn_preprocessor and processor is not None:
                model_path = structural_preprocessor_paths.get(cn_type)
                
                # Fingerprint input image
                source_image_hash = hashlib.sha256(cn_img.tobytes()).hexdigest()
                
                # Parameters
                if cn_type == flags.cn_canny:
                    params = (task_state.canny_low_threshold, task_state.canny_high_threshold)
                else:
                    params = ()
                
                cache_key = (source_image_hash, width, height, cn_type, model_path, params)
                
                cached_res = _STRUCTURAL_PREPROCESS_CACHE.get(cache_key)
                if cached_res is not None:
                    _STRUCTURAL_PREPROCESS_CACHE.move_to_end(cache_key)
                    _PREPROCESSOR_METRICS["structural_hits"] += 1.0
                    cn_img = cached_res.copy()
                else:
                    _PREPROCESSOR_METRICS["structural_misses"] += 1.0
                    try:
                        cn_img = processor(cn_type, cn_img, model_path)
                    except Exception as exc:
                        print(f'[ControlNet] Failed to preprocess {cn_type} slot {slot_index}: {exc}')
                        continue
                    _save_structural_preprocessor_output(cn_img, cn_type, slot_index)
                    
                    _STRUCTURAL_PREPROCESS_CACHE[cache_key] = cn_img.copy()
                    _STRUCTURAL_PREPROCESS_CACHE.move_to_end(cache_key)
                    while len(_STRUCTURAL_PREPROCESS_CACHE) > _STRUCTURAL_PREPROCESS_CACHE_LIMIT:
                        _STRUCTURAL_PREPROCESS_CACHE.popitem(last=False)
                        
            cn_img = HWC3(cn_img)
            task[0] = core.numpy_to_pytorch(cn_img)
            valid_tasks.append(task)
        task_state.set_cn_tasks(cn_type, valid_tasks)

    try:
        with resources.memory_phase_scope(
            resources.MemoryPhase.STRUCTURAL_PREPROCESS,
            task=task_state,
            notes={'task_count': sum(len(tasks) for tasks in structural_tasks.values())},
            end_notes={'completed': True},
        ):
            preprocess_structural_tasks(
                flags.cn_canny,
                structural_tasks.get(flags.cn_canny, []),
                lambda _cn_type, cn_img, _model_path: preprocessors.canny_pyramid(cn_img, task_state.canny_low_threshold, task_state.canny_high_threshold)
            )
            preprocess_structural_tasks(
                flags.cn_cpds,
                structural_tasks.get(flags.cn_cpds, []),
                lambda _cn_type, cn_img, _model_path: preprocessors.cpds(cn_img)
            )
            preprocess_structural_tasks(
                flags.cn_depth,
                structural_tasks.get(flags.cn_depth, []),
                structural_preprocessors.run_structural_preprocessor
            )
    finally:
        structural_preprocessors.apply_residency_policy('destroy')

    task_state.prepared_structural_cn_tasks = {
        cn_type: [list(task) for task in list(task_state.cn_tasks[cn_type])]
        for cn_type in flags.cn_structural_types
    }


def preprocess_contextual_controlnets(task_state, contextual_assets=None):
    _reset_preprocessor_metrics_once(task_state)
    width, height = task_state.width, task_state.height
    contextual_tasks = task_state.get_cn_tasks_for_channel(flags.cn_contextual)
    contextual_assets = contextual_assets or {}
    contextual_model_paths = contextual_assets.get('contextual_model_paths', {})
    clip_vision_path = contextual_assets.get('clip_vision_path')
    ip_negative_path = contextual_assets.get('ip_negative_path')
    insightface_model_names = contextual_assets.get('insightface_model_names') or ['antelopev2']
    eva_clip_path = contextual_assets.get('eva_clip_path')

    def normalize_contextual_task(task):
        if len(task) >= 4:
            return list(task)
        if len(task) == 3:
            return [task[0], task[1], task[2], 0.0]
        raise ValueError(f'Unexpected contextual task shape: {task!r}')



    def preprocess_contextual_tasks(cn_type, tasks, resize_to=None):
        valid_tasks = []
        model_path = contextual_model_paths.get(cn_type)
        if len(tasks) > 0 and model_path is None:
            print(f'[ControlNet] {cn_type} is missing its contextual model path. Skipping these tasks for now.')
            task_state.set_cn_tasks(cn_type, [])
            return

        for slot_index, task in enumerate(tasks, start=1):
            raw_img, cn_stop, cn_weight = task[:3]
            cn_img = _unpack_cn_image(raw_img, cn_type)
            if cn_img is None:
                continue
            if resize_to is not None:
                cn_img = resize_image(cn_img, width=resize_to, height=resize_to, resize_mode=0)
            try:
                if cn_type == flags.cn_pulid:
                    task[0] = pulid_runtime.preprocess(
                        cn_img,
                        model_path=model_path,
                        eva_clip_path=eva_clip_path,
                        insightface_model_names=insightface_model_names,
                    )
                else:
                    task[0] = contextual_ip_adapter.preprocess(
                        cn_img,
                        model_path=model_path,
                        clip_vision_path=clip_vision_path,
                        ip_negative_path=ip_negative_path,
                        insightface_model_names=insightface_model_names,
                        cache_kind=cn_type,
                    )
            except Exception as exc:
                print(f'[ControlNet] Failed to preprocess {cn_type} slot {slot_index}: {exc}')
                continue
            valid_tasks.append(normalize_contextual_task(task))
        task_state.set_cn_tasks(cn_type, valid_tasks)

    try:
        with resources.memory_phase_scope(
            resources.MemoryPhase.CONTEXTUAL_PREPROCESS,
            task=task_state,
            notes={'task_count': sum(len(tasks) for tasks in contextual_tasks.values())},
            end_notes={'completed': True},
        ):
            preprocess_contextual_tasks(flags.cn_ip, contextual_tasks.get(flags.cn_ip, []), resize_to=224)
            preprocess_contextual_tasks(flags.cn_pulid, contextual_tasks.get(flags.cn_pulid, []))
    finally:
        contextual_ip_adapter.apply_contextual_residency('destroy')
        pulid_runtime.apply_contextual_residency('destroy')

    all_contextual_tasks = []
    for cn_type in [flags.cn_ip]:
        all_contextual_tasks.extend(list(task_state.cn_tasks[cn_type]))

    pulid_tasks = list(task_state.cn_tasks[flags.cn_pulid])
    task_state.prepared_contextual_cn_tasks = {
        cn_type: [list(task) for task in list(task_state.cn_tasks[cn_type])]
        for cn_type in flags.cn_contextual_types
    }

    with resources.memory_phase_scope(
        resources.MemoryPhase.CONTROL_APPLY,
        task=task_state,
        notes={
            'contextual_patch_tasks': len(all_contextual_tasks),
            'pulid_patch_tasks': len(pulid_tasks),
        },
        end_notes={'completed': True},
    ):
        pass


def apply_control_nets(task_state, contextual_assets=None, structural_preprocessor_paths=None):
    """
    Applies Structural preprocessors and patches the UNet for contextual guidance.
    """
    preprocess_structural_controlnets(task_state, structural_preprocessor_paths=structural_preprocessor_paths)
    preprocess_contextual_controlnets(task_state, contextual_assets=contextual_assets)
