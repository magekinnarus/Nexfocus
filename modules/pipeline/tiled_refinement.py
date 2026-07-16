import math
import numpy as np
from typing import List, NamedTuple
import modules.core as core
import modules.flags as flags
import modules.blending as blending
from backend import resources

class TileInfo(NamedTuple):
    crop: tuple  # (x1, y1, x2, y2)
    tile_image: np.ndarray
    x: int
    y: int
    w: int
    h: int

def select_tile_resolution(full_w, full_h, min_overlap=128):
    """
    Smart Auto-Tiling 2.0: Iterates through all buckets and selects the one 
    that minimizes the total tile count (nx * ny).
    """
    buckets = []
    for s in flags.sdxl_aspect_ratios:
        w, h = map(int, s.split('*'))
        buckets.append((w, h))

    best_config = None
    min_total_tiles = float('inf')
    min_waste = float('inf')

    for bw, bh in buckets:
        # Calculate nx: how many tiles of width bw to cover full_w with min_overlap
        if full_w <= bw:
            nx = 1
            overlap_w = 0
        else:
            nx = math.ceil((full_w - min_overlap) / (bw - min_overlap))
            overlap_w = (nx * bw - full_w) / (nx - 1)
        
        # Calculate ny: how many tiles of height bh to cover full_h with min_overlap
        if full_h <= bh:
            ny = 1
            overlap_h = 0
        else:
            ny = math.ceil((full_h - min_overlap) / (bh - min_overlap))
            overlap_h = (ny * bh - full_h) / (ny - 1)

        total_tiles = nx * ny
        # Total waste is a combination of overlaps and aspect ratio mismatch
        waste = (overlap_w if nx > 1 else (bw - full_w)) + (overlap_h if ny > 1 else (bh - full_h))

        if total_tiles < min_total_tiles or (total_tiles == min_total_tiles and waste < min_waste):
            min_total_tiles = total_tiles
            min_waste = waste
            best_config = ((bw, bh), nx, ny, int(overlap_w), int(overlap_h))

    bucket, nx, ny, overlap_w, overlap_h = best_config
    print(f'[Smart Tiling] Optimized Layout: {nx}x{ny} grid using {bucket[0]}x{bucket[1]} bucket.')
    print(f'[Smart Tiling] Actual Overlap: {overlap_w}px (Horiz), {overlap_h}px (Vert)')
    
    return bucket, nx, ny, overlap_w, overlap_h

def split_into_tiles(image: np.ndarray, bucket_w: int, bucket_h: int, nx: int, ny: int, overlap_w: int, overlap_h: int) -> List[TileInfo]:
    H, W, C = image.shape
    tiles = []
    
    stride_w = bucket_w - overlap_w
    stride_h = bucket_h - overlap_h
    
    for i in range(ny):
        for j in range(nx):
            x1 = j * stride_w
            y1 = i * stride_h
            
            # Boundary correction
            if x1 + bucket_w > W: x1 = W - bucket_w
            if y1 + bucket_h > H: y1 = H - bucket_h
            
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = x1 + bucket_w, y1 + bucket_h
            
            tile_image = image[y1:y2, x1:x2]
            
            # Final check: Ensure we ARE at bucket size (in case of very small images)
            if tile_image.shape[0] != bucket_h or tile_image.shape[1] != bucket_w:
                import cv2
                tile_image = cv2.resize(tile_image, (bucket_w, bucket_h), interpolation=cv2.INTER_LANCZOS4)

            tiles.append(TileInfo(
                crop=(x1, y1, x2, y2),
                tile_image=tile_image.copy(),
                x=x1, y=y1, w=bucket_w, h=bucket_h
            ))
            
    return tiles


def stitch_tiles(tiles: List[TileInfo], full_size: tuple, bucket_w: int, bucket_h: int) -> np.ndarray:
    H, W, C = full_size
    output = np.zeros((H, W, C), dtype=np.float32)
    weights = np.zeros((H, W), dtype=np.float32)
    
    base_weight_map = blending.sin_blend_2d(bucket_w, bucket_h).cpu().numpy()
    
    for t in tiles:
        x1, y1, x2, y2 = t.crop
        output[y1:y2, x1:x2] += t.tile_image.astype(np.float32) * base_weight_map[:, :, None]
        weights[y1:y2, x1:x2] += base_weight_map
        
    output /= np.maximum(weights[:, :, None], 1e-5)
    return np.clip(output, 0, 255).astype(np.uint8)





def _resolve_tiled_prompt_blueprint(task_state, prompt_task=None):
    prompt_task = prompt_task or {}
    prompt = str(prompt_task.get('task_prompt', task_state.prompt) or '')
    negative_prompt = str(prompt_task.get('task_negative_prompt', task_state.negative_prompt) or '')

    positive_texts = tuple(str(item) for item in (prompt_task.get('positive') or [prompt]))
    negative_texts = tuple(str(item) for item in (prompt_task.get('negative') or [negative_prompt]))

    positive_top_k = int(prompt_task.get('positive_top_k', len(positive_texts)) or max(1, len(positive_texts)))
    negative_top_k = int(prompt_task.get('negative_top_k', len(negative_texts)) or max(1, len(negative_texts)))
    seed = int(prompt_task.get('task_seed', task_state.seed))

    return {
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'positive_texts': positive_texts,
        'negative_texts': negative_texts,
        'positive_top_k': positive_top_k,
        'negative_top_k': negative_top_k,
        'seed': seed,
    }


def should_retain_sdxl_warm_state(task_state) -> bool:
    from backend import process_transition
    from modules.pipeline.workflow_contracts import require_workflow_plan

    requested_key = process_transition.resolve_sdxl_process_key(
        task_state,
        workflow_plan=require_workflow_plan(task_state),
        allow_legacy_adapter=False,
    )
    if requested_key is None:
        return False

    decision = process_transition.evaluate_process_transition(requested_key)
    return decision.action == 'reuse' and decision.reason != 'lora_stack_change'


def _register_active_unified_sdxl_process(task_state) -> None:
    from backend import process_transition
    from modules.pipeline.workflow_contracts import require_workflow_plan

    policy = getattr(task_state, 'sdxl_execution_policy', None)
    execution_mode = getattr(policy, 'execution_mode', None)
    process_transition.publish_sdxl_runtime(
        task_state,
        workflow_plan=require_workflow_plan(task_state),
        route_owner=getattr(task_state, 'runtime_route_id', None) or 'super_upscale',
        safe_to_retain=(execution_mode == 'resident'),
    )


# Classified: Compatibility Bridge.
# Retained for backward-compatible routing of tiled refinement requests to the backend-owned SDXL assembly pipeline,
# while preserving stateless helpers (select_tile_resolution, split_into_tiles, stitch_tiles) as utility functions.

def apply_tiled_diffusion_refinement(task_state, upscaled_image: np.ndarray, progressbar_callback=None, prompt_task=None):
    from backend.sdxl_assembly.gateway import run_sdxl_assembly_task
    import copy
    from backend import resources

    # Early stop/skip checks
    if resources.processing_interrupted():
        if getattr(task_state, 'last_stop', None) == 'skip':
            print("[Tiled Refinement (Assembly)] User skipped tiled refinement. Stitching partially completed tiles...")
            resources.interrupt_current_processing(False)
            task_state.last_stop = False
            return np.array(upscaled_image, copy=True)

        print("[Tiled Refinement] Stop request detected before execution starts; raising InterruptProcessingException.")
        resources.throw_exception_if_processing_interrupted()

    task_dict = {
        'task_seed': task_state.seed,
        'task_prompt': task_state.prompt,
        'task_negative_prompt': task_state.negative_prompt,
    }
    if prompt_task:
        def _get_val(obj, key, default):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)
        task_dict.update({
            'task_seed': _get_val(prompt_task, 'task_seed', task_state.seed),
            'task_prompt': _get_val(prompt_task, 'task_prompt', task_state.prompt),
            'task_negative_prompt': _get_val(prompt_task, 'task_negative_prompt', task_state.negative_prompt),
            'positive': _get_val(prompt_task, 'positive', None),
            'negative': _get_val(prompt_task, 'negative', None),
        })

    # Prepare derived/temporary task state so we don't mutate main state in a confusing way
    derived_state = copy.copy(task_state)
    derived_state.upscale_gan_output_image = upscaled_image

    return run_sdxl_assembly_task(
        task_state=derived_state,
        task_dict=task_dict,
        current_task_id=0,
        total_count=1,
        all_steps=int(getattr(task_state, 'steps', 3)),
        preparation_steps=0,
        denoising_strength=float(getattr(task_state, 'upscale_refinement_denoise', 0.382)),
        final_scheduler_name=getattr(task_state, 'scheduler_name', 'karras'),
        loras=list(getattr(task_state, 'loras', []) or []),
        controlnet_paths={},
        contextual_assets={},
        base_model_additional_loras=list(getattr(task_state, 'base_model_additional_loras', None) or []),
        image_input_result=None,
        progressbar_callback=progressbar_callback,
    )
