"""
Centralized mask processing utilities for Fooocus Inpaint/Outpaint workflows.

All Gradio sketch/image component unpacking and mask color extraction goes through
this module to prevent RGBA vs RGB shape mismatch bugs.
"""
import numpy as np
import gradio as gr
from modules.util import HWC3
from PIL import Image
import os
import io
import base64
import json
import uuid
import modules.config


# ---------------------------------------------------------------------------
# In-process image cache
# Avoids repeated disk I/O + PIL decode of the same large base-image file
# on every context-mask stroke update or BB-refresh in the Gradio UI.
# ---------------------------------------------------------------------------
_IMAGE_CACHE: dict = {}   # filepath -> np.ndarray (RGB)
_IMAGE_CACHE_MAX = 8      # evict oldest entries when cache grows beyond this


def get_cached_image(filepath: str) -> "np.ndarray | None":
    """Return an RGB numpy array for *filepath*, using a simple in-process cache.

    The first call for a given path decodes the image and stores it. Subsequent
    calls for the same path return the cached array directly, skipping disk I/O
    and PIL decoding entirely. This is critical for large images (3k+ px) during rapid
    UI interactions such as context-mask stroke updates or BB refreshes.

    Cache entries are keyed by the *resolved absolute path* so that renaming or
    replacing the file will naturally produce a cache miss on the next call.
    """
    if not filepath:
        return None
    filepath = os.path.abspath(filepath)
    if filepath in _IMAGE_CACHE:
        return _IMAGE_CACHE[filepath]
    img = unpack_gradio_data(filepath)
    if img is not None:
        if len(_IMAGE_CACHE) >= _IMAGE_CACHE_MAX:
            # Evict the entry that was inserted first
            oldest_key = next(iter(_IMAGE_CACHE))
            del _IMAGE_CACHE[oldest_key]
        _IMAGE_CACHE[filepath] = img
    return img


def invalidate_image_cache(filepath: str = None):
    """Remove a specific path (or clear the entire cache) when a file changes."""
    if filepath is None:
        _IMAGE_CACHE.clear()
    else:
        _IMAGE_CACHE.pop(os.path.abspath(filepath), None)


def rgba_to_black_bg_rgb(x):
    """
    Unlike util.HWC3 which composites transparent pixels over a white background,
    masks need transparent pixels to composite over a BLACK background (0 = no mask).
    """
    if x is None:
        return None
        
    if x.dtype in (np.float32, np.float64):
        if x.max() <= 1.0:
            x = (x * 255).astype(np.uint8)
        else:
            x = x.astype(np.uint8)
            
    if x.dtype != np.uint8:
        x = x.astype(np.uint8)
    
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        # Composite over BLACK (0.0): y = color * alpha + 0.0 * (1.0 - alpha)
        y = color * alpha
        return y.clip(0, 255).astype(np.uint8)
    return x

def ensure_numpy(x, mode='RGB'):
    """
    Ensure the input is a numpy array. Handles None, str (filepath/data URL),
    PIL.Image, and np.ndarray.
    """
    if x is None:
        return None

    if isinstance(x, str):
        if x == '':
            return None
        if x.startswith('data:image'):
            try:
                _, encoded = x.split(',', 1)
                with Image.open(io.BytesIO(base64.b64decode(encoded))) as img:
                    return np.array(img.convert(mode))
            except Exception as e:
                print(f"[mask_processing] Error decoding image data URL: {e}")
                return None
        if os.path.exists(x):
            try:
                with Image.open(x) as img:
                    return np.array(img.convert(mode))
            except Exception as e:
                print(f"[mask_processing] Error loading image from {x}: {e}")
                return None
        print(f"[mask_processing] File not found: {x}")
        return None

    if isinstance(x, Image.Image):
        return np.array(x.convert(mode))

    if isinstance(x, np.ndarray):
        return x

    return None


def unpack_gradio_data(data):
    """
    Safely extract an RGB image from a Gradio image/sketch component.
    """
    if data is None:
        return None
    
    if isinstance(data, (np.ndarray, str, Image.Image)):
        return rgba_to_black_bg_rgb(ensure_numpy(data))
    
    if isinstance(data, dict):
        img = data.get('image')
        if img is not None:
            return rgba_to_black_bg_rgb(ensure_numpy(img))
        mask = data.get('mask')
        if mask is not None:
            return rgba_to_black_bg_rgb(ensure_numpy(mask))
    
    return None


def unpack_gradio_sketch(data):
    """
    Safely extract both the base image and the drawn mask from a Gradio sketch component.
    """
    if data is None:
        return None, None
    
    if isinstance(data, (np.ndarray, str, Image.Image)):
        return rgba_to_black_bg_rgb(ensure_numpy(data)), None
    
    if isinstance(data, dict):
        img = data.get('image')
        mask = data.get('mask')
        
        img_out = rgba_to_black_bg_rgb(ensure_numpy(img)) if img is not None else None
        mask_out = rgba_to_black_bg_rgb(ensure_numpy(mask)) if mask is not None else None
        
        return img_out, mask_out
    
    return None, None


def extract_mask_from_layers(data):
    """
    Extract a binary mask from Gradio 5 ImageEditor's layers.
    Combines alpha channels of all layers into a single [0, 255] uint8 mask.
    """
    if not isinstance(data, dict):
        return None
    
    layers = data.get('layers')
    if not layers:
        return None
        
    mask = None
    for layer in layers:
        if layer is None:
            continue
            
        layer_arr = ensure_numpy(layer, mode='RGBA')
        if layer_arr is None:
            continue

        if layer_arr.ndim != 3 or layer_arr.shape[2] != 4:
            continue
            
        # Extract alpha as the mask
        alpha = layer_arr[:, :, 3]
        if mask is None:
            mask = alpha
        else:
            mask = np.maximum(mask, alpha)
            
    if mask is None:
        return None
        
    return (mask > 0).astype(np.uint8) * 255


def extract_color_masks_from_layers(data):
    """
    Extract specifically Blue and White masks from ImageEditor layers.
    Blue (#0000FF) = Context Mask
    White (#FFFFFF) = Inpaint Mask
    """
    if not isinstance(data, dict):
        return None, None
    
    layers = data.get('layers')
    if not layers:
        return None, None
        
    white_mask = None
    blue_mask = None
    
    for layer in layers:
        if layer is None:
            continue
            
        layer_arr = ensure_numpy(layer, mode='RGBA')
        if layer_arr is None:
            continue

        if layer_arr.ndim != 3 or layer_arr.shape[2] != 4:
            continue
            
        r, g, b, a = layer_arr[:,:,0], layer_arr[:,:,1], layer_arr[:,:,2], layer_arr[:,:,3]
        alpha_mask = a > 0
        
        # Detect colors based on predominant channels
        white_strokes = (r > 127) & (g > 127) & (b > 127) & alpha_mask
        blue_strokes = (r < 127) & (g < 127) & (b > 127) & alpha_mask
        
        if white_strokes.any():
            m = np.zeros(layer_arr.shape[:2], dtype=np.uint8)
            m[white_strokes] = 255
            white_mask = np.maximum(white_mask, m) if white_mask is not None else m
            
        if blue_strokes.any():
            m = np.zeros(layer_arr.shape[:2], dtype=np.uint8)
            m[blue_strokes] = 255
            blue_mask = np.maximum(blue_mask, m) if blue_mask is not None else m
            
    return white_mask, blue_mask


def combine_image_and_mask(data):
    """
    Merge the 'image' and 'mask' layers from a Gradio sketch dict into a single RGB array
    using element-wise maximum. Handles RGBA vs RGB mismatches safely.
    
    Returns:
        np.ndarray (H, W, 3) or None
    """
    if data is None:
        return None
    
    if isinstance(data, (np.ndarray, str, Image.Image)):
        return ensure_numpy(data)
    
    if isinstance(data, dict):
        # Gradio 5 ImageEditor / EditorValue format
        if 'composite' in data:
            # For combine_image_and_mask, we either want the composite or 
            # the background + extracted mask. In Fooocus, this is often used
            # for mask expansion or previewing.
            if 'background' in data and data['background'] is not None:
                bg = rgba_to_black_bg_rgb(ensure_numpy(data['background']))
                mask = extract_mask_from_layers(data)
                if mask is not None:
                    # Overlay mask onto background for preview
                    mask_rgb = np.stack([mask]*3, axis=-1)
                    return np.maximum(bg, mask_rgb)
                return bg
            if 'composite' in data and data['composite'] is not None:
                return rgba_to_black_bg_rgb(ensure_numpy(data['composite']))
            return None

        # Legacy Gradio 3 sketch format
        img = data.get('image')
        mask = data.get('mask')
        
        if isinstance(img, np.ndarray) and isinstance(mask, np.ndarray) and img.ndim == 3:
            return np.maximum(rgba_to_black_bg_rgb(img), rgba_to_black_bg_rgb(mask))
        elif isinstance(img, np.ndarray):
            return rgba_to_black_bg_rgb(img)
        elif isinstance(mask, np.ndarray):
            return rgba_to_black_bg_rgb(mask)
    
    return None


def extract_color_masks(raw_mask_layer):
    """
    Extract white (inpaint) and blue (context) stroke masks from a Gradio sketch component's
    raw mask layer.
    
    The raw mask layer may be (H, W, 3) or (H, W, 4). If 4 channels, the alpha channel
    is used as a stroke presence indicator.
    
    Args:
        raw_mask_layer: np.ndarray (H, W, 3) or (H, W, 4) from Gradio sketch
        
    Returns:
        (white_mask_2d, blue_mask_2d) ??both (H, W) np.uint8, values 0 or 255
    """
    if raw_mask_layer is None:
        return None, None
    
    has_alpha = raw_mask_layer.ndim == 3 and raw_mask_layer.shape[2] == 4
    if has_alpha:
        alpha_mask = raw_mask_layer[:, :, 3] > 0
    else:
        alpha_mask = np.ones(raw_mask_layer.shape[:2], dtype=bool)
    
    r = raw_mask_layer[:, :, 0]
    g = raw_mask_layer[:, :, 1]
    b = raw_mask_layer[:, :, 2]
    
    white_strokes = (r > 127) & (g > 127) & (b > 127) & alpha_mask
    blue_strokes = (r < 127) & (g < 127) & (b > 127) & alpha_mask
    
    white_mask = np.zeros(raw_mask_layer.shape[:2], dtype=np.uint8)
    white_mask[white_strokes] = 255
    
    blue_mask = np.zeros(raw_mask_layer.shape[:2], dtype=np.uint8)
    blue_mask[blue_strokes] = 255
    
    return white_mask, blue_mask


def to_binary_mask(mask):
    """
    Convert any mask array to a clean binary 2D mask (0 or 255, uint8).
    
    Handles:
      - Float masks (0.0-1.0 or 0.0-255.0)
      - Multi-channel masks (takes max across channels)
      - 4-channel RGBA (strips alpha)
    
    Returns:
        np.ndarray (H, W) uint8 with values 0 or 255, or None
    """
    if mask is None:
        return None
    
    if isinstance(mask, (str, Image.Image)):
        mask = ensure_numpy(mask)
    
    if mask.dtype in (np.float32, np.float64):
        if mask.max() <= 1.0:
            mask = mask * 255.0
        mask = mask.astype(np.uint8)
    
    if mask.ndim == 3:
        if mask.shape[-1] == 4:
            alpha = mask[..., 3]
            rgb = mask[..., :3]
            mask = np.maximum(alpha, np.max(rgb, axis=-1))
        else:
            mask = np.max(mask, axis=-1)
    
    return (mask > 127).astype(np.uint8) * 255


def combine_masks(*masks):
    """
    Merge multiple 2D masks into one using element-wise maximum.
    Ignores None entries.
    
    Returns:
        np.ndarray (H, W) uint8, or None if all inputs are None
    """
    result = None
    for m in masks:
        if m is None:
            continue
        m2d = to_binary_mask(m)
        if m2d is None:
            continue
        if result is None:
            result = m2d
        else:
            result = np.maximum(result, m2d)
    return result


def expand_mask_direction(mask_2d, direction, pixels=32):
    """
    Expand white pixels in a 2D mask in the OPPOSITE direction of outpaint.
    E.g., if direction is 'Right', expand leftward (into original image).
    
    Args:
        mask_2d: (H, W) uint8 array, values 0 or 255
        direction: one of 'Left', 'Right', 'Top', 'Bottom' (case-sensitive)
        pixels: number of pixels to expand
        
    Returns:
        np.ndarray (H, W) uint8
    """
    result = mask_2d.copy()
    
    for _ in range(pixels):
        shifted = np.zeros_like(result)
        if direction == 'Right':      # Expand Left
            shifted[:, :-1] = result[:, 1:]
        elif direction == 'Left':     # Expand Right
            shifted[:, 1:] = result[:, :-1]
        elif direction == 'Top':      # Expand Bottom
            shifted[1:, :] = result[:-1, :]
        elif direction == 'Bottom':   # Expand Top
            shifted[:-1, :] = result[1:, :]
        result = np.maximum(result, shifted)
    
    return result


def save_to_png(numpy_img, filepath):
    """
    Saves a numpy array to a specific PNG file.
    """
    if numpy_img is None:
        return None
    from PIL import Image
    
    # Handle single channel (mask) or multi-channel
    if numpy_img.ndim == 2:
        img = Image.fromarray(numpy_img, mode='L')
    else:
        img = Image.fromarray(numpy_img)
        
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    img.save(filepath)
    invalidate_image_cache(filepath)
    return filepath


def save_to_temp_png(numpy_img):
    """
    Saves a numpy array to a temporary PNG file and returns the filepath.
    Critical for the 'type=filepath' UI memory invariant to prevent RAM bloat.
    """
    if numpy_img is None:
        return None
    import modules.util
    import modules.config
    
    _, temp_path, _ = modules.util.generate_temp_filename(
        folder=modules.config.path_temp_outputs, extension='png')
    
    return save_to_png(numpy_img, temp_path)


def save_to_staging_png(numpy_img, prefix='staged'):
    """
    Saves a numpy array to the persistent staging folder so users can reuse
    intermediate artifacts from the UI.
    """
    if numpy_img is None:
        return None

    import datetime

    staging_dir = os.path.join(modules.config.path_outputs, "staging")
    os.makedirs(staging_dir, exist_ok=True)

    safe_prefix = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in str(prefix)).strip('_')
    if safe_prefix == '':
        safe_prefix = 'staged'

    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S_%f")
    filename = f"{safe_prefix}_{time_str}.png"
    filepath = os.path.join(staging_dir, filename)
    return save_to_png(numpy_img, filepath)


def ensure_workspace_dir(workspace_id=None, prefix='mask_slot'):
    if workspace_id and all(c.isalnum() or c == '_' for c in workspace_id):
        resolved_workspace_id = workspace_id
    else:
        resolved_workspace_id = f"{prefix}_{uuid.uuid4().hex}"

    root = os.path.abspath(os.path.join(modules.config.temp_path, "workspaces"))
    os.makedirs(root, exist_ok=True)
    workspace_dir = os.path.join(root, resolved_workspace_id)
    os.makedirs(workspace_dir, exist_ok=True)
    return resolved_workspace_id, workspace_dir


def save_to_workspace_png(numpy_img, workspace_id=None, filename='base.png', prefix='mask_slot'):
    if numpy_img is None:
        return None, workspace_id

    resolved_workspace_id, workspace_dir = ensure_workspace_dir(workspace_id, prefix=prefix)
    filepath = os.path.join(workspace_dir, filename)
    save_to_png(numpy_img, filepath)
    return filepath, resolved_workspace_id

def resolve_workspace_image_path(candidate_path, workspace_id, preferred_name='base.png'):
    if candidate_path and os.path.exists(candidate_path):
        return candidate_path

    if not workspace_id:
        return ''

    _, workspace_dir = ensure_workspace_dir(workspace_id, prefix='mask_slot')
    preferred_path = os.path.join(workspace_dir, preferred_name)
    if os.path.exists(preferred_path):
        return preferred_path

    png_candidates = [
        os.path.join(workspace_dir, item)
        for item in os.listdir(workspace_dir)
        if item.lower().endswith('.png')
    ]
    if not png_candidates:
        return ''

    png_candidates.sort(key=os.path.getmtime, reverse=True)
    return png_candidates[0]



def prepare_outpaint_step1_assets(base_image_path, base_workspace_id, bb_workspace_id, mask_workspace_id, outpaint_selections, expansion_size):
    directions = outpaint_selections if isinstance(outpaint_selections, list) else []
    resolved_base_path = resolve_workspace_image_path(base_image_path, base_workspace_id, preferred_name='base.png')
    if not resolved_base_path:
        return gr.update(), gr.update(value=base_workspace_id or ""), gr.update(), gr.update(value=bb_workspace_id or ""), gr.update(value=""), gr.update(value=mask_workspace_id or ""), gr.update(value=""), gr.update(value=False), gr.update(value="Upload a Base Image first, then wait a moment for it to finish saving.")
    if len(directions) == 0:
        return gr.update(), gr.update(value=base_workspace_id or ""), gr.update(), gr.update(value=bb_workspace_id or ""), gr.update(value=""), gr.update(value=mask_workspace_id or ""), gr.update(value=""), gr.update(value=False), gr.update(value="Choose at least one Outpaint direction before preparing.")

    original_image = get_cached_image(resolved_base_path)
    if original_image is None:
        return gr.update(), gr.update(value=base_workspace_id or ""), gr.update(), gr.update(value=bb_workspace_id or ""), gr.update(value=""), gr.update(value=mask_workspace_id or ""), gr.update(value=""), gr.update(value=False), gr.update(value="Unable to read the current Base Image.")

    direction = directions[0].lower()
    try:
        expansion_size = int(expansion_size)
    except (TypeError, ValueError):
        expansion_size = 384
    from modules.pipeline.outpaint import OutpaintPipeline
    outpaint = OutpaintPipeline()
    expanded_image, generated_mask = outpaint.prepare_outpaint_canvas_only(
        original_image,
        direction,
        expansion_size=expansion_size,
        pixelate=False
    )
    # Skip blend-mask morphology during step-1 UI prep; stitch-time compositing owns it.
    ctx = outpaint.prepare(
        image=expanded_image,
        mask=generated_mask,
        outpaint_direction=direction,
        extend_factor=1.2,
        generate_blend_mask=False
    )

    expanded_filename = f"expanded_canvas_{uuid.uuid4().hex}.png"
    bb_filename = f"bb_image_{uuid.uuid4().hex}.png"
    prepared_base_path, resolved_base_workspace_id = save_to_workspace_png(
        expanded_image,
        workspace_id=base_workspace_id,
        filename=expanded_filename,
        prefix='outpaint_base'
    )
    prepared_bb_path, resolved_bb_workspace_id = save_to_workspace_png(
        ctx.bb_image,
        workspace_id=bb_workspace_id,
        filename=bb_filename,
        prefix='outpaint_bb'
    )

    notice = "Expanded canvas loaded into Base Image. BB Image ready for review."
    return (
        gr.update(value=prepared_base_path),
        gr.update(value=resolved_base_workspace_id),
        gr.update(value=prepared_bb_path),
        gr.update(value=resolved_bb_workspace_id),
        gr.update(value=""),
        gr.update(value=mask_workspace_id or ""),
        gr.update(value=""),
        gr.update(value=True),
        gr.update(value=notice)
    )

def core_compute_inpaint_step1_context(original_image, context_mask):
    """
    Core logic for inpaint step 1 context computation.
    Decoupled from Gradio and file saving.
    """
    from modules.pipeline.inpaint import InpaintPipeline
    inpaint = InpaintPipeline()
    # Skip blend-mask morphology during step-1 UI prep; stitch-time compositing owns it.
    ctx = inpaint.prepare(
        image=original_image,
        mask=context_mask,
        context_mask=None,
        extend_factor=1.2,
        generate_blend_mask=False
    )
    return ctx


def reset_inpaint_prepared_assets():
    return (
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=False)
    )


def compute_inpaint_step1_context(base_image_path, base_workspace_id, context_workspace_id, bb_workspace_id, mask_workspace_id, mask_b64):
    resolved_base_path = resolve_workspace_image_path(base_image_path, base_workspace_id, preferred_name='base.png')
    if not resolved_base_path or not mask_b64:
        return reset_inpaint_prepared_assets()

    original_image = get_cached_image(resolved_base_path)
    if original_image is None:
        return reset_inpaint_prepared_assets()

    context_mask = ensure_numpy(mask_b64, mode='L')
    if context_mask is None:
        return reset_inpaint_prepared_assets()

    ctx = core_compute_inpaint_step1_context(original_image, context_mask)
    context_path, resolved_context_workspace_id = save_to_workspace_png(
        context_mask,
        workspace_id=context_workspace_id,
        filename=f'context_mask_{uuid.uuid4().hex}.png',
        prefix='inpaint_context'
    )
    bb_path, resolved_bb_workspace_id = save_to_workspace_png(
        ctx.bb_image,
        workspace_id=bb_workspace_id,
        filename=f'bb_image_{uuid.uuid4().hex}.png',
        prefix='inpaint_bb'
    )

    return (
        gr.update(value=context_path),
        gr.update(value=resolved_context_workspace_id),
        gr.update(value=bb_path),
        gr.update(value=resolved_bb_workspace_id),
        gr.update(value=""),
        gr.update(value=mask_workspace_id or ""),
        gr.update(),
        gr.update(value=""),
        gr.update(value=json.dumps(tuple(int(v) for v in ctx.bb))),
        gr.update(value=True)
    )


def refresh_inpaint_bb_image(base_image_path, base_workspace_id, context_image_path, context_workspace_id, bb_workspace_id, mask_workspace_id, mask_b64):
    resolved_base_path = resolve_workspace_image_path(base_image_path, base_workspace_id, preferred_name='base.png')
    if not resolved_base_path:
        return (
            gr.update(),
            gr.update(),
            gr.update(value=""),
            gr.update(value=mask_workspace_id or ""),
            gr.update(value=""),
            gr.update(),
            gr.update()
        )

    original_image = get_cached_image(resolved_base_path)
    if original_image is None:
        return (
            gr.update(),
            gr.update(),
            gr.update(value=""),
            gr.update(value=mask_workspace_id or ""),
            gr.update(value=""),
            gr.update(),
            gr.update()
        )

    context_mask = None
    resolved_context_path = resolve_workspace_image_path(
        context_image_path,
        context_workspace_id,
        preferred_name='context_mask.png'
    )
    if resolved_context_path:
        context_mask = unpack_gradio_data(resolved_context_path)
        context_mask = ensure_numpy(context_mask, mode='L') if context_mask is not None else None

    if context_mask is None and mask_b64:
        context_mask = ensure_numpy(mask_b64, mode='L')

    if context_mask is None:
        return (
            gr.update(),
            gr.update(),
            gr.update(value=""),
            gr.update(value=mask_workspace_id or ""),
            gr.update(value=""),
            gr.update(),
            gr.update()
        )

    ctx = core_compute_inpaint_step1_context(original_image, context_mask)
    bb_path, resolved_bb_workspace_id = save_to_workspace_png(
        ctx.bb_image,
        workspace_id=bb_workspace_id,
        filename=f'bb_image_{uuid.uuid4().hex}.png',
        prefix='inpaint_bb'
    )

    from modules.pipeline.inference import _is_debug_console_logging_enabled

    if _is_debug_console_logging_enabled():
        print('[Debug] Refreshed BB image from current Base Image and Context Mask.')

    return (
        gr.update(value=bb_path),
        gr.update(value=resolved_bb_workspace_id),
        gr.update(value=""),
        gr.update(value=mask_workspace_id or ""),
        gr.update(value=""),
        gr.update(value=json.dumps(tuple(int(v) for v in ctx.bb))),
        gr.update(value=True)
    )


def compute_inpaint_step2_mask(workspace_id, mask_b64):
    if not mask_b64:
        return gr.update(value=""), gr.update(value=workspace_id or ""), gr.update()

    bb_mask = ensure_numpy(mask_b64, mode='L')
    if bb_mask is None:
        return gr.update(), gr.update(value=workspace_id or ""), gr.update()

    bb_mask_path, resolved_workspace_id = save_to_workspace_png(
        bb_mask,
        workspace_id=workspace_id,
        filename=f'bb_mask_{uuid.uuid4().hex}.png',
        prefix='inpaint_mask'
    )
    return gr.update(value=bb_mask_path), gr.update(value=resolved_workspace_id), gr.update()


def compute_outpaint_step2_mask(workspace_id, mask_b64):
    if not mask_b64:
        return gr.update(value=""), gr.update(value=workspace_id or ""), gr.update(value="")

    bb_mask = ensure_numpy(mask_b64, mode='L')
    if bb_mask is None:
        return gr.update(), gr.update(value=workspace_id or ""), gr.update()

    filename = f"bb_mask_{uuid.uuid4().hex}.png"
    bb_mask_path, resolved_workspace_id = save_to_workspace_png(
        bb_mask,
        workspace_id=workspace_id,
        filename=filename,
        prefix='outpaint_mask'
    )
    return gr.update(value=bb_mask_path), gr.update(value=resolved_workspace_id), gr.update()
def compute_remove_mask(workspace_id, mask_b64):
    if not mask_b64:
        return gr.update(value=""), gr.update(value=workspace_id or ""), gr.update(value="")

    remove_mask = ensure_numpy(mask_b64, mode='L')
    if remove_mask is None:
        return gr.update(), gr.update(value=workspace_id or ""), gr.update()

    remove_mask_path, resolved_workspace_id = save_to_workspace_png(
        remove_mask,
        workspace_id=workspace_id,
        filename=f'remove_mask_{uuid.uuid4().hex}.png',
        prefix='remove_mask'
    )
    return gr.update(value=remove_mask_path), gr.update(value=resolved_workspace_id), gr.update()
