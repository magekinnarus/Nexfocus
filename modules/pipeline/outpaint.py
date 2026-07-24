import torch
import numpy as np
import cv2
from dataclasses import dataclass
from PIL import Image
from modules.util import resample_image, set_image_shape_ceil, get_image_shape_ceil
from modules.upscaler import perform_upscale
from modules.core import numpy_to_pytorch
import modules.core as core

import modules.flags as flags
import modules.blending as blending


@dataclass
class OutpaintContext:
    """Carries all state between outpaint stages. No globals."""
    original_image: np.ndarray       # Full original image for final compositing
    original_mask: np.ndarray        # Full original mask (0=keep, 255=regenerate)
    bb: tuple                        # (y1, y2, x1, x2) bounding box in original image coords
    bb_image: np.ndarray             # Cropped region resized to SDXL resolution
    bb_mask: np.ndarray              # Cropped mask resized to SDXL resolution
    target_w: int                    # SDXL-snapped width
    target_h: int                    # SDXL-snapped height
    blend_mask: np.ndarray | None    # Full-image morphological gradient for stitching


class OutpaintPipeline:
    SDXL_RESOLUTIONS = [(int(s.split('*')[0]), int(s.split('*')[1])) for s in flags.sdxl_aspect_ratios]

    def __init__(self):
        pass

    def snap_to_sdxl_resolution(self, w, h):
        target_ratio = w / h
        best = min(self.SDXL_RESOLUTIONS, key=lambda r: abs(r[0]/r[1] - target_ratio))
        return best

    def _expand_canvas(self, image, y1, y2, x1, x2, pixelate=True):
        H, W, C = image.shape
        ey1, ey2, ex1, ex2 = y1, y2, x1, x2
        
        # Calculate overflow
        oy1 = max(0, -y1)
        oy2 = max(0, y2 - H)
        ox1 = max(0, -x1)
        ox2 = max(0, x2 - W)
        
        if oy1 == 0 and oy2 == 0 and ox1 == 0 and ox2 == 0:
            return image[y1:y2, x1:x2], (0, y2-y1, 0, x2-x1)
            
        # Create expanded canvas
        canvas_h = y2 - y1
        canvas_w = x2 - x1
        canvas = np.zeros((canvas_h, canvas_w, C), dtype=image.dtype)
        
        # Copy original image part
        iy1, iy2 = max(0, y1), min(H, y2)
        ix1, ix2 = max(0, x1), min(W, x2)
        cy1, cy2 = iy1 - y1, iy2 - y1
        cx1, cx2 = ix1 - x1, ix2 - x1
        canvas[cy1:cy2, cx1:cx2] = image[iy1:iy2, ix1:ix2]
        
        # Edge replication for overflow
        if oy1 > 0: canvas[:cy1, cx1:cx2] = canvas[cy1:cy1+1, cx1:cx2]
        if oy2 > 0: canvas[cy2:, cx1:cx2] = canvas[cy2-1:cy2, cx1:cx2]
        if ox1 > 0: canvas[:, :cx1] = canvas[:, cx1:cx1+1]
        if ox2 > 0: canvas[:, cx2:] = canvas[:, cx2-1:cx2]
        
        # Apply strict 8x8 pixelation to overflow regions for SDXL VAE alignment
        def pixelate_helper(slice_idx):
            h, w, c = slice_idx.shape
            if h > 0 and w > 0:
                is_single_channel = (c == 1)
                # SDXL VAE compresses 8x8 pixels into 1 latent pixel
                small = cv2.resize(slice_idx, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_NEAREST)
                large = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                if is_single_channel and large.ndim == 2:
                    large = large[:, :, None]
                return large
            return slice_idx

        if pixelate:
            if oy1 > 0: canvas[:cy1, :] = pixelate_helper(canvas[:cy1, :])
            if oy2 > 0: canvas[cy2:, :] = pixelate_helper(canvas[cy2:, :])
            if ox1 > 1: canvas[:, :cx1] = pixelate_helper(canvas[:, :cx1])
            if ox2 > 1: canvas[:, cx2:] = pixelate_helper(canvas[:, cx2:])
        
        return canvas, (cy1, cy2, cx1, cx2)

    def prepare_outpaint_canvas_only(self, image, direction, expansion_size=384, pixelate=True):
        """Phase 1 Outpaint: Pad the image with pixelated primer and return the exact mask of the new area."""
        H, W, C = image.shape
        
        y1, y2, x1, x2 = 0, H, 0, W
        if direction == 'top':
            y1 = -expansion_size
        elif direction == 'bottom':
            y2 = H + expansion_size
        elif direction == 'left':
            x1 = -expansion_size
        elif direction == 'right':
            x2 = W + expansion_size
            
        expanded_image, (cy1, cy2, cx1, cx2) = self._expand_canvas(image, y1, y2, x1, x2, pixelate=pixelate)
        
        expanded_mask = np.zeros(expanded_image.shape[:2], dtype=np.uint8)
        if direction == 'top':
            expanded_mask[:cy1, :] = 255
        elif direction == 'bottom':
            expanded_mask[cy2:, :] = 255
        elif direction == 'left':
            expanded_mask[:, :cx1] = 255
        elif direction == 'right':
            expanded_mask[:, cx2:] = 255
            
        return expanded_image, expanded_mask

    def pixelate_mask_area(self, image, mask, pixelation_ratio=8):
        """Phase 2 Primer: Applies strict 8x8 pixelation to the mapped area to preserve color edges for SDXL latents."""
        H, W, C = image.shape
        
        # Create a tiny version of the image using a strict 8x8 block size
        small = cv2.resize(image, (max(1, W // 8), max(1, H // 8)), interpolation=cv2.INTER_AREA)
        # Scale it back up using nearest neighbor for the blocky effect
        pixelated = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
        
        # Ensure mask is 3 channels for boolean indexing
        mask_3c = mask[:, :, None] if mask.ndim == 2 else mask
        if mask_3c.shape[2] == 1:
            mask_3c = np.repeat(mask_3c, 3, axis=2)
            
        # Blend: original image where mask is 0, pixelated where mask is 255
        binary_mask = (mask_3c > 127)
        primed_image = np.where(binary_mask, pixelated, image).astype(np.uint8)
        
        return primed_image

    def _box_blur(self, x, k):
        kernel_size = 2 * k + 1
        return cv2.blur(x, (kernel_size, kernel_size))

    def _max_filter_opencv(self, x, ksize=3):
        return cv2.dilate(x, np.ones((ksize, ksize), dtype=np.int16))

    def _morphological_open(self, x):
        x_int16 = np.zeros_like(x, dtype=np.int16)
        x_int16[x > 127] = 256
        for _ in range(32):
            maxed = self._max_filter_opencv(x_int16, ksize=3) - 8
            x_int16 = np.maximum(maxed, x_int16)
        return np.clip(x_int16, 0, 255).astype(np.uint8)

    def prepare(self, image, mask, extend_factor=1.2, outpaint_direction=None, generate_blend_mask=True) -> OutpaintContext:
        """Native-AR Bounding Box algorithm for Outpaint."""
        if image is None:
            raise ValueError(
                'Outpaint source image is not ready. Prepare Outpaint again, then Generate.'
            )
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError(f'Outpaint expects an HWC RGB/RGBA source image, got {image.shape}.')
        if image.shape[2] == 4:
            image = image[:, :, :3]

        if mask is None:
            # A missing optional source mask means "keep the current canvas."
            # Directional expansion and the prepared BB mask still identify
            # the area to regenerate.
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            mask = np.asarray(mask)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            if mask.ndim != 2 or mask.shape != image.shape[:2]:
                raise ValueError(
                    'Outpaint mask must match the source image dimensions '
                    f'(image={image.shape[:2]}, mask={mask.shape}).'
                )

        # 1. Resolve Bounding Box
        # In Outpaint, the mask is automatically generated by Step 1 expansion.
        mask_indices = np.where(mask > 127)
        if len(mask_indices[0]) == 0:
            H, W = image.shape[:2]
            y1, y2, x1, x2 = 0, H, 0, W
        else:
            y1, y2 = np.min(mask_indices[0]), np.max(mask_indices[0])
            x1, x2 = np.min(mask_indices[1]), np.max(mask_indices[1])
        
        H, W = image.shape[:2]
        
        if outpaint_direction is not None:
            # Outpaint Auto-Context Strategy
            if outpaint_direction in ['left', 'right']:
                target_h = H
                best_res = min(self.SDXL_RESOLUTIONS, key=lambda r: abs(r[1] - target_h))
                target_w, target_h = best_res
                
                if outpaint_direction == 'right':
                    x2 = W
                    x1 = x2 - target_w
                else: # left
                    x1 = 0
                    x2 = target_w
                y1, y2 = 0, H
 
            else: # top or bottom
                target_w = W
                best_res = min(self.SDXL_RESOLUTIONS, key=lambda r: abs(r[0] - target_w))
                target_w, target_h = best_res
                
                if outpaint_direction == 'bottom':
                    y2 = H
                    y1 = y2 - target_h
                else: # top
                    y1 = 0
                    y2 = target_h
                x1, x2 = 0, W
                
            new_w, new_h = target_w, target_h
        else:
            # Fallback (e.g. Step 2 upload without explicit direction metadata)
            # Try to infer direction from mask
            mask_indices = np.where(mask > 127)
            if len(mask_indices[0]) > 0:
                my1, my2 = np.min(mask_indices[0]), np.max(mask_indices[0])
                mx1, mx2 = np.min(mask_indices[1]), np.max(mask_indices[1])
                
                # Heuristic: Find which edge the mask is most focused on
                # For Step 1 results (square 1216x1216 from 832x1216), 
                # the mask will be a 384px block on one side.
                if mx2 >= W - 8 and mx1 > W // 2:
                    outpaint_direction = 'right'
                elif mx1 <= 8 and mx2 < W // 2:
                    outpaint_direction = 'left'
                elif my2 >= H - 8 and my1 > H // 2:
                    outpaint_direction = 'bottom'
                elif my1 <= 8 and my2 < H // 2:
                    outpaint_direction = 'top'
 
            if outpaint_direction is not None:
                # Re-run with detected direction
                return self.prepare(image, mask, extend_factor, outpaint_direction, generate_blend_mask=generate_blend_mask)
            
            # Absolute fallback
            target_w, target_h = self.snap_to_sdxl_resolution(W, H)
            y1, y2, x1, x2 = 0, H, 0, W
            new_w, new_h = target_w, target_h
        
        # 5. Handle expansion and cropping
        bb_image, _ = self._expand_canvas(image, y1, y2, x1, x2)
        bb_mask, _ = self._expand_canvas(mask[:, :, None] if mask.ndim == 2 else mask, y1, y2, x1, x2)
        bb_mask = bb_mask[:, :, 0]
        
        # 6. Resize to target resolution
        bb_image = resample_image(bb_image, target_w, target_h)
        bb_mask = resample_image(bb_mask, target_w, target_h)
        
        # 7. Generate blend mask
        blend_mask = self._morphological_open(mask) if generate_blend_mask else None
        
        return OutpaintContext(
            original_image=image,
            original_mask=mask,
            bb=(y1, y2, x1, x2),
            bb_image=bb_image,
            bb_mask=bb_mask,
            target_w=target_w,
            target_h=target_h,
            blend_mask=blend_mask
        )

    def encode(self, context: OutpaintContext, vae) -> dict:
        """VAE encode BB image and generate latent-space denoise_mask."""
        from backend import resources
        
        # 1. BB Image to tensor
        pixels = numpy_to_pytorch(context.bb_image)
        
        # 2. VAE encode (lifecycle managed by core.encode_vae)
        latent = core.encode_vae(vae=vae, pixels=pixels)['samples']
        
        # 3. Create denoise_mask in latent space
        mask_np = np.asarray(context.bb_mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]
        if mask_np.ndim != 2 or mask_np.size == 0:
            raise ValueError(f"Invalid outpaint BB mask shape: {mask_np.shape}")
        mask = torch.from_numpy(mask_np).float() / 255.0
        mask = mask[None, None, :, :] # (1, 1, H, W)
        
        # Max pool to 1/8 resolution
        denoise_mask = torch.nn.functional.max_pool2d(mask, kernel_size=8)
        
        # Threshold to binary (1.0 = regenerate, 0.0 = freeze)
        denoise_mask = (denoise_mask > 0.5).float()
        
        return {'samples': latent, 'noise_mask': denoise_mask}

    def stitch(self, context: OutpaintContext, generated_image) -> np.ndarray:
        """Preserve Fooocus's superior morphological blending for Outpaint."""
        y1, y2, x1, x2 = context.bb
        target_w, target_h = x2 - x1, y2 - y1
        
        # 1. Resize back to original BB dimensions
        content = resample_image(generated_image, target_w, target_h)
        
        # 2. Hard-paste into a full-size canvas of the original image
        result = context.original_image.copy()
        
        # Handle cases where BB might be slightly outside bounds
        H, W = result.shape[:2]
        iy1, iy2 = max(0, y1), min(H, y2)
        ix1, ix2 = max(0, x1), min(W, x2)
        
        # If BB was expanded, slice content correctly
        content_y1, content_y2 = iy1 - y1, iy2 - y1
        content_x1, content_x2 = ix1 - x1, ix2 - x1
        
        result[iy1:iy2, ix1:ix2] = content[content_y1:content_y2, content_x1:content_x2]
        
        # 3. Apply morphological gradient blend at full-image resolution
        fg = result.astype(np.float32)
        bg = context.original_image.astype(np.float32)
        blend_mask = context.blend_mask
        if blend_mask is None:
            blend_mask = self._morphological_open(context.original_mask)
            context.blend_mask = blend_mask
        w = blend_mask[:, :, None].astype(np.float32) / 255.0
        w = blending.apply_sin2_curve(w)
        
        y = fg * w + bg * (1.0 - w)
        return y.clip(0, 255).astype(np.uint8)
