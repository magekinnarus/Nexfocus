from __future__ import annotations

import time
import logging
import numpy as np
import torch
from typing import Any, Dict, Tuple, Optional
from backend.sdxl_assembly.contracts import (
    SDXLAssemblyRequest,
    SpatialContextDescriptor,
    PreparedSpatialContext,
    SpatialImageDescriptor,
    SpatialMaskDescriptor
)
from backend.sdxl_assembly.progress import log_telemetry

logger = logging.getLogger(__name__)

class SpatialContextWorker:
    """Worker representing SpatialContextWorker (owns CPU-side spatial preparation)."""

    def __init__(self, request: SDXLAssemblyRequest) -> None:
        self.request = request

    def _mask_bbox(self, mask: torch.Tensor) -> tuple[int, int, int, int]:
        # mask is [B, H, W]
        # Find active bounding box
        active = torch.nonzero(mask > 0.5, as_tuple=False)
        if active.numel() == 0:
            return (0, int(mask.shape[1]), 0, int(mask.shape[2]))
        y1 = int(active[:, 1].min().item())
        y2 = int(active[:, 1].max().item()) + 1
        x1 = int(active[:, 2].min().item())
        x2 = int(active[:, 2].max().item()) + 1
        return (y1, y2, x1, x2)

    def _bbox_area_ratio(self, bbox: tuple[int, int, int, int], height: int, width: int) -> float:
        y1, y2, x1, x2 = bbox
        bbox_area = max(0, y2 - y1) * max(0, x2 - x1)
        full_area = max(1, int(height) * int(width))
        return float(bbox_area) / float(full_area)

    def _crop_and_resize_pixels(
        self,
        pixels: torch.Tensor,
        bbox: tuple[int, int, int, int],
        target_height: int,
        target_width: int,
    ) -> torch.Tensor:
        y1, y2, x1, x2 = bbox
        crop = pixels[:, y1:y2, x1:x2, :].movedim(-1, 1)
        resized = torch.nn.functional.interpolate(
            crop,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        return resized.movedim(1, -1).contiguous().cpu()

    def _crop_and_resize_mask(
        self,
        mask: torch.Tensor,
        bbox: tuple[int, int, int, int],
        target_height: int,
        target_width: int,
    ) -> torch.Tensor:
        y1, y2, x1, x2 = bbox
        crop = mask[:, None, y1:y2, x1:x2]
        resized = torch.nn.functional.interpolate(
            crop,
            size=(target_height, target_width),
            mode="nearest",
        )
        return resized[:, 0, :, :].contiguous().cpu()

    def _build_fullres_blend_mask(self, mask: torch.Tensor) -> torch.Tensor:
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError("OpenCV is required to build blend masks.") from exc

        outputs: list[torch.Tensor] = []
        for item in mask.detach().cpu():
            mask_np = np.asarray((item > 0.5).to(dtype=torch.uint8).numpy() * 255, dtype=np.uint8)
            x_int16 = np.zeros_like(mask_np, dtype=np.int16)
            x_int16[mask_np > 127] = 256
            kernel = np.ones((3, 3), dtype=np.int16)
            for _ in range(32):
                maxed = cv2.dilate(x_int16, kernel) - 8
                x_int16 = np.maximum(maxed, x_int16)
            outputs.append(torch.from_numpy(np.clip(x_int16, 0, 255).astype(np.float32) / 255.0))
        return torch.stack(outputs, dim=0).contiguous()

    def prepare(self) -> PreparedSpatialContext:
        log_telemetry("spatial_prepare_begin")
        desc = self.request.spatial_context
        if desc is None:
            raise ValueError("No spatial_context is active on this request.")

        # Input validation
        if desc.mode not in ("image", "inpaint", "outpaint"):
            raise ValueError(f"Unsupported spatial route mode: {desc.mode}")

        mode = desc.mode
        pixels = desc.source_image.pixels
        mask = desc.source_mask.mask if desc.source_mask else None
        
        target_width = desc.target_width
        target_height = desc.target_height

        # Initialize variables
        bb_pixels = None
        bb_mask = None
        blend_mask = None
        bbox = (0, pixels.shape[1], 0, pixels.shape[2])
        bbox_area_ratio = 1.0
        mask_coverage = 0.0
        working_pixels = None
        working_mask = None

        if mode == "image":
            # If a pre-prepared image is provided, use it
            if desc.pre_bb_image is not None:
                bb_pixels = desc.pre_bb_image.pixels
                bb_mask = desc.pre_bb_mask.mask if desc.pre_bb_mask else None
                blend_mask = desc.pre_blend_mask.mask if desc.pre_blend_mask else None
                bbox = desc.bbox if desc.bbox else bbox
                bbox_area_ratio = desc.bbox_area_ratio
                mask_coverage = float(bb_mask.mean().item()) if bb_mask is not None and bb_mask.numel() else 0.0
            else:
                if mask is not None:
                    bbox = desc.bbox if desc.bbox else self._mask_bbox(mask)
                    bbox_area_ratio = self._bbox_area_ratio(bbox, pixels.shape[1], pixels.shape[2])
                    mask_coverage = float(mask.mean().item()) if mask.numel() else 0.0
                    blend_mask = self._build_fullres_blend_mask(mask)
                    
                    # Create masked pixels (pixels * (1 - mask) + 0.5 * mask)
                    mask_unsqueezed = mask.unsqueeze(-1)  # [B, H, W, 1]
                    masked_pixels = pixels * (1.0 - mask_unsqueezed) + 0.5 * mask_unsqueezed
                    
                    bb_pixels = self._crop_and_resize_pixels(masked_pixels, bbox, target_height, target_width)
                    bb_mask = self._crop_and_resize_mask(mask, bbox, target_height, target_width)
                else:
                    # No mask, bb_pixels is pixels
                    bb_pixels = pixels
                    bb_mask = None
                    blend_mask = None
                    bbox = (0, pixels.shape[1], 0, pixels.shape[2])
                    bbox_area_ratio = 1.0
                    mask_coverage = 0.0

        elif mode == "inpaint":
            if desc.pre_bb_image is not None:
                bb_pixels = desc.pre_bb_image.pixels
                bb_mask = desc.pre_bb_mask.mask if desc.pre_bb_mask else None
                blend_mask = desc.pre_blend_mask.mask if desc.pre_blend_mask else None
                bbox = desc.bbox if desc.bbox else bbox
                bbox_area_ratio = desc.bbox_area_ratio
                mask_coverage = float(bb_mask.mean().item()) if bb_mask is not None and bb_mask.numel() else 0.0
            else:
                # Build context on the fly (e.g. for testing/probes)
                from modules.pipeline.inpaint import InpaintPipeline
                inpaint = InpaintPipeline()
                
                # Input image as numpy
                image_np = (pixels[0].clamp(0.0, 1.0).mul(255.0).round().to(dtype=torch.uint8).numpy())
                mask_np = (mask[0].round().to(dtype=torch.uint8).numpy() * 255) if mask is not None else None
                
                ctx = inpaint.prepare(
                    image=image_np,
                    mask=mask_np,
                    extend_factor=1.2,
                )
                
                # Set outputs from prepared context
                bb_pixels_np = ctx.bb_image
                bb_mask_np = ctx.bb_mask
                blend_mask_np = ctx.blend_mask
                bbox = tuple(int(v) for v in ctx.bb)
                
                # Convert back to torch Tensors on CPU
                bb_pixels = torch.from_numpy(bb_pixels_np).unsqueeze(0).to(dtype=torch.float32) / 255.0
                bb_mask = torch.from_numpy(bb_mask_np).unsqueeze(0).to(dtype=torch.float32) / 255.0
                bb_mask = (bb_mask > 0.5).to(dtype=torch.float32)
                
                if blend_mask_np is not None:
                    blend_mask = torch.from_numpy(blend_mask_np).unsqueeze(0).to(dtype=torch.float32) / 255.0
                
                bbox_area_ratio = self._bbox_area_ratio(bbox, pixels.shape[1], pixels.shape[2])
                mask_coverage = float(bb_mask.mean().item()) if bb_mask.numel() else 0.0

        elif mode == "outpaint":
            if desc.pre_bb_image is not None:
                bb_pixels = desc.pre_bb_image.pixels
                bb_mask = desc.pre_bb_mask.mask if desc.pre_bb_mask else None
                blend_mask = desc.pre_blend_mask.mask if desc.pre_blend_mask else None
                bbox = desc.bbox if desc.bbox else bbox
                bbox_area_ratio = desc.bbox_area_ratio
                mask_coverage = float(bb_mask.mean().item()) if bb_mask is not None and bb_mask.numel() else 0.0
            else:
                # Build context on the fly (e.g. for testing/probes)
                from modules.pipeline.outpaint import OutpaintPipeline
                outpaint = OutpaintPipeline()
                
                image_np = (pixels[0].clamp(0.0, 1.0).mul(255.0).round().to(dtype=torch.uint8).numpy())
                mask_np = (mask[0].round().to(dtype=torch.uint8).numpy() * 255) if mask is not None else None
                
                direction = desc.outpaint_direction
                if mask_np is None:
                    if direction is None:
                        raise ValueError("Outpaint preparation requires outpaint_direction when mask is absent.")
                    working_image, working_mask = outpaint.prepare_outpaint_canvas_only(
                        image_np,
                        direction,
                        expansion_size=desc.outpaint_expansion_size,
                        pixelate=desc.outpaint_pixelate,
                    )
                    ctx = outpaint.prepare(
                        image=working_image,
                        mask=working_mask,
                        outpaint_direction=None,
                        extend_factor=1.2,
                    )
                else:
                    working_image = image_np
                    working_mask = mask_np
                    ctx = outpaint.prepare(
                        image=working_image,
                        mask=working_mask,
                        outpaint_direction=direction,
                        extend_factor=1.2,
                    )
                
                # Set outputs from prepared context
                bb_pixels_np = ctx.bb_image
                bb_mask_np = ctx.bb_mask
                blend_mask_np = ctx.blend_mask
                bbox = tuple(int(v) for v in ctx.bb)
                
                bb_pixels = torch.from_numpy(bb_pixels_np).unsqueeze(0).to(dtype=torch.float32) / 255.0
                bb_mask = torch.from_numpy(bb_mask_np).unsqueeze(0).to(dtype=torch.float32) / 255.0
                bb_mask = (bb_mask > 0.5).to(dtype=torch.float32)
                
                if blend_mask_np is not None:
                    blend_mask = torch.from_numpy(blend_mask_np).unsqueeze(0).to(dtype=torch.float32) / 255.0
                
                working_pixels = torch.from_numpy(working_image).unsqueeze(0).to(dtype=torch.float32) / 255.0
                working_mask = torch.from_numpy(working_mask).unsqueeze(0).to(dtype=torch.float32) / 255.0
                
                bbox_area_ratio = self._bbox_area_ratio(bbox, pixels.shape[1], pixels.shape[2])
                mask_coverage = float(bb_mask.mean().item()) if bb_mask.numel() else 0.0

        # Compute fingerprints for PreparedSpatialContext
        import hashlib
        def _hash_tensor(t: torch.Tensor | None) -> str:
            if t is None:
                return ""
            return hashlib.sha256(t.numpy().tobytes()).hexdigest()

        bb_pixels_fingerprint = _hash_tensor(bb_pixels)
        bb_mask_fingerprint = _hash_tensor(bb_mask) if bb_mask is not None else None

        prepared_ctx = PreparedSpatialContext(
            mode=mode,
            original_pixels=pixels,
            original_mask=mask,
            bb_pixels=bb_pixels,
            bb_mask=bb_mask,
            blend_mask=blend_mask,
            working_pixels=working_pixels,
            working_mask=working_mask,
            bbox=bbox,
            bbox_area_ratio=bbox_area_ratio,
            mask_coverage=mask_coverage,
            image_fingerprint=desc.source_image.fingerprint,
            mask_fingerprint=desc.source_mask.fingerprint if desc.source_mask else None,
            bb_pixels_fingerprint=bb_pixels_fingerprint,
            bb_mask_fingerprint=bb_mask_fingerprint,
        )
        
        log_telemetry(
            "spatial_prepare_complete",
            f"mode={mode} bbox={bbox} bbox_ratio={bbox_area_ratio:.3f} coverage={mask_coverage:.3f}"
        )
        return prepared_ctx

    def teardown_assembly_order(self) -> None:
        """Extension point for tracking teardown order."""
        pass
