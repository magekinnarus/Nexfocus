from __future__ import annotations

import gc
import logging
import os
from typing import Any

import numpy as np
import torch

from backend import resources
from backend.auxiliary_workers.execution import auxiliary_execution
from backend.auxiliary_workers.telemetry import log_auxiliary_telemetry
from ldm_patched.pfn.architecture.MAT import MAT
from modules import model_registry
import modules.config as config
from modules.blending import sin_blend_1d


logger = logging.getLogger(__name__)


def mask_unsqueeze(mask: torch.Tensor) -> torch.Tensor:
    if len(mask.shape) == 3:  # BHW -> B1HW
        mask = mask.unsqueeze(1)
    elif len(mask.shape) == 2:  # HW -> B1HW
        mask = mask.unsqueeze(0).unsqueeze(0)
    return mask


def to_torch(image: np.ndarray, mask: np.ndarray | None = None, device: str | torch.device = "cpu"):
    """Convert neutral HWC/HW uint8 arrays to float32 BCHW tensors."""
    image_t = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
    image_t = image_t.unsqueeze(0).to(device)

    if mask is not None:
        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).float() / 255.0
        mask_t = mask_unsqueeze(mask_t).to(device)
        return image_t, mask_t
    return image_t


def mask_floor(mask: torch.Tensor, threshold: float = 0.99) -> torch.Tensor:
    return (mask >= threshold).to(mask.dtype)


def pad_reflect_once(x: torch.Tensor, original_padding: tuple[int, int, int, int]) -> torch.Tensor:
    _, _, h, w = x.shape
    padding = np.array(original_padding)
    size = np.array([w, w, h, h])
    initial_padding = np.minimum(padding, size - 1)
    additional_padding = padding - initial_padding

    x = torch.nn.functional.pad(x, tuple(initial_padding), mode="reflect")
    if np.any(additional_padding > 0):
        x = torch.nn.functional.pad(x, tuple(additional_padding), mode="constant")
    return x


def resize_square(image: torch.Tensor, mask: torch.Tensor, size: int):
    _, _, h, w = image.shape
    pad_w, pad_h, prev_size = 0, 0, w
    if w == size and h == size:
        return image, mask, (pad_w, pad_h, prev_size)

    if w < h:
        pad_w = h - w
        prev_size = h
    elif h < w:
        pad_h = w - h
        prev_size = w

    image = pad_reflect_once(image, (0, pad_w, 0, pad_h))
    mask = pad_reflect_once(mask, (0, pad_w, 0, pad_h))

    if image.shape[-1] != size:
        image = torch.nn.functional.interpolate(image, size=size, mode="nearest-exact")
        mask = torch.nn.functional.interpolate(mask, size=size, mode="nearest-exact")

    return image, mask, (pad_w, pad_h, prev_size)


def undo_resize_square(image: torch.Tensor, original_size: tuple[int, int, int]) -> torch.Tensor:
    _, _, h, w = image.shape
    pad_w, pad_h, prev_size = original_size
    if prev_size != w or prev_size != h:
        image = torch.nn.functional.interpolate(image, size=prev_size, mode="bilinear", align_corners=False)
    return image[:, :, 0 : prev_size - pad_h, 0 : prev_size - pad_w]


def get_segments(length: int, tile_size: int, overlap: int):
    if length <= tile_size:
        return [(0, length, 0, 0)]

    segments = [(0, tile_size - overlap, 0, overlap)]
    while segments[-1][1] < length:
        start = segments[-1][1]
        tile_start = start - overlap
        if tile_start + tile_size >= length:
            end = length
            final_tile_start = max(0, length - tile_size)
            pad_l = start - final_tile_start
            segments.append((start, end, pad_l, 0))
            break

        end = start + tile_size - overlap * 2
        segments.append((start, end, overlap, overlap))
    return segments


def _as_uint8_image(image: np.ndarray) -> np.ndarray:
    image_np = np.asarray(image)
    if image_np.ndim != 3 or image_np.shape[2] not in (3, 4):
        raise ValueError(f"MAT expects HWC RGB/RGBA input, got {image_np.shape}.")
    if image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]
    if image_np.dtype != np.uint8:
        image_np = image_np.astype(np.float32, copy=False)
        if image_np.size and float(image_np.max()) <= 1.0:
            image_np = image_np * 255.0
        image_np = image_np.clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(image_np)


def _as_uint8_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask_np = np.asarray(mask)
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]
    if mask_np.ndim != 2 or tuple(mask_np.shape) != tuple(shape):
        raise ValueError(f"MAT expects an HW mask matching {shape}, got {mask_np.shape}.")
    if mask_np.dtype != np.uint8:
        mask_np = mask_np.astype(np.float32, copy=False)
        if mask_np.size and float(mask_np.max()) <= 1.0:
            mask_np = mask_np * 255.0
        mask_np = mask_np.clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(mask_np)


class MatInpaintWorker:
    """Ephemeral MAT object-removal worker with CPU materialization."""

    asset_id = "removals.object.mat.places512"
    default_model_name = "Places_512_FullData_G.pth"

    def __init__(self) -> None:
        self.model: Any | None = None
        self.model_name: str | None = None
        self.checkpoint_path: str | None = None
        self.device = torch.device("cpu")

    def load(self, *, model_name: str = default_model_name) -> None:
        if self.model is not None:
            raise RuntimeError("MatInpaintWorker.load called while a model is loaded.")

        resolved_name = str(model_name or self.default_model_name)
        log_auxiliary_telemetry("mat_inpaint_worker_load_begin", f"model_name={resolved_name}")
        if resolved_name == self.default_model_name:
            checkpoint_path = model_registry.ensure_asset(self.asset_id, progress=True)
        else:
            checkpoint_path = resolved_name
            if not os.path.isabs(checkpoint_path):
                checkpoint_path = os.path.join(config.path_removals, checkpoint_path)
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Object removal model not found: {resolved_name}")

        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            new_state = {
                key.replace("synthesis", "model.synthesis").replace("mapping", "model.mapping"): value
                for key, value in state_dict.items()
            }
            model = MAT()
            model.load_state_dict(new_state)
            model.eval()
            # MAT is intentionally float32 for Pascal-safe inference.
            model.to(torch.float32)
            self.model = model
            self.model_name = resolved_name
            self.checkpoint_path = checkpoint_path
            self.device = torch.device("cpu")
        except BaseException:
            self.model = None
            self.model_name = None
            self.checkpoint_path = None
            raise

        log_auxiliary_telemetry("mat_inpaint_worker_load_complete", f"model_name={resolved_name}")

    def detach(self) -> None:
        model = self.model
        if model is None:
            return
        log_auxiliary_telemetry("mat_inpaint_worker_detach_begin")
        try:
            model.to(torch.device("cpu"))
        finally:
            self.device = torch.device("cpu")
            log_auxiliary_telemetry("mat_inpaint_worker_detach_complete")

    @torch.inference_mode()
    def infer(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        seed: int = 0,
        mask_dilate: int = 16,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("MatInpaintWorker.infer called before load.")

        image_np = _as_uint8_image(image)
        mask_np = _as_uint8_mask(mask, image_np.shape[:2])
        if int(mask_dilate) > 0:
            import cv2

            kernel_size = max(1, int(mask_dilate))
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask_np = cv2.dilate(mask_np, kernel, iterations=1)

        h, w, _ = image_np.shape
        log_auxiliary_telemetry(
            "mat_inpaint_worker_infer_begin",
            f"shape={tuple(image_np.shape)} seed={seed} mask_dilate={mask_dilate}",
        )
        device = resources.get_torch_device() if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(device)
        self.device = device
        torch.manual_seed(int(seed))

        try:
            if h <= 512 and w <= 512:
                img_t, mask_t = to_torch(image_np, mask_np, device=device)
                img_sq, mask_sq, orig_info = resize_square(img_t, mask_t, 512)
                mask_sq = mask_floor(mask_sq, 0.99)
                res_sq = self.model(img_sq, mask_sq)
                res_t = undo_resize_square(res_sq, orig_info)
                comp_mask = to_torch(np.zeros_like(image_np), mask_np, device=device)[1]
                final_t = img_t * (1.0 - comp_mask) + res_t * comp_mask
                result = (
                    final_t.squeeze(0)
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                    * 255
                ).clip(0, 255).astype(np.uint8)
            else:
                logger.info("Using tiled MAT processing for %sx%s image", w, h)
                tile_size = 512
                overlap = 64
                img_t, mask_t = to_torch(image_np, mask_np, device=device)
                weight_total = torch.zeros((1, 1, h, w), device=device)
                accum = torch.zeros_like(img_t)
                h_segs = get_segments(h, tile_size, overlap)
                w_segs = get_segments(w, tile_size, overlap)

                for y_start, y_end, y_pad_l, y_pad_r in h_segs:
                    for x_start, x_end, x_pad_l, x_pad_r in w_segs:
                        tile_y_start = y_start - y_pad_l
                        tile_x_start = x_start - x_pad_l
                        tile_img = img_t[:, :, tile_y_start : tile_y_start + tile_size, tile_x_start : tile_x_start + tile_size]
                        tile_mask = mask_t[:, :, tile_y_start : tile_y_start + tile_size, tile_x_start : tile_x_start + tile_size]

                        if torch.sum(tile_mask) < 1e-4:
                            tile_res = tile_img
                        else:
                            tile_res = self.model(tile_img, mask_floor(tile_mask, 0.99))

                        weight_map = torch.ones((1, 1, tile_size, tile_size), device=device)
                        if y_pad_l > 0:
                            weight_map[:, :, :y_pad_l, :] *= sin_blend_1d(y_pad_l, device).view(1, 1, -1, 1)
                        if y_pad_r > 0:
                            weight_map[:, :, -y_pad_r:, :] *= sin_blend_1d(y_pad_r, device).flip(0).view(1, 1, -1, 1)
                        if x_pad_l > 0:
                            weight_map[:, :, :, :x_pad_l] *= sin_blend_1d(x_pad_l, device).view(1, 1, 1, -1)
                        if x_pad_r > 0:
                            weight_map[:, :, :, -x_pad_r:] *= sin_blend_1d(x_pad_r, device).flip(0).view(1, 1, 1, -1)

                        accum[:, :, tile_y_start : tile_y_start + tile_size, tile_x_start : tile_x_start + tile_size] += tile_res * weight_map
                        weight_total[:, :, tile_y_start : tile_y_start + tile_size, tile_x_start : tile_x_start + tile_size] += weight_map

                tiled_result = accum / (weight_total + 1e-8)
                final_t = img_t * (1.0 - mask_t) + tiled_result * mask_t
                result = (
                    final_t.squeeze(0)
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                    * 255
                ).clip(0, 255).astype(np.uint8)

            result = np.ascontiguousarray(result)
            log_auxiliary_telemetry("mat_inpaint_worker_infer_complete", f"result_shape={tuple(result.shape)}")
            return result
        finally:
            self.detach()

    def teardown(self) -> None:
        log_auxiliary_telemetry("mat_inpaint_worker_teardown_begin")
        model = self.model
        try:
            self.detach()
        except Exception:
            logger.debug("MAT detach failed during teardown.", exc_info=True)
        self.model = None
        self.model_name = None
        self.checkpoint_path = None
        self.device = torch.device("cpu")
        if model is not None:
            del model

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        resources.soft_empty_cache()
        log_auxiliary_telemetry("mat_inpaint_worker_teardown_complete")


def run_mat_inpaint(
    image: np.ndarray | None,
    mask: np.ndarray | None,
    *,
    model_name: str = MatInpaintWorker.default_model_name,
    seed: int = 0,
    mask_dilate: int = 16,
) -> np.ndarray | None:
    """Run one MAT request with guaranteed worker and lease cleanup."""
    if image is None or mask is None:
        return None

    with auxiliary_execution("mat_inpaint"):
        worker = MatInpaintWorker()
        try:
            worker.load(model_name=model_name)
            return worker.infer(image, mask, seed=seed, mask_dilate=mask_dilate)
        finally:
            worker.teardown()
