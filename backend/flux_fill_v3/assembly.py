from __future__ import annotations

import time
import logging
from typing import Any
import numpy as np
import torch

from backend.flux_fill_v3.contracts import (
    FluxFillRequest,
    FluxFillResult,
    FluxRuntimeIdentity,
    UNetSpineKind,
    T5PostureKind,
    VAEPostureKind,
)
from backend.flux_fill_v3.streaming_spine import StreamingUnetSpine
from backend.flux_fill_v3.resident_spine import ResidentUnetSpine
from backend.flux_fill_v3.t5_worker import DiskPagedTextWorker
from backend.flux_fill_v3.vae_worker import TransientVaeWorker

logger = logging.getLogger(__name__)


class FluxAssembly:
    """Coordinates worker units and artifact flow for a specific posture combination."""

    def __init__(
        self,
        spine: StreamingUnetSpine | ResidentUnetSpine,
        text_worker: DiskPagedTextWorker,
        vae_worker: TransientVaeWorker,
        *,
        release_spine_after_execute: bool = True,
        status_callback=None,
        progress_state=None,
    ) -> None:
        self.spine = spine
        self.text_worker = text_worker
        self.vae_worker = vae_worker
        self.release_spine_after_execute = bool(release_spine_after_execute)
        self.status_callback = status_callback
        self.progress_state = progress_state

    def _report_status(self, text: str) -> None:
        if self.status_callback is None or self.progress_state is None:
            return
        self.status_callback(
            self.progress_state,
            int(getattr(self.progress_state, 'current_progress', 0) or 0),
            text,
        )

    def execute(self, request: FluxFillRequest, callback: Any | None = None) -> FluxFillResult:
        request.validate_dispatch_ready(require_existing_assets=True)
        device = torch.device(request.device) if request.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        timings: dict[str, float] = {}

        # 1. Resolve text conditioning first so prompt-artifact failures do not
        # force us to spend VAE work or hold extra latent memory beforehand.
        self._report_status('Encoding prompt ...')
        empty_cond = self.text_worker.get_conditioning()

        # 2. Coordinate VAE worker to check cache or encode latents
        self._report_status('Encoding source image ...')
        bundle = self.vae_worker.prepare_latents(device)
        timings["vae_load_encode"] = bundle.vae_load_time
        timings["vae_encode"] = bundle.vae_encode_time

        # 3. Coordinate UNet spine denoise
        unet_start = time.perf_counter()
        self.spine.start()
        timings["unet_start"] = time.perf_counter() - unet_start

        try:
            denoise_start = time.perf_counter()
            self._report_status('Starting inference ...')
            samples, sigmas = self.spine.denoise(
                bundle, empty_cond, callback=callback
            )
            timings["unet_denoise"] = time.perf_counter() - denoise_start
        finally:
            if self.release_spine_after_execute:
                self.spine.end()

        # 4. Decode results using VAE worker. Inpaint stitch-back is route-owned,
        # so the assembly does not perform any latent-space preservation blend.
        output_image, vae_load_decode, vae_decode = self.vae_worker.decode(samples, device)
        timings["vae_load_decode"] = vae_load_decode
        timings["vae_decode"] = vae_decode

        # 5. Apply final morphological stitching if requested
        if request.blend_mode == "morphological" and request.image is not None and request.mask is not None:
            stitch_start = time.perf_counter()
            output_image = self._stitch_image(request.image, request.mask, output_image)
            timings["stitch"] = time.perf_counter() - stitch_start

        unet_spine_kind = UNetSpineKind.RESIDENT if isinstance(self.spine, ResidentUnetSpine) else UNetSpineKind.STREAMING
        runtime_identity = FluxRuntimeIdentity(
            unet_spine=unet_spine_kind,
            t5_posture=request.t5_posture,
            vae_posture=VAEPostureKind.TRANSIENT,
        )

        return FluxFillResult(
            output_image=output_image,
            seed=request.seed,
            width=output_image.shape[1],
            height=output_image.shape[0],
            runtime_identity=runtime_identity,
            timings=timings,
            metadata={
                "runtime_identity": runtime_identity.as_dict(),
                "conditioning_contract": "prompt_conditioning" if request.prompt else "empty_conditioning_only",
                "category": str(request.category or ""),
            },
        )

    def _normalize_mask_2d(self, mask: np.ndarray | None) -> np.ndarray | None:
        if mask is None:
            return None
        raw_mask = np.asarray(mask)
        if raw_mask.ndim == 3:
            raw_mask = raw_mask[:, :, 0]
        if raw_mask.ndim != 2:
            return None
        return raw_mask.astype(np.uint8, copy=False)

    def _build_morphological_alpha(self, mask: np.ndarray | None) -> np.ndarray | None:
        import cv2
        import modules.blending as blending

        raw_mask = self._normalize_mask_2d(mask)
        if raw_mask is None:
            return None

        x_int16 = np.zeros_like(raw_mask, dtype=np.int16)
        x_int16[raw_mask > 127] = 256
        kernel = np.ones((3, 3), dtype=np.int16)
        for _ in range(32):
            maxed = cv2.dilate(x_int16, kernel) - 8
            x_int16 = np.maximum(maxed, x_int16)
        alpha = np.clip(x_int16, 0, 255).astype(np.float32) / 255.0
        return blending.apply_sin2_curve(alpha)

    def _stitch_image(self, original_image: np.ndarray, mask: np.ndarray, generated_image: np.ndarray) -> np.ndarray:
        canvas = original_image.copy().astype(np.float32)
        generated = generated_image.astype(np.float32)
        alpha_2d = self._build_morphological_alpha(mask)
        if alpha_2d is None:
            return np.clip(generated, 0, 255).astype(np.uint8)
        alpha = alpha_2d[..., np.newaxis]

        merged = (generated * alpha) + (canvas * (1.0 - alpha))
        return np.clip(merged, 0, 255).astype(np.uint8)
