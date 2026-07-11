from __future__ import annotations

import logging
import math
import torch
from backend.sdxl_assembly.contracts import ColorExtractionSpec
from backend.sdxl_assembly.progress import log_telemetry

logger = logging.getLogger(__name__)

class ColorExtractionWorker:
    """Run-bound parameter-correction overlay worker for the SDXL assembly.
    Owns x_center, sigma_max, and correction parameters.
    """
    def __init__(self, spec: ColorExtractionSpec) -> None:
        self.spec = spec
        self.x_center = None
        self.sigma_max = 0.0
        self._apply_count = 0

    def prepare(self, x_center: torch.Tensor, sigma_max: float) -> None:
        """Prepare the worker with the x_center latent tensor and the maximum sigma."""
        if not torch.isfinite(x_center).all():
            raise RuntimeError("Color enhancement received a non-finite center latent.")
        self.x_center = x_center
        self.sigma_max = float(sigma_max)
        self._apply_count = 0
        log_telemetry("color_overlay_prepare", f"sigma_max={self.sigma_max:.4f}")

    def correct_denoised(self, denoised: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
        """Applies the x_center correction after CFG denoising."""
        if (
            not self.spec.enabled
            or self.x_center is None
            or not math.isfinite(self.sigma_max)
            or self.sigma_max <= 0
        ):
            return denoised

        restore_cfg = self.spec.restore_cfg
        restore_cfg_s_tmin = self.spec.restore_cfg_s_tmin

        if restore_cfg > 0:
            sigma_tensor = torch.as_tensor(sigma, device=denoised.device, dtype=denoised.dtype)
            if sigma_tensor.numel() == 0 or not torch.isfinite(sigma_tensor).all():
                return denoised

            batch_size = int(denoised.shape[0]) if denoised.ndim > 0 else 1
            if sigma_tensor.numel() not in (1, batch_size):
                return denoised

            if sigma_tensor.numel() == 1:
                sigma_tensor = sigma_tensor.reshape(1)
            else:
                sigma_tensor = sigma_tensor.reshape(batch_size)

            active = sigma_tensor > restore_cfg_s_tmin
            if not bool(active.any()):
                return denoised

            ratio = torch.clamp(sigma_tensor / self.sigma_max, min=0.0, max=1.0)
            factor = ratio.pow(restore_cfg).reshape((-1,) + (1,) * (denoised.ndim - 1))
            if factor.shape[0] == 1 and batch_size != 1:
                factor = factor.expand((batch_size,) + factor.shape[1:])
            factor = factor * active.reshape((-1,) + (1,) * (denoised.ndim - 1)).to(denoised.dtype)

            # Compute the convex blend in fp32. The algebraically equivalent
            # fp16 subtraction form can overflow when finite endpoints have
            # opposite large values, turning the entire sampled latent into
            # NaNs by decode time.
            denoised_fp32 = denoised.to(dtype=torch.float32)
            x_center_fp32 = self.x_center.to(device=denoised.device, dtype=torch.float32)
            factor_fp32 = factor.to(dtype=torch.float32)
            corrected_fp32 = torch.lerp(denoised_fp32, x_center_fp32, factor_fp32)
            if not torch.isfinite(corrected_fp32).all():
                log_telemetry("color_overlay_nonfinite", "action=use_uncorrected_denoised")
                return denoised
            denoised = corrected_fp32.to(dtype=denoised.dtype)
            self._apply_count += 1
            if self._apply_count == 1:
                log_telemetry("color_overlay_apply")

        return denoised

    def close(self) -> None:
        """Releases the run-bound x_center reference."""
        if self._apply_count:
            log_telemetry("color_overlay_apply_complete", f"count={self._apply_count}")
        self.x_center = None
        self.sigma_max = 0.0
        self._apply_count = 0
        log_telemetry("color_overlay_close")
