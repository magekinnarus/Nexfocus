from __future__ import annotations

import gc
import logging
import warnings
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from PIL import Image
from transparent_background import Remover

from backend import resources
from backend.auxiliary_workers.execution import auxiliary_execution
from backend.auxiliary_workers.telemetry import log_auxiliary_telemetry
from modules import model_registry


logger = logging.getLogger(__name__)


@contextmanager
def _background_removal_warning_scope():
    """Hide known tracing noise normally while preserving it in debug mode."""
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        yield
        return

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"torch\.meshgrid: in an upcoming release, it will be required to pass the indexing argument.*",
            category=UserWarning,
        )
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        yield


def _as_uint8_image(image: np.ndarray) -> np.ndarray:
    image_np = np.asarray(image)
    if image_np.ndim != 3 or image_np.shape[2] not in (3, 4):
        raise ValueError(f"Background removal expects HWC RGB/RGBA input, got {image_np.shape}.")

    if image_np.dtype != np.uint8:
        image_np = image_np.astype(np.float32, copy=False)
        if image_np.size and float(image_np.max()) <= 1.0:
            image_np = image_np * 255.0
        image_np = image_np.clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(image_np)


class BackgroundRemovalWorker:
    """Ephemeral InSPyReNet worker for one background-removal request."""

    asset_id = "removals.background.inspyrenet.base"

    def __init__(self) -> None:
        self.remover: Any | None = None
        self.jit: bool | None = None
        self.checkpoint_path: str | None = None

    def load(self, *, jit: bool = True) -> None:
        if self.remover is not None:
            raise RuntimeError("BackgroundRemovalWorker.load called while a model is loaded.")

        request_jit = bool(jit)
        log_auxiliary_telemetry(
            "background_removal_worker_load_begin",
            f"jit={request_jit}",
        )
        checkpoint_path = model_registry.ensure_asset(self.asset_id, progress=True)
        try:
            with _background_removal_warning_scope():
                self.remover = Remover(jit=request_jit, ckpt=checkpoint_path)
            self.jit = request_jit
            self.checkpoint_path = checkpoint_path
        except BaseException:
            self.remover = None
            self.jit = None
            self.checkpoint_path = None
            raise

        log_auxiliary_telemetry(
            "background_removal_worker_load_complete",
            f"jit={request_jit}",
        )

    def infer(self, image: np.ndarray, *, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        if self.remover is None:
            raise RuntimeError("BackgroundRemovalWorker.infer called before load.")

        image_np = _as_uint8_image(image)
        log_auxiliary_telemetry(
            "background_removal_worker_infer_begin",
            f"shape={tuple(image_np.shape)} threshold={threshold}",
        )

        try:
            pil_image = Image.fromarray(image_np)
            with _background_removal_warning_scope():
                result = self.remover.process(
                    pil_image,
                    type="rgba",
                    threshold=float(threshold),
                )
            if isinstance(result, Image.Image):
                rgba = np.asarray(result.convert("RGBA"))
            else:
                rgba = np.asarray(result)
                if rgba.ndim != 3 or rgba.shape[2] not in (3, 4):
                    raise ValueError(f"InSPyReNet returned an invalid image shape: {rgba.shape}.")
                if rgba.shape[2] == 3:
                    alpha = np.full(rgba.shape[:2] + (1,), 255, dtype=rgba.dtype)
                    rgba = np.concatenate((rgba, alpha), axis=2)
                rgba = rgba.astype(np.uint8, copy=False)

            rgba = np.ascontiguousarray(rgba.astype(np.uint8, copy=False))
            binary_mask = (rgba[:, :, 3] > 127).astype(np.uint8) * 255
            log_auxiliary_telemetry(
                "background_removal_worker_infer_complete",
                f"rgba_shape={tuple(rgba.shape)} mask_shape={tuple(binary_mask.shape)}",
            )
            return rgba, np.ascontiguousarray(binary_mask)
        finally:
            # Remover implementations differ in how they expose their inner
            # network. Detach what is exposed; teardown still drops the owner.
            self.detach()

    def detach(self) -> None:
        remover = self.remover
        if remover is None:
            return

        log_auxiliary_telemetry("background_removal_worker_detach_begin")
        candidates = [
            getattr(remover, "model", None),
            getattr(remover, "net", None),
            remover,
        ]
        seen: set[int] = set()
        for candidate in candidates:
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            try:
                if hasattr(candidate, "cpu"):
                    candidate.cpu()
                elif hasattr(candidate, "to"):
                    candidate.to("cpu")
            except Exception:
                logger.debug("Background-removal device detach failed.", exc_info=True)
        log_auxiliary_telemetry("background_removal_worker_detach_complete")

    def teardown(self) -> None:
        log_auxiliary_telemetry("background_removal_worker_teardown_begin")
        remover = self.remover
        self.detach()
        self.remover = None
        self.jit = None
        self.checkpoint_path = None

        if remover is not None:
            del remover

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        resources.soft_empty_cache()
        log_auxiliary_telemetry("background_removal_worker_teardown_complete")


def run_background_removal(
    image: np.ndarray | None,
    *,
    threshold: float = 0.5,
    jit: bool = True,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run one BGR request with guaranteed worker and lease cleanup."""
    if image is None:
        return None

    with auxiliary_execution("background_removal"):
        worker = BackgroundRemovalWorker()
        try:
            worker.load(jit=jit)
            return worker.infer(image, threshold=threshold)
        finally:
            worker.teardown()
