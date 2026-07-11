from __future__ import annotations

import os
import gc
import logging

import numpy as np
import torch

import backend.resources as resources
import ldm_patched.pfn.model_loading as loading
from backend.auxiliary_workers.execution import auxiliary_execution
from backend.auxiliary_workers.telemetry import log_auxiliary_telemetry
from modules.upscale_engine import NexUpscaleEngine
from modules.config import path_upscale_models

logger = logging.getLogger(__name__)

class GanUpscaleWorker:
    """
    Ephemeral Layer 5 worker responsible for GAN upscaling.
    Owns explicit load, infer, and teardown phases with telemetry events.
    """
    def __init__(self) -> None:
        self.model = None
        self.model_name = None

    def load(self, model_name: str | None = None) -> None:
        """
        Load the GAN upscale model by name.
        """
        if self.model is not None:
            raise RuntimeError("GanUpscaleWorker.load called while a model is already loaded.")

        log_auxiliary_telemetry("gan_upscale_worker_load_begin", f"model_name={model_name}")

        if model_name is None or model_name == "None":
            # Fallback to default Nomos2 or first available
            from modules.upscaler import list_available_models
            available = list_available_models()
            if available:
                if '4xNomos2_otf_esrgan.pth' in available:
                    model_name = '4xNomos2_otf_esrgan.pth'
                else:
                    model_name = available[0]
            else:
                raise ValueError("No upscale models found and none specified.")

        model_path = None
        for folder in path_upscale_models:
            p = os.path.join(folder, model_name)
            if os.path.exists(p):
                model_path = p
                break
                
        if model_path is None:
            raise FileNotFoundError(f"Upscale model not found: {model_name}")
            
        logger.info(f"GanUpscaleWorker loading upscale model {model_path} ...")
        
        if model_path.endswith('.safetensors'):
            from ldm_patched.modules.utils import load_torch_file
            sd = load_torch_file(model_path, device='cpu')
        else:
            sd = torch.load(model_path, map_location='cpu', weights_only=True)
        
        model = loading.load_state_dict(sd)
        model.eval()
        model.cpu()
        
        self.model = model
        self.model_name = model_name

        log_auxiliary_telemetry("gan_upscale_worker_load_complete", f"model_name={model_name}")

    def get_native_scale(self) -> int:
        """Return model scale metadata while keeping the model worker-owned."""
        if self.model is None:
            raise RuntimeError("GanUpscaleWorker.get_native_scale called but no model is loaded.")
        for attr in ("scale", "upscale", "upscale_factor", "upsampler_scale"):
            value = getattr(self.model, attr, None)
            if isinstance(value, (int, float)):
                return int(value)
        return 4

    def infer(self, img: np.ndarray, scale_override: float | None = None) -> np.ndarray:
        """
        Run upscale inference using the loaded model.
        """
        if self.model is None:
            raise RuntimeError("GanUpscaleWorker.infer called but no model is loaded.")

        if img is None:
            return None

        log_auxiliary_telemetry("gan_upscale_worker_infer_begin", f"model_name={self.model_name} img_shape={img.shape}")

        device = resources.get_torch_device()
        self.model.float()
        self.model.to(device)

        # Detect scale and color space via Spandrel (chaiNNer standard)
        native_scale = self.get_native_scale()
        is_bgr = True
        architecture_id = None
        
        if hasattr(self.model, "architecture"):
            # Spandrel Unified Metadata
            native_scale = getattr(self.model, "scale", 4)
            arch_id = self.model.architecture.id
            architecture_id = str(arch_id)
            
            # Check tags and architecture for color space
            if "RGB" in self.model.tags:
                is_bgr = False
            elif "BGR" in self.model.tags:
                is_bgr = True
            elif any(x in arch_id for x in ["RealESRGANv2", "RealPLKSR", "SCET", "SwinIR", "HAT"]):
                is_bgr = False
                
        target_scale = scale_override if scale_override is not None else native_scale
        
        def upscale_fn(t):
            return self.model(t)
            
        print(f"[GanUpscaleWorker] Upscaling image via Nex-Engine (Native Space: {'BGR' if is_bgr else 'RGB'})...")
        
        # Get the model's actual dtype for perfect precision matching in the engine
        m_dtype = next(self.model.model.parameters()).dtype if hasattr(self.model, "model") else torch.float32
        model_parameters = (
            sum(parameter.numel() for parameter in self.model.model.parameters())
            if hasattr(self.model, "model")
            else sum(parameter.numel() for parameter in self.model.parameters())
        )
        log_auxiliary_telemetry(
            "gan_upscale_worker_plan",
            f"architecture={architecture_id or 'unknown'} native_scale={native_scale} "
            f"target_scale={target_scale} dtype={m_dtype} parameters={model_parameters}",
        )

        engine = NexUpscaleEngine()
        try:
            result = engine.process(
                img,
                upscale_fn,
                native_scale,
                device,
                is_bgr=is_bgr,
                dtype=m_dtype,
                model_params=model_parameters,
                architecture_id=architecture_id,
            )
        finally:
            # The worker, not the caller, owns device detachment on every path.
            self.model.cpu()

        log_auxiliary_telemetry(
            "gan_upscale_worker_native_complete",
            f"native_scale={native_scale} native_shape={result.shape}",
        )

        # Handle scale override via bicubic resize if needed
        if target_scale != native_scale:
            import cv2
            h, w = img.shape[:2]
            new_h, new_w = int(h * target_scale), int(w * target_scale)
            result = cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        log_auxiliary_telemetry("gan_upscale_worker_infer_complete", f"model_name={self.model_name} out_shape={result.shape}")
        return result

    def teardown(self) -> None:
        """
        Unload the model and reclaim resources completely.
        """
        log_auxiliary_telemetry("gan_upscale_worker_teardown_begin")

        model = self.model
        self.model = None
        self.model_name = None

        if model is not None:
            try:
                model.cpu()
            except Exception:
                logger.debug("GanUpscaleWorker CPU detach failed during teardown.", exc_info=True)
            del model

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        resources.soft_empty_cache()

        log_auxiliary_telemetry("gan_upscale_worker_teardown_complete")


def run_gan_upscale(
    img: np.ndarray | None,
    *,
    model_name: str | None = None,
    scale_override: float | None = None,
) -> np.ndarray | None:
    """Run one strictly ephemeral GAN worker inside an auxiliary lease."""
    if img is None:
        return None

    with auxiliary_execution("gan_upscale"):
        worker = GanUpscaleWorker()
        try:
            worker.load(model_name)
            return worker.infer(img, scale_override=scale_override)
        finally:
            worker.teardown()
