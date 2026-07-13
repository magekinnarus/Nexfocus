from __future__ import annotations

import logging
from typing import Any
from backend.sdxl_assembly.contracts import SDXLAssemblyRequest
from backend.sdxl_assembly.progress import log_telemetry
from backend.cpu_compiler import SafeOpenHeaderOnly
from backend import lora as backend_lora

logger = logging.getLogger(__name__)

class GpuLoraWorker:
    """Worker representing GpuLoraWorker (materializes patches directly in GPU VRAM)."""
    
    def __init__(self, request: SDXLAssemblyRequest) -> None:
        self.request = request
        self.unet_patch_count = 0
        self.clip_patch_count = 0  # In W12a, CLIP-side GPU LoRA branch is inactive
        self.patch_artifact: dict[str, Any] | None = None
        self.last_compile_metrics: dict[str, Any] | None = None

    def materialize_patches(self) -> Any:
        """Sets up LoRA metadata for resident telemetry."""
        log_telemetry("resident_lora_parse")
        specs_len = len(self.request.lora_specs)
        if specs_len == 0:
            self.patch_artifact = {
                "kind": "identity",
                "stack_hash": self.request.lora_stack_hash,
                "unet_patch_count": 0,
                "clip_patch_count": 0,
            }
        else:
            self.patch_artifact = {
                "kind": "lora_stack",
                "stack_hash": self.request.lora_stack_hash,
                "specs": specs_len,
            }
        return self

    def apply_unet_patches(self, unet: Any) -> int:
        """Parses and applies UNet-side patches to the model patcher."""
        active_specs = [
            spec
            for spec in self.request.lora_specs
            if spec.enabled and spec.unet_weight != 0.0
        ]
        if not active_specs:
            self.unet_patch_count = 0
            if self.patch_artifact is not None:
                self.patch_artifact["unet_patch_count"] = 0
            return 0
        
        model = unet.model
        key_map = backend_lora.model_lora_keys_unet(model)
        patch_count = 0
        
        for spec in active_specs:
            lora_path = str(spec.file_identity.path)
            log_telemetry("resident_lora_parse", f"target=unet path={spec.file_identity.path.name}")
            header = SafeOpenHeaderOnly(lora_path)
            patch_dict = backend_lora.load_lora(header, key_map, log_missing=False)
            if patch_dict is None:
                patch_dict = {}

            if patch_dict:
                unet.add_patches(patch_dict, spec.unet_weight)
                patch_count += len(patch_dict)
                
        self.unet_patch_count = patch_count
        if self.patch_artifact is not None:
            self.patch_artifact["unet_patch_count"] = patch_count
        return patch_count

    def compile_unet_patches(self, unet: Any) -> dict[str, Any]:
        """Runs sequential GPU compilation on the applied patcher patches."""
        if not self.unet_patch_count:
            metrics = {"status": "noop", "patch_count": 0, "materialized_patch_keys": 0, "host_pinned_bytes": 0}
            self.last_compile_metrics = metrics
            return metrics

        import torch
        from backend.gpu_compiler import GpuArtifactCompiler

        target_device = torch.device(self.request.device)
        logger.debug("[SDXL Telemetry] Compiling UNet on GPU (device=%s)...", target_device)
        log_telemetry("resident_lora_compile_begin", f"device={target_device}")
        
        if target_device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats(target_device)
            except Exception:
                pass

        compile_metrics = GpuArtifactCompiler.compile_patcher(
            unet,
            clean_source=None,
            target_device=target_device,
            intermediate_dtype=torch.float16,
        )

        if target_device.type == "cuda" and torch.cuda.is_available():
            try:
                compile_metrics["cuda_temp_peak_bytes"] = int(torch.cuda.max_memory_allocated(target_device))
            except Exception:
                pass
        compile_metrics["cleared_patch_count"] = int(self.unet_patch_count)
        self.last_compile_metrics = dict(compile_metrics)
        if self.patch_artifact is not None:
            self.patch_artifact["compile_metrics"] = dict(compile_metrics)

        log_telemetry("resident_lora_compile_complete", f"device={target_device} metrics={compile_metrics}")
        return compile_metrics

    def apply_clip_patches(self, clip: Any) -> int:
        """CLIP branch of GpuLoraWorker.
        
        In W12a, only the UNet branch is active. This is a target-aware stub
        allowing W12c to plug in GPU CLIP patching when GPU text encoder is active.
        """
        # In W12a, CPU text worker is used, which patches CLIP on CPU.
        # This GPU CLIP branch remains inactive in W12a.
        return 0

    def teardown_assembly_order(self) -> None:
        self.patch_artifact = None
