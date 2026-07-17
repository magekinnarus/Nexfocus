from __future__ import annotations

import logging
import os
from typing import Any

import torch
from safetensors import safe_open

from backend.sdxl_assembly.contracts import SDXLAssemblyRequest
from backend.sdxl_assembly.progress import log_telemetry
from backend.cpu_compiler import SafeOpenHeaderOnly
from backend import lora as backend_lora

logger = logging.getLogger(__name__)


class _ResidentLazyWeight:
    def __init__(
        self,
        path: str,
        key: str,
        shape: list[int],
        dtype: str,
        *,
        tensor_device: torch.device,
        load_strategy: str = "safetensors",
    ) -> None:
        self.path = path
        self.key = key
        self.shape = list(shape)
        self.dtype = str(dtype)
        self._tensor_device = torch.device(tensor_device)
        self._load_strategy = str(load_strategy)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def load(self) -> torch.Tensor:
        if self._load_strategy == "safetensors":
            with safe_open(self.path, framework="pt", device=str(self._tensor_device)) as handle:
                return handle.get_tensor(self.key)
        try:
            sd = torch.load(self.path, map_location="cpu", weights_only=True)
        except Exception:
            sd = torch.load(self.path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        tensor = sd.get(self.key) if isinstance(sd, dict) else None
        if not isinstance(tensor, torch.Tensor):
            raise KeyError(f"Legacy tensor key {self.key!r} could not be reloaded from {self.path!r}.")
        return tensor.to(device=self._tensor_device)

    def item(self):
        return self.load().item()


class _ResidentSafeOpenHeaderOnly(dict):
    def __init__(self, path: str, *, tensor_device: torch.device) -> None:
        super().__init__()
        self.path = path
        self.tensor_device = torch.device(tensor_device)
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    try:
                        slice_view = handle.get_slice(key)
                        shape = list(slice_view.get_shape())
                        dtype = str(slice_view.get_dtype())
                    except Exception:
                        tensor = handle.get_tensor(key)
                        shape = list(tensor.shape)
                        dtype = str(tensor.dtype)
                    self[key] = _ResidentLazyWeight(
                        path,
                        key,
                        shape,
                        dtype,
                        tensor_device=self.tensor_device,
                        load_strategy="safetensors",
                    )
        except Exception:
            try:
                sd = torch.load(path, map_location="cpu", weights_only=True)
            except Exception:
                sd = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            for key, tensor in sd.items():
                if isinstance(tensor, torch.Tensor):
                    self[key] = _ResidentLazyWeight(
                        path,
                        key,
                        list(tensor.shape),
                        str(tensor.dtype),
                        tensor_device=self.tensor_device,
                        load_strategy="torch_load",
                    )
                else:
                    self[key] = tensor


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

    def _open_unet_lora_header(self, lora_path: str):
        if os.path.isfile(lora_path):
            try:
                return _ResidentSafeOpenHeaderOnly(
                    lora_path,
                    tensor_device=torch.device(self.request.device),
                )
            except Exception:
                logger.debug("Falling back to shared LoRA header loader for %s.", lora_path, exc_info=True)
        return SafeOpenHeaderOnly(lora_path)

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
            header = self._open_unet_lora_header(lora_path)
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

    def resolve_clip_patches(self, clip: Any) -> tuple[tuple[dict, float], ...]:
        """Parse CLIP-side patches without mutating or cloning the text encoder."""
        if not self.request.lora_specs:
            self.clip_patch_count = 0
            return ()

        from backend.sdxl_assembly.cpu_lora_worker import _PARSED_LORA_CACHE, _PARSED_LORA_CACHE_LIMIT

        patcher = clip.patcher
        model = patcher.model
        key_map = backend_lora.model_lora_keys_clip(model)
        model_class_name = model.__class__.__name__
        resolved_patches = []

        for spec in self.request.lora_specs:
            if not spec.enabled or spec.clip_weight == 0.0:
                continue

            lora_path = str(spec.file_identity.path)
            cache_key = (lora_path, "clip", model_class_name)
            current_hash = spec.file_identity.sha256

            patch_dict = None
            if cache_key in _PARSED_LORA_CACHE:
                cached_hash, cached_patch_dict = _PARSED_LORA_CACHE[cache_key]
                if cached_hash == current_hash:
                    patch_dict = cached_patch_dict
                    log_telemetry("lora_cache_hit", f"path={spec.file_identity.path.name} target=clip")
                else:
                    _PARSED_LORA_CACHE.pop(cache_key)

            if patch_dict is None:
                log_telemetry("lora_cache_miss", f"path={spec.file_identity.path.name} target=clip")
                header = SafeOpenHeaderOnly(lora_path)
                patch_dict = backend_lora.load_lora(header, key_map, log_missing=False)
                if patch_dict is None:
                    patch_dict = {}
                _PARSED_LORA_CACHE[cache_key] = (current_hash, patch_dict)
                while len(_PARSED_LORA_CACHE) > _PARSED_LORA_CACHE_LIMIT:
                    _PARSED_LORA_CACHE.pop(next(iter(_PARSED_LORA_CACHE)))

            if patch_dict:
                resolved_patches.append((patch_dict, float(spec.clip_weight)))

        self.clip_patch_count = sum(len(patch_dict) for patch_dict, _weight in resolved_patches)
        return tuple(resolved_patches)

    def apply_clip_patches(
        self,
        clip: Any,
        *,
        resolved_patches: tuple[tuple[dict, float], ...] | None = None,
    ) -> int:
        """Parses and applies CLIP-side patches to the model patcher."""
        if resolved_patches is None:
            resolved_patches = self.resolve_clip_patches(clip)

        if not resolved_patches:
            self.clip_patch_count = 0
            if self.patch_artifact is not None:
                self.patch_artifact["clip_patch_count"] = 0
            return 0

        patcher = clip.patcher
        patch_count = 0
        for patch_dict, weight in resolved_patches:
            applied_keys = patcher.add_patches(patch_dict, weight)
            patch_count += len(applied_keys or ())

        self.clip_patch_count = patch_count
        if self.patch_artifact is not None:
            self.patch_artifact["clip_patch_count"] = patch_count
        return patch_count

    def compile_clip_patches(self, clip: Any) -> dict[str, Any]:
        """Runs sequential GPU compilation on the applied CLIP patcher patches."""
        if not self.clip_patch_count:
            metrics = {"status": "noop", "patch_count": 0, "materialized_patch_keys": 0, "host_pinned_bytes": 0}
            self.last_compile_metrics = metrics
            return metrics

        import torch
        from backend.gpu_compiler import GpuArtifactCompiler

        target_device = torch.device(self.request.device)
        logger.debug("[SDXL Telemetry] Compiling CLIP on GPU (device=%s)...", target_device)
        log_telemetry("gpu_clip_lora_compile_begin", f"device={target_device}")

        baseline_vram = 0
        if target_device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats(target_device)
                baseline_vram = torch.cuda.memory_allocated(target_device)
            except Exception:
                pass

        compile_metrics = GpuArtifactCompiler.compile_patcher(
            clip.patcher,
            clean_source=None,
            target_device=target_device,
            intermediate_dtype=torch.float32,
        )

        if target_device.type == "cuda" and torch.cuda.is_available():
            try:
                peak_vram = torch.cuda.max_memory_allocated(target_device)
                compile_metrics["cuda_temp_peak_bytes"] = int(peak_vram)
                compile_metrics["cuda_baseline_allocated_bytes"] = int(baseline_vram)
                compile_metrics["cuda_peak_allocated_bytes"] = int(peak_vram)
                compile_metrics["cuda_peak_delta_bytes"] = int(peak_vram - baseline_vram)
                compile_metrics["cuda_final_allocated_bytes"] = int(torch.cuda.memory_allocated(target_device))
            except Exception:
                pass

        compile_metrics["cleared_patch_count"] = int(self.clip_patch_count)
        self.last_compile_metrics = dict(compile_metrics)
        if self.patch_artifact is not None:
            self.patch_artifact["compile_metrics"] = dict(compile_metrics)

        log_telemetry("gpu_clip_lora_compile_complete", f"device={target_device} metrics={compile_metrics}")
        return compile_metrics

    def teardown_assembly_order(self) -> None:
        self.patch_artifact = None
