from __future__ import annotations

import logging
from typing import Any, Tuple, Dict
from backend.sdxl_assembly.contracts import SDXLAssemblyRequest
from backend.sdxl_assembly.progress import log_telemetry
from backend.cpu_compiler import SafeOpenHeaderOnly
from backend import lora as backend_lora

logger = logging.getLogger(__name__)

# Cache of parsed LoRA adapter patch dictionaries (keys are lora_path, target_family, model_class_name)
_PARSED_LORA_CACHE: Dict[Tuple[str, str, str], Tuple[str, dict]] = {}
_PARSED_LORA_CACHE_LIMIT = 10

class CpuLoraWorker:
    """Worker representing CpuLoraWorker (materializes patches on CPU-pinned weights)."""
    
    def __init__(self, request: SDXLAssemblyRequest) -> None:
        self.request = request
        self.unet_patch_count = 0
        self.clip_patch_count = 0
        self.patch_artifact: dict[str, Any] | None = None

    def materialize_patches(self) -> Any:
        """Parses and compiles LoRA patches.
        
        Logs telemetry for stack parsing.
        """
        log_telemetry("lora_parse_begin")
        specs_len = len(self.request.lora_specs)
        if specs_len == 0:
            self.patch_artifact = {
                "kind": "identity",
                "stack_hash": self.request.lora_stack_hash,
                "unet_patch_count": 0,
                "clip_patch_count": 0,
            }
            log_telemetry("lora_identity_artifact", "specs=0")
        else:
            self.patch_artifact = {
                "kind": "lora_stack",
                "stack_hash": self.request.lora_stack_hash,
                "specs": specs_len,
            }
        log_telemetry("lora_parse_complete", f"specs={specs_len}")
        return self

    def apply_unet_patches(self, unet: Any) -> int:
        """Parses and applies UNet-side patches to the model patcher."""
        if not self.request.lora_specs:
            return 0
        
        model = unet.model
        key_map = backend_lora.model_lora_keys_unet(model)
        model_class_name = model.__class__.__name__
        patch_count = 0
        
        for spec in self.request.lora_specs:
            if not spec.enabled or spec.unet_weight == 0.0:
                continue
            
            lora_path = str(spec.file_identity.path)
            cache_key = (lora_path, "unet", model_class_name)
            current_hash = spec.file_identity.sha256
            
            patch_dict = None
            if cache_key in _PARSED_LORA_CACHE:
                cached_hash, cached_patch_dict = _PARSED_LORA_CACHE[cache_key]
                if cached_hash == current_hash:
                    patch_dict = cached_patch_dict
                    log_telemetry("lora_cache_hit", f"path={spec.file_identity.path.name}")
                else:
                    _PARSED_LORA_CACHE.pop(cache_key)
                    
            if patch_dict is None:
                log_telemetry("lora_cache_miss", f"path={spec.file_identity.path.name}")
                header = SafeOpenHeaderOnly(lora_path)
                patch_dict = backend_lora.load_lora(header, key_map, log_missing=False)
                if patch_dict is None:
                    patch_dict = {}
                _PARSED_LORA_CACHE[cache_key] = (current_hash, patch_dict)
                # Evict if over limit
                while len(_PARSED_LORA_CACHE) > _PARSED_LORA_CACHE_LIMIT:
                    _PARSED_LORA_CACHE.pop(next(iter(_PARSED_LORA_CACHE)))
                        
            if patch_dict:
                unet.add_patches(patch_dict, spec.unet_weight)
                patch_count += len(patch_dict)
                
        self.unet_patch_count = patch_count
        if self.patch_artifact is not None:
            self.patch_artifact["unet_patch_count"] = patch_count
        log_telemetry("lora_apply_complete", f"target=unet patches={patch_count}")
        return patch_count

    def resolve_clip_patches(self, clip: Any) -> tuple[tuple[dict, float], ...]:
        """Parse CLIP-side patches without mutating or cloning the text encoder."""
        if not self.request.lora_specs:
            self.clip_patch_count = 0
            return ()
        
        patcher = clip.patcher
        model = patcher.model
        key_map = backend_lora.model_lora_keys_clip(model)
        model_class_name = model.__class__.__name__
        resolved_patches = []
        
        for spec in self.request.lora_specs:
            if not spec.enabled:
                continue
            if spec.clip_weight == 0.0:
                logger.info(
                    "[SDXL LORA DETAIL] Explicitly or evidence-based UNet-only asset %s correctly skipping CLIP resolution.",
                    spec.file_identity.path.name
                )
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
            else:
                if (
                    getattr(spec, "decision_source", None) == "asset_evidence"
                    and getattr(spec, "evidence_status", None) == "recognized"
                ):
                    logger.warning(
                        "[SDXL LORA WARNING] Evidence-predicted CLIP content in %s "
                        "resolved zero compatible patches.",
                        spec.file_identity.path.name,
                    )
                else:
                    logger.info(
                        "[SDXL LORA DETAIL] Conservative or unavailable evidence for %s "
                        "retained the requested CLIP channel; zero compatible patches resolved.",
                        spec.file_identity.path.name,
                    )

        self.clip_patch_count = sum(len(patch_dict) for patch_dict, _weight in resolved_patches)
        return tuple(resolved_patches)

    def apply_clip_patches(
        self,
        clip: Any,
        *,
        resolved_patches: tuple[tuple[dict, float], ...] | None = None,
    ) -> int:
        """Apply previously resolved CLIP patches, parsing them when necessary."""
        if resolved_patches is None:
            resolved_patches = self.resolve_clip_patches(clip)

        patcher = clip.patcher
        for patch_dict, weight in resolved_patches:
            patcher.add_patches(patch_dict, weight)

        patch_count = sum(len(patch_dict) for patch_dict, _weight in resolved_patches)
                
        self.clip_patch_count = patch_count
        if self.patch_artifact is not None:
            self.patch_artifact["clip_patch_count"] = patch_count
        log_telemetry("lora_apply_complete", f"target=clip patches={patch_count}")
        return patch_count

    def teardown_assembly_order(self) -> None:
        self.patch_artifact = None
