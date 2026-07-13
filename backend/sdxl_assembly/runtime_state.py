from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Tuple, Any, Dict

from backend.sdxl_assembly.contracts import SDXLAssemblyRequest
from backend.sdxl_assembly.progress import log_telemetry

logger = logging.getLogger(__name__)

class LifecycleDomain(str, Enum):
    RUN_BOUND = "run_bound"
    PROMPT_CONDITIONING = "prompt_conditioning"
    MODEL_PROMPT = "model_prompt"
    SPATIAL_VAE = "spatial_vae"
    STRUCTURAL_CN = "structural_cn"
    CONTEXTUAL_CN = "contextual_cn"
    FULL_TEARDOWN = "full_teardown"


@dataclass(frozen=True)
class SDXLStreamingSpineKey:
    checkpoint_sha256: str
    device: str
    prefetch_depth: int
    prefetch_chunk_mb: int
    lora_stack_hash: str
    scheduler_signature: str


@dataclass(frozen=True)
class SDXLResidentSpineKey:
    checkpoint_sha256: str
    unet_posture: str
    device: str
    dtype: str
    scheduler_signature: str
    unet_lora_signature: Tuple[Tuple[str, float], ...]


@dataclass(frozen=True)
class SDXLPatchedTextEncoderKey:
    checkpoint_sha256: str
    clip_posture: str
    clip_lora_signature: Tuple[Tuple[str, float], ...]

# In-memory clean text encoder cache. UNet ownership belongs to the streaming
# spine, and VAE ownership is transient, so neither is cached here.
_TEXT_ENCODER_COMPONENT_CACHE: Dict[str, Any] = {}
_TEXT_ENCODER_COMPONENT_CACHE_LOCK = RLock()

# Single warm patched CLIP slot for the current checkpoint + CLIP-side LoRA stack.
_PATCHED_TEXT_ENCODER_COMPONENT_SLOT_KEY: SDXLPatchedTextEncoderKey | None = None
_PATCHED_TEXT_ENCODER_COMPONENT_SLOT: Any | None = None
_PATCHED_TEXT_ENCODER_COMPONENT_SLOT_LOCK = RLock()

# In-memory prompt conditioning caches
_PROMPT_CONDITIONING_CACHE: Dict[Tuple[str, str, int, str, Tuple[Tuple[str, float], ...]], Any] = {}
_PROMPT_CONDITIONING_CACHE_LOCK = RLock()


def _clip_lora_signature(request: SDXLAssemblyRequest) -> Tuple[Tuple[str, float], ...]:
    return tuple(
        (spec.file_identity.sha256, spec.clip_weight)
        for spec in request.lora_specs
        if spec.enabled and spec.clip_weight != 0.0
    )


def _unet_lora_signature(request: SDXLAssemblyRequest) -> Tuple[Tuple[str, float], ...]:
    return tuple(
        (spec.file_identity.sha256, spec.unet_weight)
        for spec in request.lora_specs
        if spec.enabled and spec.unet_weight != 0.0
    )


def _patched_text_encoder_key(request: SDXLAssemblyRequest) -> SDXLPatchedTextEncoderKey | None:
    clip_lora_signature = _clip_lora_signature(request)
    if not clip_lora_signature:
        return None
    return SDXLPatchedTextEncoderKey(
        checkpoint_sha256=request.checkpoint.sha256,
        clip_posture=request.clip_posture.value,
        clip_lora_signature=clip_lora_signature,
    )


def _clear_patched_text_encoder_component_slot_locked() -> bool:
    global _PATCHED_TEXT_ENCODER_COMPONENT_SLOT_KEY
    global _PATCHED_TEXT_ENCODER_COMPONENT_SLOT

    had_slot = _PATCHED_TEXT_ENCODER_COMPONENT_SLOT is not None
    _PATCHED_TEXT_ENCODER_COMPONENT_SLOT_KEY = None
    _PATCHED_TEXT_ENCODER_COMPONENT_SLOT = None
    return had_slot


def _clone_owned_component(component: Any, component_name: str, *, isolate_model: bool = False) -> Any:
    clone = getattr(component, "clone", None)
    if clone is None:
        raise TypeError(f"Cached {component_name} component does not expose clone().")
    owned = clone()
    if isolate_model:
        patcher = getattr(owned, "patcher", None)
        isolated_clone = getattr(patcher, "isolated_clone", None)
        if isolated_clone is None:
            raise TypeError(f"Cached {component_name} component cannot provide an isolated patcher clone.")
        owned.patcher = isolated_clone()
        if hasattr(owned, "cond_stage_model"):
            owned.cond_stage_model = owned.patcher.model
    return owned


def acquire_unet_component(request: SDXLAssemblyRequest) -> Any:
    """Load a UNet component owned by the streaming spine."""
    checkpoint_path = str(request.checkpoint.path)
    logger.debug("[SDXL Telemetry] Loading owned UNet component for checkpoint=%s", request.checkpoint.sha256)
    log_telemetry("unet_component_load", f"checkpoint={request.checkpoint.path.name}")

    from backend import loader
    from backend.defs import sdxl as sdxl_def
    import torch

    cpu_device = torch.device("cpu")
    return loader._stream_load_sdxl_unet_from_checkpoint(
        checkpoint_path,
        load_device=cpu_device,
        offload_device=cpu_device,
        dtype=torch.float16,
        reload_source=checkpoint_path,
        reload_prefixes=sdxl_def.PREFIXES["unet"],
        stream_chunk_bytes=int(request.prefetch_chunk_mb) * 1024 * 1024 if int(request.prefetch_chunk_mb) > 0 else None,
        raw_byte_stream=True,
    )


def acquire_resident_unet_component(request: SDXLAssemblyRequest) -> Any:
    """Load a resident UNet component directly to GPU (CUDA)."""
    checkpoint_path = str(request.checkpoint.path)
    if not checkpoint_path.lower().endswith(".safetensors"):
        raise RuntimeError(
            "Resident SDXL UNet admission requires a safetensors checkpoint "
            f"for bounded direct-GPU loading; got {request.checkpoint.path}."
        )
    logger.debug("[SDXL Telemetry] Loading owned resident UNet component for checkpoint=%s", request.checkpoint.sha256)
    log_telemetry("unet_component_load", f"checkpoint={request.checkpoint.path.name}")

    from backend import loader
    from backend.defs import sdxl as sdxl_def
    import torch

    cuda_device = torch.device(request.device)
    # The resident UNet model resides on GPU, and load/offload are CUDA.
    # No full CPU shadow survives; meta construction realizes it directly on CUDA.
    return loader._stream_load_sdxl_unet_from_checkpoint(
        checkpoint_path,
        load_device=cuda_device,
        offload_device=cuda_device,
        dtype=torch.float16,
        reload_source=checkpoint_path,
        reload_prefixes=sdxl_def.PREFIXES["unet"],
        stream_chunk_bytes=None,  # Bounded, direct-GPU safetensors load
        raw_byte_stream=True,     # Activates sequential reader meta realization
    )


def acquire_text_encoder_component(request: SDXLAssemblyRequest) -> Any:
    """Return an owned CLIP worker component without loading UNet/VAE."""
    checkpoint_path = str(request.checkpoint.path)
    key = request.checkpoint.sha256
    isolate_for_lora = any(
        spec.enabled and spec.clip_weight != 0.0
        for spec in request.lora_specs
    )

    with _TEXT_ENCODER_COMPONENT_CACHE_LOCK:
        clip = _TEXT_ENCODER_COMPONENT_CACHE.get(key)
        if clip is None:
            logger.debug("[SDXL Telemetry] Text encoder cache MISS for checkpoint=%s", request.checkpoint.sha256)
            log_telemetry("text_encoder_cache_miss", f"checkpoint={request.checkpoint.path.name}")

            from backend import loader
            from backend.defs import sdxl as sdxl_def
            import torch

            cpu_device = torch.device("cpu")
            clip = loader.load_sdxl_clip(
                checkpoint_path,
                checkpoint_path,
                load_device=cpu_device,
                offload_device=cpu_device,
                dtype=torch.float32,
                reload_source_l=checkpoint_path,
                reload_source_g=checkpoint_path,
                reload_prefixes_l=sdxl_def.PREFIXES["clip_l"],
                reload_prefixes_g=sdxl_def.PREFIXES["clip_g"],
            )
            _TEXT_ENCODER_COMPONENT_CACHE[key] = clip
        else:
            logger.debug("[SDXL Telemetry] Text encoder cache HIT for checkpoint=%s", request.checkpoint.sha256)
            log_telemetry("text_encoder_cache_hit", f"checkpoint={request.checkpoint.path.name}")

        return _clone_owned_component(clip, "CLIP", isolate_model=isolate_for_lora)


def acquire_patched_text_encoder_component(
    request: SDXLAssemblyRequest,
    *,
    lora_worker: Any,
) -> Any:
    """Return a single warm patched CLIP for the active checkpoint + CLIP-side LoRAs."""
    slot_key = _patched_text_encoder_key(request)
    if slot_key is None:
        with _PATCHED_TEXT_ENCODER_COMPONENT_SLOT_LOCK:
            _clear_patched_text_encoder_component_slot_locked()
        return acquire_text_encoder_component(request)

    with _PATCHED_TEXT_ENCODER_COMPONENT_SLOT_LOCK:
        global _PATCHED_TEXT_ENCODER_COMPONENT_SLOT_KEY
        global _PATCHED_TEXT_ENCODER_COMPONENT_SLOT

        if (
            _PATCHED_TEXT_ENCODER_COMPONENT_SLOT is not None
            and _PATCHED_TEXT_ENCODER_COMPONENT_SLOT_KEY == slot_key
        ):
            logger.debug(
                "[SDXL Telemetry] Warm patched text encoder HIT for checkpoint=%s",
                request.checkpoint.sha256,
            )
            log_telemetry("patched_text_encoder_cache_hit", f"checkpoint={request.checkpoint.path.name}")
            return _PATCHED_TEXT_ENCODER_COMPONENT_SLOT

        had_previous_slot = _clear_patched_text_encoder_component_slot_locked()
        if had_previous_slot:
            logger.debug("[SDXL Telemetry] Releasing stale warm patched text encoder before rebuild.")

        clip = acquire_text_encoder_component(request)
        lora_worker.apply_clip_patches(clip)
        if int(getattr(lora_worker, "clip_patch_count", 0) or 0) <= 0:
            logger.debug(
                "[SDXL Telemetry] Patched text encoder build resolved no CLIP-side patches for checkpoint=%s",
                request.checkpoint.sha256,
            )
            log_telemetry("patched_text_encoder_cache_bypass", f"checkpoint={request.checkpoint.path.name}")
            return clip

        from backend.cpu_compiler import CpuArtifactCompiler

        logger.debug(
            "[SDXL Telemetry] Building warm patched text encoder for checkpoint=%s patches=%d",
            request.checkpoint.sha256,
            lora_worker.clip_patch_count,
        )
        log_telemetry("patched_text_encoder_cache_miss", f"checkpoint={request.checkpoint.path.name}")
        CpuArtifactCompiler.compile_patcher(clip.patcher)
        _PATCHED_TEXT_ENCODER_COMPONENT_SLOT_KEY = slot_key
        _PATCHED_TEXT_ENCODER_COMPONENT_SLOT = clip
        return clip


def acquire_vae_component(request: SDXLAssemblyRequest) -> Any:
    """Load a VAE component for transient decode ownership."""
    source_path = str(request.vae.path) if request.vae else str(request.checkpoint.path)
    source_label = request.vae.path.name if request.vae else request.checkpoint.path.name
    logger.debug("[SDXL Telemetry] Loading transient VAE component source=%s", source_path)
    log_telemetry("vae_component_load", f"source={source_label}")

    from backend import loader
    from backend.defs import sdxl as sdxl_def
    from ldm_patched.modules import latent_formats
    import torch

    cpu_device = torch.device("cpu")
    prefixes = None if request.vae else sdxl_def.PREFIXES["vae"]
    if request.vae is None and source_path.lower().endswith(".safetensors"):
        metadata = loader._inspect_safetensors_vae_metadata(source_path, prefixes=prefixes)
        if metadata["key_count"] == 0:
            raise RuntimeError(
                "SDXL assembly transient VAE could not find embedded VAE weights "
                f"in checkpoint {request.checkpoint.path}."
            )

    # Keep the SDXL assembly VAE contract explicit. The project-wide stability
    # policy is fp32 VAE residency to avoid the known half-precision NaN path.
    return loader.load_vae(
        source_path,
        load_device=cpu_device,
        offload_device=cpu_device,
        dtype=torch.float32,
        latent_format=latent_formats.SDXL(),
        prefixes=prefixes,
    )


def _load_base_components(request: SDXLAssemblyRequest) -> Tuple[Any, Any, Any]:
    """Reject broad acquisition so workers cannot accidentally load all parts."""
    raise RuntimeError(
        "Broad SDXL base-component acquisition is disabled. "
        "Use acquire_unet_component(), acquire_text_encoder_component(), "
        "or acquire_vae_component() so each worker owns only its component."
    )


def acquire_base_components(request: SDXLAssemblyRequest) -> Tuple[Any, Any, Any]:
    """Reject broad acquisition so workers cannot accidentally load all parts."""
    return _load_base_components(request)

def lookup_prompt_conditioning(request: SDXLAssemblyRequest) -> Any | None:
    """Retrieves cached prompt conditioning if exact text-side keys match."""
    clip_lora_signature = _clip_lora_signature(request)
    key = (
        request.checkpoint.sha256,
        request.prompt_payload_hash,
        request.clip_layer,
        request.clip_posture.value,
        clip_lora_signature,
    )
    with _PROMPT_CONDITIONING_CACHE_LOCK:
        hit = key in _PROMPT_CONDITIONING_CACHE
        log_telemetry("prompt_cache_hit" if hit else "prompt_cache_miss")
        return _PROMPT_CONDITIONING_CACHE.get(key)

def remember_prompt_conditioning(request: SDXLAssemblyRequest, conditioning_payload: Any) -> None:
    """Caches prompt conditioning under text-side key."""
    clip_lora_signature = _clip_lora_signature(request)
    key = (
        request.checkpoint.sha256,
        request.prompt_payload_hash,
        request.clip_layer,
        request.clip_posture.value,
        clip_lora_signature,
    )
    with _PROMPT_CONDITIONING_CACHE_LOCK:
        _PROMPT_CONDITIONING_CACHE[key] = conditioning_payload


def release_text_encoder_component_cache(reason: str | None = None) -> None:
    logger.debug("[SDXL Telemetry] Releasing text encoder component cache reason=%s", reason)
    log_telemetry("text_encoder_cache_release", f"reason={reason or 'unspecified'}")
    with _TEXT_ENCODER_COMPONENT_CACHE_LOCK:
        _TEXT_ENCODER_COMPONENT_CACHE.clear()
    with _PATCHED_TEXT_ENCODER_COMPONENT_SLOT_LOCK:
        _clear_patched_text_encoder_component_slot_locked()
    import gc
    gc.collect()


def release_prompt_conditioning_cache(reason: str | None = None) -> None:
    logger.debug("[SDXL Telemetry] Releasing prompt conditioning cache reason=%s", reason)
    log_telemetry("prompt_cache_release", f"reason={reason or 'unspecified'}")
    with _PROMPT_CONDITIONING_CACHE_LOCK:
        _PROMPT_CONDITIONING_CACHE.clear()


def release_domain(
    domain_or_domains: Any,
    reason: str | None = None,
    *,
    assembly: Any | None = None,
    raise_on_error: bool = False,
) -> Any:
    """Release SDXL assembly lifecycle domains through the coordinator."""
    from backend.sdxl_assembly.lifecycle_coordinator import release_domains

    return release_domains(
        domain_or_domains,
        reason=reason,
        assembly=assembly,
        raise_on_error=raise_on_error,
    )


def release_model_prompt_caches(*, reason: str | None = None) -> None:
    """Release SDXL model/prompt domains without clearing warm CN or spatial artifacts."""
    clear_reason = reason or "model_prompt_domain_release"
    log_telemetry("assembly_model_prompt_cache_release", f"reason={clear_reason}")
    release_domain(LifecycleDomain.MODEL_PROMPT, reason=clear_reason)


def release_prompt_conditioning_caches(*, reason: str | None = None) -> None:
    """Release prompt-conditioning artifacts without rebuilding the warm UNet spine."""
    clear_reason = reason or "prompt_conditioning_domain_release"
    log_telemetry("assembly_prompt_conditioning_cache_release", f"reason={clear_reason}")
    release_domain(LifecycleDomain.PROMPT_CONDITIONING, reason=clear_reason)


def release_spatial_vae_caches(*, reason: str | None = None) -> None:
    """Release image/VAE-derived warm artifacts without touching ControlNet support models."""
    clear_reason = reason or "spatial_vae_domain_release"
    log_telemetry("assembly_spatial_vae_cache_release", f"reason={clear_reason}")
    release_domain(LifecycleDomain.SPATIAL_VAE, reason=clear_reason)


class SDXLStreamingRuntimeState:
    def __init__(self) -> None:
        self._lock = RLock()
        self._key: SDXLStreamingSpineKey | None = None
        self._spine: Any | None = None

    @staticmethod
    def _build_key(request: SDXLAssemblyRequest) -> SDXLStreamingSpineKey:
        return SDXLStreamingSpineKey(
            checkpoint_sha256=request.checkpoint.sha256,
            device=request.device,
            prefetch_depth=request.prefetch_depth,
            prefetch_chunk_mb=request.prefetch_chunk_mb,
            lora_stack_hash=request.lora_stack_hash,
            # Only the LCM patch changes UNet state. Ordinary scheduler changes
            # affect sampling, not the already-loaded/patched UNet spine.
            scheduler_signature="lcm" if request.scheduler == "lcm" else "standard",
        )

    def acquire(self, request: SDXLAssemblyRequest, *, lora_worker: Any | None = None) -> Tuple[Any, bool]:
        requested_key = self._build_key(request)
        stale_spine: Any | None = None

        with self._lock:
            if self._spine is not None and self._key == requested_key:
                self._spine.request = request
                if lora_worker is not None:
                    self._spine.lora_worker = lora_worker
                logger.debug(
                    "[SDXL Telemetry] Reusing cached SDXL UNet spine shell for key=%s",
                    requested_key,
                )
                log_telemetry("warm_reuse", f"checkpoint={request.checkpoint.path.name}")
                return self._spine, True

            stale_spine = self._spine
            self._spine = None
            self._key = None

        if stale_spine is not None:
            logger.debug("[SDXL Telemetry] Releasing stale streaming UNet spine before replacement.")
            _release_spine_owned_resources(stale_spine)

        # Import dynamically to avoid circular references
        from backend.sdxl_assembly.streaming_unet import StreamingUnetSpine
        spine = StreamingUnetSpine(request, lora_worker=lora_worker)
        logger.debug(
            "[SDXL Telemetry] Creating new cached SDXL UNet spine shell for key=%s",
            requested_key,
        )
        log_telemetry("cold_load", f"checkpoint={request.checkpoint.path.name}")

        with self._lock:
            self._spine = spine
            self._key = requested_key
        return spine, False

    def release(self, *, reason: str | None = None) -> bool:
        with self._lock:
            spine = self._spine
            key = self._key
            self._spine = None
            self._key = None

        if spine is None:
            return False

        logger.debug(
            "[SDXL Telemetry] Releasing active streaming UNet spine reason=%s key=%s",
            reason,
            key,
        )
        _release_spine_owned_resources(spine)
        return True

    def get_active_key(self) -> SDXLStreamingSpineKey | None:
        with self._lock:
            return self._key

_STREAMING_RUNTIME_STATE = SDXLStreamingRuntimeState()


def _release_spine_owned_resources(spine: Any) -> None:
    release_owned_resources = getattr(spine, "release_owned_resources", None)
    if callable(release_owned_resources):
        release_owned_resources()
    else:
        spine.end()

def acquire_active_sdxl_streaming_spine(
    request: SDXLAssemblyRequest,
    *,
    lora_worker: Any | None = None,
) -> Tuple[Any, bool]:
    return _STREAMING_RUNTIME_STATE.acquire(request, lora_worker=lora_worker)

def release_active_sdxl_streaming_spine(reason: str | None = None) -> bool:
    return _STREAMING_RUNTIME_STATE.release(reason=reason)


class SDXLResidentRuntimeState:
    def __init__(self) -> None:
        self._lock = RLock()
        self._key: SDXLResidentSpineKey | None = None
        self._spine: Any | None = None

    @staticmethod
    def _build_key(request: SDXLAssemblyRequest) -> SDXLResidentSpineKey:
        return SDXLResidentSpineKey(
            checkpoint_sha256=request.checkpoint.sha256,
            unet_posture=request.unet_posture.value,
            device=request.device,
            dtype="float16",  # resident UNet is fp16
            scheduler_signature="lcm" if request.scheduler == "lcm" else "standard",
            unet_lora_signature=_unet_lora_signature(request),
        )

    def acquire(self, request: SDXLAssemblyRequest, *, lora_worker: Any | None = None) -> Tuple[Any, bool]:
        requested_key = self._build_key(request)
        stale_spine: Any | None = None

        with self._lock:
            # Case 1: Exact key match -> Reuse warm resident spine immediately!
            if self._spine is not None and self._key == requested_key:
                self._spine.request = request
                if lora_worker is not None:
                    self._spine.lora_worker = lora_worker
                logger.debug(
                    "[SDXL Telemetry] Reusing warm resident UNet spine for key=%s",
                    requested_key,
                )
                log_telemetry("resident_spine_warm_reuse", f"checkpoint={request.checkpoint.path.name}")
                return self._spine, True

            # Case 2: Spine is warm, but there is a key mismatch.
            # We check if it is ONLY a change in the UNet-side LoRA signature.
            if self._spine is not None and self._key is not None:
                structural_match = (
                    self._key.checkpoint_sha256 == requested_key.checkpoint_sha256
                    and self._key.unet_posture == requested_key.unet_posture
                    and self._key.device == requested_key.device
                    and self._key.dtype == requested_key.dtype
                    and self._key.scheduler_signature == requested_key.scheduler_signature
                )
                if structural_match:
                    # In-place clean reload and GPU prepatch!
                    logger.debug(
                        "[SDXL Telemetry] In-place resident reload/prepatch for LoRA signature change. "
                        "Old key: %s, New key: %s",
                        self._key, requested_key
                    )
                    log_telemetry("resident_spine_clean_reload_begin", f"checkpoint={request.checkpoint.path.name}")
                    
                    try:
                        # 1. Update requests and workers
                        self._spine.request = request
                        from backend.sdxl_assembly.gpu_lora_worker import GpuLoraWorker
                        if lora_worker is not None:
                            self._spine.lora_worker = lora_worker
                        else:
                            self._spine.lora_worker = GpuLoraWorker(request)
                        
                        # 2. Reload clean weights from checkpoint source (restores clean state)
                        import torch
                        model = self._spine.unet.model
                        runtime_reload = getattr(self._spine.unet, "runtime_reload", None)
                        if callable(runtime_reload):
                            target_device = torch.device(request.device)
                            runtime_reload(model, target_device)
                            model.device = target_device
                        else:
                            raise RuntimeError("Resident UNet spine component missing runtime_reload callable.")
                        
                        # Clear GpuArtifactCompiler's patcher artifacts
                        self._spine.unet.patches = {}
                        self._spine.unet.weight_wrapper_patches = {}
                        self._spine.unet.backup.clear()
                        self._spine.unet.object_patches_backup.clear()
                        self._spine.unet.model.current_weight_patches_uuid = None
                        
                        # 3. Patch LCM scheduler if LCM
                        orig_scheduler = request.scheduler
                        if orig_scheduler == 'lcm':
                            from modules import core as modules_core
                            self._spine.unet = modules_core.opModelSamplingDiscrete.patch(self._spine.unet, orig_scheduler, False)[0]
                            self._spine.unet.model._nex_resident_scheduler = "lcm"
                        else:
                            self._spine.unet.model._nex_resident_scheduler = ""

                        # 4. Materialize and apply new LoRAs to UNet on GPU
                        self._spine.lora_worker.apply_unet_patches(self._spine.unet)
                        
                        # 5. Compile new patches on GPU
                        self._spine.lora_worker.compile_unet_patches(self._spine.unet)
                        
                        # 6. Publish the new key atomically
                        self._key = requested_key
                        log_telemetry("resident_spine_clean_reload_complete", f"checkpoint={request.checkpoint.path.name}")
                        return self._spine, False
                        
                    except Exception as exc:
                        # Reconfiguration failed -> Release spine completely.
                        logger.error(
                            "[SDXL Telemetry] Resident in-place reconfiguration failed! "
                            "Releasing spine completely. Error: %s", exc
                        )
                        log_telemetry("resident_spine_load_failure_cleanup", f"error={exc.__class__.__name__}")
                        stale_spine = self._spine
                        self._spine = None
                        self._key = None
                        if stale_spine is not None:
                            _release_spine_owned_resources(stale_spine)
                        raise

            # Case 3: Structural key mismatch (checkpoint, posture, device/dtype, or scheduler).
            # Release the old owner before loading the new one.
            stale_spine = self._spine
            self._spine = None
            self._key = None

        if stale_spine is not None:
            logger.debug("[SDXL Telemetry] Releasing stale resident UNet spine before replacement.")
            _release_spine_owned_resources(stale_spine)

        # Build the new resident spine
        from backend.sdxl_assembly.resident_unet import ResidentUnetSpine
        logger.debug(
            "[SDXL Telemetry] Creating new cached resident UNet spine for key=%s",
            requested_key,
        )
        log_telemetry("resident_spine_load_begin", f"checkpoint={request.checkpoint.path.name}")

        spine = None
        try:
            spine = ResidentUnetSpine(request, lora_worker=lora_worker)
            # This loads and compiles model directly to CUDA
            spine.start()
            
            with self._lock:
                self._spine = spine
                self._key = requested_key
            log_telemetry("resident_spine_load_complete", f"checkpoint={request.checkpoint.path.name}")
            return spine, False
        except Exception as exc:
            log_telemetry("resident_spine_load_failure_cleanup", f"error={exc.__class__.__name__}")
            if spine is not None:
                _release_spine_owned_resources(spine)
            raise

    def release(self, *, reason: str | None = None) -> bool:
        with self._lock:
            spine = self._spine
            key = self._key
            self._spine = None
            self._key = None

        if spine is None:
            return False

        logger.debug(
            "[SDXL Telemetry] Releasing active resident UNet spine reason=%s key=%s",
            reason,
            key,
        )
        log_telemetry("resident_spine_release_begin", f"reason={reason or 'unspecified'}")
        _release_spine_owned_resources(spine)
        log_telemetry("resident_spine_release_complete", f"reason={reason or 'unspecified'}")
        return True

    def get_active_key(self) -> SDXLResidentSpineKey | None:
        with self._lock:
            return self._key


_RESIDENT_RUNTIME_STATE = SDXLResidentRuntimeState()


def acquire_active_sdxl_resident_spine(
    request: SDXLAssemblyRequest,
    *,
    lora_worker: Any | None = None,
) -> Tuple[Any, bool]:
    return _RESIDENT_RUNTIME_STATE.acquire(request, lora_worker=lora_worker)


def release_active_sdxl_resident_spine(reason: str | None = None) -> bool:
    return _RESIDENT_RUNTIME_STATE.release(reason=reason)


def _module_tensor_inventory(module: Any) -> dict[str, Any]:
    inventory: dict[str, Any] = {
        "parameter_devices": {},
        "buffer_devices": {},
        "parameter_bytes": 0,
        "buffer_bytes": 0,
    }

    def add_tensor(bucket: dict[str, int], tensor: Any) -> int:
        if not hasattr(tensor, "device") or not hasattr(tensor, "numel"):
            return 0
        device_key = str(tensor.device)
        bucket[device_key] = int(bucket.get(device_key, 0)) + 1
        try:
            return int(tensor.numel() * tensor.element_size())
        except Exception:
            return 0

    named_parameters = getattr(module, "named_parameters", None)
    if callable(named_parameters):
        try:
            for _, tensor in named_parameters(recurse=True):
                inventory["parameter_bytes"] += add_tensor(inventory["parameter_devices"], tensor)
        except TypeError:
            for _, tensor in named_parameters():
                inventory["parameter_bytes"] += add_tensor(inventory["parameter_devices"], tensor)

    named_buffers = getattr(module, "named_buffers", None)
    if callable(named_buffers):
        try:
            for _, tensor in named_buffers(recurse=True):
                inventory["buffer_bytes"] += add_tensor(inventory["buffer_devices"], tensor)
        except TypeError:
            for _, tensor in named_buffers():
                inventory["buffer_bytes"] += add_tensor(inventory["buffer_devices"], tensor)

    inventory["model_bytes"] = int(inventory["parameter_bytes"] + inventory["buffer_bytes"])
    return inventory


def debug_component_cache_report() -> Dict[str, Any]:
    """Return coarse component ownership sizes for probe/debug output."""
    import torch
    report: Dict[str, Any] = {
        "active_spine": False,
        "active_unet_mb": 0.0,
        "active_unet_model_id": None,
        "active_unet_raw_sequential_stream": None,
        "active_unet_meta_construction": None,
        "active_unet_stream_chunk_bytes": None,
        "active_unet_realized_cpu_mb": 0.0,
        "text_cache_count": 0,
        "text_cache_mb": 0.0,
        "patched_text_cache_active": False,
        "patched_text_cache_mb": 0.0,
        "active_resident_spine": False,
        "clean_shadow_bytes": 0.0,
        "resident_parameter_devices": {},
        "resident_buffer_devices": {},
        "resident_unet_parameter_bytes": 0,
        "resident_unet_buffer_bytes": 0,
        "resident_unet_model_bytes": 0,
    }

    with _STREAMING_RUNTIME_STATE._lock:
        spine = _STREAMING_RUNTIME_STATE._spine

    with _RESIDENT_RUNTIME_STATE._lock:
        res_spine = _RESIDENT_RUNTIME_STATE._spine

    if spine is not None:
        report["active_spine"] = True
        unet = getattr(spine, "unet", None)
        if unet is not None:
            model_size = getattr(unet, "model_size", None)
            if callable(model_size):
                try:
                    report["active_unet_mb"] = float(model_size()) / (1024 * 1024)
                except Exception:
                    pass
            report["active_unet_model_id"] = id(getattr(unet, "model", unet))
            loader_info = getattr(unet, "model_options", {}).get("sdxl_assembly_loader", {})
            if isinstance(loader_info, dict):
                report["active_unet_raw_sequential_stream"] = loader_info.get("raw_sequential_stream")
                report["active_unet_meta_construction"] = loader_info.get("meta_construction")
                report["active_unet_stream_chunk_bytes"] = loader_info.get("stream_chunk_bytes")
                report["active_unet_realized_cpu_mb"] = float(loader_info.get("realized_cpu_bytes", 0)) / (1024 * 1024)

    elif res_spine is not None:
        report["active_spine"] = True
        report["active_resident_spine"] = True
        unet = getattr(res_spine, "unet", None)
        if unet is not None:
            model_size = getattr(unet, "model_size", None)
            if callable(model_size):
                try:
                    report["active_unet_mb"] = float(model_size()) / (1024 * 1024)
                except Exception:
                    pass
            report["active_unet_model_id"] = id(getattr(unet, "model", unet))
            loader_info = getattr(unet, "model_options", {}).get("sdxl_assembly_loader", {})
            if isinstance(loader_info, dict):
                report["active_unet_raw_sequential_stream"] = loader_info.get("raw_sequential_stream")
                report["active_unet_meta_construction"] = loader_info.get("meta_construction")
                report["active_unet_stream_chunk_bytes"] = loader_info.get("stream_chunk_bytes")
                report["active_unet_realized_cpu_mb"] = float(loader_info.get("realized_cpu_bytes", 0)) / (1024 * 1024)
            if hasattr(unet, "model"):
                inventory = _module_tensor_inventory(unet.model)
                report["resident_parameter_devices"] = inventory["parameter_devices"]
                report["resident_buffer_devices"] = inventory["buffer_devices"]
                report["resident_unet_parameter_bytes"] = inventory["parameter_bytes"]
                report["resident_unet_buffer_bytes"] = inventory["buffer_bytes"]
                report["resident_unet_model_bytes"] = inventory["model_bytes"]
                clean_source = getattr(unet.model, "_nex_clean_unet_source", None)
                if clean_source is not None:
                    if isinstance(clean_source, dict):
                        report["clean_shadow_bytes"] = float(sum(t.numel() * t.element_size() for t in clean_source.values() if isinstance(t, torch.Tensor)))
                    elif isinstance(clean_source, torch.Tensor):
                        report["clean_shadow_bytes"] = float(clean_source.numel() * clean_source.element_size())

    with _TEXT_ENCODER_COMPONENT_CACHE_LOCK:
        report["text_cache_count"] = len(_TEXT_ENCODER_COMPONENT_CACHE)
        for clip in _TEXT_ENCODER_COMPONENT_CACHE.values():
            patcher = getattr(clip, "patcher", None)
            model_size = getattr(patcher, "model_size", None)
            if callable(model_size):
                try:
                    report["text_cache_mb"] += float(model_size()) / (1024 * 1024)
                except Exception:
                    pass

    with _PATCHED_TEXT_ENCODER_COMPONENT_SLOT_LOCK:
        patched_clip = _PATCHED_TEXT_ENCODER_COMPONENT_SLOT
        if patched_clip is not None:
            report["patched_text_cache_active"] = True
            patcher = getattr(patched_clip, "patcher", None)
            model_size = getattr(patcher, "model_size", None)
            if callable(model_size):
                try:
                    report["patched_text_cache_mb"] = float(model_size()) / (1024 * 1024)
                except Exception:
                    pass

    return report


def clear_all_caches(*, reason: str | None = None) -> None:
    """Teardown and unload all cached workers, spines, conditioning, and base components."""
    clear_reason = reason or "global_cleanup"
    logger.debug("[SDXL Telemetry] Clearing all SDXL assembly caches reason=%s", clear_reason)
    log_telemetry("assembly_cache_clear", f"reason={clear_reason}")
    release_domain(LifecycleDomain.FULL_TEARDOWN, reason=clear_reason)
