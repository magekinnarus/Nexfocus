from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import RLock
from typing import Tuple, Any, Dict

from backend.sdxl_assembly.contracts import SDXLAssemblyRequest
from backend.sdxl_assembly.progress import log_telemetry

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class SDXLStreamingSpineKey:
    checkpoint_sha256: str
    device: str
    prefetch_depth: int
    prefetch_chunk_mb: int
    lora_stack_hash: str
    scheduler: str

# In-memory clean text encoder cache. UNet ownership belongs to the streaming
# spine, and VAE ownership is transient, so neither is cached here.
_TEXT_ENCODER_COMPONENT_CACHE: Dict[str, Any] = {}
_TEXT_ENCODER_COMPONENT_CACHE_LOCK = RLock()

# In-memory prompt conditioning caches
_PROMPT_CONDITIONING_CACHE: Dict[Tuple[str, str, int, str, Tuple[Tuple[str, float], ...]], Any] = {}
_PROMPT_CONDITIONING_CACHE_LOCK = RLock()


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

    return loader.load_vae(
        source_path,
        load_device=cpu_device,
        offload_device=cpu_device,
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
    clip_lora_signature = tuple(
        (spec.file_identity.sha256, spec.clip_weight)
        for spec in request.lora_specs
        if spec.enabled and spec.clip_weight != 0.0
    )
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
    clip_lora_signature = tuple(
        (spec.file_identity.sha256, spec.clip_weight)
        for spec in request.lora_specs
        if spec.enabled and spec.clip_weight != 0.0
    )
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
    import gc
    gc.collect()

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
            scheduler=request.scheduler,
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


def debug_component_cache_report() -> Dict[str, Any]:
    """Return coarse component ownership sizes for probe/debug output."""
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
    }

    with _STREAMING_RUNTIME_STATE._lock:
        spine = _STREAMING_RUNTIME_STATE._spine

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

    return report


def clear_all_caches() -> None:
    """Teardown and unload all cached workers, spines, conditioning, and base components."""
    release_active_sdxl_streaming_spine(reason="global_cleanup")
    
    release_text_encoder_component_cache(reason="global_cleanup")
        
    with _PROMPT_CONDITIONING_CACHE_LOCK:
        _PROMPT_CONDITIONING_CACHE.clear()

    import gc
    gc.collect()
