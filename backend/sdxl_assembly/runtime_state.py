from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import RLock
from typing import Tuple, Optional
from backend.sdxl_assembly.contracts import SDXLAssemblyRequest
from backend.sdxl_assembly.streaming_unet import StreamingUnetSpine

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class SDXLStreamingSpineKey:
    checkpoint_sha256: str
    device: str
    prefetch_depth: int
    prefetch_chunk_mb: int

class SDXLStreamingRuntimeState:
    def __init__(self) -> None:
        self._lock = RLock()
        self._key: SDXLStreamingSpineKey | None = None
        self._spine: StreamingUnetSpine | None = None

    @staticmethod
    def _build_key(request: SDXLAssemblyRequest) -> SDXLStreamingSpineKey:
        return SDXLStreamingSpineKey(
            checkpoint_sha256=request.checkpoint.sha256,
            device=request.device,
            prefetch_depth=request.prefetch_depth,
            prefetch_chunk_mb=request.prefetch_chunk_mb,
        )

    def acquire(self, request: SDXLAssemblyRequest) -> Tuple[StreamingUnetSpine, bool]:
        requested_key = self._build_key(request)
        stale_spine: StreamingUnetSpine | None = None

        with self._lock:
            if self._spine is not None and self._key == requested_key:
                self._spine.request = request
                logger.debug(
                    "[SDXL Telemetry] Reusing cached SDXL UNet spine shell for key=%s",
                    requested_key,
                )
                return self._spine, True

            stale_spine = self._spine
            self._spine = None
            self._key = None

        if stale_spine is not None:
            logger.debug("[SDXL Telemetry] Releasing stale streaming UNet spine before replacement.")
            stale_spine.end()

        spine = StreamingUnetSpine(request)
        logger.debug(
            "[SDXL Telemetry] Creating new cached SDXL UNet spine shell for key=%s",
            requested_key,
        )

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
        spine.end()
        return True

    def get_active_key(self) -> SDXLStreamingSpineKey | None:
        with self._lock:
            return self._key

_STREAMING_RUNTIME_STATE = SDXLStreamingRuntimeState()

def acquire_active_sdxl_streaming_spine(request: SDXLAssemblyRequest) -> Tuple[StreamingUnetSpine, bool]:
    return _STREAMING_RUNTIME_STATE.acquire(request)

def release_active_sdxl_streaming_spine(reason: str | None = None) -> bool:
    return _STREAMING_RUNTIME_STATE.release(reason=reason)

def clear_all_caches() -> None:
    """Teardown and unload all cached workers and spines."""
    release_active_sdxl_streaming_spine(reason="global_cleanup")
