from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import RLock
from backend.flux_fill_v3.contracts import FluxFillRequest, FluxLatentArtifactBundle
from backend.flux_fill_v3.streaming_spine import StreamingUnetSpine
from backend.flux_fill_v3.resident_spine import ResidentUnetSpine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FluxStreamingSpineKey:
    unet_path: str
    device: str
    prefetch_depth: int
    prefetch_chunk_mb: int


class FluxStreamingRuntimeState:
    def __init__(self) -> None:
        self._lock = RLock()
        self._key: FluxStreamingSpineKey | None = None
        self._spine: StreamingUnetSpine | None = None

    @staticmethod
    def _build_key(request: FluxFillRequest) -> FluxStreamingSpineKey:
        return FluxStreamingSpineKey(
            unet_path=str(request.unet_path),
            device=str(request.device or ""),
            prefetch_depth=int(getattr(request, "prefetch_depth", 1) or 0),
            prefetch_chunk_mb=int(getattr(request, "prefetch_chunk_mb", 64) or 0),
        )

    def acquire(self, request: FluxFillRequest) -> tuple[StreamingUnetSpine, bool]:
        requested_key = self._build_key(request)
        stale_spine: StreamingUnetSpine | None = None

        with self._lock:
            if self._spine is not None and self._key == requested_key:
                self._spine.request = request
                logger.debug(
                    "[Flux Telemetry] Reusing cached Flux UNet spine shell for key=%s started=%s",
                    requested_key,
                    bool(self._spine.started and self._spine.unet_patcher is not None),
                )
                return self._spine, True

            stale_spine = self._spine
            self._spine = None
            self._key = None

        if stale_spine is not None:
            logger.debug("[Flux Telemetry] Releasing stale streaming UNet spine before replacement.")
            stale_spine.end()

        spine = StreamingUnetSpine(request)
        logger.debug(
            "[Flux Telemetry] Creating new cached Flux UNet spine shell for key=%s",
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
            "[Flux Telemetry] Releasing active streaming UNet spine reason=%s key=%s",
            reason,
            key,
        )
        spine.end()
        return True

    def get_active_key(self) -> FluxStreamingSpineKey | None:
        with self._lock:
            return self._key


_STREAMING_RUNTIME_STATE = FluxStreamingRuntimeState()


def acquire_active_flux_streaming_spine(request: FluxFillRequest) -> tuple[StreamingUnetSpine, bool]:
    return _STREAMING_RUNTIME_STATE.acquire(request)


def get_active_flux_streaming_spine_key() -> FluxStreamingSpineKey | None:
    return _STREAMING_RUNTIME_STATE.get_active_key()


@dataclass(frozen=True)
class FluxResidentSpineKey:
    unet_path: str
    device: str


class FluxResidentRuntimeState:
    def __init__(self) -> None:
        self._lock = RLock()
        self._key: FluxResidentSpineKey | None = None
        self._spine: ResidentUnetSpine | None = None

    @staticmethod
    def _build_key(request: FluxFillRequest) -> FluxResidentSpineKey:
        return FluxResidentSpineKey(
            unet_path=str(request.unet_path),
            device=str(request.device or ""),
        )

    def acquire(self, request: FluxFillRequest) -> tuple[ResidentUnetSpine, bool]:
        requested_key = self._build_key(request)
        stale_spine: ResidentUnetSpine | None = None

        with self._lock:
            if self._spine is not None and self._key == requested_key:
                self._spine.request = request
                logger.debug(
                    "[Flux Telemetry] Reusing cached resident Flux UNet spine shell for key=%s started=%s",
                    requested_key,
                    bool(self._spine.started and self._spine.unet_patcher is not None),
                )
                return self._spine, True

            stale_spine = self._spine
            self._spine = None
            self._key = None

        if stale_spine is not None:
            logger.debug("[Flux Telemetry] Releasing stale resident UNet spine before replacement.")
            stale_spine.end()

        spine = ResidentUnetSpine(request)
        logger.debug(
            "[Flux Telemetry] Creating new cached resident Flux UNet spine shell for key=%s",
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
            "[Flux Telemetry] Releasing active resident UNet spine reason=%s key=%s",
            reason,
            key,
        )
        spine.end()
        return True

    def get_active_key(self) -> FluxResidentSpineKey | None:
        with self._lock:
            return self._key


_RESIDENT_RUNTIME_STATE = FluxResidentRuntimeState()


def acquire_active_flux_resident_spine(request: FluxFillRequest) -> tuple[ResidentUnetSpine, bool]:
    return _RESIDENT_RUNTIME_STATE.acquire(request)


def get_active_flux_resident_spine_key() -> FluxResidentSpineKey | None:
    return _RESIDENT_RUNTIME_STATE.get_active_key()


class FluxLatentArtifactState:
    def __init__(self) -> None:
        self._lock = RLock()
        self._bundle: FluxLatentArtifactBundle | None = None

    def get_bundle(self, fingerprint: str) -> FluxLatentArtifactBundle | None:
        with self._lock:
            if self._bundle is not None and self._bundle.fingerprint == fingerprint:
                return self._bundle
            return None

    def set_bundle(self, bundle: FluxLatentArtifactBundle) -> None:
        with self._lock:
            self._bundle = bundle

    def release(self) -> bool:
        with self._lock:
            if self._bundle is None:
                return False
            self._bundle = None
            return True


_LATENT_ARTIFACT_STATE = FluxLatentArtifactState()


def get_cached_latent_artifact_bundle(fingerprint: str) -> FluxLatentArtifactBundle | None:
    return _LATENT_ARTIFACT_STATE.get_bundle(fingerprint)


def set_cached_latent_artifact_bundle(bundle: FluxLatentArtifactBundle) -> None:
    _LATENT_ARTIFACT_STATE.set_bundle(bundle)


def release_flux_latent_artifacts() -> bool:
    return _LATENT_ARTIFACT_STATE.release()


def release_active_flux_resident_spine(*args, **kwargs) -> bool:
    reason = kwargs.get("reason")
    released_streaming = _STREAMING_RUNTIME_STATE.release(reason=reason)
    released_resident = _RESIDENT_RUNTIME_STATE.release(reason=reason)

    released_t5 = False
    try:
        from backend.flux_fill_v3.cpu_resident_text_worker import CpuResidentTextEncoderCache
        released_t5 = CpuResidentTextEncoderCache.teardown()
    except Exception:
        pass

    if released_streaming or released_resident:
        try:
            from backend.host_cache import flush_pinned_host_cache
            flush_pinned_host_cache()
        except Exception:
            pass

    return released_streaming or released_resident or released_t5
