from backend.flux_fill_v3.contracts import FluxFillRequest, T5PostureKind, UNetSpineKind
from backend.flux_fill_v3.assembly import FluxAssembly
from backend.flux_fill_v3.runtime_state import (
    acquire_active_flux_streaming_spine,
    acquire_active_flux_resident_spine,
)
from backend.flux_fill_v3.t5_worker import DiskPagedTextWorker
from backend.flux_fill_v3.cpu_resident_text_worker import CpuResidentTextWorker
from backend.flux_fill_v3.vae_worker import TransientVaeWorker


class FluxAssemblyDirector:
    """Authoritative director for assembly selection and instantiation."""

    @staticmethod
    def _select_text_worker(request: FluxFillRequest) -> DiskPagedTextWorker | CpuResidentTextWorker:
        if request.t5_posture == T5PostureKind.DISK_PAGED:
            return DiskPagedTextWorker(request)
        if request.t5_posture == T5PostureKind.CPU_RESIDENT:
            return CpuResidentTextWorker(request)
        raise NotImplementedError(f"Unsupported Flux Fill text-worker posture: {request.t5_posture!r}")

    @staticmethod
    def select_assembly(
        request: FluxFillRequest,
        *,
        status_callback=None,
        progress_state=None,
    ) -> FluxAssembly:
        if request.unet_spine == UNetSpineKind.RESIDENT:
            spine, _reused = acquire_active_flux_resident_spine(request)
        else:
            spine, _reused = acquire_active_flux_streaming_spine(request)

        text_worker = FluxAssemblyDirector._select_text_worker(request)
        vae_worker = TransientVaeWorker(request)
        return FluxAssembly(
            spine,
            text_worker,
            vae_worker,
            release_spine_after_execute=False,
            status_callback=status_callback,
            progress_state=progress_state,
        )
