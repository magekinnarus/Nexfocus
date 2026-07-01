from __future__ import annotations

from backend.sdxl_assembly.contracts import SDXLAssemblyRequest
from backend.sdxl_assembly.runtime_state import acquire_active_sdxl_streaming_spine
from backend.sdxl_assembly.streaming_unet import StreamingUnetSpine
from backend.sdxl_assembly.cpu_pinned_text import CpuPinnedTextEncoderWorker
from backend.sdxl_assembly.transient_vae import TransientVaeWorker
from backend.sdxl_assembly.streaming_lora import StreamingLoraPatchWorker

class SDXLAssemblyAssembler:
    """Factory class to acquire posture-specific workers and spines."""

    @staticmethod
    def acquire_unet_spine(
        request: SDXLAssemblyRequest,
        lora_worker: StreamingLoraPatchWorker | None = None,
    ) -> StreamingUnetSpine:
        spine, _reused = acquire_active_sdxl_streaming_spine(request, lora_worker=lora_worker)
        return spine

    @staticmethod
    def acquire_text_worker(
        request: SDXLAssemblyRequest,
        lora_worker: StreamingLoraPatchWorker | None = None,
    ) -> CpuPinnedTextEncoderWorker:
        return CpuPinnedTextEncoderWorker(request, lora_worker=lora_worker)

    @staticmethod
    def acquire_vae_worker(request: SDXLAssemblyRequest) -> TransientVaeWorker:
        return TransientVaeWorker(request)

    @staticmethod
    def acquire_lora_worker(request: SDXLAssemblyRequest) -> StreamingLoraPatchWorker:
        return StreamingLoraPatchWorker(request)
