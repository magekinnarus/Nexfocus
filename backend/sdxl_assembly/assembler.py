from __future__ import annotations

from backend.sdxl_assembly.contracts import SDXLAssemblyRequest, UNetPostureKind
from backend.sdxl_assembly.runtime_state import acquire_active_sdxl_streaming_spine, acquire_active_sdxl_resident_spine
from backend.sdxl_assembly.streaming_unet import StreamingUnetSpine
from backend.sdxl_assembly.resident_unet import ResidentUnetSpine
from backend.sdxl_assembly.cpu_text_encode_worker import CpuTextEncodeWorker
from backend.sdxl_assembly.vae_decode_worker import TransientVaeDecodeWorker
from backend.sdxl_assembly.cpu_lora_worker import CpuLoraWorker
from backend.sdxl_assembly.gpu_lora_worker import GpuLoraWorker
from backend.sdxl_assembly.vae_encode_worker import VaeEncodeWorker
from backend.sdxl_assembly.spatial_context_worker import SpatialContextWorker
from backend.sdxl_assembly.stream_st_preprocess_worker import StreamingStructuralPreprocessWorker
from backend.sdxl_assembly.stream_st_cn_worker import StreamingStructuralControlWorker
from backend.sdxl_assembly.stream_ctx_cn_worker import StreamingContextualControlWorker

class SDXLAssemblyAssembler:
    """Factory class to acquire posture-specific workers and spines."""

    @staticmethod
    def acquire_unet_spine(
        request: SDXLAssemblyRequest,
        lora_worker: CpuLoraWorker | GpuLoraWorker | None = None,
    ) -> StreamingUnetSpine | ResidentUnetSpine:
        if request.unet_posture == UNetPostureKind.RESIDENT:
            spine, _reused = acquire_active_sdxl_resident_spine(request, lora_worker=lora_worker)
            return spine
        else:
            spine, _reused = acquire_active_sdxl_streaming_spine(request, lora_worker=lora_worker)
            return spine

    @staticmethod
    def acquire_text_encode_worker(
        request: SDXLAssemblyRequest,
        lora_worker: CpuLoraWorker | GpuLoraWorker | None = None,
    ) -> CpuTextEncodeWorker | GpuTextEncodeWorker:
        from backend.sdxl_assembly.contracts import TextEncoderPostureKind
        if request.clip_posture == TextEncoderPostureKind.GPU_RESIDENT:
            from backend.sdxl_assembly.gpu_text_encode_worker import GpuTextEncodeWorker
            return GpuTextEncodeWorker(request, lora_worker=lora_worker)
        return CpuTextEncodeWorker(request, lora_worker=lora_worker)

    @staticmethod
    def acquire_vae_decode_worker(request: SDXLAssemblyRequest) -> TransientVaeDecodeWorker:
        return TransientVaeDecodeWorker(request)

    @staticmethod
    def acquire_cpu_lora_worker(request: SDXLAssemblyRequest) -> CpuLoraWorker:
        return CpuLoraWorker(request)

    @staticmethod
    def acquire_gpu_lora_worker(request: SDXLAssemblyRequest) -> GpuLoraWorker:
        return GpuLoraWorker(request)

    @staticmethod
    def acquire_vae_encode_worker(request: SDXLAssemblyRequest) -> VaeEncodeWorker:
        return VaeEncodeWorker(request)

    @staticmethod
    def acquire_spatial_context_worker(request: SDXLAssemblyRequest) -> SpatialContextWorker:
        return SpatialContextWorker(request)

    @staticmethod
    def acquire_st_preprocess_worker(request: SDXLAssemblyRequest) -> StreamingStructuralPreprocessWorker:
        return StreamingStructuralPreprocessWorker(request)

    @staticmethod
    def acquire_st_control_worker(request: SDXLAssemblyRequest) -> StreamingStructuralControlWorker:
        return StreamingStructuralControlWorker(request)

    @staticmethod
    def acquire_ctx_control_worker(request: SDXLAssemblyRequest) -> StreamingContextualControlWorker:
        return StreamingContextualControlWorker(request)

