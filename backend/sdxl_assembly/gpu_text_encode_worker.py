from backend.sdxl_assembly.contracts import SDXLAssemblyRequest

class GpuTextEncodeWorker:
    """Worker representing GpuTextEncodeWorker (CLIP-L and CLIP-G on GPU).
    
    Deferred to W08+.
    """
    def __init__(self, request: SDXLAssemblyRequest) -> None:
        raise NotImplementedError("GPU-pinned text encoder worker posture is deferred to W08+ and not supported in W02.")
