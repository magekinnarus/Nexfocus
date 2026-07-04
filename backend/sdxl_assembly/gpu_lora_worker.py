from backend.sdxl_assembly.contracts import SDXLAssemblyRequest

class GpuLoraWorker:
    """Worker representing GpuLoraWorker (materializes patches directly in GPU VRAM).
    
    Deferred to W08+.
    """
    def __init__(self, request: SDXLAssemblyRequest) -> None:
        raise NotImplementedError("Resident LoRA patch worker posture is deferred to W08+ and not supported in W02.")
