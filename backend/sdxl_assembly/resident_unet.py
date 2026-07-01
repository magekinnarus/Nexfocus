from backend.sdxl_assembly.contracts import SDXLAssemblyRequest

class ResidentUnetSpine:
    """Worker representing ResidentUnetSpine (GPU-pinned resident weights).
    
    Deferred to W08.
    """
    def __init__(self, request: SDXLAssemblyRequest) -> None:
        raise NotImplementedError("Resident UNet spine posture is deferred to W08 and not supported in W02.")
