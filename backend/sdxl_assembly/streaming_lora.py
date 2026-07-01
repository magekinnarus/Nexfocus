from typing import Any
from backend.sdxl_assembly.contracts import SDXLAssemblyRequest

class StreamingLoraPatchWorker:
    """Worker representing StreamingLoraPatchWorker (materializes patches on CPU-pinned weights).
    
    W02 Shim only; does not perform real LoRA patching.
    """
    def __init__(self, request: SDXLAssemblyRequest) -> None:
        self.request = request

    def materialize_patches(self) -> Any:
        """Parses and compiles LoRA patches.
        
        W02 shim returns mock patches.
        """
        return None

    def teardown_assembly_order(self) -> None:
        """Extension point for tracking teardown order."""
        pass
