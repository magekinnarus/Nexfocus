from typing import Any, Dict
from backend.sdxl_assembly.contracts import SDXLAssemblyRequest

class CpuPinnedTextEncoderWorker:
    """Worker representing CpuPinnedTextEncoderWorker (CLIP-L and CLIP-G on CPU).
    
    W02 Shim only; does not perform real text encoding.
    """
    def __init__(self, request: SDXLAssemblyRequest) -> None:
        self.request = request

    def get_conditioning(self) -> Dict[str, Any]:
        """Encodes positive and negative prompts on CPU.
        
        W02 shim returns mock dict.
        """
        return {
            "positive_embeds": None,
            "negative_embeds": None,
        }

    def teardown_assembly_order(self) -> None:
        """Extension point for tracking teardown order."""
        pass
