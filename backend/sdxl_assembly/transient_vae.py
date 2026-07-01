from typing import Any, Tuple
import numpy as np
import torch
from backend.sdxl_assembly.contracts import SDXLAssemblyRequest

class TransientVaeWorker:
    """Worker representing TransientVaeWorker (loads/unloads VAE transiently).
    
    W02 Shim only; does not perform real VAE encoding/decoding.
    """
    def __init__(self, request: SDXLAssemblyRequest) -> None:
        self.request = request

    def prepare_latents(self, device: torch.device) -> Any:
        """Extension point for loading VAE and encoding/preparing latents.
        
        W02 shim returns mock latent bundle.
        """
        # Returns a mock object representing latent bundle
        return type("MockLatentBundle", (), {"samples": torch.zeros((1, 4, 64, 64)), "fingerprint": "mock_fp"})()

    def decode(self, latent: torch.Tensor, device: torch.device) -> Tuple[np.ndarray, float, float]:
        """Decodes latents to HWC RGB numpy image.
        
        W02 shim returns a mock blank image.
        """
        mock_image = np.zeros((self.request.height, self.request.width, 3), dtype=np.uint8)
        return mock_image, 0.0, 0.0

    def teardown_assembly_order(self) -> None:
        """Extension point for tracking teardown order."""
        pass
