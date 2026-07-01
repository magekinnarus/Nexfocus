from typing import Any, Optional
import torch
from backend.sdxl_assembly.contracts import SDXLAssemblyRequest, SDXLAssemblyResult

class StreamingUnetSpine:
    """Worker representing StreamingUnetSpine (CPU-pinned weights streamed slice-by-slice).
    
    W02 Shim only; no real UNet execution is performed.
    """
    def __init__(self, request: SDXLAssemblyRequest) -> None:
        self.request = request
        self.is_active = False

    def start(self) -> None:
        self.is_active = True

    def denoise(self, latent: torch.Tensor, conditioning: Any, callback: Optional[Any] = None) -> torch.Tensor:
        """Runs the denoise loop.
        
        W02 shim returns mock latent.
        """
        # Extension Point: Per-step ControlNet residual/application hook
        self._apply_step_control_residual(step=0)
        return latent

    def end(self) -> None:
        self.is_active = False

    # Extension Points for ControlNet / Adapters
    def load_control_model(self, control_model: Any) -> None:
        """Extension point for loading structural ControlNet models."""
        raise NotImplementedError("ControlNet models are not supported in W02 streaming UNet spine.")

    def _apply_step_control_residual(self, step: int) -> None:
        """Extension point for applying per-step control residuals/hooks."""
        pass

    def teardown_assembly_order(self) -> None:
        """Extension point for tracking teardown order."""
        pass
