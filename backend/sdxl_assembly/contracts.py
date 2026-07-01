from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Posture Enums
class UNetPostureKind(str, Enum):
    STREAMING = "streaming"
    RESIDENT = "resident"

class TextEncoderPostureKind(str, Enum):
    CPU_PINNED = "cpu_pinned"
    GPU_PINNED = "gpu_pinned"

class VAEPostureKind(str, Enum):
    TRANSIENT = "transient"

class LoraPatchPostureKind(str, Enum):
    STREAMING = "streaming"
    RESIDENT = "resident"

@dataclass(frozen=True)
class ResolvedFileIdentity:
    path: Path
    sha256: str
    size_bytes: int
    modified_ns: int

@dataclass(frozen=True)
class SDXLLoraSpec:
    file_identity: ResolvedFileIdentity
    unet_weight: float
    clip_weight: float
    enabled: bool = True

@dataclass(frozen=True)
class SDXLRuntimeIdentity:
    checkpoint: ResolvedFileIdentity
    vae: Optional[ResolvedFileIdentity]
    unet_posture: UNetPostureKind
    clip_posture: TextEncoderPostureKind
    vae_posture: VAEPostureKind
    lora_posture: LoraPatchPostureKind

    def as_dict(self) -> Dict[str, str | None]:
        return {
            "unet_posture": self.unet_posture.value,
            "clip_posture": self.clip_posture.value,
            "vae_posture": self.vae_posture.value,
            "lora_posture": self.lora_posture.value,
            "checkpoint_sha256": self.checkpoint.sha256,
            "vae_sha256": self.vae.sha256 if self.vae else None,
        }

@dataclass(frozen=True)
class SDXLAssemblyRequest:
    # Queue and route identity: one request per concrete prompt/image task
    request_id: str
    route_id: str
    image_index: int
    image_count: int
    
    # Model Identity
    checkpoint: ResolvedFileIdentity
    vae: Optional[ResolvedFileIdentity]
    model_variant_key: str
    
    # Prompt Payloads (frozen copies)
    prompt: str
    negative_prompt: str
    positive_texts: Tuple[str, ...]
    negative_texts: Tuple[str, ...]
    
    # Denoise Inputs
    width: int
    height: int
    steps: int
    cfg: float
    sampler: str
    scheduler: str
    seed: int
    clip_layer: int = -2
    style_selections: Tuple[str, ...] = field(default_factory=tuple)
    prompt_payload_hash: str = ""
    
    # LoRA stack snapshot
    lora_specs: Tuple[SDXLLoraSpec, ...] = field(default_factory=tuple)
    lora_stack_hash: str = ""
    
    # Selected Postures
    unet_posture: UNetPostureKind = UNetPostureKind.STREAMING
    clip_posture: TextEncoderPostureKind = TextEncoderPostureKind.CPU_PINNED
    vae_posture: VAEPostureKind = VAEPostureKind.TRANSIENT
    lora_posture: LoraPatchPostureKind = LoraPatchPostureKind.STREAMING
    
    # Streaming parameters
    prefetch_depth: int = 1
    prefetch_chunk_mb: int = 64
    
    # Execution Seams
    device: str = "cuda"
    tiled: bool = False
    denoise_strength: Optional[float] = None
    sharpness: float = 2.0
    adaptive_cfg: float = 7.0
    adm_scaler_positive: float = 1.5
    adm_scaler_negative: float = 0.8
    adm_scaler_end: float = 0.3
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Explicit future extension descriptors for ControlNet/adapters/spatial inputs
    # left empty/unsupported in W02 so later streaming ControlNet slices can attach.
    structural_tasks: Dict[str, Any] = field(default_factory=dict)
    controlnet_paths: Dict[str, Any] = field(default_factory=dict)
    contextual_tasks: Dict[str, Any] = field(default_factory=dict)
    contextual_assets: Dict[str, Any] = field(default_factory=dict)
    initial_latent: Any = None
    disable_initial_latent: bool = False
    
    def validate(self) -> None:
        """Enforces minimum parameter checks on an already-resolved snapshot."""
        if not self.checkpoint.path.exists():
            raise SDXLAssemblyValidationError(f"Base checkpoint does not exist: {self.checkpoint.path}")
        if self.vae and not self.vae.path.exists():
            raise SDXLAssemblyValidationError(f"VAE checkpoint does not exist: {self.vae.path}")
        if self.steps < 1:
            raise SDXLAssemblyValidationError(f"Steps must be >= 1, got {self.steps}")
        if self.prefetch_depth < 0:
            raise SDXLAssemblyValidationError(f"Prefetch depth must be >= 0, got {self.prefetch_depth}")
        if self.prefetch_chunk_mb < 1:
            raise SDXLAssemblyValidationError(f"Prefetch chunk MB must be >= 1, got {self.prefetch_chunk_mb}")
        if self.width <= 0 or self.height <= 0:
            raise SDXLAssemblyValidationError(f"Width and Height must be positive, got {self.width}x{self.height}")

@dataclass(frozen=True)
class SDXLAssemblyResult:
    output_image: np.ndarray  # HWC RGB
    seed: int
    width: int
    height: int
    runtime_identity: SDXLRuntimeIdentity
    timings: Dict[str, float]
    metadata: Dict[str, Any]

class SDXLAssemblyEligibilityError(Exception):
    """Raised when a request is ineligible for the new assembly pipeline."""
    pass

class SDXLAssemblyValidationError(ValueError):
    """Raised when request static validation fails."""
    pass
