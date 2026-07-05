from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

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
class SpatialImageDescriptor:
    fingerprint: str
    pixels: torch.Tensor  # shape [B, H, W, C], float32 CPU tensor, clamped to [0.0, 1.0]

@dataclass(frozen=True)
class SpatialMaskDescriptor:
    fingerprint: str
    mask: torch.Tensor  # shape [B, H, W], float32 CPU tensor, values 0.0 or 1.0

@dataclass(frozen=True)
class SpatialContextDescriptor:
    mode: str  # "image", "inpaint", or "outpaint"
    source_image: SpatialImageDescriptor
    source_mask: Optional[SpatialMaskDescriptor] = None
    target_width: int = 1024
    target_height: int = 1024
    denoise_strength: Optional[float] = None
    
    # Bounding box geometry
    bbox: Optional[Tuple[int, int, int, int]] = None
    bbox_area_ratio: float = 1.0
    
    # Pre-prepared properties if resolved early (fully frozen!)
    pre_bb_image: Optional[SpatialImageDescriptor] = None
    pre_bb_mask: Optional[SpatialMaskDescriptor] = None
    pre_blend_mask: Optional[SpatialMaskDescriptor] = None
    
    # Outpaint specific
    outpaint_direction: Optional[str] = None
    outpaint_expansion_size: int = 0
    outpaint_pixelate: bool = False

@dataclass(frozen=True)
class PreparedSpatialContext:
    mode: str
    original_pixels: torch.Tensor # CPU float32 [B, H, W, C]
    bb_pixels: torch.Tensor # CPU float32 [B, H, W, C] (resized to target w/h)
    image_fingerprint: str
    bb_pixels_fingerprint: str

    original_mask: Optional[torch.Tensor] = None # CPU float32 [B, H, W]
    bb_mask: Optional[torch.Tensor] = None # CPU float32 [B, H, W] (resized to target w/h)
    blend_mask: Optional[torch.Tensor] = None # CPU float32 [B, H, W] (blend mask)
    
    # For outpaint working images
    working_pixels: Optional[torch.Tensor] = None
    working_mask: Optional[torch.Tensor] = None
    
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    bbox_area_ratio: float = 1.0
    mask_coverage: float = 0.0
    
    mask_fingerprint: Optional[str] = None
    bb_mask_fingerprint: Optional[str] = None

    def get_cache_key(self, vae_identity: str) -> str:
        # Construct a stable key from the image/mask/geometry domain plus VAE identity.
        import hashlib
        import json
        payload = {
            "mode": self.mode,
            "vae_identity": vae_identity,
            "image_fingerprint": self.image_fingerprint,
            "mask_fingerprint": self.mask_fingerprint,
            "bb_pixels_fingerprint": self.bb_pixels_fingerprint,
            "bb_mask_fingerprint": self.bb_mask_fingerprint,
            "bbox": tuple(int(v) for v in self.bbox),
            "bbox_area_ratio": round(float(self.bbox_area_ratio), 8),
            "mask_coverage": round(float(self.mask_coverage), 8),
            "bb_shape": tuple(int(dim) for dim in self.bb_pixels.shape),
            "working_shape": (
                tuple(int(dim) for dim in self.working_pixels.shape)
                if self.working_pixels is not None
                else None
            ),
        }
        payload_str = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

@dataclass(frozen=True)
class SpatialAssemblyArtifacts:
    # The actual latent tensors (remain CPU parked!)
    route_latent: torch.Tensor          # Initial latent samples [B, 4, H, W]
    source_fingerprint: str
    image_fingerprint: str
    route_latent_fingerprint: str

    masked_latent: Optional[torch.Tensor] = None # Masked source latent if inpaint/outpaint [B, 4, H, W]
    bb_latent: Optional[torch.Tensor] = None     # Bounding box region latent if inpaint/outpaint [B, 4, H, W]
    denoise_mask: Optional[torch.Tensor] = None  # Denoise mask tensor [B, 1, H, W]
    blend_mask: Optional[torch.Tensor] = None    # Blend mask for compositing [B, 1, H, W]
    
    # Metadata and Fingerprints
    mask_fingerprint: Optional[str] = None
    masked_latent_fingerprint: Optional[str] = None
    bb_latent_fingerprint: Optional[str] = None
    denoise_mask_fingerprint: Optional[str] = None
    blend_mask_fingerprint: Optional[str] = None
    
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    bbox_area_ratio: float = 1.0
    mask_coverage: float = 0.0
    
    # Telemetry
    cache_hit: bool = False
    encode_wall: float = 0.0

def make_spatial_image_descriptor(pixels: Any) -> SpatialImageDescriptor:
    """Creates an immutable, normalized, copy-on-creation SpatialImageDescriptor."""
    if pixels is None:
        raise ValueError("pixels cannot be None for SpatialImageDescriptor")
    tensor = torch.as_tensor(pixels).detach().cpu()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[-1] < 3:
        raise ValueError("source_pixels must have shape [B, H, W, C] or [H, W, C] with at least 3 channels.")
    
    tensor = tensor[..., :3].to(dtype=torch.float32).clone()  # Clone to ensure no mutation
    if tensor.numel() and float(tensor.max().item()) > 1.0:
        tensor = tensor / 255.0
    tensor = tensor.clamp(0.0, 1.0)
    
    import hashlib
    h = hashlib.sha256(tensor.numpy().tobytes())
    fingerprint = h.hexdigest()
    return SpatialImageDescriptor(fingerprint=fingerprint, pixels=tensor)

def make_spatial_mask_descriptor(mask: Any, image_descriptor: SpatialImageDescriptor) -> SpatialMaskDescriptor:
    """Creates an immutable, normalized, copy-on-creation SpatialMaskDescriptor matching image dimensions."""
    if mask is None:
        raise ValueError("mask cannot be None for SpatialMaskDescriptor")
    tensor = torch.as_tensor(mask).detach().cpu()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3 and tensor.shape[:2] == image_descriptor.pixels.shape[1:3]:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim == 4:
        tensor = tensor.amax(dim=-1)
    if tensor.ndim != 3:
        raise ValueError("source_mask must have shape [B, H, W], [H, W], or [B, H, W, C].")
        
    pixels_shape = image_descriptor.pixels.shape
    if tensor.shape[0] == 1 and pixels_shape[0] > 1:
        tensor = tensor.repeat(int(pixels_shape[0]), 1, 1)
        
    if tensor.shape[0] != pixels_shape[0] or tensor.shape[1] != pixels_shape[1] or tensor.shape[2] != pixels_shape[2]:
        raise ValueError("source_mask must match source_pixels spatial shape and batch.")
        
    tensor = tensor.to(dtype=torch.float32).clone()
    if tensor.numel() and float(tensor.max().item()) > 1.0:
        tensor = tensor / 255.0
    tensor = (tensor > 0.5).to(dtype=torch.float32)
    
    import hashlib
    h = hashlib.sha256(tensor.numpy().tobytes())
    fingerprint = h.hexdigest()
    return SpatialMaskDescriptor(fingerprint=fingerprint, mask=tensor)


@dataclass(frozen=True)
class SDXLStructuralControlDescriptor:
    slot_index: int
    control_type: str
    image_pixels: torch.Tensor  # shape [B, H, W, C] or [H, W, C]
    image_fingerprint: str
    
    preprocessor_id: Optional[str]
    preprocessor_path: Optional[Path]
    preprocessor_params: Dict[str, Any]
    
    target_width: int
    target_height: int
    
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_type: str  # "controlnet" or "control_lora" or "lllite"
    
    weight: float = 1.0
    start_percent: float = 0.0
    end_percent: float = 1.0
    
    unsupported_mode_errors: Tuple[str, ...] = field(default_factory=tuple)
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StructuralHintArtifact:
    slot_index: int
    control_type: str
    hint_tensor: torch.Tensor  # CPU tensor [1, C, H, W], float32 [0.0, 1.0]
    hint_fingerprint: str
    cache_hit: bool = False
    preprocess_wall: float = 0.0


@dataclass(frozen=True)
class SDXLContextualControlDescriptor:
    ui_slot_index: int
    control_type: str  # "ImagePrompt" or "PuLID"
    image_pixels: torch.Tensor  # shape [B, H, W, C]
    image_fingerprint: str
    source_image_role: str
    
    # Model/Support identities
    model_path: Path
    model_sha256: str
    clip_vision_path: Optional[Path] = None
    clip_vision_sha256: Optional[str] = None
    ip_negative_path: Optional[Path] = None
    ip_negative_sha256: Optional[str] = None
    eva_clip_path: Optional[Path] = None
    eva_clip_sha256: Optional[str] = None
    insightface_model_names: Tuple[str, ...] = field(default_factory=tuple)
    
    # Payload-affecting preprocess parameters
    preprocess_params: Dict[str, Any] = field(default_factory=dict)
    
    # Application parameters (do NOT invalidate the payload artifact cache)
    weight: float = 1.0
    start_percent: float = 0.0
    end_percent: float = 1.0
    
    unsupported_mode_errors: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ContextualPayloadArtifact:
    ui_slot_index: int
    control_type: str
    payload: Tuple[List[torch.Tensor], List[torch.Tensor]]  # (ip_conds, ip_unconds) CPU-parked tensors
    payload_fingerprint: str
    cache_hit: bool = False
    preprocess_wall: float = 0.0


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
    spatial_context: Optional[SpatialContextDescriptor] = None
    
    # Structural Control descriptors added in W07
    structural_controls: Tuple[SDXLStructuralControlDescriptor, ...] = field(default_factory=tuple)
    
    # Contextual Control descriptors added in W08
    contextual_controls: Tuple[SDXLContextualControlDescriptor, ...] = field(default_factory=tuple)
    
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

        import modules.flags as flags
        for desc in self.structural_controls:
            if desc.slot_index <= 0:
                raise SDXLAssemblyValidationError(f"Structural control slot index must be positive, got {desc.slot_index}")
            if desc.control_type not in getattr(flags, "cn_structural_types", []):
                raise SDXLAssemblyValidationError(f"Unsupported structural control type: {desc.control_type}")
            if not desc.checkpoint_path.exists():
                raise SDXLAssemblyValidationError(f"Structural control checkpoint does not exist: {desc.checkpoint_path}")
            if len(desc.unsupported_mode_errors) > 0:
                raise SDXLAssemblyValidationError(f"Unsupported structural mode: {desc.unsupported_mode_errors}")

        for desc in self.contextual_controls:
            if desc.ui_slot_index < 0 or desc.ui_slot_index > 3:
                raise SDXLAssemblyValidationError(f"Contextual control slot index must be in range 0..3, got {desc.ui_slot_index}")
            if desc.control_type == "FaceID V2" or desc.control_type == "FaceSwap":
                raise SDXLAssemblyValidationError(f"FaceID V2 is explicitly retired on the new assembly path.")
            if desc.control_type not in getattr(flags, "cn_contextual_types", []):
                raise SDXLAssemblyValidationError(f"Unsupported contextual control type: {desc.control_type}")
            if not desc.model_path.exists():
                raise SDXLAssemblyValidationError(f"Contextual control model path does not exist: {desc.model_path}")
            if len(desc.unsupported_mode_errors) > 0:
                raise SDXLAssemblyValidationError(f"Unsupported contextual mode: {desc.unsupported_mode_errors}")

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
