from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.staging_manager import ExecutionClass, PlacementSolver, ResidencyMode


EXECUTION_FAMILY_STANDARD = "standard_sdxl"
GPU_PREFERRED_VRAM_THRESHOLD_MB = 12 * 1024

CLIP_RESIDENCY_CPU_ONLY = "cpu_only"
CLIP_RESIDENCY_GPU_THEN_OFFLOAD = "gpu_then_offload"
CLIP_RESIDENCY_GPU_RESIDENT = "gpu_resident"

VAE_ENCODE_CPU_DEFAULT = "cpu_default"
VAE_ENCODE_GPU_PREFERRED = "gpu_preferred"
VAE_POSTURE_TRANSIENT_GPU = "transient_gpu"
VAE_POSTURE_GPU_RESIDENT = "gpu_resident"

SDXL_RESIDENCY_CLASS_FULL = "full_resident"
SDXL_RESIDENCY_CLASS_UNIFIED_STREAMING = "unified_streaming"


@dataclass(frozen=True)
class SDXLExecutionPolicy:
    enabled: bool
    architecture: str | None = None
    runtime_family: str | None = None
    execution_mode: str | None = None
    hardware_tier: str | None = None
    # Legacy compatibility field. Active resident SDXL no longer uses clean-shadow posture.
    allow_cpu_shadow: bool = False
    prefer_clip_gpu: bool = False
    prefer_gpu_vae_encode: bool = False
    stream_budget_mb: float = 256.0
    # Legacy compatibility field retained for callers that still pass it explicitly.
    resident_clean_source_device: str = "cpu"
    notes: tuple[str, ...] = field(default_factory=tuple)

    # Compatibility fields retained for the maintained SDXL callers.
    execution_class: Any = None
    execution_family: str | None = None
    residency_class: str | None = None
    clip_residency_mode: str | None = None
    vae_encode_mode: str | None = None
    keep_clip_loaded: bool | None = None

    def __post_init__(self) -> None:
        runtime_family = "unified_sdxl"
        execution_mode = self.execution_mode
        prefer_clip_gpu = self.prefer_clip_gpu
        prefer_gpu_vae_encode = self.prefer_gpu_vae_encode

        if self.residency_class is None:
            residency_class = (
                SDXL_RESIDENCY_CLASS_UNIFIED_STREAMING
                if execution_mode == "streaming"
                else SDXL_RESIDENCY_CLASS_FULL
            )
        else:
            residency_class = normalize_residency_class(self.residency_class)
        if execution_mode is None:
            execution_mode = "streaming" if residency_class == SDXL_RESIDENCY_CLASS_UNIFIED_STREAMING else "resident"

        if self.clip_residency_mode is not None:
            prefer_clip_gpu = self.clip_residency_mode == CLIP_RESIDENCY_GPU_RESIDENT
        elif self.keep_clip_loaded is not None:
            prefer_clip_gpu = bool(self.keep_clip_loaded)

        if self.vae_encode_mode is not None:
            prefer_gpu_vae_encode = self.vae_encode_mode in (VAE_POSTURE_TRANSIENT_GPU, VAE_POSTURE_GPU_RESIDENT)

        object.__setattr__(self, "runtime_family", runtime_family)
        object.__setattr__(self, "execution_mode", execution_mode)
        object.__setattr__(self, "prefer_clip_gpu", prefer_clip_gpu)
        object.__setattr__(self, "prefer_gpu_vae_encode", prefer_gpu_vae_encode)
        object.__setattr__(self, "execution_family", EXECUTION_FAMILY_STANDARD)
        object.__setattr__(self, "residency_class", residency_class)
        object.__setattr__(self, "clip_residency_mode", self.clip_residency_mode or (CLIP_RESIDENCY_GPU_RESIDENT if prefer_clip_gpu else CLIP_RESIDENCY_CPU_ONLY))
        if self.vae_encode_mode is None or self.vae_encode_mode == VAE_POSTURE_GPU_RESIDENT:
            object.__setattr__(self, "vae_encode_mode", VAE_POSTURE_TRANSIENT_GPU if prefer_gpu_vae_encode else VAE_ENCODE_CPU_DEFAULT)
        if self.keep_clip_loaded is None:
            object.__setattr__(self, "keep_clip_loaded", prefer_clip_gpu)
        if self.execution_class is None or self.execution_class not in {
            ExecutionClass.SDXL_STREAMING_T1,
            ExecutionClass.SDXL_RESIDENT_T2,
        }:
            object.__setattr__(
                self,
                "execution_class",
                ExecutionClass.SDXL_STREAMING_T1 if execution_mode == "streaming" else ExecutionClass.SDXL_RESIDENT_T2,
            )

    def cache_domain(self) -> tuple[str | None, str | None, str | None]:
        return self.execution_family, self.residency_class, self.clip_residency_mode


def _sdxl_process_class(policy) -> str:
    from backend import process_transition

    return process_transition.PROCESS_CLASS_STANDARD_SDXL


def _sdxl_route_family(policy, base_model_name=None) -> str:
    return "sdxl"


def resolve_sdxl_process_key(
    *,
    base_model_name,
    vae_name=None,
    clip_name=None,
    sdxl_policy=None,
    loras=None,
):
    from backend import process_transition

    identity = [str(base_model_name or ""), str(clip_name or "")]
    if loras:
        identity.extend(str(lora) for lora in sorted(loras))

    return process_transition.build_process_key(
        family=process_transition.PROCESS_FAMILY_SDXL,
        process_class=_sdxl_process_class(sdxl_policy),
        authoritative_identity=tuple(identity),
        execution_family=getattr(sdxl_policy, "execution_family", None) if sdxl_policy is not None else None,
        residency_class=getattr(sdxl_policy, "residency_class", None) if sdxl_policy is not None else None,
        route_family="sdxl",
    )


def normalize_residency_class(residency_class: Any | None) -> str:
    normalized = str(residency_class or "").strip().lower()
    if normalized in {SDXL_RESIDENCY_CLASS_FULL, SDXL_RESIDENCY_CLASS_UNIFIED_STREAMING}:
        return normalized
    return SDXL_RESIDENCY_CLASS_FULL


def resolve_sdxl_execution_policy(
    *,
    architecture: str | None,
    base_model_name: Any | None,
    profile: Any | None = None,
    requested_residency_class: Any | None = None,
) -> SDXLExecutionPolicy:
    normalized_architecture = str(architecture or "").strip().lower() or None
    task_id = "flux_fill" if normalized_architecture == "flux" else "sdxl"

    if profile is not None:
        plan = PlacementSolver.solve(
            vram_total_mb=getattr(profile, "total_vram_mb", 8192.0),
            ram_total_mb=getattr(profile, "total_ram_mb", 16384.0),
            task_id=task_id,
        )
    else:
        plan = PlacementSolver.solve_from_system(task_id=task_id)

    unet_streaming = plan.execution_class == ExecutionClass.SDXL_STREAMING_T1 or plan.unet.mode in {
        ResidencyMode.CPU_RESIDENT,
        ResidencyMode.DISK_PAGED,
    }
    notes = [plan.tier.name, plan.execution_class.name, plan.unet.mode.name]
    stream_budget_mb = 0.0 if plan.unet.mode == ResidencyMode.GPU_RESIDENT else 256.0

    return SDXLExecutionPolicy(
        enabled=True,
        architecture=normalized_architecture,
        runtime_family="unified_sdxl",
        execution_mode="streaming" if unet_streaming else "resident",
        hardware_tier=plan.tier.name,
        allow_cpu_shadow=False,
        prefer_clip_gpu=(plan.clip.mode == ResidencyMode.GPU_RESIDENT),
        prefer_gpu_vae_encode=(plan.vae.mode == ResidencyMode.GPU_RESIDENT),
        stream_budget_mb=stream_budget_mb,
        notes=tuple(notes),
    )
