from __future__ import annotations

import enum
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Public enums and compatibility types
# ---------------------------------------------------------------------------


class ResidencyMode(enum.Enum):
    GPU_RESIDENT = "gpu_resident"
    CPU_RESIDENT = "cpu_resident"
    CPU_PINNED_STREAMING = "cpu_pinned_streaming"
    DISK_PAGED = "disk_paged"
    OFFLOADED = "offloaded"
    TRANSIENT_GPU = "transient_gpu"
    OPPORTUNISTIC_GPU = "opportunistic_gpu"
    CPU_ONLY = "cpu_only"


class HardwareTier(enum.Enum):
    LOW_VRAM = "LOW_VRAM"
    NORMAL_VRAM = "NORMAL_VRAM"
    HIGH_VRAM = "HIGH_VRAM"


class ExecutionClass(enum.Enum):
    SDXL_STREAMING_T1 = "SDXL_STREAMING_T1"
    SDXL_RESIDENT_T2 = "SDXL_RESIDENT_T2"
    SDXL_GPU_GREEDY_T3PLUS = "SDXL_GPU_GREEDY_T3PLUS"
    FLUX_STREAMING_T3 = "FLUX_STREAMING_T3"
    FLUX_RESIDENT_T4 = "FLUX_RESIDENT_T4"
    FLUX_RESIDENT_T5 = "FLUX_RESIDENT_T5"
    FLUX_RESIDENT_T6 = "FLUX_RESIDENT_T6"


STREAMING_EXECUTION_CLASSES = {
    ExecutionClass.SDXL_STREAMING_T1,
    ExecutionClass.FLUX_STREAMING_T3,
}

FLUX_RUNTIME_FAMILY_NATIVE_FP8 = "native_fp8"
FLUX_RUNTIME_POSTURE_STREAMING = "streaming"
FLUX_RUNTIME_POSTURE_RESIDENT = "resident"
FLUX_STREAMING_PROFILE_OPEN_C64_D1_S1 = "open_c64_d1_s1"
FLUX_STREAMING_PROFILE_OPEN_C128_D1_S1 = "open_c128_d1_s1"
FLUX_FILL_STREAMING_PROFILE_OPEN_C64_D1_S1 = FLUX_STREAMING_PROFILE_OPEN_C64_D1_S1
FLUX_FILL_STREAMING_PROFILE_OPEN_C128_D1_S1 = FLUX_STREAMING_PROFILE_OPEN_C128_D1_S1
FLUX_RESIDENT_LOAD_STANDARD = "resident_cpu_shadow"
FLUX_RESIDENT_LOAD_STICKY_NO_CPU_SHADOW = "sticky_no_cpu_shadow"
HIGH_VRAM_MIN_TOTAL_VRAM_MB = 14 * 1024.0
FLUX_RESIDENT_EXECUTION_CLASSES = {
    ExecutionClass.FLUX_RESIDENT_T4,
    ExecutionClass.FLUX_RESIDENT_T5,
    ExecutionClass.FLUX_RESIDENT_T6,
}
SDXL_RESIDENT_EXECUTION_CLASSES = {
    ExecutionClass.SDXL_RESIDENT_T2,
    ExecutionClass.SDXL_GPU_GREEDY_T3PLUS,
}


DEFAULT_PLATFORM_BASELINE_MB = {
    "windows": 650.0,
    "linux": 300.0,
}


# ---------------------------------------------------------------------------
# A Priori model universe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InferenceCostProfile:
    family: str
    variant: str
    weights_mb: float
    overhead_mb: float
    vae_mb: float
    clip_mb: float
    t5_mb: float = 0.0
    streaming_workspace_mb: float = 0.0
    lora_artifact_cpu_mb: float = 0.0
    lora_dense_cpu_mb: float = 0.0
    lora_transient_gpu_mb: float = 0.0
    conditioning_mb: float = 0.0


@dataclass(frozen=True)
class T5CostProfile:
    disk_paged_fp16_mb: float = 9334.0
    disk_paged_q8_mb: float = 2762.0


MODEL_UNIVERSE_COSTS: Dict[str, InferenceCostProfile] = {
    "sdxl_q8": InferenceCostProfile(
        family="sdxl",
        variant="sdxl_q8",
        weights_mb=2670.0,
        overhead_mb=1000.0,
        vae_mb=320.0,
        clip_mb=1135.0,
        streaming_workspace_mb=780.0,
        lora_artifact_cpu_mb=32.0,
        lora_dense_cpu_mb=24.0,
        lora_transient_gpu_mb=16.0,
    ),
    "sdxl_fp16": InferenceCostProfile(
        family="sdxl",
        variant="sdxl_fp16",
        weights_mb=5135.0,
        overhead_mb=900.0,
        vae_mb=320.0,
        clip_mb=1135.0,
        streaming_workspace_mb=720.0,
        lora_artifact_cpu_mb=32.0,
        lora_dense_cpu_mb=24.0,
        lora_transient_gpu_mb=16.0,
    ),
    "flux_fill_fp8": InferenceCostProfile(
        family="flux",
        variant="flux_fill_fp8",
        weights_mb=11351.0,
        overhead_mb=2800.0,
        vae_mb=640.0,
        clip_mb=235.0,
        t5_mb=4827.0,
        streaming_workspace_mb=1100.0,
        conditioning_mb=8.0,
    ),
}


# Legacy name retained for compatibility with existing imports.
COST_REGISTRY = MODEL_UNIVERSE_COSTS
SYSTEM_OVERHEAD = DEFAULT_PLATFORM_BASELINE_MB


@dataclass(frozen=True)
class HardwareProfile:
    total_vram_mb: float
    total_ram_mb: float
    platform: str
    system_reserved_mb: float

    @property
    def available_vram_mb(self) -> float:
        return max(self.total_vram_mb - self.system_reserved_mb, 0.0)


@dataclass(frozen=True)
class TaskRequest:
    family: str
    requested_variant: str | None = None
    lora_count: int = 0


@dataclass(frozen=True)
class ResolvedRequest:
    family: str
    requested_variant: str | None
    model_variant: str
    execution_class: ExecutionClass
    hardware_tier: HardwareTier
    profile: InferenceCostProfile
    t5_mode: str | None
    vram_total_mb: float
    total_ram_mb: float
    lora_count: int = 0


@dataclass(frozen=True)
class ComponentPlacement:
    component_id: str
    load_device: str
    compute_device: str
    offload_device: str
    residency_mode: str
    required_gpu_mb: float
    preferred_gpu_mb: float
    pinned_cpu_mb: float
    host_ram_mb: float
    transient_gpu_mb: float
    evict_before: tuple[str, ...]
    phase_scope: tuple[str, ...]
    family: str | None = None
    variant: str | None = None
    reusable: bool = False
    current_device: str | None = None
    current_residency_mode: str | None = None
    current_fingerprint: str | None = None

    @property
    def mode(self) -> ResidencyMode:
        if self.residency_mode == ResidencyMode.GPU_RESIDENT.value:
            return ResidencyMode.GPU_RESIDENT
        if self.residency_mode == ResidencyMode.DISK_PAGED.value or self.residency_mode.startswith("disk_paged"):
            return ResidencyMode.DISK_PAGED
        if self.residency_mode == ResidencyMode.OFFLOADED.value:
            return ResidencyMode.OFFLOADED
        if self.residency_mode in {
            ResidencyMode.TRANSIENT_GPU.value,
            ResidencyMode.OPPORTUNISTIC_GPU.value,
        }:
            return ResidencyMode.GPU_RESIDENT
        if self.residency_mode.startswith("cpu_"):
            return ResidencyMode.CPU_RESIDENT
        return ResidencyMode.CPU_RESIDENT

    @property
    def device(self) -> torch.device:
        if self.residency_mode in {
            ResidencyMode.GPU_RESIDENT.value,
            ResidencyMode.TRANSIENT_GPU.value,
            ResidencyMode.OPPORTUNISTIC_GPU.value,
        }:
            return torch.device("cuda")
        return torch.device("cpu")

    @property
    def budget_mb(self) -> float:
        return self.required_gpu_mb + self.preferred_gpu_mb + self.host_ram_mb + self.transient_gpu_mb

    def as_legacy_device_plan(self) -> "DevicePlan":
        return DevicePlan(
            device=self.device,
            mode=self.mode,
            budget_mb=self.budget_mb,
        )


@dataclass(frozen=True)
class PhasePlan:
    phase: str
    required_components: tuple[str, ...]
    preferred_gpu_components: tuple[str, ...]
    required_headroom_mb: float
    optional_headroom_mb: float
    evict_ledger_entries: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PlacementPlan:
    execution_class: ExecutionClass
    task_family: str
    model_variant: str
    t5_mode: str | None
    components: Dict[str, ComponentPlacement]
    phase_plans: Dict[str, PhasePlan]
    hardware_tier: HardwareTier
    overhead_mb: float
    ledger_gpu_mb: float
    ledger_pinned_cpu_mb: float
    available_gpu_mb: float
    available_ram_mb: float
    runtime_family: str | None = None
    runtime_posture: str | None = None
    streaming_profile: str | None = None
    resident_load_strategy: str | None = None
    fallback_model_variant: str | None = None
    reusable_components: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def tier(self) -> HardwareTier:
        return self.hardware_tier

    def component(self, component_id: str) -> ComponentPlacement | None:
        return self.components.get(component_id)

    @property
    def unet(self) -> "DevicePlan":
        return self.components.get("unet", _empty_device_plan())

    @property
    def clip(self) -> "DevicePlan":
        return self.components.get("clip", _empty_device_plan())

    @property
    def vae(self) -> "DevicePlan":
        return self.components.get("vae", _empty_device_plan())

    @property
    def t5(self) -> "DevicePlan | None":
        component = self.components.get("t5")
        if component is None:
            return None
        return component.as_legacy_device_plan()

    @property
    def conditioning(self) -> "DevicePlan | None":
        component = self.components.get("conditioning")
        if component is None:
            return None
        return component.as_legacy_device_plan()


@dataclass(frozen=True)
class DevicePlan:
    device: torch.device
    mode: ResidencyMode
    budget_mb: float


def _empty_device_plan() -> DevicePlan:
    return DevicePlan(device=torch.device("cpu"), mode=ResidencyMode.OFFLOADED, budget_mb=0.0)


@dataclass(frozen=True)
class LedgerEntry:
    identity: str
    family: str | None
    variant: str | None
    current_device: str
    residency_mode: str
    pinned_cpu_mb: float
    host_ram_mb: float
    gpu_mb: float
    fingerprint: str | None
    last_used_phase: str | None
    last_used_timestamp: float | None
    reusable: bool
    transient_gpu_mb: float = 0.0

    @property
    def name(self) -> str:
        return self.identity

    @property
    def device(self) -> torch.device:
        if self.current_device.startswith("cuda"):
            return torch.device("cuda")
        return torch.device("cpu")

    @property
    def size_mb(self) -> float:
        return self.gpu_mb + self.host_ram_mb + self.transient_gpu_mb

    @property
    def mode(self) -> ResidencyMode:
        if self.residency_mode == ResidencyMode.GPU_RESIDENT.value:
            return ResidencyMode.GPU_RESIDENT
        if self.residency_mode == ResidencyMode.DISK_PAGED.value or self.residency_mode.startswith("disk_paged"):
            return ResidencyMode.DISK_PAGED
        if self.residency_mode == ResidencyMode.OFFLOADED.value:
            return ResidencyMode.OFFLOADED
        if self.residency_mode in {
            ResidencyMode.TRANSIENT_GPU.value,
            ResidencyMode.OPPORTUNISTIC_GPU.value,
        }:
            return ResidencyMode.GPU_RESIDENT
        return ResidencyMode.CPU_RESIDENT


LoadedResource = LedgerEntry


# ---------------------------------------------------------------------------
# Resource ledger
# ---------------------------------------------------------------------------


class ResourceLedger:
    def __init__(self) -> None:
        self._resources: Dict[str, LedgerEntry] = {}

    @staticmethod
    def _normalize_device(device: Any | None) -> str:
        if device is None:
            return "cpu"
        if isinstance(device, torch.device):
            if device.index is None:
                return device.type
            return f"{device.type}:{device.index}"
        return str(device)

    @staticmethod
    def _default_gpu_mb(device_name: str, size_mb: float | None) -> float:
        if device_name.startswith("cuda"):
            return float(size_mb or 0.0)
        return 0.0

    @staticmethod
    def _default_cpu_mb(device_name: str, size_mb: float | None) -> float:
        if device_name.startswith("cuda"):
            return 0.0
        return float(size_mb or 0.0)

    def register_load(
        self,
        identity: str,
        device: Any | None = None,
        size_mb: float | None = None,
        fingerprint: str | None = None,
        mode: ResidencyMode | str | None = None,
        *,
        family: str | None = None,
        variant: str | None = None,
        current_device: Any | None = None,
        residency_mode: str | None = None,
        pinned_cpu_mb: float | None = None,
        host_ram_mb: float | None = None,
        gpu_mb: float | None = None,
        last_used_phase: str | None = None,
        last_used_timestamp: float | None = None,
        reusable: bool = True,
        transient_gpu_mb: float = 0.0,
    ) -> LedgerEntry:
        current_device_name = self._normalize_device(current_device if current_device is not None else device)
        residency_mode_name = (
            residency_mode
            if residency_mode is not None
            else (mode.value if isinstance(mode, ResidencyMode) else str(mode) if mode is not None else ResidencyMode.CPU_RESIDENT.value)
        )
        gpu_mb_value = float(gpu_mb if gpu_mb is not None else self._default_gpu_mb(current_device_name, size_mb))
        pinned_cpu_value = float(
            pinned_cpu_mb if pinned_cpu_mb is not None else self._default_cpu_mb(current_device_name, size_mb)
        )
        host_ram_value = float(host_ram_mb if host_ram_mb is not None else pinned_cpu_value)
        entry = LedgerEntry(
            identity=identity,
            family=family,
            variant=variant,
            current_device=current_device_name,
            residency_mode=residency_mode_name,
            pinned_cpu_mb=pinned_cpu_value,
            host_ram_mb=host_ram_value,
            gpu_mb=gpu_mb_value,
            fingerprint=fingerprint,
            last_used_phase=last_used_phase,
            last_used_timestamp=last_used_timestamp if last_used_timestamp is not None else time.time(),
            reusable=reusable,
            transient_gpu_mb=float(transient_gpu_mb),
        )
        self._resources[identity] = entry
        return entry

    def register_evict(self, identity: str) -> None:
        self._resources.pop(identity, None)

    def touch(self, identity: str, phase: str, timestamp: float | None = None) -> None:
        entry = self._resources.get(identity)
        if entry is None:
            return
        self._resources[identity] = LedgerEntry(
            identity=entry.identity,
            family=entry.family,
            variant=entry.variant,
            current_device=entry.current_device,
            residency_mode=entry.residency_mode,
            pinned_cpu_mb=entry.pinned_cpu_mb,
            host_ram_mb=entry.host_ram_mb,
            gpu_mb=entry.gpu_mb,
            fingerprint=entry.fingerprint,
            last_used_phase=phase,
            last_used_timestamp=timestamp if timestamp is not None else time.time(),
            reusable=entry.reusable,
            transient_gpu_mb=entry.transient_gpu_mb,
        )

    def get_state(self) -> Dict[str, LedgerEntry]:
        return dict(self._resources)

    def get_entry(self, identity: str) -> LedgerEntry | None:
        return self._resources.get(identity)

    def matches(self, component_id: str, family: str, variant: str, *, residency_mode: str | None = None) -> LedgerEntry | None:
        entry = self._resources.get(component_id)
        if entry is None or not entry.reusable:
            return None
        if entry.family != family or entry.variant != variant:
            return None
        if residency_mode is not None and entry.residency_mode != residency_mode:
            return None
        return entry

    def get_total_vram_used(self) -> float:
        return sum(entry.gpu_mb for entry in self._resources.values())

    def get_total_pinned_cpu_mb(self) -> float:
        return sum(entry.pinned_cpu_mb for entry in self._resources.values())

    def get_total_host_ram_mb(self) -> float:
        return sum(entry.host_ram_mb for entry in self._resources.values())

    def available_gpu_mb(self, total_vram_mb: float, system_reserved_mb: float = 0.0) -> float:
        return max(total_vram_mb - system_reserved_mb - self.get_total_vram_used(), 0.0)

    def available_ram_mb(self, total_ram_mb: float) -> float:
        return max(total_ram_mb - self.get_total_host_ram_mb(), 0.0)

    def plan_gpu_evictions(
        self,
        *,
        required_free_gpu_mb: float,
        total_vram_mb: float,
        system_reserved_mb: float = 0.0,
        protected_identities: Iterable[str] = (),
    ) -> tuple[str, ...]:
        shortage_mb = max(required_free_gpu_mb - self.available_gpu_mb(total_vram_mb, system_reserved_mb), 0.0)
        if shortage_mb <= 0.0:
            return ()
        protected = set(protected_identities)
        candidates = sorted(
            (
                entry
                for identity, entry in self._resources.items()
                if identity not in protected and entry.gpu_mb > 0.0
            ),
            key=lambda entry: (entry.last_used_timestamp is None, entry.last_used_timestamp or 0.0),
        )
        freed_mb = 0.0
        evictions: list[str] = []
        for entry in candidates:
            evictions.append(entry.identity)
            freed_mb += entry.gpu_mb
            if freed_mb >= shortage_mb:
                break
        return tuple(evictions)


# ---------------------------------------------------------------------------
# Registry and solver
# ---------------------------------------------------------------------------


class ModelUniverseRegistry:
    def __init__(self) -> None:
        self._costs = dict(MODEL_UNIVERSE_COSTS)
        self._t5_costs = T5CostProfile()

    @property
    def costs(self) -> Mapping[str, InferenceCostProfile]:
        return self._costs

    @property
    def t5_costs(self) -> T5CostProfile:
        return self._t5_costs

    def parse_task_id(self, task_id: str) -> tuple[str, str | None]:
        normalized = str(task_id or "").strip().lower()
        if normalized in self._costs:
            return self._costs[normalized].family, normalized
        if normalized.startswith("sdxl"):
            return "sdxl", None
        if normalized.startswith("flux"):
            return "flux", None
        if normalized.startswith("t5"):
            return "t5", None
        raise ValueError(f"Unknown task family or variant: {task_id!r}")

    def profile_for_variant(self, variant: str) -> InferenceCostProfile:
        try:
            return self._costs[variant]
        except KeyError as exc:
            raise ValueError(f"Unknown model variant: {variant!r}") from exc

    def default_variant_for(self, family: str, execution_class: ExecutionClass) -> str:
        family = family.lower()
        if family == "sdxl":
            return "sdxl_fp16"
        if family == "flux":
            return "flux_fill_fp8"
        raise ValueError(f"No default variant for family {family!r}")

    def canonical_variant_for_request(
        self,
        family: str,
        requested_variant: str | None,
        execution_class: ExecutionClass,
    ) -> str:
        # Execution class is the authority; concrete legacy task ids are treated
        # as hints, not policy overrides.
        return self.default_variant_for(family, execution_class)


class ExecutionClassSolver:
    def __init__(self, registry: ModelUniverseRegistry | None = None) -> None:
        self.registry = registry or ModelUniverseRegistry()

    @staticmethod
    def hardware_tier_for_vram(vram_total_mb: float, total_ram_mb: float = 16384.0) -> HardwareTier:
        if vram_total_mb <= 6144.0:
            return HardwareTier.LOW_VRAM
        if vram_total_mb < HIGH_VRAM_MIN_TOTAL_VRAM_MB:
            return HardwareTier.NORMAL_VRAM
        return HardwareTier.HIGH_VRAM

    def resolve_execution_class(self, family: str, vram_total_mb: float, total_ram_mb: float = 16384.0) -> tuple[ExecutionClass, HardwareTier]:
        family = family.lower()
        tier = self.hardware_tier_for_vram(vram_total_mb, total_ram_mb)
        if family == "sdxl":
            if vram_total_mb < 8192.0:
                return ExecutionClass.SDXL_STREAMING_T1, tier
            return ExecutionClass.SDXL_RESIDENT_T2, tier
        if family == "flux":
            if tier in (HardwareTier.LOW_VRAM, HardwareTier.NORMAL_VRAM):
                return ExecutionClass.FLUX_STREAMING_T3, tier
            return ExecutionClass.FLUX_RESIDENT_T6, tier
        raise ValueError(f"Unsupported family for execution-class resolution: {family!r}")

    def resolve_t5_mode(
        self,
        total_ram_mb: float,
        execution_class: ExecutionClass,
        ledger: ResourceLedger | None = None,
    ) -> str:
        del total_ram_mb, execution_class, ledger
        return "disk_paged_fp16"

    def resolve_request(
        self,
        *,
        task_id: str,
        vram_total_mb: float,
        total_ram_mb: float,
        ledger: ResourceLedger | None = None,
        lora_count: int = 0,
    ) -> ResolvedRequest:
        family, requested_variant = self.registry.parse_task_id(task_id)
        execution_class, hardware_tier = self.resolve_execution_class(family, vram_total_mb, total_ram_mb)
        resolved_variant = self.registry.canonical_variant_for_request(family, requested_variant, execution_class)
        profile = self.registry.profile_for_variant(resolved_variant)
        t5_mode = None
        if family == "flux":
            t5_mode = self.resolve_t5_mode(total_ram_mb, execution_class, ledger)
        return ResolvedRequest(
            family=family,
            requested_variant=requested_variant,
            model_variant=resolved_variant,
            execution_class=execution_class,
            hardware_tier=hardware_tier,
            profile=profile,
            t5_mode=t5_mode,
            vram_total_mb=float(vram_total_mb),
            total_ram_mb=float(total_ram_mb),
            lora_count=lora_count,
        )


class PlacementPlanner:
    def __init__(
        self,
        registry: ModelUniverseRegistry | None = None,
        solver: ExecutionClassSolver | None = None,
    ) -> None:
        self.registry = registry or ModelUniverseRegistry()
        self.solver = solver or ExecutionClassSolver(self.registry)

    @staticmethod
    def _platform_baseline(platform: str) -> float:
        normalized = platform.lower()
        if normalized in {"nt", "windows", "win32"}:
            return DEFAULT_PLATFORM_BASELINE_MB["windows"]
        if normalized in {"posix", "linux"}:
            return DEFAULT_PLATFORM_BASELINE_MB["linux"]
        return DEFAULT_PLATFORM_BASELINE_MB["linux"]

    @staticmethod
    def _map_mode_to_legacy(component: ComponentPlacement) -> ResidencyMode:
        if component.residency_mode == ResidencyMode.GPU_RESIDENT.value:
            return ResidencyMode.GPU_RESIDENT
        if component.residency_mode == ResidencyMode.DISK_PAGED.value:
            return ResidencyMode.DISK_PAGED
        if component.residency_mode == ResidencyMode.OFFLOADED.value:
            return ResidencyMode.OFFLOADED
        if component.residency_mode == ResidencyMode.TRANSIENT_GPU.value:
            return ResidencyMode.GPU_RESIDENT
        if component.residency_mode == ResidencyMode.OPPORTUNISTIC_GPU.value:
            return ResidencyMode.GPU_RESIDENT
        return ResidencyMode.CPU_RESIDENT

    @staticmethod
    def _map_device_name(component: ComponentPlacement) -> torch.device:
        if component.compute_device.startswith("cuda") or component.load_device.startswith("cuda"):
            return torch.device("cuda")
        return torch.device("cpu")

    def _ledger_hit(
        self,
        ledger: ResourceLedger,
        component_id: str,
        family: str,
        variant: str,
        residency_mode: str | None = None,
    ) -> LedgerEntry | None:
        return ledger.matches(component_id, family, variant, residency_mode=residency_mode)

    def _build_component(
        self,
        *,
        component_id: str,
        family: str,
        variant: str,
        residency_mode: str,
        load_device: str,
        compute_device: str,
        offload_device: str,
        required_gpu_mb: float,
        preferred_gpu_mb: float,
        pinned_cpu_mb: float,
        host_ram_mb: float,
        transient_gpu_mb: float,
        evict_before: tuple[str, ...],
        phase_scope: tuple[str, ...],
        ledger: ResourceLedger,
    ) -> ComponentPlacement:
        hit = self._ledger_hit(ledger, component_id, family, variant, residency_mode=residency_mode)
        if hit is not None:
            load_device = hit.current_device
            compute_device = "cuda" if hit.current_device.startswith("cuda") else "cpu"
            offload_device = hit.current_device
            return ComponentPlacement(
                component_id=component_id,
                load_device=load_device,
                compute_device=compute_device,
                offload_device=offload_device,
                residency_mode=residency_mode,
                required_gpu_mb=required_gpu_mb,
                preferred_gpu_mb=preferred_gpu_mb,
                pinned_cpu_mb=pinned_cpu_mb,
                host_ram_mb=host_ram_mb,
                transient_gpu_mb=transient_gpu_mb,
                evict_before=evict_before,
                phase_scope=phase_scope,
                family=family,
                variant=variant,
                reusable=True,
                current_device=hit.current_device,
                current_residency_mode=hit.residency_mode,
                current_fingerprint=hit.fingerprint,
            )
        return ComponentPlacement(
            component_id=component_id,
            load_device=load_device,
            compute_device=compute_device,
            offload_device=offload_device,
            residency_mode=residency_mode,
            required_gpu_mb=required_gpu_mb,
            preferred_gpu_mb=preferred_gpu_mb,
            pinned_cpu_mb=pinned_cpu_mb,
            host_ram_mb=host_ram_mb,
            transient_gpu_mb=transient_gpu_mb,
            evict_before=evict_before,
            phase_scope=phase_scope,
            family=family,
            variant=variant,
            reusable=False,
        )

    def _resolve_flux_runtime_contract(
        self,
        *,
        request: ResolvedRequest,
    ) -> dict[str, str | None]:
        if request.family != "flux":
            return {
                "runtime_family": None,
                "runtime_posture": None,
                "streaming_profile": None,
                "resident_load_strategy": None,
                "fallback_model_variant": None,
            }

        runtime_family = FLUX_RUNTIME_FAMILY_NATIVE_FP8
        fallback_model_variant = None

        if request.execution_class in FLUX_RESIDENT_EXECUTION_CLASSES:
            return {
                "runtime_family": runtime_family,
                "runtime_posture": FLUX_RUNTIME_POSTURE_RESIDENT,
                "streaming_profile": None,
                "resident_load_strategy": FLUX_RESIDENT_LOAD_STICKY_NO_CPU_SHADOW,
                "fallback_model_variant": fallback_model_variant,
            }

        # Otherwise, streaming posture
        if float(request.vram_total_mb) <= 8192.0:
            streaming_profile = FLUX_STREAMING_PROFILE_OPEN_C64_D1_S1
        else:
            streaming_profile = FLUX_STREAMING_PROFILE_OPEN_C128_D1_S1

        return {
            "runtime_family": runtime_family,
            "runtime_posture": FLUX_RUNTIME_POSTURE_STREAMING,
            "streaming_profile": streaming_profile,
            "resident_load_strategy": None,
            "fallback_model_variant": fallback_model_variant,
        }

    @staticmethod
    def _uses_streaming_posture(
        *,
        request: ResolvedRequest,
        runtime_posture: str | None,
    ) -> bool:
        if request.execution_class in STREAMING_EXECUTION_CLASSES:
            return True
        return request.family == "flux" and runtime_posture == FLUX_RUNTIME_POSTURE_STREAMING

    def _prefer_vae_resident(
        self,
        *,
        request: ResolvedRequest,
        runtime_contract: dict[str, str | None],
        greedy: bool,
    ) -> bool:
        if request.family != "flux":
            # SDXL VAE residency was retired after repeated warm-run lifecycle
            # regressions. Keep it opportunistic for every SDXL execution class.
            return False
        if greedy:
            return True

        runtime_posture = str(runtime_contract.get("runtime_posture") or "").strip().lower()
        if runtime_posture == FLUX_RUNTIME_POSTURE_RESIDENT:
            return True

        streaming_profile = str(runtime_contract.get("streaming_profile") or "").strip().lower()
        vram_total_mb = float(request.vram_total_mb)
        if streaming_profile == FLUX_STREAMING_PROFILE_OPEN_C64_D1_S1:
            return vram_total_mb >= 8192.0
        if streaming_profile == FLUX_STREAMING_PROFILE_OPEN_C128_D1_S1:
            return vram_total_mb >= 10240.0
        return False

    def _streaming_unet(
        self,
        *,
        family: str,
        variant: str,
        profile: InferenceCostProfile,
        ledger: ResourceLedger,
    ) -> ComponentPlacement:
        return self._build_component(
            component_id="unet",
            family=family,
            variant=variant,
            residency_mode=ResidencyMode.CPU_PINNED_STREAMING.value,
            load_device="pinned_cpu",
            compute_device="cuda",
            offload_device="pinned_cpu",
            required_gpu_mb=profile.streaming_workspace_mb,
            preferred_gpu_mb=0.0,
            pinned_cpu_mb=profile.weights_mb,
            host_ram_mb=profile.weights_mb,
            transient_gpu_mb=profile.streaming_workspace_mb,
            evict_before=("decode",),
            phase_scope=("prompt_encode", "diffusion", "finalize"),
            ledger=ledger,
        )

    def _resident_unet(
        self,
        *,
        family: str,
        variant: str,
        profile: InferenceCostProfile,
        ledger: ResourceLedger,
    ) -> ComponentPlacement:
        return self._build_component(
            component_id="unet",
            family=family,
            variant=variant,
            residency_mode=ResidencyMode.GPU_RESIDENT.value,
            load_device="cuda",
            compute_device="cuda",
            offload_device="cuda",
            required_gpu_mb=profile.weights_mb + profile.overhead_mb,
            preferred_gpu_mb=profile.weights_mb,
            pinned_cpu_mb=0.0,
            host_ram_mb=0.0,
            transient_gpu_mb=profile.overhead_mb,
            evict_before=(),
            phase_scope=("prompt_encode", "diffusion", "finalize"),
            ledger=ledger,
        )

    def _sdxl_clip(
        self,
        *,
        family: str,
        variant: str,
        profile: InferenceCostProfile,
        greedy: bool,
        ledger: ResourceLedger,
        available_gpu_mb: float,
    ) -> ComponentPlacement:
        residency_mode = ResidencyMode.CPU_ONLY.value
        load_device = "cpu"
        compute_device = "cpu"
        preferred_gpu_mb = 0.0
        required_gpu_mb = 0.0
        pinned_cpu_mb = profile.clip_mb
        host_ram_mb = profile.clip_mb
        offload_device = "cpu"
        return self._build_component(
            component_id="clip",
            family=family,
            variant=variant,
            residency_mode=residency_mode,
            load_device=load_device,
            compute_device=compute_device,
            offload_device=offload_device,
            required_gpu_mb=required_gpu_mb,
            preferred_gpu_mb=preferred_gpu_mb,
            pinned_cpu_mb=pinned_cpu_mb,
            host_ram_mb=host_ram_mb,
            transient_gpu_mb=0.0,
            evict_before=(),
            phase_scope=("prompt_encode",),
            ledger=ledger,
        )

    def _flux_clip(
        self,
        *,
        family: str,
        variant: str,
        profile: InferenceCostProfile,
        greedy: bool,
        ledger: ResourceLedger,
        available_gpu_mb: float,
    ) -> ComponentPlacement:
        if greedy and available_gpu_mb >= profile.clip_mb:
            residency_mode = ResidencyMode.OPPORTUNISTIC_GPU.value
            load_device = "cuda"
            compute_device = "cuda"
            preferred_gpu_mb = profile.clip_mb
            required_gpu_mb = 120.0
            pinned_cpu_mb = 0.0
            host_ram_mb = 0.0
            offload_device = "cuda"
        else:
            residency_mode = ResidencyMode.CPU_ONLY.value
            load_device = "cpu"
            compute_device = "cpu"
            preferred_gpu_mb = 0.0
            required_gpu_mb = 0.0
            pinned_cpu_mb = profile.clip_mb
            host_ram_mb = profile.clip_mb
            offload_device = "cpu"
        return self._build_component(
            component_id="clip",
            family=family,
            variant=variant,
            residency_mode=residency_mode,
            load_device=load_device,
            compute_device=compute_device,
            offload_device=offload_device,
            required_gpu_mb=required_gpu_mb,
            preferred_gpu_mb=preferred_gpu_mb,
            pinned_cpu_mb=pinned_cpu_mb,
            host_ram_mb=host_ram_mb,
            transient_gpu_mb=120.0 if greedy else 0.0,
            evict_before=("diffusion",) if greedy else (),
            phase_scope=("prompt_encode",),
            ledger=ledger,
        )

    def _vae(
        self,
        *,
        family: str,
        variant: str,
        profile: InferenceCostProfile,
        execution_class: ExecutionClass,
        greedy: bool,
        prefer_resident: bool | None = None,
        ledger: ResourceLedger,
        available_gpu_mb: float,
    ) -> ComponentPlacement:
        if prefer_resident is None:
            resident_preferred = greedy or execution_class in {
                ExecutionClass.SDXL_RESIDENT_T2,
                ExecutionClass.SDXL_GPU_GREEDY_T3PLUS,
                ExecutionClass.FLUX_RESIDENT_T4,
                ExecutionClass.FLUX_RESIDENT_T5,
                ExecutionClass.FLUX_RESIDENT_T6,
            }
        else:
            resident_preferred = bool(prefer_resident)
        if resident_preferred and available_gpu_mb >= profile.vae_mb:
            residency_mode = ResidencyMode.GPU_RESIDENT.value
            load_device = "cuda"
            compute_device = "cuda"
            offload_device = "cuda"
            required_gpu_mb = profile.vae_mb
            preferred_gpu_mb = profile.vae_mb
            pinned_cpu_mb = 0.0
            host_ram_mb = 0.0
            transient_gpu_mb = profile.vae_mb
        else:
            residency_mode = ResidencyMode.TRANSIENT_GPU.value
            load_device = "cpu"
            compute_device = "cuda"
            offload_device = "cpu"
            required_gpu_mb = profile.vae_mb
            preferred_gpu_mb = 0.0
            pinned_cpu_mb = profile.vae_mb
            host_ram_mb = profile.vae_mb
            transient_gpu_mb = profile.vae_mb
        return self._build_component(
            component_id="vae",
            family=family,
            variant=variant,
            residency_mode=residency_mode,
            load_device=load_device,
            compute_device=compute_device,
            offload_device=offload_device,
            required_gpu_mb=required_gpu_mb,
            preferred_gpu_mb=preferred_gpu_mb,
            pinned_cpu_mb=pinned_cpu_mb,
            host_ram_mb=host_ram_mb,
            transient_gpu_mb=transient_gpu_mb,
            evict_before=("finalize",),
            phase_scope=("decode", "finalize"),
            ledger=ledger,
        )

    def _t5(
        self,
        *,
        family: str,
        variant: str,
        t5_mode: str,
        ledger: ResourceLedger,
        available_ram_mb: float,
    ) -> ComponentPlacement:
        costs = self.registry.t5_costs
        if t5_mode == "disk_paged_fp16":
            pinned_cpu_mb = 0.0
            host_ram_mb = costs.disk_paged_fp16_mb
            residency_mode = t5_mode
            load_device = "disk"
            offload_device = "disk"
        else:
            pinned_cpu_mb = 0.0
            host_ram_mb = costs.disk_paged_q8_mb
            residency_mode = t5_mode
            load_device = "disk"
            offload_device = "disk"
        return self._build_component(
            component_id="t5",
            family=family,
            variant=variant,
            residency_mode=residency_mode,
            load_device=load_device,
            compute_device="cpu",
            offload_device=offload_device,
            required_gpu_mb=0.0,
            preferred_gpu_mb=0.0,
            pinned_cpu_mb=pinned_cpu_mb if t5_mode != "disk_paged_q8" and available_ram_mb > 0.0 else pinned_cpu_mb,
            host_ram_mb=host_ram_mb,
            transient_gpu_mb=0.0,
            evict_before=(),
            phase_scope=("prompt_encode",),
            ledger=ledger,
        )

    def _lora(
        self,
        *,
        family: str,
        variant: str,
        profile: InferenceCostProfile,
        lora_count: int,
        ledger: ResourceLedger,
        streaming: bool,
    ) -> ComponentPlacement | None:
        if lora_count <= 0:
            return None
        pinned_cpu_mb = float(lora_count) * (profile.lora_artifact_cpu_mb + profile.lora_dense_cpu_mb)
        transient_gpu_mb = float(lora_count) * profile.lora_transient_gpu_mb
        residency_mode = ResidencyMode.CPU_PINNED_STREAMING.value if streaming else ResidencyMode.CPU_RESIDENT.value
        return self._build_component(
            component_id="lora",
            family=family,
            variant=variant,
            residency_mode=residency_mode,
            load_device="pinned_cpu",
            compute_device="cuda" if streaming else "cpu",
            offload_device="pinned_cpu",
            required_gpu_mb=transient_gpu_mb,
            preferred_gpu_mb=0.0,
            pinned_cpu_mb=pinned_cpu_mb,
            host_ram_mb=pinned_cpu_mb,
            transient_gpu_mb=transient_gpu_mb,
            evict_before=("finalize",),
            phase_scope=("prompt_encode", "diffusion"),
            ledger=ledger,
        )

    def _phase_plans(
        self,
        *,
        request: ResolvedRequest,
        components: Mapping[str, ComponentPlacement],
        streaming: bool,
    ) -> Dict[str, PhasePlan]:
        greedy = request.execution_class == ExecutionClass.SDXL_GPU_GREEDY_T3PLUS
        unet = components.get("unet")
        clip = components.get("clip")
        vae = components.get("vae")
        t5 = components.get("t5")
        lora = components.get("lora")

        def needed_gpu(component: ComponentPlacement | None) -> float:
            if component is None or component.reusable:
                return 0.0
            return component.required_gpu_mb

        def preferred_gpu(component: ComponentPlacement | None) -> float:
            if component is None or component.reusable:
                return 0.0
            return component.preferred_gpu_mb

        prompt_required = ("clip",)
        prompt_preferred: tuple[str, ...] = ()
        prompt_required_headroom = needed_gpu(clip)
        prompt_optional_headroom = preferred_gpu(clip)
        if request.family == "flux" and t5 is not None:
            prompt_required = ("clip", "t5")
            prompt_required_headroom = max(prompt_required_headroom, 0.0)
        if greedy and clip is not None:
            prompt_preferred = ("clip",)
            prompt_optional_headroom = preferred_gpu(clip) + preferred_gpu(t5)

        diffusion_required = ("unet",)
        diffusion_preferred: tuple[str, ...] = ()
        diffusion_required_headroom = needed_gpu(unet)
        diffusion_optional_headroom = preferred_gpu(unet)
        if greedy:
            diffusion_preferred = tuple(
                name for name in ("clip", "vae", "lora") if components.get(name) is not None
            )
            diffusion_optional_headroom += sum(
                preferred_gpu(component) for component in (clip, vae, lora) if component is not None
            )
        if streaming and lora is not None:
            diffusion_required_headroom += needed_gpu(lora)

        decode_required = ("vae",)
        decode_preferred: tuple[str, ...] = ("vae",) if vae is not None and vae.preferred_gpu_mb > 0 else ()
        decode_required_headroom = needed_gpu(vae)
        decode_optional_headroom = preferred_gpu(vae)

        return {
            "startup": PhasePlan(
                phase="startup",
                required_components=(),
                preferred_gpu_components=(),
                required_headroom_mb=0.0,
                optional_headroom_mb=0.0,
            ),
            "model_refresh": PhasePlan(
                phase="model_refresh",
                required_components=tuple(
                    name for name in ("unet", "clip", "vae", "t5", "lora") if components.get(name) is not None
                ),
                preferred_gpu_components=tuple(
                    name
                    for name in ("unet", "clip", "vae")
                    if components.get(name) is not None and components[name].preferred_gpu_mb > 0.0
                ),
                required_headroom_mb=sum(needed_gpu(component) for component in components.values()),
                optional_headroom_mb=sum(preferred_gpu(component) for component in components.values()),
            ),
            "prompt_encode": PhasePlan(
                phase="prompt_encode",
                required_components=prompt_required,
                preferred_gpu_components=prompt_preferred,
                required_headroom_mb=prompt_required_headroom,
                optional_headroom_mb=prompt_optional_headroom,
            ),
            "diffusion": PhasePlan(
                phase="diffusion",
                required_components=diffusion_required,
                preferred_gpu_components=diffusion_preferred,
                required_headroom_mb=diffusion_required_headroom,
                optional_headroom_mb=diffusion_optional_headroom,
            ),
            "decode": PhasePlan(
                phase="decode",
                required_components=decode_required,
                preferred_gpu_components=decode_preferred,
                required_headroom_mb=decode_required_headroom,
                optional_headroom_mb=decode_optional_headroom,
            ),
            "finalize": PhasePlan(
                phase="finalize",
                required_components=(),
                preferred_gpu_components=(),
                required_headroom_mb=0.0,
                optional_headroom_mb=0.0,
            ),
        }

    def _startup_plan(self, request: ResolvedRequest, hardware: HardwareProfile, ledger: ResourceLedger) -> PlacementPlan:
        runtime_contract = self._resolve_flux_runtime_contract(request=request)
        components = {
            "unet": ComponentPlacement(
                component_id="unet",
                load_device="cpu",
                compute_device="cpu",
                offload_device="cpu",
                residency_mode=ResidencyMode.OFFLOADED.value,
                required_gpu_mb=0.0,
                preferred_gpu_mb=0.0,
                pinned_cpu_mb=0.0,
                host_ram_mb=0.0,
                transient_gpu_mb=0.0,
                evict_before=(),
                phase_scope=("startup",),
                family=request.family,
                variant=request.model_variant,
            ),
            "clip": ComponentPlacement(
                component_id="clip",
                load_device="cpu",
                compute_device="cpu",
                offload_device="cpu",
                residency_mode=ResidencyMode.OFFLOADED.value,
                required_gpu_mb=0.0,
                preferred_gpu_mb=0.0,
                pinned_cpu_mb=0.0,
                host_ram_mb=0.0,
                transient_gpu_mb=0.0,
                evict_before=(),
                phase_scope=("startup",),
                family=request.family,
                variant=request.model_variant,
            ),
            "vae": ComponentPlacement(
                component_id="vae",
                load_device="cpu",
                compute_device="cpu",
                offload_device="cpu",
                residency_mode=ResidencyMode.OFFLOADED.value,
                required_gpu_mb=0.0,
                preferred_gpu_mb=0.0,
                pinned_cpu_mb=0.0,
                host_ram_mb=0.0,
                transient_gpu_mb=0.0,
                evict_before=(),
                phase_scope=("startup",),
                family=request.family,
                variant=request.model_variant,
            ),
        }
        phase_plans = self._phase_plans(
            request=request,
            components=components,
            streaming=self._uses_streaming_posture(
                request=request,
                runtime_posture=runtime_contract["runtime_posture"],
            ),
        )
        return PlacementPlan(
            execution_class=request.execution_class,
            task_family=request.family,
            model_variant=request.model_variant,
            t5_mode=request.t5_mode,
            components=components,
            phase_plans=phase_plans,
            hardware_tier=request.hardware_tier,
            overhead_mb=0.0,
            ledger_gpu_mb=ledger.get_total_vram_used(),
            ledger_pinned_cpu_mb=ledger.get_total_pinned_cpu_mb(),
            available_gpu_mb=hardware.available_vram_mb,
            available_ram_mb=hardware.total_ram_mb,
            runtime_family=runtime_contract["runtime_family"],
            runtime_posture=runtime_contract["runtime_posture"],
            streaming_profile=runtime_contract["streaming_profile"],
            resident_load_strategy=runtime_contract["resident_load_strategy"],
            fallback_model_variant=runtime_contract["fallback_model_variant"],
            reusable_components=(),
            notes=("startup",),
        )

    def plan(
        self,
        *,
        hardware: HardwareProfile,
        task_request: TaskRequest,
        current_ledger: ResourceLedger | None = None,
        is_startup: bool = False,
    ) -> PlacementPlan:
        ledger = current_ledger or ResourceLedger()
        request = self.solver.resolve_request(
            task_id=task_request.family if task_request.requested_variant is None else task_request.requested_variant,
            vram_total_mb=hardware.total_vram_mb,
            total_ram_mb=hardware.total_ram_mb,
            ledger=ledger,
            lora_count=task_request.lora_count,
        )
        if is_startup:
            return self._startup_plan(request, hardware, ledger)

        available_gpu_mb = ledger.available_gpu_mb(hardware.total_vram_mb, hardware.system_reserved_mb)
        available_ram_mb = ledger.available_ram_mb(hardware.total_ram_mb)
        planning_gpu_mb = hardware.available_vram_mb
        profile = request.profile
        runtime_contract = self._resolve_flux_runtime_contract(request=request)
        streaming = self._uses_streaming_posture(
            request=request,
            runtime_posture=runtime_contract["runtime_posture"],
        )
        greedy = request.execution_class == ExecutionClass.SDXL_GPU_GREEDY_T3PLUS

        components: Dict[str, ComponentPlacement] = {}
        if streaming:
            components["unet"] = self._streaming_unet(
                family=request.family,
                variant=request.model_variant,
                profile=profile,
                ledger=ledger,
            )
        else:
            components["unet"] = self._resident_unet(
                family=request.family,
                variant=request.model_variant,
                profile=profile,
                ledger=ledger,
            )

        if request.family == "sdxl":
            components["clip"] = self._sdxl_clip(
                family=request.family,
                variant=request.model_variant,
                profile=profile,
                greedy=greedy,
                ledger=ledger,
                available_gpu_mb=planning_gpu_mb,
            )
        elif request.family == "flux":
            components["clip"] = self._flux_clip(
                family=request.family,
                variant=request.model_variant,
                profile=profile,
                greedy=greedy or request.execution_class in {
                    ExecutionClass.FLUX_RESIDENT_T6,
                },
                ledger=ledger,
                available_gpu_mb=planning_gpu_mb,
            )
            components["conditioning"] = ComponentPlacement(
                component_id="conditioning",
                load_device="cpu",
                compute_device="cpu",
                offload_device="cpu",
                residency_mode=ResidencyMode.CPU_ONLY.value,
                required_gpu_mb=0.0,
                preferred_gpu_mb=0.0,
                pinned_cpu_mb=profile.conditioning_mb,
                host_ram_mb=profile.conditioning_mb,
                transient_gpu_mb=0.0,
                evict_before=(),
                phase_scope=("prompt_encode",),
                family=request.family,
                variant=request.model_variant,
                reusable=False,
            )

        vae_component = self._vae(
            family=request.family,
            variant=request.model_variant,
            profile=profile,
            execution_class=request.execution_class,
            greedy=greedy or request.execution_class in {ExecutionClass.FLUX_RESIDENT_T6},
            prefer_resident=self._prefer_vae_resident(
                request=request,
                runtime_contract=runtime_contract,
                greedy=greedy or request.execution_class in {ExecutionClass.FLUX_RESIDENT_T6},
            ),
            ledger=ledger,
            available_gpu_mb=planning_gpu_mb,
        )
        components["vae"] = vae_component

        if request.family == "flux":
            components["t5"] = self._t5(
                family=request.family,
                variant=request.model_variant,
                t5_mode=request.t5_mode or "disk_paged_fp16",
                ledger=ledger,
                available_ram_mb=available_ram_mb,
            )

        if request.lora_count > 0:
            lora_component = self._lora(
                family=request.family,
                variant=request.model_variant,
                profile=profile,
                lora_count=request.lora_count,
                ledger=ledger,
                streaming=streaming,
            )
            if lora_component is not None:
                components["lora"] = lora_component

        phase_plans = self._phase_plans(
            request=request,
            components=components,
            streaming=streaming,
        )
        reusable_components = tuple(
            sorted(
                component_id
                for component_id, component in components.items()
                if component.reusable
            )
        )
        model_refresh = phase_plans["model_refresh"]
        model_refresh_evictions = ledger.plan_gpu_evictions(
            required_free_gpu_mb=model_refresh.required_headroom_mb,
            total_vram_mb=hardware.total_vram_mb,
            system_reserved_mb=hardware.system_reserved_mb,
            protected_identities=reusable_components,
        )
        phase_plans = dict(phase_plans)
        phase_plans["model_refresh"] = PhasePlan(
            phase=model_refresh.phase,
            required_components=model_refresh.required_components,
            preferred_gpu_components=model_refresh.preferred_gpu_components,
            required_headroom_mb=model_refresh.required_headroom_mb,
            optional_headroom_mb=model_refresh.optional_headroom_mb,
            evict_ledger_entries=model_refresh_evictions,
        )
        total_required_gpu_mb = sum(component.required_gpu_mb for component in components.values())
        total_preferred_gpu_mb = sum(component.preferred_gpu_mb for component in components.values())
        notes = [
            f"platform={hardware.platform}",
            f"required_gpu_mb={total_required_gpu_mb:.1f}",
            f"preferred_gpu_mb={total_preferred_gpu_mb:.1f}",
        ]
        if runtime_contract["runtime_family"] is not None:
            notes.append(f"runtime_family={runtime_contract['runtime_family']}")
        if runtime_contract["runtime_posture"] is not None:
            notes.append(f"runtime_posture={runtime_contract['runtime_posture']}")
        if runtime_contract["streaming_profile"] is not None:
            notes.append(f"streaming_profile={runtime_contract['streaming_profile']}")
        if runtime_contract["resident_load_strategy"] is not None:
            notes.append(f"resident_load_strategy={runtime_contract['resident_load_strategy']}")
        if runtime_contract["fallback_model_variant"] is not None:
            notes.append(f"fallback_model_variant={runtime_contract['fallback_model_variant']}")
        if request.family == "flux" and streaming:
            notes.append("overlap_status=cpu_bound_or_overlap_unproven")
        if request.requested_variant is not None and request.requested_variant != request.model_variant:
            notes.append(
                f"variant_overridden={request.requested_variant}->{request.model_variant}"
            )
        if model_refresh_evictions:
            notes.append(f"model_refresh_evictions={','.join(model_refresh_evictions)}")
        return PlacementPlan(
            execution_class=request.execution_class,
            task_family=request.family,
            model_variant=request.model_variant,
            t5_mode=request.t5_mode,
            components=components,
            phase_plans=phase_plans,
            hardware_tier=request.hardware_tier,
            overhead_mb=profile.overhead_mb,
            ledger_gpu_mb=ledger.get_total_vram_used(),
            ledger_pinned_cpu_mb=ledger.get_total_pinned_cpu_mb(),
            available_gpu_mb=available_gpu_mb,
            available_ram_mb=available_ram_mb,
            runtime_family=runtime_contract["runtime_family"],
            runtime_posture=runtime_contract["runtime_posture"],
            streaming_profile=runtime_contract["streaming_profile"],
            resident_load_strategy=runtime_contract["resident_load_strategy"],
            fallback_model_variant=runtime_contract["fallback_model_variant"],
            reusable_components=reusable_components,
            notes=tuple(notes),
        )


# ---------------------------------------------------------------------------
# Facade for compatibility and system entry points
# ---------------------------------------------------------------------------


class PlacementSolver:
    registry = ModelUniverseRegistry()
    solver = ExecutionClassSolver(registry)
    planner = PlacementPlanner(registry, solver)

    @staticmethod
    def get_hardware_tier(vram_mb: float, ram_mb: float = 16384.0) -> HardwareTier:
        return PlacementSolver.solver.hardware_tier_for_vram(vram_mb, ram_mb)

    @staticmethod
    def solve(
        vram_total_mb: float,
        ram_total_mb: float,
        task_id: str,
        is_startup: bool = False,
        current_ledger: ResourceLedger | None = None,
        *,
        lora_count: int = 0,
    ) -> PlacementPlan:
        platform = os.environ.get("FOOOCUS_PLATFORM", os.name)
        hardware = HardwareProfile(
            total_vram_mb=float(vram_total_mb),
            total_ram_mb=float(ram_total_mb),
            platform=platform,
            system_reserved_mb=PlacementSolver.planner._platform_baseline(platform),
        )
        family, requested_variant = PlacementSolver.registry.parse_task_id(task_id)
        task_request = TaskRequest(
            family=family,
            requested_variant=requested_variant,
            lora_count=lora_count,
        )
        return PlacementSolver.planner.plan(
            hardware=hardware,
            task_request=task_request,
            current_ledger=current_ledger,
            is_startup=is_startup,
        )

    @staticmethod
    def solve_from_system(
        task_id: str,
        is_startup: bool = False,
        current_ledger: ResourceLedger | None = None,
        *,
        lora_count: int = 0,
    ) -> PlacementPlan:
        try:
            from backend import resources
            vram_total_mb = float(resources.get_total_memory() / (1024**2))
        except Exception:
            vram_total_mb = 0.0
        try:
            import psutil

            ram_total_mb = float(psutil.virtual_memory().total / (1024**2))
        except Exception:
            ram_total_mb = 16384.0
        return PlacementSolver.solve(
            vram_total_mb=vram_total_mb,
            ram_total_mb=ram_total_mb,
            task_id=task_id,
            is_startup=is_startup,
            current_ledger=current_ledger,
            lora_count=lora_count,
        )


def _warn_legacy_vram_mode_if_needed(current_vram_state: Any) -> None:
    state_name = getattr(current_vram_state, "name", str(current_vram_state))
    if state_name not in {"LOW_VRAM", "NO_VRAM"}:
        return
    warned = getattr(_warn_legacy_vram_mode_if_needed, "_warned", set())
    if state_name in warned:
        return
    logging.warning(
        "[Nex-Memory] Legacy VRAM flag behavior (%s) is deprecated; staging policy now follows execution class and phase-aware placement.",
        state_name,
    )
    warned = set(warned)
    warned.add(state_name)
    _warn_legacy_vram_mode_if_needed._warned = warned
