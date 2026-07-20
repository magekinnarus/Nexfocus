import os
from dataclasses import dataclass
import torch
import numpy as np
import logging
from PIL import Image
from typing import Any, List, Tuple

from modules.flux_fill_surface import (
    FLUX_FILL_BLEND_ALPHA,
    FLUX_FILL_BLEND_MORPHOLOGICAL,
    FLUX_FILL_INPAINT_ROUTE_FLUX,
    FLUX_FILL_INPAINT_ROUTE_SDXL,
    OBJR_ENGINE_CHOICES,
    OBJR_ENGINE_FLUX_FILL,
    OBJR_ENGINE_MAT,
    is_flux_fill_inpaint_route,
    is_flux_fill_route_family,
    normalize_flux_fill_blend_mode,
    normalize_flux_fill_inpaint_route,
    normalize_objr_engine,
)
import modules.mask_processing as mask_processing
from backend.auxiliary_workers.mat_inpaint_worker import (
    MatInpaintWorker,
    get_segments,
    mask_floor,
    mask_unsqueeze,
    pad_reflect_once,
    resize_square,
    run_mat_inpaint,
    to_torch,
    undo_resize_square,
)
from modules.util import HWC3
from backend.legacy_runtime_errors import LegacyFluxArchivedError

logger = logging.getLogger(__name__)

FLUX_FILL_GUIDANCE_DEFAULT = 15.0
FLUX_FILL_CONDITIONING_EMPTY = "empty"
FLUX_FILL_CONDITIONING_PROMPT = "prompt"
FLUX_FILL_EMPTY_CONDITIONING_ASSET_ID = "inpaint.flux_fill.empty_conditioning"
FLUX_FILL_CONDITIONING_BY_KIND = {
    FLUX_FILL_CONDITIONING_EMPTY: FLUX_FILL_EMPTY_CONDITIONING_ASSET_ID,
}
FLUX_FILL_PROMPT_CACHE_TEMP = "temp"
FLUX_FILL_PROMPT_CACHE_PERMANENT = "permanent"
FLUX_FILL_MASK_GROW = 16
FLUX_FILL_MASK_BLUR = 6
FLUX_FILL_EMPTY_CONDITIONING_RELATIVE_PATH = os.path.join("flux", "flux_empty_conditioning.pt")

FLUX_FILL_VRAM_CLASS_RESIDENT = "16gb_plus"
FLUX_FILL_VRAM_CLASS_CONSTRAINED = "8gb_class"
FLUX_FILL_RUNTIME_POSTURE_RESIDENT = "resident"
FLUX_FILL_RUNTIME_POSTURE_HYBRID = "streaming"
FLUX_FILL_TEXT_ENCODER_ROUTE_BUDGET_MB = {
    "": 0.0,
    "flux_fill": 0.0,
    "removal": 2048.0,
    "upscale": 4096.0,
    "txt2img": 6144.0,
    "image_input": 8192.0,
    "inpaint": 8192.0,
    "outpaint": 8192.0,
    "sdxl": 8192.0,
}


@dataclass(frozen=True)
class FluxFillRouteReconciliation:
    decision: str
    reason: str
    target_signature: tuple[str, str, str] | None = None
    active_signature_before: tuple[str, str, str] | None = None
    active_signature_after: tuple[str, str, str] | None = None
    session_started: bool = False
    session_reused: bool = False
    session_replaced: bool = False
    session_torn_down: bool = False
    next_route_family: str | None = None
    text_encoder_kept: bool | None = None
    text_encoder_action: str | None = None
    text_encoder_reason: str | None = None


@dataclass(frozen=True)
class FluxFillHardwareProfile:
    profile_name: str
    total_ram_mb: float
    available_ram_mb: float
    total_vram_mb: float
    available_vram_mb: float
    is_colab: bool
    vram_class: str
    runtime_posture: str
    gpu_name: str | None = None
    cuda_capability: str | None = None
    flux_acceleration_class: str | None = None
    tensor_core_accelerated: bool = False


@dataclass(frozen=True)
class _FluxFillPolicyContext:
    profile_name: str
    total_ram_mb: float
    available_ram_mb: float
    total_vram_mb: float
    available_vram_mb: float
    is_colab: bool
    gpu_name: str | None
    cuda_capability: str | None
    flux_acceleration_class: str | None
    tensor_core_accelerated: bool
    placement_plan: Any
def inspect_flux_fill_hardware(profile: Any | None = None) -> Any:
    raise LegacyFluxArchivedError()

def evaluate_flux_fill_text_encoder_residency(profile: Any | None = None, *, next_route_family: Any | None = None) -> Any:
    from backend.flux_fill_v3.contracts import UNetSpineKind
    from backend.flux_fill_v3.activation import resolve_flux_fill_t5_posture

    total_ram_gb = None
    if profile is not None and hasattr(profile, "total_ram_mb"):
        total_ram_gb = profile.total_ram_mb / 1024.0

    unet_spine = UNetSpineKind.STREAMING
    if profile is not None and getattr(profile, "runtime_posture", None) == "resident":
        unet_spine = UNetSpineKind.RESIDENT

    t5_posture = resolve_flux_fill_t5_posture(unet_spine, total_ram_gb)

    return {
        "keep_resident": False,
        "t5_posture": t5_posture.value,
        "unet_spine": unet_spine.value,
    }

def select_flux_fill_tier(profile: Any | None = None) -> str:
    raise LegacyFluxArchivedError()

def normalize_flux_fill_t5_variant(variant: str | None) -> str:
    raise LegacyFluxArchivedError()

def should_keep_flux_fill_text_encoder_resident(profile: Any | None = None, *, next_route_family: Any | None = None) -> bool:
    return evaluate_flux_fill_text_encoder_residency(profile, next_route_family=next_route_family).get("keep_resident", False)

def reconcile_flux_fill_text_encoder_residency(*, profile: Any | None = None, next_route_family: Any | None = None) -> Any:
    return {"text_encoder_action": "cleared"}

def select_flux_fill_t5_variant(profile: Any | None = None, *, variant: str | None = None) -> str:
    from backend.flux_fill_v3.contracts import UNetSpineKind
    from backend.flux_fill_v3.activation import resolve_flux_fill_t5_posture

    total_ram_gb = None
    if profile is not None and hasattr(profile, "total_ram_mb"):
        total_ram_gb = profile.total_ram_mb / 1024.0

    unet_spine = UNetSpineKind.STREAMING
    if profile is not None and getattr(profile, "runtime_posture", None) == "resident":
        unet_spine = UNetSpineKind.RESIDENT

    resolve_flux_fill_t5_posture(unet_spine, total_ram_gb)
    if variant is not None and str(variant).strip().lower() not in {"", "fp16"}:
        raise ValueError("Only the native Flux Fill fp16 text encoder is supported.")
    return "fp16"

def get_flux_fill_t5_asset_id(variant: str | None = None, *, profile: Any | None = None) -> str:
    raise LegacyFluxArchivedError()

def ensure_flux_fill_t5_asset(variant: str | None = None, *, profile: Any | None = None, progress: bool = True) -> tuple[str, str, str]:
    raise LegacyFluxArchivedError()

def normalize_flux_fill_conditioning(conditioning: str | None) -> str:
    return "empty"

def normalize_flux_fill_prompt_cache(cache_mode: str | None) -> str:
    return "temp"

def generate_flux_fill_prompt_conditioning_cache(prompt: str, **kwargs) -> str:
    raise LegacyFluxArchivedError()

def prepare_flux_fill_prompt_conditioning_cache_path(prompt: str, **kwargs) -> str:
    raise LegacyFluxArchivedError()

def generate_flux_fill_prompt_conditioning(prompt: str, **kwargs) -> Any:
    raise LegacyFluxArchivedError()

def get_flux_fill_conditioning_cache_path(conditioning: str | None = None, *, progress: bool = True) -> str:
    raise LegacyFluxArchivedError()

def get_flux_empty_conditioning_cache_path(conditioning: str | None = None, *, progress: bool = True) -> str:
    raise LegacyFluxArchivedError()

def safe_resolve_flux_fill_asset_paths(**kwargs) -> dict[str, Any]:
    raise LegacyFluxArchivedError()

def resolve_flux_fill_asset_paths(**kwargs) -> dict[str, Any]:
    raise LegacyFluxArchivedError()

def get_active_flux_fill_session() -> Any:
    return None

def get_active_flux_fill_session_signature() -> Any:
    return None

def has_active_flux_fill_session() -> bool:
    return False

def ensure_active_flux_fill_session(**kwargs) -> Any:
    raise LegacyFluxArchivedError()

def reconcile_active_flux_fill_session(**kwargs) -> Any:
    from collections import namedtuple
    Reconciliation = namedtuple("Reconciliation", ["decision", "text_encoder_action"])
    return Reconciliation("ignored", "cleared")

def end_active_flux_fill_session(*args, **kwargs) -> Any:
    return None

def _select_flux_fill_mode(image, mode=None):
    return "context_crop"


@torch.inference_mode()
def remove_object_flux_fill(
    image: np.ndarray,
    mask: np.ndarray,
    seed: int = 0,
    mask_dilate: int = FLUX_FILL_MASK_GROW,
    *,
    mask_blur: int = FLUX_FILL_MASK_BLUR,
    tier: str | None = None,
    conditioning: str | None = None,
    prompt: str | None = None,
    prompt_cache: str | None = FLUX_FILL_PROMPT_CACHE_TEMP,
    blend_mode: str | None = FLUX_FILL_BLEND_MORPHOLOGICAL,
    guidance: float = FLUX_FILL_GUIDANCE_DEFAULT,
    steps: int = 30,
    sampler: str = "euler",
    scheduler: str = "normal",
    callback: Any | None = None,
    disable_pbar: bool = True,
    progress: bool = True,
    mode: str | None = None,
) -> np.ndarray:
    raise LegacyFluxArchivedError()


@torch.inference_mode()
def run_flux_fill_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    seed: int = 0,
    mask_dilate: int = FLUX_FILL_MASK_GROW,
    *,
    mask_blur: int = FLUX_FILL_MASK_BLUR,
    tier: str | None = None,
    conditioning: str | None = None,
    prompt: str | None = None,
    prompt_cache: str | None = FLUX_FILL_PROMPT_CACHE_TEMP,
    blend_mode: str | None = FLUX_FILL_BLEND_MORPHOLOGICAL,
    guidance: float = FLUX_FILL_GUIDANCE_DEFAULT,
    steps: int = 30,
    sampler: str = "euler",
    scheduler: str = "normal",
    callback: Any | None = None,
    disable_pbar: bool = True,
    progress: bool = True,
    mode: str | None = None,
) -> np.ndarray:
    raise LegacyFluxArchivedError()


def remove_object_with_engine(
    image: np.ndarray,
    mask: np.ndarray,
    seed: int = 0,
    mask_dilate: int = FLUX_FILL_MASK_GROW,
    *,
    engine: str | None = OBJR_ENGINE_MAT,
    flux_tier: str | None = None,
    flux_conditioning: str | None = None,
    flux_prompt: str | None = None,
    flux_prompt_cache: str | None = FLUX_FILL_PROMPT_CACHE_TEMP,
    flux_mask_blur: int = FLUX_FILL_MASK_BLUR,
    flux_blend_mode: str | None = FLUX_FILL_BLEND_MORPHOLOGICAL,
    flux_steps: int = 30,
    flux_sampler: str = "euler",
    flux_scheduler: str = "normal",
    flux_callback: Any | None = None,
    flux_disable_pbar: bool = True,
) -> np.ndarray:
    selected_engine = normalize_objr_engine(engine)
    if selected_engine == OBJR_ENGINE_FLUX_FILL:
        return remove_object_flux_fill(
            image,
            mask,
            seed=seed,
            mask_dilate=mask_dilate,
            mask_blur=flux_mask_blur,
            tier=flux_tier,
            conditioning=flux_conditioning,
            prompt=flux_prompt,
            prompt_cache=flux_prompt_cache,
            blend_mode=flux_blend_mode,
            steps=flux_steps,
            sampler=flux_sampler,
            scheduler=flux_scheduler,
            callback=flux_callback,
            disable_pbar=flux_disable_pbar,
        )
    return remove_object(image, mask, seed=seed, mask_dilate=mask_dilate)

def remove_object_from_file(
    image_path: str,
    mask_path: str,
    seed: int = 0,
    mask_dilate: int = FLUX_FILL_MASK_GROW,
    *,
    engine: str | None = OBJR_ENGINE_MAT,
    flux_tier: str | None = None,
    flux_conditioning: str | None = None,
    flux_prompt: str | None = None,
    flux_prompt_cache: str | None = FLUX_FILL_PROMPT_CACHE_TEMP,
    flux_mask_blur: int = FLUX_FILL_MASK_BLUR,
    flux_blend_mode: str | None = FLUX_FILL_BLEND_MORPHOLOGICAL,
    flux_steps: int = 30,
    flux_sampler: str = "euler",
    flux_scheduler: str = "normal",
    flux_callback: Any | None = None,
    flux_disable_pbar: bool = True,
) -> str:
    """Filepath invariant wrapper with explicit MAT/Flux dispatch."""
    with Image.open(image_path) as img:
        img_np = HWC3(np.array(img.convert('RGBA')))
    with Image.open(mask_path) as msk:
        msk_np = np.array(msk.convert('L'))

    res_np = remove_object_with_engine(
        img_np,
        msk_np,
        seed=seed,
        mask_dilate=mask_dilate,
        engine=engine,
        flux_tier=flux_tier,
        flux_conditioning=flux_conditioning,
        flux_prompt=flux_prompt,
        flux_prompt_cache=flux_prompt_cache,
        flux_mask_blur=flux_mask_blur,
        flux_blend_mode=flux_blend_mode,
        flux_steps=flux_steps,
        flux_sampler=flux_sampler,
        flux_scheduler=flux_scheduler,
        flux_callback=flux_callback,
        flux_disable_pbar=flux_disable_pbar,
    )

    return mask_processing.save_to_temp_png(res_np)


def prepare_flux_fill_mask(mask: np.ndarray, *, grow: int = FLUX_FILL_MASK_GROW, blur: int = FLUX_FILL_MASK_BLUR) -> np.ndarray:
    import cv2

    mask_np = np.asarray(mask)
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]
    if mask_np.ndim != 2:
        raise ValueError(f"Flux Fill mask must be HW or HWC, got shape {mask_np.shape}.")

    mask_np = np.where(mask_np > 0, 255, 0).astype(np.uint8)
    if grow > 0:
        kernel_size = max(1, int(grow) * 2 + 1)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_np = cv2.dilate(mask_np, kernel, iterations=1)
    if blur > 0:
        kernel_size = max(3, int(blur) * 2 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        mask_np = cv2.GaussianBlur(mask_np, (kernel_size, kernel_size), 0)
    return mask_np.clip(0, 255).astype(np.uint8)


_expand_flux_fill_mask = prepare_flux_fill_mask


def load_model(model_name: str = MatInpaintWorker.default_model_name):
    """Retired compatibility entry point; MAT lifetime belongs to the worker."""
    raise RuntimeError(
        "modules.objr_engine.load_model is retired; use MatInpaintWorker or "
        "run_mat_inpaint()."
    )


def unload_model() -> None:
    """No-op compatibility hook; every MAT worker tears down per request."""
    return None


def remove_object(
    image: np.ndarray,
    mask: np.ndarray,
    seed: int = 0,
    mask_dilate: int = FLUX_FILL_MASK_GROW,
) -> np.ndarray:
    """MAT compatibility bridge backed by a fresh ephemeral worker."""
    result = run_mat_inpaint(image, mask, seed=seed, mask_dilate=mask_dilate)
    if result is None:
        raise ValueError("MAT object removal requires an image and mask.")
    return result


