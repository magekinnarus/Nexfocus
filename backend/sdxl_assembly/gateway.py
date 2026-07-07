from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from backend import resources
from backend.sdxl_assembly.contracts import SDXLAssemblyEligibilityError, SDXLAssemblyResult, SDXLAssemblyRequest
from backend.sdxl_assembly.request_builder import determine_eligibility, build_assembly_request
from backend.sdxl_assembly.director import SDXLAssemblyDirector
from backend.sdxl_assembly.progress import SDXLAssemblyProgressCallback
from backend.sdxl_assembly.lifecycle_coordinator import release_for_changes, LifecycleChange

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _GatewayRequestState:
    checkpoint_sha256: str
    vae_sha256: str | None
    posture_signature: tuple[str, str, str, str]
    lora_stack_hash: str
    prompt_payload_hash: str
    spatial_signature: Any
    structural_signature: tuple[Any, ...]
    contextual_signature: tuple[Any, ...]


def _freeze_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze_value(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze_value(item) for item in value), key=repr))
    return value


def _spatial_context_signature(spatial_context: Any) -> Any:
    if spatial_context is None:
        return None

    source_image = getattr(spatial_context, "source_image", None)
    source_mask = getattr(spatial_context, "source_mask", None)
    pre_bb_image = getattr(spatial_context, "pre_bb_image", None)
    pre_bb_mask = getattr(spatial_context, "pre_bb_mask", None)
    pre_blend_mask = getattr(spatial_context, "pre_blend_mask", None)

    return (
        str(getattr(spatial_context, "mode", "") or ""),
        getattr(source_image, "fingerprint", None),
        getattr(source_mask, "fingerprint", None),
        int(getattr(spatial_context, "target_width", 0) or 0),
        int(getattr(spatial_context, "target_height", 0) or 0),
        getattr(spatial_context, "denoise_strength", None),
        _freeze_value(getattr(spatial_context, "bbox", None)),
        float(getattr(spatial_context, "bbox_area_ratio", 1.0) or 1.0),
        getattr(pre_bb_image, "fingerprint", None),
        getattr(pre_bb_mask, "fingerprint", None),
        getattr(pre_blend_mask, "fingerprint", None),
        str(getattr(spatial_context, "outpaint_direction", "") or ""),
        int(getattr(spatial_context, "outpaint_expansion_size", 0) or 0),
        bool(getattr(spatial_context, "outpaint_pixelate", False)),
    )


def _structural_control_signature(control: Any) -> Any:
    slot_idx = getattr(control, "slot_index", None)
    slot_idx = -1 if slot_idx is None else int(slot_idx)
    return (
        slot_idx,
        str(getattr(control, "control_type", "") or ""),
        getattr(control, "image_fingerprint", None),
        str(getattr(control, "preprocessor_id", "") or ""),
        str(getattr(control, "preprocessor_path", "") or ""),
        _freeze_value(getattr(control, "preprocessor_params", {})),
        int(getattr(control, "target_width", 0) or 0),
        int(getattr(control, "target_height", 0) or 0),
        str(getattr(control, "checkpoint_path", "") or ""),
        str(getattr(control, "checkpoint_sha256", "") or ""),
        str(getattr(control, "checkpoint_type", "") or ""),
        _freeze_value(getattr(control, "unsupported_mode_errors", ())),
        _freeze_value(getattr(control, "extra_params", {})),
    )


def _contextual_control_signature(control: Any) -> Any:
    ui_slot_idx = getattr(control, "ui_slot_index", None)
    ui_slot_idx = -1 if ui_slot_idx is None else int(ui_slot_idx)
    return (
        ui_slot_idx,
        str(getattr(control, "control_type", "") or ""),
        getattr(control, "image_fingerprint", None),
        str(getattr(control, "source_image_role", "") or ""),
        str(getattr(control, "model_path", "") or ""),
        str(getattr(control, "model_sha256", "") or ""),
        str(getattr(control, "clip_vision_path", "") or ""),
        str(getattr(control, "clip_vision_sha256", "") or ""),
        str(getattr(control, "ip_negative_path", "") or ""),
        str(getattr(control, "ip_negative_sha256", "") or ""),
        str(getattr(control, "eva_clip_path", "") or ""),
        str(getattr(control, "eva_clip_sha256", "") or ""),
        _freeze_value(getattr(control, "insightface_model_names", ())),
        _freeze_value(getattr(control, "preprocess_params", {})),
        _freeze_value(getattr(control, "unsupported_mode_errors", ())),
    )


def _build_gateway_request_state(request: SDXLAssemblyRequest) -> _GatewayRequestState:
    structural_signature = tuple(
        sorted(
            (_structural_control_signature(control) for control in request.structural_controls),
            key=lambda item: item[0],
        )
    )
    contextual_signature = tuple(
        sorted(
            (_contextual_control_signature(control) for control in request.contextual_controls),
            key=lambda item: item[0],
        )
    )
    return _GatewayRequestState(
        checkpoint_sha256=str(request.checkpoint.sha256 or ""),
        vae_sha256=request.vae.sha256 if request.vae is not None else None,
        posture_signature=(
            str(request.unet_posture.value),
            str(request.clip_posture.value),
            str(request.vae_posture.value),
            str(request.lora_posture.value),
        ),
        lora_stack_hash=str(request.lora_stack_hash or ""),
        prompt_payload_hash=str(request.prompt_payload_hash or ""),
        spatial_signature=_spatial_context_signature(request.spatial_context),
        structural_signature=structural_signature,
        contextual_signature=contextual_signature,
    )


def _calculate_gateway_changes(
    previous_state: _GatewayRequestState,
    request_state: _GatewayRequestState,
) -> list[LifecycleChange]:
    changes: list[LifecycleChange] = []

    def add(change: LifecycleChange) -> None:
        if change not in changes:
            changes.append(change)

    if previous_state.checkpoint_sha256 != request_state.checkpoint_sha256:
        add(LifecycleChange.CHECKPOINT_CHANGE)
    if previous_state.vae_sha256 != request_state.vae_sha256:
        add(LifecycleChange.SPATIAL_VAE_CHANGE)
    if previous_state.posture_signature != request_state.posture_signature:
        add(LifecycleChange.SPINE_POSTURE_CHANGE)
    if previous_state.lora_stack_hash != request_state.lora_stack_hash:
        add(LifecycleChange.LORA_STACK_CHANGE)
    if previous_state.prompt_payload_hash != request_state.prompt_payload_hash:
        add(LifecycleChange.PROMPT_CHANGE)
    if previous_state.spatial_signature != request_state.spatial_signature:
        add(LifecycleChange.SPATIAL_VAE_CHANGE)
    if previous_state.structural_signature != request_state.structural_signature:
        add(LifecycleChange.STRUCTURAL_CN_CHANGE)
    if previous_state.contextual_signature != request_state.contextual_signature:
        add(LifecycleChange.CONTEXTUAL_CN_CHANGE)

    return changes

def is_eligible_for_sdxl_assembly(
    task_state: Any,
    loras: List[Tuple[str, float]],
    controlnet_paths: Optional[Dict[str, str]] = None,
    contextual_assets: Optional[Dict[str, Any]] = None,
    image_input_result: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[str]]:
    """Gateway check to determine if a request should go to the new assembly lane or old path."""
    return determine_eligibility(
        task_state=task_state,
        loras=loras,
        controlnet_paths=controlnet_paths,
        contextual_assets=contextual_assets,
        image_input_result=image_input_result,
    )

_LAST_REQUEST_STATE: Optional[_GatewayRequestState] = None

def clear_gateway_state() -> None:
    """Clear last request tracker to prevent stale request comparisons."""
    global _LAST_REQUEST_STATE
    _LAST_REQUEST_STATE = None

def run_sdxl_assembly_task(
    task_state: Any,
    task_dict: Dict[str, Any],
    current_task_id: int,
    total_count: int,
    all_steps: int,
    preparation_steps: int,
    denoising_strength: Optional[float],
    final_scheduler_name: str,
    *,
    loras: List[Tuple[str, float]],
    controlnet_paths: Optional[Dict[str, str]] = None,
    contextual_assets: Optional[Dict[str, Any]] = None,
    base_model_additional_loras: Optional[List[Tuple[str, float]]] = None,
    image_input_result: Optional[Dict[str, Any]] = None,
    progressbar_callback: Optional[Any] = None,
    preview_runtime_holder: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Gateway entry point that executes an eligible task via the SDXL Assembly lane."""
    # 1. Build frozen request
    try:
        request = build_assembly_request(
            task_state=task_state,
            task_dict=task_dict,
            current_task_id=current_task_id,
            total_count=total_count,
            all_steps=all_steps,
            preparation_steps=preparation_steps,
            denoising_strength=denoising_strength,
            final_scheduler_name=final_scheduler_name,
            loras=loras,
            controlnet_paths=controlnet_paths,
            contextual_assets=contextual_assets,
            base_model_additional_loras=base_model_additional_loras,
            image_input_result=image_input_result,
        )
    except resources.InterruptProcessingException:
        raise
    except Exception as e:
        logger.error(f"[SDXL Assembly] Request building/freeze failed: {e}")
        raise RuntimeError(f"Request building/freeze failed: {e}") from e

    # 1.5. Detect changes from last request and trigger domain releases
    global _LAST_REQUEST_STATE
    request_state = _build_gateway_request_state(request)
    if _LAST_REQUEST_STATE is not None:
        changes = _calculate_gateway_changes(_LAST_REQUEST_STATE, request_state)
        if changes:
            release_for_changes(changes, reason="gateway_transition")

    _LAST_REQUEST_STATE = request_state

    # 2. Select assembly
    try:
        assembly = SDXLAssemblyDirector.select_assembly(request)
    except resources.InterruptProcessingException:
        raise
    except Exception as e:
        logger.error(f"[SDXL Assembly] Assembly selection failed: {e}")
        raise RuntimeError(f"Assembly selection failed: {e}") from e

    if preview_runtime_holder is not None:
        preview_runtime_holder["assembly"] = assembly

    # 3. Create progress callback
    progress_cb = SDXLAssemblyProgressCallback(request, progressbar_callback)

    # 4. Execute
    try:
        result: SDXLAssemblyResult = assembly.execute(request, callback=progress_cb)
    finally:
        assembly.close()
        if preview_runtime_holder is not None:
            preview_runtime_holder["assembly"] = None

    return result.output_image
