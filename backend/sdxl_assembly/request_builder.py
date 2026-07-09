from __future__ import annotations

import os
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np
import psutil
import torch

import modules.config as config
import modules.flags as flags
import modules.mask_processing as mask_processing
import modules.model_taxonomy as model_taxonomy
from modules.util import HWC3, get_file_from_folder_list
from backend.sdxl_assembly.contracts import (
    SDXLAssemblyRequest,
    SDXLLoraSpec,
    ResolvedFileIdentity,
    UNetPostureKind,
    TextEncoderPostureKind,
    VAEPostureKind,
    LoraPatchPostureKind,
    SDXLAssemblyEligibilityError,
    SDXLAssemblyValidationError,
    SpatialContextDescriptor,
    SDXLStructuralControlDescriptor,
)

logger = logging.getLogger(__name__)

# File identity cache to avoid expensive SHA-256 recalculation on model files
_FILE_IDENTITY_CACHE: Dict[Tuple[str, int, int], ResolvedFileIdentity] = {}


def _first_non_none(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def _task_attr_or_none(task_state: Any, name: str) -> Any:
    state_dict = getattr(task_state, "__dict__", None)
    if isinstance(state_dict, dict):
        return state_dict.get(name)
    return None


def _dict_or_empty(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _has_active_tasks(tasks: Any) -> bool:
    if tasks is None:
        return False
    try:
        return len(tasks) > 0
    except TypeError:
        return bool(tasks)


def _normalize_cn_path_map(paths: Any) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for cn_type, path in _dict_or_empty(paths).items():
        resolved_type = flags.resolve_cn_type(cn_type, default=cn_type)
        normalized[resolved_type] = path
    return normalized


def _resolve_controlnet_paths(
    controlnet_paths: Optional[Dict[str, str]],
    image_input_result: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    if image_input_result:
        resolved.update(_normalize_cn_path_map(image_input_result.get("controlnet_paths") or {}))
    if controlnet_paths:
        resolved.update(_normalize_cn_path_map(controlnet_paths))
    return resolved


def _resolve_contextual_assets(
    contextual_assets: Optional[Dict[str, Any]],
    image_input_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    resolved: Dict[str, Any] = {}
    if image_input_result:
        resolved.update(_dict_or_empty(image_input_result.get("contextual_assets") or {}))
    if contextual_assets:
        explicit_assets = _dict_or_empty(contextual_assets)
        merged_model_paths = dict(_dict_or_empty(resolved.get("contextual_model_paths")))
        merged_model_paths.update(_dict_or_empty(explicit_assets.get("contextual_model_paths")))
        resolved.update(explicit_assets)
        if merged_model_paths:
            resolved["contextual_model_paths"] = merged_model_paths
    return resolved


def _resolve_structural_preprocessor_paths(
    image_input_result: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    if not image_input_result:
        return {}
    return _normalize_cn_path_map(image_input_result.get("structural_preprocessor_paths") or {})


def _path_exists(path_str: Any) -> bool:
    return bool(path_str) and os.path.exists(str(path_str))


def _require_existing_path(path_str: Any, label: str) -> str:
    if not path_str:
        raise SDXLAssemblyEligibilityError(f"{label} is missing.")
    path = str(path_str)
    if not os.path.exists(path):
        raise SDXLAssemblyEligibilityError(f"{label} does not exist: {path}")
    return path


def _make_control_image_descriptor(raw_img: Any, label: str):
    from backend.sdxl_assembly.contracts import make_spatial_image_descriptor

    if raw_img is None:
        raise SDXLAssemblyEligibilityError(f"{label} is missing its input image asset.")

    candidate = str(raw_img) if isinstance(raw_img, os.PathLike) else raw_img
    if not isinstance(candidate, (torch.Tensor, np.ndarray, list, tuple)):
        unpacked = mask_processing.unpack_gradio_data(candidate)
        if unpacked is None:
            raise SDXLAssemblyEligibilityError(
                f"{label} could not resolve its input image asset into pixels."
            )
        candidate = HWC3(unpacked)

    try:
        return make_spatial_image_descriptor(candidate)
    except (TypeError, ValueError) as exc:
        raise SDXLAssemblyEligibilityError(
            f"{label} could not normalize its input image asset: {exc}"
        ) from exc


def _active_cn_task_types(
    task_state: Any,
    prepared_structural: Dict[str, Any],
    prepared_contextual: Dict[str, Any],
) -> Tuple[set[str], set[str], set[str]]:
    structural_types: set[str] = set()
    contextual_types: set[str] = set()
    all_types: set[str] = set()

    def remember(cn_type: Any, tasks: Any) -> None:
        if not _has_active_tasks(tasks):
            return
        normalized_type = flags.resolve_cn_type(cn_type, default=cn_type)
        all_types.add(normalized_type)
        channel = flags.get_cn_channel(normalized_type)
        if channel == flags.cn_structural:
            structural_types.add(normalized_type)
        elif channel == flags.cn_contextual:
            contextual_types.add(normalized_type)

    for cn_type, tasks in _dict_or_empty(getattr(task_state, "cn_tasks", {}) or {}).items():
        remember(cn_type, tasks)
    for cn_type, tasks in prepared_structural.items():
        remember(cn_type, tasks)
    for cn_type, tasks in prepared_contextual.items():
        remember(cn_type, tasks)

    return structural_types, contextual_types, all_types


def _structural_preprocessor_required(task_state: Any, cn_type: str) -> bool:
    return cn_type == flags.cn_depth and not bool(getattr(task_state, "skipping_cn_preprocessor", False))

def get_file_identity(path_str: str) -> ResolvedFileIdentity:
    """Calculates or retrieves cached file identity metadata."""
    if not path_str or not os.path.exists(path_str):
        raise FileNotFoundError(f"File not found: {path_str}")
        
    abs_path = os.path.abspath(path_str)
    stat = os.stat(abs_path)
    size = stat.st_size
    mtime_ns = stat.st_mtime_ns
    
    cache_key = (abs_path, size, mtime_ns)
    if cache_key in _FILE_IDENTITY_CACHE:
        return _FILE_IDENTITY_CACHE[cache_key]
        
    # Calculate SHA-256 in chunks
    digest = hashlib.sha256()
    with open(abs_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    sha256 = digest.hexdigest()
    
    identity = ResolvedFileIdentity(
        path=Path(abs_path),
        sha256=sha256,
        size_bytes=size,
        modified_ns=mtime_ns
    )
    _FILE_IDENTITY_CACHE[cache_key] = identity
    return identity


def _resolve_requested_vae_path(task_state: Any) -> Optional[str]:
    """Resolve the assembly VAE selection to a concrete file path.

    SDXL's default/shared VAE contract is the explicit `sdxl_vae.safetensors`
    asset under the configured VAE roots, not a checkpoint-embedded fallback.
    """
    vae_name = str(getattr(task_state, 'vae_name', '') or '').strip()
    if vae_name in {'', flags.default_vae, 'Default (model)', 'Default (Same as model)'}:
        vae_name = flags.default_vae
    if not vae_name:
        return None
    return get_file_from_folder_list(vae_name, config.path_vae)

def determine_eligibility(
    task_state: Any,
    loras: Optional[List[Tuple[str, float]]] = None,
    controlnet_paths: Optional[Dict[str, str]] = None,
    contextual_assets: Optional[Dict[str, Any]] = None,
    image_input_result: Optional[Dict[str, Any]] = None,
    *,
    force_eligible: bool = False,
) -> Tuple[bool, Optional[str]]:
    """Determines if a generation request is eligible for the new SDXL assembly lane.
    
    W09 production lane supports:
    - txt2img, inpaint, outpaint routes.
    - ControlNet structural: PyraCanny, Depth, CPDS.
    - ControlNet contextual: ImagePrompt, PuLID.
    
    Rejects: FaceID V2, FaceSwap, MLSD (retired); Resident SDXL/ControlNet (deferred);
             Flux Fill, removal, upscale, tiled refinement, non-SDXL base models.
    """
    # 0. Check for retired FaceID V2 / FaceSwap
    resolved_controlnet_paths = _resolve_controlnet_paths(controlnet_paths, image_input_result)
    resolved_contextual_assets = _resolve_contextual_assets(contextual_assets, image_input_result)
    contextual_model_paths = _normalize_cn_path_map(
        resolved_contextual_assets.get('contextual_model_paths', {})
    )
    structural_preprocessor_paths = _resolve_structural_preprocessor_paths(image_input_result)
    prepared_ctx = _dict_or_empty(getattr(task_state, 'prepared_contextual_cn_tasks', {}) or {})
    cn_tasks = _dict_or_empty(getattr(task_state, 'cn_tasks', {}) or {})
    for name in ["FaceID V2", "FaceSwap", "cn_faceid", "cn_ip_face"]:
        if name in contextual_model_paths or name in prepared_ctx or name in cn_tasks:
            return False, "FaceID V2 / FaceSwap is explicitly retired on the new assembly path."

    # 0.5. Check for retired MLSD
    prepared_structural = _dict_or_empty(getattr(task_state, 'prepared_structural_cn_tasks', {}) or {})
    for name in ["MLSD", "cn_mlsd"]:
        if name in resolved_controlnet_paths or name in prepared_structural or name in cn_tasks:
            return False, "MLSD is explicitly retired on the active ControlNet path."

    # 1. Check for resident SDXL or resident ControlNet
    policy = getattr(task_state, 'sdxl_execution_policy', None)
    if policy is not None and getattr(policy, 'execution_mode', None) == 'resident':
        return False, "Resident SDXL posture is not supported on the new assembly lane"

    if force_eligible:
        return True, None

    # 2. Check for tiled refinement
    if bool(getattr(task_state, 'tiled', False)):
        return False, "Tiled refinement is active"

    # 3. Check for spatial latent requests / initial latent (direct/probe override, not inpaint/outpaint VAE latents)
    if getattr(task_state, 'initial_latent', None) is not None:
        return False, "Spatial latent / initial_latent is active"

    # 4. Resolve Route Intent
    from modules.route_intent import (
        normalize_current_tab,
        resolve_route_intent,
        route_family_for_route_id,
    )
    intent = resolve_route_intent(task_state)
    
    # 5. Route check
    target_route = intent.route_id
    if target_route == "txt2img":
        requested_route_id = str(getattr(task_state, "requested_route_id", "") or "").strip().lower()
        requested_route_family = str(getattr(task_state, "requested_route_family", "") or "").strip().lower()
        requested_route_family = route_family_for_route_id(requested_route_id) or requested_route_family or ""
        goals = {
            str(goal or "").strip().lower()
            for goal in (getattr(task_state, "goals", []) or [])
        }
        current_tab = normalize_current_tab(getattr(task_state, "current_tab", ""))
        input_image_active = bool(getattr(task_state, "input_image_checkbox", False))

        if requested_route_family == "image_input" and requested_route_id in {"inpaint", "outpaint"}:
            target_route = requested_route_id
        elif input_image_active and ("inpaint" in goals or current_tab == "inpaint"):
            target_route = "inpaint"
        elif input_image_active and ("outpaint" in goals or current_tab == "outpaint"):
            target_route = "outpaint"

    if target_route not in {"txt2img", "inpaint", "outpaint"}:
        return False, f"Route '{target_route}' is not eligible for SDXL Assembly"

    # 5.5. Fail-closed route-specific spatial asset checks
    from unittest.mock import Mock
    def _is_invalid_asset(val: Any) -> bool:
        if val is None:
            return True
        if isinstance(val, Mock) or (hasattr(val, "__class__") and val.__class__.__name__ in ("Mock", "MagicMock", "NonCallableMock", "NonCallableMagicMock")):
            return True
        return False

    if target_route == "inpaint":
        img = None
        mask = None
        if image_input_result:
            img = image_input_result.get('inpaint_image')
            mask = image_input_result.get('inpaint_mask')
            if _is_invalid_asset(mask):
                mask = getattr(task_state, 'context_mask', None)

        if _is_invalid_asset(img):
            img = getattr(task_state, 'inpaint_input_image', None)
        if _is_invalid_asset(mask):
            mask = getattr(task_state, 'inpaint_mask_image', None)
            if _is_invalid_asset(mask):
                mask = getattr(task_state, 'context_mask', None)

        if _is_invalid_asset(img) or _is_invalid_asset(mask):
            return False, "Inpaint route requires valid inpaint image and mask assets"

    elif target_route == "outpaint":
        img = None
        if image_input_result:
            img = image_input_result.get('outpaint_image')
        if _is_invalid_asset(img):
            img = getattr(task_state, 'outpaint_input_image', None)

        if _is_invalid_asset(img):
            return False, "Outpaint route requires valid outpaint image asset"

    # 5.6. Fail-closed ControlNet input image checks
    for cn_type, tasks in cn_tasks.items():
        if _has_active_tasks(tasks):
            for t in tasks:
                if not t or _is_invalid_asset(t[0]):
                    return False, f"ControlNet '{cn_type}' is enabled but missing its input image asset"

    for cn_type, tasks in prepared_structural.items():
        if _has_active_tasks(tasks):
            for t in tasks:
                if not t or _is_invalid_asset(t[0]):
                    return False, f"Structural ControlNet '{cn_type}' is enabled but missing its input image asset"

    for cn_type, tasks in prepared_ctx.items():
        if _has_active_tasks(tasks):
            for t in tasks:
                if not t or _is_invalid_asset(t[0]):
                    return False, f"Contextual ControlNet '{cn_type}' is enabled but missing its input image asset"

    # 6. Check for active ControlNet types and the resolved assets needed to execute them.
    active_structural_types, active_contextual_types, active_task_types = _active_cn_task_types(
        task_state,
        prepared_structural,
        prepared_ctx,
    )
    active_cn_types = set(active_task_types)
    active_cn_types.update(resolved_controlnet_paths.keys())
    active_cn_types.update(contextual_model_paths.keys())

    for cn_type in active_cn_types:
        normalized_type = flags.resolve_cn_type(cn_type, default=cn_type)
        if normalized_type not in flags.cn_all_types:
            return False, f"ControlNet type '{cn_type}' is not supported on the new assembly lane"

    for cn_type in sorted(active_structural_types):
        checkpoint_path = resolved_controlnet_paths.get(cn_type)
        if not _path_exists(checkpoint_path):
            return False, (
                f"Structural ControlNet '{cn_type}' requires a resolved checkpoint path "
                "before it can use the SDXL assembly lane."
            )
        if _structural_preprocessor_required(task_state, cn_type):
            preprocessor_path = structural_preprocessor_paths.get(cn_type)
            if not _path_exists(preprocessor_path):
                return False, (
                    f"Structural ControlNet '{cn_type}' requires a resolved preprocessor path "
                    "before it can use the SDXL assembly lane."
                )

    for cn_type in sorted(active_contextual_types):
        model_path = contextual_model_paths.get(cn_type)
        if not _path_exists(model_path):
            return False, (
                f"Contextual ControlNet '{cn_type}' requires a resolved model path "
                "before it can use the SDXL assembly lane."
            )
        if cn_type == flags.cn_ip:
            if not _path_exists(resolved_contextual_assets.get('clip_vision_path')):
                return False, "ImagePrompt requires a resolved CLIP vision path before it can use the SDXL assembly lane."
            if not _path_exists(resolved_contextual_assets.get('ip_negative_path')):
                return False, "ImagePrompt requires a resolved IP negative path before it can use the SDXL assembly lane."
        elif cn_type == flags.cn_pulid:
            if not _path_exists(resolved_contextual_assets.get('eva_clip_path')):
                return False, "PuLID requires a resolved EVA-CLIP path before it can use the SDXL assembly lane."

    # 7. Check for SDXL architecture
    try:
        model_name = str(getattr(task_state, 'base_model_name', '') or '').strip()
        checkpoint_path = get_file_from_folder_list(model_name, config.paths_checkpoints)
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            return False, f"Base model path does not exist: {checkpoint_path}"
        resolved_taxonomy = config.resolve_model_taxonomy(checkpoint_path)
        if resolved_taxonomy.architecture != model_taxonomy.ARCHITECTURE_SDXL:
            return False, f"Model architecture {resolved_taxonomy.architecture} is not SDXL"
    except Exception as e:
        return False, f"Failed base model verification: {e}"

    return True, None

def build_assembly_request(
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
    force_eligible: bool = False,
) -> SDXLAssemblyRequest:
    """Builds a frozen per-image SDXLAssemblyRequest from input parameters."""
    eligible, reason = determine_eligibility(
        task_state=task_state,
        loras=loras,
        controlnet_paths=controlnet_paths,
        contextual_assets=contextual_assets,
        image_input_result=image_input_result,
        force_eligible=force_eligible,
    )

    if not eligible:
        raise SDXLAssemblyEligibilityError(f"Request is not eligible for SDXL Assembly: {reason}")

    # Resolve Checkpoint
    model_name = str(getattr(task_state, 'base_model_name', '') or '').strip()
    checkpoint_path = get_file_from_folder_list(model_name, config.paths_checkpoints)
    checkpoint_identity = get_file_identity(checkpoint_path)

    # Resolve VAE
    vae_path = _resolve_requested_vae_path(task_state)
    vae_identity = get_file_identity(vae_path) if vae_path else None

    # Resolve LoRA Specs
    resolved_input_loras = list(
        _task_attr_or_none(task_state, 'loras_processed')
        or loras
        or _task_attr_or_none(task_state, 'loras')
        or []
    )
    resolved_additional = list(
        base_model_additional_loras
        or _task_attr_or_none(task_state, 'base_model_additional_loras')
        or []
    )
    all_lora_tuples = resolved_input_loras + resolved_additional
    lora_specs_list = []
    
    # We resolve lookup path for each LoRA
    for lora_path, weight in all_lora_tuples:
        candidate = str(lora_path).strip()
        if candidate in {'', 'None'}:
            continue
        # Resolve path
        resolved_lora_path = candidate if os.path.exists(candidate) else get_file_from_folder_list(candidate, config.paths_lora_lookup)
        if not resolved_lora_path or not os.path.exists(resolved_lora_path):
            raise FileNotFoundError(f"Could not resolve LoRA file: {candidate}")
            
        spec_identity = get_file_identity(resolved_lora_path)
        lora_specs_list.append(SDXLLoraSpec(
            file_identity=spec_identity,
            unet_weight=float(weight),
            clip_weight=float(weight), # default to same weight for CLIP
        ))
        
    lora_specs = tuple(lora_specs_list)

    # Hashes
    # Prompt payload details
    prompt = str(task_dict.get('task_prompt', task_state.prompt) or '')
    neg_prompt = str(task_dict.get('task_negative_prompt', task_state.negative_prompt) or '')
    
    pos_texts = tuple(str(item) for item in (task_dict.get('positive') or [task_dict.get('task_prompt', task_state.prompt)]))
    neg_texts = tuple(str(item) for item in (task_dict.get('negative') or [task_dict.get('task_negative_prompt', task_state.negative_prompt)]))

    prompt_hash_payload = (prompt + "||" + neg_prompt + "||" + "".join(pos_texts) + "||" + "".join(neg_texts)).encode("utf-8")
    prompt_payload_hash = hashlib.sha256(prompt_hash_payload).hexdigest()

    # LoRA stack hash
    lora_hash_payload = "".join(f"{spec.file_identity.sha256}:{spec.unet_weight}:{spec.clip_weight}" for spec in lora_specs).encode("utf-8")
    lora_stack_hash = hashlib.sha256(lora_hash_payload).hexdigest()

    # Postures (Flux Fill user settings policy)
    policy = getattr(task_state, 'sdxl_execution_policy', None)
    unet_posture = UNetPostureKind.STREAMING
    clip_posture = TextEncoderPostureKind.CPU_PINNED
    vae_posture = VAEPostureKind.TRANSIENT
    lora_posture = LoraPatchPostureKind.STREAMING
    prefetch_depth = int(
        _first_non_none(
            _task_attr_or_none(task_state, 'prefetch_depth'),
            getattr(policy, 'prefetch_depth', None),
            default=1,
        )
    )
    prefetch_chunk_mb = int(
        _first_non_none(
            _task_attr_or_none(task_state, 'prefetch_chunk_mb'),
            getattr(policy, 'prefetch_chunk_mb', None),
            default=64,
        )
    )
    execution_metadata: Dict[str, Any] = {
        "pin_unet_host": bool(
            _first_non_none(
                _task_attr_or_none(task_state, 'pin_unet_host'),
                getattr(policy, 'pin_unet_host', None),
                default=False,
            )
        ),
        "release_warm_unet_after_task": bool(
            _first_non_none(
                _task_attr_or_none(task_state, 'release_warm_unet_after_task'),
                getattr(policy, 'release_warm_unet_after_task', None),
                default=False,
            )
        ),
        "release_text_encoder_after_task": bool(
            _first_non_none(
                _task_attr_or_none(task_state, 'release_text_encoder_after_task'),
                getattr(policy, 'release_text_encoder_after_task', None),
                default=False,
            )
        ),
    }

    if policy is not None and getattr(policy, 'execution_mode', None) == 'resident':
        logger.debug(
            "[SDXL Assembly] Ignoring legacy resident SDXL policy for W04 route cutover; "
            "the new route currently admits only the accepted streaming assembly."
        )

    # Retrieve quality configs
    sharpness = float(getattr(task_state, 'sharpness', 2.0))
    adaptive_cfg = float(getattr(task_state, 'adaptive_cfg', 7.0))
    adm_pos = float(getattr(task_state, 'adm_scaler_positive', 1.5))
    adm_neg = float(getattr(task_state, 'adm_scaler_negative', 0.8))
    adm_end = float(getattr(task_state, 'adm_scaler_end', 0.3))

    # Resolve structural ControlNets
    structural_controls_list = []
    if hasattr(task_state, 'get_cn_tasks_for_channel'):
        struct_tasks_dict = task_state.get_cn_tasks_for_channel(flags.cn_structural)
    else:
        struct_tasks_dict = {}

    structural_preprocessor_paths = _resolve_structural_preprocessor_paths(image_input_result)
    cn_paths = _resolve_controlnet_paths(controlnet_paths, image_input_result)

    for cn_type in getattr(flags, 'cn_structural_types', []):
        tasks = struct_tasks_dict.get(cn_type, [])
        for task in tasks:
            raw_img, cn_stop, cn_weight = task[:3]
            cn_start = task[3] if len(task) >= 4 else 0.0
            ckpt_path_str = _require_existing_path(
                cn_paths.get(cn_type),
                f"Structural ControlNet '{cn_type}' checkpoint path",
            )

            controlnet_identity = get_file_identity(ckpt_path_str)
            checkpoint_type = determine_controlnet_type(ckpt_path_str)

            # Accept either already-prepared tensors or unresolved UI-backed image payloads.
            image_desc = _make_control_image_descriptor(
                raw_img,
                f"Structural ControlNet '{cn_type}'",
            )

            # Resolve preprocessor
            preprocessor_id = cn_type
            if getattr(task_state, 'skipping_cn_preprocessor', False):
                preprocessor_id = "None"

            preprocessor_path = None
            preprocessor_path_str = structural_preprocessor_paths.get(cn_type)
            if _structural_preprocessor_required(task_state, cn_type):
                preprocessor_path_str = _require_existing_path(
                    preprocessor_path_str,
                    f"Structural ControlNet '{cn_type}' preprocessor path",
                )
            if preprocessor_path_str:
                preprocessor_path = Path(preprocessor_path_str)

            # Params
            preprocessor_params = {}
            if cn_type == flags.cn_canny:
                preprocessor_params = {
                    "low_threshold": int(getattr(task_state, 'canny_low_threshold', 64)),
                    "high_threshold": int(getattr(task_state, 'canny_high_threshold', 128))
                }

            # Target resolution
            target_width = int(getattr(task_state, 'width', 1024))
            target_height = int(getattr(task_state, 'height', 1024))

            unsupported_mode_errors = []
            if not controlnet_identity.path.exists():
                unsupported_mode_errors.append(f"Checkpoint path does not exist: {controlnet_identity.path}")

            slot_index = int(task[4]) if len(task) >= 5 else len(structural_controls_list)
            desc = SDXLStructuralControlDescriptor(
                slot_index=slot_index,
                control_type=cn_type,
                image_pixels=image_desc.pixels,
                image_fingerprint=image_desc.fingerprint,
                preprocessor_id=preprocessor_id,
                preprocessor_path=preprocessor_path,
                preprocessor_params=preprocessor_params,
                target_width=target_width,
                target_height=target_height,
                checkpoint_path=Path(ckpt_path_str),
                checkpoint_sha256=controlnet_identity.sha256,
                checkpoint_type=checkpoint_type,
                weight=float(cn_weight),
                start_percent=float(cn_start),
                end_percent=float(cn_stop),
                unsupported_mode_errors=tuple(unsupported_mode_errors)
            )
            structural_controls_list.append(desc)

    # Resolve contextual ControlNets
    contextual_controls_list = []
    if hasattr(task_state, 'get_cn_tasks_for_channel'):
        contextual_tasks_dict = task_state.get_cn_tasks_for_channel(flags.cn_contextual)
    else:
        contextual_tasks_dict = {}

    contextual_assets_resolved = _resolve_contextual_assets(contextual_assets, image_input_result)

    contextual_model_paths = _normalize_cn_path_map(contextual_assets_resolved.get('contextual_model_paths', {}))
    clip_vision_path_str = contextual_assets_resolved.get('clip_vision_path')
    ip_negative_path_str = contextual_assets_resolved.get('ip_negative_path')
    eva_clip_path_str = contextual_assets_resolved.get('eva_clip_path')
    insightface_model_names = list(contextual_assets_resolved.get('insightface_model_names') or ['antelopev2'])

    clip_vision_identity = None
    if clip_vision_path_str and os.path.exists(clip_vision_path_str):
        clip_vision_identity = get_file_identity(clip_vision_path_str)

    ip_negative_identity = None
    if ip_negative_path_str and os.path.exists(ip_negative_path_str):
        ip_negative_identity = get_file_identity(ip_negative_path_str)

    eva_clip_identity = None
    if eva_clip_path_str and os.path.exists(eva_clip_path_str):
        eva_clip_identity = get_file_identity(eva_clip_path_str)

    for cn_type in getattr(flags, 'cn_contextual_types', []):
        tasks = contextual_tasks_dict.get(cn_type, [])
        for idx, task in enumerate(tasks):
            raw_img = task[0]
            cn_stop = task[1]
            cn_weight = task[2]
            cn_start = task[3] if len(task) >= 4 else 0.0

            if len(task) < 5:
                raise SDXLAssemblyEligibilityError(
                    "Contextual direct/probe requests require an explicit ui_slot_index; "
                    "the live TaskState grouping path still does not preserve truthful slot continuity."
                )

            ui_slot_index = int(task[4])
            
            model_path_str = _require_existing_path(
                contextual_model_paths.get(cn_type),
                f"Contextual ControlNet '{cn_type}' model path",
            )
            if cn_type == flags.cn_ip:
                clip_vision_path_str = _require_existing_path(
                    clip_vision_path_str,
                    "ImagePrompt CLIP vision path",
                )
                ip_negative_path_str = _require_existing_path(
                    ip_negative_path_str,
                    "ImagePrompt IP negative path",
                )
                if clip_vision_identity is None:
                    clip_vision_identity = get_file_identity(clip_vision_path_str)
                if ip_negative_identity is None:
                    ip_negative_identity = get_file_identity(ip_negative_path_str)
            elif cn_type == flags.cn_pulid:
                eva_clip_path_str = _require_existing_path(
                    eva_clip_path_str,
                    "PuLID EVA-CLIP path",
                )
                if eva_clip_identity is None:
                    eva_clip_identity = get_file_identity(eva_clip_path_str)

            model_identity = get_file_identity(model_path_str)

            # Accept either already-prepared tensors or unresolved UI-backed image payloads.
            image_desc = _make_control_image_descriptor(
                raw_img,
                f"Contextual ControlNet '{cn_type}'",
            )

            source_image_role = "face_image" if cn_type == flags.cn_pulid else "image_prompt"

            preprocess_params = {}
            if cn_type == flags.cn_ip:
                preprocess_params = {"resize_to": 224}
            elif cn_type == flags.cn_pulid:
                preprocess_params = {"resize_to": 512, "crop": "norm_crop"}

            unsupported_mode_errors = []
            if not model_identity.path.exists():
                unsupported_mode_errors.append(f"Model path does not exist: {model_identity.path}")

            from backend.sdxl_assembly.contracts import SDXLContextualControlDescriptor
            desc = SDXLContextualControlDescriptor(
                ui_slot_index=ui_slot_index,
                control_type=cn_type,
                image_pixels=image_desc.pixels,
                image_fingerprint=image_desc.fingerprint,
                source_image_role=source_image_role,
                model_path=Path(model_path_str),
                model_sha256=model_identity.sha256,
                clip_vision_path=Path(clip_vision_path_str) if clip_vision_path_str else None,
                clip_vision_sha256=clip_vision_identity.sha256 if clip_vision_identity else None,
                ip_negative_path=Path(ip_negative_path_str) if ip_negative_path_str else None,
                ip_negative_sha256=ip_negative_identity.sha256 if ip_negative_identity else None,
                eva_clip_path=Path(eva_clip_path_str) if eva_clip_path_str else None,
                eva_clip_sha256=eva_clip_identity.sha256 if eva_clip_identity else None,
                insightface_model_names=tuple(insightface_model_names),
                preprocess_params=preprocess_params,
                weight=float(cn_weight),
                start_percent=float(cn_start),
                end_percent=float(cn_stop),
                unsupported_mode_errors=tuple(unsupported_mode_errors)
            )
            contextual_controls_list.append(desc)

    request = SDXLAssemblyRequest(
        request_id=f"req_{current_task_id}_{int(time.time())}",
        route_id="txt2img_assembly",
        image_index=current_task_id,
        image_count=total_count,
        checkpoint=checkpoint_identity,
        vae=vae_identity,
        model_variant_key="sdxl",
        prompt=prompt,
        negative_prompt=neg_prompt,
        positive_texts=pos_texts,
        negative_texts=neg_texts,
        width=int(task_state.width),
        height=int(task_state.height),
        steps=int(task_state.steps),
        cfg=float(task_state.cfg_scale),
        sampler=str(task_state.sampler_name),
        scheduler=str(final_scheduler_name),
        seed=int(task_dict['task_seed']),
        clip_layer=-abs(int(getattr(task_state, 'clip_skip', 1) or 1)),
        style_selections=tuple(getattr(task_state, 'style_selections', []) or []),
        prompt_payload_hash=prompt_payload_hash,
        lora_specs=lora_specs,
        lora_stack_hash=lora_stack_hash,
        unet_posture=unet_posture,
        clip_posture=clip_posture,
        vae_posture=vae_posture,
        lora_posture=lora_posture,
        prefetch_depth=prefetch_depth,
        prefetch_chunk_mb=prefetch_chunk_mb,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tiled=bool(getattr(task_state, 'tiled', False)),
        denoise_strength=float(denoising_strength) if denoising_strength is not None else None,
        sharpness=sharpness,
        adaptive_cfg=adaptive_cfg,
        adm_scaler_positive=adm_pos,
        adm_scaler_negative=adm_neg,
        adm_scaler_end=adm_end,
        metadata=execution_metadata,
        spatial_context=build_spatial_context_descriptor(task_state, image_input_result),
        structural_controls=tuple(structural_controls_list),
        contextual_controls=tuple(contextual_controls_list),
    )
    
    # Static validate
    request.validate()
    return request


_CONTROLNET_TYPE_CACHE: Dict[str, str] = {}

def determine_controlnet_type(ckpt_path: str) -> str:
    path_str = str(ckpt_path)
    if path_str in _CONTROLNET_TYPE_CACHE:
        return _CONTROLNET_TYPE_CACHE[path_str]

    from backend import utils as backend_utils
    try:
        controlnet_data = backend_utils.load_torch_file(path_str)
        if any("lllite" in key.lower() for key in controlnet_data.keys()):
            t = "lllite"
        elif "lora_controlnet" in controlnet_data:
            t = "control_lora"
        else:
            t = "controlnet"
    except Exception as e:
        t = f"error: {str(e)}"

    _CONTROLNET_TYPE_CACHE[path_str] = t
    return t


def build_spatial_context_descriptor(
    task_state: Any,
    image_input_result: Optional[Dict[str, Any]] = None,
) -> Optional[SpatialContextDescriptor]:
    """Helper to construct an immutable SpatialContextDescriptor from mutable inputs."""
    from unittest.mock import Mock
    def _clean_mock(val: Any) -> Any:
        if isinstance(val, Mock) or (hasattr(val, "__class__") and val.__class__.__name__ in ("Mock", "MagicMock", "NonCallableMock", "NonCallableMagicMock")):
            return None
        return val

    goals = set(getattr(task_state, 'goals', []) or [])
    image_input_result = image_input_result or {}
    
    mode = None
    source_pixels = None
    source_mask = None
    
    if 'outpaint' in goals:
        mode = "outpaint"
        source_pixels = image_input_result.get('outpaint_image')
        source_mask = image_input_result.get('outpaint_mask')
    elif 'inpaint' in goals:
        mode = "inpaint"
        source_pixels = image_input_result.get('inpaint_image')
        source_mask = getattr(task_state, 'context_mask', None)
        if _clean_mock(source_mask) is None:
            source_mask = image_input_result.get('inpaint_mask')
    elif image_input_result.get('inpaint_image') is not None:
        mode = "inpaint"
        source_pixels = image_input_result.get('inpaint_image')
        source_mask = image_input_result.get('inpaint_mask')
    elif image_input_result.get('outpaint_image') is not None:
        mode = "outpaint"
        source_pixels = image_input_result.get('outpaint_image')
        source_mask = image_input_result.get('outpaint_mask')
    else:
        # Check for plain image/latent or img2img mode
        source_pixels = image_input_result.get('source_pixels')
        if _clean_mock(source_pixels) is None:
            source_pixels = getattr(task_state, 'source_pixels', None)
        source_mask = image_input_result.get('source_mask')
        if _clean_mock(source_mask) is None:
            source_mask = getattr(task_state, 'source_mask', None)
        if _clean_mock(source_pixels) is not None:
            mode = "image"
            
    source_pixels = _clean_mock(source_pixels)
    source_mask = _clean_mock(source_mask)
            
    if mode is None or source_pixels is None:
        return None

    # Normalization & hashing is handled by factory helpers
    from backend.sdxl_assembly.contracts import (
        make_spatial_image_descriptor,
        make_spatial_mask_descriptor,
        SpatialContextDescriptor,
    )
    
    source_image_desc = make_spatial_image_descriptor(source_pixels)
    source_mask_desc = None
    if source_mask is not None:
        source_mask_desc = make_spatial_mask_descriptor(source_mask, source_image_desc)
        
    target_width = int(getattr(task_state, 'width', 1024))
    target_height = int(getattr(task_state, 'height', 1024))
    
    bbox = None
    bbox_area_ratio = 1.0
    pre_bb_image = None
    pre_bb_mask = None
    pre_blend_mask = None
    
    inpaint_context = getattr(task_state, 'inpaint_context', None)
    if inpaint_context is not None:
        if getattr(inpaint_context, 'bb', None) is not None:
            bbox = tuple(int(v) for v in inpaint_context.bb)
        if getattr(inpaint_context, 'target_w', None) is not None:
            target_width = int(inpaint_context.target_w)
        if getattr(inpaint_context, 'target_h', None) is not None:
            target_height = int(inpaint_context.target_h)
            
        if getattr(inpaint_context, 'bb_image', None) is not None:
            pre_bb_image = make_spatial_image_descriptor(inpaint_context.bb_image)
        if getattr(inpaint_context, 'bb_mask', None) is not None:
            pre_bb_mask = make_spatial_mask_descriptor(inpaint_context.bb_mask, pre_bb_image)
        if getattr(inpaint_context, 'blend_mask', None) is not None:
            pre_blend_mask = make_spatial_mask_descriptor(inpaint_context.blend_mask, source_image_desc)
            
        if bbox is not None:
            y1, y2, x1, x2 = bbox
            bbox_area = max(0, y2 - y1) * max(0, x2 - x1)
            full_area = max(1, source_image_desc.pixels.shape[1] * source_image_desc.pixels.shape[2])
            bbox_area_ratio = float(bbox_area) / float(full_area)

    outpaint_direction = getattr(task_state, 'outpaint_direction', None)
    if isinstance(outpaint_direction, list) and len(outpaint_direction) > 0:
        outpaint_direction = outpaint_direction[0].lower()
    elif isinstance(outpaint_direction, str):
        outpaint_direction = outpaint_direction.lower()
        
    outpaint_expansion_size = int(getattr(task_state, 'inpaint_outpaint_expansion_size', 384) or 384)
    outpaint_pixelate = bool(getattr(task_state, 'inpaint_pixelate_primer', True))
    denoise_strength = float(getattr(task_state, 'inpaint_strength', 1.0)) if 'inpaint' in goals else (
        float(getattr(task_state, 'outpaint_strength', 1.0)) if 'outpaint' in goals else None
    )

    return SpatialContextDescriptor(
        mode=mode,
        source_image=source_image_desc,
        source_mask=source_mask_desc,
        target_width=target_width,
        target_height=target_height,
        denoise_strength=denoise_strength,
        bbox=bbox,
        bbox_area_ratio=bbox_area_ratio,
        pre_bb_image=pre_bb_image,
        pre_bb_mask=pre_bb_mask,
        pre_blend_mask=pre_blend_mask,
        outpaint_direction=outpaint_direction,
        outpaint_expansion_size=outpaint_expansion_size,
        outpaint_pixelate=outpaint_pixelate,
    )
