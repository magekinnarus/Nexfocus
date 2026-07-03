from __future__ import annotations

import os
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

import psutil
import torch

import modules.config as config
import modules.flags as flags
import modules.model_taxonomy as model_taxonomy
from modules.util import get_file_from_folder_list
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

def determine_eligibility(
    task_state: Any,
    loras: Optional[List[Tuple[str, float]]] = None,
    controlnet_paths: Optional[Dict[str, str]] = None,
    contextual_assets: Optional[Dict[str, Any]] = None,
    image_input_result: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[str]]:
    """Determines if a generation request is eligible for the new SDXL assembly lane.
    
    Rejects: inpaint/outpaint, image inputs, ControlNet, contextual adapters, tiled refinement, spatial latents.
    """
    # 1. Check for image inputs
    if image_input_result:
        for k, v in image_input_result.items():
            if v is not None:
                return False, f"image_input_result key '{k}' is active"

    # 2. Check for inpaint / outpaint goals
    goals = set(getattr(task_state, 'goals', []) or [])
    if 'inpaint' in goals or 'outpaint' in goals:
        return False, f"Inpaint/Outpaint goals are active: {goals}"

    # 3. Check for tiled refinement
    if bool(getattr(task_state, 'tiled', False)):
        return False, "Tiled refinement is active"

    # 4. Check for structural ControlNet
    if controlnet_paths:
        return False, "Structural ControlNet paths are active"
    prepared_structural = getattr(task_state, 'prepared_structural_cn_tasks', {}) or {}
    if any(tasks for tasks in prepared_structural.values()):
        return False, "Prepared structural ControlNet tasks are active"

    # 5. Check for contextual adapters
    if contextual_assets:
        return False, "Contextual adapter assets are active"
    prepared_contextual = getattr(task_state, 'prepared_contextual_cn_tasks', {}) or {}
    if any(tasks for tasks in prepared_contextual.values()):
        return False, "Prepared contextual adapter tasks are active"

    # 6. Check for spatial latent requests / initial latent
    if getattr(task_state, 'initial_latent', None) is not None:
        return False, "Spatial latent / initial_latent is active"

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
) -> SDXLAssemblyRequest:
    """Builds a frozen per-image SDXLAssemblyRequest from input parameters."""
    eligible, reason = determine_eligibility(
        task_state=task_state,
        loras=loras,
        controlnet_paths=controlnet_paths,
        contextual_assets=contextual_assets,
        image_input_result=image_input_result,
    )
    if not eligible:
        raise SDXLAssemblyEligibilityError(f"Request is not eligible for SDXL Assembly: {reason}")

    # Resolve Checkpoint
    model_name = str(getattr(task_state, 'base_model_name', '') or '').strip()
    checkpoint_path = get_file_from_folder_list(model_name, config.paths_checkpoints)
    checkpoint_identity = get_file_identity(checkpoint_path)

    # Resolve VAE
    vae_name = str(getattr(task_state, 'vae_name', '') or '').strip()
    if vae_name in {'', flags.default_vae, 'Default (model)', 'Default (Same as model)'}:
        vae_path = flags.default_vae
    else:
        vae_path = get_file_from_folder_list(vae_name, config.path_vae)
        
    vae_identity = None
    if vae_path and os.path.exists(vae_path):
        vae_identity = get_file_identity(vae_path)

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
    )
    
    # Static validate
    request.validate()
    return request
