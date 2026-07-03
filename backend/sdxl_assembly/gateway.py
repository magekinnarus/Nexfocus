from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from backend import resources
from backend.sdxl_assembly.contracts import SDXLAssemblyEligibilityError, SDXLAssemblyResult
from backend.sdxl_assembly.request_builder import determine_eligibility, build_assembly_request
from backend.sdxl_assembly.director import SDXLAssemblyDirector
from backend.sdxl_assembly.progress import SDXLAssemblyProgressCallback

logger = logging.getLogger(__name__)

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
