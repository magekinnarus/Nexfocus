from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from modules.flux_fill_surface import (
    FLUX_FILL_BLEND_MORPHOLOGICAL,
    FLUX_FILL_INPAINT_ROUTE_SDXL,
    OBJR_ENGINE_MAT,
)
import numpy as np


@dataclass
class TaskState:
    # --- Generation Parameters ---
    generate_image_grid: bool = False
    prompt: str = ""
    negative_prompt: str = ""
    style_selections: List[str] = field(default_factory=list)
    steps: int = 30
    original_steps: int = 30
    aspect_ratios_selection: str = "1024x1024"
    image_number: int = 1
    output_format: str = "png"
    seed: int = -1
    sharpness: float = 2.0
    cfg_scale: float = 4.0
    base_model_name: str = ""
    vae_name: str = ""
    clip_model_name: str = ""
    loras: List[Any] = field(default_factory=list)
    lora_channel_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    base_model_additional_loras: List[Any] = field(default_factory=list)
    input_image_checkbox: bool = False
    current_tab: str = "uov"
    uov_method: str = "Disabled"
    uov_input_image: Optional[np.ndarray] = None
    upscale_model: str = "None"
    upscale_scale_override: float = 0
    upscale_prompt: str = ""
    upscale_gan_output_image: Optional[Union[np.ndarray, str]] = None
    upscale_gan_tile_size: int = 256
    upscale_diffusion_refinement: bool = False
    upscale_refinement_tile_overlap: int = 128
    upscale_refinement_denoise: float = 0.3
    outpaint_selections: List[str] = field(default_factory=list)
    outpaint_input_image: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None
    outpaint_bb_image: Optional[np.ndarray] = None
    outpaint_bb_mask_data: str = ""
    outpaint_mask_image: Optional[np.ndarray] = None
    outpaint_additional_prompt: str = ""
    inpaint_input_image: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None
    inpaint_context_mask_image: Optional[np.ndarray] = None
    inpaint_bbox: str = ""
    inpaint_additional_prompt: str = ""
    inpaint_mask_image: Optional[np.ndarray] = None
    inpaint_bb_image: Optional[np.ndarray] = None
    inpaint_route: str = FLUX_FILL_INPAINT_ROUTE_SDXL
    remove_base_image: Optional[np.ndarray] = None
    remove_prompt: str = ""
    remove_mask_data: str = ""
    remove_bg_enabled: bool = False
    remove_obj_enabled: bool = False
    objr_engine: str = OBJR_ENGINE_MAT
    flux_fill_conditioning: str = "empty"
    flux_fill_prompt_cache: str = "temp"
    flux_fill_model_variant: str = ""
    flux_fill_unet_path: str = ""
    flux_fill_ae_path: str = ""
    flux_fill_conditioning_cache_path: str = ""
    flux_fill_clip_l_path: str = ""
    flux_fill_t5_path: str = ""
    flux_fill_t5_posture: str = ""
    flux_fill_disk_paged_t5_gc_interval: str = "auto"
    flux_fill_runtime_posture: str = "auto"
    sdxl_assembly_posture: str = "auto"
    prefetch_depth: int = 1
    prefetch_chunk_mb: int = 64
    objr_mask_dilate: int = 16
    objr_mask_blur: int = 6
    objr_blend_mode: str = FLUX_FILL_BLEND_MORPHOLOGICAL
    disable_preview: bool = False
    preview_update_interval: int = 1
    preview_max_side: int = 0
    disable_intermediate_results: bool = False
    disable_seed_increment: bool = False
    adm_scaler_positive: float = 1.5
    adm_scaler_negative: float = 0.8
    adm_scaler_end: float = 0.3
    adaptive_cfg: float = 7.0
    clip_skip: int = 1
    sampler_name: str = "dpmpp_2m_sde_gpu"
    scheduler_name: str = "karras"
    overwrite_width: int = -1
    overwrite_height: int = -1
    overwrite_upscale_strength: float = -1.0
    mixing_image_prompt_and_inpaint: bool = False
    mixing_image_prompt_and_outpaint: bool = False
    skipping_cn_preprocessor: bool = False
    canny_low_threshold: int = 64
    canny_high_threshold: int = 128
    controlnet_softness: float = 0.25
    debugging_inpaint_preprocessor: bool = False
    inpaint_disable_initial_latent: bool = False
    inpaint_engine: str = "None"
    inpaint_strength: float = 1.0
    inpaint_respective_field: float = 0.618
    inpaint_erode_or_dilate: int = 0
    inpaint_step2_checkbox: bool = False
    outpaint_step2_checkbox: bool = False
    outpaint_engine: str = "None"
    outpaint_strength: float = 1.0
    inpaint_outpaint_expansion_size: int = 384
    inpaint_pixelate_primer: bool = False
    context_mask: Optional[np.ndarray] = None
    outpaint_direction: Optional[str] = None
    save_metadata_to_images: bool = True
    metadata_scheme: Any = None  # modules.flags.MetadataScheme
    cn_tasks: Dict[str, List[Any]] = field(default_factory=dict)
    cn_tasks_by_channel: Dict[str, Dict[str, List[Any]]] = field(default_factory=dict)

    # --- Runtime State ---
    requested_route_id: str = ""
    requested_route_family: str = ""
    # Layer 0/1 queue-frozen workflow truth.  Raw UI ControlNet maps remain
    # available as a compatibility shell, but execution must use this plan.
    requested_source_surface: str = ""
    workflow_selection: Any = None
    workflow_plan: Any = None
    planned_cn_tasks: Dict[str, List[Any]] = field(default_factory=dict)
    planned_cn_tasks_by_channel: Dict[str, Dict[str, List[Any]]] = field(default_factory=dict)
    runtime_route_id: str = ""
    runtime_route_family: str = ""
    runtime_route_display_name: str = ""
    process_transition_action: str = ""
    process_transition_reason: str = ""
    process_transition_previous_family: str = ""
    process_transition_requested_family: str = ""
    process_transition_reuse_allowed: bool = False
    yields: List[Any] = field(default_factory=list)
    results: List[Any] = field(default_factory=list)
    last_stop: Union[bool, str] = False
    processing: bool = False
    current_progress: int = 0
    current_status_text: str = ""
    goals: List[str] = field(default_factory=list)
    initial_latent: Optional[Dict[str, Any]] = None
    denoising_strength: float = 1.0
    tiled: bool = False
    positive_cond: Optional[Any] = None
    negative_cond: Optional[Any] = None
    width: int = 1024
    height: int = 1024
    use_expansion: bool = False
    inpaint_context: object = None
    use_style: bool = True
    sdxl_execution_policy: object = None
    sdxl_execution_family: str = ""
    sdxl_residency_class: str = ""
    prepared_contextual_cn_tasks: Dict[str, List[Any]] = field(default_factory=dict)
    prepared_structural_cn_tasks: Dict[str, List[Any]] = field(default_factory=dict)

    def __post_init__(self):
        self.ensure_cn_task_maps()

    def ensure_cn_task_maps(self):
        from modules import flags

        normalized_tasks = {cn_type: [] for cn_type in flags.cn_all_types}
        for cn_type, tasks in self.cn_tasks.items():
            normalized_type = flags.resolve_cn_type(cn_type, default=None)
            if normalized_type is None:
                continue
            for task in tasks:
                task_list = list(task)
                if len(task_list) == 3:
                    task_list.append(0.0)
                if len(task_list) == 4:
                    task_list.append(len(normalized_tasks[normalized_type]))
                normalized_tasks[normalized_type].append(task_list)

        self.cn_tasks = normalized_tasks
        self.cn_tasks_by_channel = {
            flags.cn_structural: {cn_type: list(self.cn_tasks[cn_type]) for cn_type in flags.cn_structural_types},
            flags.cn_contextual: {cn_type: list(self.cn_tasks[cn_type]) for cn_type in flags.cn_contextual_types},
        }

    def set_workflow_plan(self, plan):
        """Bind the immutable plan and create only a worker-owned task mirror.

        The mirror preserves the established worker payload shape and may be
        enriched with prepared tensors.  Its slot/type membership is copied
        from the immutable plan and cannot be used to admit a new slot.
        """
        from modules import flags

        plan.validate()
        existing = getattr(self, "workflow_plan", None)
        if existing is not None:
            if existing is plan:
                return existing
            raise RuntimeError("Workflow plan is already bound; queued workflow truth cannot be replaced")
        self.workflow_plan = plan
        self.planned_cn_tasks = plan.materialize_cn_tasks()
        self.planned_cn_tasks_by_channel = {
            flags.cn_structural: {
                cn_type: list(self.planned_cn_tasks.get(cn_type, []))
                for cn_type in flags.cn_structural_types
            },
            flags.cn_contextual: {
                cn_type: list(self.planned_cn_tasks.get(cn_type, []))
                for cn_type in flags.cn_contextual_types
            },
        }
        return plan

    def _active_plan_slot_keys(self):
        plan = getattr(self, "workflow_plan", None)
        if plan is None:
            return None
        return {
            (item.control_type, int(item.ui_slot_index))
            for item in plan.controlnet_overlay.active_slot_descriptors
        }

    def add_cn_task(self, cn_type, task):
        from modules import flags

        normalized_type = flags.resolve_cn_type(cn_type, default=None)
        if normalized_type is None:
            return False

        channel = flags.get_cn_channel(normalized_type)
        if channel is None:
            return False

        task_list = list(task)
        if len(task_list) == 3:
            task_list.append(0.0)
        if len(task_list) == 4:
            task_list.append(len(self.cn_tasks[normalized_type]))

        self.cn_tasks[normalized_type].append(task_list)
        self.cn_tasks_by_channel[channel][normalized_type].append(task_list)
        return True

    def set_cn_tasks(self, cn_type, tasks):
        from modules import flags

        normalized_type = flags.resolve_cn_type(cn_type, default=None)
        if normalized_type is None:
            return False

        channel = flags.get_cn_channel(normalized_type)
        if channel is None:
            return False

        normalized_list = []
        for task in tasks:
            task_list = list(task)
            if len(task_list) == 3:
                task_list.append(0.0)
            if len(task_list) == 4:
                task_list.append(len(normalized_list))
            normalized_list.append(task_list)

        plan_slot_keys = self._active_plan_slot_keys()
        if plan_slot_keys is not None:
            # Prepared workers can replace payloads for planned slots, but
            # cannot add a slot/type that the queue-frozen plan did not admit.
            normalized_list = [
                task_list for task_list in normalized_list
                if len(task_list) >= 5
                and (normalized_type, int(task_list[4])) in plan_slot_keys
            ]
            self.planned_cn_tasks[normalized_type] = normalized_list
            self.planned_cn_tasks_by_channel[channel][normalized_type] = list(normalized_list)
            return True

        self.cn_tasks[normalized_type] = normalized_list
        self.cn_tasks_by_channel[channel][normalized_type] = list(normalized_list)
        return True

    def get_cn_tasks_for_channel(self, channel):
        if getattr(self, "workflow_plan", None) is not None:
            return self.planned_cn_tasks_by_channel.get(channel, {})
        return self.cn_tasks_by_channel.get(channel, {})
