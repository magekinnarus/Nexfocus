from __future__ import annotations

from dataclasses import dataclass
import io
import os
import threading
import urllib.parse

import modules.async_worker as worker
import modules.flags as flags
from modules.flux_fill_surface import (
    FLUX_FILL_INPAINT_ROUTE_FLUX,
    OBJR_ENGINE_FLUX_FILL,
    OBJR_ENGINE_MAT,
    normalize_flux_fill_inpaint_route,
    normalize_objr_engine,
)
from modules.route_intent import resolve_route_intent
import numpy as np
from PIL import Image


@dataclass
class CompletedTaskRecord:
    task_id: str
    prompt: str
    model_name: str
    seed: object
    images: list[str]
    workflow_name: str = ""
    prompt_label: str = "Prompt"
    model_label: str = "Model"
    show_prompt: bool = True


_state_mutex = threading.RLock()
completed_tasks_history: list[CompletedTaskRecord] = []
_last_seen_active_task = None
_last_active_task_id: str | None = None
_last_progress_state = {"visible": False, "number": 0, "text": ""}
_last_preview_value = None
_last_preview_revision = 0
_last_preview_encoded_bytes: bytes | None = None
_last_preview_encoded_media_type = "image/png"
_last_preview_encoded_cache_key: tuple[int, int, int] | None = None


def build_file_url(path: str) -> str:
    return f"/file={urllib.parse.quote(str(path), safe='')}"


def build_completed_image_url(task_id: str, image_index: int) -> str:
    safe_task_id = urllib.parse.quote(str(task_id or ""), safe="")
    return f"/runtime_surface_api/completed_image/{safe_task_id}/{int(image_index)}"


def build_preview_image_url(revision: int | None = None) -> str:
    base_url = "/runtime_surface_api/preview_image"
    if revision is None:
        return base_url
    return f"{base_url}?revision={int(revision)}"


def build_prompt_preview(prompt: str, *, limit: int = 40) -> str:
    prompt_preview = str(prompt or "")[:limit]
    if len(str(prompt or "")) > limit:
        prompt_preview += "..."
    if not prompt_preview.strip():
        prompt_preview = "Image generation"
    return prompt_preview


def _normalize_text(value) -> str:
    return str(value or "").strip()


def _merge_prompt_text(primary: str, secondary: str) -> str:
    primary = _normalize_text(primary)
    secondary = _normalize_text(secondary)
    if secondary == "":
        return primary
    if primary == "":
        return secondary
    return secondary + "\n" + primary


def _normalize_objr_engine_name(value) -> str:
    try:
        return normalize_objr_engine(value)
    except ValueError:
        normalized = _normalize_text(value).lower().replace("(", "").replace(")", "")
        if "flux" in normalized:
            return OBJR_ENGINE_FLUX_FILL
        return OBJR_ENGINE_MAT


def _is_flux_fill_inpaint_route(value) -> bool:
    return normalize_flux_fill_inpaint_route(value) == FLUX_FILL_INPAINT_ROUTE_FLUX


def _is_upscale_request(state) -> bool:
    return resolve_route_intent(state, prefer_runtime_route=True).wants_upscale


def _has_outpaint_request(state) -> bool:
    return resolve_route_intent(state, prefer_runtime_route=True).wants_outpaint


def _has_inpaint_request(state) -> bool:
    return resolve_route_intent(state, prefer_runtime_route=True).wants_inpaint


def _is_removal_request(state) -> bool:
    return resolve_route_intent(state, prefer_runtime_route=True).wants_removal


def _resolve_task_display_fields(state) -> dict:
    base_model_name = _normalize_text(getattr(state, "base_model_name", ""))
    seed = getattr(state, "seed", "")
    goals = set(getattr(state, "goals", []) or [])
    remove_bg_enabled = bool(getattr(state, "remove_bg_enabled", False) or (flags.remove_bg in goals))
    remove_obj_enabled = bool(getattr(state, "remove_obj_enabled", False) or (flags.remove_obj in goals))
    current_tab = _normalize_text(getattr(state, "current_tab", "")).lower()
    runtime_route_id = _normalize_text(getattr(state, "runtime_route_id", "")).lower()

    workflow_name = "Txt2Img"
    prompt_text = _normalize_text(getattr(state, "prompt", ""))
    prompt_label = "Prompt"
    model_label = "Model"
    model_name = base_model_name
    show_prompt = bool(prompt_text)

    if _is_removal_request(state):
        selected_engine = _normalize_objr_engine_name(getattr(state, "objr_engine", ""))
        prompt_text = _normalize_text(getattr(state, "remove_prompt", ""))
        prompt_label = "Remove Prompt"
        model_label = "Engine"
        show_prompt = bool(prompt_text)
        if remove_bg_enabled and remove_obj_enabled:
            workflow_name = "Background + Object Removal"
            model_name = f"Background Removal + {selected_engine}"
        elif remove_bg_enabled:
            workflow_name = "Background Removal"
            model_name = "Background Removal"
        else:
            workflow_name = "Flux Fill Object Removal" if "Flux Fill" in selected_engine else "Object Removal"
            model_name = selected_engine
    elif _is_upscale_request(state):
        method = _normalize_text(getattr(state, "uov_method", "")).lower()
        upscale_model_name = _normalize_text(getattr(state, "upscale_model", ""))
        model_label = "Engine"
        if "color enhancement" in method or "color-enhanced-upscale" in method:
            workflow_name = "Color Enhancement"
            prompt_text = _normalize_text(getattr(state, "upscale_prompt", ""))
            prompt_label = "Upscale Prompt"
            model_label = "Model"
            show_prompt = bool(prompt_text)
            model_name = base_model_name or "Selected SDXL Model"
        elif "super-upscale" in method:
            workflow_name = "Super Upscale"
            show_prompt = bool(prompt_text)
            prompt_label = "Prompt"
            model_label = "Pipeline"
            target_label = "Provided Upscale Target"
            model_name = target_label if base_model_name == "" else f"{target_label} + {base_model_name}"
        else:
            workflow_name = "Upscale"
            prompt_text = ""
            show_prompt = False
            model_name = upscale_model_name if upscale_model_name not in {"", "None"} else "Default Upscaler"
    elif _has_outpaint_request(state):
        workflow_name = "Outpaint"
        prompt_text = _merge_prompt_text(getattr(state, "prompt", ""), getattr(state, "outpaint_additional_prompt", ""))
        prompt_label = "Outpaint Prompt"
        show_prompt = bool(prompt_text)
        engine_name = _normalize_text(getattr(state, "outpaint_engine", ""))
        if engine_name not in {"", "None"}:
            model_name = engine_name if base_model_name == "" else f"{base_model_name} + Inpaint Patch {engine_name}"
        else:
            model_name = base_model_name
    elif _has_inpaint_request(state):
        prompt_text = _merge_prompt_text(getattr(state, "prompt", ""), getattr(state, "inpaint_additional_prompt", ""))
        prompt_label = "Inpaint Prompt"
        show_prompt = bool(prompt_text)
        if runtime_route_id == "flux_inpaint" or _is_flux_fill_inpaint_route(getattr(state, "inpaint_route", "")):
            workflow_name = "Flux Inpaint"
            model_label = "Engine"
            model_name = "Flux Fill"
        else:
            workflow_name = "Inpaint"
            engine_name = _normalize_text(getattr(state, "inpaint_engine", ""))
            if engine_name not in {"", "None"}:
                model_name = engine_name if base_model_name == "" else f"{base_model_name} + Inpaint Patch {engine_name}"
            else:
                model_name = base_model_name

    if model_name == "":
        model_name = workflow_name

    return {
        "workflow_name": workflow_name,
        "prompt_text": prompt_text,
        "prompt_preview": build_prompt_preview(prompt_text) if show_prompt else "",
        "prompt_label": prompt_label,
        "model_label": model_label,
        "model_name": model_name,
        "seed": seed,
        "show_prompt": bool(show_prompt),
    }


def reset_runtime_surface_state():
    global _last_seen_active_task, _last_active_task_id
    global _last_preview_value, _last_preview_revision, _last_progress_state
    global _last_preview_encoded_bytes, _last_preview_encoded_media_type, _last_preview_encoded_cache_key
    with _state_mutex:
        completed_tasks_history.clear()
        _last_seen_active_task = None
        _last_active_task_id = None
        _last_preview_value = None
        _last_preview_revision = 0
        _last_preview_encoded_bytes = None
        _last_preview_encoded_media_type = "image/png"
        _last_preview_encoded_cache_key = None
        _last_progress_state = {"visible": False, "number": 0, "text": ""}


def set_progress_state(*, visible: bool, number: int = 0, text: str = ""):
    with _state_mutex:
        _last_progress_state["visible"] = bool(visible)
        _last_progress_state["number"] = int(number or 0)
        _last_progress_state["text"] = str(text or "")


def _set_preview_value(value):
    global _last_preview_value, _last_preview_revision
    global _last_preview_encoded_bytes, _last_preview_encoded_media_type, _last_preview_encoded_cache_key
    _last_preview_value = value
    _last_preview_revision += 1
    _last_preview_encoded_bytes = None
    _last_preview_encoded_media_type = "image/png"
    _last_preview_encoded_cache_key = None


def get_preview_state():
    with _state_mutex:
        return _last_preview_value, _last_preview_revision


def _normalize_preview_array(value) -> tuple[np.ndarray, str] | tuple[None, None]:
    if value is None:
        return None, None

    try:
        array = np.asarray(value)
    except Exception:
        return None, None

    if array.size == 0:
        return None, None

    if np.issubdtype(array.dtype, np.floating):
        if float(np.nanmax(array)) <= 1.0:
            array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.ndim == 2:
        return array, "L"
    if array.ndim != 3:
        return None, None

    channels = int(array.shape[2])
    if channels == 1:
        return array[:, :, 0], "L"
    if channels == 3:
        return array, "RGB"
    if channels == 4:
        return array, "RGBA"
    return None, None


def has_preview_image() -> bool:
    with _state_mutex:
        preview_value = _last_preview_value
        if preview_value is None:
            return False
        if isinstance(preview_value, str):
            return bool(preview_value) and os.path.exists(preview_value)
        normalized_array, _ = _normalize_preview_array(preview_value)
        return normalized_array is not None


def get_preview_image_path() -> str | None:
    with _state_mutex:
        preview_value = _last_preview_value
        if not isinstance(preview_value, str):
            return None
        preview_path = str(preview_value or "")
        if not preview_path or not os.path.exists(preview_path):
            return None
        return preview_path


def _coerce_preview_bound(value) -> int:
    try:
        return max(0, int(value or 0))
    except Exception:
        return 0


def _resolve_preview_target_size(width: int, height: int, *, max_width: int, max_height: int) -> tuple[int, int]:
    width = max(1, int(width or 1))
    height = max(1, int(height or 1))
    max_width = _coerce_preview_bound(max_width)
    max_height = _coerce_preview_bound(max_height)
    if max_width <= 0 and max_height <= 0:
        return width, height

    scale_candidates = [1.0]
    if max_width > 0:
        scale_candidates.append(float(max_width) / float(width))
    if max_height > 0:
        scale_candidates.append(float(max_height) / float(height))

    scale = min(scale_candidates)
    if scale >= 1.0:
        return width, height

    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    return target_width, target_height


def _encode_preview_image_bytes(preview_value, *, max_width: int = 0, max_height: int = 0) -> tuple[bytes | None, str | None]:
    image_mode = None
    if isinstance(preview_value, str):
        preview_path = str(preview_value or "")
        if not preview_path or not os.path.exists(preview_path):
            return None, None
        try:
            with Image.open(preview_path) as source_image:
                preview_image = source_image.copy()
        except Exception:
            return None, None
    else:
        normalized_array, image_mode = _normalize_preview_array(preview_value)
        if normalized_array is None or image_mode is None:
            return None, None
        preview_image = Image.fromarray(normalized_array, mode=image_mode)

    target_width, target_height = _resolve_preview_target_size(
        preview_image.width,
        preview_image.height,
        max_width=max_width,
        max_height=max_height,
    )
    if (target_width, target_height) != (preview_image.width, preview_image.height):
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        preview_image = preview_image.resize((target_width, target_height), resample=resampling)

    image_buffer = io.BytesIO()
    preview_image.save(image_buffer, format="PNG")
    return image_buffer.getvalue(), "image/png"


def get_preview_image_bytes(*, max_width: int = 0, max_height: int = 0) -> tuple[bytes | None, str | None, int]:
    global _last_preview_encoded_bytes, _last_preview_encoded_media_type, _last_preview_encoded_cache_key

    with _state_mutex:
        preview_value = _last_preview_value
        preview_revision = int(_last_preview_revision)
        if preview_value is None:
            return None, None, preview_revision
        requested_max_width = _coerce_preview_bound(max_width)
        requested_max_height = _coerce_preview_bound(max_height)
        cache_key = (preview_revision, requested_max_width, requested_max_height)
        if (
            _last_preview_encoded_bytes is not None
            and _last_preview_encoded_cache_key == cache_key
        ):
            return _last_preview_encoded_bytes, _last_preview_encoded_media_type, preview_revision
        preview_snapshot = preview_value

    encoded_bytes, media_type = _encode_preview_image_bytes(
        preview_snapshot,
        max_width=requested_max_width,
        max_height=requested_max_height,
    )
    if encoded_bytes is None or media_type is None:
        return None, None, preview_revision

    with _state_mutex:
        if preview_revision == int(_last_preview_revision):
            _last_preview_encoded_bytes = encoded_bytes
            _last_preview_encoded_media_type = media_type
            _last_preview_encoded_cache_key = cache_key

    return encoded_bytes, media_type, preview_revision


def _record_completed_task(task, images):
    if not images:
        return False

    task_id = getattr(task, "task_id", None)
    if task_id is None or any(record.task_id == task_id for record in completed_tasks_history):
        return False

    state = getattr(task, "state", None)
    display = _resolve_task_display_fields(state)
    completed_tasks_history.append(
        CompletedTaskRecord(
            task_id=task_id,
            prompt=display["prompt_text"],
            model_name=display["model_name"],
            seed=display["seed"],
            images=list(images),
            workflow_name=display["workflow_name"],
            prompt_label=display["prompt_label"],
            model_label=display["model_label"],
            show_prompt=display["show_prompt"],
        )
    )
    if len(completed_tasks_history) > 50:
        completed_tasks_history.pop(0)
    return True


def _drain_task_events(task):
    latest_preview_value = None
    latest_progress_pct = None
    latest_progress_msg = None
    finished_images = None

    while len(task.yields) > 0:
        flag, product = task.yields.pop(0)
        if flag == "preview":
            pct, msg, img = product
            latest_progress_pct = pct
            latest_progress_msg = msg
            if img is not None:
                latest_preview_value = img
        elif flag == "finish":
            finished_images = list(product) if isinstance(product, list) else [product]
            _record_completed_task(task, finished_images)

    return latest_preview_value, latest_progress_pct, latest_progress_msg, finished_images


def drain_worker_state():
    global _last_seen_active_task, _last_active_task_id
    with _state_mutex:
        active_task = worker.get_active_task()
        active_task_id = getattr(active_task, "task_id", None)

        previous_task = None
        if (
            _last_seen_active_task is not None
            and getattr(_last_seen_active_task, "task_id", None) != active_task_id
        ):
            previous_task = _last_seen_active_task

        if previous_task is not None:
            _, _, _, previous_finished_images = _drain_task_events(previous_task)
            if previous_finished_images:
                _set_preview_value(previous_finished_images[0])

        if active_task is not None and _last_active_task_id != active_task_id:
            set_progress_state(visible=True, number=1, text="Waiting for task to start ...")
            _last_active_task_id = active_task_id

        if active_task is not None:
            latest_preview_value, latest_progress_pct, latest_progress_msg, finished_images = _drain_task_events(active_task)

            if finished_images:
                _set_preview_value(finished_images[0])

            if latest_preview_value is not None:
                _set_preview_value(latest_preview_value)

            if latest_progress_msg is not None:
                set_progress_state(
                    visible=True,
                    number=latest_progress_pct or 0,
                    text=latest_progress_msg,
                )

            _last_seen_active_task = active_task
        else:
            if _last_active_task_id is not None:
                set_progress_state(visible=False, number=0, text="")
            _last_active_task_id = None
            if previous_task is not None:
                _last_seen_active_task = None


def _serialize_task(task):
    if task is None:
        return None

    state = getattr(task, "state", None)
    display = _resolve_task_display_fields(state)
    return {
        "task_id": getattr(task, "task_id", ""),
        "workflow_name": display["workflow_name"],
        "prompt_preview": display["prompt_preview"],
        "prompt_label": display["prompt_label"],
        "show_prompt": display["show_prompt"],
        "model_label": display["model_label"],
        "model_name": display["model_name"],
        "seed": display["seed"],
        "progress": max(0, min(int(getattr(state, "current_progress", 0) or 0), 100)),
        "status_text": str(getattr(state, "current_status_text", "") or "").strip(),
    }


def get_completed_image_path(task_id: str, image_index: int) -> str | None:
    if image_index < 0:
        return None

    with _state_mutex:
        for record in completed_tasks_history:
            if record.task_id != task_id:
                continue
            if image_index >= len(record.images):
                return None
            image_path = str(record.images[image_index] or "")
            if not image_path or not os.path.exists(image_path):
                return None
            return image_path
    return None


def remove_completed_task(task_id: str) -> bool:
    normalized_task_id = str(task_id or "").strip()
    if not normalized_task_id:
        return False

    with _state_mutex:
        for index, record in enumerate(completed_tasks_history):
            if record.task_id != normalized_task_id:
                continue
            completed_tasks_history.pop(index)
            return True
    return False


def get_runtime_snapshot():
    drain_worker_state()
    with _state_mutex:
        active_task = worker.get_active_task()
        pending_tasks = list(worker.async_tasks)
        active_payload = _serialize_task(active_task)
        preview_available = False
        preview_value = _last_preview_value
        if isinstance(preview_value, str):
            preview_available = bool(preview_value) and os.path.exists(preview_value)
        elif preview_value is not None:
            preview_available = _normalize_preview_array(preview_value)[0] is not None
        if active_payload is not None and not active_payload["status_text"]:
            active_payload["status_text"] = "Waiting for task to start ..."

        return {
            "progress": dict(_last_progress_state),
            "preview": {
                "revision": int(_last_preview_revision),
                "available": preview_available,
                "image_url": build_preview_image_url(_last_preview_revision) if preview_available else None,
            },
            "running": active_payload,
            "pending": [_serialize_task(task) for task in pending_tasks],
            "completed": [
                {
                    "task_id": record.task_id,
                    "workflow_name": str(record.workflow_name or ""),
                    "prompt_preview": build_prompt_preview(record.prompt),
                    "prompt_label": str(record.prompt_label or "Prompt"),
                    "show_prompt": bool(record.show_prompt and str(record.prompt or "").strip()),
                    "model_label": str(record.model_label or "Model"),
                    "model_name": str(record.model_name or ""),
                    "seed": record.seed,
                    "images": list(record.images),
                    "image_urls": [
                        build_completed_image_url(record.task_id, image_index)
                        for image_index, _ in enumerate(record.images)
                    ],
                }
                for record in reversed(completed_tasks_history)
            ],
            "queue_count": len(pending_tasks) + (1 if active_task is not None else 0),
        }


def request_skip_active():
    active_task = worker.get_active_task()
    if active_task is not None:
        worker.request_interrupt("skip", active_task)


def _clear_progress_if_idle():
    if worker.get_active_task() is None and len(worker.async_tasks) == 0:
        set_progress_state(visible=False, number=0, text="")


def request_cancel_task(task_id: str):
    worker.cancel_task(task_id)
    _clear_progress_if_idle()


def request_delete_completed_task(task_id: str) -> bool:
    return remove_completed_task(task_id)


def request_clear_all():
    active_task = worker.get_active_task()
    if active_task is not None:
        worker.request_interrupt("stop", active_task)
    while len(worker.async_tasks) > 0:
        task = worker.async_tasks.pop(0)
        task.yields.append(["finish", []])
    _clear_progress_if_idle()
