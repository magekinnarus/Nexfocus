import logging
import inspect
import time
import psutil
import torch
from contextlib import contextmanager
from threading import RLock
from typing import Any, Callable, Optional
from backend import resources
from backend.sdxl_assembly.contracts import SDXLAssemblyRequest

logger = logging.getLogger(__name__)
_TELEMETRY_SINKS: list[Callable[[dict[str, Any]], None]] = []
_TELEMETRY_SINKS_LOCK = RLock()
_STEP_MEMORY_TELEMETRY_INTERVAL = 5


def _callback_accepts_positional_count(callback: Callable, count: int) -> Optional[bool]:
    """Return whether *callback* can accept exactly ``count`` positional args.

    Signature inspection lets the assembly boundary distinguish the sampler
    callback shape from the legacy UI progressbar shape without invoking a
    callback once with the wrong arguments.  Some extension callables do not
    expose a signature; ``None`` keeps the historical five-argument behavior
    for those callables.
    """
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return None

    try:
        signature.bind(*([object()] * count))
    except TypeError:
        return False
    return True


def _resolve_callback_progress_state(callback: Callable) -> Any:
    """Resolve the live task state used by the legacy UI progress callback.

    ``modules.async_worker.progressbar`` is an unbound three-argument
    function, so the task state is not carried by the callback itself.  Its
    module globals expose ``get_active_task`` while the worker is running;
    use that seam lazily to avoid importing the worker from this module.
    """
    explicit_state = getattr(callback, "_sdxl_progress_state", None)
    if explicit_state is not None:
        return explicit_state

    callback_globals = getattr(callback, "__globals__", None)
    get_active_task = (
        callback_globals.get("get_active_task")
        if isinstance(callback_globals, dict)
        else None
    )
    if not callable(get_active_task):
        return None

    try:
        active_task = get_active_task()
    except Exception:
        return None
    return getattr(active_task, "state", None) if active_task is not None else None


def _should_forward_text_only_progress(raw_callback: Callable | None, completed_steps: int, total_steps: int) -> bool:
    if raw_callback is None or not bool(getattr(raw_callback, "_sdxl_forward_text_only", False)):
        return True
    resolved_total_steps = max(int(total_steps), 1)
    cadence = max(5, resolved_total_steps // 10 or 1)
    return (
        completed_steps <= 1
        or completed_steps >= resolved_total_steps
        or (completed_steps % cadence) == 0
    )


def add_telemetry_sink(sink: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
    """Register a diagnostic sink for structured telemetry snapshots."""
    with _TELEMETRY_SINKS_LOCK:
        _TELEMETRY_SINKS.append(sink)

    def unregister() -> None:
        with _TELEMETRY_SINKS_LOCK:
            try:
                _TELEMETRY_SINKS.remove(sink)
            except ValueError:
                pass

    return unregister


@contextmanager
def telemetry_sink(sink: Callable[[dict[str, Any]], None]):
    unregister = add_telemetry_sink(sink)
    try:
        yield
    finally:
        unregister()


def _emit_telemetry_snapshot(snapshot: dict[str, Any]) -> None:
    with _TELEMETRY_SINKS_LOCK:
        sinks = list(_TELEMETRY_SINKS)
    for sink in sinks:
        try:
            sink(dict(snapshot))
        except Exception as exc:
            logger.debug("[SDXL Telemetry] Telemetry sink failed: %s", exc)

def log_telemetry(event: str, extra_msg: str = "") -> None:
    """Logs a standardized SDXL telemetry memory snapshot at DEBUG level."""
    try:
        # Get host memory stats
        virtual_mem = psutil.virtual_memory()
        ram_total = float(virtual_mem.total) / (1024 * 1024)
        ram_free = float(virtual_mem.available) / (1024 * 1024)
        ram_used = ram_total - ram_free
        
        # Get process RSS
        process = psutil.Process()
        proc_rss = float(process.memory_info().rss) / (1024 * 1024)
        
        # Get GPU memory stats if CUDA is available
        vram_total = 0.0
        vram_free = 0.0
        if torch.cuda.is_available():
            try:
                vram_free_bytes, vram_total_bytes = torch.cuda.mem_get_info()
                vram_total = float(vram_total_bytes) / (1024 * 1024)
                vram_free = float(vram_free_bytes) / (1024 * 1024)
            except Exception:
                pass
                
        metrics_str = (
            f"ram_total={ram_total:.1f}MB "
            f"ram_free={ram_free:.1f}MB "
            f"ram_used={ram_used:.1f}MB "
            f"vram_total={vram_total:.1f}MB "
            f"vram_free={vram_free:.1f}MB "
            f"proc_rss={proc_rss:.1f}MB"
        )
        
        log_line = f"[SDXL Telemetry] {event} | {metrics_str}"
        if extra_msg:
            log_line += f" | {extra_msg}"
            
        logger.debug(log_line)
        _emit_telemetry_snapshot(
            {
                "event": event,
                "extra": extra_msg,
                "timestamp_s": time.time(),
                "ram_total_mb": ram_total,
                "ram_free_mb": ram_free,
                "ram_used_mb": ram_used,
                "vram_total_mb": vram_total,
                "vram_free_mb": vram_free,
                "proc_rss_mb": proc_rss,
            }
        )
    except Exception as e:
        logger.debug(f"[SDXL Telemetry] Failed to gather memory telemetry: {e}")

class SDXLAssemblyProgressCallback:
    """Progress callback wrapper that hooks into the sampling loop and logs telemetry."""
    def __init__(
        self,
        request: SDXLAssemblyRequest,
        raw_callback: Optional[Callable] = None,
        *,
        progress_state: Any = None,
    ) -> None:
        self.request = request
        self.raw_callback = raw_callback
        self.progress_state = progress_state
        self.start_time = time.perf_counter()
        self.last_step_time = self.start_time

    def _forward_raw_callback(
        self,
        step: int,
        x0: Any,
        x: Any,
        total_steps: int,
        y: Any,
    ) -> None:
        if self.raw_callback is None:
            return

        accepts_sampler_shape = _callback_accepts_positional_count(self.raw_callback, 5)
        if accepts_sampler_shape is not False:
            self.raw_callback(step, x0, x, total_steps, y)
            return

        # The route layer historically supplies progressbar(task_state,
        # number, text).  It is a valid callback at the route boundary, but
        # it is not a sampler callback and must not receive five arguments.
        accepts_progressbar_shape = _callback_accepts_positional_count(self.raw_callback, 3)
        if accepts_progressbar_shape is True:
            progress_state = self.progress_state
            if progress_state is None:
                progress_state = _resolve_callback_progress_state(self.raw_callback)
            if progress_state is None:
                logger.debug(
                    "[SDXL Assembly] Skipping legacy progress callback step=%s "
                    "because no live task state is available",
                    step,
                )
                return

            current_progress = int(getattr(progress_state, "current_progress", 0) or 0)
            status_text = (
                f"Sampling step {step + 1}/{total_steps}, "
                f"image {int(getattr(self.request, 'image_index', 0)) + 1}/"
                f"{int(getattr(self.request, 'image_count', 1) or 1)} ..."
            )
            # The surrounding route owns the global percentage for nested
            # stages such as W11c.  Preserve that percentage and update only
            # the status text while forwarding the legacy callback shape.
            self.raw_callback(progress_state, current_progress, status_text)
            return

        # Preserve the historical invocation for opaque callables and report
        # any genuine callback error through the existing error boundary.
        self.raw_callback(step, x0, x, total_steps, y)

    def __call__(self, step: int, x0: Any, x: Any, total_steps: int, y: Any) -> None:
        now = time.perf_counter()
        step_duration = now - self.last_step_time
        self.last_step_time = now
        
        extra_msg = f"step={step} total_steps={total_steps} step_time={step_duration:.3f}s"
        completed_steps = int(step) + 1
        resolved_total_steps = max(int(total_steps), 1)
        should_emit_memory_snapshot = (
            completed_steps <= 1
            or completed_steps >= resolved_total_steps
            or (completed_steps % _STEP_MEMORY_TELEMETRY_INTERVAL) == 0
        )
        if should_emit_memory_snapshot:
            log_telemetry("spine_stream_step", extra_msg)
        else:
            logger.debug("[SDXL Telemetry] spine_stream_step | %s", extra_msg)
        
        if self.raw_callback is not None and _should_forward_text_only_progress(
            self.raw_callback,
            completed_steps,
            resolved_total_steps,
        ):
            try:
                self._forward_raw_callback(step, x0, x, total_steps, y)
            except resources.InterruptProcessingException:
                raise
            except Exception as e:
                logger.error(f"[SDXL Assembly] Error in raw progress callback: {e}")
