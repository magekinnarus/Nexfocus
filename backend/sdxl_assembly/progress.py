import logging
import time
import psutil
import torch
from contextlib import contextmanager
from threading import RLock
from typing import Any, Callable, Optional
from backend.sdxl_assembly.contracts import SDXLAssemblyRequest

logger = logging.getLogger(__name__)
_TELEMETRY_SINKS: list[Callable[[dict[str, Any]], None]] = []
_TELEMETRY_SINKS_LOCK = RLock()


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
    def __init__(self, request: SDXLAssemblyRequest, raw_callback: Optional[Callable] = None) -> None:
        self.request = request
        self.raw_callback = raw_callback
        self.start_time = time.perf_counter()
        self.last_step_time = self.start_time

    def __call__(self, step: int, x0: Any, x: Any, total_steps: int, y: Any) -> None:
        now = time.perf_counter()
        step_duration = now - self.last_step_time
        self.last_step_time = now
        
        extra_msg = f"step={step} total_steps={total_steps} step_time={step_duration:.3f}s"
        log_telemetry("spine_stream_step", extra_msg)
        
        if self.raw_callback is not None:
            try:
                self.raw_callback(step, x0, x, total_steps, y)
            except Exception as e:
                logger.error(f"[SDXL Assembly] Error in raw progress callback: {e}")
