from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from threading import RLock
from typing import Any, Callable

import psutil
import torch


logger = logging.getLogger(__name__)
_TELEMETRY_SINKS: list[Callable[[dict[str, Any]], None]] = []
_TELEMETRY_SINKS_LOCK = RLock()


def add_telemetry_sink(sink: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
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


def _emit_snapshot(snapshot: dict[str, Any]) -> None:
    with _TELEMETRY_SINKS_LOCK:
        sinks = list(_TELEMETRY_SINKS)
    for sink in sinks:
        try:
            sink(dict(snapshot))
        except Exception:
            logger.debug("[Auxiliary Telemetry] Telemetry sink failed.", exc_info=True)


def log_auxiliary_telemetry(event: str, extra_msg: str = "") -> None:
    """Emit process-aware telemetry without coupling auxiliary workers to SDXL."""
    try:
        virtual_mem = psutil.virtual_memory()
        ram_total = float(virtual_mem.total) / (1024 * 1024)
        ram_free = float(virtual_mem.available) / (1024 * 1024)
        ram_used = ram_total - ram_free
        proc_rss = float(psutil.Process().memory_info().rss) / (1024 * 1024)

        vram_total = 0.0
        vram_free = 0.0
        if torch.cuda.is_available():
            try:
                vram_free_bytes, vram_total_bytes = torch.cuda.mem_get_info()
                vram_total = float(vram_total_bytes) / (1024 * 1024)
                vram_free = float(vram_free_bytes) / (1024 * 1024)
            except Exception:
                pass

        metrics = (
            f"ram_total={ram_total:.1f}MB "
            f"ram_free={ram_free:.1f}MB "
            f"ram_used={ram_used:.1f}MB "
            f"vram_total={vram_total:.1f}MB "
            f"vram_free={vram_free:.1f}MB "
            f"proc_rss={proc_rss:.1f}MB"
        )
        message = f"[Auxiliary Telemetry] {event} | {metrics}"
        if extra_msg:
            message += f" | {extra_msg}"
        logger.debug(message)
        _emit_snapshot(
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
    except Exception:
        logger.debug("[Auxiliary Telemetry] Failed to gather memory telemetry.", exc_info=True)
