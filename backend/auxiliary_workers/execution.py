from __future__ import annotations

from contextlib import contextmanager
from threading import Lock, RLock
from typing import Iterator

from backend.auxiliary_workers.telemetry import log_auxiliary_telemetry


_EXECUTION_LOCK = Lock()
_STATE_LOCK = RLock()
_ACTIVE_AUXILIARY_WORKER: str | None = None


def active_auxiliary_worker() -> str | None:
    with _STATE_LOCK:
        return _ACTIVE_AUXILIARY_WORKER


@contextmanager
def auxiliary_execution(worker_name: str) -> Iterator[None]:
    """Serialize one posture-agnostic auxiliary GPU execution window.

    The selected assembly must already be in its auxiliary-ready state. This
    lease deliberately performs no model eviction, cache sweep, hardware-tier
    selection, or posture substitution.
    """
    global _ACTIVE_AUXILIARY_WORKER

    resolved_name = str(worker_name or "auxiliary_worker")
    log_auxiliary_telemetry("auxiliary_admission_requested", f"worker={resolved_name}")
    with _EXECUTION_LOCK:
        with _STATE_LOCK:
            _ACTIVE_AUXILIARY_WORKER = resolved_name
        log_auxiliary_telemetry("auxiliary_admission_granted", f"worker={resolved_name}")
        try:
            yield
        except BaseException as exc:
            log_auxiliary_telemetry(
                "auxiliary_execution_failed",
                f"worker={resolved_name} error_type={type(exc).__name__}",
            )
            raise
        finally:
            with _STATE_LOCK:
                _ACTIVE_AUXILIARY_WORKER = None
            log_auxiliary_telemetry("auxiliary_execution_released", f"worker={resolved_name}")
