"""Isolated Flux prompt-conditioning artifact subprocess.

This is the runtime-owned replacement for the retired maintainer generator
under ``tools/``.  It intentionally remains a single-shot subprocess so the
large text encoder and its host allocations die with the worker process.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CLIP_ROOT = Path(r"D:\AI\Imagine\models\clip")
DEFAULT_CLIP_L_PATH = DEFAULT_CLIP_ROOT / "clip_l.safetensors"
DEFAULT_FP16_T5_PATH = DEFAULT_CLIP_ROOT / "t5xxl_fp16.safetensors"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


class MemorySampler:
    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = psutil.Process() if psutil is not None else None
        self.peak_rss_bytes = 0
        self.peak_vms_bytes = 0
        self.peak_pagefile_bytes = 0
        self.peak_private_bytes = 0
        self.peak_uss_bytes = 0

    def __enter__(self) -> "MemorySampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set() and self._process is not None:
            try:
                info = self._process.memory_info()
                self.peak_rss_bytes = max(self.peak_rss_bytes, int(getattr(info, "rss", 0)))
                self.peak_vms_bytes = max(self.peak_vms_bytes, int(getattr(info, "vms", 0)))
                self.peak_pagefile_bytes = max(self.peak_pagefile_bytes, int(getattr(info, "pagefile", 0)))
                self.peak_private_bytes = max(
                    self.peak_private_bytes,
                    int(getattr(info, "private", getattr(info, "private_bytes", 0))),
                )
            except Exception:
                pass
            try:
                full = self._process.memory_full_info()
                self.peak_uss_bytes = max(self.peak_uss_bytes, int(getattr(full, "uss", 0)))
            except Exception:
                pass
            time.sleep(self.interval_s)

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def _capture_process_memory_snapshot() -> dict[str, float] | None:
    if psutil is None:
        return None
    try:
        process = psutil.Process()
        info = process.memory_info()
        try:
            full = process.memory_full_info()
        except Exception:
            full = None
        return {
            "rss_mb": float(getattr(info, "rss", 0)) / (1024 * 1024),
            "vms_mb": float(getattr(info, "vms", 0)) / (1024 * 1024),
            "pagefile_mb": float(getattr(info, "pagefile", 0)) / (1024 * 1024),
            "private_mb": float(getattr(info, "private", getattr(info, "private_bytes", 0))) / (1024 * 1024),
            "uss_mb": float(getattr(full, "uss", 0)) / (1024 * 1024) if full is not None else 0.0,
        }
    except Exception:
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-shot Flux fp16 streaming prompt-conditioning artifact generator."
    )
    parser.add_argument("--prompt", required=True, help="Prompt text to encode.")
    parser.add_argument("--output", required=True, help="Output .pt artifact path.")
    parser.add_argument("--clip-l", default=str(DEFAULT_CLIP_L_PATH), help="Path to Flux CLIP-L weights.")
    parser.add_argument("--fp16-t5", default=str(DEFAULT_FP16_T5_PATH), help="Path to the fp16 T5 safetensors weights.")
    parser.add_argument("--embedding-directory", default=None, help="Optional embedding directory.")
    parser.add_argument("--metrics-json", default=None, help="Optional metrics JSON output path.")
    parser.add_argument(
        "--disk-paged-t5-gc-interval",
        type=int,
        default=None,
        help="Optional fixed disk-paged T5 GC interval override.",
    )
    parser.add_argument("--traceback", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    # Backend imports initialize the application argument manager. Do not let
    # this worker's private CLI flags leak into that independent parser.
    sys.argv = [sys.argv[0]]
    prompt_text = str(args.prompt or "").strip()
    if not prompt_text:
        raise ValueError("--prompt must be a non-empty string.")
    if Path(args.fp16_t5).suffix.lower() != ".safetensors":
        raise ValueError(
            "--fp16-t5 must point to a .safetensors checkpoint for disk-paged worker execution."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = (
        Path(args.metrics_json)
        if args.metrics_json
        else output_path.with_suffix(output_path.suffix + ".metrics.json")
    )

    from backend import resources
    from backend.flux_fill_v3.t5_worker import (
        load_flux_prompt_text_encoder,
        save_flux_prompt_conditioning_cache,
    )

    gc.collect()
    try:
        resources.soft_empty_cache(force=True)
    except Exception:
        pass

    encoder = None
    try:
        with MemorySampler() as memory:
            phase_snapshots: list[dict[str, Any]] = []
            total_start = time.perf_counter()
            phase_snapshots.append(
                {"phase": "pre_load", "elapsed_wall": 0.0, "memory": _capture_process_memory_snapshot()}
            )
            load_start = time.perf_counter()
            encoder = load_flux_prompt_text_encoder(
                clip_l_path=Path(args.clip_l),
                t5_path=Path(args.fp16_t5),
                embedding_directory=Path(args.embedding_directory) if args.embedding_directory else None,
                t5_loader_policy="stream_safetensors_runtime",
                low_ram_gc=True,
                disk_paged_t5_gc_interval=args.disk_paged_t5_gc_interval,
            )
            model_load_wall = time.perf_counter() - load_start
            loader_metadata = dict(getattr(encoder, "_nex_load_metadata", {}) or {})
            phase_snapshots.append(
                {
                    "phase": "post_load",
                    "elapsed_wall": time.perf_counter() - total_start,
                    "memory": _capture_process_memory_snapshot(),
                }
            )

            encode_start = time.perf_counter()
            encode_cpu_start = time.process_time()
            cross_attn, pooled_output = encoder.encode(prompt_text)
            encode_wall = time.perf_counter() - encode_start
            encode_cpu_proc = time.process_time() - encode_cpu_start
            phase_snapshots.append(
                {
                    "phase": "post_encode",
                    "elapsed_wall": time.perf_counter() - total_start,
                    "memory": _capture_process_memory_snapshot(),
                }
            )

            save_start = time.perf_counter()
            conditioning = save_flux_prompt_conditioning_cache(
                output_path,
                cross_attn=cross_attn,
                pooled_output=pooled_output,
                metadata={
                    "prompt": prompt_text,
                    "clip_l_path": str(args.clip_l),
                    "t5_path": str(args.fp16_t5),
                    "t5_format": "safetensors",
                    "generator": "backend/flux_fill_v3/prompt_conditioning_artifact_worker.py",
                    "conditioning_kind": "prompt",
                    "transport": "pt_cache",
                    "text_encoder_resident": False,
                    "t5_loader_policy": "stream_safetensors_runtime",
                    "low_ram_gc": True,
                    "disk_paged_t5_gc_interval": args.disk_paged_t5_gc_interval,
                    "posture": "disk_paged_t5",
                    "loader_metadata": loader_metadata,
                    "phase_snapshots": phase_snapshots,
                },
            )
            save_wall = time.perf_counter() - save_start
            total_wall = time.perf_counter() - total_start
            phase_snapshots.append(
                {"phase": "post_save", "elapsed_wall": total_wall, "memory": _capture_process_memory_snapshot()}
            )

        payload = {
            "status": "ok",
            "prompt": prompt_text,
            "output_path": str(output_path),
            "metrics_path": str(metrics_path),
            "total_wall": total_wall,
            "model_load_wall": model_load_wall,
            "encode_wall": encode_wall,
            "encode_cpu_proc": encode_cpu_proc,
            "save_wall": save_wall,
            "peak_rss_mb": float(memory.peak_rss_bytes) / (1024 * 1024),
            "peak_vms_mb": float(memory.peak_vms_bytes) / (1024 * 1024),
            "peak_pagefile_mb": float(memory.peak_pagefile_bytes) / (1024 * 1024),
            "peak_private_mb": float(memory.peak_private_bytes) / (1024 * 1024),
            "peak_uss_mb": float(memory.peak_uss_bytes) / (1024 * 1024),
            "conditioning_shape": list(conditioning.cross_attn.shape),
            "pooled_shape": list(conditioning.pooled_output.shape),
            "conditioning_dtype": str(conditioning.cross_attn.dtype),
            "pooled_dtype": str(conditioning.pooled_output.dtype),
            "loader_metadata": loader_metadata,
            "low_ram_gc": True,
            "disk_paged_t5_gc_interval": args.disk_paged_t5_gc_interval,
            "t5_loader_policy": "stream_safetensors_runtime",
            "posture_label": "disk_paged_t5",
            "phase_snapshots": phase_snapshots,
            "process_isolated": True,
            "pid": int(os.getpid()),
        }
        metrics_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        print(json.dumps(payload, default=_json_default))
        return 0
    except Exception as exc:
        error = {
            "status": "error",
            "error": {"type": exc.__class__.__name__, "message": str(exc)},
            "output_path": str(output_path),
            "metrics_path": str(metrics_path),
        }
        if args.traceback:
            error["traceback"] = traceback.format_exc()
        metrics_path.write_text(json.dumps(error, indent=2, default=_json_default), encoding="utf-8")
        print(json.dumps(error, default=_json_default))
        return 1
    finally:
        try:
            if encoder is not None:
                try:
                    resources.eject_model(encoder.patcher)
                except Exception:
                    detach = getattr(encoder.patcher, "detach", None)
                    if callable(detach):
                        detach()
        except Exception:
            pass
        try:
            del encoder
        except Exception:
            pass
        gc.collect()
        try:
            resources.soft_empty_cache(force=True)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
