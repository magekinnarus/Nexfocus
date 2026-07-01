from __future__ import annotations

import argparse
import csv
import ctypes
import os
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

# Add workspace to path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Fooocus/Nex imports initialize the app-wide args parser. Keep probe CLI args
# for this script, but hide them from imported app modules.
_PROBE_ARGV = sys.argv[1:]
sys.argv = [sys.argv[0]]

from backend.sdxl_assembly.director import SDXLAssemblyDirector
from backend.sdxl_assembly.progress import telemetry_sink
from backend.sdxl_assembly.request_builder import build_assembly_request
from backend.sdxl_assembly.runtime_state import clear_all_caches, debug_component_cache_report


DEFAULT_PROMPT = (
    "masterpiece, best quality, a beautiful anime girl, detailed background, "
    "hyperrealistic"
)
DEFAULT_NEGATIVE_PROMPT = "ugly, blurry, deformed, low quality"
DEFAULT_CHECKPOINT = "sdxl\\base\\innovision_v10.safetensors"
DEFAULT_VAE = "sdxl_vae.safetensors"
DEFAULT_LORA = "SDXL\\base\\TWbabe.safetensors:0.5"


def _current_process_rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        pass

    if os.name != "nt":
        return None

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(
        handle,
        ctypes.byref(counters),
        counters.cb,
    )
    if not ok:
        return None
    return counters.WorkingSetSize / (1024 * 1024)


def _print_memory(label: str, *, enabled: bool) -> None:
    if not enabled:
        return

    rss_mb = _current_process_rss_mb()
    parts = [f"[memory] {label}:"]
    parts.append(f"RAM rss={rss_mb:.1f} MB" if rss_mb is not None else "RAM rss=unavailable")

    try:
        import torch

        if torch.cuda.is_available():
            parts.append(f"CUDA allocated={torch.cuda.memory_allocated() / (1024 * 1024):.1f} MB")
            parts.append(f"reserved={torch.cuda.memory_reserved() / (1024 * 1024):.1f} MB")
            parts.append(f"peak={torch.cuda.max_memory_allocated() / (1024 * 1024):.1f} MB")
    except Exception as exc:
        parts.append(f"CUDA memory=unavailable ({exc})")

    print(" ".join(parts))


def _print_component_report(label: str, *, enabled: bool) -> None:
    if not enabled:
        return
    report = debug_component_cache_report()
    print(
        f"[components] {label}: "
        f"active_spine={report['active_spine']} "
        f"active_unet_mb={report['active_unet_mb']:.1f} "
        f"active_unet_model_id={report['active_unet_model_id']} "
        f"active_unet_raw_stream={report['active_unet_raw_sequential_stream']} "
        f"active_unet_meta_construction={report['active_unet_meta_construction']} "
        f"active_unet_stream_chunk_bytes={report['active_unet_stream_chunk_bytes']} "
        f"active_unet_realized_cpu_mb={report['active_unet_realized_cpu_mb']:.1f} "
        f"text_cache_count={report['text_cache_count']} "
        f"text_cache_mb={report['text_cache_mb']:.1f}"
    )


class MemorySampler:
    def __init__(self, *, enabled: bool, interval: float = 0.25) -> None:
        self.enabled = enabled
        self.interval = max(0.05, float(interval))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_rss_mb = 0.0
        self.peak_label = ""
        self.records: list[dict[str, float | str]] = []
        self._started_at = 0.0

    def start(self, label: str) -> None:
        if not self.enabled:
            return
        self.peak_label = label
        self._started_at = time.perf_counter()
        current = _current_process_rss_mb()
        self.peak_rss_mb = float(current or 0.0)
        self._record("start")
        self._thread = threading.Thread(target=self._run, name="w03-probe-memory-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.enabled or self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._record("stop")

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            rss_mb = _current_process_rss_mb()
            if rss_mb is not None and rss_mb > self.peak_rss_mb:
                self.peak_rss_mb = float(rss_mb)
            self._record("sample", rss_mb=rss_mb)

    def mark_telemetry(self, snapshot: dict[str, object]) -> None:
        event = str(snapshot.get("event") or "unknown")
        self._record(
            f"telemetry:{event}",
            rss_mb=_as_float_or_none(snapshot.get("proc_rss_mb")),
            extra_fields={
                "telemetry_event": event,
                "telemetry_extra": str(snapshot.get("extra") or ""),
                "telemetry_ram_used_mb": _as_float_or_zero(snapshot.get("ram_used_mb")),
                "telemetry_ram_free_mb": _as_float_or_zero(snapshot.get("ram_free_mb")),
                "telemetry_vram_free_mb": _as_float_or_zero(snapshot.get("vram_free_mb")),
                "telemetry_vram_total_mb": _as_float_or_zero(snapshot.get("vram_total_mb")),
            },
        )

    def _record(
        self,
        marker: str,
        *,
        rss_mb: float | None = None,
        extra_fields: dict[str, float | str] | None = None,
    ) -> None:
        if not self.enabled:
            return
        if rss_mb is None:
            rss_mb = _current_process_rss_mb()
        record: dict[str, float | str] = {
            "elapsed_s": time.perf_counter() - self._started_at if self._started_at > 0 else 0.0,
            "marker": marker,
            "label": self.peak_label,
            "rss_mb": float(rss_mb or 0.0),
            "cuda_allocated_mb": 0.0,
            "cuda_reserved_mb": 0.0,
            "cuda_peak_allocated_mb": 0.0,
            "telemetry_event": "",
            "telemetry_extra": "",
            "telemetry_ram_used_mb": 0.0,
            "telemetry_ram_free_mb": 0.0,
            "telemetry_vram_free_mb": 0.0,
            "telemetry_vram_total_mb": 0.0,
        }
        try:
            import torch

            if torch.cuda.is_available():
                record["cuda_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
                record["cuda_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
                record["cuda_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        except Exception:
            pass
        if extra_fields:
            record.update(extra_fields)
        self.records.append(record)


def _write_memory_csv(output_dir: Path, prefix: str, chunk_mb: int, timestamp: str, records: list[dict[str, float | str]]) -> Path | None:
    if not records:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{prefix}_{chunk_mb}mb_{timestamp}_memory.csv"
    fieldnames = [
        "elapsed_s",
        "marker",
        "label",
        "rss_mb",
        "cuda_allocated_mb",
        "cuda_reserved_mb",
        "cuda_peak_allocated_mb",
        "telemetry_event",
        "telemetry_extra",
        "telemetry_ram_used_mb",
        "telemetry_ram_free_mb",
        "telemetry_vram_free_mb",
        "telemetry_vram_total_mb",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return csv_path.resolve()


def _as_float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _as_float_or_zero(value: object) -> float:
    converted = _as_float_or_none(value)
    return float(converted or 0.0)


def _print_worker_telemetry(snapshot: dict[str, object], *, enabled: bool) -> None:
    if not enabled:
        return
    event = str(snapshot.get("event") or "unknown")
    parts = [f"[telemetry] {event}:"]
    proc_rss = _as_float_or_none(snapshot.get("proc_rss_mb"))
    ram_used = _as_float_or_none(snapshot.get("ram_used_mb"))
    ram_free = _as_float_or_none(snapshot.get("ram_free_mb"))
    vram_free = _as_float_or_none(snapshot.get("vram_free_mb"))
    if proc_rss is not None:
        parts.append(f"proc_rss={proc_rss:.1f} MB")
    if ram_used is not None:
        parts.append(f"system_ram_used={ram_used:.1f} MB")
    if ram_free is not None:
        parts.append(f"system_ram_free={ram_free:.1f} MB")
    if vram_free is not None:
        parts.append(f"vram_free={vram_free:.1f} MB")
    extra = str(snapshot.get("extra") or "")
    if extra:
        parts.append(f"| {extra}")
    print(" ".join(parts))


def _parse_lora_specs(raw_loras: list[str], *, no_lora: bool) -> list[tuple[str, float]]:
    if no_lora:
        return []

    parsed: list[tuple[str, float]] = []
    for raw in raw_loras:
        value = str(raw or "").strip()
        if not value or value.lower() in {"none", "no", "false"}:
            continue

        if ":" in value:
            path, weight_text = value.rsplit(":", 1)
            weight = float(weight_text)
        else:
            path = value
            weight = 1.0

        parsed.append((path, weight))
    return parsed


def _build_task_state(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        last_stop=False,
        base_model_name=args.checkpoint,
        vae_name=args.vae,
        goals=['txt2img'],
        tiled=False,
        prepared_structural_cn_tasks={},
        prepared_contextual_cn_tasks={},
        initial_latent=None,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        cfg_scale=args.cfg,
        sampler_name=args.sampler,
        clip_skip=args.clip_skip,
        style_selections=[],
        sdxl_execution_policy=None,
        sharpness=args.sharpness,
        adaptive_cfg=args.adaptive_cfg,
        adm_scaler_positive=args.adm_positive,
        adm_scaler_negative=args.adm_negative,
        adm_scaler_end=args.adm_end,
        use_expansion=False,
        disable_intermediate_results=True,
    )


def _build_task_dict(task_state: SimpleNamespace, seed: int) -> dict[str, object]:
    return {
        'task_seed': seed,
        'task_prompt': task_state.prompt,
        'task_negative_prompt': task_state.negative_prompt,
        'positive': [task_state.prompt],
        'negative': [task_state.negative_prompt],
    }


def _print_timings(chunk_mb: int, duration: float, timings: dict[str, float], steps: int) -> None:
    print(f"{chunk_mb}MB variation completed in {duration:.3f} seconds.")
    print("Detailed timings:")
    for key, value in timings.items():
        print(f"  {key}: {value:.3f} seconds")
        if key == "unet_denoise" and steps > 0:
            print(f"  --> Denoise speed: {value / steps:.3f} seconds per step ({steps} steps)")


def _save_image(output_dir: Path, prefix: str, chunk_mb: int, timestamp: str, image) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{prefix}_{chunk_mb}mb_{timestamp}.png"
    Image.fromarray(image).save(image_path)
    return image_path


def run_probe(args: argparse.Namespace) -> list[Path]:
    print("====================================================")
    print("SDXL Assembly Streaming W03 Execution Probe Tool")
    print("====================================================")
    _print_memory("probe_start", enabled=args.memory_trace)

    if args.clear_cache_start:
        clear_all_caches()
        _print_memory("after_initial_clear", enabled=args.memory_trace)

    task_state = _build_task_state(args)
    task_dict = _build_task_dict(task_state, args.seed)
    raw_loras = args.lora if args.lora is not None else [DEFAULT_LORA]
    loras = _parse_lora_specs(raw_loras, no_lora=args.no_lora)

    print(f"Checkpoint: {task_state.base_model_name}")
    print(f"VAE override: {task_state.vae_name}")
    print(f"LoRA stack: {loras if loras else 'identity/no LoRA'}")
    print(f"Prompt: {task_state.prompt}")
    print(f"Size/steps/seed: {task_state.width}x{task_state.height}, {task_state.steps} steps, seed {args.seed}")
    print(f"Sampler/scheduler: {task_state.sampler_name}/{args.scheduler}")
    print(f"Prefetch chunks: {args.chunk_mb}")
    print(f"UNet host pinning: {'enabled' if args.pin_unet_host else 'disabled'}")
    print(f"Release warm UNet after task: {'yes' if args.release_warm_unet else 'no'}")
    print(f"Release text encoder after task: {'yes' if args.release_text_encoder else 'no'}")
    _print_memory("before_request_build", enabled=args.memory_trace)

    base_request = build_assembly_request(
        task_state=task_state,
        task_dict=task_dict,
        current_task_id=0,
        total_count=1,
        all_steps=task_state.steps,
        preparation_steps=0,
        denoising_strength=args.denoising_strength,
        final_scheduler_name=args.scheduler,
        loras=loras,
    )
    base_request = replace(
        base_request,
        metadata={
            **base_request.metadata,
            "pin_unet_host": bool(args.pin_unet_host),
            "release_warm_unet_after_task": bool(args.release_warm_unet),
            "release_text_encoder_after_task": bool(args.release_text_encoder),
        },
    )
    _print_memory("after_request_build", enabled=args.memory_trace)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_paths: list[Path] = []

    try:
        for chunk_mb in args.chunk_mb:
            request = replace(base_request, prefetch_chunk_mb=int(chunk_mb))
            print(f"\n--- Running {request.prefetch_chunk_mb}MB prefetch chunk variation ---")
            _print_memory(f"{request.prefetch_chunk_mb}mb_before_select", enabled=args.memory_trace)

            assembly = SDXLAssemblyDirector.select_assembly(request)
            _print_memory(f"{request.prefetch_chunk_mb}mb_after_select", enabled=args.memory_trace)
            started = time.perf_counter()
            sampler = MemorySampler(enabled=args.memory_trace, interval=args.memory_sample_interval)
            sampler.start(f"{request.prefetch_chunk_mb}mb_execute")
            callback = None
            if args.memory_step_trace:
                def callback(step, _x0, _x, total_steps, _y):
                    _print_memory(
                        f"{request.prefetch_chunk_mb}mb_step_{step + 1}_of_{total_steps}",
                        enabled=args.memory_trace,
                    )
            try:
                def worker_telemetry_sink(snapshot: dict[str, object]) -> None:
                    sampler.mark_telemetry(snapshot)
                    _print_worker_telemetry(snapshot, enabled=args.worker_telemetry)

                with telemetry_sink(worker_telemetry_sink):
                    result = assembly.execute(request, callback=callback)
                _print_memory(f"{request.prefetch_chunk_mb}mb_after_execute", enabled=args.memory_trace)
                _print_component_report(f"{request.prefetch_chunk_mb}mb_after_execute", enabled=args.component_trace)
            finally:
                sampler.stop()
                assembly.close()
                _print_memory(f"{request.prefetch_chunk_mb}mb_after_close", enabled=args.memory_trace)
                _print_component_report(f"{request.prefetch_chunk_mb}mb_after_close", enabled=args.component_trace)
                if args.memory_trace:
                    print(
                        f"[memory] {request.prefetch_chunk_mb}mb_execute_peak: "
                        f"RAM rss={sampler.peak_rss_mb:.1f} MB"
                    )
                    if args.memory_csv:
                        csv_path = _write_memory_csv(
                            Path(args.output_dir),
                            args.output_prefix,
                            request.prefetch_chunk_mb,
                            timestamp,
                            sampler.records,
                        )
                        if csv_path is not None:
                            print(f"[memory] timeline csv: {csv_path}")

            duration = time.perf_counter() - started
            _print_timings(request.prefetch_chunk_mb, duration, result.timings, request.steps)

            output_path = _save_image(
                Path(args.output_dir),
                args.output_prefix,
                request.prefetch_chunk_mb,
                timestamp,
                result.output_image,
            )
            output_paths.append(output_path.resolve())
            print(f"Saved generated image to: {output_paths[-1]}")
            _print_memory(f"{request.prefetch_chunk_mb}mb_after_save", enabled=args.memory_trace)
    finally:
        if args.clear_cache_end:
            clear_all_caches()
            _print_memory("after_final_clear", enabled=args.memory_trace)
            _print_component_report("after_final_clear", enabled=args.component_trace)

    return output_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the W03 SDXL assembly direct streaming probe without UI route cutover."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--vae", default=DEFAULT_VAE)
    parser.add_argument("--lora", action="append", default=None, help="LoRA path or path:weight. Repeatable.")
    parser.add_argument("--no-lora", action="store_true", help="Ignore all LoRA arguments and run identity/no-LoRA.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=7.0)
    parser.add_argument("--sampler", default="dpmpp_2m")
    parser.add_argument("--scheduler", default="beta")
    parser.add_argument("--denoising-strength", type=float, default=1.0)
    parser.add_argument("--clip-skip", type=int, default=2)
    parser.add_argument("--sharpness", type=float, default=2.0)
    parser.add_argument("--adaptive-cfg", type=float, default=7.0)
    parser.add_argument("--adm-positive", type=float, default=1.5)
    parser.add_argument("--adm-negative", type=float, default=0.8)
    parser.add_argument("--adm-end", type=float, default=0.3)
    parser.add_argument("--chunk-mb", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--output-prefix", default="w03_probe_output")
    parser.add_argument("--pin-unet-host", action="store_true", help="Opt into full CUDA pinned-host UNet tensors.")
    parser.add_argument("--release-warm-unet", action="store_true", help="Release warm UNet at assembly close.")
    parser.add_argument("--release-text-encoder", action="store_true", help="Release cached CLIP text encoder at assembly close.")
    parser.add_argument("--no-clear-cache-start", dest="clear_cache_start", action="store_false")
    parser.add_argument("--no-clear-cache-end", dest="clear_cache_end", action="store_false")
    parser.add_argument("--no-memory-trace", dest="memory_trace", action="store_false")
    parser.add_argument("--memory-step-trace", action="store_true")
    parser.add_argument("--no-memory-csv", dest="memory_csv", action="store_false")
    parser.add_argument("--no-component-trace", dest="component_trace", action="store_false")
    parser.add_argument("--no-worker-telemetry", dest="worker_telemetry", action="store_false")
    parser.add_argument("--memory-sample-interval", type=float, default=0.25)
    parser.set_defaults(
        clear_cache_start=True,
        clear_cache_end=True,
        memory_trace=True,
        memory_csv=True,
        component_trace=True,
        worker_telemetry=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    if argv is None:
        argv = _PROBE_ARGV
    args = parser.parse_args(argv)
    run_probe(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Generation failed with error: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
