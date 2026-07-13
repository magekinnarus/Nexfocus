#!/usr/bin/env python3
"""Validation probe for the SDXL resident UNet spine and GPU LoRA lifecycle.

The real-mode path is intended for Colab Free/T4 W12a field evidence.  It
performs a cold resident safetensors load, exact-key warm reuse, UNet-side LoRA
reload/prepatch, optional second LoRA stack change, LoRA removal, and explicit
release while recording memory/device telemetry.

Use ``--mock`` for lightweight local logic checks when real SDXL assets are not
available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keep this probe out of the UI argparse/model-load path.
_fake_args = SimpleNamespace(
    colab=Path("/content").exists(),
    preset="",
    output_path="",
    temp_path="",
    skip_model_load=True,
    disable_metadata=True,
)
sys.modules.setdefault(
    "args_manager",
    SimpleNamespace(
        args=_fake_args,
        args_parser=SimpleNamespace(args=_fake_args, parser=SimpleNamespace()),
    ),
)

import torch

try:
    import psutil
except Exception:  # pragma: no cover - optional runtime dependency
    psutil = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate_sdxl_resident_spine")


def mock_environment() -> None:
    """Mock heavy loader/compiler functions for local diagnostic runs."""
    logger.info("Mocking environment for diagnostic-only rehearsal...")

    import backend.gpu_compiler as gpu_compiler
    import backend.loader as loader
    import backend.lora as backend_lora

    class FakeModel:
        def __init__(self) -> None:
            self.parameters_dict = {"param1": torch.nn.Parameter(torch.empty(2, 2, device="meta"))}
            self.buffers_dict: dict[str, torch.Tensor] = {}
            self.current_weight_patches_uuid = None
            self.model_loaded_weight_memory = 0
            self.model_lowvram = False
            self.lowvram_patch_counter = 0
            self.device = torch.device("cpu")
            self.model_sampling = SimpleNamespace(
                sigma_max=1.0,
                noise_scaling=lambda sigma, noise, latent, max_denoise=False: noise,
                inverse_noise_scaling=lambda sigma, samples: samples,
            )

        def state_dict(self):
            return {"param1": self.parameters_dict["param1"]}

        def named_modules(self):
            return [("", self)]

        def named_parameters(self, recurse=True):
            return [("param1", self.parameters_dict["param1"])]

        def named_buffers(self, recurse=True):
            return []

        def requires_grad_(self, val):
            return self

        def eval(self):
            return self

    class FakePatcher:
        def __init__(self, name: str) -> None:
            self.name = name
            self.model = FakeModel()
            self.patches: dict[str, Any] = {}
            self.weight_wrapper_patches: dict[str, Any] = {}
            self.backup: dict[str, Any] = {}
            self.object_patches_backup: dict[str, Any] = {}
            self.runtime_release_to_meta = True
            self.runtime_reload = None
            self.load_device = torch.device("cpu")
            self.offload_device = torch.device("cpu")
            self.current_device = torch.device("cpu")
            self.model_options: dict[str, Any] = {
                "sdxl_assembly_loader": {
                    "direct_safetensors_load": True,
                    "raw_sequential_stream": True,
                    "meta_construction": True,
                    "stream_chunk_bytes": None,
                    "realized_cpu_bytes": 0,
                }
            }

        def model_size(self):
            return 1024

        def add_patches(self, patches, weight):
            for key, value in patches.items():
                self.patches[key] = [(weight, value, 1.0, None, lambda x: x)]
            return list(patches.keys())

        def patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False):
            self.current_device = torch.device(device_to or "cpu")
            return self.model

        def detach(self):
            self.current_device = self.offload_device

        def current_loaded_device(self):
            return self.current_device

    def fake_stream_load(ckpt_path, **kwargs):
        unet = FakePatcher("unet")
        load_device = torch.device(kwargs.get("load_device") or "cpu")
        unet.load_device = load_device
        unet.offload_device = torch.device(kwargs.get("offload_device") or load_device)

        def reload_func(model, device):
            model.parameters_dict["param1"] = torch.nn.Parameter(torch.zeros(2, 2, device=device))
            model.device = torch.device(device)

        unet.runtime_reload = reload_func
        unet.runtime_reload(unet.model, load_device)
        return unet

    def fake_gpu_compile(patcher, **kwargs):
        patch_count = len(getattr(patcher, "patches", {}) or {})
        patcher.patches = {}
        return {
            "status": "compiled" if patch_count else "noop",
            "patch_count": patch_count,
            "materialized_patch_keys": patch_count,
            "host_pinned_bytes": 0,
            "cuda_temp_peak_bytes": 0,
            "cleared_patch_count": patch_count,
        }

    loader._stream_load_sdxl_unet_from_checkpoint = fake_stream_load
    gpu_compiler.GpuArtifactCompiler.compile_patcher = fake_gpu_compile
    backend_lora.load_lora = lambda *args, **kwargs: {"param1": torch.ones(2, 2)}
    backend_lora.model_lora_keys_unet = lambda m: {"param1": "param1"}

    import backend.sdxl_assembly.gpu_lora_worker as gpu_lora_worker

    gpu_lora_worker.SafeOpenHeaderOnly = lambda path: {"path": path}


def _sha256_file(path: Path, *, fast_identity: bool = False) -> str:
    stat = path.stat()
    if fast_identity:
        return f"fast:{path.name}:{stat.st_size}:{stat.st_mtime_ns}"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path, *, fast_identity: bool = False):
    from backend.sdxl_assembly.contracts import ResolvedFileIdentity

    path = Path(path)
    if path.exists():
        stat = path.stat()
        return ResolvedFileIdentity(
            path=path,
            sha256=_sha256_file(path, fast_identity=fast_identity),
            size_bytes=int(stat.st_size),
            modified_ns=int(stat.st_mtime_ns),
        )
    return ResolvedFileIdentity(path=path, sha256=f"missing:{path.name}", size_bytes=1, modified_ns=1)


def _memory_snapshot(label: str, device: torch.device) -> dict[str, Any]:
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass

    snapshot: dict[str, Any] = {
        "label": label,
        "timestamp_s": time.time(),
        "device": str(device),
    }
    if psutil is not None:
        try:
            process = psutil.Process(os.getpid())
            vm = psutil.virtual_memory()
            snapshot.update(
                {
                    "proc_rss_bytes": int(process.memory_info().rss),
                    "ram_total_bytes": int(vm.total),
                    "ram_available_bytes": int(vm.available),
                }
            )
        except Exception as exc:
            snapshot["psutil_error"] = f"{type(exc).__name__}: {exc}"

    if device.type == "cuda" and torch.cuda.is_available():
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            snapshot.update(
                {
                    "cuda_name": torch.cuda.get_device_name(device),
                    "cuda_total_bytes": int(total_bytes),
                    "cuda_free_bytes": int(free_bytes),
                    "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                    "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                    "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                }
            )
        except Exception as exc:
            snapshot["cuda_error"] = f"{type(exc).__name__}: {exc}"
    return snapshot


def _clean_shadow_bytes(unet: Any) -> int:
    model = getattr(unet, "model", None)
    clean_source = getattr(model, "_nex_clean_unet_source", None)
    if clean_source is None:
        return 0
    if isinstance(clean_source, dict):
        return int(
            sum(
                tensor.numel() * tensor.element_size()
                for tensor in clean_source.values()
                if isinstance(tensor, torch.Tensor)
            )
        )
    if isinstance(clean_source, torch.Tensor):
        return int(clean_source.numel() * clean_source.element_size())
    return 0


def _state_record(label: str, spine: Any, device: torch.device, events: list[dict[str, Any]]) -> dict[str, Any]:
    from backend.sdxl_assembly.runtime_state import debug_component_cache_report

    unet = getattr(spine, "unet", None) if spine is not None else None
    lora_worker = getattr(spine, "lora_worker", None) if spine is not None else None
    return {
        "label": label,
        "spine_id": id(spine) if spine is not None else None,
        "unet_id": id(getattr(unet, "model", unet)) if unet is not None else None,
        "memory": _memory_snapshot(label, device),
        "cache_report": debug_component_cache_report(),
        "clean_shadow_bytes": _clean_shadow_bytes(unet),
        "lora_compile_metrics": dict(getattr(lora_worker, "last_compile_metrics", {}) or {}),
        "telemetry_event_count": len(events),
    }


def _event_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        name = str(event.get("event"))
        counts[name] = counts.get(name, 0) + 1
    return counts


def _make_request(
    *,
    checkpoint_identity: Any,
    lora_specs: tuple[Any, ...],
    lora_stack_hash: str,
    device: str,
):
    from backend.sdxl_assembly.contracts import (
        LoraPatchPostureKind,
        SDXLAssemblyRequest,
        TextEncoderPostureKind,
        UNetPostureKind,
        VAEPostureKind,
    )

    return SDXLAssemblyRequest(
        request_id=f"w12a_probe_{lora_stack_hash or 'clean'}",
        route_id="txt2img_assembly",
        image_index=0,
        image_count=1,
        checkpoint=checkpoint_identity,
        vae=None,
        model_variant_key="sdxl",
        prompt="resident spine validation",
        negative_prompt="",
        positive_texts=("resident spine validation",),
        negative_texts=("",),
        width=64,
        height=64,
        steps=1,
        cfg=1.0,
        sampler="euler",
        scheduler="karras",
        seed=123,
        device=device,
        unet_posture=UNetPostureKind.RESIDENT,
        clip_posture=TextEncoderPostureKind.CPU_PINNED,
        vae_posture=VAEPostureKind.TRANSIENT,
        lora_posture=LoraPatchPostureKind.RESIDENT,
        lora_specs=lora_specs,
        lora_stack_hash=lora_stack_hash,
    )


def _make_lora_spec(path: Path, *, weight: float, fast_identity: bool):
    from backend.sdxl_assembly.contracts import SDXLLoraSpec

    return SDXLLoraSpec(
        file_identity=_identity(path, fast_identity=fast_identity),
        unet_weight=float(weight),
        clip_weight=0.0,
        enabled=True,
    )


def _configured_main_lora_paths(args: argparse.Namespace) -> list[Path]:
    paths = [
        Path(path)
        for path in (getattr(args, "lora_paths", None) or getattr(args, "loras", None) or [])
        if path
    ]
    if paths:
        return paths

    legacy_lora_a = getattr(args, "lora_a", None)
    return [Path(legacy_lora_a)] if legacy_lora_a else []


def _configured_main_lora_weights(args: argparse.Namespace, count: int) -> list[float]:
    weights = [float(value) for value in (getattr(args, "lora_weights", None) or [])]
    if weights and len(weights) != count:
        raise ValueError(f"--lora-weight count ({len(weights)}) must match --lora count ({count}).")
    if weights:
        return weights
    if count == 1 and getattr(args, "lora_a", None):
        return [float(getattr(args, "lora_a_weight", 1.0))]
    return [1.0 for _ in range(count)]


def _lora_stack_hash(label: str, specs: tuple[Any, ...]) -> str:
    if not specs:
        return "clean"
    digest = hashlib.sha256()
    digest.update(label.encode("utf-8"))
    for spec in specs:
        identity = spec.file_identity
        digest.update(str(identity.path).encode("utf-8"))
        digest.update(str(identity.sha256).encode("utf-8"))
        digest.update(str(identity.size_bytes).encode("utf-8"))
        digest.update(str(spec.unet_weight).encode("utf-8"))
    return f"{label}_{digest.hexdigest()[:16]}"


def run_validation(args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    try:
        main_lora_paths = _configured_main_lora_paths(args)
        main_lora_weights = _configured_main_lora_weights(args, len(main_lora_paths))
    except ValueError as exc:
        return False, {"status": "failed", "error": str(exc)}

    alternate_lora_path = Path(getattr(args, "lora_b", "")) if getattr(args, "lora_b", None) else None

    if args.mock:
        mock_environment()
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            return False, {"status": "failed", "error": "CUDA is not available; use --mock for local diagnostics."}
        if not args.checkpoint:
            return False, {"status": "failed", "error": "--checkpoint is required for real W12a field validation."}
        if not main_lora_paths:
            return False, {"status": "failed", "error": "At least one --lora or --lora-a path is required for real W12a GPU LoRA evidence."}
        preflight_assets = [("checkpoint", Path(args.checkpoint))]
        preflight_assets.extend((f"lora_{index}", path) for index, path in enumerate(main_lora_paths, start=1))
        if alternate_lora_path is not None:
            preflight_assets.append(("lora_b", alternate_lora_path))
        for label, path in preflight_assets:
            if path.suffix.lower() != ".safetensors":
                return False, {"status": "failed", "error": f"{label} must be a .safetensors file: {path}"}
            if not path.is_file():
                return False, {"status": "failed", "error": f"{label} file does not exist: {path}"}
        device = torch.device(args.device)

    from backend.sdxl_assembly.contracts import SDXLLoraSpec
    from backend.sdxl_assembly.lifecycle_coordinator import LifecycleChange, release_for_changes
    from backend.sdxl_assembly.progress import telemetry_sink
    from backend.sdxl_assembly.runtime_state import (
        acquire_active_sdxl_resident_spine,
        clear_all_caches,
        debug_component_cache_report,
        release_active_sdxl_resident_spine,
    )

    checkpoint_path = Path(args.checkpoint or "checkpoint.safetensors")
    checkpoint_identity = _identity(checkpoint_path, fast_identity=args.fast_identity)

    if args.mock and not main_lora_paths:
        main_lora_specs = (SDXLLoraSpec(
            file_identity=_identity(Path("mock_lora_a.safetensors"), fast_identity=True),
            unet_weight=1.0,
            clip_weight=0.0,
        ),)
        main_lora_paths_for_summary = [Path("mock_lora_a.safetensors")]
        main_lora_weights_for_summary = [1.0]
    else:
        main_lora_specs = tuple(
            _make_lora_spec(path, weight=weight, fast_identity=args.fast_identity)
            for path, weight in zip(main_lora_paths, main_lora_weights)
        )
        main_lora_paths_for_summary = main_lora_paths
        main_lora_weights_for_summary = main_lora_weights

    lora_b = None
    if alternate_lora_path is not None:
        lora_b = _make_lora_spec(alternate_lora_path, weight=args.lora_b_weight, fast_identity=args.fast_identity)

    events: list[dict[str, Any]] = []
    records: dict[str, Any] = {}
    release_result = None

    clear_all_caches(reason="w12a_probe_start")
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    with telemetry_sink(lambda snapshot: events.append(dict(snapshot))):
        req_clean = _make_request(
            checkpoint_identity=checkpoint_identity,
            lora_specs=(),
            lora_stack_hash="clean",
            device=str(device),
        )

        logger.info("Step 1: cold resident safetensors load")
        spine_clean, cold_reused = acquire_active_sdxl_resident_spine(req_clean)
        spine_clean.start()
        records["cold_clean"] = _state_record("cold_clean", spine_clean, device, events)
        records["cold_clean"]["reused"] = bool(cold_reused)

        logger.info("Step 2: exact-key warm reuse")
        spine_clean_warm, warm_reused = acquire_active_sdxl_resident_spine(req_clean)
        spine_clean_warm.start()
        records["warm_clean"] = _state_record("warm_clean", spine_clean_warm, device, events)
        records["warm_clean"]["reused"] = bool(warm_reused)

        logger.info("Step 3: UNet LoRA stack change through clean reload + GPU prepatch")
        req_lora_a = _make_request(
            checkpoint_identity=checkpoint_identity,
            lora_specs=main_lora_specs,
            lora_stack_hash=_lora_stack_hash("lora_stack", main_lora_specs),
            device=str(device),
        )
        spine_lora_a, lora_a_reused = acquire_active_sdxl_resident_spine(req_lora_a)
        spine_lora_a.start()
        records["lora_a"] = _state_record("lora_a", spine_lora_a, device, events)
        records["lora_a"]["reused"] = bool(lora_a_reused)

        logger.info("Step 4: same LoRA stack warm reuse")
        spine_lora_a_warm, lora_a_warm_reused = acquire_active_sdxl_resident_spine(req_lora_a)
        spine_lora_a_warm.start()
        records["warm_lora_a"] = _state_record("warm_lora_a", spine_lora_a_warm, device, events)
        records["warm_lora_a"]["reused"] = bool(lora_a_warm_reused)

        if lora_b is not None:
            logger.info("Step 5: optional second UNet LoRA stack change")
            req_lora_b = _make_request(
                checkpoint_identity=checkpoint_identity,
                lora_specs=(lora_b,),
                lora_stack_hash=_lora_stack_hash("lora_b", (lora_b,)),
                device=str(device),
            )
            spine_lora_b, lora_b_reused = acquire_active_sdxl_resident_spine(req_lora_b)
            spine_lora_b.start()
            records["lora_b"] = _state_record("lora_b", spine_lora_b, device, events)
            records["lora_b"]["reused"] = bool(lora_b_reused)

        logger.info("Step 6: LoRA removal back to clean identity")
        spine_removed, removal_reused = acquire_active_sdxl_resident_spine(req_clean)
        spine_removed.start()
        records["lora_removed"] = _state_record("lora_removed", spine_removed, device, events)
        records["lora_removed"]["reused"] = bool(removal_reused)

        logger.info("Step 7: explicit resident release")
        before_release_memory = _memory_snapshot("before_release", device)
        released = release_active_sdxl_resident_spine(reason="w12a_probe_release")
        release_for_changes([LifecycleChange.CHECKPOINT_CHANGE], reason="w12a_probe_release_regression")
        after_release_memory = _memory_snapshot("after_release", device)
        release_result = {
            "released": bool(released),
            "before": before_release_memory,
            "after": after_release_memory,
            "cache_report_after": debug_component_cache_report(),
        }

    counts = _event_counts(events)
    lora_compile_metrics = records["lora_a"].get("lora_compile_metrics", {})
    lora_patch_count = int(lora_compile_metrics.get("patch_count", 0) or 0)
    clean_shadow_values = [int(record.get("clean_shadow_bytes", 0) or 0) for record in records.values()]
    cold_report = records["cold_clean"]["cache_report"]
    cold_param_devices = cold_report.get("resident_parameter_devices", {})
    release_after = release_result or {}

    checks = {
        "cold_load_not_reused": records["cold_clean"]["reused"] is False,
        "direct_safetensors_load": bool(cold_report.get("active_unet_raw_sequential_stream")),
        "meta_construction": bool(cold_report.get("active_unet_meta_construction")),
        "clean_shadow_zero_all_steps": all(value == 0 for value in clean_shadow_values),
        "settled_resident_spine_active": bool(cold_report.get("active_resident_spine")),
        "resident_parameters_on_requested_device": (
            args.mock
            or (
                bool(cold_param_devices)
                and all(str(device) == str(dev) or str(dev).startswith("cuda") for dev in cold_param_devices)
            )
        ),
        "same_clean_stack_reused": records["warm_clean"]["reused"] is True,
        "lora_change_reused_same_spine_object": (
            records["lora_a"]["spine_id"] == records["cold_clean"]["spine_id"]
            and records["lora_a"]["reused"] is False
        ),
        "same_lora_stack_reused": records["warm_lora_a"]["reused"] is True,
        "lora_compile_produced_unet_patches": lora_patch_count > 0,
        "lora_removal_reused_same_spine_object": (
            records["lora_removed"]["spine_id"] == records["cold_clean"]["spine_id"]
            and records["lora_removed"]["reused"] is False
        ),
        "explicit_release_cleared_resident_spine": not bool(
            (release_after.get("cache_report_after") or {}).get("active_resident_spine")
        ),
        "failure_free_telemetry": counts.get("resident_spine_load_failure_cleanup", 0) == 0,
        "linux_runtime": platform.system().lower() == "linux",
    }

    field_acceptance_ready = (not args.mock) and all(checks.values())
    mock_rehearsal_ready = bool(
        args.mock
        and all(value for key, value in checks.items() if key != "linux_runtime")
    )

    summary = {
        "status": "completed" if field_acceptance_ready or mock_rehearsal_ready else "incomplete",
        "mock": bool(args.mock),
        "run_name": args.run_name,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" and torch.cuda.is_available() else None,
        },
        "inputs": {
            "checkpoint": str(checkpoint_path),
            "lora_stack": [str(path) for path in main_lora_paths_for_summary],
            "lora_weights": [float(weight) for weight in main_lora_weights_for_summary],
            "alternate_lora_b": str(alternate_lora_path) if alternate_lora_path else None,
            "fast_identity": bool(args.fast_identity),
        },
        "records": records,
        "release": release_result,
        "telemetry_event_counts": counts,
        "telemetry_tail": events[-20:],
        "acceptance_checks": checks,
        "field_acceptance_ready": field_acceptance_ready,
        "mock_rehearsal_ready": mock_rehearsal_ready,
    }
    return bool(field_acceptance_ready or mock_rehearsal_ready), summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run W12a resident SDXL spine validation.")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode without real model assets.")
    parser.add_argument("--checkpoint", help="Path to an SDXL safetensors checkpoint for real validation.")
    parser.add_argument("--lora", dest="lora_paths", action="append", help="UNet-targeting safetensors LoRA path. Repeat to validate a multi-LoRA stack.")
    parser.add_argument("--lora-weight", dest="lora_weights", action="append", type=float, help="Weight for each repeated --lora path. Defaults to 1.0 for each LoRA.")
    parser.add_argument("--lora-a", help="Path to a UNet-targeting safetensors LoRA for stack-change validation.")
    parser.add_argument("--lora-b", help="Optional second UNet-targeting safetensors LoRA for A->B stack-change validation.")
    parser.add_argument("--lora-a-weight", type=float, default=1.0)
    parser.add_argument("--lora-b-weight", type=float, default=0.8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-name", default="w12a_resident_spine_probe")
    parser.add_argument("--output-json", help="Path to write the structured summary JSON.")
    parser.add_argument("--fast-identity", action="store_true", help="Use size/mtime identity instead of SHA256 hashing.")
    args = parser.parse_args()

    ok, summary = run_validation(args)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("Summary written to %s", output_path)
    print(json.dumps(summary, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
