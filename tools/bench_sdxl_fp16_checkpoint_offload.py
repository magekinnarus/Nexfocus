import argparse
import gc
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Optional

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ORIGINAL_SYS_ARGV = list(sys.argv)
sys.argv = [sys.argv[0]]

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import loader, resources
from backend import float_ops as backend_float_ops
from backend import utils as backend_utils
from backend.cpu_compiler import CpuArtifactCompiler
from backend.weight_ops import calculate_weight, get_key_weight, string_to_seed
from backend.gguf.direct_sdxl_runtime import DirectSDXLGGUFRuntime, DirectSDXLGGUFRunConfig
from modules.gguf_headless_runner import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_POSITIVE_PROMPT,
    QualityConfig,
    ScenarioConfig,
    collect_environment,
    write_environment_report,
)
from tools.bench_sdxl_pinned_residency_matrix import (
    MemorySampler,
    _append_jsonl,
    _as_folder_list,
    _capture_phase_memory,
    _json_default,
    _parse_lora_spec,
    _prompt_hash,
    _resolve_local_model_path,
    _save_png,
    _write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark fp16 SDXL checkpoint CPU-resident low-vram execution with optional host pinning and LoRA.",
    )
    parser.add_argument("--runs", type=int, default=1, help="Total runs including the cold run.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "P4-M13-W07b-R"))
    parser.add_argument("--scenario-name", default="fp16_checkpoint_offload")
    parser.add_argument("--checkpoint-path", default="", help="Optional fp16 SDXL checkpoint override.")
    parser.add_argument(
        "--execution-shape",
        default="registered_patch_offload",
        choices=("registered_patch_offload", "materialized_patch_offload", "materialized_patch_streamlike"),
        help="Select the fp16 execution shape to benchmark.",
    )
    parser.add_argument(
        "--clip-modes",
        nargs="+",
        default=("cpu_only", "gpu_then_offload"),
        choices=("gpu_then_offload", "cpu_only"),
        help="Benchmark one or both CLIP residency modes.",
    )
    parser.add_argument(
        "--unet-budget-mb",
        type=int,
        default=None,
        help="Optional low-vram budget to pass to the UNet attach path.",
    )
    parser.add_argument(
        "--streamlike-budget-mb",
        type=int,
        default=256,
        help="Tight UNet budget used by the stream-like execution shape.",
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--cfg", type=float, default=7.0)
    parser.add_argument("--sampler", default="euler_ancestral")
    parser.add_argument("--scheduler", default="karras")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--clip-layer", type=int, default=-2)
    parser.add_argument(
        "--lora",
        dest="lora_specs",
        action="append",
        default=[],
        help="Optional LoRA spec in the form path[:weight]. Repeat the flag for multiple LoRAs.",
    )
    parser.add_argument(
        "--lora-state",
        default="both",
        choices=("off", "on", "both"),
        help="Select whether to run LoRA-off control rows, LoRA-on rows, or both.",
    )
    parser.add_argument(
        "--pin-unet-host",
        dest="pin_unet_host",
        action="store_true",
        default=True,
        help="Pin fp16 UNet CPU tensors before low-vram attach.",
    )
    parser.add_argument(
        "--no-pin-unet-host",
        dest="pin_unet_host",
        action="store_false",
        help="Disable host pinning for the fp16 UNet CPU tensors.",
    )
    parser.add_argument(
        "--stage-conditioning-to-gpu",
        action="store_true",
        default=False,
        help="Stage encoded prompt conditioning to GPU before ADM construction.",
    )
    parser.add_argument(
        "--materialize-patched-artifacts",
        action="store_true",
        default=False,
        help="Fully materialize CLIP and UNet LoRA patches on CPU before low-vram attach.",
    )
    parser.add_argument(
        "--use-cpu-compiler",
        action="store_true",
        default=False,
        help="Use backend.cpu_compiler for UNet CPU materialization instead of the local sequential seal path.",
    )
    parser.add_argument(
        "--cpu-compiler-workers",
        type=int,
        default=None,
        help="Optional explicit worker count for backend.cpu_compiler.",
    )
    parser.add_argument(
        "--cpu-compiler-torch-threads",
        type=int,
        default=None,
        help="Optional torch thread count to use inside each cpu_compiler worker.",
    )
    parser.add_argument("--notes", default="")
    parser.add_argument("--traceback", action="store_true")
    return parser.parse_args(ORIGINAL_SYS_ARGV[1:])


def _resolve_fp16_checkpoint(override: str = "") -> str:
    import modules.config as config

    folders = _as_folder_list(getattr(config, "paths_checkpoints", []))
    if override:
        resolved = _resolve_local_model_path(str(override), folders)
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"Resolved checkpoint does not exist: {resolved}")
        return resolved

    selected = getattr(config, "default_base_model_name", None) or getattr(config, "default_model", None)
    if selected and str(selected).lower() != "none":
        resolved = _resolve_local_model_path(str(selected), folders)
        if os.path.isfile(resolved) and not resolved.lower().endswith(".gguf"):
            taxonomy = config.resolve_model_taxonomy(resolved, root_keys=("checkpoints",), folder_paths=folders)
            if taxonomy.architecture == "sdxl":
                return resolved

    candidates = []
    for folder in folders:
        root = Path(folder)
        if not root.exists():
            continue
        for suffix in ("*.safetensors", "*.ckpt"):
            candidates.extend(root.rglob(suffix))

    seen = set()
    for candidate in candidates:
        resolved = str(candidate)
        normalized = resolved.lower()
        if normalized.endswith(".gguf") or normalized in seen:
            continue
        seen.add(normalized)
        taxonomy = config.resolve_model_taxonomy(resolved, root_keys=("checkpoints",), folder_paths=folders)
        if taxonomy.architecture == "sdxl":
            return resolved
        if "sdxl" in normalized or "sdxl" in str(candidate.parent).lower():
            return resolved

    raise FileNotFoundError("No fp16/non-GGUF SDXL checkpoint was found in config.txt checkpoint roots.")


def _build_scenario(args: argparse.Namespace, checkpoint_path: str) -> ScenarioConfig:
    scenario = ScenarioConfig(
        name=args.scenario_name,
        unet_path=checkpoint_path,
        clip_l_path=checkpoint_path,
        clip_g_path=checkpoint_path,
        vae_path=checkpoint_path,
        prompt=DEFAULT_POSITIVE_PROMPT,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        width=args.width,
        height=args.height,
        steps=args.steps,
        cfg=args.cfg,
        sampler=args.sampler,
        scheduler=args.scheduler,
        seed=args.seed,
        clip_layer=args.clip_layer,
        quality=QualityConfig(
            sharpness=0.0,
            adaptive_cfg=args.cfg,
            adm_scale_positive=1.5,
            adm_scale_negative=0.8,
            adm_scaler_end=0.3,
        ),
        notes="FP16 SDXL checkpoint experiment using checkpoint-extracted CLIP/VAE with CPU-resident UNet low-vram attach.",
    )
    if args.notes:
        scenario = replace(scenario, notes=(scenario.notes + " " + args.notes).strip())
    return scenario


def _pin_module_tensors(module: torch.nn.Module) -> int:
    if not torch.cuda.is_available():
        return 0

    pinned_bytes = 0
    for _, submodule in module.named_modules():
        for _, param in submodule.named_parameters(recurse=False):
            if param is None or param.device.type != "cpu" or param.is_pinned():
                continue
            pinned = param.data.contiguous().pin_memory()
            param.data = pinned
            pinned_bytes += pinned.numel() * pinned.element_size()
        for name, buf in submodule.named_buffers(recurse=False):
            if buf is None or buf.device.type != "cpu" or buf.is_pinned():
                continue
            pinned = buf.contiguous().pin_memory()
            submodule._buffers[name] = pinned
            pinned_bytes += pinned.numel() * pinned.element_size()
    return pinned_bytes


def _seal_materialized_patcher(patcher: Any, *, pin_host: bool = False) -> dict[str, Any]:
    cpu_device = torch.device("cpu")
    patch_count = len(getattr(patcher, "patches", {}) or {})
    if patch_count > 0:
        for key in list(patcher.patches.keys()):
            weight, set_func, convert_func = get_key_weight(patcher.model, key)
            preserved_dtype = weight.dtype
            temp_weight = weight.to(device=cpu_device, dtype=preserved_dtype, copy=True)
            if convert_func is not None:
                temp_weight = convert_func(temp_weight, inplace=True)
            out_weight = calculate_weight(
                patcher.patches[key],
                temp_weight,
                key,
                intermediate_dtype=preserved_dtype,
                original_weights={},
            )
            if set_func is None:
                out_weight = backend_float_ops.stochastic_rounding(
                    out_weight,
                    preserved_dtype,
                    seed=string_to_seed(key),
                )
                backend_utils.set_attr_param(patcher.model, key, out_weight)
            else:
                set_func(out_weight, inplace_update=False, seed=string_to_seed(key))
        patcher.model.to(cpu_device)
        patcher.model.device = cpu_device
        patcher.model.model_loaded_weight_memory = patcher.model_size()
        patcher.model.model_lowvram = False
        patcher.model.lowvram_patch_counter = 0
    else:
        patcher.patch_model(
            device_to=cpu_device,
            lowvram_model_memory=0,
            load_weights=True,
            force_patch_weights=False,
        )

    host_pinned_bytes = 0
    if pin_host:
        host_pinned_bytes = _pin_module_tensors(patcher.model)

    patcher.patches = {}
    patcher.weight_wrapper_patches = {}
    patcher.backup.clear()
    patcher.object_patches_backup.clear()
    patcher.model.current_weight_patches_uuid = None
    patcher.model.model_loaded_weight_memory = patcher.model_size()
    patcher.model.model_lowvram = False
    patcher.model.lowvram_patch_counter = 0
    patcher.model.device = cpu_device
    patcher.current_device = cpu_device
    patcher.load_device = cpu_device
    patcher.offload_device = cpu_device
    patcher.patches_uuid = uuid.uuid4()

    for module in patcher.model.modules():
        if hasattr(module, "weight_function"):
            module.weight_function = []
        if hasattr(module, "bias_function"):
            module.bias_function = []
        if hasattr(module, "comfy_patched_weights"):
            try:
                del module.comfy_patched_weights
            except Exception:
                pass

    return {
        "materialized_patch_keys": patch_count,
        "host_pinned_bytes": host_pinned_bytes,
    }


class FP16CheckpointRuntime(DirectSDXLGGUFRuntime):
    route_label = "fp16_checkpoint_offload"

    def __init__(
        self,
        config: DirectSDXLGGUFRunConfig,
        *,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
        unet_budget_mb: Optional[int] = None,
        pin_unet_host: bool = True,
    ) -> None:
        super().__init__(config, device=device, unet_budget_mb=unet_budget_mb)
        self.checkpoint_path = checkpoint_path
        self.pin_unet_host = bool(pin_unet_host)
        self.host_pin_wall = 0.0
        self.host_pinned_bytes = 0

    def load_components(self) -> float:
        if self._loaded:
            return 0.0

        start = time.perf_counter()
        cpu_device = torch.device("cpu")
        self.unet, self.clip, self.vae = loader.load_sdxl_checkpoint(
            self.checkpoint_path,
            load_device=cpu_device,
            offload_device=cpu_device,
            unet_dtype=torch.float16,
            clip_load_device=cpu_device,
            clip_offload_device=cpu_device,
            vae_offload_device=cpu_device,
        )
        self.unet.runtime_release_to_meta = False
        self.clip.clip_layer(self.config.clip_layer)

        if self.pin_unet_host:
            pin_start = time.perf_counter()
            self.host_pinned_bytes = _pin_module_tensors(self.unet.model.diffusion_model)
            self.host_pin_wall = time.perf_counter() - pin_start

        if self.config.quality:
            loader.patch_unet_for_quality(self.unet, self.config.quality)

        self._cold_model_load_cpu = time.perf_counter() - start
        self._loaded = True
        return self._cold_model_load_cpu


class FP16CheckpointStreamlikeRuntime(FP16CheckpointRuntime):
    route_label = "fp16_materialized_streamlike"

    def __init__(
        self,
        config: DirectSDXLGGUFRunConfig,
        *,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
        unet_budget_mb: Optional[int] = None,
        pin_unet_host: bool = True,
        streamlike_budget_mb: int = 256,
    ) -> None:
        super().__init__(
            config,
            checkpoint_path=checkpoint_path,
            device=device,
            unet_budget_mb=unet_budget_mb,
            pin_unet_host=pin_unet_host,
        )
        self.streamlike_budget_mb = max(64, int(streamlike_budget_mb))

    def _clean_unet_budget_bytes(self) -> int:
        return self.streamlike_budget_mb * 1024 * 1024


def _build_runtime_config(
    scenario: ScenarioConfig,
    *,
    clip_residency_mode: str,
    stage_prompt_conditioning_to_device: bool,
) -> DirectSDXLGGUFRunConfig:
    return DirectSDXLGGUFRunConfig(
        unet_path=scenario.unet_path,
        clip_l_path=scenario.clip_l_path,
        clip_g_path=scenario.clip_g_path,
        vae_path=scenario.vae_path,
        prompt=scenario.prompt,
        negative_prompt=scenario.negative_prompt,
        width=scenario.width,
        height=scenario.height,
        steps=scenario.steps,
        cfg=scenario.cfg,
        sampler=scenario.sampler,
        scheduler=scenario.scheduler,
        seed=scenario.seed,
        clip_layer=scenario.clip_layer,
        clip_residency_mode=clip_residency_mode,
        denoise=1.0,
        batch_size=1,
        quality=scenario.quality.as_sampling_dict(),
        route_family="fp16_checkpoint",
        process_class="SDXL_FP16_CHECKPOINT_OFFLOAD",
        execution_class=None,
        stage_prompt_conditioning_to_device=stage_prompt_conditioning_to_device,
    )


def _apply_loras_to_runtime(runtime: Any, *, scenario: ScenarioConfig, lora_specs: list[tuple[str, float]]) -> dict[str, Any]:
    import modules.core as core
    import modules.model_taxonomy as model_taxonomy

    if not lora_specs:
        return {
            "enabled": False,
            "spec_count": 0,
            "artifact_count": 0,
            "artifact_sources": [],
            "artifact_scales": [],
            "refresh_loras_wall": 0.0,
        }

    model = core.StableDiffusionModel(
        unet=runtime.unet,
        clip=runtime.clip,
        filename=scenario.unet_path,
        architecture=model_taxonomy.ARCHITECTURE_SDXL,
    )
    start = time.perf_counter()
    model.refresh_loras(lora_specs)
    refresh_loras_wall = time.perf_counter() - start
    runtime.unet = model.unet_with_lora
    runtime.clip = model.clip_with_lora
    runtime.unet.runtime_release_to_meta = False

    return {
        "enabled": True,
        "spec_count": len(lora_specs),
        "artifact_count": len(model.lora_artifact_registry),
        "artifact_sources": [artifact.source_path for artifact in model.lora_artifact_registry],
        "artifact_scales": [float(artifact.default_scale) for artifact in model.lora_artifact_registry],
        "refresh_loras_wall": refresh_loras_wall,
    }


def _materialize_runtime_artifacts_on_cpu(
    runtime: Any,
    *,
    pin_unet_host: bool,
    use_cpu_compiler: bool = False,
    cpu_compiler_workers: Optional[int] = None,
    cpu_compiler_torch_threads: Optional[int] = None,
) -> dict[str, Any]:
    probe = {
        "enabled": True,
        "clip_materialize_wall": 0.0,
        "unet_materialize_wall": 0.0,
        "clip_patch_keys": 0,
        "unet_patch_keys": 0,
        "extra_host_pinned_bytes": 0,
    }

    if runtime.clip is not None:
        start = time.perf_counter()
        clip_result = _seal_materialized_patcher(runtime.clip.patcher, pin_host=False)
        probe["clip_materialize_wall"] = time.perf_counter() - start
        probe["clip_patch_keys"] = int(clip_result.get("materialized_patch_keys", 0))

    if runtime.unet is not None:
        start = time.perf_counter()
        if use_cpu_compiler:
            unet_result = CpuArtifactCompiler.compile_streaming_unet_patcher(
                runtime.unet,
                pin_unet_host=pin_unet_host,
                num_workers=cpu_compiler_workers,
                torch_threads_per_worker=cpu_compiler_torch_threads,
            )
        else:
            unet_result = _seal_materialized_patcher(runtime.unet, pin_host=pin_unet_host)
        probe["unet_materialize_wall"] = time.perf_counter() - start
        probe["unet_patch_keys"] = int(unet_result.get("materialized_patch_keys", 0))
        probe["extra_host_pinned_bytes"] = int(unet_result.get("host_pinned_bytes", 0))
        probe["compiler_backend"] = "cpu_compiler" if use_cpu_compiler else "sequential_seal"

    return probe


def _combo_name(clip_mode: str, lora_enabled: bool) -> str:
    return f"{clip_mode}_{'lora_on' if lora_enabled else 'lora_off'}"


def _run_combo(
    *,
    scenario: ScenarioConfig,
    checkpoint_path: str,
    clip_mode: str,
    lora_specs: list[tuple[str, float]],
    run_dir: Path,
    run_label: str,
    unet_budget_mb: Optional[int],
    streamlike_budget_mb: int,
    pin_unet_host: bool,
    stage_conditioning_to_gpu: bool,
    execution_shape: str,
    use_cpu_compiler: bool,
    cpu_compiler_workers: Optional[int],
    cpu_compiler_torch_threads: Optional[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    materialize_patched_artifacts = execution_shape in (
        "materialized_patch_offload",
        "materialized_patch_streamlike",
    )
    config = _build_runtime_config(
        scenario,
        clip_residency_mode=clip_mode,
        stage_prompt_conditioning_to_device=stage_conditioning_to_gpu,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime_cls = FP16CheckpointStreamlikeRuntime if execution_shape == "materialized_patch_streamlike" else FP16CheckpointRuntime
    runtime_kwargs = {
        "config": config,
        "checkpoint_path": checkpoint_path,
        "device": device,
        "unet_budget_mb": unet_budget_mb,
        "pin_unet_host": (pin_unet_host and not materialize_patched_artifacts),
    }
    if runtime_cls is FP16CheckpointStreamlikeRuntime:
        runtime_kwargs["streamlike_budget_mb"] = streamlike_budget_mb
    runtime = runtime_cls(**runtime_kwargs)

    phase_memory = []
    payload: Optional[dict[str, Any]] = None
    lora_probe = {
        "enabled": False,
        "spec_count": 0,
        "artifact_count": 0,
        "artifact_sources": [],
        "artifact_scales": [],
        "refresh_loras_wall": 0.0,
    }
    materialize_probe = {
        "enabled": False,
        "clip_materialize_wall": 0.0,
        "unet_materialize_wall": 0.0,
        "clip_patch_keys": 0,
        "unet_patch_keys": 0,
        "extra_host_pinned_bytes": 0,
        "compiler_backend": "none",
        "compiler_workers": cpu_compiler_workers,
        "compiler_torch_threads": cpu_compiler_torch_threads,
    }

    try:
        start = time.perf_counter()
        with MemorySampler() as memory:
            runtime.load_components()
            phase_memory.append(_capture_phase_memory("after_load_components"))

            if lora_specs:
                lora_probe = _apply_loras_to_runtime(runtime, scenario=scenario, lora_specs=lora_specs)
                phase_memory.append(_capture_phase_memory("after_lora_refresh"))
                if materialize_patched_artifacts:
                    materialize_probe = _materialize_runtime_artifacts_on_cpu(
                        runtime,
                        pin_unet_host=pin_unet_host,
                        use_cpu_compiler=use_cpu_compiler,
                        cpu_compiler_workers=cpu_compiler_workers,
                        cpu_compiler_torch_threads=cpu_compiler_torch_threads,
                    )
                    runtime.host_pinned_bytes = max(
                        int(getattr(runtime, "host_pinned_bytes", 0)),
                        int(materialize_probe.get("extra_host_pinned_bytes", 0)),
                    )
                    phase_memory.append(_capture_phase_memory("after_materialize_patched_artifacts"))

            prepared_inputs, prep_metrics = runtime.prepare_inputs()
            phase_memory.append(_capture_phase_memory("after_prepare_inputs"))

            denoise_result = runtime.denoise_prepared_inputs(prepared_inputs)
            phase_memory.append(_capture_phase_memory("after_denoise"))

            images, vae_attach, vae_decode = runtime.decode_latent(denoise_result.samples)
            phase_memory.append(_capture_phase_memory("after_decode"))

            image_path = run_dir / f"{scenario.name}_{_combo_name(clip_mode, bool(lora_specs))}_{run_label}.png"
            image_save_start = time.perf_counter()
            _save_png(image_path, images[0])
            image_save = time.perf_counter() - image_save_start

            total_wall = time.perf_counter() - start
            payload = {
                "scenario": scenario.name,
                "route": runtime.route_label,
                "route_label": runtime.route_label,
                "execution_shape": execution_shape,
                "use_cpu_compiler": bool(use_cpu_compiler),
                "cpu_compiler_workers": cpu_compiler_workers,
                "cpu_compiler_torch_threads": cpu_compiler_torch_threads,
                "run_label": run_label,
                "checkpoint_path": checkpoint_path,
                "prompt_hash": _prompt_hash(scenario.prompt, scenario.negative_prompt),
                "quant_model": Path(checkpoint_path).name,
                "resolution": f"{scenario.width}x{scenario.height}",
                "steps": scenario.steps,
                "cfg": scenario.cfg,
                "sampler": scenario.sampler,
                "scheduler": scenario.scheduler,
                "seed": scenario.seed,
                "batch_size": 1,
                "clip_residency_mode": clip_mode,
                "pin_unet_host": pin_unet_host,
                "host_pin_wall": runtime.host_pin_wall,
                "host_pinned_bytes": runtime.host_pinned_bytes,
                "lora_probe": lora_probe,
                "materialize_probe": materialize_probe,
                "benchmark": {
                    "route_label": runtime.route_label,
                    "execution_shape": execution_shape,
                    "use_cpu_compiler": bool(use_cpu_compiler),
                    "cpu_compiler_workers": cpu_compiler_workers,
                    "cpu_compiler_torch_threads": cpu_compiler_torch_threads,
                    "clip_residency_mode": runtime.config.clip_residency_mode,
                    "cold_model_load_cpu": getattr(runtime, "_cold_model_load_cpu", 0.0),
                    "clip_residency_attach": prep_metrics["clip_residency_attach"],
                    "clip_residency_offload": prep_metrics["clip_residency_offload"],
                    "clip_encode": prep_metrics["clip_encode"],
                    "conditioning_stage_to_device": prep_metrics.get("conditioning_stage_to_device", 0.0),
                    "adm_build": prep_metrics["adm_build"],
                    "latent_noise_prep": prep_metrics["latent_noise_prep"],
                    "sampler_model_attach": denoise_result.sampler_model_attach,
                    "cond_prepare_explicit": denoise_result.cond_prepare_duration,
                    "denoise_wall": denoise_result.denoise_wall,
                    "denoise_s_per_it": denoise_result.denoise_wall / max(1, runtime.config.steps),
                    "denoise_cpu_proc": denoise_result.denoise_cpu_proc,
                    "gguf_dequant": float(denoise_result.gguf_trace_stats.get("dequant_seconds", 0.0)),
                    "gguf_dequant_cpu_proc": float(denoise_result.gguf_trace_stats.get("dequant_cpu_process_seconds", 0.0)),
                    "vae_attach": vae_attach,
                    "vae_decode": vae_decode,
                    "image_save": image_save,
                    "total_wall": total_wall,
                },
                "phase_memory": [asdict(snapshot) for snapshot in phase_memory],
                "image_path": str(image_path),
                "total_wall": total_wall,
                "peak_rss_bytes": memory.snapshot.peak_rss_bytes,
                "peak_vram_allocated_bytes": memory.snapshot.peak_vram_allocated_bytes,
                "peak_vram_reserved_bytes": memory.snapshot.peak_vram_reserved_bytes,
                "notes": scenario.notes,
            }
    finally:
        runtime.close()
        gc.collect()
        if payload is not None:
            payload["post_close_memory"] = asdict(_capture_phase_memory("after_close"))

    if payload is None:
        raise RuntimeError("Benchmark payload was not produced.")
    return payload, dict(payload["benchmark"])


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    effective_execution_shape = (
        "materialized_patch_offload"
        if args.materialize_patched_artifacts and args.execution_shape == "registered_patch_offload"
        else args.execution_shape
    )

    run_root = Path(args.output_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = run_root / f"{args.scenario_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        checkpoint_path = _resolve_fp16_checkpoint(args.checkpoint_path)
        scenario = _build_scenario(args, checkpoint_path)
        environment = collect_environment("fp16_checkpoint_offload", scenario)
        write_environment_report(environment, run_dir)

        lora_specs = [_parse_lora_spec(spec) for spec in args.lora_specs]
        if not lora_specs:
            lora_states = (False,)
        elif args.lora_state == "off":
            lora_states = (False,)
        elif args.lora_state == "on":
            lora_states = (True,)
        else:
            lora_states = (False, True)
        if args.lora_state == "on" and not lora_specs:
            raise ValueError("--lora-state on requires at least one --lora spec")

        results: list[dict[str, Any]] = []
        run_labels = ["cold"] + [f"warm_{index}" for index in range(1, args.runs)]
        for clip_mode in tuple(args.clip_modes):
            for lora_enabled in lora_states:
                combo_dir = run_dir / _combo_name(clip_mode, lora_enabled)
                combo_dir.mkdir(parents=True, exist_ok=True)
                active_loras = lora_specs if lora_enabled else []
                for run_label in run_labels:
                    payload, raw_result = _run_combo(
                        scenario=scenario,
                        checkpoint_path=checkpoint_path,
                        clip_mode=clip_mode,
                        lora_specs=active_loras,
                        run_dir=combo_dir,
                        run_label=run_label,
                        unet_budget_mb=args.unet_budget_mb,
                        streamlike_budget_mb=args.streamlike_budget_mb,
                        pin_unet_host=args.pin_unet_host,
                        stage_conditioning_to_gpu=args.stage_conditioning_to_gpu,
                        execution_shape=effective_execution_shape,
                        use_cpu_compiler=args.use_cpu_compiler,
                        cpu_compiler_workers=args.cpu_compiler_workers,
                        cpu_compiler_torch_threads=args.cpu_compiler_torch_threads,
                    )
                    _append_jsonl(combo_dir / "benchmark_results.jsonl", payload)
                    results.append(payload)
                    print(
                        json.dumps(
                            {
                                "run": run_label,
                                "execution_shape": payload.get("execution_shape", ""),
                                "cpu_compiler": bool(payload.get("use_cpu_compiler", False)),
                                "compiler_workers": payload.get("cpu_compiler_workers", None),
                                "compiler_torch_threads": payload.get("cpu_compiler_torch_threads", None),
                                "clip_mode": clip_mode,
                                "lora": "on" if lora_enabled else "off",
                                "checkpoint": Path(checkpoint_path).name,
                                "denoise_s_per_it": round(float(raw_result.get("denoise_s_per_it", 0.0)), 4),
                                "total_wall": round(float(payload.get("total_wall", 0.0)), 4),
                                "peak_rss_mb": round(float(payload.get("peak_rss_bytes", 0)) / (1024 * 1024), 2),
                                "peak_vram_mb": round(float(payload.get("peak_vram_reserved_bytes", 0)) / (1024 * 1024), 2),
                                "host_pinned_mb": round(float(payload.get("host_pinned_bytes", 0)) / (1024 * 1024), 2),
                                "image_path": payload.get("image_path", ""),
                            },
                            default=_json_default,
                        )
                    )

        summary = {
            "environment": asdict(environment),
            "scenario": asdict(scenario),
            "matrix": {
                "clip_modes": list(tuple(args.clip_modes)),
                "lora_enabled": bool(lora_specs),
                "lora_state": args.lora_state,
                "pin_unet_host": bool(args.pin_unet_host),
                "stage_conditioning_to_gpu": bool(args.stage_conditioning_to_gpu),
                "materialize_patched_artifacts": effective_execution_shape != "registered_patch_offload",
                "execution_shape": effective_execution_shape,
                "use_cpu_compiler": bool(args.use_cpu_compiler),
                "cpu_compiler_workers": args.cpu_compiler_workers,
                "cpu_compiler_torch_threads": args.cpu_compiler_torch_threads,
                "unet_budget_mb": args.unet_budget_mb,
                "streamlike_budget_mb": args.streamlike_budget_mb,
            },
            "checkpoint_path": checkpoint_path,
            "results": results,
            "output_dir": str(run_dir),
            "notes": "Tool-only fp16 SDXL checkpoint CPU-resident low-vram benchmark.",
        }
        _write_json(run_dir / "summary.json", summary)
        print(json.dumps({"summary": str(run_dir / "summary.json"), "output_dir": str(run_dir)}, default=_json_default))
        return 0
    except Exception as exc:
        error = {
            "status": "error",
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
            },
            "output_dir": str(run_dir),
        }
        if args.traceback:
            import traceback

            error["traceback"] = traceback.format_exc()
        _write_json(run_dir / "error.json", error)
        print(json.dumps(error, default=_json_default))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
