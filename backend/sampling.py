import torch
import math
import logging
import time
from functools import partial
from typing import Any, Callable, List, Dict, Optional, Union, Tuple

# Local imports
from . import schedulers
from . import k_diffusion
from . import precision

# Re-export key constants for registration
SCHEDULER_NAMES = schedulers.SCHEDULER_NAMES

from . import anisotropic

# Note: We avoid importing from ldm_patched directly to keep the backend clean.
# ModelPatcher and BaseModel are expected to be passed as generic objects or Any.

from .cond_utils import (
    add_area_dims, get_area_and_mult, cond_equal_size, can_concat_cond,
    cond_cat, calc_cond_batch, resolve_areas_and_cond_masks_multidim,
    calculate_start_end_timesteps, encode_model_conds, process_conds,
    reset_cond_batch_trace_stats, consume_cond_batch_trace_stats
)

class KSamplerX0Inpaint:
    def __init__(self, model: Any, sigmas: torch.Tensor):
        self.inner_model = model
        self.sigmas = sigmas
        self.noise = None
        self.latent_image = None
        
    def __call__(self, x: torch.Tensor, sigma: torch.Tensor, denoise_mask: Optional[torch.Tensor] = None, model_options: Dict[str, Any] = {}, seed: Optional[int] = None) -> torch.Tensor:
        if denoise_mask is not None:
            if "denoise_mask_function" in model_options:
                denoise_mask = model_options["denoise_mask_function"](sigma, denoise_mask, extra_options={"model": self.inner_model, "sigmas": self.sigmas})
            latent_mask = 1. - denoise_mask
            x = x * denoise_mask + self.inner_model.inner_model.model_sampling.noise_scaling(sigma, self.noise, self.latent_image) * latent_mask
        
        out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
        
        if denoise_mask is not None:
            out = out * denoise_mask + self.latent_image * latent_mask
        return out

class Sampler:
    def sample(self, model_wrap: Any, sigmas: torch.Tensor, extra_args: Dict[str, Any], callback: Optional[Callable], noise: torch.Tensor, latent_image: Optional[torch.Tensor] = None, denoise_mask: Optional[torch.Tensor] = None, disable_pbar: bool = False) -> torch.Tensor:
        pass

    def max_denoise(self, model_wrap: Any, sigmas: torch.Tensor) -> bool:
        max_sigma = float(model_wrap.inner_model.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma

class KSAMPLER(Sampler):
    def __init__(self, sampler_function: Callable, extra_options: Dict[str, Any] = {}, inpaint_options: Dict[str, Any] = {}):
        self.sampler_function = sampler_function
        self.extra_options = extra_options
        self.inpaint_options = inpaint_options

    def sample(self, model_wrap: Any, sigmas: torch.Tensor, extra_args: Dict[str, Any], callback: Optional[Callable], noise: torch.Tensor, latent_image: Optional[torch.Tensor] = None, denoise_mask: Optional[torch.Tensor] = None, disable_pbar: bool = False) -> torch.Tensor:
        extra_args["denoise_mask"] = denoise_mask
        model_k = KSamplerX0Inpaint(model_wrap, sigmas)
        model_k.latent_image = latent_image
        
        if self.inpaint_options.get("random", False):
            generator = torch.manual_seed(extra_args.get("seed", 41) + 1)
            model_k.noise = torch.randn(noise.shape, generator=generator, device="cpu").to(noise.dtype).to(noise.device)
        else:
            model_k.noise = noise

        noise = model_wrap.inner_model.model_sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model_wrap, sigmas))

        k_callback = None
        total_steps = len(sigmas) - 1
        if callback is not None:
            k_callback = lambda x: callback(x["i"], x["denoised"], x["x"], total_steps, x.get("denoised", None))

        samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
        samples = model_wrap.inner_model.model_sampling.inverse_noise_scaling(sigmas[-1], samples)
        return samples

# Sampler Registry
KSAMPLER_NAMES = [
    "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2",
    "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral",
    "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_cfg_pp",
    "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm",
    "ipndm", "ipndm_v", "deis", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral",
    "res_multistep_ancestral_cfg_pp", "gradient_estimation", "gradient_estimation_cfg_pp",
    "er_sde", "seeds_2", "seeds_3", "sa_solver", "sa_solver_pece"
]

SAMPLER_NAMES = KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]

def sampler_names() -> List[str]:
    return SAMPLER_NAMES

def ksampler(sampler_name: str, extra_options: Dict[str, Any] = {}, inpaint_options: Dict[str, Any] = {}) -> KSAMPLER:
    if sampler_name == "dpm_fast":
        def dpm_fast_function(model, noise, sigmas, extra_args, callback, disable):
            if len(sigmas) <= 1: return noise
            sigma_min = sigmas[-1] if sigmas[-1] > 0 else sigmas[-2]
            return k_diffusion.sample_dpm_fast(model, noise, sigma_min, sigmas[0], len(sigmas) - 1, extra_args=extra_args, callback=callback, disable=disable)
        sampler_function = dpm_fast_function
    elif sampler_name == "dpm_adaptive":
        def dpm_adaptive_function(model, noise, sigmas, extra_args, callback, disable, **extra_options):
            if len(sigmas) <= 1: return noise
            sigma_min = sigmas[-1] if sigmas[-1] > 0 else sigmas[-2]
            return k_diffusion.sample_dpm_adaptive(model, noise, sigma_min, sigmas[0], extra_args=extra_args, callback=callback, disable=disable, **extra_options)
        sampler_function = dpm_adaptive_function
    else:
        func_name = f"sample_{sampler_name.replace('_cfg_pp', '')}"
        sampler_function = getattr(k_diffusion, func_name, None)
        if sampler_function is None:
            raise ValueError(f"Sampler {sampler_name} not implemented in k_diffusion as {func_name}")
            
    return KSAMPLER(sampler_function, extra_options, inpaint_options)

def sample_sdxl(
    model: Any,
    noise: torch.Tensor,
    positive: Any,
    negative: Any,
    cfg: float,
    steps: int,
    sampler_name: str,
    scheduler: str,
    denoise: float = 1.0,
    seed: int = None,
    latent_image: torch.Tensor = None,
    denoise_mask: torch.Tensor = None,
    callback: Callable = None,
    disable_pbar: bool = False,
    model_options: Dict[str, Any] = {}
) -> torch.Tensor:
    """Main entry point for SDXL sampling."""
    device = noise.device
    ksampler_inst = KSampler(model, steps, device, sampler_name, scheduler, denoise, model_options=model_options)
    return ksampler_inst.sample(
        noise, positive, negative, cfg,
        latent_image=latent_image,
        denoise_mask=denoise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed
    )

def sampler_priority() -> List[str]:
    return ["euler", "euler_ancestral", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde_gpu"]

def scheduler_names() -> List[str]:
    return schedulers.scheduler_names()

class KSampler:
    SCHEDULERS = scheduler_names()
    SAMPLERS = sampler_names()
    DISCARD_PENULTIMATE_SIGMA_SAMPLERS = set(('dpm_2', 'dpm_2_ancestral', 'uni_pc', 'uni_pc_bh2'))

    def __init__(self, model: Any, steps: int, device: torch.device, sampler: str = None, scheduler: str = None, denoise: float = None, model_options: Dict[str, Any] = {}):
        self.model = model
        self.device = device
        if scheduler not in self.SCHEDULERS: scheduler = self.SCHEDULERS[0]
        if sampler not in self.SAMPLERS: sampler = self.SAMPLERS[0]
        self.scheduler = scheduler
        self.sampler = sampler
        self.denoise = denoise
        self.model_options = model_options
        self.quality = model_options.get("quality", {})
        
        # Apply quality patches to UNet (Timed ADM, precision casting)
        from . import loader
        loader.patch_unet_for_quality(self.model, self.quality)

        self.set_steps(steps, denoise)

    def calculate_sigmas(self, steps: int) -> torch.Tensor:
        discard_penultimate_sigma = False
        if self.sampler in self.DISCARD_PENULTIMATE_SIGMA_SAMPLERS:
            steps += 1
            discard_penultimate_sigma = True
        
        model_sampling = self.model.get_model_object("model_sampling")
        sigmas = schedulers.calculate_sigmas(model_sampling, self.scheduler, steps, model=self.model.model)
        
        if discard_penultimate_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas

    def set_steps(self, steps: int, denoise: float = None):
        self.steps = steps
        if denoise is None or denoise > 0.9999:
            self.sigmas = self.calculate_sigmas(steps).to(self.device)
        else:
            if denoise <= 0.0:
                self.sigmas = torch.FloatTensor([])
            else:
                new_steps = int(steps/denoise)
                sigmas = self.calculate_sigmas(new_steps).to(self.device)
                self.sigmas = sigmas[-(steps + 1):]

    def sample(self, noise, positive, negative, cfg, latent_image=None, start_step=None, last_step=None, force_full_denoise=False, denoise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
        if sigmas is None: sigmas = self.sigmas
        if last_step is not None and last_step < (len(sigmas) - 1):
            sigmas = sigmas[:last_step + 1]
            if force_full_denoise: sigmas[-1] = 0
        if start_step is not None:
            if start_step < (len(sigmas) - 1): sigmas = sigmas[start_step:]
            else: return latent_image if latent_image is not None else torch.zeros_like(noise)

        sampler_inst = ksampler(self.sampler)
        cfg_guider = prepare_sampler_conds(
            self.model,
            noise,
            positive,
            negative,
            cfg,
            sampler_name=self.sampler,
            latent_image=latent_image,
            denoise_mask=denoise_mask,
            seed=seed,
            model_options=self.model_options,
            quality=self.quality,
        )

        with precision.autocast_context(self.device):
            return sample_prepared_sdxl(
                cfg_guider,
                noise,
                sigmas,
                sampler=sampler_inst,
                latent_image=latent_image,
                denoise_mask=denoise_mask,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=seed,
                attach_model=True,
            )

_sampler_trace_stats = {}
def reset_sampler_trace_stats():
    _sampler_trace_stats.clear()
def consume_sampler_trace_stats():
    snapshot = dict(_sampler_trace_stats)
    reset_sampler_trace_stats()
    return snapshot
def _record_sampler_trace(name, *, wall_seconds, cpu_process_seconds):
    entry = _sampler_trace_stats.setdefault(name, {
        "calls": 0,
        "wall_seconds": 0.0,
        "cpu_process_seconds": 0.0,
    })
    entry["calls"] += 1
    entry["wall_seconds"] += wall_seconds
    entry["cpu_process_seconds"] += cpu_process_seconds
def cfg_function(model: Any, cond_pred: torch.Tensor, uncond_pred: torch.Tensor, cond_scale: float, x: torch.Tensor, timestep: torch.Tensor, model_options: Dict[str, Any] = {}, cfg_pp: bool = False, adaptive_cfg: float = 0.0, diffusion_progress: float = 0.0) -> torch.Tensor:
    if "sampler_cfg_function" in model_options:
        args = {
            "cond_denoised": cond_pred, 
            "uncond_denoised": uncond_pred, 
            "cond_scale": cond_scale, 
            "timestep": timestep, 
            "input": x, 
            "model": model, 
            "model_options": model_options
        }
        return x - model_options["sampler_cfg_function"](args)
    
    # Fooocus Adaptive CFG
    if adaptive_cfg > 0.0 and cond_scale > adaptive_cfg:
        # Scale terms to EPS for Fooocus logic similarity
        cond_eps = x - cond_pred
        uncond_eps = x - uncond_pred
        
        real_eps = uncond_eps + cond_scale * (cond_eps - uncond_eps)
        mimic_eps = uncond_eps + adaptive_cfg * (cond_eps - uncond_eps)
        
        # Blend by progress: real_eps * progress + mimic_eps * (1 - progress)
        final_eps = real_eps * diffusion_progress + mimic_eps * (1.0 - diffusion_progress)
        return x - final_eps

    if cfg_pp:
        # CFG++: cond + (scale - 1) * (cond - uncond)
        cfg_result = cond_pred + (cond_scale - 1.0) * (cond_pred - uncond_pred)
    else:
        # Standard CFG: uncond + scale * (cond - uncond)
        # ComfyUI often uses: uncond + scale * (cond - uncond)
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale
    
    for fn in model_options.get("sampler_post_cfg_function", []):
        args = {
            "denoised": cfg_result, 
            "cond_denoised": cond_pred, 
            "uncond_denoised": uncond_pred, 
            "cond_scale": cond_scale, 
            "model": model, 
            "sigma": timestep, 
            "model_options": model_options, 
            "input": x
        }
        cfg_result = fn(args)
        
    return cfg_result

def sampling_function(model: Any, x: torch.Tensor, timestep: torch.Tensor, uncond: Any, cond: Any, cond_scale: float, model_options: Dict[str, Any] = {}, seed: Optional[int] = None, cfg_pp: bool = False, sharpness: float = 0.0, adaptive_cfg: float = 0.0) -> torch.Tensor:
    # Calculate diffusion progress (0.0 -> 1.0)
    # sigma is passed as timestep
    progress_start = time.perf_counter()
    progress_cpu_start = time.process_time()
    model_sampling = model.model_sampling
    t = model_sampling.timestep(timestep)
    diffusion_progress = max(0.0, min(1.0, 1.0 - t.item() / 999.0))
    _record_sampler_trace(
        "progress",
        wall_seconds=time.perf_counter() - progress_start,
        cpu_process_seconds=time.process_time() - progress_cpu_start,
    )
    if math.isclose(cond_scale, 1.0) and not model_options.get("disable_cfg1_optimization", False):
        uncond_ = None
    else:
        uncond_ = uncond
    conds = [cond, uncond_]
    cond_batch_start = time.perf_counter()
    cond_batch_cpu_start = time.process_time()
    if "sampler_calc_cond_batch_function" in model_options:
        args = {"conds": conds, "input": x, "sigma": timestep, "model": model, "model_options": model_options}
        out = model_options["sampler_calc_cond_batch_function"](args)
    else:
        out = calc_cond_batch(model, conds, x, timestep, model_options)
    _record_sampler_trace(
        "calc_cond_batch",
        wall_seconds=time.perf_counter() - cond_batch_start,
        cpu_process_seconds=time.process_time() - cond_batch_cpu_start,
    )
    # out[0] is positive_x0, out[1] is negative_x0
    cond_pred = out[0]
    uncond_pred = out[1]
    # Fooocus Sharpness (Anisotropic Filtering)
    sharpness_start = time.perf_counter()
    sharpness_cpu_start = time.process_time()
    if sharpness > 0.0:
        alpha = 0.001 * sharpness * diffusion_progress
        if alpha >= 0.01:
            positive_eps = x - cond_pred
            # adaptive_anisotropic_filter(x=eps, g=x0)
            degraded_eps = anisotropic.adaptive_anisotropic_filter(x=positive_eps, g=cond_pred)
            # Blend: degraded * alpha + original * (1 - alpha)
            positive_eps_weighted = degraded_eps * alpha + positive_eps * (1.0 - alpha)
            # Update cond_pred (x0) back from weighted eps
            cond_pred = x - positive_eps_weighted
    _record_sampler_trace(
        "sharpness",
        wall_seconds=time.perf_counter() - sharpness_start,
        cpu_process_seconds=time.process_time() - sharpness_cpu_start,
    )
    pre_cfg_start = time.perf_counter()
    pre_cfg_cpu_start = time.process_time()
    for fn in model_options.get("sampler_pre_cfg_function", []):
        args = {"conds": conds, "conds_out": [cond_pred, uncond_pred], "cond_scale": cond_scale, "timestep": timestep,
                "input": x, "sigma": timestep, "model": model, "model_options": model_options}
        out = fn(args)
        cond_pred = out[0]
        uncond_pred = out[1]
    _record_sampler_trace(
        "pre_cfg_hooks",
        wall_seconds=time.perf_counter() - pre_cfg_start,
        cpu_process_seconds=time.process_time() - pre_cfg_cpu_start,
    )
    cfg_start = time.perf_counter()
    cfg_cpu_start = time.process_time()
    result = cfg_function(model, cond_pred, uncond_pred, cond_scale, x, timestep, model_options=model_options, cfg_pp=cfg_pp, adaptive_cfg=adaptive_cfg, diffusion_progress=diffusion_progress)
    _record_sampler_trace(
        "cfg_function",
        wall_seconds=time.perf_counter() - cfg_start,
        cpu_process_seconds=time.process_time() - cfg_cpu_start,
    )
    return result

def _begin_sampling_trace_capture():
    reset_sampler_trace_stats()
    reset_cond_batch_trace_stats()
    apply_model_trace = None
    try:
        from ldm_patched.modules import model_base as apply_model_trace
        apply_model_trace.reset_apply_model_trace_stats()
    except Exception:
        apply_model_trace = None

    return apply_model_trace


def _emit_sampling_perf_logs(
    *,
    model_load_duration: float,
    cond_duration: float,
    denoise_duration: float,
    denoise_cpu_duration: float,
    total_duration: float,
    apply_model_trace: Any,
):
    perf_stats: Dict[str, Any] = {
        "sampler_trace": {},
        "cond_batch_trace": {},
        "apply_model_trace": {},
    }
    perf_message = (
        f"[Nex-Perf] sampler timings model_load={model_load_duration:.3f}s "
        f"cond_prep={cond_duration:.3f}s denoise={denoise_duration:.3f}s "
        f"denoise_cpu_proc={denoise_cpu_duration:.3f}s total={total_duration:.3f}s"
    )
    print(perf_message)
    logging.info(perf_message)


    sampler_trace = consume_sampler_trace_stats()
    if sampler_trace:
        perf_stats["sampler_trace"] = sampler_trace
        sampler_parts = []
        for trace_name, trace_stats in sorted(sampler_trace.items(), key=lambda item: item[1].get('wall_seconds', 0.0), reverse=True):
            calls = trace_stats.get('calls', 0) or 1
            avg_trace_ms = (trace_stats.get('wall_seconds', 0.0) / calls) * 1000.0
            sampler_parts.append(
                f"{trace_name}:calls={trace_stats.get('calls', 0)},wall={trace_stats.get('wall_seconds', 0.0):.3f}s,"
                f"cpu_proc={trace_stats.get('cpu_process_seconds', 0.0):.3f}s,avg={avg_trace_ms:.3f}ms"
            )
        sampler_trace_message = f"[Nex-Perf] sampler function trace {'; '.join(sampler_parts)}"
        print(sampler_trace_message)
        logging.info(sampler_trace_message)

    cond_batch_trace = consume_cond_batch_trace_stats()
    if cond_batch_trace:
        perf_stats["cond_batch_trace"] = cond_batch_trace
        cond_batch_parts = []
        for trace_name, trace_stats in sorted(cond_batch_trace.items(), key=lambda item: item[1].get('wall_seconds', 0.0), reverse=True):
            calls = trace_stats.get('calls', 0) or 1
            avg_trace_ms = (trace_stats.get('wall_seconds', 0.0) / calls) * 1000.0
            cond_batch_parts.append(
                f"{trace_name}:calls={trace_stats.get('calls', 0)},wall={trace_stats.get('wall_seconds', 0.0):.3f}s,"
                f"cpu_proc={trace_stats.get('cpu_process_seconds', 0.0):.3f}s,avg={avg_trace_ms:.3f}ms"
            )
        cond_batch_trace_message = f"[Nex-Perf] cond batch trace {'; '.join(cond_batch_parts)}"
        print(cond_batch_trace_message)
        logging.info(cond_batch_trace_message)

    if apply_model_trace is not None:
        apply_trace = apply_model_trace.consume_apply_model_trace_stats()
        if apply_trace:
            perf_stats["apply_model_trace"] = apply_trace
            apply_parts = []
            for trace_name, trace_stats in sorted(apply_trace.items(), key=lambda item: item[1].get('wall_seconds', 0.0), reverse=True):
                calls = trace_stats.get('calls', 0) or 1
                avg_trace_ms = (trace_stats.get('wall_seconds', 0.0) / calls) * 1000.0
                apply_parts.append(
                    f"{trace_name}:calls={trace_stats.get('calls', 0)},wall={trace_stats.get('wall_seconds', 0.0):.3f}s,"
                    f"cpu_proc={trace_stats.get('cpu_process_seconds', 0.0):.3f}s,avg={avg_trace_ms:.3f}ms"
                )
            apply_trace_message = f"[Nex-Perf] apply_model trace {'; '.join(apply_parts)}"
            print(apply_trace_message)
            logging.info(apply_trace_message)
    return perf_stats


def prepare_sampler_conds(
    model: Any,
    noise: torch.Tensor,
    positive: Any,
    negative: Any,
    cfg: float,
    *,
    sampler_name: str,
    latent_image: Optional[torch.Tensor] = None,
    denoise_mask: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
    model_options: Optional[Dict[str, Any]] = None,
    quality: Optional[Dict[str, Any]] = None,
    guider: Optional["CFGGuider"] = None,
    inner_model: Any = None,
) -> "CFGGuider":
    model_options = model_options or {}
    cfg_guider = guider or CFGGuider(model)
    cfg_guider.set_conds(positive, negative)
    cfg_guider.set_cfg(cfg, cfg_pp="_cfg_pp" in sampler_name)
    cfg_guider.set_quality(quality if quality is not None else model_options.get("quality", {}))
    cfg_guider.prepare_conds(
        noise,
        latent_image=latent_image,
        denoise_mask=denoise_mask,
        seed=seed,
        inner_model=inner_model,
    )
    return cfg_guider


def sample_prepared_sdxl(
    guider: "CFGGuider",
    noise: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    sampler: Any,
    latent_image: Optional[torch.Tensor] = None,
    denoise_mask: Optional[torch.Tensor] = None,
    callback: Optional[Callable] = None,
    disable_pbar: bool = False,
    seed: Optional[int] = None,
    attach_model: bool = True,
) -> torch.Tensor:
    if sigmas.shape[-1] == 0:
        return latent_image

    if not guider.prepared:
        raise ValueError("Sampler conditions must be prepared before calling sample_prepared_sdxl().")

    guider.ensure_inner_model()
    apply_model_trace = _begin_sampling_trace_capture()
    sample_total_start = time.perf_counter()
    model_load_duration = 0.0

    if attach_model:
        from . import resources
        load_start = time.perf_counter()
        resources.load_models_gpu([guider.model_patcher])
        model_load_duration = time.perf_counter() - load_start

    denoise_start = time.perf_counter()
    denoise_cpu_start = time.process_time()
    try:
        with torch.inference_mode():
            return sampler.sample(guider, sigmas, {}, callback, noise, latent_image, denoise_mask, disable_pbar)
    finally:
        denoise_duration = time.perf_counter() - denoise_start
        denoise_cpu_duration = time.process_time() - denoise_cpu_start
        total_duration = time.perf_counter() - sample_total_start
        perf_stats = _emit_sampling_perf_logs(
            model_load_duration=model_load_duration,
            cond_duration=guider.cond_prep_duration,
            denoise_duration=denoise_duration,
            denoise_cpu_duration=denoise_cpu_duration,
            total_duration=total_duration,
            apply_model_trace=apply_model_trace,
        )
        try:
            guider.last_sampling_perf_stats = dict(perf_stats)
            guider.model_options["_nex_sampling_perf"] = dict(perf_stats)
            guider.model_patcher.model_options["_nex_sampling_perf"] = dict(perf_stats)
        except Exception:
            pass


class CFGGuider:
    def __init__(self, model_patcher: Any):
        self.model_patcher = model_patcher
        self.model_options = getattr(model_patcher, "model_options", {})
        self.original_conds = {}
        self.cfg = 1.0
        self.conds = {}
        self.inner_model = None
        self.cfg_pp = False
        self.quality = {}
        self.prepared = False
        self.cond_prep_duration = 0.0

    def set_conds(self, positive: Any, negative: Any):
        self.inner_set_conds({"positive": positive, "negative": negative})
        self.prepared = False

    def set_cfg(self, cfg: float, cfg_pp: bool = False):
        self.cfg = cfg
        self.cfg_pp = cfg_pp

    def set_quality(self, quality: Dict[str, Any]):
        self.quality = quality

    def inner_set_conds(self, conds: Dict[str, Any]):
        for k, v in conds.items():
            self.original_conds[k] = self.convert_cond(v)

    def convert_cond(self, cond: Any) -> List[Dict[str, Any]]:
        import uuid
        out = []
        if isinstance(cond, list) and len(cond) > 0 and isinstance(cond[0], dict):
            for c in cond:
                temp = c.copy()
                temp["uuid"] = temp.get("uuid", uuid.uuid4())
                out.append(temp)
            return out

        for c in cond:
            temp = c[1].copy()
            model_conds = temp.get("model_conds", {})
            if c[0] is not None:
                temp["cross_attn"] = c[0]
            temp["model_conds"] = model_conds
            temp["uuid"] = uuid.uuid4()
            out.append(temp)
        return out

    def clone_original_conds(self):
        self.conds = {}
        for k in self.original_conds:
            self.conds[k] = [c.copy() for c in self.original_conds[k]]
        self.prepared = False
        return self.conds

    def ensure_inner_model(self, inner_model: Any = None):
        if inner_model is not None:
            self.inner_model = inner_model
        if self.inner_model is None:
            self.inner_model = self.model_patcher.model
        return self.inner_model

    def prepare_conds(
        self,
        noise: torch.Tensor,
        latent_image: Optional[torch.Tensor] = None,
        denoise_mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        inner_model: Any = None,
    ) -> Dict[str, Any]:
        self.clone_original_conds()
        prepared_model = self.ensure_inner_model(inner_model=inner_model)
        cond_start = time.perf_counter()
        self.conds = process_conds(
            prepared_model,
            noise,
            self.conds,
            noise.device,
            latent_image=latent_image,
            denoise_mask=denoise_mask,
            seed=seed,
        )
        self.cond_prep_duration = time.perf_counter() - cond_start
        self.prepared = True
        return self.conds

    def predict_noise(self, x: torch.Tensor, timestep: torch.Tensor, model_options: Dict[str, Any] = {}, seed: Optional[int] = None) -> torch.Tensor:
        return sampling_function(
            self.inner_model,
            x,
            timestep,
            self.conds.get("negative"),
            self.conds.get("positive"),
            self.cfg,
            model_options=model_options,
            seed=seed,
            cfg_pp=self.cfg_pp,
            sharpness=self.quality.get("sharpness", 0.0),
            adaptive_cfg=self.quality.get("adaptive_cfg", 0.0)
        )

    def __call__(self, *args, **kwargs):
        return self.predict_noise(*args, **kwargs)

    def sample(self, noise: torch.Tensor, latent_image: torch.Tensor, sampler: Any, sigmas: torch.Tensor, denoise_mask: Optional[torch.Tensor] = None, callback: Optional[Callable] = None, disable_pbar: bool = False, seed: Optional[int] = None) -> torch.Tensor:
        self.prepare_conds(
            noise,
            latent_image=latent_image,
            denoise_mask=denoise_mask,
            seed=seed,
        )

        return sample_prepared_sdxl(
            self,
            noise,
            sigmas,
            sampler=sampler,
            latent_image=latent_image,
            denoise_mask=denoise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
            attach_model=True,
        )
