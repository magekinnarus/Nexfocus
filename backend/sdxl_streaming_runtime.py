"""Nex-owned production home for the streaming SDXL runtime spine.

Separated from resident runtime execution to ensure clean debugging and
zero cross-mode state leakage.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import os
import time
from collections import OrderedDict
from typing import Any

import torch
from backend import conditioning, decode, lora as backend_lora, loader, precision, resources, sampling
from backend.cpu_compiler import CpuArtifactCompiler, SafeOpenHeaderOnly
from backend.sdxl_runtime_contract import (
    BaseModelAvailability,
    CompiledUnetArtifact,
    GpuAttachedExecutionState,
    InjectedFeatureArtifact,
    PromptConditioningArtifact,
    StructuralConditioningArtifact,
    SpatialConditioningArtifact,
)
from backend.sdxl_unified_runtime import (
    UnifiedSDXLRuntime,
    UnifiedSDXLRuntimeConfig,
    UnifiedSDXLPreparedInputs,
    UnifiedSDXLDenoiseResult,
    SharedSDXLBaseComponents,
    SharedSDXLCompiledUnetComponents,
    _SHARED_SDXL_BASE_COMPONENT_CACHE,
    _SHARED_SDXL_VAE_CACHE,
    _detach_cached_component,
    _is_default_shared_sdxl_vae_selection,
    _load_shared_sdxl_vae_for_device,
    _release_shared_sdxl_base_components,
    _soft_empty_cache_force,
)


_SHARED_SDXL_COMPILED_UNET_CACHE: OrderedDict[
    tuple[str, str, str, tuple[str, ...], bool],
    SharedSDXLCompiledUnetComponents,
] = OrderedDict()
_SHARED_SDXL_COMPILED_UNET_CACHE_LIMIT = 1


def _release_shared_sdxl_compiled_unet(entry: SharedSDXLCompiledUnetComponents) -> None:
    _detach_cached_component(entry.unet)


def clear_streaming_cache() -> None:
    while _SHARED_SDXL_COMPILED_UNET_CACHE:
        _, entry = _SHARED_SDXL_COMPILED_UNET_CACHE.popitem(last=False)
        _release_shared_sdxl_compiled_unet(entry)


class SDXLStreamingRuntime(UnifiedSDXLRuntime):
    """CPU-authoritative streaming SDXL execution runtime."""

    route_label = "sdxl_streaming_runtime"

    def __init__(self, config: UnifiedSDXLRuntimeConfig) -> None:
        super().__init__(config)

    def load_components(self) -> float:
        if self._loaded:
            return 0.0

        checkpoint_path = self._require_checkpoint_path()
        vae_path = self._require_optional_vae_path()
        start = time.perf_counter()
        cpu_device = torch.device("cpu")

        from backend.sdxl_runtime_policy import resolve_sdxl_execution_policy
        self.policy = self.config.runtime_policy or resolve_sdxl_execution_policy(
            architecture="sdxl",
            base_model_name=checkpoint_path,
        )

        cache_key = self._base_component_cache_key(
            checkpoint_path=checkpoint_path,
            is_resident=False,
        )
        allow_default_vae_fallback = _is_default_shared_sdxl_vae_selection(self.config.vae_path)
        vae_to_use = None
        if vae_path:
            vae_cache_key = (vae_path, str(cpu_device), str(cpu_device))
            vae_to_use = _SHARED_SDXL_VAE_CACHE.get(vae_cache_key)
            if vae_to_use is None:
                vae_to_use = _load_shared_sdxl_vae_for_device(
                    vae_path,
                    load_device=cpu_device,
                    offload_device=cpu_device,
                    allow_default_fallback=allow_default_vae_fallback,
                )
                if vae_to_use is not None:
                    _SHARED_SDXL_VAE_CACHE[vae_cache_key] = vae_to_use
        loaded_vae = vae_to_use
        shared_base = _SHARED_SDXL_BASE_COMPONENT_CACHE.get(cache_key)
        if shared_base is not None:
            _SHARED_SDXL_BASE_COMPONENT_CACHE.move_to_end(cache_key)
        self._base_component_cache_hit = shared_base is not None

        if shared_base is None:
            # CPU-authoritative UNet & VAE loading for streaming runtime
            loaded_unet, loaded_clip, loaded_vae = loader.load_sdxl_checkpoint(
                checkpoint_path,
                load_device=cpu_device,
                offload_device=cpu_device,
                unet_dtype=torch.float16,
                clip_load_device=cpu_device,
                clip_offload_device=cpu_device,
                vae_load_device=cpu_device,
                vae_offload_device=cpu_device,
                vae_source=vae_to_use,
            )
            if vae_to_use is None and loaded_vae is not None and allow_default_vae_fallback and vae_path:
                _SHARED_SDXL_VAE_CACHE[vae_cache_key] = loaded_vae
                vae_to_use = loaded_vae

            shared_base = SharedSDXLBaseComponents(
                unet=loaded_unet,
                clip=loaded_clip,
                checkpoint_fingerprint=self._fingerprint_source_path(checkpoint_path),
            )
            self._remember_shared_base_components(cache_key, shared_base)

        self.unet, self.clip = self._clone_shared_base_components(shared_base)
        self.vae = self._clone_component_for_runtime(loaded_vae)
        self._compiled_unet_cache_hit = False

        if self.unet is not None:
            self.unet.runtime_release_to_meta = False
            # Apply scheduler-specific patch to the UNet
            orig_scheduler = self.config.original_scheduler_name or self.config.scheduler
            if orig_scheduler == 'lcm':
                from modules import core as modules_core
                self.unet = modules_core.opModelSamplingDiscrete.patch(self.unet, orig_scheduler, False)[0]
        if self.clip is not None:
            self.clip.runtime_policy = self.policy
            if hasattr(self.clip, "clip_layer"):
                self.clip.clip_layer(self.config.clip_layer)
        if self.vae is not None:
            self.vae.runtime_policy = self.policy

        self._checkpoint_fingerprint = shared_base.checkpoint_fingerprint or self._fingerprint_source_path(checkpoint_path)
        self.base_model = BaseModelAvailability(
            family="sdxl",
            variant=self.config.model_variant,
            source_path=checkpoint_path,
            fingerprint=self._checkpoint_fingerprint,
            loaded=True,
            reusable=True,
        )
        self._clip_identity = self._build_clip_identity()
        self._cold_model_load_cpu = 0.0 if self._base_component_cache_hit else (time.perf_counter() - start)
        cache_fingerprint = (self._checkpoint_fingerprint or checkpoint_path)[:12]
        logging.info(
            "[Nex-BaseModelCache] %s route=%s fingerprint=%s",
            "hit" if self._base_component_cache_hit else "miss",
            self.route_label,
            cache_fingerprint,
        )
        self._loaded = True
        return self._cold_model_load_cpu

    def _is_vae_resident(self) -> bool:
        return False

    def _materialize_lora_stack(self) -> dict[str, float]:
        clip_patch_count = 0
        unet_patch_count = 0
        clip_host_pinned_bytes = 0
        unet_host_pinned_bytes = 0
        self._compiled_unet_cache_hit = False
        streaming_signature = self._streaming_compiled_unet_cache_key()

        current_streaming_signature = self._current_streaming_unet_signature()

        if not self._resolved_lora_specs:
            if current_streaming_signature is not None:
                self._restore_clean_streaming_unet_shell()
            unet_compile_start = time.perf_counter()
            unet_compile = self._compile_patcher(
                self.unet,
                pin_model_host=self._should_pin_unet_host_for_compile(),
            )
            unet_compile_wall = time.perf_counter() - unet_compile_start
            self._remember_streaming_unet_signature(None)
            unet_patch_count = int(unet_compile.get("patch_count", 0))
            unet_host_pinned_bytes = int(unet_compile.get("host_pinned_bytes", 0))
            return {
                "spec_count": 0.0,
                "clip_patch_count": 0.0,
                "unet_patch_count": float(unet_patch_count),
                "clip_host_pinned_bytes": float(clip_host_pinned_bytes),
                "unet_host_pinned_bytes": float(unet_host_pinned_bytes),
                "clip_compile_wall": 0.0,
                "unet_compile_wall": float(unet_compile_wall),
                "compiled_unet_cache_hit": 0.0,
                "clip_compile_metrics": {"status": "noop", "patch_count": 0, "host_pinned_bytes": 0},
                "unet_compile_metrics": unet_compile,
            }

        clip_patch_count = self._apply_lora_specs_to_patcher(
            self.clip.patcher,
            self.clip.patcher.model,
            target_family="clip",
        )
        clip_compile_wall = 0.0
        clip_compile_metrics: dict[str, Any] = {"status": "noop", "patch_count": 0, "host_pinned_bytes": 0}
        if clip_patch_count > 0:
            clip_compile_start = time.perf_counter()
            clip_compile = self._compile_patcher(self.clip.patcher, pin_model_host=False)
            clip_compile_wall = time.perf_counter() - clip_compile_start
            clip_compile_metrics = clip_compile
            clip_host_pinned_bytes = int(clip_compile.get("host_pinned_bytes", 0))

        cache_fingerprint = self._streaming_compiled_unet_cache_fingerprint(streaming_signature)
        self._compiled_unet_cache_hit = current_streaming_signature == streaming_signature
        logging.info(
            "[Nex-CompiledUnetCache] %s route=%s fingerprint=%s",
            "hit" if self._compiled_unet_cache_hit else "miss",
            self.route_label,
            cache_fingerprint[:12],
        )
        if self._compiled_unet_cache_hit:
            unet_compile = self._current_streaming_unet_compile_metrics() or {
                "status": "compiled",
                "patch_count": 0,
                "materialized_patch_keys": 0,
                "host_pinned_bytes": 0,
            }
            unet_compile_wall = 0.0
        else:
            if current_streaming_signature is not None:
                self._restore_clean_streaming_unet_shell()
            unet_compile_start = time.perf_counter()
            self._apply_lora_specs_to_patcher(
                self.unet,
                self.unet.model,
                target_family="unet",
            )
            unet_compile = self._compile_patcher(
                self.unet,
                pin_model_host=self._should_pin_unet_host_for_compile(),
            )
            unet_compile_wall = time.perf_counter() - unet_compile_start
            self._remember_streaming_unet_signature(streaming_signature, unet_compile)
        self.unet.runtime_release_to_meta = False

        unet_patch_count = int(unet_compile.get("patch_count", 0))
        unet_host_pinned_bytes = int(unet_compile.get("host_pinned_bytes", 0))

        return {
            "spec_count": float(len(self._resolved_lora_specs)),
            "clip_patch_count": float(clip_patch_count),
            "unet_patch_count": float(unet_patch_count),
            "clip_host_pinned_bytes": float(clip_host_pinned_bytes),
            "unet_host_pinned_bytes": float(unet_host_pinned_bytes),
            "clip_compile_wall": float(clip_compile_wall),
            "unet_compile_wall": float(unet_compile_wall),
            "compiled_unet_cache_hit": 1.0 if self._compiled_unet_cache_hit else 0.0,
            "clip_compile_metrics": clip_compile_metrics,
            "unet_compile_metrics": unet_compile,
        }

    def denoise_prepared_inputs(
        self,
        prepared_inputs: UnifiedSDXLPreparedInputs,
        *,
        callback: Any = None,
        disable_pbar: bool = True,
    ) -> UnifiedSDXLDenoiseResult:
        _soft_empty_cache_force()
        self.load_components()
        self._validate_prepared_inputs(prepared_inputs)
        self.prepared_inputs = prepared_inputs

        attach_device = self._execution_device()
        budget_bytes = self._clean_unet_budget_bytes(attach_device)
        headroom_mb = self._device_headroom_mb(attach_device)
        unet_attach_start = time.perf_counter()
        self._attach_compiled_unet(attach_device, budget_bytes=budget_bytes)
        unet_attach_wall = time.perf_counter() - unet_attach_start

        conditioning_attach_start = time.perf_counter()
        attached_payload = self._build_attached_payload(prepared_inputs, attach_device)
        conditioning_attach_wall = time.perf_counter() - conditioning_attach_start
        self._attached_payload = attached_payload

        state = self._transition_execution_state(
            prepared_inputs,
            active_phase="diffusion",
            attached_component_ids=self._build_attached_component_ids(prepared_inputs),
            device=attach_device,
            stream_budget_mb=float(budget_bytes) / (1024 * 1024),
            headroom_mb=headroom_mb,
        )

        denoise_start = time.perf_counter()
        denoise_cpu_start = time.process_time()
        try:
            with torch.inference_mode(), precision.autocast_context(attach_device):
                samples = self._run_prepared_denoise(
                    attached_payload,
                    device=attach_device,
                    callback=callback,
                    disable_pbar=disable_pbar,
                )
        finally:
            self._park_compiled_unet_before_decode()
        denoise_wall = time.perf_counter() - denoise_start
        denoise_cpu_proc = time.process_time() - denoise_cpu_start
        latent_cpu = samples.detach().cpu()

        metrics = {
            "execution_device": 1.0 if attach_device.type == "cuda" else 0.0,
            "unet_attach_cpu": float(unet_attach_wall),
            "conditioning_attach_cpu": float(conditioning_attach_wall),
            "prepared_conditioning_reused": 1.0,
            "prepared_unet_reused": 1.0,
            "prepared_structural_reused": 1.0 if prepared_inputs.structural_conditioning is not None else 0.0,
            "prepared_spatial_reused": 1.0 if prepared_inputs.spatial_conditioning is not None else 0.0,
            "denoise_mask_attached": 1.0 if attached_payload.get("denoise_mask") is not None else 0.0,
            "attached_component_count": float(len(state.attached_component_ids)),
            "stream_budget_mb": float(state.stream_budget_mb),
            "headroom_mb": float(state.headroom_mb),
            "denoise_wall": float(denoise_wall),
            "denoise_cpu_proc": float(denoise_cpu_proc),
            "cond_prepare_explicit": float(attached_payload.get("cond_prepare_duration", 0.0)),
        }
        return UnifiedSDXLDenoiseResult(
            samples=latent_cpu,
            execution_state=state,
            metrics=metrics,
        )

    def decode_latent(self, latent: torch.Tensor, tiled: bool = False) -> tuple[torch.Tensor, float, float]:
        self.load_components()
        decode_device = self._execution_device()
        self._park_compiled_unet_before_decode()

        # Enforce explicit decode-stage memory separation: reclaim UNet memory
        gc.collect()
        _soft_empty_cache_force()

        attach_start = time.perf_counter()
        self._attach_vae(decode_device)
        vae_attach = time.perf_counter() - attach_start

        self._transition_execution_state(
            self.prepared_inputs,
            active_phase="decode",
            attached_component_ids=self._build_decode_component_ids(),
            device=decode_device,
            stream_budget_mb=0.0,
            headroom_mb=self._device_headroom_mb(decode_device),
        )

        decode_start = time.perf_counter()
        try:
            with torch.inference_mode():
                decoded_patch = decode.decode_preloaded_vae(self.vae, latent, tiled=tiled)
                images = self._compose_decoded_images(decoded_patch)
        finally:
            resources.eject_model(getattr(self.vae, "patcher", None))
            _soft_empty_cache_force()
            self._attached_payload = None
            self._transition_execution_state(
                self.prepared_inputs,
                active_phase="finalize",
                attached_component_ids=(),
                device=decode_device,
                stream_budget_mb=0.0,
                headroom_mb=self._device_headroom_mb(decode_device),
            )
        vae_decode = time.perf_counter() - decode_start
        return images, vae_attach, vae_decode

    def close(self) -> None:
        self._park_compiled_unet_before_decode()
        self._detach_component(self.unet)
        resources.eject_model(getattr(self.vae, "patcher", None))
        self._detach_component(getattr(self.clip, "patcher", None))
        self._attached_payload = None
        self.execution_state = None
        self.prepared_inputs = None
        self.base_model = None
        self.compiled_unet = None
        self.conditioning = None
        self.structural_conditioning = None
        self.spatial_conditioning = None
        self.injected_features = {}
        self.unet = None
        self.clip = None
        self.vae = None
        self._unload_controlnets()
        self._loaded = False
        self._base_component_cache_hit = False
        self._compiled_unet_cache_hit = False
        self._prepare_metrics = {}
        self._checkpoint_fingerprint = None
        gc.collect()
        _soft_empty_cache_force()

    def _park_compiled_unet_before_decode(self) -> None:
        if self._park_streaming_compiled_unet_shell():
            return
        self._detach_component(self.unet)

    def _streaming_compiled_unet_cache_key(self) -> tuple[str, str, str, tuple[str, ...], bool]:
        source_fingerprint = self._checkpoint_fingerprint or self._require_checkpoint_path()
        scheduler_identity = str(self.config.original_scheduler_name or self.config.scheduler or "")
        return (
            str(source_fingerprint),
            self._execution_class_label(),
            scheduler_identity,
            self._lora_signature(),
            bool(self.config.pin_base_unet_without_lora),
        )

    def _streaming_compiled_unet_cache_fingerprint(
        self,
        cache_key: tuple[str, str, str, tuple[str, ...], bool],
    ) -> str:
        return self._hash_payload(cache_key)

    def _remember_shared_compiled_unet(
        self,
        cache_key: tuple[str, str, str, tuple[str, ...], bool],
        entry: SharedSDXLCompiledUnetComponents,
    ) -> None:
        _SHARED_SDXL_COMPILED_UNET_CACHE[cache_key] = entry
        _SHARED_SDXL_COMPILED_UNET_CACHE.move_to_end(cache_key)
        while len(_SHARED_SDXL_COMPILED_UNET_CACHE) > _SHARED_SDXL_COMPILED_UNET_CACHE_LIMIT:
            _, evicted = _SHARED_SDXL_COMPILED_UNET_CACHE.popitem(last=False)
            _release_shared_sdxl_compiled_unet(evicted)

    def _build_streaming_compiled_unet_entry(
        self,
        cache_key: tuple[str, str, str, tuple[str, ...], bool],
    ) -> SharedSDXLCompiledUnetComponents:
        source_unet = getattr(self.unet, "isolated_clone", None)
        if callable(source_unet):
            compiled_unet = source_unet()
        else:
            compiled_unet = self._clone_component_for_runtime(self.unet)

        if compiled_unet is None:
            raise RuntimeError("Compiled UNet cache requested without a source UNet shell.")

        compiled_unet.runtime_release_to_meta = False
        if self._resolved_lora_specs:
            self._apply_lora_specs_to_patcher(
                compiled_unet,
                compiled_unet.model,
                target_family="unet",
            )
        compile_metrics = self._compile_patcher(
            compiled_unet,
            pin_model_host=self._should_pin_unet_host_for_compile(),
        )
        artifact_fingerprint = self._build_compiled_unet_fingerprint(
            unet_compile_metrics=compile_metrics,
        )
        return SharedSDXLCompiledUnetComponents(
            unet=compiled_unet,
            compile_metrics=dict(compile_metrics),
            artifact_fingerprint=artifact_fingerprint,
            cache_fingerprint=self._streaming_compiled_unet_cache_fingerprint(cache_key),
        )

    def _current_streaming_unet_signature(self) -> Any:
        model = getattr(self.unet, "model", None)
        return getattr(model, "_nex_streaming_unet_signature", None) if model is not None else None

    def _current_streaming_unet_compile_metrics(self) -> dict[str, Any] | None:
        model = getattr(self.unet, "model", None)
        metrics = getattr(model, "_nex_streaming_unet_compile_metrics", None) if model is not None else None
        return dict(metrics) if isinstance(metrics, dict) else None

    def _remember_streaming_unet_signature(
        self,
        signature: tuple[str, str, str, tuple[str, ...], bool] | None,
        compile_metrics: dict[str, Any] | None = None,
    ) -> None:
        model = getattr(self.unet, "model", None)
        if model is None:
            return
        setattr(model, "_nex_streaming_unet_signature", signature)
        if compile_metrics is None:
            if hasattr(model, "_nex_streaming_unet_compile_metrics"):
                delattr(model, "_nex_streaming_unet_compile_metrics")
            return
        setattr(model, "_nex_streaming_unet_compile_metrics", dict(compile_metrics))

    def _restore_clean_streaming_unet_shell(self) -> bool:
        if self.unet is not None:
            self.unet.unpatch_model(unpatch_weights=True)
            self._clear_patcher_artifacts(self.unet)
        runtime_reload = getattr(self.unet, "runtime_reload", None)
        if not callable(runtime_reload):
            return False
        target_device = getattr(self.unet, "offload_device", None) or torch.device("cpu")
        runtime_reload(self.unet.model, target_device)
        self.unet.model.device = target_device
        self._remember_streaming_unet_signature(None)
        return True

    def _should_pin_unet_host_for_compile(self) -> bool:
        if not self._resolved_lora_specs:
            return bool(self.config.pin_base_unet_without_lora)
        # Streaming SDXL already has a clean CPU source of truth. Pinning the
        # compiled LoRA shell forces a second near-full host copy during
        # materialization, which is too expensive for the 32 GB baseline.
        return False

    def _park_streaming_compiled_unet_shell(self) -> bool:
        if self.unet is None or not self._resolved_lora_specs:
            return False
        if self._current_streaming_unet_signature() is None:
            return False

        offload_device = getattr(self.unet, "offload_device", None) or torch.device("cpu")
        current_device_getter = getattr(self.unet, "current_loaded_device", None)
        if callable(current_device_getter):
            current_device = current_device_getter()
        else:
            current_device = getattr(getattr(self.unet, "model", None), "device", None)

        if current_device is None:
            current_device = offload_device
        if not isinstance(current_device, torch.device):
            current_device = torch.device(current_device)

        if current_device == offload_device:
            model = getattr(self.unet, "model", None)
            if model is not None:
                model.device = offload_device
            if hasattr(self.unet, "current_device"):
                self.unet.current_device = offload_device
            return True

        partial_unload = getattr(self.unet, "partially_unload", None)
        if not callable(partial_unload):
            return False

        memory_to_free = 0
        loaded_size = getattr(self.unet, "loaded_size", None)
        if callable(loaded_size):
            try:
                memory_to_free = int(loaded_size())
            except Exception:
                memory_to_free = 0
        if memory_to_free <= 0:
            memory_to_free = int(getattr(getattr(self.unet, "model", None), "model_loaded_weight_memory", 0) or 0)
        if memory_to_free <= 0:
            memory_to_free = max(1, int(self.unet.model_size()))

        try:
            partial_unload(offload_device, memory_to_free=memory_to_free)
            model = getattr(self.unet, "model", None)
            if model is not None:
                model.device = offload_device
            if hasattr(self.unet, "current_device"):
                self.unet.current_device = offload_device
            return True
        except Exception:
            logging.debug("Failed to park compiled streaming UNet shell without unpatching.", exc_info=True)
            return False
