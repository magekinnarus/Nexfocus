"""Nex-owned clean-slate resident runtime managing resident-only lifecycles for UNet, CLIP, and VAE.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import os
import time
from types import SimpleNamespace
from typing import Any

import torch
from safetensors import safe_open
from backend import conditioning, decode, k_diffusion, loader, precision, resources, sampling
from backend.cpu_compiler import CpuArtifactCompiler, SafeOpenHeaderOnly
from backend.gpu_compiler import GpuArtifactCompiler
from backend.lora_artifacts import compute_file_hash
from backend.sdxl_runtime_contract import (
    BaseModelAvailability,
    CompiledUnetArtifact,
    GpuAttachedExecutionState,
    InjectedFeatureArtifact,
    PromptConditioningArtifact,
    StructuralConditioningArtifact,
    SpatialConditioningArtifact,
    UnifiedSDXLRuntimeProtocol,
    UnifiedSDXLRuntimeSeams,
)
from backend.sdxl_unified_runtime_artifacts import (
    UnifiedSDXLRuntimeArtifactMixin,
    clear_spatial_latent_cache,
)
from backend.sdxl_unified_runtime_execution import UnifiedSDXLRuntimeExecutionMixin
from backend import lora as backend_lora
from backend.sdxl_unified_runtime import (
    UnifiedSDXLRuntime,
    UnifiedSDXLRuntimeConfig,
    UnifiedSDXLPreparedInputs,
    UnifiedSDXLDenoiseResult,
    _SHARED_SDXL_BASE_COMPONENT_CACHE,
    _SHARED_SDXL_VAE_CACHE,
    _is_default_shared_sdxl_vae_selection,
    _load_shared_sdxl_vae_for_device,
    _release_shared_sdxl_base_components,
    _resolve_shared_sdxl_vae_path,
    _looks_like_shared_sdxl_vae_asset,
    _soft_empty_cache_force,
    SharedSDXLBaseComponents,
    _get_lora_file_hash,
    _PARSED_LORA_ADAPTER_CACHE,
    _PARSED_LORA_ADAPTER_CACHE_LIMIT,
)



class _ResidentLazyWeight:
    def __init__(
        self,
        path: str,
        key: str,
        shape: list[int],
        dtype: str,
        *,
        tensor_device: torch.device,
        load_strategy: str = "safetensors",
    ) -> None:
        self.path = path
        self.key = key
        self.shape = list(shape)
        self.dtype = str(dtype)
        self._tensor_device = torch.device(tensor_device)
        self._load_strategy = str(load_strategy)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def load(self) -> torch.Tensor:
        if self._load_strategy == "safetensors":
            with safe_open(self.path, framework="pt", device=str(self._tensor_device)) as handle:
                return handle.get_tensor(self.key)
        try:
            sd = torch.load(self.path, map_location="cpu", weights_only=True)
        except Exception:
            sd = torch.load(self.path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        tensor = sd.get(self.key) if isinstance(sd, dict) else None
        if not isinstance(tensor, torch.Tensor):
            raise KeyError(f"Legacy tensor key {self.key!r} could not be reloaded from {self.path!r}.")
        return tensor.to(device=self._tensor_device)

    def item(self):
        return self.load().item()


class _ResidentSafeOpenHeaderOnly(dict):
    def __init__(self, path: str, *, tensor_device: torch.device) -> None:
        super().__init__()
        self.path = path
        self.tensor_device = torch.device(tensor_device)
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    try:
                        slice_view = handle.get_slice(key)
                        shape = list(slice_view.get_shape())
                        dtype = str(slice_view.get_dtype())
                    except Exception:
                        tensor = handle.get_tensor(key)
                        shape = list(tensor.shape)
                        dtype = str(tensor.dtype)
                    self[key] = _ResidentLazyWeight(
                        path,
                        key,
                        shape,
                        dtype,
                        tensor_device=self.tensor_device,
                        load_strategy="safetensors",
                    )
        except Exception:
            try:
                sd = torch.load(path, map_location="cpu", weights_only=True)
            except Exception:
                sd = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            for key, tensor in sd.items():
                if isinstance(tensor, torch.Tensor):
                    self[key] = _ResidentLazyWeight(
                        path,
                        key,
                        list(tensor.shape),
                        str(tensor.dtype),
                        tensor_device=self.tensor_device,
                        load_strategy="torch_load",
                    )
                else:
                    self[key] = tensor


class ResidentSDXLRuntime(UnifiedSDXLRuntime):
    """Resident-only SDXL runtime spine."""

    route_label = "sdxl_resident_runtime"

    seams = UnifiedSDXLRuntimeSeams(
        task_start_owner="modules.async_worker",
        prompt_conditioning_owner="backend.conditioning",
        compiled_unet_owner="backend.sdxl_resident_runtime",
        denoise_owner="backend.sdxl_resident_runtime",
        decode_owner="backend.sdxl_resident_runtime",
    )

    def __init__(self, config: UnifiedSDXLRuntimeConfig) -> None:
        self.config = config
        self.unet: Any = None
        self.clip: Any = None
        self.vae: Any = None
        self.policy: Any = None
        self.base_model: BaseModelAvailability | None = None
        self.compiled_unet: CompiledUnetArtifact | None = None
        self.conditioning: PromptConditioningArtifact | None = None
        self.structural_conditioning: StructuralConditioningArtifact | None = None
        self.spatial_conditioning: SpatialConditioningArtifact | None = None
        self.injected_features: dict[str, InjectedFeatureArtifact] = {}
        self.execution_state: GpuAttachedExecutionState | None = None
        self.prepared_inputs: UnifiedSDXLPreparedInputs | None = None
        self._loaded = False
        self._cold_model_load_cpu = 0.0
        self._base_component_cache_hit = False
        self._compiled_unet_cache_hit = False
        self._prepare_metrics: dict[str, float] = {}
        self._resolved_lora_specs: tuple[tuple[str, float], ...] = self._normalize_lora_specs(
            self.config.lora_specs
        )
        self._checkpoint_fingerprint: str | None = None
        self._clip_identity: str = self._build_clip_identity()
        self._attached_payload: dict[str, Any] | None = None
        self._loaded_controlnets: dict[str, Any] = {}
        self._borrowed_controlnet_paths: set[str] = set()
        self._structural_controlnets_prefetched = False
        self._cold_load_metric = 0.0
        self._warm_reuse_metric = 0.0
        self._true_invalidation_metric = 0.0
        self._lora_parse_cold_time = 0.0
        self._lora_parse_warm_time = 0.0
        self._lora_parse_hits = 0.0
        self._lora_parse_misses = 0.0

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

        is_resident = True
        cache_key = self._base_component_cache_key(
            checkpoint_path=checkpoint_path,
            is_resident=is_resident,
        )
        allow_default_vae_fallback = _is_default_shared_sdxl_vae_selection(self.config.vae_path)
        vae_source = None
        if vae_path:
            vae_cache_key = (vae_path, str(cpu_device), str(cpu_device))
            vae_source = _SHARED_SDXL_VAE_CACHE.get(vae_cache_key)
            if vae_source is None:
                vae_source = _load_shared_sdxl_vae_for_device(
                    vae_path,
                    load_device=cpu_device,
                    offload_device=cpu_device,
                    allow_default_fallback=allow_default_vae_fallback,
                )
                if vae_source is not None:
                    _SHARED_SDXL_VAE_CACHE[vae_cache_key] = vae_source
        loaded_vae = vae_source
        shared_base = _SHARED_SDXL_BASE_COMPONENT_CACHE.get(cache_key)
        if shared_base is not None:
            _SHARED_SDXL_BASE_COMPONENT_CACHE.move_to_end(cache_key)
        self._base_component_cache_hit = shared_base is not None

        if shared_base is None:
            cuda_device = resources.get_torch_device()
            # Resident SDXL keeps the checkpoint-weight UNet authoritative on GPU.
            # The shared VAE remains CPU-cached and only attaches when decode work needs it.
            loaded_unet, loaded_clip, loaded_vae = loader.load_sdxl_checkpoint(
                checkpoint_path,
                load_device=cuda_device,
                offload_device=cuda_device,
                unet_dtype=torch.float16,
                clip_load_device=cpu_device,
                clip_offload_device=cpu_device,
                vae_load_device=cpu_device,
                vae_offload_device=cpu_device,
                vae_source=vae_source,
            )
            if vae_source is None and loaded_vae is not None and allow_default_vae_fallback and vae_path:
                _SHARED_SDXL_VAE_CACHE[vae_cache_key] = loaded_vae
                vae_source = loaded_vae

            shared_base = SharedSDXLBaseComponents(
                unet=loaded_unet,
                clip=loaded_clip,
                checkpoint_fingerprint=self._fingerprint_source_path(checkpoint_path),
            )
            self._remember_shared_base_components(cache_key, shared_base)

        self.unet, self.clip = self._clone_shared_base_components(shared_base)
        self.vae = self._clone_component_for_runtime(loaded_vae)
        self._compiled_unet_cache_hit = False
        self._discard_legacy_clean_snapshots()

        if self.unet is not None:
            # Resident SDXL always reloads clean UNet weights from the runtime source.
            # The deprecated CPU/GPU clean-shadow policy is intentionally ignored here.
            self.unet.runtime_release_to_meta = True
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

        from backend import process_transition
        active_key = process_transition.get_active_process_key()
        is_invalidation = False
        is_cold_load = False
        if not self._base_component_cache_hit:
            if active_key is not None:
                is_invalidation = True
            elif len(_SHARED_SDXL_BASE_COMPONENT_CACHE) > 0:
                is_invalidation = True
            else:
                is_cold_load = True

        self._cold_load_metric = 1.0 if is_cold_load else 0.0
        self._warm_reuse_metric = 1.0 if self._base_component_cache_hit else 0.0
        self._true_invalidation_metric = 1.0 if is_invalidation else 0.0

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

    def prepare_inputs(self) -> tuple[UnifiedSDXLPreparedInputs, dict[str, float]]:
        if self.prepared_inputs is not None:
            return self.prepared_inputs, dict(self._prepare_metrics)

        self.load_components()
        if self.base_model is None or self.clip is None or self.unet is None:
            raise RuntimeError("Resident SDXL runtime failed to load the base components.")

        # Materialize UNet LoRAs (handles UNet Clean-Source Snapshot lifecycle)
        lora_metrics = self._materialize_lora_stack()

        prompt_stage = conditioning.build_sdxl_text_conditioning_fingerprint(
            prompt=self.config.prompt,
            negative_prompt=self.config.negative_prompt,
            positive_texts=self.config.positive_texts or (self.config.prompt,),
            negative_texts=self.config.negative_texts or (self.config.negative_prompt,),
            positive_top_k=self.config.positive_top_k,
            negative_top_k=self.config.negative_top_k,
            model_identity=self.base_model.fingerprint or self.base_model.source_path or self.config.model_variant,
            text_encoder_identity=self._clip_identity,
            clip_patch_uuid=self._lora_signature(),
            clip_layer_idx=self.config.clip_layer,
            lora_artifacts_state=self._resolved_lora_specs,
            route_family_reconciliation_signature=(self.route_label, self.seams.compiled_unet_owner),
            residency_class="cpu",
            route_family=self.route_label,
            execution_family=self.config.execution_class,
            clip_residency_mode="cpu",
        )
        prompt_fingerprint = prompt_stage.digest()

        encode_start = time.perf_counter()
        encoded_prompt_pair = conditioning.load_prompt_conditioning_from_cache(prompt_stage)
        prompt_cache_hit = encoded_prompt_pair is not None

        clip_patch_count = 0
        clip_host_pinned_bytes = 0
        clip_compile_wall = 0.0
        clip_compile_metrics = self._default_compile_metrics()

        if prompt_cache_hit:
            logging.info(
                "[Nex-PromptCache] hit route=%s fingerprint=%s",
                self.route_label,
                prompt_fingerprint[:12],
            )
        else:
            # CLIP CPU-only patch, encode, and immediate restore lifecycle
            clip_patcher = getattr(self.clip, "patcher", self.clip)
            if clip_patcher is not None:
                if self._clip_component_needs_restore(clip_patcher):
                    self._restore_clean_clip_component(clip_patcher)
                if self._resolved_lora_specs:
                    clip_patch_count = self._apply_lora_specs_to_patcher(
                        clip_patcher,
                        clip_patcher.model,
                        target_family="clip",
                    )
                    if clip_patch_count > 0:
                        # Compile CLIP on CPU
                        clip_compile_start = time.perf_counter()
                        clip_compile = CpuArtifactCompiler.compile_patcher(clip_patcher)
                        clip_compile_wall = time.perf_counter() - clip_compile_start
                        clip_compile_metrics = clip_compile
                        clip_host_pinned_bytes = int(clip_compile.get("host_pinned_bytes", 0))

            encoded_prompt_pair = conditioning.encode_prompt_pair_sdxl(
                self.clip,
                self.config.prompt,
                self.config.negative_prompt,
                positive_texts=self.config.positive_texts or (self.config.prompt,),
                negative_texts=self.config.negative_texts or (self.config.negative_prompt,),
                positive_top_k=self.config.positive_top_k,
                negative_top_k=self.config.negative_top_k,
                use_explicit_residency=True,
            )
            conditioning.remember_prompt_conditioning_cache(prompt_stage, encoded_prompt_pair)
            logging.info(
                "[Nex-PromptCache] miss route=%s fingerprint=%s",
                self.route_label,
                prompt_fingerprint[:12],
            )

            # Immediate CLIP unpatch and restore to free backup memory
            if clip_patcher is not None and clip_patch_count > 0:
                self._restore_clean_clip_component(clip_patcher)

        conditioning_encode_wall = time.perf_counter() - encode_start

        adm_start = time.perf_counter()
        adm_pair = conditioning.build_sdxl_adm_pair(
            encoded_prompt_pair,
            self.config.width,
            self.config.height,
            target_width=self.config.width,
            target_height=self.config.height,
            adm_scale_positive=float((self.config.quality or {}).get("adm_scaler_positive", 1.5)),
            adm_scale_negative=float((self.config.quality or {}).get("adm_scaler_negative", 0.8)),
        )
        adm_build_wall = time.perf_counter() - adm_start

        conditioning_fingerprint = self._hash_payload(
            {
                "positive": encoded_prompt_pair["positive"],
                "negative": encoded_prompt_pair["negative"],
                "adm_pair": adm_pair,
            }
        )
        pooled_fingerprint = self._hash_payload(
            {
                "positive": encoded_prompt_pair["positive"]["pooled"],
                "negative": encoded_prompt_pair["negative"]["pooled"],
            }
        )

        self.conditioning = PromptConditioningArtifact(
            family="sdxl",
            variant=self.config.model_variant,
            prompt_fingerprint=prompt_fingerprint,
            clip_identity=self._clip_identity,
            clip_layer_idx=self.config.clip_layer,
            conditioning_fingerprint=conditioning_fingerprint,
            pooled_fingerprint=pooled_fingerprint,
            reusable=True,
        )

        injected_features, injected_payload, injected_metrics = self._prepare_injected_feature_artifacts()
        self.injected_features = injected_features

        unet_compile_metrics = lora_metrics["unet_compile_metrics"]
        compiled_unet_wall = float(lora_metrics["unet_compile_wall"])

        self.compiled_unet = CompiledUnetArtifact(
            family="sdxl",
            variant=self.config.model_variant,
            execution_class=self._execution_class_label(),
            source_path=self.config.checkpoint_path,
            source_fingerprint=self._checkpoint_fingerprint,
            artifact_fingerprint=self._build_compiled_unet_fingerprint(
                unet_compile_metrics=unet_compile_metrics,
            ),
            pinned_cpu_mb=self._measure_pinned_bytes(self.unet.model) / (1024 * 1024),
            gpu_mb=0.0,
            reusable=True,
        )

        structural_conditioning, structural_payload, structural_metrics = self._prepare_structural_conditioning_artifacts()
        self.structural_conditioning = structural_conditioning
        spatial_conditioning, spatial_payload, spatial_metrics = self._prepare_spatial_conditioning_artifacts()
        self.spatial_conditioning = spatial_conditioning

        # Aggressively tear down preprocessors/adapters used to generate BB+Mask artifacts
        resources.cleanup_memory(
            reason="preprocessors_teardown",
            unload_models=False,
            force_cache=True,
            target_phase=resources.MemoryPhase.DIFFUSION,
            task=self._memory_task_hint(),
        )

        self.execution_state = GpuAttachedExecutionState(
            execution_class=self._execution_class_label(),
            device="cuda",
            active_phase="prepare_inputs",
            attached_component_ids=(),
            stream_budget_mb=0.0,
            headroom_mb=0.0,
        )

        self.prepared_inputs = UnifiedSDXLPreparedInputs(
            base_model=self.base_model,
            compiled_unet=self.compiled_unet,
            conditioning=self.conditioning,
            structural_conditioning=self.structural_conditioning,
            spatial_conditioning=self.spatial_conditioning,
            injected_features=dict(self.injected_features),
            gpu_attached_execution_state=self.execution_state,
            payload={
                "encoded_prompt_pair": encoded_prompt_pair,
                "adm_pair": adm_pair,
                "prompt_fingerprint": prompt_fingerprint,
                "conditioning_fingerprint": conditioning_fingerprint,
                "pooled_fingerprint": pooled_fingerprint,
                "lora_specs": self._resolved_lora_specs,
                "base_model_fingerprint": self.base_model.fingerprint,
                "compiled_unet_fingerprint": self.compiled_unet.artifact_fingerprint,
                "initial_latent": self.config.initial_latent,
                "denoise_strength": self.config.denoise_strength,
                **structural_payload,
                **spatial_payload,
                **injected_payload,
            },
            metrics={
                "base_model_load_cpu": float(self._cold_model_load_cpu),
                "base_model_cache_hit": 1.0 if self._base_component_cache_hit else 0.0,
                "cold_load": float(self._cold_load_metric),
                "warm_reuse": float(self._warm_reuse_metric),
                "true_invalidation": float(self._true_invalidation_metric),
                "compiled_unet_cache_hit": float(lora_metrics.get("compiled_unet_cache_hit", 0.0)),
                "lora_parse_cold_time": float(self._lora_parse_cold_time),
                "lora_parse_warm_time": float(self._lora_parse_warm_time),
                "lora_parse_hits": float(self._lora_parse_hits),
                "lora_parse_misses": float(self._lora_parse_misses),
                "conditioning_encode_cpu": float(conditioning_encode_wall),
                "conditioning_adm_cpu": float(adm_build_wall),
                "conditioning_cache_hit": 1.0 if prompt_cache_hit else 0.0,
                "compiled_unet_cpu": float(compiled_unet_wall),
                "lora_spec_count": float(lora_metrics["spec_count"]),
                "clip_patch_count": float(clip_patch_count),
                "unet_patch_count": float(lora_metrics["unet_patch_count"]),
                "clip_host_pinned_bytes": float(clip_host_pinned_bytes),
                "unet_host_pinned_bytes": float(lora_metrics["unet_host_pinned_bytes"]),
                "clip_compile_cpu": float(clip_compile_wall),
                "conditioning_artifact_count": 1.0,
                "structural_artifact_count": 1.0 if structural_conditioning is not None else 0.0,
                "spatial_artifact_count": 1.0 if spatial_conditioning is not None else 0.0,
                "injected_feature_count": float(len(self.injected_features)),
                **structural_metrics,
                **spatial_metrics,
                **injected_metrics,
            },
        )
        self._prepare_metrics = dict(self.prepared_inputs.metrics)
        return self.prepared_inputs, dict(self._prepare_metrics)

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
        # Keep resident UNet state warm, but release the shared VAE so it does not stay on GPU.
        resources.eject_model(getattr(self.vae, "patcher", None))
        _soft_empty_cache_force()
        self._attached_payload = None
        self.execution_state = None
        self.prepared_inputs = None
        self.base_model = None
        self.compiled_unet = None
        self.conditioning = None
        self.structural_conditioning = None
        self.spatial_conditioning = None
        self.injected_features = {}
        # Keep references warm in the loader cache
        self.unet = None
        self.clip = None
        self.vae = None
        self._unload_controlnets()
        self._loaded = False
        self._base_component_cache_hit = False
        self._compiled_unet_cache_hit = False
        self._prepare_metrics = {}
        gc.collect()
        
        # Aggressively teardown any residual request-local support preprocessors/adapters
        resources.cleanup_memory(
            reason="resident_request_complete",
            unload_models=False,
            force_cache=True,
            target_phase=resources.MemoryPhase.FINALIZE,
            task=self._memory_task_hint(),
        )

    def patched_weights_for_block(self, block_id: str) -> Any:
        if self.compiled_unet is None or self.unet is None:
            raise RuntimeError("Resident SDXL runtime has no compiled UNet artifact available.")
        execution_unet = self.unet
        if self._attached_payload is not None:
            execution_unet = self._attached_payload.get("execution_unet") or execution_unet
        return {
            "block_id": str(block_id),
            "artifact_fingerprint": self.compiled_unet.artifact_fingerprint,
            "execution_unet": execution_unet,
            "execution_state": self.execution_state,
            "attached": bool(self._attached_payload is not None),
        }

    def injected_features_for_block(self, block_id: str, timestep: Any, context: Any) -> Any:
        _ = timestep
        _ = context
        payload = self._attached_payload or (self.prepared_inputs.payload if self.prepared_inputs is not None else {})
        contextual_tasks = (payload or {}).get("contextual_tasks") or {}
        if str(block_id) != "attn2":
            return None
        return contextual_tasks or None

    def encode_spatial_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """Route BB-plus-mask artifact preparation/latent encode through resident VAE authority."""
        self.load_components()
        if self.vae is None:
            raise RuntimeError("VAE is not loaded.")
        decode_device = self._execution_device()
        self._attach_vae(decode_device)
        try:
            from backend import encode as vae_encode
            if hasattr(self.vae, "first_stage_model"):
                self.vae.first_stage_model.to(device=decode_device)
            res = vae_encode.encode_pixels(self.vae, pixels)
            return res["samples"]
        finally:
            resources.eject_model(getattr(self.vae, "patcher", None))
            _soft_empty_cache_force()

    def decode_spatial_latents(self, latent: torch.Tensor, tiled: bool = False) -> torch.Tensor:
        """Expose a resident VAE decode seam that tiled refinement and Super-Upscale can adopt."""
        self.load_components()
        if self.vae is None:
            raise RuntimeError("VAE is not loaded.")
        decode_device = self._execution_device()
        self._attach_vae(decode_device)
        try:
            from backend import decode as vae_decode
            if hasattr(self.vae, "first_stage_model"):
                if vae_decode._should_force_fp32_vae_decode(self.vae):
                    self.vae.first_stage_model.to(device=decode_device, dtype=torch.float32)
                else:
                    self.vae.first_stage_model.to(device=decode_device)
            return vae_decode.decode_latent(self.vae, latent, tiled=tiled)
        finally:
            resources.eject_model(getattr(self.vae, "patcher", None))
            _soft_empty_cache_force()

    def _encode_spatial_pixels_for_artifacts(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.encode_spatial_pixels(pixels).detach().cpu()

    def _materialize_lora_stack(self) -> dict[str, Any]:
        self._compiled_unet_cache_hit = False
        desired_signature = self._lora_signature()
        
        current_unet_signature = self._current_resident_lora_signature(self.unet)

        has_compiled_state = (
            getattr(self.unet.model, "_nex_resident_compile_metrics", None) is not None
            or (
                not desired_signature
                and getattr(self.unet.model, "_nex_clean_unet_source", None) is None
            )
        )

        if desired_signature == current_unet_signature and has_compiled_state:
            self._compiled_unet_cache_hit = True
            unet_compile = self._current_resident_compile_metrics(self.unet)
            unet_patch_count = int(unet_compile.get("patch_count", 0))
            unet_host_pinned_bytes = int(unet_compile.get("host_pinned_bytes", 0))
            return {
                "spec_count": float(len(self._resolved_lora_specs)),
                "unet_patch_count": float(unet_patch_count),
                "unet_host_pinned_bytes": float(unet_host_pinned_bytes),
                "unet_compile_wall": 0.0,
                "compiled_unet_cache_hit": 1.0,
                "unet_compile_metrics": unet_compile,
            }

        model = getattr(self.unet, "model", None)
        needs_restore = (
            getattr(model, "_nex_clean_unet_source", None) is not None
            or current_unet_signature != ()
            or bool(getattr(model, "_patched_marker", False))
        )
        if needs_restore:
            self._restore_clean_unet_component()

        if not self._resolved_lora_specs:
            self._remember_resident_lora_state(self.unet, (), self._default_compile_metrics())
            unet_patch_count = 0.0
            unet_host_pinned_bytes = 0.0
            unet_compile_wall = 0.0
            unet_compile = self._default_compile_metrics()
        else:
            self._apply_lora_specs_to_patcher(
                self.unet,
                self.unet.model,
                target_family="unet",
            )
            unet_compile_start = time.perf_counter()
            unet_compile = GpuArtifactCompiler.compile_patcher(
                self.unet,
                clean_source=None,
                target_device=self._execution_device(),
                intermediate_dtype=torch.float16,
            )
            unet_compile_wall = time.perf_counter() - unet_compile_start
            unet_patch_count = int(unet_compile.get("patch_count", 0))
            unet_host_pinned_bytes = int(unet_compile.get("host_pinned_bytes", 0))
            
            self._remember_resident_lora_state(self.unet, desired_signature, unet_compile)

        return {
            "spec_count": float(len(self._resolved_lora_specs)),
            "unet_patch_count": float(unet_patch_count),
            "unet_host_pinned_bytes": float(unet_host_pinned_bytes),
            "unet_compile_wall": float(unet_compile_wall),
            "compiled_unet_cache_hit": 0.0,
            "unet_compile_metrics": unet_compile,
        }

    def _restore_clean_unet_component(self) -> bool:
        if self.unet is None:
            return False
        model = getattr(self.unet, "model", None)
        if model is None:
            return False

        if hasattr(model, "_patched_marker"):
            delattr(model, "_patched_marker")

        clean_source = getattr(model, "_nex_clean_unet_source", None)
        runtime_reload = getattr(self.unet, "runtime_reload", None)
        if clean_source is not None:
            model._nex_clean_unet_source = None
            logging.info("[ResidentSDXLRuntime] Discarded legacy clean UNet snapshot from memory.")
        if callable(runtime_reload):
            start_r = time.perf_counter()
            target_device = self.unet.current_loaded_device()
            if target_device is not None and not isinstance(target_device, torch.device):
                target_device = torch.device(target_device)
            runtime_reload(model, target_device)
            model.device = target_device
            end_r = time.perf_counter()
            logging.info(
                f"[ResidentSDXLRuntime] Reloaded clean UNet weights from source "
                f"in {((end_r - start_r) * 1000):.2f} ms"
            )
        elif clean_source is not None:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in clean_source:
                        param.copy_(clean_source[name])
                for name, buf in model.named_buffers():
                    if name in clean_source:
                        buf.copy_(clean_source[name])

        self._clear_patcher_artifacts(self.unet)
        self._remember_resident_lora_state(self.unet, (), self._default_compile_metrics())
        return True

    def _restore_clean_clip_component(self, clip_patcher: Any) -> None:
        if clip_patcher is None:
            return
        model = getattr(clip_patcher, "model", None)
        if model is None:
            return

        if hasattr(model, "_patched_marker"):
            delattr(model, "_patched_marker")

        clean_clip = getattr(model, "_nex_clean_clip_source", None)
        runtime_reload = getattr(clip_patcher, "runtime_reload", None)
        if clean_clip is not None and callable(runtime_reload):
            model._nex_clean_clip_source = None
            logging.info("[ResidentSDXLRuntime] Discarded legacy clean CLIP snapshot from memory.")
        elif clean_clip is not None:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in clean_clip:
                        param.copy_(clean_clip[name])
                for name, buf in model.named_buffers():
                    if name in clean_clip:
                        buf.copy_(clean_clip[name])
            model._nex_clean_clip_source = None
            logging.info("[ResidentSDXLRuntime] Discarded clean CLIP snapshot from memory.")
        if callable(runtime_reload):
            target_device = getattr(model, "device", None) or torch.device("cpu")
            if target_device is not None and not isinstance(target_device, torch.device):
                target_device = torch.device(target_device)
            runtime_reload(model, target_device)
            model.device = target_device

        self._clear_patcher_artifacts(clip_patcher)
        self._remember_resident_lora_state(clip_patcher, (), self._default_compile_metrics())

    def _require_checkpoint_path(self) -> str:
        checkpoint_path = str(self.config.checkpoint_path or "").strip()
        if not checkpoint_path:
            raise ValueError("Resident SDXL runtime requires config.checkpoint_path.")
        return checkpoint_path

    def _require_optional_vae_path(self) -> str | None:
        import modules.flags as flags

        value = str(self.config.vae_path or "").strip()
        if _is_default_shared_sdxl_vae_selection(value):
            return _resolve_shared_sdxl_vae_path() or flags.default_vae
        if _looks_like_shared_sdxl_vae_asset(value) and not os.path.isfile(value):
            resolved = _resolve_shared_sdxl_vae_path()
            if resolved is not None:
                return resolved
        return value or None

    def _discard_legacy_clean_snapshots(self) -> None:
        for patcher, attr_name, label in (
            (self.unet, "_nex_clean_unet_source", "UNet"),
            (getattr(self.clip, "patcher", self.clip), "_nex_clean_clip_source", "CLIP"),
        ):
            model = getattr(patcher, "model", None) if patcher is not None else None
            if model is None or getattr(model, attr_name, None) is None:
                continue
            setattr(model, attr_name, None)
            logging.info("[ResidentSDXLRuntime] Cleared cached %s clean snapshot during runtime attach.", label)

    def _clip_component_needs_restore(self, clip_patcher: Any) -> bool:
        if clip_patcher is None:
            return False
        model = getattr(clip_patcher, "model", None)
        if model is None:
            return False
        return bool(
            getattr(model, "_patched_marker", False)
            or getattr(model, "_nex_clean_clip_source", None) is not None
            or getattr(clip_patcher, "patches", None)
        )

    def _memory_task_hint(self):
        has_controlnet = bool(self.config.structural_tasks or self.config.contextual_tasks or self.config.controlnet_paths)
        has_source_pixels = self.config.source_pixels is not None or self.config.source_mask is not None
        current_tab = "txt2img"
        state = {}

        if self.config.outpaint_direction:
            current_tab = "outpaint"
            state["outpaint_input_image"] = object()
            state["mixing_image_prompt_and_outpaint"] = has_controlnet
        elif has_source_pixels:
            current_tab = "inpaint"
            state["inpaint_input_image"] = object()
            state["mixing_image_prompt_and_inpaint"] = has_controlnet
        elif has_controlnet:
            current_tab = "ip"

        state["current_tab"] = current_tab
        state["input_image_checkbox"] = has_source_pixels or has_controlnet
        state["cn_tasks"] = {
            **{key: list(tasks) for key, tasks in (self.config.structural_tasks or {}).items() if tasks},
            **{key: list(tasks) for key, tasks in (self.config.contextual_tasks or {}).items() if tasks},
        }
        return SimpleNamespace(**state)

    def _execution_class_label(self) -> str:
        return str(self.config.execution_class or self.route_label)

    def _execution_device(self) -> torch.device:
        try:
            return resources.get_torch_device()
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _clean_unet_budget_bytes(self, device: torch.device) -> int:
        if device.type != "cuda":
            return 0
        if self.config.streamlike_budget_mb <= 0:
            return 0
        return max(64, int(self.config.streamlike_budget_mb)) * 1024 * 1024

    def _device_headroom_mb(self, device: torch.device) -> float:
        try:
            return float(resources.get_free_memory(device)) / (1024 * 1024)
        except Exception:
            return 0.0

    def _attach_compiled_unet(self, device: torch.device, *, budget_bytes: int = 0) -> None:
        if self.unet is None:
            raise RuntimeError("Compiled UNet is not loaded.")
        model_size = int(self.unet.model_size())
        lowvram_model_memory = 0 if budget_bytes <= 0 or budget_bytes >= model_size else int(budget_bytes)
        self.unet.patch_model(device_to=device, lowvram_model_memory=lowvram_model_memory)

    def _attach_vae(self, device: torch.device) -> None:
        if self.vae is None:
            raise RuntimeError("VAE is not loaded.")
        self.vae.patcher.patch_model(device_to=device, lowvram_model_memory=0)

    def _detach_component(self, component: Any) -> None:
        if component is None:
            return
        detach = getattr(component, "detach", None)
        if callable(detach):
            try:
                detach()
            except Exception:
                pass

    def _park_compiled_unet_before_decode(self) -> None:
        return

    def _should_pin_unet_host_for_compile(self) -> bool:
        if not self._resolved_lora_specs:
            return bool(self.config.pin_base_unet_without_lora)
        return True

    def _base_component_cache_key(
        self,
        *,
        checkpoint_path: str,
        is_resident: bool,
    ) -> tuple[str, str]:
        residency_label = "resident" if is_resident else "cpu"
        return (checkpoint_path, residency_label)

    def _clone_component_for_runtime(self, component: Any) -> Any:
        if component is None:
            return None
        clone = getattr(component, "clone", None)
        if callable(clone):
            return clone()
        return component

    def _clone_shared_base_components(
        self,
        shared_base: SharedSDXLBaseComponents,
    ) -> tuple[Any, Any]:
        return (
            self._clone_component_for_runtime(shared_base.unet),
            self._clone_component_for_runtime(shared_base.clip),
        )

    def _remember_shared_base_components(
        self,
        cache_key: tuple[str, str],
        shared_base: SharedSDXLBaseComponents,
    ) -> None:
        _SHARED_SDXL_BASE_COMPONENT_CACHE[cache_key] = shared_base
        _SHARED_SDXL_BASE_COMPONENT_CACHE.move_to_end(cache_key)
        while len(_SHARED_SDXL_BASE_COMPONENT_CACHE) > 1:
            _, evicted = _SHARED_SDXL_BASE_COMPONENT_CACHE.popitem(last=False)
            _release_shared_sdxl_base_components(evicted)

    def _normalize_lora_specs(self, lora_specs: Any) -> tuple[tuple[str, float], ...]:
        normalized: list[tuple[str, float]] = []
        for spec in lora_specs or ():
            if not spec:
                continue
            path, strength = spec
            normalized.append((str(path), float(strength)))
        return tuple(normalized)

    def _fingerprint_source_path(self, source_path: str | None) -> str | None:
        if not source_path:
            return None
        if os.path.isfile(source_path):
            try:
                return compute_file_hash(source_path)
            except Exception:
                pass
        return hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()

    def _lora_signature(self) -> tuple[str, ...]:
        return tuple(f"{path}:{strength:g}" for path, strength in self._resolved_lora_specs)

    def _hash_payload(self, payload: Any) -> str:
        digest = hashlib.sha256()
        digest.update(repr(self._freeze_value(payload)).encode("utf-8"))
        return digest.hexdigest()

    def _freeze_value(self, value: Any) -> Any:
        if hasattr(value, "__dataclass_fields__"):
            from dataclasses import asdict
            return self._freeze_value(asdict(value))
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            return (
                "tensor",
                tuple(int(dim) for dim in tensor.shape),
                str(tensor.dtype),
                hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
            )
        if isinstance(value, dict):
            return tuple((str(key), self._freeze_value(item)) for key, item in sorted(value.items(), key=lambda item: str(item[0])))
        if isinstance(value, (list, tuple)):
            return tuple(self._freeze_value(item) for item in value)
        if isinstance(value, set):
            return tuple(sorted(self._freeze_value(item) for item in value))
        return value

    def _measure_pinned_bytes(self, module: Any) -> int:
        if module is None:
            return 0
        total = 0
        for tensor in list(module.parameters()) + list(module.buffers()):
            if isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu" and tensor.is_pinned():
                total += tensor.numel() * tensor.element_size()
        return total

    @staticmethod
    def _default_compile_metrics() -> dict[str, Any]:
        return {
            "status": "noop",
            "patch_count": 0,
            "materialized_patch_keys": 0,
            "host_pinned_bytes": 0,
        }

    def _current_resident_lora_signature(self, patcher: Any) -> tuple[str, ...]:
        model = getattr(patcher, "model", None)
        signature = getattr(model, "_nex_resident_lora_signature", ()) if model is not None else ()
        if signature is None:
            return ()
        return tuple(str(item) for item in signature)

    def _current_resident_compile_metrics(self, patcher: Any) -> dict[str, Any]:
        model = getattr(patcher, "model", None)
        metrics = getattr(model, "_nex_resident_compile_metrics", None) if model is not None else None
        if isinstance(metrics, dict):
            return dict(metrics)
        return self._default_compile_metrics()

    def _remember_resident_lora_state(
        self,
        patcher: Any,
        signature: tuple[str, ...],
        compile_metrics: dict[str, Any] | None = None,
    ) -> None:
        model = getattr(patcher, "model", None)
        if model is None:
            return
        setattr(model, "_nex_resident_lora_signature", tuple(signature))
        setattr(
            model,
            "_nex_resident_compile_metrics",
            dict(compile_metrics or self._default_compile_metrics()),
        )

    def _clear_patcher_artifacts(self, patcher: Any) -> None:
        if patcher is None:
            return
        for attr in ("patches", "weight_wrapper_patches", "backup", "object_patches_backup"):
            value = getattr(patcher, attr, None)
            if hasattr(value, "clear"):
                try:
                    value.clear()
                except Exception:
                    pass
        model = getattr(patcher, "model", None)
        if model is None:
            return
        if hasattr(model, "current_weight_patches_uuid"):
            model.current_weight_patches_uuid = None
        if hasattr(model, "lowvram_patch_counter"):
            model.lowvram_patch_counter = 0
        if hasattr(model, "model_lowvram"):
            model.model_lowvram = False

    def _build_clip_identity(self) -> str:
        clip_identity = self.config.checkpoint_path
        if not clip_identity and self.base_model is not None and self.base_model.source_path:
            clip_identity = self.base_model.source_path
        return f"{self.config.model_variant}:{clip_identity}:{self.config.clip_layer}"

    def _apply_lora_specs_to_patcher(self, patcher: Any, model: Any, *, target_family: str) -> int:
        if patcher is None or model is None:
            return 0

        key_map = (
            backend_lora.model_lora_keys_clip(model)
            if target_family == "clip"
            else backend_lora.model_lora_keys_unet(model)
        )
        patch_count = 0
        model_class_name = model.__class__.__name__

        for lora_path, strength in self._resolved_lora_specs:
            materialization_tag = str(self._execution_device()) if target_family == "unet" else "cpu"
            cache_key = (lora_path, target_family, model_class_name, materialization_tag)
            current_hash = _get_lora_file_hash(lora_path)
            
            cached_entry = _PARSED_LORA_ADAPTER_CACHE.get(cache_key)
            if cached_entry is not None:
                cached_hash, cached_patch_dict = cached_entry
                if cached_hash == current_hash:
                    start_time = time.perf_counter()
                    _PARSED_LORA_ADAPTER_CACHE.move_to_end(cache_key)
                    self._lora_parse_hits += 1.0
                    patch_dict = cached_patch_dict
                    self._lora_parse_warm_time += time.perf_counter() - start_time
                else:
                    _PARSED_LORA_ADAPTER_CACHE.pop(cache_key)
                    cached_entry = None

            if cached_entry is None:
                start_time = time.perf_counter()
                header = self._open_lora_header(lora_path, target_family=target_family)
                patch_dict = backend_lora.load_lora(header, key_map, log_missing=False)
                self._lora_parse_misses += 1.0
                self._lora_parse_cold_time += time.perf_counter() - start_time
                
                if patch_dict:
                    _PARSED_LORA_ADAPTER_CACHE[cache_key] = (current_hash, patch_dict)
                    _PARSED_LORA_ADAPTER_CACHE.move_to_end(cache_key)
                    while len(_PARSED_LORA_ADAPTER_CACHE) > _PARSED_LORA_ADAPTER_CACHE_LIMIT:
                        _PARSED_LORA_ADAPTER_CACHE.popitem(last=False)

            if not patch_dict:
                continue
            patcher.add_patches(patch_dict, strength)
            patch_count += len(patch_dict)
        return patch_count

    def _open_lora_header(self, lora_path: str, *, target_family: str):
        if target_family == "unet" and os.path.isfile(lora_path):
            try:
                return _ResidentSafeOpenHeaderOnly(lora_path, tensor_device=self._execution_device())
            except Exception:
                logging.debug("Falling back to shared LoRA header loader for %s.", lora_path, exc_info=True)
        return SafeOpenHeaderOnly(lora_path)

    def _build_compiled_unet_fingerprint(
        self,
        *,
        unet_compile_metrics: dict[str, Any],
    ) -> str:
        digest = hashlib.sha256()
        base_fp = (self.base_model.fingerprint if self.base_model else None) or self.config.checkpoint_path or ""
        digest.update(base_fp.encode("utf-8"))
        digest.update(self._execution_class_label().encode("utf-8"))
        digest.update(repr(self._lora_signature()).encode("utf-8"))
        digest.update(repr(self.config.clip_layer).encode("utf-8"))
        digest.update(repr(self.config.batch_size).encode("utf-8"))
        digest.update(repr(self.config.steps).encode("utf-8"))
        digest.update(repr(self.config.scheduler).encode("utf-8"))
        digest.update(repr(self.config.original_scheduler_name).encode("utf-8"))
        digest.update(repr(unet_compile_metrics.get("patch_count", 0)).encode("utf-8"))
        digest.update(repr(unet_compile_metrics.get("host_pinned_bytes", 0)).encode("utf-8"))
        return digest.hexdigest()
