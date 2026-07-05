from __future__ import annotations

import logging
import time
import hashlib
import json
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import safetensors.torch as sf

from backend.sdxl_assembly.contracts import (
    SDXLAssemblyRequest,
    SDXLContextualControlDescriptor,
    ContextualPayloadArtifact
)
from backend.sdxl_assembly.progress import log_telemetry
import backend.resources as runtime_resources
from backend.ops import use_patched_ops

# Donor module wrappers
from backend.ip_adapter import (
    PulidAdapterModel,
    IPAdapterModel,
    _normalize_pulid_state_dict,
    detect_model_kind,
    _sorted_kv_modules,
    manual_cast,
    load_insightface
)
from modules.core import numpy_to_pytorch
import ldm_patched.modules.clip_vision
from ldm_patched.modules.model_patcher import ModelPatcher
import ldm_patched.ldm.modules.attention as attention
from insightface.utils import face_align

logger = logging.getLogger(__name__)


def _offload_patcher(patcher: Any) -> None:
    if patcher is None:
        return
    try:
        patcher.detach()
    except Exception:
        pass


def _offload_clip_vision_model(clip_model: Any) -> None:
    if clip_model is None:
        return
    patcher = getattr(clip_model, 'patcher', None)
    if patcher is not None:
        _offload_patcher(patcher)
    else:
        try:
            clip_model.model.to('cpu')
        except Exception:
            pass


def _offload_contextual_entry(entry: Dict[str, Any]) -> None:
    if not isinstance(entry, dict):
        return
    model = entry.get('model')
    if model is not None:
        try:
            model.to(getattr(model, 'offload_device', torch.device('cpu')), dtype=getattr(model, 'dtype', None))
        except Exception:
            try:
                model.to(getattr(model, 'offload_device', torch.device('cpu')))
            except Exception:
                pass
    _offload_patcher(entry.get('image_proj_model'))
    _offload_patcher(entry.get('ip_layers'))


def _offload_module(module: Any) -> None:
    if module is None:
        return
    try:
        module.to('cpu')
    except Exception:
        pass


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    tensor = torch.clamp(torch.from_numpy(image).float() / 255.0, 0, 1)
    return tensor[..., [2, 1, 0]]


def to_gray(img: torch.Tensor) -> torch.Tensor:
    x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    return x.repeat(1, 3, 1, 1)


class StreamingContextualControlWorker:
    """Worker managing contextual ControlNet preprocessing, caching, and runtime attention patching."""

    # Class-level support model caches (private to assembly lane)
    _CONTEXTUAL_MODELS: Dict[str, Dict[str, Any]] = {}
    _CLIP_VISION_MODELS: Dict[str, Any] = {}
    _IP_NEGATIVES: Dict[str, Any] = {}
    _EVA_CLIP_MODELS: Dict[str, Any] = {}
    _FACE_PARSERS: Dict[str, Any] = {}
    _INSIGHTFACE_APPS: Dict[str, Any] = {}

    # Class-level payload cache (limit 8, OrderedDict)
    _PAYLOAD_CACHE: OrderedDict[str, ContextualPayloadArtifact] = OrderedDict()
    _PAYLOAD_CACHE_LIMIT = 8

    def __init__(self, request: SDXLAssemblyRequest) -> None:
        self.request = request
        self.unet_spine: Optional[Any] = None

    @classmethod
    def clear_payload_cache(cls) -> None:
        cls._PAYLOAD_CACHE.clear()
        log_telemetry("contextual_payload_cache_cleared")

    @classmethod
    def clear_support_cache(cls) -> None:
        # Offload/release all entries
        for entry in cls._CONTEXTUAL_MODELS.values():
            _offload_contextual_entry(entry)
        cls._CONTEXTUAL_MODELS.clear()

        for clip_model in cls._CLIP_VISION_MODELS.values():
            _offload_clip_vision_model(clip_model)
        cls._CLIP_VISION_MODELS.clear()

        cls._IP_NEGATIVES.clear()

        for module in cls._EVA_CLIP_MODELS.values():
            _offload_module(module)
        cls._EVA_CLIP_MODELS.clear()

        for parser in cls._FACE_PARSERS.values():
            _offload_module(parser)
        cls._FACE_PARSERS.clear()

        cls._INSIGHTFACE_APPS.clear()
        
        log_telemetry("contextual_support_cache_cleared")

    def release_owned_resources(self) -> None:
        self.clear_support_cache()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_contextual_cache_key(self, desc: SDXLContextualControlDescriptor) -> str:
        payload = {
            "image_fingerprint": desc.image_fingerprint,
            "source_image_role": desc.source_image_role,
            "control_type": desc.control_type,
            "model_sha256": desc.model_sha256,
            "clip_vision_sha256": desc.clip_vision_sha256,
            "ip_negative_sha256": desc.ip_negative_sha256,
            "eva_clip_sha256": desc.eva_clip_sha256,
            "insightface_model_names": sorted(list(desc.insightface_model_names)),
            "preprocess_params": desc.preprocess_params,
        }
        payload_str = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

    def load_contextual_model_local(self, model_path: Path) -> Dict[str, Any]:
        path_str = str(model_path)
        if path_str in self._CONTEXTUAL_MODELS:
            return self._CONTEXTUAL_MODELS[path_str]

        load_device = runtime_resources.get_torch_device()
        offload_device = torch.device("cpu")
        use_fp16 = runtime_resources.should_use_fp16(device=load_device)
        dtype = torch.float16 if use_fp16 else torch.float32

        if path_str.lower().endswith(".safetensors"):
            raw_state_dict = sf.load_file(path_str)
        else:
            raw_state_dict = torch.load(path_str, map_location="cpu", weights_only=True)

        state_dict = _normalize_pulid_state_dict(raw_state_dict)
        model_kind = detect_model_kind(state_dict)

        if model_kind == "faceid_v2":
            raise RuntimeError("FaceID V2 is explicitly retired on the new assembly path.")

        with use_patched_ops(manual_cast):
            if model_kind == "pulid":
                model = PulidAdapterModel(state_dict)
            else:
                plus = "latents" in state_dict["image_proj"]
                cross_attention_dim = state_dict["ip_adapter"]["1.to_k_ip.weight"].shape[1]
                sdxl = cross_attention_dim == 2048
                if plus:
                    clip_extra_context_tokens = state_dict["image_proj"]["latents"].shape[1]
                    clip_embeddings_dim = state_dict["image_proj"]["latents"].shape[2]
                else:
                    clip_extra_context_tokens = state_dict["image_proj"]["proj.weight"].shape[0] // cross_attention_dim
                    clip_embeddings_dim = state_dict["image_proj"]["proj.weight"].shape[1]

                model = IPAdapterModel(
                    state_dict,
                    plus=plus,
                    cross_attention_dim=cross_attention_dim,
                    clip_embeddings_dim=clip_embeddings_dim,
                    clip_extra_context_tokens=clip_extra_context_tokens,
                    sdxl_plus=sdxl and plus,
                )

        model.load_device = load_device
        model.offload_device = offload_device
        model.dtype = dtype
        model.to(offload_device, dtype=dtype)

        entry = {
            "kind": model_kind,
            "model": model,
            "image_proj_model": ModelPatcher(model=model.image_proj_model, load_device=load_device, offload_device=offload_device),
            "ip_layers": ModelPatcher(model=model.ip_layers, load_device=load_device, offload_device=offload_device),
            "ip_unconds": None,
        }
        self._CONTEXTUAL_MODELS[path_str] = entry
        return entry

    def load_clip_vision_local(self, path: Path) -> Any:
        path_str = str(path)
        if path_str in self._CLIP_VISION_MODELS:
            return self._CLIP_VISION_MODELS[path_str]

        clip_model = ldm_patched.modules.clip_vision.load(path_str)
        self._CLIP_VISION_MODELS[path_str] = clip_model
        return clip_model

    def ensure_ip_negative_local(self, path: Path) -> Any:
        path_str = str(path)
        if path_str in self._IP_NEGATIVES:
            return self._IP_NEGATIVES[path_str]

        data = sf.load_file(path_str)["data"]
        self._IP_NEGATIVES[path_str] = data
        return data

    def load_eva_clip_local(self, path: Path) -> Any:
        path_str = str(path)
        if path_str in self._EVA_CLIP_MODELS:
            return self._EVA_CLIP_MODELS[path_str]

        from backend.eva_clip.factory import create_model_and_transforms
        model, _, _ = create_model_and_transforms(
            'EVA02-CLIP-L-14-336',
            pretrained=path_str,
            force_custom_clip=True,
            device='cpu',
        )
        visual = model.visual
        self._EVA_CLIP_MODELS[path_str] = visual
        return visual

    def load_face_parser_local(self, device: torch.device) -> Any:
        cache_key = str(device)
        if cache_key in self._FACE_PARSERS:
            return self._FACE_PARSERS[cache_key]

        import modules.config as config
        from extras.facexlib.parsing import init_parsing_model

        parser = init_parsing_model(
            model_name='bisenet',
            device=str(device),
            model_rootpath=config.path_insightface,
        )
        self._FACE_PARSERS[cache_key] = parser
        return parser

    def load_insightface_local(self, model_name: str) -> Any:
        if model_name in self._INSIGHTFACE_APPS:
            return self._INSIGHTFACE_APPS[model_name]

        app = load_insightface(model_name=model_name)
        self._INSIGHTFACE_APPS[model_name] = app
        return app

    def _detect_faces_local(self, face_app: Any, bgr_image: np.ndarray) -> List[Any]:
        for size in range(640, 256, -64):
            face_app.det_model.input_size = (size, size)
            faces = face_app.get(bgr_image)
            if faces:
                return faces
        return []

    def preprocess(self) -> Dict[int, ContextualPayloadArtifact]:
        """Preprocesses the contextual control tasks with lazy caching and CPU parking."""
        prepared_payloads = {}
        if not self.request.contextual_controls:
            return prepared_payloads

        for desc in self.request.contextual_controls:
            cache_key = self._get_contextual_cache_key(desc)
            cached_artifact = self._PAYLOAD_CACHE.get(cache_key)

            if cached_artifact is not None:
                # Move to end for LRU behavior
                self._PAYLOAD_CACHE.move_to_end(cache_key)
                # Re-check ui_slot_index matches request
                updated_artifact = ContextualPayloadArtifact(
                    ui_slot_index=desc.ui_slot_index,
                    control_type=cached_artifact.control_type,
                    payload=cached_artifact.payload,
                    payload_fingerprint=cached_artifact.payload_fingerprint,
                    cache_hit=True,
                    preprocess_wall=0.0
                )
                prepared_payloads[desc.ui_slot_index] = updated_artifact
                log_telemetry("contextual_payload_hit", f"slot={desc.ui_slot_index} control_type={desc.control_type}")
                continue

            # Cache miss: Execute preprocessing branch
            log_telemetry("contextual_payload_miss", f"slot={desc.ui_slot_index} control_type={desc.control_type}")
            start_wall = time.perf_counter()

            # Load primary contextual adapter model details
            entry = self.load_contextual_model_local(desc.model_path)
            adapter_model = entry["model"]
            load_device = adapter_model.load_device
            dtype = adapter_model.dtype

            img_np = (desc.image_pixels.numpy() * 255.0).astype(np.uint8)
            # Remove batch dimension if present
            if len(img_np.shape) == 4 and img_np.shape[0] == 1:
                img_np = img_np[0]

            if desc.control_type == "ImagePrompt":
                if not desc.clip_vision_path or not desc.ip_negative_path:
                    raise RuntimeError("ImagePrompt requires CLIP vision path and IP negative path.")

                clip_model = self.load_clip_vision_local(desc.clip_vision_path)
                try:
                    clip_patcher = getattr(clip_model, 'patcher', None)
                    if clip_patcher is not None:
                        runtime_resources.load_model_gpu(clip_patcher)
                    else:
                        try:
                            clip_model.model.to(load_device)
                        except Exception:
                            pass

                    # Encode image prompt
                    outputs = clip_model.encode_image(numpy_to_pytorch(img_np))
                    cond = outputs.penultimate_hidden_states if adapter_model.plus else outputs.image_embeds
                    cond = cond.to(device=load_device, dtype=dtype)

                    runtime_resources.load_model_gpu(entry["image_proj_model"])
                    cond = entry["image_proj_model"].model(cond).to(device=load_device, dtype=dtype)

                    runtime_resources.load_model_gpu(entry["ip_layers"])
                    kv_modules = _sorted_kv_modules(entry["ip_layers"].model)

                    negative = self.ensure_ip_negative_local(desc.ip_negative_path)
                    negative = negative.to(device=load_device, dtype=dtype)

                    ip_unconds = [module(negative).cpu() for module in kv_modules]
                    ip_conds = [module(cond).cpu() for module in kv_modules]

                    res_payload = (ip_conds, ip_unconds)

                finally:
                    _offload_contextual_entry(entry)
                    _offload_clip_vision_model(clip_model)

            elif desc.control_type == "PuLID":
                if not desc.eva_clip_path:
                    raise RuntimeError("PuLID requires EVA-CLIP path.")

                try:
                    eva_clip = self.load_eva_clip_local(desc.eva_clip_path)
                    face_parser = self.load_face_parser_local(device=load_device)

                    bgr_image = np.ascontiguousarray(img_np[:, :, ::-1])
                    faces = []
                    for model_name in desc.insightface_model_names:
                        face_app = self.load_insightface_local(model_name)
                        faces = self._detect_faces_local(face_app, bgr_image)
                        if faces:
                            break

                    if not faces:
                        raise RuntimeError("PuLID preprocessing could not detect a face in the reference image.")

                    eva_clip = eva_clip.to(load_device, dtype=dtype)
                    runtime_resources.load_model_gpu(entry["image_proj_model"])

                    bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
                    cond_embeddings = []
                    uncond_embeddings = []

                    image_size = eva_clip.image_size if isinstance(eva_clip.image_size, int) else eva_clip.image_size[0]
                    interpolation = T.InterpolationMode.BICUBIC if 'cuda' in load_device.type else T.InterpolationMode.BILINEAR

                    for face in faces:
                        iface_embeds = torch.from_numpy(face.embedding).unsqueeze(0).to(load_device, dtype=dtype)
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore",
                                message="`estimate` is deprecated since version 0.26",
                                category=FutureWarning,
                            )
                            aligned = face_align.norm_crop(bgr_image, landmark=face.kps, image_size=512)
                        
                        face_tensor = image_to_tensor(aligned).unsqueeze(0).permute(0, 3, 1, 2).to(load_device)
                        parsing_input = TF.normalize(face_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        parsing_out = face_parser(parsing_input)[0].argmax(dim=1, keepdim=True)
                        bg = sum((parsing_out == label) for label in bg_label).bool()
                        white_image = torch.ones_like(face_tensor)
                        face_features_image = torch.where(bg, white_image, to_gray(face_tensor))
                        face_features_image = TF.resize(face_features_image, [image_size, image_size], interpolation=interpolation)
                        face_features_image = TF.normalize(face_features_image, eva_clip.image_mean, eva_clip.image_std).to(load_device, dtype=dtype)

                        id_cond_vit, id_vit_hidden = eva_clip(
                            face_features_image,
                            return_all_features=False,
                            return_hidden=True,
                            shuffle=False,
                        )
                        id_cond_vit = id_cond_vit.to(load_device, dtype=dtype)
                        id_vit_hidden = [hidden.to(load_device, dtype=dtype) for hidden in id_vit_hidden]
                        id_cond_vit = torch.div(id_cond_vit, torch.norm(id_cond_vit, 2, 1, True))

                        id_cond = torch.cat([iface_embeds, id_cond_vit], dim=-1)
                        id_uncond = torch.zeros_like(id_cond)
                        id_hidden_uncond = [torch.zeros_like(hidden) for hidden in id_vit_hidden]

                        cond_embeddings.append(entry['image_proj_model'].model(id_cond, id_vit_hidden))
                        uncond_embeddings.append(entry['image_proj_model'].model(id_uncond, id_hidden_uncond))

                    cond = torch.mean(torch.cat(cond_embeddings, dim=0), dim=0, keepdim=True).to(load_device, dtype=dtype)
                    uncond = torch.mean(torch.cat(uncond_embeddings, dim=0), dim=0, keepdim=True).to(load_device, dtype=dtype)

                    zero_tensor = torch.zeros((cond.size(0), 8, cond.size(-1)), dtype=dtype, device=load_device)
                    cond = torch.cat([cond, zero_tensor], dim=1)
                    uncond = torch.cat([uncond, zero_tensor], dim=1)

                    runtime_resources.load_model_gpu(entry['ip_layers'])
                    kv_modules = _sorted_kv_modules(entry['ip_layers'].model)
                    ip_conds = [module(cond).cpu() for module in kv_modules]
                    ip_unconds = [module(uncond).cpu() for module in kv_modules]

                    res_payload = (ip_conds, ip_unconds)

                finally:
                    for module in self._EVA_CLIP_MODELS.values():
                        _offload_module(module)
                    for parser in self._FACE_PARSERS.values():
                        _offload_module(parser)
                    _offload_contextual_entry(entry)

            else:
                raise KeyError(f"Unsupported contextual type: {desc.control_type}")

            # Park on CPU and add to cache
            payload_artifact = ContextualPayloadArtifact(
                ui_slot_index=desc.ui_slot_index,
                control_type=desc.control_type,
                payload=res_payload,
                payload_fingerprint=desc.image_fingerprint,
                cache_hit=False,
                preprocess_wall=time.perf_counter() - start_wall
            )

            self._PAYLOAD_CACHE[cache_key] = payload_artifact
            self._PAYLOAD_CACHE.move_to_end(cache_key)

            while len(self._PAYLOAD_CACHE) > self._PAYLOAD_CACHE_LIMIT:
                self._PAYLOAD_CACHE.popitem(last=False)

            prepared_payloads[desc.ui_slot_index] = payload_artifact
            log_telemetry("contextual_payload_parked_cpu", f"slot={desc.ui_slot_index} control_type={desc.control_type} time={payload_artifact.preprocess_wall:.3f}s")

        return prepared_payloads

    def attach_unet_patches(self, unet_spine: Any) -> None:
        """Hooks attention layers of the UNet spine with the unified attention patcher."""
        if not self.request.contextual_controls:
            return

        self.unet_spine = unet_spine
        unet = unet_spine.unet

        log_telemetry("contextual_control_attach_begin")
        start_time = time.perf_counter()

        # Build task lookup for the attention patcher
        active_tasks = []
        for desc in self.request.contextual_controls:
            cache_key = self._get_contextual_cache_key(desc)
            artifact = self._PAYLOAD_CACHE.get(cache_key)
            if artifact is None:
                raise RuntimeError(f"Preprocessed payload not found in cache for slot {desc.ui_slot_index}")
            active_tasks.append((artifact.payload, desc.weight, desc.start_percent, desc.end_percent, desc.control_type))

        transformer_options = unet.model_options["transformer_options"]
        if "patches_replace" not in transformer_options:
            transformer_options["patches_replace"] = {}
        if "attn2" not in transformer_options["patches_replace"]:
            transformer_options["patches_replace"]["attn2"] = {}

        target_keys = [
            ("input", 4, 0), ("input", 5, 0), ("input", 7, 0), ("input", 8, 0),
            ("middle", 0, 0),
            ("output", 0, 0), ("output", 1, 0), ("output", 2, 0), ("output", 3, 0), ("output", 4, 0), ("output", 5, 0)
        ]

        def sdp(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, extra_options: Dict[str, Any]) -> torch.Tensor:
            return attention.optimized_attention(q, k, v, heads=extra_options['n_heads'], mask=None)

        def make_unified_attn_patcher(ip_index: int) -> Any:
            def unified_attn_patcher(n: torch.Tensor, context_attn2: torch.Tensor, value_attn2: torch.Tensor, extra_options: Dict[str, Any]) -> torch.Tensor:
                percentage = 0.5
                model_sampling = getattr(extra_options.get('model', None), 'model_sampling', None)
                if model_sampling is not None:
                    try:
                        timestep = model_sampling.timestep(extra_options.get("sigmas", torch.tensor([0.0]))).cpu().numpy()[0]
                        # timesteps run from 999 down to 0
                        percentage = 1.0 - (timestep / 999.0)
                    except Exception:
                        pass

                q = n
                out = sdp(q, context_attn2, value_attn2, extra_options)

                # Process ImagePrompt tasks first (via concatenation)
                ip_k_list = [context_attn2]
                ip_v_list = [value_attn2]
                has_ip = False

                for payload, weight, start_pct, end_pct, control_type in active_tasks:
                    if percentage < start_pct or percentage >= end_pct:
                        continue
                    if control_type == "ImagePrompt":
                        conds, unconds = payload
                        ip_k_c = conds[ip_index * 2].to(device=q.device, dtype=q.dtype)
                        ip_v_c = conds[ip_index * 2 + 1].to(device=q.device, dtype=q.dtype)
                        ip_k_uc = unconds[ip_index * 2].to(device=q.device, dtype=q.dtype)
                        ip_v_uc = unconds[ip_index * 2 + 1].to(device=q.device, dtype=q.dtype)

                        if q.shape[0] == 2:
                            k_concat = torch.cat([ip_k_uc, ip_k_c], dim=0)
                            v_concat = torch.cat([ip_v_uc, ip_v_c], dim=0)
                        else:
                            k_concat = ip_k_c
                            v_concat = ip_v_c

                        ip_k_list.append(k_concat * weight)
                        ip_v_list.append(v_concat)
                        has_ip = True

                if has_ip:
                    out = sdp(q, torch.cat(ip_k_list, dim=1), torch.cat(ip_v_list, dim=1), extra_options)

                # Process PuLID tasks second (via orthogonal projection)
                for payload, weight, start_pct, end_pct, control_type in active_tasks:
                    if percentage < start_pct or percentage >= end_pct:
                        continue
                    if control_type == "PuLID":
                        conds, unconds = payload
                        ip_k_c = conds[ip_index * 2].to(device=q.device, dtype=q.dtype)
                        ip_v_c = conds[ip_index * 2 + 1].to(device=q.device, dtype=q.dtype)
                        ip_k_uc = unconds[ip_index * 2].to(device=q.device, dtype=q.dtype)
                        ip_v_uc = unconds[ip_index * 2 + 1].to(device=q.device, dtype=q.dtype)

                        if q.shape[0] == 2:
                            ip_k = torch.cat([ip_k_uc, ip_k_c], dim=0)
                            ip_v = torch.cat([ip_v_uc, ip_v_c], dim=0)
                        else:
                            ip_k = ip_k_c
                            ip_v = ip_v_c

                        out_ip = sdp(q, ip_k, ip_v, extra_options)

                        # Orthogonal projection math
                        out_fp = out.to(dtype=torch.float32)
                        out_ip_fp = out_ip.to(dtype=torch.float32)
                        attn_map = q.to(dtype=torch.float32) @ ip_k.transpose(-2, -1).to(dtype=torch.float32)
                        attn_mean = attn_map.softmax(dim=-1).mean(dim=1, keepdim=True)
                        attn_mean = attn_mean[:, :, :5].sum(dim=-1, keepdim=True)

                        projection = (
                            torch.sum((out_fp * out_ip_fp), dim=-2, keepdim=True)
                            / torch.sum((out_fp * out_fp), dim=-2, keepdim=True).clamp(min=1e-6)
                            * out_fp
                        )
                        orthogonal = out_ip_fp + (attn_mean - 1) * projection
                        out = (out_fp + weight * orthogonal).to(dtype=q.dtype)

                return out

            return unified_attn_patcher

        for idx, key in enumerate(target_keys):
            transformer_options["patches_replace"]["attn2"][key] = make_unified_attn_patcher(idx)

        attach_time = time.perf_counter() - start_time
        log_telemetry("contextual_control_attach_complete", f"patched_layers={len(target_keys)} time={attach_time:.3f}s")

    def end(self) -> None:
        """Clears request-local contextual patch state to prevent leakage to subsequent requests."""
        if self.unet_spine and self.unet_spine.unet:
            opts = self.unet_spine.unet.model_options
            if "transformer_options" in opts:
                if "patches_replace" in opts["transformer_options"]:
                    if "attn2" in opts["transformer_options"]["patches_replace"]:
                        opts["transformer_options"]["patches_replace"]["attn2"].clear()
                        log_telemetry("contextual_request_state_cleared")
        self.unet_spine = None
