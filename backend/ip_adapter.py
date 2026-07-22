import math
import os
import shutil
import warnings
import hashlib
import gc
from collections import OrderedDict

import numpy as np
import safetensors.torch as sf
import torch
import torch.nn as nn
import ldm_patched.ldm.modules.attention as attention
import ldm_patched.modules.clip_vision
import backend.resources as runtime_resources
from einops import rearrange
from einops.layers.torch import Rearrange
from ldm_patched.modules.model_patcher import ModelPatcher
from ldm_patched.modules.ops import manual_cast

from backend.ops import use_patched_ops
from backend.pulid_encoders import IDEncoder
from modules.core import numpy_to_pytorch

_CONTEXTUAL_PAYLOAD_CACHE: OrderedDict[tuple, tuple[list[torch.Tensor], list[torch.Tensor]]] = OrderedDict()
_CONTEXTUAL_PAYLOAD_CACHE_LIMIT = 8


def _clone_contextual_payload(payload):
    if payload is None:
        return None
    ip_conds, ip_unconds = payload
    cloned_conds = [t.clone() if isinstance(t, torch.Tensor) else t for t in ip_conds]
    cloned_unconds = [t.clone() if isinstance(t, torch.Tensor) else t for t in ip_unconds]
    return (cloned_conds, cloned_unconds)


def clear_contextual_payload_cache() -> None:
    _CONTEXTUAL_PAYLOAD_CACHE.clear()


def _normalize_contextual_cache_kind(kind):
    normalized = str(kind or "").strip().lower().replace(" ", "_")
    if normalized in {"imageprompt", "image_prompt"}:
        return "image_prompt"
    if normalized in {"faceidv2", "faceid_v2"}:
        return "faceid_v2"
    if normalized == "pulid":
        return "pulid"
    return normalized or None


def _build_contextual_payload_cache_key(
    img,
    *,
    cn_type,
    model_path,
    clip_vision_path=None,
    ip_negative_path=None,
    insightface_model_names=None,
):
    cn_img_hash = hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()
    insightface_model_names_tuple = tuple(insightface_model_names) if insightface_model_names is not None else None
    return (cn_img_hash, cn_type, model_path, clip_vision_path, ip_negative_path, insightface_model_names_tuple)


SD_V12_CHANNELS = [320] * 4 + [640] * 4 + [1280] * 4 + [1280] * 6 + [640] * 6 + [320] * 6 + [1280] * 2
SD_XL_CHANNELS = [640] * 8 + [1280] * 40 + [1280] * 60 + [640] * 12 + [1280] * 20

clip_vision_models = {}
ip_negative = {}
contextual_models = {}
insightface_apps = {}


def _offload_patcher(patcher):
    if patcher is None:
        return
    try:
        patcher.detach()
    except Exception:
        pass


def _offload_clip_vision_model(clip_model):
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


def _offload_contextual_entry(entry):
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


def apply_contextual_residency(mode='offload', *, clip_vision_action=None, insightface_action=None):
    global clip_vision_models, ip_negative, contextual_models, insightface_apps

    clip_vision_action = clip_vision_action or mode
    insightface_action = insightface_action or mode
    actions = {
        'mode': mode,
        'clip_vision_action': clip_vision_action,
        'insightface_action': insightface_action,
        'contextual_models': len(contextual_models),
        'clip_vision_models': len(clip_vision_models),
        'insightface_apps': len(insightface_apps),
    }

    for entry in contextual_models.values():
        _offload_contextual_entry(entry)
    for clip_model in clip_vision_models.values():
        _offload_clip_vision_model(clip_model)

    if mode == 'destroy':
        contextual_models = {}
    if clip_vision_action == 'destroy':
        clip_vision_models = {}
    if insightface_action == 'destroy':
        insightface_apps = {}
    if mode == 'destroy':
        ip_negative = {}

    return actions


def release_contextual_preprocess_support(*, reclaim_device_memory=True):
    """Destroy all contextual support while preserving reusable CPU payloads."""
    actions = apply_contextual_residency(
        'destroy',
        clip_vision_action='destroy',
        insightface_action='destroy',
    )
    actions['payload_cache_entries'] = len(_CONTEXTUAL_PAYLOAD_CACHE)

    if reclaim_device_memory:
        gc.collect()
        runtime_resources.soft_empty_cache(force=True)
    return actions


def has_contextual_preprocess_support():
    return bool(
        contextual_models
        or clip_vision_models
        or ip_negative
        or insightface_apps
    )


def release_pulid_preprocess_support(*, reclaim_device_memory=True):
    """Compatibility alias for payload-only contextual retention."""
    return release_contextual_preprocess_support(
        reclaim_device_memory=reclaim_device_memory,
    )


def sdp(q, k, v, extra_options):
    return attention.optimized_attention(q, k, v, heads=extra_options["n_heads"], mask=None)


def masked_mean(tensor, *, dim, mask=None):
    if mask is None:
        return tensor.mean(dim=dim)

    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, "b n -> b n 1")
    masked_tensor = tensor.masked_fill(~mask, 0.0)
    return masked_tensor.sum(dim=dim) / denom.clamp(min=1e-5)


def reshape_tensor(x, heads):
    batch, length, width = x.shape
    x = x.view(batch, length, heads, -1)
    x = x.transpose(1, 2)
    return x.reshape(batch, heads, length, -1)


def _largest_face(faces):
    return sorted(faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))[-1]


def _sorted_kv_modules(ip_layers_model):
    def sort_key(item):
        name = item[0]
        parts = name.split("_")
        try:
            index = int(parts[0])
        except Exception:
            index = 0
        is_v = 1 if "_to_v_" in name else 0
        return (index, is_v, name)

    return [module for _, module in sorted(ip_layers_model.to_kvs.items(), key=sort_key)]


class FeedForward(nn.Sequential):
    def __init__(self, dim, mult=4):
        inner_dim = int(dim * mult)
        super().__init__(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False),
            nn.GELU(),
            nn.Linear(inner_dim, dim, bias=False),
        )


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        x = self.norm1(x)
        latents = self.norm2(latents)

        batch, length, _ = latents.shape
        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        out = out.permute(0, 2, 1, 3).reshape(batch, length, -1)

        return self.to_out(out)


class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        max_seq_len=257,
        apply_pos_emb=False,
        num_latents_mean_pooled=0,
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim) if apply_pos_emb else None
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        self.to_latents_from_mean_pooled_seq = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange("b (n d) -> b n d", n=num_latents_mean_pooled),
            )
            if num_latents_mean_pooled > 0
            else None
        )
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        if self.pos_emb is not None:
            seq_len, device = x.shape[1], x.device
            pos_emb = self.pos_emb(torch.arange(seq_len, device=device))
            x = x + pos_emb

        latents = self.latents.repeat(x.size(0), 1, 1)
        x = self.proj_in(x)

        if self.to_latents_from_mean_pooled_seq:
            meanpooled_seq = masked_mean(x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim=-2)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        return self.norm_out(latents)


class FacePerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim=768,
        depth=4,
        dim_head=64,
        heads=16,
        embedding_dim=1280,
        output_dim=768,
        ff_mult=4,
    ):
        super().__init__()
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, latents, x):
        x = self.proj_in(x)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        latents = self.proj_out(latents)
        return self.norm_out(latents)


class ImageProjModel(nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        x = self.proj(image_embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        return self.norm(x)


class ProjModelFaceIdPlus(nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, clip_embeddings_dim=1280, num_tokens=4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            nn.GELU(),
            nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = nn.LayerNorm(cross_attention_dim)
        self.perceiver_resampler = FacePerceiverResampler(
            dim=cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=cross_attention_dim // 64,
            embedding_dim=clip_embeddings_dim,
            output_dim=cross_attention_dim,
            ff_mult=4,
        )

    def forward(self, id_embeds, clip_embeds, scale=1.0, shortcut=False):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        out = self.perceiver_resampler(x, clip_embeds)
        if shortcut:
            out = x + scale * out
        return out


class ToKV(nn.Module):
    def __init__(self, cross_attention_dim=None, state_dict=None):
        super().__init__()
        self.to_kvs = nn.ModuleDict()

        if state_dict is None:
            channels = SD_XL_CHANNELS if cross_attention_dim == 2048 else SD_V12_CHANNELS
            for index, channel in enumerate(channels):
                self.to_kvs[f"{index}_to_k_ip"] = nn.Linear(cross_attention_dim, channel, bias=False)
                self.to_kvs[f"{index}_to_v_ip"] = nn.Linear(cross_attention_dim, channel, bias=False)
            return

        for key, value in state_dict.items():
            if not key.endswith(("to_k_ip.weight", "to_v_ip.weight")):
                continue
            clean_key = key.replace(".weight", "").replace(".", "_")
            self.to_kvs[clean_key] = nn.Linear(value.shape[1], value.shape[0], bias=False)
            self.to_kvs[clean_key].weight.data = value

    def load_state_dict_ordered(self, state_dict):
        ordered = []
        for index in range(4096):
            for suffix in ["k", "v"]:
                key = f"{index}.to_{suffix}_ip.weight"
                if key in state_dict:
                    ordered.append((f"{index}_to_{suffix}_ip", state_dict[key]))
        for clean_key, value in ordered:
            self.to_kvs[clean_key].weight = torch.nn.Parameter(value, requires_grad=False)


class IPAdapterModel(nn.Module):
    def __init__(
        self,
        state_dict,
        plus,
        cross_attention_dim=768,
        clip_embeddings_dim=1024,
        clip_extra_context_tokens=4,
        sdxl_plus=False,
    ):
        super().__init__()
        self.plus = plus
        self.kind = "image_prompt"
        if self.plus:
            self.image_proj_model = Resampler(
                dim=1280 if sdxl_plus else cross_attention_dim,
                depth=4,
                dim_head=64,
                heads=20 if sdxl_plus else 12,
                num_queries=clip_extra_context_tokens,
                embedding_dim=clip_embeddings_dim,
                output_dim=cross_attention_dim,
                ff_mult=4,
            )
        else:
            self.image_proj_model = ImageProjModel(
                cross_attention_dim=cross_attention_dim,
                clip_embeddings_dim=clip_embeddings_dim,
                clip_extra_context_tokens=clip_extra_context_tokens,
            )

        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        self.ip_layers = ToKV(cross_attention_dim=cross_attention_dim)
        self.ip_layers.load_state_dict_ordered(state_dict["ip_adapter"])


class FaceIDAdapterModel(nn.Module):
    def __init__(self, state_dict, cross_attention_dim=2048, clip_embeddings_dim=1280, num_tokens=4):
        super().__init__()
        self.kind = "faceid_v2"
        self.plus = True
        self.faceid_shortcut = True
        self.faceid_scale = 1.0
        self.image_proj_model = ProjModelFaceIdPlus(
            cross_attention_dim=cross_attention_dim,
            id_embeddings_dim=512,
            clip_embeddings_dim=clip_embeddings_dim,
            num_tokens=num_tokens,
        )
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        self.ip_layers = ToKV(state_dict=state_dict["ip_adapter"])


class PulidAdapterModel(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        self.kind = "pulid"
        self.image_proj_model = IDEncoder()
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        self.ip_layers = ToKV(state_dict=state_dict["ip_adapter"])


def _ensure_clip_vision(path):
    if not isinstance(path, str):
        return None
    if path not in clip_vision_models:
        clip_vision_models[path] = ldm_patched.modules.clip_vision.load(path)
    return clip_vision_models[path]


def _select_clip_vision_model(path):
    clip_model = _ensure_clip_vision(path)
    if clip_model is not None:
        return clip_model
    return next(iter(clip_vision_models.values()), None)


def _ensure_ip_negative(path):
    if not isinstance(path, str):
        return None
    if path not in ip_negative:
        ip_negative[path] = sf.load_file(path)["data"]
    return ip_negative[path]


def _normalize_pulid_state_dict(state_dict):
    if "image_proj" in state_dict and "ip_adapter" in state_dict:
        return state_dict

    normalized = {"image_proj": {}, "ip_adapter": {}}
    for key, value in state_dict.items():
        if key.startswith("image_proj."):
            normalized["image_proj"][key.replace("image_proj.", "")] = value
        elif key.startswith("ip_adapter."):
            normalized["ip_adapter"][key.replace("ip_adapter.", "")] = value
        elif key.startswith("id_adapter."):
            normalized["image_proj"][key.replace("id_adapter.", "")] = value
        elif key.startswith("id_adapter_attn_layers."):
            clean_key = key.replace("id_adapter_attn_layers.", "")
            clean_key = clean_key.replace(".id_to_k.weight", ".to_k_ip.weight")
            clean_key = clean_key.replace(".id_to_v.weight", ".to_v_ip.weight")
            normalized["ip_adapter"][clean_key] = value
    return normalized


def detect_model_kind(state_dict):
    ip_state = state_dict.get("ip_adapter", {})
    image_proj_state = state_dict.get("image_proj", {})

    if "0.to_q_lora.down.weight" in ip_state or "perceiver_resampler.proj_in.weight" in image_proj_state:
        return "faceid_v2"
    if "id_embedding_mapping.0.weight" in image_proj_state and any(key.endswith("to_k_ip.weight") for key in ip_state):
        return "pulid"
    if "body.0.weight" in image_proj_state and any(key.startswith("mapping_") for key in image_proj_state):
        return "pulid"
    return "image_prompt"


def load_contextual_model(model_path, clip_vision_path=None, ip_negative_path=None):
    global contextual_models

    if not isinstance(model_path, str):
        return None

    _ensure_clip_vision(clip_vision_path)
    _ensure_ip_negative(ip_negative_path)

    if model_path in contextual_models:
        return contextual_models[model_path]

    load_device = runtime_resources.get_torch_device()
    offload_device = torch.device("cpu")
    use_fp16 = runtime_resources.should_use_fp16(device=load_device)
    dtype = torch.float16 if use_fp16 else torch.float32

    if model_path.lower().endswith(".safetensors"):
        raw_state_dict = sf.load_file(model_path)
    else:
        raw_state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    state_dict = _normalize_pulid_state_dict(raw_state_dict)
    model_kind = detect_model_kind(state_dict)

    with use_patched_ops(manual_cast):
        if model_kind == "faceid_v2":
            model = FaceIDAdapterModel(state_dict)
        elif model_kind == "pulid":
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
    contextual_models[model_path] = entry
    return entry


def load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_path):
    return load_contextual_model(ip_adapter_path, clip_vision_path=clip_vision_path, ip_negative_path=ip_negative_path)


def _prepare_insightface_model_dir(root, model_name):
    models_root = os.path.join(root, "models")
    target_dir = os.path.join(models_root, model_name)
    legacy_dir = os.path.join(root, model_name)
    nested_dir = os.path.join(target_dir, model_name)

    os.makedirs(models_root, exist_ok=True)

    if os.path.isdir(legacy_dir):
        os.makedirs(target_dir, exist_ok=True)
        for name in os.listdir(legacy_dir):
            src = os.path.join(legacy_dir, name)
            dst = os.path.join(target_dir, name)
            if os.path.exists(dst):
                continue
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

    if os.path.isdir(nested_dir):
        for name in os.listdir(nested_dir):
            src = os.path.join(nested_dir, name)
            dst = os.path.join(target_dir, name)
            if os.path.exists(dst):
                continue
            shutil.move(src, dst)
        try:
            os.rmdir(nested_dir)
        except OSError:
            pass

    return target_dir


def load_insightface(model_name="antelopev2", providers=None, root=None):
    global insightface_apps

    if model_name in insightface_apps:
        return insightface_apps[model_name]

    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise ImportError("InsightFace is required for FaceID V2 and PuLID support.") from exc

    if root is None:
        import modules.config as config

        root = config.path_insightface

    if providers is None:
        providers = ["CPUExecutionProvider"]
        try:
            import onnxruntime

            available = set(onnxruntime.get_available_providers())
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            providers = ["CPUExecutionProvider"]

    _prepare_insightface_model_dir(root, model_name)
    app = FaceAnalysis(name=model_name, root=root, providers=providers)
    ctx_id = 0 if any(provider.startswith("CUDA") for provider in providers) else -1
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    insightface_apps[model_name] = app
    return app


def preprocess(img, model_path, clip_vision_path=None, ip_negative_path=None, insightface_model_names=None, cache_kind=None):
    if has_contextual_preprocess_support():
        release_contextual_preprocess_support(reclaim_device_memory=True)
    try:
        return _preprocess(
            img,
            model_path,
            clip_vision_path=clip_vision_path,
            ip_negative_path=ip_negative_path,
            insightface_model_names=insightface_model_names,
            cache_kind=cache_kind,
        )
    finally:
        if has_contextual_preprocess_support():
            release_contextual_preprocess_support(reclaim_device_memory=True)


def _preprocess(img, model_path, clip_vision_path=None, ip_negative_path=None, insightface_model_names=None, cache_kind=None):
    try:
        from backend.sdxl_unified_runtime import _PREPROCESSOR_METRICS
    except ImportError:
        _PREPROCESSOR_METRICS = None

    cache_key = None
    normalized_cache_kind = _normalize_contextual_cache_kind(cache_kind)
    if normalized_cache_kind is not None:
        cache_key = _build_contextual_payload_cache_key(
            img,
            cn_type=normalized_cache_kind,
            model_path=model_path,
            clip_vision_path=clip_vision_path,
            ip_negative_path=ip_negative_path,
            insightface_model_names=insightface_model_names,
        )
        cached_val = _CONTEXTUAL_PAYLOAD_CACHE.get(cache_key)
        if cached_val is not None:
            _CONTEXTUAL_PAYLOAD_CACHE.move_to_end(cache_key)
            if _PREPROCESSOR_METRICS is not None:
                _PREPROCESSOR_METRICS["contextual_hits"] += 1.0
            return _clone_contextual_payload(cached_val)

    entry = load_contextual_model(model_path, clip_vision_path=clip_vision_path, ip_negative_path=ip_negative_path)
    if entry is None:
        return None

    if cache_key is None:
        cn_type = _normalize_contextual_cache_kind(entry["kind"]) or entry["kind"]
        cache_key = _build_contextual_payload_cache_key(
            img,
            cn_type=cn_type,
            model_path=model_path,
            clip_vision_path=clip_vision_path,
            ip_negative_path=ip_negative_path,
            insightface_model_names=insightface_model_names,
        )
        cached_val = _CONTEXTUAL_PAYLOAD_CACHE.get(cache_key)
        if cached_val is not None:
            _CONTEXTUAL_PAYLOAD_CACHE.move_to_end(cache_key)
            if _PREPROCESSOR_METRICS is not None:
                _PREPROCESSOR_METRICS["contextual_hits"] += 1.0
            return _clone_contextual_payload(cached_val)

    if _PREPROCESSOR_METRICS is not None:
        _PREPROCESSOR_METRICS["contextual_misses"] += 1.0

    if entry["kind"] == "faceid_v2":
        res = preprocess_faceid(
            img,
            model_path,
            clip_vision_path=clip_vision_path,
            insightface_model_names=insightface_model_names,
        )
    elif entry["kind"] == "pulid":
        raise NotImplementedError("PuLID preprocessing is not wired yet in the backend contextual path.")
    else:
        res = preprocess_ip_adapter(
            img,
            model_path,
            clip_vision_path=clip_vision_path,
            ip_negative_path=ip_negative_path,
        )

    if res is not None:
        _CONTEXTUAL_PAYLOAD_CACHE[cache_key] = _clone_contextual_payload(res)
        _CONTEXTUAL_PAYLOAD_CACHE.move_to_end(cache_key)
        while len(_CONTEXTUAL_PAYLOAD_CACHE) > _CONTEXTUAL_PAYLOAD_CACHE_LIMIT:
            _CONTEXTUAL_PAYLOAD_CACHE.popitem(last=False)

    return res


def preprocess_ip_adapter(img, model_path, clip_vision_path=None, ip_negative_path=None):
    entry = contextual_models[model_path]
    clip_model = _select_clip_vision_model(clip_vision_path)
    if clip_model is None:
        raise RuntimeError("CLIP vision must be loaded before preprocessing IP-Adapter inputs.")

    try:
        clip_patcher = getattr(clip_model, 'patcher', None)
        if clip_patcher is not None:
            runtime_resources.load_model_gpu(clip_patcher)
        else:
            try:
                clip_model.model.to(runtime_resources.get_torch_device())
            except Exception:
                pass

        outputs = clip_model.encode_image(numpy_to_pytorch(img))
        adapter_model = entry["model"]
        image_proj_model = entry["image_proj_model"]
        ip_layers = entry["ip_layers"]

        cond = outputs.penultimate_hidden_states if adapter_model.plus else outputs.image_embeds
        cond = cond.to(device=adapter_model.load_device, dtype=adapter_model.dtype)

        runtime_resources.load_model_gpu(image_proj_model)
        cond = image_proj_model.model(cond).to(device=adapter_model.load_device, dtype=adapter_model.dtype)

        runtime_resources.load_model_gpu(ip_layers)
        kv_modules = _sorted_kv_modules(ip_layers.model)

        negative = _ensure_ip_negative(ip_negative_path)
        if negative is None:
            raise RuntimeError("IP-Adapter negative embedding is required for contextual preprocessing.")
        negative = negative.to(device=adapter_model.load_device, dtype=adapter_model.dtype)
        ip_unconds = [module(negative).cpu() for module in kv_modules]

        ip_conds = [module(cond).cpu() for module in kv_modules]
        return ip_conds, ip_unconds
    finally:
        _offload_contextual_entry(entry)
        _offload_clip_vision_model(clip_model)


def _detect_faces(face_app, bgr_image):
    for size in range(640, 256, -64):
        face_app.det_model.input_size = (size, size)
        faces = face_app.get(bgr_image)
        if faces:
            return faces
    return []


def preprocess_faceid(img, model_path, clip_vision_path=None, insightface_model_names=None):
    from insightface.utils import face_align

    entry = contextual_models[model_path]
    clip_model = _ensure_clip_vision(clip_vision_path)
    if clip_model is None:
        raise RuntimeError("CLIP vision must be loaded before preprocessing FaceID inputs.")

    names = insightface_model_names or ["antelopev2", "buffalo_l"]
    face_app = None
    faces = []
    bgr_image = np.ascontiguousarray(img[:, :, ::-1])
    for model_name in names:
        face_app = load_insightface(model_name=model_name)
        faces = _detect_faces(face_app, bgr_image)
        if faces:
            break

    if not faces or face_app is None:
        raise RuntimeError("FaceID V2 preprocessing could not detect a face in the reference image.")

    adapter_model = entry["model"]
    image_proj_model = entry["image_proj_model"]
    ip_layers = entry["ip_layers"]

    cond_embeds = []
    uncond_embeds = []

    try:
        clip_patcher = getattr(clip_model, 'patcher', None)
        if clip_patcher is not None:
            runtime_resources.load_model_gpu(clip_patcher)
        else:
            try:
                clip_model.model.to(runtime_resources.get_torch_device())
            except Exception:
                pass

        runtime_resources.load_model_gpu(image_proj_model)

        for face in sorted(faces, key=lambda current: (current.bbox[2] - current.bbox[0]) * (current.bbox[3] - current.bbox[1]), reverse=True):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="`estimate` is deprecated since version 0.26",
                    category=FutureWarning,
                )
                aligned = face_align.norm_crop(bgr_image, landmark=face.kps, image_size=256)
            aligned_rgb = np.ascontiguousarray(aligned[:, :, ::-1].copy())
            outputs = clip_model.encode_image(numpy_to_pytorch(aligned_rgb))
            clip_embeds = outputs.penultimate_hidden_states.to(device=adapter_model.load_device, dtype=adapter_model.dtype)
            cond_id = torch.from_numpy(face.normed_embedding).unsqueeze(0).to(device=adapter_model.load_device, dtype=adapter_model.dtype)
            zero_id = torch.zeros_like(cond_id)
            zero_clip = torch.zeros_like(clip_embeds)

            cond_embeds.append(
                image_proj_model.model(
                    cond_id,
                    clip_embeds,
                    scale=adapter_model.faceid_scale,
                    shortcut=adapter_model.faceid_shortcut,
                )
            )
            uncond_embeds.append(
                image_proj_model.model(
                    zero_id,
                    zero_clip,
                    scale=adapter_model.faceid_scale,
                    shortcut=adapter_model.faceid_shortcut,
                )
            )

        cond = torch.mean(torch.cat(cond_embeds, dim=0), dim=0, keepdim=True).to(device=adapter_model.load_device, dtype=adapter_model.dtype)
        uncond = torch.mean(torch.cat(uncond_embeds, dim=0), dim=0, keepdim=True).to(device=adapter_model.load_device, dtype=adapter_model.dtype)

        runtime_resources.load_model_gpu(ip_layers)
        kv_modules = _sorted_kv_modules(ip_layers.model)
        ip_conds = [module(cond).cpu() for module in kv_modules]
        ip_unconds = [module(uncond).cpu() for module in kv_modules]
        return ip_conds, ip_unconds
    finally:
        _offload_contextual_entry(entry)
        _offload_clip_vision_model(clip_model)


def patch_model(model, tasks):
    new_model = model.clone()

    def normalize_task(task):
        if len(task) >= 4:
            return task[0], float(task[1]), float(task[2]), float(task[3])
        if len(task) == 3:
            return task[0], float(task[1]), float(task[2]), 0.0
        raise ValueError(f"Unexpected contextual task shape: {task!r}")

    def make_attn_patcher(ip_index):
        def patcher(n, context_attn2, value_attn2, extra_options):
            org_dtype = n.dtype
            current_step = float(model.model.diffusion_model.current_step.detach().cpu().numpy()[0])
            cond_or_uncond = extra_options["cond_or_uncond"]

            q = n
            k = [context_attn2]
            v = [value_attn2]

            for task in tasks:
                (cs, ucs), cn_stop, cn_weight, cn_start = normalize_task(task)
                if current_step < cn_start or current_step >= cn_stop:
                    continue

                ip_k_c = cs[ip_index * 2].to(q)
                ip_v_c = cs[ip_index * 2 + 1].to(q)
                ip_k_uc = ucs[ip_index * 2].to(q)
                ip_v_uc = ucs[ip_index * 2 + 1].to(q)

                ip_k = torch.cat([(ip_k_c, ip_k_uc)[i] for i in cond_or_uncond], dim=0)
                ip_v = torch.cat([(ip_v_c, ip_v_uc)[i] for i in cond_or_uncond], dim=0)

                ip_v_mean = torch.mean(ip_v, dim=1, keepdim=True)
                ip_v_offset = ip_v - ip_v_mean

                _, _, channels = ip_k.shape
                channel_penalty = float(channels) / 1280.0
                weight = cn_weight * channel_penalty

                ip_k = ip_k * weight
                ip_v = ip_v_offset + ip_v_mean * weight

                k.append(ip_k)
                v.append(ip_v)

            out = sdp(q, torch.cat(k, dim=1), torch.cat(v, dim=1), extra_options)
            return out.to(dtype=org_dtype)

        return patcher

    def set_model_patch_replace(model_clone, number, key):
        transformer_options = model_clone.model_options["transformer_options"]
        if "patches_replace" not in transformer_options:
            transformer_options["patches_replace"] = {}
        if "attn2" not in transformer_options["patches_replace"]:
            transformer_options["patches_replace"]["attn2"] = {}
        if key not in transformer_options["patches_replace"]["attn2"]:
            transformer_options["patches_replace"]["attn2"][key] = make_attn_patcher(number)

    number = 0
    for block_id in [4, 5, 7, 8]:
        block_indices = range(2) if block_id in [4, 5] else range(10)
        for index in block_indices:
            set_model_patch_replace(new_model, number, ("input", block_id, index))
            number += 1

    for block_id in range(6):
        block_indices = range(2) if block_id in [3, 4, 5] else range(10)
        for index in block_indices:
            set_model_patch_replace(new_model, number, ("output", block_id, index))
            number += 1

    for index in range(10):
        set_model_patch_replace(new_model, number, ("middle", 0, index))
        number += 1

    return new_model

