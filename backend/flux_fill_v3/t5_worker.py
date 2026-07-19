from __future__ import annotations

import gc
import json
import logging
import hashlib
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import T5TokenizerFast

import backend.patching as patching
from backend import precision
import ldm_patched.modules.ops as base_ops
import ldm_patched.modules.sd1_clip as sd1_clip
import ldm_patched.modules.utils as comfy_utils
from backend import resources
from backend.flux_fill_v3.contracts import (
    FluxFillRequest,
    T5PostureKind,
)
from backend.flux_fill_v3.conditioning_loader import (
    FluxEmptyConditioning,
    format_flux_conditioning_memory_summary,
    load_flux_empty_conditioning_cache,
)
from ldm_patched.ldm.modules.attention import optimized_attention_for_device

logger = logging.getLogger(__name__)

_FLUX_FILL_V3_ASSET_ROOT = Path(__file__).resolve().parent / "assets"
_T5_FIXED_LENGTH = 256
_CLIP_L_KEY = "text_model.encoder.layers.1.mlp.fc1.weight"
_T5_KEY = "encoder.block.23.layer.1.DenseReluDense.wi_1.weight"
_T5_KEY_OLD = "encoder.block.23.layer.1.DenseReluDense.wi.weight"
_T5_LAZY_GC_DEFAULT_INTERVAL = 4
_T5_LAZY_GC_LOW_HEADROOM_INTERVAL = 2
_T5_LAZY_GC_CRITICAL_INTERVAL = 1
_T5_LAZY_GC_RECHECK_BLOCKS = 2


def _describe_cache_path(path: Path) -> str:
    try:
        size_mb = float(path.stat().st_size) / (1024 * 1024)
        return f"exists={path.exists()} size={size_mb:.3f}MB"
    except OSError:
        return f"exists={path.exists()} size=n/a"


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        coerced = int(default)
    return max(1, coerced)


def _normalize_headroom_mb(value: Any) -> float | None:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    if normalized <= 0.0:
        return None
    return normalized


def _resolve_disk_paged_t5_gc_interval(
    *,
    free_ram_mb: float | None,
    low_headroom_mb: float | None,
    critical_headroom_mb: float | None,
    profile_name: str = "",
    healthy_interval: int | None = None,
) -> int:
    resolved_healthy_interval = _coerce_positive_int(
        healthy_interval if healthy_interval is not None else _T5_LAZY_GC_DEFAULT_INTERVAL,
        _T5_LAZY_GC_DEFAULT_INTERVAL,
    )
    if free_ram_mb is not None:
        if critical_headroom_mb is not None and free_ram_mb < critical_headroom_mb:
            return _T5_LAZY_GC_CRITICAL_INTERVAL
        if low_headroom_mb is not None and free_ram_mb < low_headroom_mb:
            return _T5_LAZY_GC_LOW_HEADROOM_INTERVAL
        return resolved_healthy_interval

    if healthy_interval is not None:
        return resolved_healthy_interval
    if profile_name in {"colab_free", "local_low_vram"}:
        return _T5_LAZY_GC_LOW_HEADROOM_INTERVAL
    return _T5_LAZY_GC_DEFAULT_INTERVAL


def _resolve_disk_paged_t5_gc_config(*, override_interval: int | None = None) -> dict[str, Any]:
    policy_summary: dict[str, Any] = {}
    profile_name = ""
    free_ram_mb: float | None = None

    try:
        policy_summary = dict(resources.memory_policy_summary() or {})
    except Exception:
        policy_summary = {}

    try:
        profile = resources.active_memory_environment_profile()
        profile_name = str(getattr(profile, "name", "") or "").strip().lower()
    except Exception:
        profile_name = ""

    low_headroom_mb = _normalize_headroom_mb(policy_summary.get("low_ram_headroom_mb")) or 3072.0
    critical_headroom_mb = _normalize_headroom_mb(policy_summary.get("critical_ram_headroom_mb")) or 1536.0

    try:
        snapshot = resources.capture_memory_snapshot(notes={"tag": "disk_paged_t5_gc_config"})
        free_ram_mb = _normalize_headroom_mb(getattr(snapshot, "free_ram_mb", None))
    except Exception:
        free_ram_mb = None

    if override_interval is not None:
        interval = _resolve_disk_paged_t5_gc_interval(
            free_ram_mb=free_ram_mb,
            low_headroom_mb=low_headroom_mb,
            critical_headroom_mb=critical_headroom_mb,
            profile_name=profile_name,
            healthy_interval=override_interval,
        )
        return {
            "interval": interval,
            "healthy_interval": _coerce_positive_int(override_interval, _T5_LAZY_GC_DEFAULT_INTERVAL),
            "recheck_blocks": 1 if interval <= 1 else _T5_LAZY_GC_RECHECK_BLOCKS,
            "low_headroom_mb": low_headroom_mb,
            "critical_headroom_mb": critical_headroom_mb,
            "initial_free_ram_mb": free_ram_mb,
            "profile_name": profile_name,
            "adaptive": True,
            "override_interval": interval,
        }

    interval = _resolve_disk_paged_t5_gc_interval(
        free_ram_mb=free_ram_mb,
        low_headroom_mb=low_headroom_mb,
        critical_headroom_mb=critical_headroom_mb,
        profile_name=profile_name,
    )
    recheck_blocks = 1 if interval <= 1 else _T5_LAZY_GC_RECHECK_BLOCKS
    return {
        "interval": interval,
        "healthy_interval": interval,
        "recheck_blocks": recheck_blocks,
        "low_headroom_mb": low_headroom_mb,
        "critical_headroom_mb": critical_headroom_mb,
        "initial_free_ram_mb": free_ram_mb,
        "profile_name": profile_name,
        "adaptive": True,
        "override_interval": None,
    }


def flux_t5_tokenizer_path() -> Path:
    return _FLUX_FILL_V3_ASSET_ROOT / "t5_tokenizer"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _pick_t5_ops(model_options: dict[str, Any] | None) -> Any:
    custom_ops = (model_options or {}).get("custom_operations")
    return custom_ops if custom_ops is not None else base_ops.manual_cast


def _normalize_t5_loader_policy(policy: str | None, *, t5_path: str | Path | None = None) -> str:
    if policy is None or str(policy).strip() == "":
        if t5_path is not None:
            resolved = Path(t5_path)
            if resolved.suffix.lower() == ".safetensors":
                return "stream_safetensors_runtime"
        return "stream_safetensors_runtime"

    value = str(policy).strip().lower().replace("-", "_").replace(" ", "_")
    if value not in {"stream_safetensors_runtime", "eager"}:
        logger.warning(
            "[Flux Telemetry] Rejected unsupported T5 loader policy policy=%s",
            policy,
        )
        raise ValueError(
            f"Unsupported or rejected T5 loader policy: {policy!r}. Only 'stream_safetensors_runtime' and 'eager' are accepted."
        )
    if t5_path is not None:
        resolved = Path(t5_path)
        if resolved.suffix.lower() != ".safetensors":
            logger.warning(
                "[Flux Telemetry] Rejected non-safetensors T5 path path=%s suffix=%s",
                resolved,
                resolved.suffix,
            )
            raise ValueError("T5 path must end in .safetensors.")
    return value


def _load_t5_tokenizer_fast():
    return T5TokenizerFast


class FixedLengthT5Tokenizer:
    def __init__(
        self,
        tokenizer_path: str | Path | None = None,
        *,
        fixed_length: int = _T5_FIXED_LENGTH,
    ) -> None:
        tokenizer_path = Path(tokenizer_path) if tokenizer_path is not None else flux_t5_tokenizer_path()
        tokenizer_cls = _load_t5_tokenizer_fast()
        self.tokenizer = tokenizer_cls.from_pretrained(str(tokenizer_path))
        self.fixed_length = int(fixed_length)
        empty = self.tokenizer("")["input_ids"]
        self.start_token = None
        self.end_token = int(empty[0])
        self.pad_token = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)

    def tokenize_with_weights(self, text: str, return_word_ids: bool = False, **_: Any) -> list[list[tuple[int, float] | tuple[int, float, int]]]:
        parsed_weights = sd1_clip.token_weights(sd1_clip.escape_important(text), 1.0)
        token_stream: list[tuple[int, float, int]] = []
        word_id = 1

        for weighted_segment, weight in parsed_weights:
            words = sd1_clip.unescape_important(weighted_segment).replace("\n", " ").split(" ")
            for word in (piece for piece in words if piece != ""):
                token_ids = self.tokenizer(word)["input_ids"][:-1]
                for token_id in token_ids:
                    if len(token_stream) >= self.fixed_length - 1:
                        break
                    token_stream.append((int(token_id), float(weight), word_id))
                if len(token_stream) >= self.fixed_length - 1:
                    break
                word_id += 1
            if len(token_stream) >= self.fixed_length - 1:
                break

        token_stream.append((self.end_token, 1.0, 0))
        while len(token_stream) < self.fixed_length:
            token_stream.append((self.pad_token, 1.0, 0))

        if return_word_ids:
            return [token_stream]
        return [[(token_id, weight) for token_id, weight, _ in token_stream]]


class T5LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, *, dtype=None, device=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, dtype=dtype, device=device))
        self.variance_epsilon = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = f"{prefix}weight"
        value = state_dict.get(key)
        if value is None:
            missing_keys.append(key)
            return
        if hasattr(value, "load") and callable(value.load):
            value = value.load()
        if not isinstance(value, torch.Tensor):
            error_msgs.append(f"{key} expected tensor-like value, got {type(value).__name__}.")
            return
        self.weight.data.copy_(value.to(device=self.weight.device, dtype=self.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return base_ops.cast_to_input(self.weight, x) * x


_ACTIVATIONS = {
    "gelu_pytorch_tanh": lambda tensor: torch.nn.functional.gelu(tensor, approximate="tanh"),
    "relu": torch.nn.functional.relu,
}


class T5DenseActDense(torch.nn.Module):
    def __init__(self, model_dim: int, ff_dim: int, ff_activation: str, *, dtype=None, device=None, operations=base_ops.manual_cast):
        super().__init__()
        self.wi = operations.Linear(model_dim, ff_dim, bias=False, dtype=dtype, device=device)
        self.wo = operations.Linear(ff_dim, model_dim, bias=False, dtype=dtype, device=device)
        self.act = _ACTIVATIONS[ff_activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wo(self.act(self.wi(x)))


class T5DenseGatedActDense(torch.nn.Module):
    def __init__(self, model_dim: int, ff_dim: int, ff_activation: str, *, dtype=None, device=None, operations=base_ops.manual_cast):
        super().__init__()
        self.wi_0 = operations.Linear(model_dim, ff_dim, bias=False, dtype=dtype, device=device)
        self.wi_1 = operations.Linear(model_dim, ff_dim, bias=False, dtype=dtype, device=device)
        self.wo = operations.Linear(ff_dim, model_dim, bias=False, dtype=dtype, device=device)
        self.act = _ACTIVATIONS[ff_activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wo(self.act(self.wi_0(x)) * self.wi_1(x))


class T5LayerFF(torch.nn.Module):
    def __init__(self, model_dim: int, ff_dim: int, ff_activation: str, gated_act: bool, *, dtype=None, device=None, operations=base_ops.manual_cast):
        super().__init__()
        dense_cls = T5DenseGatedActDense if gated_act else T5DenseActDense
        self.DenseReluDense = dense_cls(model_dim, ff_dim, ff_activation, dtype=dtype, device=device, operations=operations)
        self.layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.DenseReluDense(self.layer_norm(x))


class T5Attention(torch.nn.Module):
    def __init__(self, model_dim: int, inner_dim: int, num_heads: int, relative_attention_bias: bool, *, dtype=None, device=None, operations=base_ops.manual_cast):
        super().__init__()
        self.q = operations.Linear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.k = operations.Linear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.v = operations.Linear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.o = operations.Linear(inner_dim, model_dim, bias=False, dtype=dtype, device=device)
        self.num_heads = num_heads
        self.relative_attention_bias = None
        if relative_attention_bias:
            self.relative_attention_num_buckets = 32
            self.relative_attention_max_distance = 128
            self.relative_attention_bias = operations.Embedding(self.relative_attention_num_buckets, self.num_heads, device=device, dtype=dtype)

    @staticmethod
    def _relative_position_bucket(relative_position: torch.Tensor, *, bidirectional: bool = True, num_buckets: int = 32, max_distance: int = 128) -> torch.Tensor:
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / torch.log(torch.tensor(max_distance / max_exact, device=relative_position.device))
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1))
        return relative_buckets + torch.where(is_small, relative_position, relative_position_if_large)

    def compute_bias(self, query_length: int, key_length: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        buckets = self._relative_position_bucket(relative_position)
        values = self.relative_attention_bias(buckets, out_dtype=dtype)
        return values.permute(2, 0, 1).unsqueeze(0).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        past_bias: torch.Tensor | None = None,
        optimized_attention=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        if self.relative_attention_bias is not None:
            past_bias = self.compute_bias(x.shape[1], x.shape[1], device=x.device, dtype=x.dtype)

        if past_bias is not None:
            mask = mask + past_bias if mask is not None else past_bias

        out = optimized_attention(q, k * ((k.shape[-1] / self.num_heads) ** 0.5), v, self.num_heads, mask)
        return self.o(out), past_bias


class T5LayerSelfAttention(torch.nn.Module):
    def __init__(self, model_dim: int, inner_dim: int, num_heads: int, relative_attention_bias: bool, *, dtype=None, device=None, operations=base_ops.manual_cast):
        super().__init__()
        self.SelfAttention = T5Attention(model_dim, inner_dim, num_heads, relative_attention_bias, dtype=dtype, device=device, operations=operations)
        self.layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device)

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        past_bias: torch.Tensor | None = None,
        optimized_attention=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        output, past_bias = self.SelfAttention(self.layer_norm(x), mask=mask, past_bias=past_bias, optimized_attention=optimized_attention)
        return x + output, past_bias


class T5Block(torch.nn.Module):
    def __init__(
        self,
        model_dim: int,
        inner_dim: int,
        ff_dim: int,
        ff_activation: str,
        gated_act: bool,
        num_heads: int,
        relative_attention_bias: bool,
        *,
        dtype=None,
        device=None,
        operations=base_ops.manual_cast,
    ):
        super().__init__()
        self.layer = torch.nn.ModuleList(
            [
                T5LayerSelfAttention(model_dim, inner_dim, num_heads, relative_attention_bias, dtype=dtype, device=device, operations=operations),
                T5LayerFF(model_dim, ff_dim, ff_activation, gated_act, dtype=dtype, device=device, operations=operations),
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        past_bias: torch.Tensor | None = None,
        optimized_attention=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x, past_bias = self.layer[0](x, mask=mask, past_bias=past_bias, optimized_attention=optimized_attention)
        return self.layer[1](x), past_bias


class T5Stack(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        model_dim: int,
        inner_dim: int,
        ff_dim: int,
        ff_activation: str,
        gated_act: bool,
        num_heads: int,
        relative_attention: bool,
        *,
        dtype=None,
        device=None,
        operations=base_ops.manual_cast,
    ):
        super().__init__()
        self.block = torch.nn.ModuleList(
            [
                T5Block(
                    model_dim,
                    inner_dim,
                    ff_dim,
                    ff_activation,
                    gated_act,
                    num_heads,
                    relative_attention_bias=((not relative_attention) or (index == 0)),
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
                for index in range(num_layers)
            ]
        )
        self.final_layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor, *, attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])).expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )
            mask = mask.masked_fill(mask.to(torch.bool), -torch.finfo(x.dtype).max)

        optimized_attention = optimized_attention_for_device(x.device, mask=attention_mask is not None, small_input=True)
        past_bias = None
        use_gc = getattr(self, "_t5_lazy_runtime", False)
        adaptive_gc = bool(getattr(self, "_t5_lazy_gc_adaptive", True))
        gc_interval = _coerce_positive_int(
            getattr(self, "_t5_lazy_gc_interval", _T5_LAZY_GC_DEFAULT_INTERVAL),
            _T5_LAZY_GC_DEFAULT_INTERVAL,
        )
        healthy_gc_interval = _coerce_positive_int(
            getattr(self, "_t5_lazy_gc_healthy_interval", gc_interval),
            gc_interval,
        )
        recheck_blocks = _coerce_positive_int(
            getattr(self, "_t5_lazy_gc_recheck_blocks", _T5_LAZY_GC_RECHECK_BLOCKS),
            _T5_LAZY_GC_RECHECK_BLOCKS,
        )
        low_headroom_mb = _normalize_headroom_mb(getattr(self, "_t5_lazy_gc_low_headroom_mb", None))
        critical_headroom_mb = _normalize_headroom_mb(getattr(self, "_t5_lazy_gc_critical_headroom_mb", None))
        total_blocks = len(self.block)

        for index, block in enumerate(self.block, start=1):
            x, past_bias = block(x, mask=mask, past_bias=past_bias, optimized_attention=optimized_attention)

            if use_gc and adaptive_gc and gc_interval > 1 and index < total_blocks and index % recheck_blocks == 0:
                try:
                    snapshot = resources.capture_memory_snapshot(
                        notes={
                            "tag": "disk_paged_t5_gc_recheck",
                            "block_index": index,
                            "total_blocks": total_blocks,
                            "gc_interval": gc_interval,
                        }
                    )
                    free_ram_mb = _normalize_headroom_mb(getattr(snapshot, "free_ram_mb", None))
                    next_interval = _resolve_disk_paged_t5_gc_interval(
                        free_ram_mb=free_ram_mb,
                        low_headroom_mb=low_headroom_mb,
                        critical_headroom_mb=critical_headroom_mb,
                        healthy_interval=healthy_gc_interval,
                    )
                    if next_interval < gc_interval:
                        gc_interval = next_interval
                        if gc_interval <= 1:
                            recheck_blocks = 1
                except Exception:
                    pass

            if use_gc and ((index % gc_interval) == 0 or index == total_blocks):
                gc.collect()
        return self.final_layer_norm(x), None


class T5(torch.nn.Module):
    def __init__(self, config_dict: dict[str, Any], dtype, device, operations) -> None:
        super().__init__()
        model_dim = config_dict["d_model"]
        inner_dim = config_dict["d_kv"] * config_dict["num_heads"]
        self.encoder = T5Stack(
            config_dict["num_layers"],
            model_dim,
            inner_dim,
            config_dict["d_ff"],
            config_dict["dense_act_fn"],
            config_dict["is_gated_act"],
            config_dict["num_heads"],
            config_dict["model_type"] != "umt5",
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.shared = operations.Embedding(config_dict["vocab_size"], model_dim, device=device, dtype=dtype)
        self.dtype = dtype

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, embeddings) -> None:
        self.shared = embeddings

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, embeds: torch.Tensor | None = None, **kwargs):
        if input_ids is None:
            x = embeds
        else:
            x = self.shared(input_ids, out_dtype=kwargs.get("dtype", torch.float32))
        if self.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
            x = torch.nan_to_num(x)
        return self.encoder(x, attention_mask=attention_mask)


class T5XXLTextEncoder(torch.nn.Module, sd1_clip.ClipTokenWeightEncoder):
    def __init__(self, *, device="cpu", dtype=None, model_options: dict[str, Any] | None = None) -> None:
        super().__init__()
        config_path = _FLUX_FILL_V3_ASSET_ROOT / "t5_config_xxl.json"
        config = _load_json(config_path)
        operations = _pick_t5_ops(model_options)
        self.transformer = T5(config, dtype, device, operations)
        self.special_tokens = {"end": 1, "pad": 0}

    def set_clip_options(self, options: dict[str, Any]) -> None:
        return None

    def reset_clip_options(self) -> None:
        return None

    def encode(self, tokens):
        embedding_weight = self.transformer.get_input_embeddings().weight
        device = getattr(embedding_weight, "device", torch.device("cpu"))
        if not isinstance(device, torch.device):
            device = torch.device(device)
        input_ids = torch.LongTensor(tokens).to(device)
        attention_mask = (input_ids != self.special_tokens["pad"]).long()
        encoded, _ = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return encoded.float(), None

    def load_sd(self, sd):
        return self.transformer.load_state_dict(sd, strict=False)


class FluxClipModel(torch.nn.Module):
    def __init__(self, *, dtype_t5=None, device="cpu", dtype=None, model_options: dict[str, Any] | None = None):
        super().__init__()
        model_options = model_options or {}
        dtype_t5 = precision.pick_weight_dtype(dtype_t5, dtype, device)
        self.clip_l = sd1_clip.SDClipModel(device=device, dtype=dtype)
        self.t5xxl = T5XXLTextEncoder(device=device, dtype=dtype_t5, model_options=model_options)
        self.dtypes = {dtype, dtype_t5}

    def set_clip_options(self, options: dict[str, Any]) -> None:
        return None

    def reset_clip_options(self) -> None:
        return None

    def clip_layer(self, layer_idx: int) -> None:
        self.clip_l.clip_layer(layer_idx)

    def reset_clip_layer(self) -> None:
        self.clip_l.reset_clip_layer()

    def encode_token_weights(self, token_weight_pairs):
        t5_out, _ = self.t5xxl.encode_token_weights(token_weight_pairs["t5xxl"])
        _, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs["l"])
        return t5_out, l_pooled

    def load_sd(self, sd):
        if _CLIP_L_KEY in sd:
            return self.clip_l.load_sd(sd)
        return self.t5xxl.load_sd(sd)


class FluxTokenizer:
    def __init__(self, embedding_directory=None):
        self.clip_l = sd1_clip.SDTokenizer(embedding_directory=embedding_directory)
        self.t5xxl = FixedLengthT5Tokenizer()

    def tokenize_with_weights(self, text: str, return_word_ids: bool = False, **kwargs):
        return {
            "l": self.clip_l.tokenize_with_weights(text, return_word_ids),
            "t5xxl": self.t5xxl.tokenize_with_weights(text, return_word_ids, **kwargs),
        }

    def state_dict(self) -> dict[str, Any]:
        return {}


@dataclass
class FluxPromptTextEncoder:
    cond_stage_model: FluxClipModel
    tokenizer: FluxTokenizer
    patcher: Any

    def tokenize(self, text: str):
        return self.tokenizer.tokenize_with_weights(text)

    def encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenize(text)
        load_device = getattr(self.patcher, "load_device", None)
        if getattr(load_device, "type", None) != "cpu":
            resources.load_models_gpu([self.patcher], force_full_load=True)
        try:
            with torch.inference_mode():
                cond, pooled = self.cond_stage_model.encode_token_weights(tokens)
            return cond, pooled
        finally:
            try:
                resources.eject_model(self.patcher)
            except Exception:
                detach = getattr(self.patcher, "detach", None)
                if callable(detach):
                    detach()


def _load_text_encoder_state_dict(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.suffix.lower() == ".gguf":
        raise ValueError(
            "GGUF text-encoder checkpoints are not supported by flux_fill_v3. "
            "Use the native safetensors Flux Fill assets."
        )
    return comfy_utils.load_torch_file(str(path), safe_load=True), {}


def _normalize_checkpoint_dtype(value: Any) -> torch.dtype | None:
    dtype = getattr(value, "dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None and isinstance(value, torch.dtype):
        return value

    dtype_text = str(dtype if dtype is not None else value).strip()
    mapping = {
        "F16": torch.float16,
        "F32": torch.float32,
        "BF16": torch.bfloat16,
        "F8_E4M3": getattr(torch, "float8_e4m3fn", None),
        "F8_E4M3FN": getattr(torch, "float8_e4m3fn", None),
        "F8_E5M2": getattr(torch, "float8_e5m2", None),
        "torch.float16": torch.float16,
        "torch.float32": torch.float32,
        "torch.bfloat16": torch.bfloat16,
        "float8_e4m3fn": getattr(torch, "float8_e4m3fn", None),
        "float8_e5m2": getattr(torch, "float8_e5m2", None),
        "torch.float8_e4m3fn": getattr(torch, "float8_e4m3fn", None),
        "torch.float8_e5m2": getattr(torch, "float8_e5m2", None),
    }
    return mapping.get(dtype_text)


def _detect_t5_dtype(state_dict: dict[str, Any]) -> torch.dtype | None:
    for key in ("encoder.final_layer_norm.weight", _T5_KEY, _T5_KEY_OLD):
        tensor = state_dict.get(key)
        if tensor is not None:
            detected = _normalize_checkpoint_dtype(tensor)
            if detected is not None:
                return detected
    return None


def _map_t5_source_key(source_key: str) -> str:
    if source_key == "encoder.embed_tokens.weight":
        return "shared.weight"
    return source_key


class LazySafetensorsLayer(torch.nn.Module):
    comfy_cast_weights = True

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        matched = False
        for key, value in state_dict.items():
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix):]
            if suffix == "weight":
                self.weight = value
                matched = True
            elif suffix == "bias":
                self.bias = value
                matched = True
            else:
                unexpected_keys.append(key)
        if not matched:
            missing_keys.append(prefix + "weight")

    @staticmethod
    def _materialize(value: Any, *, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor | None:
        if value is None:
            return None
        if hasattr(value, "load") and callable(value.load):
            value = value.load()
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected tensor-like lazy weight, got {type(value).__name__}.")
        if dtype is None:
            return value.to(device=device)
        return value.to(device=device, dtype=dtype)


class LazySafetensorsOps(base_ops.manual_cast):
    class Linear(LazySafetensorsLayer, base_ops.manual_cast.Linear):
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None

        def forward(self, input):
            weight = self._materialize(self.weight, device=input.device, dtype=input.dtype)
            bias = self._materialize(self.bias, device=input.device, dtype=input.dtype) if self.bias is not None else None
            return torch.nn.functional.linear(input, weight, bias)

    class Embedding(LazySafetensorsLayer, base_ops.manual_cast.Embedding):
        def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.padding_idx = padding_idx
            self.max_norm = max_norm
            self.norm_type = norm_type
            self.scale_grad_by_freq = scale_grad_by_freq
            self.sparse = sparse
            self.weight = None
            self.bias = None

        def forward(self, input, out_dtype=None):
            weight_dtype = out_dtype if out_dtype is not None else None
            weight = self._materialize(self.weight, device=input.device, dtype=weight_dtype)
            out = torch.nn.functional.embedding(
                input,
                weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
            if out_dtype is not None:
                out = out.to(dtype=out_dtype)
            return out


def _build_lazy_t5_state_dict(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from backend.cpu_compiler import SafeOpenHeaderOnly

    header = SafeOpenHeaderOnly(str(path))
    lazy_state_dict: dict[str, Any] = {}
    duplicate_source_keys: list[str] = []
    for source_key, value in header.items():
        target_key = _map_t5_source_key(source_key)
        if target_key in lazy_state_dict:
            duplicate_source_keys.append(source_key)
            continue
        lazy_state_dict[target_key] = value
    return lazy_state_dict, {
        "custom_operations": LazySafetensorsOps,
        "lazy_safetensors_runtime": True,
        "lazy_duplicate_source_keys": duplicate_source_keys,
    }


def load_flux_prompt_text_encoder(
    *,
    clip_l_path: str | Path,
    t5_path: str | Path,
    embedding_directory: str | Path | None = None,
    t5_loader_policy: str | None = None,
    low_ram_gc: bool = False,
    disk_paged_t5_gc_interval: int | None = None,
) -> FluxPromptTextEncoder:
    clip_l_path = Path(clip_l_path)
    t5_path = Path(t5_path)
    clip_l_sd, clip_l_options = _load_text_encoder_state_dict(clip_l_path)
    t5_loader_policy = _normalize_t5_loader_policy(t5_loader_policy, t5_path=t5_path)

    if t5_loader_policy != "stream_safetensors_runtime":
        raise ValueError(
            f"Unsupported T5 loader policy: {t5_loader_policy!r}. Only 'stream_safetensors_runtime' is accepted."
        )

    if t5_path.suffix.lower() != ".safetensors":
        raise ValueError("stream_safetensors_runtime requires a .safetensors T5 checkpoint.")

    t5_sd, t5_options = _build_lazy_t5_state_dict(t5_path)
    detected_t5_dtype = _detect_t5_dtype(t5_sd)

    load_device = torch.device("cpu")
    offload_device = torch.device("cpu")
    dtype = precision.text_encoder_dtype(load_device)
    model_options = {}
    model_options.update(clip_l_options)
    model_options.update(t5_options)
    initial_device = torch.device("cpu")
    model_options["initial_device"] = initial_device

    cond_stage_model = FluxClipModel(
        dtype_t5=detected_t5_dtype,
        device=initial_device,
        dtype=dtype,
        model_options=model_options,
    )
    tokenizer = FluxTokenizer(embedding_directory=embedding_directory)
    patcher = patching.NexModelPatcher(cond_stage_model, load_device=load_device, offload_device=offload_device)

    missing, unexpected = cond_stage_model.load_sd(clip_l_sd)
    if missing:
        logger.debug("Flux CLIP-L missing keys: %s", missing)
    if unexpected:
        logger.debug("Flux CLIP-L unexpected keys: %s", unexpected)

    missing, unexpected = cond_stage_model.load_sd(t5_sd)
    if missing:
        logger.debug("Flux T5 missing keys: %s", missing)
    if unexpected:
        logger.debug("Flux T5 unexpected keys: %s", unexpected)

    encoder = FluxPromptTextEncoder(cond_stage_model=cond_stage_model, tokenizer=tokenizer, patcher=patcher)
    gc_config: dict[str, Any] = {}
    if low_ram_gc:
        gc_config = _resolve_disk_paged_t5_gc_config(override_interval=disk_paged_t5_gc_interval)
        t5_encoder = cond_stage_model.t5xxl.transformer.encoder
        setattr(t5_encoder, "_t5_lazy_runtime", True)
        setattr(t5_encoder, "_t5_lazy_gc_adaptive", bool(gc_config.get("adaptive", True)))
        setattr(t5_encoder, "_t5_lazy_gc_interval", gc_config["interval"])
        setattr(t5_encoder, "_t5_lazy_gc_healthy_interval", gc_config.get("healthy_interval", gc_config["interval"]))
        setattr(t5_encoder, "_t5_lazy_gc_recheck_blocks", gc_config["recheck_blocks"])
        setattr(t5_encoder, "_t5_lazy_gc_low_headroom_mb", gc_config["low_headroom_mb"])
        setattr(t5_encoder, "_t5_lazy_gc_critical_headroom_mb", gc_config["critical_headroom_mb"])
        logger.debug(
            "[Flux Telemetry] Configured disk-paged T5 GC cadence profile=%s free_ram_mb=%s adaptive=%s interval=%s healthy_interval=%s recheck_blocks=%s override_interval=%s low_headroom_mb=%s critical_headroom_mb=%s",
            gc_config.get("profile_name") or "unknown",
            gc_config.get("initial_free_ram_mb"),
            gc_config.get("adaptive"),
            gc_config["interval"],
            gc_config.get("healthy_interval"),
            gc_config["recheck_blocks"],
            gc_config.get("override_interval"),
            gc_config["low_headroom_mb"],
            gc_config["critical_headroom_mb"],
        )
    setattr(
        encoder,
        "_nex_load_metadata",
        {
            "clip_l_path": str(clip_l_path),
            "t5_path": str(t5_path),
            "t5_loader_policy": t5_loader_policy,
            "t5_detected_dtype": str(detected_t5_dtype) if detected_t5_dtype is not None else None,
            "t5_source_kind": "safetensors_lazy_runtime",
            "t5_full_state_dict_materialized": False,
            "t5_stream_runtime": True,
            "t5_lazy_runtime": bool(low_ram_gc),
            "t5_lazy_gc_adaptive": gc_config.get("adaptive"),
            "t5_lazy_gc_interval": gc_config.get("interval"),
            "t5_lazy_gc_healthy_interval": gc_config.get("healthy_interval"),
            "t5_lazy_gc_recheck_blocks": gc_config.get("recheck_blocks"),
            "t5_lazy_gc_low_headroom_mb": gc_config.get("low_headroom_mb"),
            "t5_lazy_gc_critical_headroom_mb": gc_config.get("critical_headroom_mb"),
            "t5_lazy_gc_initial_free_ram_mb": gc_config.get("initial_free_ram_mb"),
            "t5_lazy_gc_profile_name": gc_config.get("profile_name"),
            "t5_lazy_gc_override_interval": gc_config.get("override_interval"),
            "t5_lazy_duplicate_source_keys": list(t5_options.get("lazy_duplicate_source_keys", [])),
        },
    )
    return encoder


def get_prompt_cache_path(prompt: str, clip_l_path: Path | str, t5_path: Path | str, cache_mode: str | None = "temp") -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", prompt.strip().lower()).strip("_")
    slug = (slug or "prompt")[:48]
    payload = "\n".join([prompt.strip(), str(clip_l_path), str(t5_path)]).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    filename = f"{slug}_{digest}.pt"
    from modules import config
    if str(cache_mode).strip().lower() == "permanent":
        roots = getattr(config, "paths_clips", None)
        if isinstance(roots, (list, tuple)) and len(roots) > 0:
            clip_root = Path(roots[0])
        else:
            root = getattr(config, "path_clip", None)
            clip_root = Path(root) if root else Path("models") / "clip"
        dest_dir = clip_root / "flux" / "generated_conditioning"
    else:
        dest_dir = Path(config.path_temp_outputs) / "flux_conditioning"
    return dest_dir / filename


def save_flux_prompt_conditioning_cache(
    path: str | Path,
    *,
    cross_attn: torch.Tensor,
    pooled_output: torch.Tensor,
    metadata: dict[str, Any] | None = None,
) -> FluxEmptyConditioning:
    payload = {
        "cross_attn": cross_attn.detach().cpu(),
        "pooled_output": pooled_output.detach().cpu(),
        "metadata": dict(metadata or {}),
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(output_path))
    return FluxEmptyConditioning(
        cross_attn=payload["cross_attn"],
        pooled_output=payload["pooled_output"],
        metadata=payload["metadata"],
    )


def _resolve_request_conditioning_cache_path(request: FluxFillRequest) -> Path:
    explicit_path = str(getattr(request, "conditioning_cache_path", "") or "").strip()
    if explicit_path:
        return Path(explicit_path)

    prompt_text = str(getattr(request, "prompt", "") or "").strip()
    clip_l_path = getattr(request, "clip_l_path", None)
    t5_path = getattr(request, "t5_path", None)
    if prompt_text and clip_l_path and t5_path:
        return get_prompt_cache_path(
            prompt_text,
            clip_l_path=clip_l_path,
            t5_path=t5_path,
            cache_mode="temp",
        )

    return Path(explicit_path)


def _flux_prompt_conditioning_generator_script() -> Path:
    return Path(__file__).resolve().with_name("prompt_conditioning_artifact_worker.py")


def _flux_prompt_conditioning_metrics_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".metrics.json")


def _extract_json_payload_from_process_output(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in str(stdout or "").splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("Artifact generator did not emit a JSON object on stdout.")


def _summarize_process_output(text: str, *, limit: int = 400) -> str:
    cleaned = " ".join(str(text or "").splitlines()).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _require_disk_paged_t5_checkpoint(t5_path: str | Path) -> Path:
    resolved = Path(t5_path)
    if resolved.suffix.lower() != ".safetensors":
        logger.warning(
            "[Flux Telemetry] Rejected non-safetensors T5 path path=%s suffix=%s",
            resolved,
            resolved.suffix,
        )
        raise ValueError(
            "Flux disk-paged T5 artifact generation requires a .safetensors T5 checkpoint."
        )
    return resolved


def generate_flux_prompt_conditioning_artifact(
    *,
    prompt_text: str,
    clip_l_path: str | Path,
    t5_path: str | Path,
    cache_path: str | Path,
    disk_paged_t5_gc_interval: int | None = None,
) -> dict[str, Any]:
    script_path = _flux_prompt_conditioning_generator_script()
    if not script_path.exists():
        raise FileNotFoundError(f"Flux prompt-conditioning generator script not found: {script_path}")

    cache_path = Path(cache_path)
    t5_path = _require_disk_paged_t5_checkpoint(t5_path)
    metrics_path = _flux_prompt_conditioning_metrics_path(cache_path)
    repo_root = script_path.parents[2]
    command = [
        sys.executable,
        str(script_path),
        "--prompt",
        str(prompt_text),
        "--output",
        str(cache_path),
        "--clip-l",
        str(clip_l_path),
        "--fp16-t5",
        str(t5_path),
        "--metrics-json",
        str(metrics_path),
    ]
    if disk_paged_t5_gc_interval is not None:
        command.extend(
            [
                "--disk-paged-t5-gc-interval",
                str(int(disk_paged_t5_gc_interval)),
            ]
        )

    logger.debug(
        "[Flux Telemetry] Launching isolated prompt-conditioning generator posture=disk_paged_t5 policy=stream_safetensors_runtime low_ram_gc=True gc_interval_override=%s cache_path=%s metrics_path=%s %s",
        disk_paged_t5_gc_interval,
        cache_path,
        metrics_path,
        format_flux_conditioning_memory_summary(tag="conditioning_generator_launch"),
    )

    gc.collect()
    try:
        resources.soft_empty_cache(force=True)
    except Exception:
        pass

    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    wall = time.perf_counter() - start

    try:
        payload = _extract_json_payload_from_process_output(completed.stdout)
    except Exception:
        payload = {}

    if completed.returncode != 0 or payload.get("status") != "ok":
        stdout_summary = _summarize_process_output(completed.stdout)
        stderr_summary = _summarize_process_output(completed.stderr)
        error_message = None
        if isinstance(payload.get("error"), dict):
            error_message = payload["error"].get("message")
        raise RuntimeError(
            "Flux prompt-conditioning artifact generation failed "
            f"(posture=disk_paged_t5, policy=stream_safetensors_runtime, exit_code={completed.returncode}, "
            f"error={error_message!r}, stdout={stdout_summary!r}, stderr={stderr_summary!r})."
        )

    logger.debug(
        "[Flux Telemetry] Isolated prompt-conditioning generator completed in %.3fs posture=disk_paged_t5 policy=stream_safetensors_runtime low_ram_gc=True gc_interval_override=%s output_path=%s metrics_path=%s %s",
        wall,
        disk_paged_t5_gc_interval,
        payload.get("output_path") or payload.get("output") or str(cache_path),
        payload.get("metrics_path") or str(metrics_path),
        format_flux_conditioning_memory_summary(tag="conditioning_generator_complete"),
    )
    return payload


def _load_or_generate_prompt_conditioning(
    request: FluxFillRequest,
) -> FluxEmptyConditioning:
    prompt_text = str(request.prompt or "").strip()
    clip_l_path = request.clip_l_path
    t5_path = request.t5_path
    if not clip_l_path or not t5_path:
        raise ValueError("Prompt-conditioned Flux Fill requires explicit clip_l_path and t5_path in request.")
    t5_path = _require_disk_paged_t5_checkpoint(t5_path)
    gc_interval_override = getattr(request, "disk_paged_t5_gc_interval", None)

    cache_path = _resolve_request_conditioning_cache_path(request)
    logger.debug(
        "[Flux Telemetry] T5 conditioning begin posture=disk_paged_t5 prompt_len=%d low_ram_gc=True gc_interval_override=%s cache_path=%s %s %s",
        len(prompt_text),
        gc_interval_override,
        cache_path,
        _describe_cache_path(cache_path),
        format_flux_conditioning_memory_summary(tag="conditioning_begin"),
    )
    logger.debug(f"[Flux Telemetry] Checking prompt conditioning cache at: {cache_path}")
    if cache_path.exists():
        logger.debug(f"[Flux Telemetry] Prompt conditioning cache HIT for path: {cache_path}")
        try:
            return load_flux_empty_conditioning_cache(cache_path)
        except Exception:
            logger.exception(
                "[Flux Telemetry] Prompt conditioning cache reuse failed path=%s posture=disk_paged_t5 %s",
                cache_path,
                format_flux_conditioning_memory_summary(tag="conditioning_cache_reuse_failed"),
            )
            raise

    logger.debug(
        "[Flux Telemetry] Prompt conditioning cache MISS. Launching isolated artifact generator clip_l=%s t5=%s posture=disk_paged_t5 policy=stream_safetensors_runtime low_ram_gc=True gc_interval_override=%s",
        clip_l_path,
        t5_path,
        gc_interval_override,
    )
    generate_flux_prompt_conditioning_artifact(
        prompt_text=prompt_text,
        clip_l_path=clip_l_path,
        t5_path=t5_path,
        cache_path=cache_path,
        disk_paged_t5_gc_interval=gc_interval_override,
    )
    logger.debug(
        "[Flux Telemetry] Prompt conditioning artifact ready posture=disk_paged_t5 cache_path=%s %s",
        cache_path,
        format_flux_conditioning_memory_summary(tag="conditioning_artifact_ready"),
    )
    return load_flux_empty_conditioning_cache(cache_path)


class DiskPagedTextWorker:
    """Greenfield Disk Paged T5 Posture/Worker Contract."""
    def __init__(self, request: FluxFillRequest) -> None:
        self.request = request

    def get_conditioning(self) -> FluxEmptyConditioning:
        cpu_resident_released = False
        try:
            from backend.flux_fill_v3.cpu_resident_text_worker import CpuResidentTextEncoderCache
            cpu_resident_released = CpuResidentTextEncoderCache.teardown()
        except Exception:
            cpu_resident_released = False

        if cpu_resident_released:
            logger.info(
                "[Flux Telemetry] DiskPagedTextWorker request tearing down warm CPU-resident text encoder before disk-paged execution."
            )

        if not self.request.prompt or not str(self.request.prompt).strip():
            logger.debug(
                "[Flux Telemetry] Empty prompt, loading empty conditioning cache. %s",
                format_flux_conditioning_memory_summary(tag="disk_paged_empty_prompt"),
            )
            return load_flux_empty_conditioning_cache(self.request.conditioning_cache_path)

        return _load_or_generate_prompt_conditioning(self.request)
