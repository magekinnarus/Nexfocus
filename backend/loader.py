import ctypes
import json
import torch
import logging
from typing import Any, Dict
import gc
import os
import struct
from safetensors import safe_open
import torch
from .defs import sdxl as sdxl_def
from .defs import sd15 as sd15_def
from backend import resources, clip, patching, conditioning
from ldm_patched.modules import model_base, latent_formats, supported_models_base
from ldm_patched.ldm.models.autoencoder import AutoencoderKL, AutoencodingEngine
import torch.nn as nn
from . import utils

def heal_model_weights(model, name_prefix="Model"):
    """
    Checks for NaNs/Infs in model weights and heals them in-place.
    """
    # print(f"Checking and Healing {name_prefix} weights...")
    bad_params_count = 0
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            bad_params_count += 1
            logging.warning(f"CRITICAL: Bad values in {name_prefix} parameter: {name}. HEALING...")
            with torch.no_grad():
                if "weight" in name and ("layer_norm" in name or "layernorm" in name):
                     param.data.nan_to_num_(nan=1.0, posinf=1.0, neginf=-1.0)
                else:
                     param.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    
    if bad_params_count > 0:
        logging.info(f"Healed {bad_params_count} parameters in {name_prefix}.")

class EmbeddingFP32Wrapper(nn.Module):
    """
    Force embedding output to FP32 to trigger safe compute paths in ldm_patched.
    """
    def __init__(self, original_embedding):
        super().__init__()
        self.original = original_embedding
    
    def forward(self, x):
        return self.original(x).float()
    
    def __getattr__(self, name):
        if name in ["original", "forward", "__init__", "__class__", "__dir__"]:
             return super().__getattr__(name) 
        return getattr(self.original, name)

def resolve_source(source, device=None):
    """
    Ensures the source is a state dict. If it's a path, loads it.
    """
    if isinstance(source, str):
        return utils.load_torch_file(source, device=device)
    return source


def _log_sdxl_assembly_telemetry(event: str, extra_msg: str = "") -> None:
    try:
        from backend.sdxl_assembly.progress import log_telemetry

        log_telemetry(event, extra_msg)
    except Exception:
        pass


def _safe_open_device_arg(device):
    if device is None:
        return "cpu"
    if not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type == "cpu":
        return "cpu"
    if device.type == "cuda":
        return 0 if device.index is None else int(device.index)
    return "cpu"


def _strip_checkpoint_prefix(key, prefix):
    new_key = key[len(prefix):]
    if new_key.startswith("."):
        new_key = new_key[1:]
    return new_key


def _extract_prefixed_safetensors_state_dict(ckpt_path, prefixes, *, device=None):
    extracted = {}
    with safe_open(ckpt_path, framework="pt", device=_safe_open_device_arg(device)) as handle:
        for key in handle.keys():
            for prefix in prefixes:
                if key.startswith(prefix):
                    extracted[_strip_checkpoint_prefix(key, prefix)] = handle.get_tensor(key)
                    break
    return extracted


def _load_prefixed_safetensors_into_module(
    ckpt_path,
    prefixes,
    module,
    *,
    device=None,
    dtype=None,
    chunk_bytes=None,
    realize_cpu_targets=False,
    realize_pinned_targets=False,
    load_metrics=None,
    raw_byte_stream=False,
):
    target_device = torch.device(device) if device is not None else None
    state_entries = module.state_dict()
    state_owners = _build_module_state_owner_index(module)
    loaded_keys = set()
    unexpected_keys = []
    realized_pinned_bytes = 0
    realized_pinned_tensor_count = 0
    realized_cpu_bytes = 0
    realized_cpu_tensor_count = 0
    fallback_raw_keys = []

    if raw_byte_stream:
        with _open_safetensors_sequential_reader(ckpt_path) as reader:
            header, data_base_offset = _read_safetensors_header(reader)
            for key, entry in header.items():
                if key == "__metadata__":
                    continue
                matched_prefix = _match_safetensors_prefix(key, prefixes)
                if matched_prefix is None:
                    continue

                target_key = _strip_checkpoint_prefix(key, matched_prefix)
                target_tensor, realized_bytes = _resolve_streaming_target_tensor(
                    state_entries,
                    state_owners,
                    target_key,
                    target_dtype=dtype,
                    realize_cpu_targets=realize_cpu_targets,
                    realize_pinned_targets=realize_pinned_targets,
                )
                if target_tensor is None:
                    unexpected_keys.append(target_key)
                    continue

                if realized_bytes > 0:
                    if bool(realize_pinned_targets and torch.cuda.is_available()):
                        realized_pinned_bytes += int(realized_bytes)
                        realized_pinned_tensor_count += 1
                    else:
                        realized_cpu_bytes += int(realized_bytes)
                        realized_cpu_tensor_count += 1

                if _stream_raw_safetensors_entry_into_target(
                    reader,
                    data_base_offset,
                    entry,
                    target_tensor,
                    dtype=dtype,
                    chunk_bytes=chunk_bytes,
                ):
                    loaded_keys.add(target_key)
                    continue

                fallback_raw_keys.append((key, target_key, target_tensor))
    else:
        with safe_open(ckpt_path, framework="pt", device=_safe_open_device_arg(target_device)) as handle:
            for key in handle.keys():
                matched_prefix = _match_safetensors_prefix(key, prefixes)
                if matched_prefix is None:
                    continue

                target_key = _strip_checkpoint_prefix(key, matched_prefix)
                target_tensor, realized_bytes = _resolve_streaming_target_tensor(
                    state_entries,
                    state_owners,
                    target_key,
                    target_dtype=dtype,
                    realize_cpu_targets=realize_cpu_targets,
                    realize_pinned_targets=realize_pinned_targets,
                )
                if target_tensor is None:
                    unexpected_keys.append(target_key)
                    continue

                if realized_bytes > 0:
                    if bool(realize_pinned_targets and torch.cuda.is_available()):
                        realized_pinned_bytes += int(realized_bytes)
                        realized_pinned_tensor_count += 1
                    else:
                        realized_cpu_bytes += int(realized_bytes)
                        realized_cpu_tensor_count += 1

                _stream_safetensors_key_into_target(
                    handle,
                    key,
                    target_tensor,
                    dtype=dtype,
                    chunk_bytes=chunk_bytes,
                )
                loaded_keys.add(target_key)

    if fallback_raw_keys:
        with safe_open(ckpt_path, framework="pt", device=_safe_open_device_arg(target_device)) as handle:
            for key, target_key, target_tensor in fallback_raw_keys:
                _stream_safetensors_key_into_target(
                    handle,
                    key,
                    target_tensor,
                    dtype=dtype,
                    chunk_bytes=chunk_bytes,
                )
                loaded_keys.add(target_key)

    missing_keys = [key for key in state_entries.keys() if key not in loaded_keys]
    if isinstance(load_metrics, dict):
        load_metrics["realized_pinned_bytes"] = int(realized_pinned_bytes)
        load_metrics["realized_pinned_tensor_count"] = int(realized_pinned_tensor_count)
        if realized_cpu_bytes > 0 or realized_cpu_tensor_count > 0:
            load_metrics["realized_cpu_bytes"] = int(realized_cpu_bytes)
            load_metrics["realized_cpu_tensor_count"] = int(realized_cpu_tensor_count)
    return missing_keys, unexpected_keys


def _normalize_prefixes(prefixes):
    if prefixes is None:
        return None
    if isinstance(prefixes, (list, tuple, set)):
        return [str(prefix) for prefix in prefixes]
    return [str(prefixes)]


def _build_prefixed_safetensors_key_map(handle, prefixes=None):
    key_map = {}
    prefix_candidates = _normalize_prefixes(prefixes) or [""]
    for key in handle.keys():
        for prefix in prefix_candidates:
            if key.startswith(prefix):
                key_map[_strip_checkpoint_prefix(key, prefix)] = key
                break
    return key_map


def _match_safetensors_prefix(key, prefixes):
    for prefix in prefixes:
        if key.startswith(prefix):
            return prefix
    return None


def _open_safetensors_sequential_reader(path):
    flags = int(os.O_RDONLY)
    flags |= int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_SEQUENTIAL", 0))
    fd = os.open(path, flags)
    return os.fdopen(fd, "rb", buffering=0)


def _read_safetensors_header(reader):
    header_len_bytes = reader.read(8)
    if len(header_len_bytes) != 8:
        raise EOFError("Could not read safetensors header length.")
    header_len = int(struct.unpack("<Q", header_len_bytes)[0])
    header_bytes = reader.read(header_len)
    if len(header_bytes) != header_len:
        raise EOFError("Could not read full safetensors header payload.")
    header = json.loads(header_bytes)
    if not isinstance(header, dict):
        raise ValueError("Safetensors header payload must decode to a dictionary.")
    return header, 8 + header_len


def _torch_dtype_from_safetensors(dtype_value):
    mapping = {
        "BOOL": torch.bool,
        "U8": torch.uint8,
        "I8": torch.int8,
        "I16": torch.int16,
        "U16": getattr(torch, "uint16", None),
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "I32": torch.int32,
        "U32": getattr(torch, "uint32", None),
        "F32": torch.float32,
        "I64": torch.int64,
        "U64": getattr(torch, "uint64", None),
        "F64": torch.float64,
        "F8_E4M3": getattr(torch, "float8_e4m3fn", None),
        "F8_E4M3FN": getattr(torch, "float8_e4m3fn", None),
        "F8_E5M2": getattr(torch, "float8_e5m2", None),
    }
    return mapping.get(str(dtype_value).strip().upper())


def _stream_raw_safetensors_entry_into_target(
    reader,
    data_base_offset,
    entry,
    target_tensor,
    *,
    dtype=None,
    chunk_bytes=None,
):
    if not isinstance(target_tensor, torch.Tensor):
        return False
    if target_tensor.device.type != "cpu" or not target_tensor.is_contiguous():
        return False
    if dtype is not None and target_tensor.dtype != dtype:
        return False

    entry_dtype = _torch_dtype_from_safetensors(entry.get("dtype"))
    if entry_dtype is None or entry_dtype != target_tensor.dtype:
        return False

    entry_shape = [int(dim) for dim in entry.get("shape", [])]
    if list(target_tensor.shape) != entry_shape:
        return False

    offsets = entry.get("data_offsets")
    if not isinstance(offsets, (list, tuple)) or len(offsets) != 2:
        return False
    start_offset = int(offsets[0])
    end_offset = int(offsets[1])
    total_bytes = int(end_offset - start_offset)
    expected_bytes = int(target_tensor.numel() * target_tensor.element_size())
    if total_bytes != expected_bytes:
        return False

    chunk_limit = int(chunk_bytes) if chunk_bytes is not None else 0
    chunk_size = max(1, min(total_bytes, chunk_limit if chunk_limit > 0 else total_bytes))
    reader.seek(int(data_base_offset) + start_offset)
    target_ptr = int(target_tensor.data_ptr())
    copied_bytes = 0

    while copied_bytes < total_bytes:
        this_chunk = min(chunk_size, total_bytes - copied_bytes)
        remaining = this_chunk
        chunk_offset = 0
        while remaining > 0:
            dest = (ctypes.c_char * remaining).from_address(target_ptr + copied_bytes + chunk_offset)
            read_bytes = reader.readinto(dest)
            if read_bytes is None:
                read_bytes = 0
            if read_bytes <= 0:
                raise EOFError(
                    f"Unexpected EOF while streaming safetensors payload at byte {copied_bytes + chunk_offset}."
                )
            remaining -= int(read_bytes)
            chunk_offset += int(read_bytes)
        copied_bytes += this_chunk

    return True


def _build_module_state_owner_index(module):
    owners = {}
    for module_name, submodule in module.named_modules():
        key_prefix = f"{module_name}." if module_name else ""
        for param_name, _ in submodule.named_parameters(recurse=False):
            owners[key_prefix + param_name] = (submodule, "param", param_name)
        for buffer_name, _ in submodule.named_buffers(recurse=False):
            owners[key_prefix + buffer_name] = (submodule, "buffer", buffer_name)
    return owners


def _resolve_streaming_target_tensor(
    state_entries,
    state_owners,
    target_key,
    *,
    target_dtype=None,
    realize_cpu_targets=False,
    realize_pinned_targets=False,
):
    fallback_tensor = state_entries.get(target_key)
    owner = state_owners.get(target_key)
    if owner is None:
        return fallback_tensor, 0

    submodule, tensor_kind, tensor_name = owner
    if tensor_kind == "param":
        current_tensor = submodule._parameters.get(tensor_name)
    else:
        current_tensor = submodule._buffers.get(tensor_name)
    if current_tensor is None:
        return fallback_tensor, 0

    live_tensor = current_tensor.data if tensor_kind == "param" else current_tensor
    if not realize_pinned_targets and not realize_cpu_targets:
        return live_tensor, 0

    device_type = getattr(getattr(live_tensor, "device", None), "type", None)
    if device_type not in {"cpu", "meta"}:
        return live_tensor, 0
    if device_type == "cpu" and live_tensor.is_pinned():
        return live_tensor, 0

    should_pin = bool(realize_pinned_targets and torch.cuda.is_available())
    if device_type == "cpu" and not should_pin:
        return live_tensor, 0

    empty_kwargs = {"device": "cpu"}
    if target_dtype is not None and torch.is_floating_point(live_tensor):
        empty_kwargs["dtype"] = target_dtype
    if should_pin:
        empty_kwargs["pin_memory"] = True
    realized_target = torch.empty_like(live_tensor, **empty_kwargs)
    realized_bytes = int(realized_target.numel() * realized_target.element_size())
    if tensor_kind == "param":
        submodule._parameters[tensor_name] = torch.nn.Parameter(
            realized_target,
            requires_grad=bool(getattr(current_tensor, "requires_grad", False)),
        )
        return submodule._parameters[tensor_name].data, realized_bytes

    submodule._buffers[tensor_name] = realized_target
    return submodule._buffers[tensor_name], realized_bytes


def _copy_tensor_into_target(target_tensor, source_tensor, *, dtype=None, transpose=False):
    target_dtype = target_tensor.dtype
    if dtype is not None and torch.is_floating_point(target_tensor):
        target_dtype = dtype
    copy_tensor = source_tensor.t() if transpose else source_tensor
    copy_device = getattr(copy_tensor, "device", None)
    copy_dtype = getattr(copy_tensor, "dtype", None)
    if copy_device != target_tensor.device or copy_dtype != target_dtype:
        copy_tensor = copy_tensor.to(device=target_tensor.device, dtype=target_dtype)
    with torch.no_grad():
        target_tensor.copy_(copy_tensor)


def _safetensors_dtype_size(dtype_value):
    dtype_text = str(dtype_value).strip().upper()
    sizes = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "F8_E4M3": 1,
        "F8_E4M3FN": 1,
        "F8_E5M2": 1,
        "I16": 2,
        "U16": 2,
        "F16": 2,
        "BF16": 2,
        "I32": 4,
        "U32": 4,
        "F32": 4,
        "I64": 8,
        "U64": 8,
        "F64": 8,
    }
    return sizes.get(dtype_text)


def _stream_safetensors_key_into_target(handle, key, target_tensor, *, dtype=None, chunk_bytes=None):
    if not isinstance(target_tensor, torch.Tensor):
        source_tensor = handle.get_tensor(key)
        _copy_tensor_into_target(target_tensor, source_tensor, dtype=dtype)
        return

    effective_chunk_bytes = int(chunk_bytes) if chunk_bytes is not None else 0
    can_chunk = (
        effective_chunk_bytes > 0
        and target_tensor.device.type == "cpu"
        and target_tensor.is_pinned()
        and target_tensor.ndim >= 1
    )
    if can_chunk:
        try:
            source_slice = handle.get_slice(key)
            source_shape = list(source_slice.get_shape())
            source_dtype_size = _safetensors_dtype_size(source_slice.get_dtype()) or target_tensor.element_size()
        except Exception:
            source_slice = None
            source_shape = []
            source_dtype_size = target_tensor.element_size()

        if source_shape:
            row_elems = 1
            for dim in source_shape[1:]:
                row_elems *= int(dim)
            row_bytes = max(1, int(row_elems) * int(source_dtype_size))
            rows_per_chunk = max(1, effective_chunk_bytes // row_bytes)
            if int(source_shape[0]) > rows_per_chunk:
                tail = (slice(None),) * (len(source_shape) - 1)
                for start in range(0, int(source_shape[0]), rows_per_chunk):
                    end = min(int(source_shape[0]), start + rows_per_chunk)
                    source_chunk = source_slice[(slice(start, end),) + tail]
                    target_chunk = target_tensor[(slice(start, end),) + tail]
                    _copy_tensor_into_target(target_chunk, source_chunk, dtype=dtype)
                return

    source_tensor = handle.get_tensor(key)
    _copy_tensor_into_target(target_tensor, source_tensor, dtype=dtype)


def _collect_safetensors_matches(handle, prefixes, *, fallback_to_all=False):
    prefix_candidates = _normalize_prefixes(prefixes) or []
    raw_keys = list(handle.keys())
    matches = []

    for key in raw_keys:
        for prefix in prefix_candidates:
            if key.startswith(prefix):
                matches.append((key, _strip_checkpoint_prefix(key, prefix)))
                break

    if matches or not fallback_to_all:
        return matches

    return [(key, key) for key in raw_keys]


def _stream_load_sdxl_clip_l_into_encoder(
    source_path,
    target_encoder,
    *,
    prefixes=None,
    dtype=None,
):
    state_entries = target_encoder.transformer.state_dict()
    loaded_keys = set()
    unexpected_keys = []

    with safe_open(source_path, framework="pt", device="cpu") as handle:
        matches = _collect_safetensors_matches(
            handle,
            prefixes or clip.CLIP_L_PREFIXES,
            fallback_to_all=prefixes is None,
        )
        for raw_key, normalized_key in matches:
            target_key = normalized_key.replace("text_model.", "")
            target_tensor = state_entries.get(target_key)
            if target_tensor is None:
                unexpected_keys.append(target_key)
                continue

            source_tensor = handle.get_tensor(raw_key)
            _copy_tensor_into_target(target_tensor, source_tensor, dtype=dtype)
            loaded_keys.add(target_key)

    missing_keys = [key for key in state_entries.keys() if key not in loaded_keys]
    return missing_keys, unexpected_keys


def _stream_load_sdxl_clip_g_into_encoder(
    source_path,
    target_encoder,
    *,
    prefixes=None,
    dtype=None,
):
    state_entries = target_encoder.transformer.state_dict()
    loaded_keys = set()
    unexpected_keys = []
    projection = getattr(target_encoder, "text_projection", None)
    logit_scale = getattr(target_encoder, "logit_scale", None)

    with safe_open(source_path, framework="pt", device="cpu") as handle:
        matches = _collect_safetensors_matches(
            handle,
            prefixes or clip.CLIP_G_PREFIXES,
            fallback_to_all=prefixes is None,
        )
        for raw_key, normalized_key in matches:
            target_key = normalized_key
            target_key = target_key.replace("transformer.resblocks.", "encoder.layers.")
            target_key = target_key.replace("ln_1.", "layer_norm1.")
            target_key = target_key.replace("ln_2.", "layer_norm2.")
            target_key = target_key.replace("mlp.c_fc.", "mlp.fc1.")
            target_key = target_key.replace("mlp.c_proj.", "mlp.fc2.")
            target_key = target_key.replace("attn.out_proj.", "self_attn.out_proj.")
            target_key = target_key.replace("token_embedding.weight", "embeddings.token_embedding.weight")
            target_key = target_key.replace("positional_embedding", "embeddings.position_embedding.weight")
            target_key = target_key.replace("ln_final.", "final_layer_norm.")
            target_key = target_key.replace("text_model.", "")

            source_tensor = handle.get_tensor(raw_key)

            if "attn.in_proj_weight" in target_key:
                base = target_key.replace("attn.in_proj_weight", "self_attn.")
                hidden_size = source_tensor.shape[0] // 3
                split_targets = (
                    (base + "q_proj.weight", source_tensor[:hidden_size]),
                    (base + "k_proj.weight", source_tensor[hidden_size:hidden_size * 2]),
                    (base + "v_proj.weight", source_tensor[hidden_size * 2:]),
                )
                for split_key, split_tensor in split_targets:
                    target_tensor = state_entries.get(split_key)
                    if target_tensor is None:
                        unexpected_keys.append(split_key)
                        continue
                    _copy_tensor_into_target(target_tensor, split_tensor, dtype=dtype)
                    loaded_keys.add(split_key)
                continue

            if "attn.in_proj_bias" in target_key:
                base = target_key.replace("attn.in_proj_bias", "self_attn.")
                hidden_size = source_tensor.shape[0] // 3
                split_targets = (
                    (base + "q_proj.bias", source_tensor[:hidden_size]),
                    (base + "k_proj.bias", source_tensor[hidden_size:hidden_size * 2]),
                    (base + "v_proj.bias", source_tensor[hidden_size * 2:]),
                )
                for split_key, split_tensor in split_targets:
                    target_tensor = state_entries.get(split_key)
                    if target_tensor is None:
                        unexpected_keys.append(split_key)
                        continue
                    _copy_tensor_into_target(target_tensor, split_tensor, dtype=dtype)
                    loaded_keys.add(split_key)
                continue

            if target_key == "text_projection" and projection is not None:
                _copy_tensor_into_target(projection.data, source_tensor, dtype=dtype)
                loaded_keys.add(target_key)
                continue

            if target_key == "text_projection.weight" and projection is not None:
                _copy_tensor_into_target(projection.data, source_tensor, dtype=dtype, transpose=True)
                loaded_keys.add(target_key)
                continue

            if target_key == "logit_scale" and logit_scale is not None:
                _copy_tensor_into_target(logit_scale.data, source_tensor, dtype=dtype)
                loaded_keys.add(target_key)
                continue

            target_tensor = state_entries.get(target_key)
            if target_tensor is None:
                unexpected_keys.append(target_key)
                continue
            _copy_tensor_into_target(target_tensor, source_tensor, dtype=dtype)
            loaded_keys.add(target_key)

    missing_keys = [key for key in state_entries.keys() if key not in loaded_keys]
    if projection is not None and "text_projection" not in loaded_keys and "text_projection.weight" not in loaded_keys:
        missing_keys.append("text_projection")
    if logit_scale is not None and "logit_scale" not in loaded_keys:
        missing_keys.append("logit_scale")
    return missing_keys, unexpected_keys


def _load_sdxl_clip_source_into_model(
    target_model,
    source,
    *,
    force_type=None,
    prefixes=None,
    dtype=None,
):
    if isinstance(source, str) and source.lower().endswith(".safetensors"):
        if force_type == "l":
            missing, unexpected = _stream_load_sdxl_clip_l_into_encoder(
                source,
                target_model.clip_l,
                prefixes=prefixes,
                dtype=dtype,
            )
            if missing:
                logging.debug("SDXL CLIP-L: Missing keys while streaming load: %s", missing)
            if unexpected:
                logging.debug("SDXL CLIP-L: Unexpected keys while streaming load: %s", unexpected)
            return
        if force_type == "g":
            missing, unexpected = _stream_load_sdxl_clip_g_into_encoder(
                source,
                target_model.clip_g,
                prefixes=prefixes,
                dtype=dtype,
            )
            if missing:
                logging.debug("SDXL CLIP-G: Missing keys while streaming load: %s", missing)
            if unexpected:
                logging.debug("SDXL CLIP-G: Unexpected keys while streaming load: %s", unexpected)
            return

    sd = resolve_source(source)
    if force_type is None:
        target_model.load_sd(sd)
    else:
        target_model.load_sd(sd, force_type=force_type)


def _inspect_safetensors_vae_metadata(source_path, *, prefixes=None):
    with safe_open(source_path, framework="pt", device="cpu") as handle:
        key_map = _build_prefixed_safetensors_key_map(handle, prefixes=prefixes)
        decoder_conv_in = None
        post_quant_conv = None
        if "decoder.conv_in.weight" in key_map:
            decoder_conv_in = handle.get_tensor(key_map["decoder.conv_in.weight"])
        if "post_quant_conv.weight" in key_map:
            post_quant_conv = handle.get_tensor(key_map["post_quant_conv.weight"])
        return {
            "key_count": len(key_map),
            "decoder_conv_in_shape": None if decoder_conv_in is None else tuple(decoder_conv_in.shape),
            "post_quant_conv_shape": None if post_quant_conv is None else tuple(post_quant_conv.shape),
            "has_downsample": "encoder.down.2.downsample.conv.weight" in key_map,
            "has_upsample": "decoder.up.3.upsample.conv.weight" in key_map,
        }


def _extract_prefixed_state_dict(source, prefixes, *, device=None):
    if isinstance(source, str) and source.lower().endswith(".safetensors"):
        return _extract_prefixed_safetensors_state_dict(source, prefixes, device=device)

    sd = resolve_source(source)
    extracted = {}
    for key, value in sd.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                extracted[_strip_checkpoint_prefix(key, prefix)] = value.to(device=device) if device is not None and hasattr(value, "to") else value
                break
    return extracted


def _module_is_meta(module):
    for tensor in list(module.parameters()) + list(module.buffers()):
        device = getattr(tensor, "device", None)
        if device is not None:
            return device.type == "meta"
    return False


def _reload_unet_weights(target_model, source, *, device, dtype=None, prefixes=None):
    diffusion_model = target_model.diffusion_model
    source_is_safetensors = isinstance(source, str) and source.lower().endswith(".safetensors")
    module_is_meta = _module_is_meta(diffusion_model)
    if module_is_meta and source_is_safetensors:
        load_prefixes = prefixes if prefixes is not None else [""]
        missing, unexpected = _load_prefixed_safetensors_into_module(
            source,
            load_prefixes,
            diffusion_model,
            device=device,
            dtype=dtype,
            realize_cpu_targets=True,
            raw_byte_stream=True,
        )
        if missing:
            logging.debug("UNet reload: Missing keys while streaming load: %s", missing)
        if unexpected:
            logging.debug("UNet reload: Unexpected keys while streaming load: %s", unexpected)
        gc.collect()
        return

    if module_is_meta and hasattr(diffusion_model, "to_empty"):
        diffusion_model.to_empty(device=device)
    if dtype is not None:
        diffusion_model.to(device=device, dtype=dtype)
    else:
        diffusion_model.to(device=device)

    if source_is_safetensors:
        load_prefixes = prefixes if prefixes is not None else [""]
        missing, unexpected = _load_prefixed_safetensors_into_module(
            source,
            load_prefixes,
            diffusion_model,
            device=device,
            dtype=dtype,
            raw_byte_stream=True,
        )
        if missing:
            logging.debug("UNet reload: Missing keys while streaming load: %s", missing)
        if unexpected:
            logging.debug("UNet reload: Unexpected keys while streaming load: %s", unexpected)
        gc.collect()
        return

    if prefixes is not None:
        sd = _extract_prefixed_state_dict(source, prefixes, device=device)
    else:
        sd = resolve_source(source, device=device)

    diffusion_model.load_state_dict(sd, strict=False)

    del sd
    gc.collect()


def _build_unet_runtime_reload(source, *, dtype=None, prefixes=None):
    if not isinstance(source, str):
        return None

    def _reload(target_model, target_device):
        _reload_unet_weights(
            target_model,
            source,
            device=target_device,
            dtype=dtype,
            prefixes=prefixes,
        )

    return _reload


def _reload_sdxl_clip_weights(
    target_model,
    source_l,
    source_g,
    *,
    device,
    dtype=None,
    prefixes_l=None,
    prefixes_g=None,
):
    if hasattr(target_model, "to"):
        if dtype is None:
            target_model.to(device=device)
        else:
            target_model.to(device=device, dtype=dtype)

    streamable_sources = (
        isinstance(source_l, str)
        and source_l.lower().endswith(".safetensors")
        and isinstance(source_g, str)
        and source_g.lower().endswith(".safetensors")
    )
    if streamable_sources:
        _load_sdxl_clip_source_into_model(
            target_model,
            source_l,
            force_type="l",
            prefixes=prefixes_l,
            dtype=dtype,
        )
        _load_sdxl_clip_source_into_model(
            target_model,
            source_g,
            force_type="g",
            prefixes=prefixes_g,
            dtype=dtype,
        )
        gc.collect()
        return

    share_source = source_l == source_g and prefixes_l == prefixes_g
    sd_l = resolve_source(source_l)
    sd_g = sd_l if share_source else resolve_source(source_g)

    try:
        if sd_l is not None:
            if share_source:
                target_model.load_sd(sd_l)
            else:
                target_model.load_sd(sd_l, force_type="l")
        if sd_g is not None and not share_source:
            target_model.load_sd(sd_g, force_type="g")
    finally:
        if not share_source:
            del sd_g
        del sd_l
        gc.collect()


def _build_sdxl_clip_runtime_reload(
    source_l,
    source_g,
    *,
    dtype=None,
    prefixes_l=None,
    prefixes_g=None,
):
    if source_l is None or source_g is None:
        return None

    def _reload(target_model, target_device):
        _reload_sdxl_clip_weights(
            target_model,
            source_l,
            source_g,
            device=target_device,
            dtype=dtype,
            prefixes_l=prefixes_l,
            prefixes_g=prefixes_g,
        )

    return _reload


def _stream_load_sdxl_unet_from_checkpoint(
    ckpt_path,
    *,
    load_device=None,
    offload_device=None,
    dtype=None,
    reload_source=None,
    reload_prefixes=None,
    stream_chunk_bytes=None,
    raw_byte_stream=True,
):
    load_device = torch.device(load_device or resources.get_torch_device())
    offload_device = torch.device(offload_device or resources.unet_offload_device())
    effective_dtype = dtype or torch.float16
    use_raw_stream = bool(raw_byte_stream and str(ckpt_path).lower().endswith(".safetensors"))
    use_meta_construction = bool(use_raw_stream and load_device.type == "cpu")

    runtime_reload = _build_unet_runtime_reload(
        reload_source if reload_source is not None else ckpt_path,
        dtype=effective_dtype,
        prefixes=reload_prefixes,
    )

    _log_sdxl_assembly_telemetry(
        "sdxl_unet_shell_construct_begin",
        f"meta_construct={use_meta_construction} dtype={effective_dtype}",
    )
    model = model_base.SDXL(
        model_config=ModelConfig(sdxl_def.UNET_CONFIG, latent_formats.SDXL()),
        device=torch.device("meta") if use_meta_construction else None,
    )
    _log_sdxl_assembly_telemetry(
        "sdxl_unet_shell_construct_complete",
        f"meta_construct={use_meta_construction}",
    )
    if not use_meta_construction:
        model.diffusion_model.to(device=load_device, dtype=effective_dtype)
        _log_sdxl_assembly_telemetry(
            "sdxl_unet_shell_to_device_complete",
            f"device={load_device} dtype={effective_dtype}",
        )

    load_metrics: dict[str, Any] = {}
    _log_sdxl_assembly_telemetry(
        "sdxl_unet_weight_stream_begin",
        f"raw_stream={use_raw_stream} meta_construct={use_meta_construction}",
    )
    missing, unexpected = _load_prefixed_safetensors_into_module(
        ckpt_path,
        reload_prefixes or sdxl_def.PREFIXES["unet"],
        model.diffusion_model,
        device=load_device,
        dtype=effective_dtype,
        chunk_bytes=stream_chunk_bytes,
        realize_cpu_targets=use_meta_construction,
        load_metrics=load_metrics,
        raw_byte_stream=use_raw_stream,
    )
    _log_sdxl_assembly_telemetry(
        "sdxl_unet_weight_stream_complete",
        (
            f"raw_stream={use_raw_stream} meta_construct={use_meta_construction} "
            f"realized_cpu_mb={int(load_metrics.get('realized_cpu_bytes', 0)) / (1024 * 1024):.1f}"
        ),
    )
    if missing:
        logging.debug("SDXL UNet: Missing keys while streaming load: %s", missing)
    if unexpected:
        logging.debug("SDXL UNet: Unexpected keys while streaming load: %s", unexpected)

    patcher = patching.NexModelPatcher(
        model,
        load_device=load_device,
        offload_device=offload_device,
        runtime_reload=runtime_reload,
        runtime_release_to_meta=runtime_reload is not None,
    )
    patcher.model_options["sdxl_assembly_loader"] = {
        "direct_safetensors_load": bool(str(ckpt_path).lower().endswith(".safetensors")),
        "raw_sequential_stream": use_raw_stream,
        "meta_construction": use_meta_construction,
        "stream_chunk_bytes": int(stream_chunk_bytes) if stream_chunk_bytes is not None else None,
        "realized_cpu_bytes": int(load_metrics.get("realized_cpu_bytes", 0)),
        "realized_cpu_tensor_count": int(load_metrics.get("realized_cpu_tensor_count", 0)),
        "realized_pinned_bytes": int(load_metrics.get("realized_pinned_bytes", 0)),
        "realized_pinned_tensor_count": int(load_metrics.get("realized_pinned_tensor_count", 0)),
    }
    return patcher

class ModelConfig(supported_models_base.BASE):
    """Mock config object for model instantiation, inheriting from BASE for compatibility."""
    def __init__(self, unet_config, latent_format):
        super().__init__(unet_config)
        self.latent_format = latent_format

class CLIP:
    """Isolated CLIP container to avoid modules.sd baggage."""
    def __init__(self, cond_stage_model, tokenizer, load_device, offload_device):
        self.cond_stage_model = cond_stage_model
        self.tokenizer = tokenizer
        self.patcher = patching.NexModelPatcher(
            self.cond_stage_model,
            load_device=load_device,
            offload_device=offload_device
        )
        self.layer_idx = None
        self.fcs_cond_cache = {}

    def clip_layer(self, layer_idx):
        self.layer_idx = layer_idx

    def add_patches(self, patches, weight):
        return self.patcher.add_patches(patches, weight)

    def clone(self):
        n = CLIP(self.cond_stage_model, self.tokenizer, self.patcher.load_device, self.patcher.offload_device)
        n.patcher = self.patcher.clone()
        n.layer_idx = self.layer_idx
        return n

    def tokenize(self, text, return_word_ids=False):
        return self.tokenizer.tokenize_with_weights(text, return_word_ids)

    def _apply_clip_layer_selection(self):
        if self.layer_idx is not None:
            self.cond_stage_model.clip_layer(self.layer_idx)
        else:
            self.cond_stage_model.reset_clip_layer()

    def encode_from_tokens_resident(self, tokens, return_pooled=False):
        self._apply_clip_layer_selection()
        cond, pooled = self.cond_stage_model.encode_token_weights(tokens)
        if return_pooled:
            return cond, pooled
        return cond

    def encode_from_tokens(self, tokens, return_pooled=False):
        resources.prepare_models_for_stage(
            [self.patcher],
            stage_name="text_encode",
            target_phase=resources.MemoryPhase.PROMPT_ENCODE,
            force_full_load=True,
        )
        return self.encode_from_tokens_resident(tokens, return_pooled=return_pooled)

class VAE:
    """Isolated VAE container to avoid modules.sd baggage."""
    def __init__(self, first_stage_model, load_device, offload_device, latent_format=None):
        self.first_stage_model = first_stage_model
        # Use SD15 as default if not specified (backward compatibility)
        if latent_format is None:
            logging.debug("VAE: No latent_format provided, defaulting to SD15.")
            latent_format = latent_formats.SD15()
        self.latent_format = latent_format
        self.patcher = patching.NexModelPatcher(
            self.first_stage_model,
            load_device=load_device,
            offload_device=offload_device
        )

    def clone(self):
        n = VAE(
            self.first_stage_model,
            self.patcher.load_device,
            self.patcher.offload_device,
            latent_format=self.latent_format,
        )
        n.patcher = self.patcher.clone()
        return n

    def decode(self, samples, tiled=False, tile_size=64):
        from . import decode
        return decode.decode_latent(self, samples, tiled=tiled, tile_size=tile_size)

    def encode(self, pixels):
        from . import encode
        return encode.encode_pixels(self, pixels)

# --- SDXL Support ---

def load_sdxl_unet(
    source,
    load_device=None,
    offload_device=None,
    dtype=None,
    reload_source=None,
    reload_prefixes=None,
    *,
    execution_class=None,
):
    """
    Loads the SDXL UNet using sdxl_def.UNET_CONFIG.
    Supports .gguf integration.
    """
    load_device = load_device or resources.get_torch_device()
    offload_device = offload_device or resources.unet_offload_device()
    effective_dtype = dtype or torch.float16
    
    custom_operations = None
    patcher_class = patching.NexModelPatcher
    runtime_reload = None

    if isinstance(source, str) and source.endswith(".gguf"):
        from backend.gguf.loader import gguf_sd_loader, is_streaming_execution_class
        from backend.gguf.ops import GGMLOps
        from backend.gguf.patcher import GGUFModelPatcher

        streaming = is_streaming_execution_class(execution_class)
        if streaming:
            load_device = torch.device("cpu") if load_device is None else torch.device(load_device)
            offload_device = torch.device("cpu") if offload_device is None else torch.device(offload_device)
            if load_device.type != "cpu" or offload_device.type != "cpu":
                raise RuntimeError("Streaming-class SDXL GGUF loads must stage weights on CPU pinned host memory.")
        sd = gguf_sd_loader(source, pin_memory=streaming, execution_class=execution_class, require_pinned_host=streaming)
        custom_operations = GGMLOps
        patcher_class = GGUFModelPatcher
    else:
        sd = resolve_source(source, device=load_device)
        runtime_reload = _build_unet_runtime_reload(
            reload_source if reload_source is not None else source,
            dtype=effective_dtype,
            prefixes=reload_prefixes,
        )

    model = model_base.SDXL(
        model_config=ModelConfig(sdxl_def.UNET_CONFIG, latent_formats.SDXL()),
        operations=custom_operations
    )
    
    # User Requirement: SDXL UNet should be in fp16 to avoid casting to fp32 (saving RAM/VRAM)
    dtype = effective_dtype
        
    if dtype is not None:
        model.diffusion_model.to(device=load_device, dtype=dtype)
    else:
        model.diffusion_model.to(device=load_device)
        
    model.diffusion_model.load_state_dict(sd, strict=False)
    
    patcher_kwargs = {
        "load_device": load_device,
        "offload_device": offload_device,
        "runtime_reload": runtime_reload,
        "runtime_release_to_meta": runtime_reload is not None,
    }
    if isinstance(source, str) and source.endswith(".gguf"):
        patcher_kwargs["preserve_source_artifact"] = is_streaming_execution_class(execution_class)
    return patcher_class(model, **patcher_kwargs)

def patch_unet_for_quality(unet_patcher: Any, quality: Dict[str, Any]):
    """
    Monkey-patches the UNet's forward pass to support Timed ADM.
    """
    if not quality:
        return
        
    unet = unet_patcher.model.diffusion_model
    if hasattr(unet, "_nex_quality_patched"):
        return
    unet._nex_quality_patched = True

    adm_scaler_end = quality.get("adm_scaler_end", 0.3)
    
    original_forward = unet.forward
    
    def nex_patched_forward(x, timesteps, context=None, y=None, control=None, transformer_options={}, **kwargs):
        # Prevent per-layer upcasting slowness (~3-4x penalty on Windows/NVIDIA)
        # model_base.apply_model() does NOT cast everything correctly on all paths.
        from backend import precision
        x, timesteps, context, y, control = precision.cast_unet_inputs(
            x, timesteps, context=context, y=y, control=control, weight_dtype=unet.dtype
        )

        if y is not None:
             # timed_adm(y, timestep, model, adm_scaler_end)
             y = conditioning.timed_adm(y, timesteps, unet_patcher.model, adm_scaler_end=adm_scaler_end)
             
        return original_forward(x, timesteps, context=context, y=y, control=control, transformer_options=transformer_options, **kwargs)
        
    unet.forward = nex_patched_forward
    logging.info(f"[Nex] Quality: UNet patched for Timed ADM (scaler_end={adm_scaler_end})")

def patch_controlnet_for_quality(controlnet: Any, quality: Dict[str, Any]):
    """
    Monkey-patches ControlNet's forward pass to support Timed ADM and Softness.
    Accepts either a raw ControlNet module or a backend wrapper object.
    """
    if not quality:
        return

    target = getattr(controlnet, "control_model", controlnet)
    if target is None or not hasattr(target, "forward"):
        setattr(controlnet, "_nex_pending_quality", dict(quality))
        return

    if hasattr(target, "_nex_quality_patched"):
        return
    target._nex_quality_patched = True

    controlnet_softness = quality.get("controlnet_softness", 0.0)
    original_forward = target.forward

    def nex_patched_forward(x, hint, timesteps, context, y=None, **kwargs):
        if y is not None:
            y = conditioning.timed_adm(y, timesteps, target, adm_scaler_end=quality.get("adm_scaler_end", 0.3))

        outs = original_forward(x, hint, timesteps, context, y=y, **kwargs)

        if controlnet_softness > 0 and isinstance(outs, list):
            for i in range(len(outs)):
                k = 1.0 - float(i) / (len(outs) - 1) if len(outs) > 1 else 1.0
                outs[i] = outs[i] * (1.0 - controlnet_softness * k)
        return outs

    target.forward = nex_patched_forward
    logging.info(f"[Nex] Quality: ControlNet patched (softness={controlnet_softness})")

def load_sdxl_clip(
    source_l,
    source_g,
    load_device=None,
    offload_device=None,
    dtype=None,
    *,
    reload_source_l=None,
    reload_source_g=None,
    reload_prefixes_l=None,
    reload_prefixes_g=None,
):
    """
    Loads SDXL CLIP (L and G) and returns a clean CLIP container.
    """
    load_device = load_device or resources.text_encoder_load_device()
    offload_device = offload_device or resources.text_encoder_offload_device()

    streamable_sources = (
        isinstance(source_l, str)
        and source_l.lower().endswith(".safetensors")
        and isinstance(source_g, str)
        and source_g.lower().endswith(".safetensors")
    )
    if streamable_sources:
        same_source = False
        sd_l = None
        sd_g = None
    else:
        same_source = source_g is source_l
        if not same_source and isinstance(source_l, str) and isinstance(source_g, str):
            same_source = source_l == source_g
        sd_l = resolve_source(source_l)
        sd_g = sd_l if same_source else resolve_source(source_g)
    
    # Use Nex implementations
    tokenizer = clip.NexSDXLTokenizer()
    
    # SDXL CLIP should stay resident in fp32 so CPU/GPU prompt encode does not
    # repeatedly upcast fp16 weights to match fp32 activations at runtime.
    if dtype is None:
        dtype = torch.float32 
        
    model = clip.NexSDXLClipModel(device=offload_device, dtype=dtype)
    
    with torch.no_grad():
        if streamable_sources:
            _load_sdxl_clip_source_into_model(
                model,
                source_l,
                force_type="l",
                prefixes=reload_prefixes_l,
                dtype=dtype,
            )
            _load_sdxl_clip_source_into_model(
                model,
                source_g,
                force_type="g",
                prefixes=reload_prefixes_g,
                dtype=dtype,
            )
        else:
            if sd_l is not None:
                if same_source:
                    model.load_sd(sd_l)
                else:
                    model.load_sd(sd_l, force_type="l")
            if sd_g is not None and not same_source:
                model.load_sd(sd_g, force_type="g")
    
    clip_container = CLIP(model, tokenizer, load_device, offload_device)
    effective_reload_source_l = (
        reload_source_l
        if reload_source_l is not None
        else (source_l if isinstance(source_l, str) else None)
    )
    effective_reload_source_g = (
        reload_source_g
        if reload_source_g is not None
        else (source_g if isinstance(source_g, str) else None)
    )
    clip_container.patcher.runtime_reload = _build_sdxl_clip_runtime_reload(
        effective_reload_source_l,
        effective_reload_source_g,
        dtype=dtype,
        prefixes_l=reload_prefixes_l,
        prefixes_g=reload_prefixes_g,
    )
    clip_container.patcher.runtime_release_to_meta = False
    return clip_container

def load_vae(source, load_device=None, offload_device=None, dtype=None, latent_format=None, *, prefixes=None):
    """
    Loads VAE/AE (SD15/SDXL/Flux compatible) and returns a clean VAE container.
    """
    load_device = load_device or resources.get_torch_device()
    offload_device = offload_device or resources.vae_offload_device()

    # Try to infer latent format from filename if not provided.
    if latent_format is None and isinstance(source, str):
        normalized_source = source.replace("\\", "/").lower()
        if "/flux_fill/" in normalized_source or "/flux/" in normalized_source:
            latent_format = latent_formats.Flux()
            logging.info(f"VAE: Inferred Flux latent format from {source}")

    if latent_format is None and isinstance(source, str):
        from modules import model_taxonomy
        arch = model_taxonomy.infer_architecture_from_filename(source)
        if arch == model_taxonomy.ARCHITECTURE_SDXL:
            latent_format = latent_formats.SDXL()
            logging.info(f"VAE: Inferred SDXL latent format from {source}")
        elif arch == model_taxonomy.ARCHITECTURE_SD15:
            latent_format = latent_formats.SD15()
            logging.info(f"VAE: Inferred SD15 latent format from {source}")

    streamable_source = isinstance(source, str) and source.lower().endswith(".safetensors")
    vae_metadata = None
    if streamable_source:
        vae_metadata = _inspect_safetensors_vae_metadata(source, prefixes=prefixes)
        decoder_conv_in_shape = vae_metadata["decoder_conv_in_shape"]
        post_quant_conv_shape = vae_metadata["post_quant_conv_shape"]
    else:
        sd = resolve_source(source, device=load_device)
        decoder_conv_in_shape = tuple(sd["decoder.conv_in.weight"].shape) if "decoder.conv_in.weight" in sd else None
        post_quant_conv_shape = tuple(sd["post_quant_conv.weight"].shape) if "post_quant_conv.weight" in sd else None

    if latent_format is None and decoder_conv_in_shape is not None:
        latent_channels = decoder_conv_in_shape[1]
        if latent_channels == latent_formats.Flux.latent_channels:
            latent_format = latent_formats.Flux()
            logging.info("VAE: Inferred Flux latent format from 16-channel AE state dict")

    # Generic VAE config; derive latent/embed width from the state dict for Flux AE.
    ddconfig = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
    if decoder_conv_in_shape is not None:
        ddconfig["z_channels"] = decoder_conv_in_shape[1]
        has_downsample = vae_metadata["has_downsample"] if vae_metadata is not None else ('encoder.down.2.downsample.conv.weight' in sd)
        has_upsample = vae_metadata["has_upsample"] if vae_metadata is not None else ('decoder.up.3.upsample.conv.weight' in sd)
        if not has_downsample and not has_upsample:
            ddconfig['ch_mult'] = [1, 2, 4]

    if post_quant_conv_shape is not None:
        model = AutoencoderKL(ddconfig=ddconfig, embed_dim=post_quant_conv_shape[1])
    elif decoder_conv_in_shape is not None:
        model = AutoencodingEngine(
            regularizer_config={'target': "ldm_patched.ldm.models.autoencoder.DiagonalGaussianRegularizer"},
            encoder_config={'target': "ldm_patched.ldm.modules.diffusionmodules.model.Encoder", 'params': ddconfig},
            decoder_config={'target': "ldm_patched.ldm.modules.diffusionmodules.model.Decoder", 'params': ddconfig},
        )
    else:
        model = AutoencoderKL(ddconfig=ddconfig, embed_dim=4)

    # User Requirement: VAE should be in fp32
    if dtype is None:
        dtype = torch.float32

    if dtype is not None:
        model.to(dtype)

    if streamable_source:
        missing, unexpected = _load_prefixed_safetensors_into_module(
            source,
            prefixes or [""],
            model,
            device=torch.device("cpu"),
            dtype=dtype,
        )
    else:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        del sd
    if missing:
        logging.debug("VAE: Missing keys while loading: %s", missing)
    if unexpected:
        logging.debug("VAE: Unexpected keys while loading: %s", unexpected)
    gc.collect()

    return VAE(model.eval(), load_device, offload_device, latent_format=latent_format)


def load_sdxl_checkpoint(
    ckpt_path,
    load_device=None,
    offload_device=None,
    unet_dtype=None,
    *,
    clip_load_device=None,
    clip_offload_device=None,
    vae_load_device=None,
    vae_offload_device=None,
    vae_source=None,
):
    """
    Loads SDXL components sequentially and clears raw data immediately.
    """
    logging.info(f"Loading SDXL checkpoint from: {ckpt_path}")
    load_device = load_device or resources.get_torch_device()
    clip_dtype = torch.float32
    external_vae_source = vae_source
    vae_load_device = vae_load_device or vae_offload_device or resources.vae_offload_device()

    if isinstance(ckpt_path, str) and ckpt_path.lower().endswith(".safetensors"):
        if external_vae_source is not None:
            if isinstance(external_vae_source, VAE):
                logging.info("Reusing preloaded external SDXL VAE instance...")
                vae = external_vae_source
            else:
                logging.info("Streaming external SDXL VAE override instead of checkpoint VAE...")
                vae = load_vae(
                    external_vae_source,
                    load_device=vae_load_device,
                    offload_device=vae_offload_device,
                    latent_format=latent_formats.SDXL(),
                )
                gc.collect()
        else:
            logging.info("Streaming VAE from safetensors checkpoint directly into CPU module...")
            vae_metadata = _inspect_safetensors_vae_metadata(ckpt_path, prefixes=sdxl_def.PREFIXES["vae"])
            if vae_metadata["key_count"] > 0:
                vae = load_vae(
                    ckpt_path,
                    load_device=vae_load_device,
                    offload_device=vae_offload_device,
                    latent_format=latent_formats.SDXL(),
                    prefixes=sdxl_def.PREFIXES["vae"],
                )
            else:
                logging.warning("SDXL checkpoint is missing embedded VAE weights; continuing without checkpoint VAE.")
                vae = None
            gc.collect()

        logging.info("Streaming CLIP from safetensors checkpoint directly into CPU module...")
        clip = load_sdxl_clip(
            ckpt_path,
            ckpt_path,
            load_device=clip_load_device,
            offload_device=clip_offload_device,
            dtype=clip_dtype,
            reload_source_l=ckpt_path,
            reload_source_g=ckpt_path,
            reload_prefixes_l=sdxl_def.PREFIXES["clip_l"],
            reload_prefixes_g=sdxl_def.PREFIXES["clip_g"],
        )
        gc.collect()

        logging.info("Extracting UNet from safetensors checkpoint directly to load device...")
        unet = _stream_load_sdxl_unet_from_checkpoint(
            ckpt_path,
            load_device=load_device,
            offload_device=offload_device,
            dtype=unet_dtype,
            reload_source=ckpt_path,
            reload_prefixes=sdxl_def.PREFIXES["unet"],
        )
        gc.collect()
        return unet, clip, vae

    sd = utils.load_torch_file(ckpt_path)
    gc.collect()

    # VAE (fp32 default)
    if external_vae_source is not None:
        if isinstance(external_vae_source, VAE):
            logging.info("Reusing preloaded external SDXL VAE instance...")
            vae = external_vae_source
        else:
            logging.info("Loading external SDXL VAE override instead of checkpoint VAE...")
            vae = load_vae(
                external_vae_source,
                load_device=vae_load_device,
                offload_device=vae_offload_device,
                latent_format=latent_formats.SDXL(),
            )
            gc.collect()
    else:
        logging.info("Extracting VAE...")
        vae_sd = {}
        keys = list(sd.keys())
        for k in keys:
            for p in sdxl_def.PREFIXES["vae"]:
                if k.startswith(p):
                    new_key = k[len(p):]
                    if new_key.startswith("."): new_key = new_key[1:]
                    vae_sd[new_key] = sd.pop(k)
                    break
        
        if len(vae_sd) > 0:
            vae = load_vae(
                vae_sd,
                load_device=vae_load_device,
                offload_device=vae_offload_device,
                latent_format=latent_formats.SDXL(),
            )
        else:
            logging.warning("SDXL checkpoint is missing embedded VAE weights; continuing without checkpoint VAE.")
            vae = None
        del vae_sd
        gc.collect()
    
    # CLIP (fp16 default)
    logging.info("Extracting CLIP...")
    clip_l_sd = {}
    clip_g_sd = {}
    keys = list(sd.keys())
    for k in keys:
        for p in sdxl_def.PREFIXES["clip_l"]:
            if k.startswith(p):
                new_key = k[len(p):]
                if new_key.startswith("."): new_key = new_key[1:]
                clip_l_sd[new_key] = sd.pop(k)
                break
        
        if k in sd: 
            for p in sdxl_def.PREFIXES["clip_g"]:
                if k.startswith(p):
                    new_key = k[len(p):]
                    if new_key.startswith("."): new_key = new_key[1:]
                    clip_g_sd[new_key] = sd.pop(k)
                    break
                    
    clip = load_sdxl_clip(
        clip_l_sd,
        clip_g_sd,
        load_device=clip_load_device,
        offload_device=clip_offload_device,
        dtype=clip_dtype,
        reload_source_l=ckpt_path,
        reload_source_g=ckpt_path,
        reload_prefixes_l=sdxl_def.PREFIXES["clip_l"],
        reload_prefixes_g=sdxl_def.PREFIXES["clip_g"],
    )
    del clip_l_sd
    del clip_g_sd
    gc.collect()

    # UNet (fp16 default)
    logging.info("Extracting UNet...")
    unet_sd = {}
    keys = list(sd.keys())
    for k in keys:
        for p in sdxl_def.PREFIXES["unet"]:
            if k.startswith(p):
                new_key = k[len(p):]
                if new_key.startswith("."): new_key = new_key[1:]
                unet_sd[new_key] = sd.pop(k)
                break
    
    if len(sd) > 0:
        logging.info(f"Remaining keys in checkpoint: {len(sd)}")
    
    logging.debug("Deleting original checkpoint storage...")
    del sd
    gc.collect() 

    logging.info("Loading UNet Model...")
    unet = load_sdxl_unet(
        unet_sd,
        load_device=load_device,
        offload_device=offload_device,
        dtype=unet_dtype,
        reload_source=ckpt_path,
        reload_prefixes=sdxl_def.PREFIXES["unet"],
    )
    del unet_sd
    gc.collect()
    
    return unet, clip, vae

# --- SD 1.5 Support ---

def load_sd15_unet(source, load_device=None, offload_device=None, dtype=None, reload_source=None, reload_prefixes=None):
    """
    Loads the SD 1.5 UNet using sd15_def.UNET_CONFIG.
    """
    load_device = load_device or resources.get_torch_device()
    offload_device = offload_device or resources.unet_offload_device()
    effective_dtype = dtype or torch.float16
    
    sd = resolve_source(source, device=load_device)
    runtime_reload = _build_unet_runtime_reload(
        reload_source if reload_source is not None else source,
        dtype=effective_dtype,
        prefixes=reload_prefixes,
    )

    model = model_base.BaseModel(
        model_config=ModelConfig(sd15_def.UNET_CONFIG, latent_formats.SD15()),
    )
    
    # User Requirement: SD1.5 UNet should be in fp16
    dtype = effective_dtype
        
    model.diffusion_model.to(device=load_device, dtype=dtype)
    model.diffusion_model.load_state_dict(sd, strict=False)
    
    return patching.NexModelPatcher(
        model,
        load_device=load_device,
        offload_device=offload_device,
        runtime_reload=runtime_reload,
        runtime_release_to_meta=runtime_reload is not None,
    )

def load_sd15_clip(source, load_device=None, offload_device=None, dtype=None):
    """
    Loads SD 1.5 CLIP (L only) and returns a clean CLIP container.
    """
    load_device = load_device or resources.text_encoder_load_device()
    offload_device = offload_device or resources.text_encoder_offload_device()
    
    sd = resolve_source(source)
    
    tokenizer, encoder = clip.create_sd15_clip(sd)
    
    # NexClipEncoder implements the cond_stage_model interface directly.
    return CLIP(encoder, tokenizer, load_device, offload_device)

def load_sd15_checkpoint(ckpt_path, load_device=None, unet_dtype=None):
    """
    Loads SD 1.5 components sequentially.
    """
    logging.info(f"Loading SD 1.5 checkpoint from: {ckpt_path}")
    load_device = load_device or resources.get_torch_device()

    if isinstance(ckpt_path, str) and ckpt_path.lower().endswith(".safetensors"):
        logging.info("Extracting VAE from safetensors checkpoint...")
        vae_sd = _extract_prefixed_safetensors_state_dict(
            ckpt_path,
            sd15_def.PREFIXES["vae"],
            device=torch.device("cpu"),
        )
        vae = load_vae(vae_sd, latent_format=latent_formats.SD15())
        del vae_sd
        gc.collect()

        logging.info("Extracting CLIP from safetensors checkpoint...")
        clip_sd = _extract_prefixed_safetensors_state_dict(
            ckpt_path,
            sd15_def.PREFIXES["clip"],
            device=torch.device("cpu"),
        )
        clip = load_sd15_clip(clip_sd, dtype=unet_dtype)
        heal_model_weights(clip.patcher.model, "CLIP")
        del clip_sd
        gc.collect()

        # Precision Injection (SD1.5 specific fix for NaN overflows)
        try:
            sd1_clip_model = clip.cond_stage_model
            transformer = None

            if hasattr(sd1_clip_model, 'transformer'):
                 transformer = sd1_clip_model.transformer
            elif hasattr(sd1_clip_model, 'clip_l'):
                 transformer = sd1_clip_model.clip_l.transformer
            elif hasattr(sd1_clip_model, 'clip'):
                 transformer = sd1_clip_model.clip.transformer

            if transformer is not None and hasattr(transformer, 'text_model'):
                 transformer = transformer.text_model

            if transformer is not None and hasattr(transformer, 'embeddings'):
               embeddings = transformer.embeddings
               if not isinstance(embeddings, EmbeddingFP32Wrapper):
                   transformer.embeddings = EmbeddingFP32Wrapper(embeddings)
        except Exception as e:
            logging.error(f"FAILED to apply precision injection to CLIP: {e}")

        logging.info("Extracting UNet from safetensors checkpoint directly to load device...")
        unet_sd = _extract_prefixed_safetensors_state_dict(
            ckpt_path,
            sd15_def.PREFIXES["unet"],
            device=load_device,
        )
        unet = load_sd15_unet(
            unet_sd,
            load_device=load_device,
            dtype=unet_dtype,
            reload_source=ckpt_path,
            reload_prefixes=sd15_def.PREFIXES["unet"],
        )
        heal_model_weights(unet.model, "UNet")
        del unet_sd
        gc.collect()
        return unet, clip, vae

    sd = utils.load_torch_file(ckpt_path)
    gc.collect()

    # VAE (fp32)
    logging.info("Extracting VAE...")
    vae_sd = {}
    keys = list(sd.keys())
    for k in keys:
        for p in sd15_def.PREFIXES["vae"]:
            if k.startswith(p):
                new_key = k[len(p):]
                if new_key.startswith("."): new_key = new_key[1:]
                vae_sd[new_key] = sd.pop(k)
                break
    
    vae = load_vae(vae_sd, latent_format=latent_formats.SD15())
    del vae_sd
    gc.collect()
    
    # CLIP (fp16)
    logging.info("Extracting CLIP...")
    clip_sd = {}
    keys = list(sd.keys())
    for k in keys:
        for p in sd15_def.PREFIXES["clip"]:
            if k.startswith(p):
                new_key = k[len(p):]
                if new_key.startswith("."): new_key = new_key[1:]
                clip_sd[new_key] = sd.pop(k)
                break
                     
    clip = load_sd15_clip(clip_sd, dtype=unet_dtype)
    heal_model_weights(clip.patcher.model, "CLIP")

    # Precision Injection (SD1.5 specific fix for NaN overflows)
    try:
        sd1_clip_model = clip.cond_stage_model
        transformer = None
        
        if hasattr(sd1_clip_model, 'transformer'): 
             transformer = sd1_clip_model.transformer
        elif hasattr(sd1_clip_model, 'clip_l'): 
             transformer = sd1_clip_model.clip_l.transformer
        elif hasattr(sd1_clip_model, 'clip'):
             transformer = sd1_clip_model.clip.transformer
             
        if transformer is not None and hasattr(transformer, 'text_model'):
             transformer = transformer.text_model

        if transformer is not None and hasattr(transformer, 'embeddings'):
           embeddings = transformer.embeddings
           if not isinstance(embeddings, EmbeddingFP32Wrapper):
               transformer.embeddings = EmbeddingFP32Wrapper(embeddings)
    except Exception as e:
        logging.error(f"FAILED to apply precision injection to CLIP: {e}")

    del clip_sd
    gc.collect()

    # UNet (fp16)
    logging.info("Extracting UNet...")
    unet_sd = {}
    keys = list(sd.keys())
    for k in keys:
        for p in sd15_def.PREFIXES["unet"]:
            if k.startswith(p):
                new_key = k[len(p):]
                if new_key.startswith("."): new_key = new_key[1:]
                unet_sd[new_key] = sd.pop(k)
                break
    
    if len(sd) > 0:
        logging.info(f"Remaining keys in checkpoint: {len(sd)}")
    
    logging.debug("Deleting original checkpoint storage...")
    del sd
    gc.collect() 

    logging.info("Loading UNet Model...")
    unet = load_sd15_unet(
        unet_sd,
        load_device=load_device,
        dtype=unet_dtype,
        reload_source=ckpt_path,
        reload_prefixes=sd15_def.PREFIXES["unet"],
    )
    heal_model_weights(unet.model, "UNet")
    del unet_sd
    gc.collect()
    
    return unet, clip, vae
