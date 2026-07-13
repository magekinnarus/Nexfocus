import gc
import logging
from typing import Any

import torch

from backend import float_ops as backend_float_ops, utils as backend_utils
from backend.weight_ops import get_key_weight, string_to_seed
import ldm_patched.modules.weight_adapter as weight_adapter
from ldm_patched.modules.weight_adapter.base import pad_tensor_to_shape, weight_decompose


def _identity(x):
    return x


def _resolve_tensor(w: Any, device: torch.device, dtype: torch.dtype) -> Any:
    if w is None:
        return None
    if hasattr(w, "load") and callable(w.load):
        w = w.load()
    if isinstance(w, torch.Tensor):
        return w.to(device=device, dtype=dtype, non_blocking=True)
    return w


def _resolve_scalar(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, torch.Tensor):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _clear_cached_adapter_tensor(item: Any) -> None:
    clear_cached = getattr(item, "clear_materialized_tensor", None)
    if callable(clear_cached):
        clear_cached()
        return
    if hasattr(item, "_tensor"):
        item._tensor = None


class GpuArtifactCompiler:
    """
    GPU-native, single-threaded compiler for merging LoRA weights directly into a
    GPU-resident base model.
    """

    @classmethod
    @torch.no_grad()
    def compile_patcher(
        cls,
        patcher: Any,
        clean_source: dict[str, torch.Tensor] | None,
        target_device: torch.device,
        intermediate_dtype: torch.dtype,
    ) -> dict[str, Any]:
        patcher.model.requires_grad_(False)
        patcher.model.eval()

        clean_source = clean_source or {}
        patch_count = len(getattr(patcher, "patches", {}) or {})
        if patch_count == 0:
            return {"status": "noop", "patch_count": 0, "materialized_patch_keys": 0, "host_pinned_bytes": 0}

        logging.info(
            f"[GpuArtifactCompiler] Compiling {patch_count} patches on device {target_device} "
            "sequentially (single-threaded)."
        )

        try:
            for key in list(patcher.patches.keys()):
                weight, set_func, convert_func = get_key_weight(patcher.model, key)
                preserved_dtype = weight.dtype

                clean_weight = clean_source.get(key, weight)
                temp_weight = clean_weight.to(device=target_device, dtype=intermediate_dtype, copy=True)
                if convert_func is not None:
                    temp_weight = convert_func(temp_weight, inplace=True)

                patches = patcher.patches[key]
                out_weight = cls._patch_single_layer_worker(
                    key,
                    temp_weight,
                    patches,
                    intermediate_dtype,
                    target_device,
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

                del temp_weight, out_weight
        finally:
            for _, patches in list(getattr(patcher, "patches", {}).items()):
                for patch in patches:
                    if len(patch) > 1:
                        patch_payload = patch[1]
                        if isinstance(
                            patch_payload,
                            (
                                weight_adapter.LoRAAdapter,
                                weight_adapter.LoHaAdapter,
                                weight_adapter.LoKrAdapter,
                                weight_adapter.GLoRAAdapter,
                            ),
                        ):
                            for weight_entry in getattr(patch_payload, "weights", ()):
                                if weight_entry is not None:
                                    _clear_cached_adapter_tensor(weight_entry)
                        elif isinstance(patch_payload, tuple):
                            for item in patch_payload:
                                _clear_cached_adapter_tensor(item)

            patcher.patches = {}
            patcher.weight_wrapper_patches = {}
            patcher.backup.clear()
            patcher.object_patches_backup.clear()
            patcher.model.current_weight_patches_uuid = None
            patcher.model.model_loaded_weight_memory = patcher.model_size()
            patcher.model.model_lowvram = False
            patcher.model.lowvram_patch_counter = 0
            patcher.model.device = target_device

            gc.collect()
            if target_device.type == "cuda":
                torch.cuda.empty_cache()

        return {
            "status": "compiled",
            "patch_count": patch_count,
            "materialized_patch_keys": patch_count,
            "host_pinned_bytes": 0,
        }

    @classmethod
    def _patch_single_layer_worker(
        cls,
        key: str,
        base_tensor: torch.Tensor,
        patches: list[Any],
        intermediate_dtype: torch.dtype,
        target_device: torch.device,
    ) -> torch.Tensor:
        weight = base_tensor
        for patch in patches:
            strength_patch, patch_payload, strength_model, offset, function = patch

            if offset is not None:
                weight = weight.narrow(offset[0], offset[1], offset[2])

            if strength_model != 1.0:
                weight.mul_(strength_model)

            function = function or _identity

            if isinstance(patch_payload, weight_adapter.LoRAAdapter):
                v = patch_payload.weights
                mat1 = _resolve_tensor(v[0], target_device, intermediate_dtype)
                mat2 = _resolve_tensor(v[1], target_device, intermediate_dtype)
                alpha_val = _resolve_scalar(v[2])
                mid = _resolve_tensor(v[3], target_device, intermediate_dtype)
                dora_scale = _resolve_tensor(v[4], target_device, intermediate_dtype)
                reshape = v[5]

                if reshape is not None:
                    weight = pad_tensor_to_shape(weight, reshape)

                alpha = alpha_val / mat2.shape[0] if alpha_val is not None else 1.0

                if mid is not None:
                    final_shape = [mat2.shape[1], mat2.shape[0], mid.shape[2], mid.shape[3]]
                    mat2 = torch.mm(
                        mat2.transpose(0, 1).flatten(start_dim=1),
                        mid.transpose(0, 1).flatten(start_dim=1),
                    ).reshape(final_shape).transpose(0, 1)

                if dora_scale is None and weight.ndim in (2, 4) and mat1.ndim == 2 and mat2.ndim in (2, 4):
                    m1_flat = mat1.flatten(start_dim=1)
                    m2_flat = mat2.flatten(start_dim=1)
                    weight_view = weight.view(weight.shape[0], -1)
                    weight_view.addmm_(m1_flat, m2_flat, alpha=strength_patch * alpha)
                else:
                    lora_diff = torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1)).reshape(weight.shape)
                    if dora_scale is not None:
                        weight = weight_decompose(
                            dora_scale,
                            weight,
                            lora_diff,
                            alpha,
                            strength_patch,
                            intermediate_dtype,
                            function,
                        )
                    else:
                        weight.add_(function((strength_patch * alpha) * lora_diff).to(weight.dtype))

                del mat1, mat2, mid, dora_scale

            elif isinstance(patch_payload, weight_adapter.LoHaAdapter):
                v = patch_payload.weights
                w1a = _resolve_tensor(v[0], target_device, intermediate_dtype)
                w1b = _resolve_tensor(v[1], target_device, intermediate_dtype)
                alpha_val = _resolve_scalar(v[2])
                w2a = _resolve_tensor(v[3], target_device, intermediate_dtype)
                w2b = _resolve_tensor(v[4], target_device, intermediate_dtype)
                t1 = _resolve_tensor(v[5], target_device, intermediate_dtype)
                t2 = _resolve_tensor(v[6], target_device, intermediate_dtype)
                dora_scale = _resolve_tensor(v[7], target_device, intermediate_dtype)

                alpha = alpha_val / w1b.shape[0] if alpha_val is not None else 1.0

                if t1 is not None and t2 is not None:
                    m1 = torch.einsum("i j k l, j r, i p -> p r k l", t1, w1b, w1a)
                    m2 = torch.einsum("i j k l, j r, i p -> p r k l", t2, w2b, w2a)
                else:
                    m1 = torch.mm(w1a, w1b)
                    m2 = torch.mm(w2a, w2b)

                lora_diff = (m1 * m2).reshape(weight.shape)
                if dora_scale is not None:
                    weight = weight_decompose(
                        dora_scale,
                        weight,
                        lora_diff,
                        alpha,
                        strength_patch,
                        intermediate_dtype,
                        function,
                    )
                else:
                    weight.add_(function((strength_patch * alpha) * lora_diff).to(weight.dtype))

                del w1a, w1b, w2a, w2b, t1, t2, dora_scale, m1, m2

            elif isinstance(patch_payload, weight_adapter.LoKrAdapter):
                v = patch_payload.weights
                w1 = _resolve_tensor(v[0], target_device, intermediate_dtype)
                w2 = _resolve_tensor(v[1], target_device, intermediate_dtype)
                alpha_val = _resolve_scalar(v[2])
                w1_a = _resolve_tensor(v[3], target_device, intermediate_dtype)
                w1_b = _resolve_tensor(v[4], target_device, intermediate_dtype)
                w2_a = _resolve_tensor(v[5], target_device, intermediate_dtype)
                w2_b = _resolve_tensor(v[6], target_device, intermediate_dtype)
                t2 = _resolve_tensor(v[7], target_device, intermediate_dtype)
                dora_scale = _resolve_tensor(v[8], target_device, intermediate_dtype)
                dim = None

                if w1 is None:
                    dim = w1_b.shape[0]
                    w1 = torch.mm(w1_a, w1_b)

                if w2 is None:
                    dim = w2_b.shape[0]
                    if t2 is None:
                        w2 = torch.mm(w2_a, w2_b)
                    else:
                        w2 = torch.einsum("i j k l, j r, i p -> p r k l", t2, w2_b, w2_a)

                if len(w2.shape) == 4:
                    w1 = w1.unsqueeze(2).unsqueeze(2)
                alpha = alpha_val / dim if alpha_val is not None and dim is not None else 1.0

                lora_diff = torch.kron(w1, w2).reshape(weight.shape)
                if dora_scale is not None:
                    weight = weight_decompose(
                        dora_scale,
                        weight,
                        lora_diff,
                        alpha,
                        strength_patch,
                        intermediate_dtype,
                        function,
                    )
                else:
                    weight.add_(function((strength_patch * alpha) * lora_diff).to(weight.dtype))

                del w1, w2, w1_a, w1_b, w2_a, w2_b, t2, dora_scale

            elif isinstance(patch_payload, weight_adapter.GLoRAAdapter):
                v = patch_payload.weights
                a1 = _resolve_tensor(v[0], target_device, intermediate_dtype).flatten(start_dim=1)
                a2 = _resolve_tensor(v[1], target_device, intermediate_dtype).flatten(start_dim=1)
                b1 = _resolve_tensor(v[2], target_device, intermediate_dtype).flatten(start_dim=1)
                b2 = _resolve_tensor(v[3], target_device, intermediate_dtype).flatten(start_dim=1)
                alpha_val = _resolve_scalar(v[4])
                dora_scale = _resolve_tensor(v[5], target_device, intermediate_dtype)

                alpha = alpha_val / v[0].shape[0] if alpha_val is not None else 1.0

                lora_diff = (
                    torch.mm(b2, b1) + torch.mm(torch.mm(weight.flatten(start_dim=1), a2), a1)
                ).reshape(weight.shape)
                if dora_scale is not None:
                    weight = weight_decompose(
                        dora_scale,
                        weight,
                        lora_diff,
                        alpha,
                        strength_patch,
                        intermediate_dtype,
                        function,
                    )
                else:
                    weight.add_(function((strength_patch * alpha) * lora_diff).to(weight.dtype))

                del a1, a2, b1, b2, dora_scale

            elif isinstance(patch_payload, tuple):
                if len(patch_payload) == 1:
                    patch_type = "diff"
                    v = patch_payload
                elif len(patch_payload) == 2:
                    patch_type = patch_payload[0]
                    v = patch_payload[1]
                else:
                    patch_type = patch_payload[0]
                    v = patch_payload[1:]

                if patch_type == "diff":
                    w1 = _resolve_tensor(v[0], target_device, intermediate_dtype)
                    if strength_patch != 0.0 and w1.shape == weight.shape:
                        weight.add_(w1.to(weight.dtype), alpha=strength_patch)
                    del w1
                elif patch_type == "lora":
                    mat1 = _resolve_tensor(v[0], target_device, intermediate_dtype)
                    mat2 = _resolve_tensor(v[1], target_device, intermediate_dtype)
                    alpha_val = _resolve_scalar(v[2])
                    mid = _resolve_tensor(v[3], target_device, intermediate_dtype)
                    alpha = alpha_val / mat2.shape[0] if alpha_val is not None else 1.0
                    if mid is not None:
                        final_shape = [mat2.shape[1], mat2.shape[0], mid.shape[2], mid.shape[3]]
                        mat2 = torch.mm(
                            mat2.transpose(0, 1).flatten(start_dim=1),
                            mid.transpose(0, 1).flatten(start_dim=1),
                        ).reshape(final_shape).transpose(0, 1)
                    if weight.ndim in (2, 4) and mat1.ndim == 2 and mat2.ndim in (2, 4):
                        m1_flat = mat1.flatten(start_dim=1)
                        m2_flat = mat2.flatten(start_dim=1)
                        weight_view = weight.view(weight.shape[0], -1)
                        weight_view.addmm_(m1_flat, m2_flat, alpha=strength_patch * alpha)
                    else:
                        lora_diff = torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1)).reshape(weight.shape)
                        weight.add_((strength_patch * alpha) * lora_diff.to(weight.dtype))
                    del mat1, mat2, mid
                elif patch_type == "fooocus":
                    w1 = _resolve_tensor(v[0], target_device, intermediate_dtype)
                    w_min = _resolve_tensor(v[1], target_device, intermediate_dtype)
                    w_max = _resolve_tensor(v[2], target_device, intermediate_dtype)
                    w1 = (w1 / 255.0) * (w_max - w_min) + w_min
                    if strength_patch != 0.0 and w1.shape == weight.shape:
                        weight.add_(w1.to(weight.dtype), alpha=strength_patch)
                    del w1, w_min, w_max
                elif patch_type == "set":
                    w1 = _resolve_tensor(v[0], target_device, weight.dtype)
                    weight.copy_(w1)
                    del w1

        return weight


ResidentLoRACompiler = GpuArtifactCompiler
