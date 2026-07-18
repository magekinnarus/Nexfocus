import concurrent.futures
import gc
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import torch
from safetensors import safe_open

from backend import float_ops as backend_float_ops, utils as backend_utils
from backend import lora as backend_lora
from backend.weight_ops import get_key_weight, string_to_seed
import ldm_patched.modules.weight_adapter as weight_adapter
from ldm_patched.modules.weight_adapter.base import weight_decompose, pad_tensor_to_shape


def _identity(x):
    return x


@dataclass(frozen=True)
class LoRAPatchDef:
    lora_path: str
    strength: float = 1.0


class LazyWeight:
    def __init__(
        self,
        path: str,
        key: str,
        shape: list[int],
        dtype: str,
        tensor: torch.Tensor | None = None,
        *,
        load_strategy: str = "safetensors",
    ) -> None:
        self.path = path
        self.key = key
        self.shape = list(shape)
        self.dtype = self._normalize_dtype(dtype)
        self._tensor = tensor
        self._load_strategy = str(load_strategy)

    @staticmethod
    def _normalize_dtype(dtype: Any) -> str:
        if isinstance(dtype, torch.dtype):
            return str(dtype)
        mapping = {
            "F16": str(torch.float16),
            "F32": str(torch.float32),
            "BF16": str(torch.bfloat16),
            "I64": str(torch.int64),
            "I32": str(torch.int32),
            "I16": str(torch.int16),
            "I8": str(torch.int8),
            "U8": str(torch.uint8),
            "BOOL": str(torch.bool),
        }
        return mapping.get(str(dtype), str(dtype))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def load(self) -> torch.Tensor:
        if self._tensor is not None:
            return self._tensor
        if self._load_strategy == "safetensors":
            with safe_open(self.path, framework="pt", device="cpu") as handle:
                return handle.get_tensor(self.key)
        if self._load_strategy == "torch_load":
            try:
                sd = torch.load(self.path, map_location="cpu", weights_only=True)
            except Exception:
                sd = torch.load(self.path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            tensor = sd.get(self.key) if isinstance(sd, dict) else None
            if not isinstance(tensor, torch.Tensor):
                raise KeyError(f"Legacy tensor key {self.key!r} could not be reloaded from {self.path!r}.")
            return tensor
        raise RuntimeError(f"Unsupported LazyWeight load strategy: {self._load_strategy!r}")

    def clear_materialized_tensor(self) -> None:
        self._tensor = None

    def item(self):
        return self.load().item()

    def __repr__(self) -> str:
        return f"LazyWeight({self.path!r}, {self.key!r}, {self.shape!r}, {self.dtype!r})"


class SafeOpenHeaderOnly(dict):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
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
                    self[key] = LazyWeight(path, key, shape, dtype, load_strategy="safetensors")
        except Exception:
            try:
                sd = torch.load(path, map_location="cpu", weights_only=True)
            except Exception:
                sd = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            for key, tensor in sd.items():
                if isinstance(tensor, torch.Tensor):
                    self[key] = LazyWeight(
                        path,
                        key,
                        list(tensor.shape),
                        str(tensor.dtype),
                        tensor=tensor,
                        load_strategy="torch_load",
                    )
                else:
                    self[key] = tensor


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


def _pin_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
    if not isinstance(tensor, torch.Tensor):
        return tensor, 0
    if not torch.cuda.is_available():
        return tensor, 0
    if tensor.device.type != "cpu" or tensor.is_pinned():
        return tensor, 0

    pinned = tensor.contiguous().pin_memory()
    return pinned, pinned.numel() * pinned.element_size()


def _measure_pinned_module_tensors(module: torch.nn.Module) -> int:
    if module is None:
        return 0

    pinned_bytes = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu" and tensor.is_pinned():
            pinned_bytes += tensor.numel() * tensor.element_size()
    return pinned_bytes


def _pin_module_tensors(module: torch.nn.Module) -> int:
    if not torch.cuda.is_available():
        return 0

    pinned_bytes = 0
    for _, submodule in module.named_modules():
        for _, param in submodule.named_parameters(recurse=False):
            if param is None or param.device.type != "cpu" or param.is_pinned():
                continue
            pinned, _ = _pin_tensor(param.data)
            param.data = pinned
            pinned_bytes += pinned.numel() * pinned.element_size()
        for name, buf in submodule.named_buffers(recurse=False):
            if buf is None or buf.device.type != "cpu" or buf.is_pinned():
                continue
            pinned, _ = _pin_tensor(buf)
            submodule._buffers[name] = pinned
            pinned_bytes += pinned.numel() * pinned.element_size()
    return pinned_bytes


class CpuArtifactCompiler:
    """
    Multi-threaded, CPU-native artifact compiler for materializing LoRA deltas.

    Generic patcher compilation is non-pinning by default. Host pinning is a
    streaming-UNet transfer policy and new assembly code must opt into it via
    ``compile_streaming_unet_patcher``.
    """

    @staticmethod
    def _resolve_worker_count(num_workers: Optional[int], task_count: int) -> int:
        if task_count <= 0:
            return 1
        if num_workers is not None:
            return max(1, min(int(num_workers), task_count))

        cpu_count = os.cpu_count() or 4
        # Keep default parallelism conservative: PyTorch matmul already uses
        # its own CPU worker pool, so broad fan-out tends to amplify RAM spikes.
        return max(1, min(task_count, 4, max(1, cpu_count // 2)))

    @staticmethod
    def _resolve_torch_threads_per_worker(torch_threads_per_worker: Optional[int]) -> int:
        if torch_threads_per_worker is None:
            return 1
        return max(1, int(torch_threads_per_worker))

    @classmethod
    def _run_tasks(
        cls,
        num_workers: int,
        task_count: int,
        submitter,
        *,
        torch_threads_per_worker: int = 1,
    ):
        if task_count <= 0:
            return

        torch_threads_per_worker = cls._resolve_torch_threads_per_worker(torch_threads_per_worker)

        if num_workers <= 1:
            original_threads = torch.get_num_threads()
            original_interop = torch.get_num_interop_threads()
            for index in range(task_count):
                try:
                    torch.set_num_threads(torch_threads_per_worker)
                    torch.set_num_interop_threads(1)
                except Exception:
                    pass
                with torch.inference_mode():
                    submitter(index)()
            try:
                torch.set_num_threads(original_threads)
                torch.set_num_interop_threads(original_interop)
            except Exception:
                pass
            return

        original_threads = torch.get_num_threads()
        original_interop = torch.get_num_interop_threads()
        try:
            torch.set_num_threads(torch_threads_per_worker)
            torch.set_num_interop_threads(1)
        except Exception:
            original_threads = None
            original_interop = None

        def _inference_wrap(fn):
            def wrapped():
                with torch.inference_mode():
                    return fn()
            return wrapped

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(_inference_wrap(submitter(index))) for index in range(task_count)]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
        finally:
            if original_threads is not None:
                try:
                    torch.set_num_threads(original_threads)
                    torch.set_num_interop_threads(original_interop)
                except Exception:
                    pass

    @classmethod
    @torch.no_grad()
    def compile_patcher(
        cls,
        patcher: Any,
        *,
        pin_unet_host: bool = False,
        num_workers: Optional[int] = None,
        torch_threads_per_worker: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Compile a generic NexModelPatcher in-place on CPU.

        ``pin_unet_host`` is retained temporarily for legacy callers. New code
        must use ``compile_streaming_unet_patcher`` when compiling a streaming
        UNet that may require page-locked host tensors.
        """
        return cls._compile_patcher_impl(
            patcher,
            pin_model_host=bool(pin_unet_host),
            num_workers=num_workers,
            torch_threads_per_worker=torch_threads_per_worker,
        )

    @classmethod
    @torch.no_grad()
    def compile_streaming_unet_patcher(
        cls,
        patcher: Any,
        *,
        pin_unet_host: bool,
        num_workers: Optional[int] = None,
        torch_threads_per_worker: Optional[int] = None,
    ) -> dict[str, Any]:
        """Compile a streaming UNet and apply its explicit host-pinning policy."""
        # Route through the legacy-observable entry point until W13 removes
        # old runtime instrumentation around compile_patcher.
        return cls.compile_patcher(
            patcher,
            pin_unet_host=bool(pin_unet_host),
            num_workers=num_workers,
            torch_threads_per_worker=torch_threads_per_worker,
        )

    @classmethod
    def _compile_patcher_impl(
        cls,
        patcher: Any,
        *,
        pin_model_host: bool,
        num_workers: Optional[int],
        torch_threads_per_worker: Optional[int],
    ) -> dict[str, Any]:
        patcher.model.requires_grad_(False)
        patcher.model.eval()
        
        target_device = torch.device("cpu")
        for param in list(patcher.model.parameters()) + list(patcher.model.buffers()):
            target_device = param.device
            break

        if target_device.type != "cpu":
            raise AssertionError(
                f"CpuArtifactCompiler cannot compile for GPU target device {target_device}. "
                f"Use GpuArtifactCompiler for GPU/Resident compilation."
            )

        patch_count = len(getattr(patcher, "patches", {}) or {})
        if patch_count == 0:
            if pin_model_host:
                _pin_module_tensors(patcher.model)
            return {"status": "noop", "patch_count": 0}

        num_workers = cls._resolve_worker_count(num_workers, patch_count)

        logging.info(f"[CpuArtifactCompiler] Compiling {patch_count} patches on device {target_device} across {num_workers} worker threads.")

        tasks = list(patcher.patches.keys())
        cls._run_tasks(
            num_workers,
            len(tasks),
            lambda index: (
                lambda: cls._compile_single_key_patcher(
                    patcher,
                    tasks[index],
                    target_device,
                    pin_output=pin_model_host,
                )
            ),
            torch_threads_per_worker=torch_threads_per_worker,
        )

        # Clean up patcher
        patcher.patches = {}
        patcher.weight_wrapper_patches = {}
        patcher.backup.clear()
        patcher.object_patches_backup.clear()
        patcher.model.current_weight_patches_uuid = None
        patcher.model.model_loaded_weight_memory = patcher.model_size()
        patcher.model.model_lowvram = False
        patcher.model.lowvram_patch_counter = 0
        patcher.model.device = target_device

        host_pinned_bytes = 0
        if pin_model_host:
            _pin_module_tensors(patcher.model)
            host_pinned_bytes = _measure_pinned_module_tensors(patcher.model)

        gc.collect()
        return {
            "status": "compiled",
            "materialized_patch_keys": patch_count,
            "host_pinned_bytes": host_pinned_bytes,
        }

    @classmethod
    @torch.no_grad()
    def compile_loras_into_model(
        cls,
        model: torch.nn.Module,
        lora_specs: list[tuple[str, float]],
        *,
        pin_unet_host: bool = True,
        num_workers: Optional[int] = None,
        torch_threads_per_worker: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Isolated compiler that mathematically merges LoRAs directly into an instantiated
        UNet, entirely bypassing Comfy/legacy module instantiations (e.g. core.StableDiffusionModel).
        """
        from collections import defaultdict

        key_map = backend_lora.model_lora_keys_unet(model)

        compiled_patches = defaultdict(list)
        for lora_path, strength in lora_specs:
            lora_sd = SafeOpenHeaderOnly(lora_path)
            patch_dict = backend_lora.load_lora(lora_sd, key_map, log_missing=False)
            for target_key, payload in patch_dict.items():
                compiled_patches[target_key].append((strength, payload, 1.0, None, _identity))

        target_device = torch.device("cpu")
        for param in list(model.parameters()) + list(model.buffers()):
            target_device = param.device
            break

        if target_device.type != "cpu":
            raise AssertionError(
                f"CpuArtifactCompiler cannot compile for GPU target device {target_device}. "
                f"Use GpuArtifactCompiler for GPU/Resident compilation."
            )

        patch_count = len(compiled_patches)
        if patch_count == 0:
            if pin_unet_host:
                _pin_module_tensors(model)
            return {"status": "noop", "patch_count": 0}

        num_workers = cls._resolve_worker_count(num_workers, patch_count)

        logging.info(f"[CpuArtifactCompiler] Isolated compile: {patch_count} patches on device {target_device} across {num_workers} threads.")

        model.requires_grad_(False)
        model.eval()

        tasks = list(compiled_patches.items())
        cls._run_tasks(
            num_workers,
            len(tasks),
            lambda index: (
                lambda: cls._compile_single_key_direct(
                    model,
                    tasks[index][0],
                    *get_key_weight(model, tasks[index][0]),
                    tasks[index][1],
                    target_device,
                    pin_output=pin_unet_host,
                )
            ),
            torch_threads_per_worker=torch_threads_per_worker,
        )

        host_pinned_bytes = 0
        if pin_unet_host:
            _pin_module_tensors(model)
            host_pinned_bytes = _measure_pinned_module_tensors(model)

        gc.collect()
        return {
            "status": "compiled",
            "materialized_patch_keys": patch_count,
            "host_pinned_bytes": host_pinned_bytes,
        }

    @classmethod
    @torch.no_grad()
    def compile_unet(
        cls,
        base_state_dict: dict[str, torch.Tensor],
        lora_patch_defs: list[LoRAPatchDef],
        *,
        key_map: dict[str, str],
        pin_unet_host: bool = True,
        num_workers: Optional[int] = None,
        torch_threads_per_worker: Optional[int] = None,
    ) -> dict[str, Any]:
        from collections import defaultdict

        compiled_patches = defaultdict(list)
        for patch_def in lora_patch_defs:
            header = SafeOpenHeaderOnly(patch_def.lora_path)
            patch_dict = backend_lora.load_lora(header, key_map, log_missing=False)
            for target_key, payload in patch_dict.items():
                compiled_patches[target_key].append((patch_def.strength, payload, 1.0, None, _identity))

        result = cls.compile_standalone(
            base_state_dict,
            compiled_patches,
            pin_unet_host=pin_unet_host,
            num_workers=num_workers,
            torch_threads_per_worker=torch_threads_per_worker,
        )
        result["patch_count"] = len(compiled_patches)
        return result

    @classmethod
    def compile_standalone(
        cls,
        base_state_dict: dict[str, torch.Tensor],
        compiled_patches: dict[str, list[Any]],
        *,
        pin_unet_host: bool = True,
        num_workers: Optional[int] = None,
        torch_threads_per_worker: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Entrypoint for standalone dictionary patching.
        """
        target_device = torch.device("cpu")
        for tensor in base_state_dict.values():
            if isinstance(tensor, torch.Tensor):
                target_device = tensor.device
                break

        if target_device.type != "cpu":
            raise AssertionError(
                f"CpuArtifactCompiler cannot compile for GPU target device {target_device}. "
                f"Use GpuArtifactCompiler for GPU/Resident compilation."
            )

        patch_count = len(compiled_patches)
        if patch_count == 0:
            return {"status": "noop", "patch_count": 0}

        num_workers = cls._resolve_worker_count(num_workers, patch_count)

        logging.info(f"[CpuArtifactCompiler] Standalone compile: {patch_count} target keys on device {target_device} across {num_workers} threads.")

        keys = list(compiled_patches.keys())
        cls._run_tasks(
            num_workers,
            len(keys),
            lambda index: (
                lambda: cls._compile_single_key_standalone(
                    base_state_dict,
                    keys[index],
                    compiled_patches[keys[index]],
                    target_device,
                )
            ),
            torch_threads_per_worker=torch_threads_per_worker,
        )

        if pin_unet_host:
            for tensor in base_state_dict.values():
                if isinstance(tensor, torch.Tensor) and not tensor.is_pinned():
                    try:
                        tensor.pin_memory()
                    except Exception:
                        pass

        gc.collect()
        return {"status": "compiled", "materialized_patch_keys": patch_count}

    @classmethod
    def _compile_single_key_patcher(
        cls,
        patcher: Any,
        key: str,
        target_device: torch.device,
        *,
        pin_output: bool = False,
    ):
        weight, set_func, convert_func = get_key_weight(patcher.model, key)
        preserved_dtype = weight.dtype
        # We work directly on target_device
        temp_weight = weight.to(device=target_device, dtype=preserved_dtype, copy=False)
        if convert_func is not None:
            temp_weight = convert_func(temp_weight, inplace=True)

        patches = patcher.patches[key]
        out_weight = cls._patch_single_layer_worker(key, temp_weight, patches, preserved_dtype)

        if set_func is None:
            out_weight = backend_float_ops.stochastic_rounding(
                out_weight,
                preserved_dtype,
                seed=string_to_seed(key),
            )
            if pin_output:
                out_weight, _ = _pin_tensor(out_weight)
            backend_utils.set_attr_param(patcher.model, key, out_weight)
        else:
            if pin_output:
                out_weight, _ = _pin_tensor(out_weight)
            set_func(out_weight, inplace_update=False, seed=string_to_seed(key))

    @classmethod
    def _compile_single_key_direct(
        cls,
        model: torch.nn.Module,
        key: str,
        weight: Any,
        set_func: Any,
        convert_func: Any,
        patches: list[Any],
        target_device: torch.device,
        *,
        pin_output: bool = False,
    ):
        preserved_dtype = weight.dtype
        temp_weight = weight.to(device=target_device, dtype=preserved_dtype, copy=False)
        if convert_func is not None:
            temp_weight = convert_func(temp_weight, inplace=True)

        out_weight = cls._patch_single_layer_worker(key, temp_weight, patches, preserved_dtype)

        if set_func is None:
            out_weight = backend_float_ops.stochastic_rounding(
                out_weight,
                preserved_dtype,
                seed=string_to_seed(key),
            )
            if pin_output:
                out_weight, _ = _pin_tensor(out_weight)
            from backend import utils as backend_utils
            backend_utils.set_attr_param(model, key, out_weight)
        else:
            if pin_output:
                out_weight, _ = _pin_tensor(out_weight)
            set_func(out_weight, inplace_update=False, seed=string_to_seed(key))

    @classmethod
    def _compile_single_key_standalone(
        cls,
        base_state_dict: dict[str, torch.Tensor],
        key: str,
        patches: list[Any],
        target_device: torch.device,
    ):
        weight = base_state_dict[key]
        preserved_dtype = weight.dtype
        temp_weight = weight.to(device=target_device, dtype=preserved_dtype, copy=False)
        out_weight = cls._patch_single_layer_worker(key, temp_weight, patches, preserved_dtype)
        out_weight = backend_float_ops.stochastic_rounding(
            out_weight,
            preserved_dtype,
            seed=string_to_seed(key),
        )
        base_state_dict[key] = out_weight

    @classmethod
    def _patch_single_layer_worker(
        cls,
        key: str,
        base_tensor: torch.Tensor,
        patches: list[Any],
        preserved_dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Isolated worker that safely merges patches into a single UNet layer.
        Operates strictly on CPU, without VRAM interaction, and discards all
        intermediaries immediately upon completion.
        """
        weight = base_tensor
        for patch in patches:
            strength_patch, patch_payload, strength_model, offset, function = patch

            if offset is not None:
                weight = weight.narrow(offset[0], offset[1], offset[2])

            if strength_model != 1.0:
                weight.mul_(strength_model)

            intermediate_dtype = preserved_dtype
            function = function or _identity

            if isinstance(patch_payload, weight_adapter.LoRAAdapter):
                v = patch_payload.weights
                mat1 = _resolve_tensor(v[0], weight.device, intermediate_dtype)
                mat2 = _resolve_tensor(v[1], weight.device, intermediate_dtype)
                alpha_val = _resolve_scalar(v[2])
                mid = _resolve_tensor(v[3], weight.device, intermediate_dtype)
                dora_scale = _resolve_tensor(v[4], weight.device, intermediate_dtype)
                reshape = v[5]

                if reshape is not None:
                    weight = pad_tensor_to_shape(weight, reshape)

                if alpha_val is not None:
                    alpha = alpha_val / mat2.shape[0]
                else:
                    alpha = 1.0

                if mid is not None:
                    final_shape = [mat2.shape[1], mat2.shape[0], mid.shape[2], mid.shape[3]]
                    mat2 = torch.mm(mat2.transpose(0, 1).flatten(start_dim=1),
                                    mid.transpose(0, 1).flatten(start_dim=1)).reshape(final_shape).transpose(0, 1)

                if dora_scale is None and weight.ndim in (2, 4) and mat1.ndim == 2 and mat2.ndim in (2, 4):
                    m1_flat = mat1.flatten(start_dim=1)
                    m2_flat = mat2.flatten(start_dim=1)
                    weight_view = weight.view(weight.shape[0], -1)
                    weight_view.addmm_(m1_flat, m2_flat, alpha=strength_patch * alpha)
                else:
                    lora_diff = torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1)).reshape(weight.shape)
                    if dora_scale is not None:
                        weight = weight_decompose(dora_scale, weight, lora_diff, alpha, strength_patch, intermediate_dtype, function)
                    else:
                        weight.add_(function((strength_patch * alpha) * lora_diff).to(weight.dtype))

            elif isinstance(patch_payload, weight_adapter.LoHaAdapter):
                v = patch_payload.weights
                w1a = _resolve_tensor(v[0], weight.device, intermediate_dtype)
                w1b = _resolve_tensor(v[1], weight.device, intermediate_dtype)
                alpha_val = _resolve_scalar(v[2])
                w2a = _resolve_tensor(v[3], weight.device, intermediate_dtype)
                w2b = _resolve_tensor(v[4], weight.device, intermediate_dtype)
                t1 = _resolve_tensor(v[5], weight.device, intermediate_dtype)
                t2 = _resolve_tensor(v[6], weight.device, intermediate_dtype)
                dora_scale = _resolve_tensor(v[7], weight.device, intermediate_dtype)

                if alpha_val is not None:
                    alpha = alpha_val / w1b.shape[0]
                else:
                    alpha = 1.0

                if t1 is not None and t2 is not None:
                    m1 = torch.einsum('i j k l, j r, i p -> p r k l', t1, w1b, w1a)
                    m2 = torch.einsum('i j k l, j r, i p -> p r k l', t2, w2b, w2a)
                else:
                    m1 = torch.mm(w1a, w1b)
                    m2 = torch.mm(w2a, w2b)

                lora_diff = (m1 * m2).reshape(weight.shape)
                if dora_scale is not None:
                    weight = weight_decompose(dora_scale, weight, lora_diff, alpha, strength_patch, intermediate_dtype, function)
                else:
                    weight.add_(function((strength_patch * alpha) * lora_diff).to(weight.dtype))

            elif isinstance(patch_payload, weight_adapter.LoKrAdapter):
                v = patch_payload.weights
                w1 = _resolve_tensor(v[0], weight.device, intermediate_dtype)
                w2 = _resolve_tensor(v[1], weight.device, intermediate_dtype)
                alpha_val = _resolve_scalar(v[2])
                w1_a = _resolve_tensor(v[3], weight.device, intermediate_dtype)
                w1_b = _resolve_tensor(v[4], weight.device, intermediate_dtype)
                w2_a = _resolve_tensor(v[5], weight.device, intermediate_dtype)
                w2_b = _resolve_tensor(v[6], weight.device, intermediate_dtype)
                t2 = _resolve_tensor(v[7], weight.device, intermediate_dtype)
                dora_scale = _resolve_tensor(v[8], weight.device, intermediate_dtype)
                dim = None

                if w1 is None:
                    dim = w1_b.shape[0]
                    w1 = torch.mm(w1_a, w1_b)

                if w2 is None:
                    dim = w2_b.shape[0]
                    if t2 is None:
                        w2 = torch.mm(w2_a, w2_b)
                    else:
                        w2 = torch.einsum('i j k l, j r, i p -> p r k l', t2, w2_b, w2_a)

                if len(w2.shape) == 4:
                    w1 = w1.unsqueeze(2).unsqueeze(2)
                if alpha_val is not None and dim is not None:
                    alpha = alpha_val / dim
                else:
                    alpha = 1.0

                lora_diff = torch.kron(w1, w2).reshape(weight.shape)
                if dora_scale is not None:
                    weight = weight_decompose(dora_scale, weight, lora_diff, alpha, strength_patch, intermediate_dtype, function)
                else:
                    weight.add_(function((strength_patch * alpha) * lora_diff).to(weight.dtype))

            elif isinstance(patch_payload, weight_adapter.GLoRAAdapter):
                v = patch_payload.weights
                a1 = _resolve_tensor(v[0], weight.device, intermediate_dtype).flatten(start_dim=1)
                a2 = _resolve_tensor(v[1], weight.device, intermediate_dtype).flatten(start_dim=1)
                b1 = _resolve_tensor(v[2], weight.device, intermediate_dtype).flatten(start_dim=1)
                b2 = _resolve_tensor(v[3], weight.device, intermediate_dtype).flatten(start_dim=1)
                alpha_val = _resolve_scalar(v[4])
                dora_scale = _resolve_tensor(v[5], weight.device, intermediate_dtype)

                if alpha_val is not None:
                    alpha = alpha_val / v[0].shape[0]
                else:
                    alpha = 1.0

                lora_diff = (torch.mm(b2, b1) + torch.mm(torch.mm(weight.flatten(start_dim=1), a2), a1)).reshape(weight.shape)
                if dora_scale is not None:
                    weight = weight_decompose(dora_scale, weight, lora_diff, alpha, strength_patch, intermediate_dtype, function)
                else:
                    weight.add_(function((strength_patch * alpha) * lora_diff).to(weight.dtype))

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
                    w1 = _resolve_tensor(v[0], weight.device, intermediate_dtype)
                    if strength_patch != 0.0:
                        if w1.shape == weight.shape:
                            weight.add_(w1.to(weight.dtype), alpha=strength_patch)
                elif patch_type == "lora":
                    mat1 = _resolve_tensor(v[0], weight.device, intermediate_dtype)
                    mat2 = _resolve_tensor(v[1], weight.device, intermediate_dtype)
                    alpha_val = _resolve_scalar(v[2])
                    mid = _resolve_tensor(v[3], weight.device, intermediate_dtype)
                    if alpha_val is not None:
                        alpha = alpha_val / mat2.shape[0]
                    else:
                        alpha = 1.0
                    if mid is not None:
                        final_shape = [mat2.shape[1], mat2.shape[0], mid.shape[2], mid.shape[3]]
                        mat2 = torch.mm(mat2.transpose(0, 1).flatten(start_dim=1),
                                        mid.transpose(0, 1).flatten(start_dim=1)).reshape(final_shape).transpose(0, 1)
                    if weight.ndim in (2, 4) and mat1.ndim == 2 and mat2.ndim in (2, 4):
                        m1_flat = mat1.flatten(start_dim=1)
                        m2_flat = mat2.flatten(start_dim=1)
                        weight_view = weight.view(weight.shape[0], -1)
                        weight_view.addmm_(m1_flat, m2_flat, alpha=strength_patch * alpha)
                    else:
                        lora_diff = torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1)).reshape(weight.shape)
                        weight.add_((strength_patch * alpha) * lora_diff.to(weight.dtype))
                elif patch_type == "fooocus":
                    w1 = _resolve_tensor(v[0], weight.device, intermediate_dtype)
                    w_min = _resolve_tensor(v[1], weight.device, intermediate_dtype)
                    w_max = _resolve_tensor(v[2], weight.device, intermediate_dtype)
                    w1 = (w1 / 255.0) * (w_max - w_min) + w_min
                    if strength_patch != 0.0 and w1.shape == weight.shape:
                        weight.add_(w1.to(weight.dtype), alpha=strength_patch)
                elif patch_type == "set":
                    w1 = _resolve_tensor(v[0], weight.device, weight.dtype)
                    weight.copy_(w1)

        return base_tensor
