import psutil
import logging
from enum import Enum
import torch
import platform
import ctypes
import os
from typing import Any, Tuple
import backend.memory_governor as memory_governor

class VRAMState(Enum):
    DISABLED = 0    # No vram present: no need to move models to vram
    NO_VRAM = 1     # Very low vram: enable all the options to save vram
    LOW_VRAM = 2
    NORMAL_VRAM = 3
    HIGH_VRAM = 4
    SHARED = 5      # No dedicated vram: memory shared between CPU and GPU but models still need to be moved between both.

class CPUState(Enum):
    GPU = 0
    CPU = 1
    MPS = 2

class ResourcesConfig:
    def __init__(self):
        self.deterministic = False
        self.directml = None
        self.cpu = False
        self.lowvram = False
        self.novram = False
        self.highvram = False
        self.gpu_only = False
        self.reserve_vram = None
        self.disable_smart_memory = False
        self.disable_xformers = False
        self.use_pytorch_cross_attention = False
        self.use_split_cross_attention = False
        self.use_quad_cross_attention = False
        self.supports_fp8_compute = False
        self.fp32_unet = False
        self.fp64_unet = False
        self.bf16_unet = False
        self.fp16_unet = False
        self.fp8_e4m3fn_unet = False
        self.fp8_e5m2_unet = False
        self.fp8_e8m0fnu_unet = False
        self.fp8_e4m3fn_text_enc = False
        self.fp8_e5m2_text_enc = False
        self.fp16_text_enc = False
        self.bf16_text_enc = False
        self.fp32_text_enc = False
        self.cpu_vae = False
        self.fp16_vae = False
        self.bf16_vae = False
        self.fp32_vae = False
        self.force_upcast_attention = False
        self.async_offload = False
        self.force_channels_last = False
        self.use_sage_attention = False
        self.use_flash_attention = False
        self.force_fp16 = False
        self.force_fp32 = False
        self.fast = []
        self.disable_ipex_optimize = False

config = ResourcesConfig()

# Determine VRAM State
vram_state = VRAMState.NORMAL_VRAM
set_vram_to = VRAMState.NORMAL_VRAM
cpu_state = CPUState.GPU
total_vram = 0

def _memory_mb(value):
    return float(value) / (1024 ** 2)

def get_supported_float8_types():
    float8_types = []
    try:
        float8_types.append(torch.float8_e4m3fn)
    except:
        pass
    try:
        float8_types.append(torch.float8_e4m3fnuz)
    except:
        pass
    try:
        float8_types.append(torch.float8_e5m2)
    except:
        pass
    try:
        float8_types.append(torch.float8_e5m2fnuz)
    except:
        pass
    try:
        float8_types.append(torch.float8_e8m0fnu)
    except:
        pass
    return float8_types

FLOAT8_TYPES = get_supported_float8_types()

xpu_available = False
torch_version = ""
try:
    torch_version = torch.version.__version__
    temp = torch_version.split(".")
    torch_version_numeric = (int(temp[0]), int(temp[1]))
    xpu_available = (torch_version_numeric[0] < 2 or (torch_version_numeric[0] == 2 and torch_version_numeric[1] <= 4)) and torch.xpu.is_available()
except:
    pass

lowvram_available = True

# We'll set this later after config is potentially updated
def apply_config():
    global vram_state, set_vram_to, cpu_state, lowvram_available, directml_enabled, directml_device
    
    if config.deterministic:
        logging.info("Using deterministic algorithms for pytorch")
        torch.use_deterministic_algorithms(True, warn_only=True)

    directml_enabled = False
    if config.directml is not None:
        import torch_directml
        directml_enabled = True
        device_index = config.directml
        if device_index < 0:
            directml_device = torch_directml.device()
        else:
            directml_device = torch_directml.device(device_index)
        logging.info("Using directml with device: {}".format(torch_directml.device_name(device_index)))
        lowvram_available = False 

    if config.cpu:
        cpu_state = CPUState.CPU

    if config.lowvram:
        set_vram_to = VRAMState.LOW_VRAM
        lowvram_available = True
        logging.warning('[Nex-Memory] --lowvram compatibility mode is active; prefer memory_environment_profile for stage-aware residency.')
    elif config.novram:
        set_vram_to = VRAMState.NO_VRAM
        logging.warning('[Nex-Memory] --novram compatibility mode is active; prefer memory_environment_profile for stage-aware residency.')
    elif config.highvram or config.gpu_only:
        vram_state = VRAMState.HIGH_VRAM

    if lowvram_available:
        if set_vram_to in (VRAMState.LOW_VRAM, VRAMState.NO_VRAM):
            vram_state = set_vram_to

    if cpu_state != CPUState.GPU:
        vram_state = VRAMState.DISABLED

    if cpu_state == CPUState.MPS:
        vram_state = VRAMState.SHARED

apply_config()

ipex = None
try:
    import intel_extension_for_pytorch as ipex_mod
    ipex = ipex_mod
    _ = torch.xpu.device_count()
    xpu_available = xpu_available or torch.xpu.is_available()
except:
    xpu_available = xpu_available or (hasattr(torch, "xpu") and torch.xpu.is_available())

try:
    if torch.backends.mps.is_available():
        cpu_state = CPUState.MPS
        import torch.mps
except:
    pass

try:
    import torch_npu  # noqa: F401
    _ = torch.npu.device_count()
    npu_available = torch.npu.is_available()
except:
    npu_available = False

try:
    import torch_mlu  # noqa: F401
    _ = torch.mlu.device_count()
    mlu_available = torch.mlu.is_available()
except:
    mlu_available = False

try:
    ixuca_available = hasattr(torch, "corex")
except:
    ixuca_available = False

def is_intel_xpu():
    global cpu_state
    global xpu_available
    if cpu_state == CPUState.GPU:
        if xpu_available:
            return True
    return False

def is_ascend_npu():
    global npu_available
    if npu_available:
        return True
    return False

def is_mlu():
    global mlu_available
    if mlu_available:
        return True
    return False

def is_ixuca():
    global ixuca_available
    if ixuca_available:
        return True
    return False

def get_torch_device():
    global directml_enabled
    global cpu_state
    if directml_enabled:
        global directml_device
        return directml_device
    if cpu_state == CPUState.MPS:
        return torch.device("mps")
    if cpu_state == CPUState.CPU:
        return torch.device("cpu")
    else:
        if is_intel_xpu():
            return torch.device("xpu", torch.xpu.current_device())
        elif is_ascend_npu():
            return torch.device("npu", torch.npu.current_device())
        elif is_mlu():
            return torch.device("mlu", torch.mlu.current_device())
        else:
            return torch.device(torch.cuda.current_device())

def get_total_memory(dev=None, torch_total_too=False):
    global directml_enabled
    if dev is None:
        dev = get_torch_device()

    if hasattr(dev, 'type') and (dev.type == 'cpu' or dev.type == 'mps'):
        mem_total = psutil.virtual_memory().total
        mem_total_torch = mem_total
    else:
        if directml_enabled:
            mem_total = 1024 * 1024 * 1024 #TODO
            mem_total_torch = mem_total
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_reserved = stats['reserved_bytes.all.current']
            mem_total_xpu = torch.xpu.get_device_properties(dev).total_memory
            mem_total_torch = mem_reserved
            mem_total = mem_total_xpu
        elif is_ascend_npu():
            stats = torch.npu.memory_stats(dev)
            mem_reserved = stats['reserved_bytes.all.current']
            _, mem_total_npu = torch.npu.mem_get_info(dev)
            mem_total_torch = mem_reserved
            mem_total = mem_total_npu
        elif is_mlu():
            stats = torch.mlu.memory_stats(dev)
            mem_reserved = stats['reserved_bytes.all.current']
            _, mem_total_mlu = torch.mlu.mem_get_info(dev)
            mem_total_torch = mem_reserved
            mem_total = mem_total_mlu
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_reserved = stats['reserved_bytes.all.current']
            _, mem_total_cuda = torch.cuda.mem_get_info(dev)
            mem_total_torch = mem_reserved
            mem_total = mem_total_cuda

    if torch_total_too:
        return (mem_total, mem_total_torch)
    else:
        return mem_total

def get_free_memory(dev=None, torch_free_too=False):
    global directml_enabled
    if dev is None:
        dev = get_torch_device()

    if hasattr(dev, 'type') and (dev.type == 'cpu' or dev.type == 'mps'):
        mem_free_total = psutil.virtual_memory().available
        mem_free_torch = mem_free_total
    else:
        if directml_enabled:
            mem_free_total = 1024 * 1024 * 1024 #TODO
            mem_free_torch = mem_free_total
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_xpu = torch.xpu.get_device_properties(dev).total_memory - mem_reserved
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_xpu + mem_free_torch
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_cuda + mem_free_torch

    if torch_free_too:
        return (mem_free_total, mem_free_torch)
    else:
        return mem_free_total

def mac_version():
    try:
        return tuple(int(n) for n in platform.mac_ver()[0].split("."))
    except:
        return None

total_vram = float(get_total_memory(get_torch_device())) / (1024 * 1024)
total_ram = psutil.virtual_memory().total / (1024 * 1024)
logging.info("Total VRAM {:0.0f} MB, total RAM {:0.0f} MB".format(total_vram, total_ram))

try:
    OOM_EXCEPTION = torch.cuda.OutOfMemoryError
except:
    OOM_EXCEPTION = Exception

XFORMERS_VERSION = ""
XFORMERS_ENABLED_VAE = True
if config.disable_xformers:
    XFORMERS_IS_AVAILABLE = False
else:
    try:
        import xformers
        import xformers.ops
        XFORMERS_IS_AVAILABLE = True
        try:
            XFORMERS_IS_AVAILABLE = xformers._has_cpp_library
        except:
            pass
        try:
            XFORMERS_VERSION = xformers.version.__version__
        except:
            pass
    except:
        XFORMERS_IS_AVAILABLE = False

def is_nvidia():
    global cpu_state
    if cpu_state == CPUState.GPU:
        if torch.version.cuda:
            return True
    return False

def is_amd():
    global cpu_state
    if cpu_state == CPUState.GPU:
        if torch.version.hip:
            return True
    return False

MIN_WEIGHT_MEMORY_RATIO = 0.4
if is_nvidia():
    MIN_WEIGHT_MEMORY_RATIO = 0.0

ENABLE_PYTORCH_ATTENTION = False
if config.use_pytorch_cross_attention:
    ENABLE_PYTORCH_ATTENTION = True
    XFORMERS_IS_AVAILABLE = False

try:
    if is_nvidia():
        if torch_version_numeric[0] >= 2:
            if ENABLE_PYTORCH_ATTENTION == False and config.use_split_cross_attention == False and config.use_quad_cross_attention == False:
                ENABLE_PYTORCH_ATTENTION = True
    if is_intel_xpu() or is_ascend_npu() or is_mlu() or is_ixuca():
        if config.use_split_cross_attention == False and config.use_quad_cross_attention == False:
            ENABLE_PYTORCH_ATTENTION = True
except:
    pass

if ENABLE_PYTORCH_ATTENTION:
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

def get_torch_device_name(device):
    if hasattr(device, 'type'):
        if device.type == "cuda":
            try:
                allocator_backend = torch.cuda.get_allocator_backend()
            except:
                allocator_backend = ""
            return "{} {} : {}".format(device, torch.cuda.get_device_name(device), allocator_backend)
        elif device.type == "xpu":
            return "{} {}".format(device, torch.xpu.get_device_name(device))
        else:
            return "{}".format(device.type)
    elif is_intel_xpu():
        return "{} {}".format(device, torch.xpu.get_device_name(device))
    elif is_ascend_npu():
        return "{} {}".format(device, torch.npu.get_device_name(device))
    elif is_mlu():
        return "{} {}".format(device, torch.mlu.get_device_name(device))
    else:
        return "CUDA {}: {}".format(device, torch.cuda.get_device_name(device))

def is_directml_enabled():
    global directml_enabled
    if 'directml_enabled' in globals() and directml_enabled:
        return True
    return False

def device_supports_non_blocking(device):
    if is_device_mps(device):
        return False
    if is_intel_xpu():
        return True
    if config.deterministic:
        return False
    if is_directml_enabled():
        return False
    return True

def supports_fp8_compute(device=None):
    if getattr(config, 'supports_fp8_compute', False):
        return True

    if device is not None and not isinstance(device, torch.device):
        device = torch.device(device)
    if device is not None and device.type != "cuda":
        return False

    if not is_nvidia():
        return False

    props = torch.cuda.get_device_properties(device)
    if props.major >= 9:
        return True
    if props.major < 8:
        return False
    if props.minor < 9:
        return False

    if torch_version_numeric < (2, 3):
        return False

    if any(platform.win32_ver()):
        if torch_version_numeric < (2, 4):
            return False

    return True

def is_device_type(device, type):
    if hasattr(device, 'type'):
        return device.type == type
    return False

def is_device_cpu(device): return is_device_type(device, 'cpu')
def is_device_mps(device): return is_device_type(device, 'mps')
def is_device_xpu(device): return is_device_type(device, 'xpu')
def is_device_cuda(device): return is_device_type(device, 'cuda')

def should_use_fp16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    if device is not None and is_device_cpu(device): return False
    if config.force_fp16: return True
    if config.force_fp32: return False
    if cpu_state == CPUState.CPU: return False
    return True

def should_use_bf16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    if device is not None and is_device_cpu(device): return False
    if config.force_fp32: return False
    return torch.cuda.is_bf16_supported()

def unet_manual_cast(weight_dtype, inference_device, supported_dtypes=[torch.float16, torch.bfloat16, torch.float32]):
    if weight_dtype == torch.float32 or weight_dtype == torch.float64:
        return None

    fp16_supported = should_use_fp16(inference_device, prioritize_performance=False)
    if fp16_supported and weight_dtype == torch.float16:
        return None

    bf16_supported = should_use_bf16(inference_device)
    if bf16_supported and weight_dtype == torch.bfloat16:
        return None

    fp16_supported = should_use_fp16(inference_device, prioritize_performance=True)
    for dt in supported_dtypes:
        if dt == torch.float16 and fp16_supported:
            return torch.float16
        if dt == torch.bfloat16 and bf16_supported:
            return torch.bfloat16

    return torch.float32

def text_encoder_dtype(device=None):
    from ldm_patched.modules.args_parser import args
    if args.clip_in_fp8_e4m3fn:
        return torch.float8_e4m3fn
    elif args.clip_in_fp8_e5m2:
        return torch.float8_e5m2
    elif args.clip_in_fp16:
        return torch.float16
    elif args.clip_in_fp32:
        return torch.float32

    if is_device_cpu(device):
        return torch.float16
    return torch.float16

def supports_cast(device, dtype):
    if dtype == torch.float32:
        return True
    if dtype == torch.float16:
        return True
    if directml_enabled:
        return False
    if dtype == torch.bfloat16:
        return True
    if is_device_mps(device):
        return False
    if dtype == torch.float8_e4m3fn:
        return True
    if dtype == torch.float8_e5m2:
        return True
    return False

def pick_weight_dtype(dtype, fallback_dtype, device=None):
    from backend.utils import dtype_size
    if dtype is None:
        dtype = fallback_dtype
    elif dtype_size(dtype) > dtype_size(fallback_dtype):
        dtype = fallback_dtype

    if not supports_cast(device, dtype):
        dtype = fallback_dtype

    return dtype

def extra_reserved_memory():
    res = 400 * 1024 * 1024
    if any(platform.win32_ver()):
        res = 600 * 1024 * 1024
        if total_vram < 4096:
            res = 250 * 1024 * 1024
        elif total_vram > (15 * 1024):
            res += 100 * 1024 * 1024
    if config.reserve_vram is not None:
        res = config.reserve_vram * 1024 * 1024 * 1024
    return res

def minimum_inference_memory():
    if total_vram < 4096:
        return (1024 * 1024 * 1024) * 0.4 + extra_reserved_memory()
    return (1024 * 1024 * 1024) * 0.8 + extra_reserved_memory()

def unet_dtype(device=None, model_params=0, supported_dtypes=None, weight_dtype=None):
    if supported_dtypes is None:
        supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]
    if model_params < 0:
        model_params = 1000000000000000000000
    
    if config.bf16_unet:
        return torch.bfloat16
    if config.fp16_unet:
        return torch.float16
    if config.fp8_e4m3fn_unet:
        return torch.float8_e4m3fn
    if config.fp8_e5m2_unet:
        return torch.float8_e5m2
    
    fp8_dtype = None
    if weight_dtype in FLOAT8_TYPES:
        fp8_dtype = weight_dtype

    if fp8_dtype is not None:
        if supports_fp8_compute(device):
            return fp8_dtype

        free_model_memory = maximum_vram_for_weights(device)
        if model_params * 2 > free_model_memory:
            return fp8_dtype

    if torch.float16 in supported_dtypes and should_use_fp16(device=device, model_params=model_params):
        return torch.float16
    if torch.bfloat16 in supported_dtypes and should_use_bf16(device, model_params=model_params):
        return torch.bfloat16

    for dt in supported_dtypes:
        if dt == torch.float16 and should_use_fp16(device=device, model_params=model_params, manual_cast=True):
            if torch.float16 in supported_dtypes:
                return torch.float16
        if dt == torch.bfloat16 and should_use_bf16(device, model_params=model_params, manual_cast=True):
            if torch.bfloat16 in supported_dtypes:
                return torch.bfloat16

    return torch.float32

def maximum_vram_for_weights(device=None):
    return (float(get_total_memory(device)) * 0.88 - minimum_inference_memory())

def vae_device():
    if config.cpu_vae:
        return torch.device("cpu")
    return get_torch_device()

def vae_offload_device():
    return torch.device("cpu")

def unet_offload_device():
    return torch.device("cpu")

def text_encoder_load_device():
    return torch.device("cpu")

def text_encoder_offload_device():
    return torch.device("cpu")

def intermediate_device():
    from ldm_patched.modules.args_parser import args
    if getattr(args, "always_gpu", False):
        return get_torch_device()
    else:
        return torch.device("cpu")

def soft_empty_cache(force=False):
    if not memory_governor.should_flush_cache(force=force):
        return

    if cpu_state == CPUState.MPS:
        torch.mps.empty_cache()
    elif is_intel_xpu():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    memory_governor.note_cache_flush()

def _classify_model_role(model_patcher):
    patcher_name = type(model_patcher).__name__
    model_obj = getattr(model_patcher, "model", None)
    model_name = type(model_obj).__name__ if model_obj is not None else "UnknownModel"
    model_name_lower = model_name.lower()

    role = "model"
    if model_obj is not None:
        if hasattr(model_obj, "diffusion_model"):
            role = "unet"
        elif 'controlnet' in model_name_lower:
            role = "controlnet"
        elif model_name == "CLIP" or hasattr(model_obj, "tokenizer") or "clip" in model_name_lower:
            role = "clip"
        elif (
            model_name == "VAE"
            or hasattr(model_obj, "first_stage_model")
            or "autoencoder" in model_name_lower
            or "autoencoding" in model_name_lower
        ):
            role = "vae"

    return role, patcher_name, model_name

def _describe_model_for_logs(model_patcher):
    role, patcher_name, model_name = _classify_model_role(model_patcher)
    return f"{role}:{patcher_name}/{model_name}"

def _residency_plan_for_phase(target_phase=None, task=None):
    phase_name = memory_governor.normalize_phase(target_phase) if target_phase is not None else memory_governor.current_phase()
    return memory_governor.plan_for_task(task=task, phase=phase_name)

def _emit_residency_log(prefix, *, plan, notes=None, role=None, item=None, action=None):
    payload = {
        'profile': plan.notes.get('profile'),
        'phase': plan.notes.get('phase'),
        'pinned': ','.join(plan.pinned) or '-',
        'warm': ','.join(plan.warm) or '-',
        'evictable': ','.join(plan.evictable) or '-',
    }
    if role is not None:
        payload['role'] = role
    if item is not None:
        payload['item'] = item
    if action is not None:
        payload['action'] = action
    if notes:
        payload.update(notes)

    extras = ' '.join(f"{key}={value}" for key, value in payload.items())
    message = f"[Nex-Residency] {prefix} {extras}"
    print(message)
    logging.info(message)


SDXL_RESIDENCY_CLASS_FULL = "full_resident"
SDXL_RESIDENCY_CLASS_UNIFIED_STREAMING = "unified_streaming"


def normalize_sdxl_residency_class(residency_class=None):
    if residency_class is not None:
        normalized = str(residency_class).strip().lower()
        if normalized in {
            SDXL_RESIDENCY_CLASS_FULL,
            SDXL_RESIDENCY_CLASS_UNIFIED_STREAMING,
        }:
            return normalized
    return SDXL_RESIDENCY_CLASS_FULL


def get_component_plan(role: str, policy: Any = None) -> Tuple[torch.device, str]:
    if policy is None:
        from backend import sdxl_runtime_policy
        policy = sdxl_runtime_policy.resolve_sdxl_execution_policy(
            architecture="sdxl",
            base_model_name=None,
        )
    
    # Defaults
    dev = torch.device("cpu")
    mode = "offloaded"
    
    if policy is not None:
        if role == "unet":
            execution_mode = getattr(policy, "execution_mode", None)
            if execution_mode is not None:
                is_streaming = (execution_mode == "streaming")
            else:
                normalized_residency_class = normalize_sdxl_residency_class(getattr(policy, "residency_class", None))
                is_streaming = normalized_residency_class == SDXL_RESIDENCY_CLASS_UNIFIED_STREAMING
            if is_streaming:
                return torch.device("cpu"), "cpu_resident"
            return get_torch_device(), "gpu_resident"
        
        if role == "clip":
            prefer_clip_gpu = getattr(policy, "prefer_clip_gpu", False)
            if not prefer_clip_gpu:
                from backend import sdxl_runtime_policy
                clip_res_mode = getattr(policy, "clip_residency_mode", None)
                prefer_clip_gpu = (clip_res_mode == sdxl_runtime_policy.CLIP_RESIDENCY_GPU_RESIDENT)
            if prefer_clip_gpu:
                return get_torch_device(), "gpu_resident"
            return torch.device("cpu"), "cpu_resident"
            
        if role == "vae":
            prefer_gpu_vae = getattr(policy, "prefer_gpu_vae_encode", False)
            if not prefer_gpu_vae:
                from backend import sdxl_runtime_policy
                vae_enc_mode = getattr(policy, "vae_encode_mode", None)
                prefer_gpu_vae = vae_enc_mode in {
                    sdxl_runtime_policy.VAE_ENCODE_GPU_PREFERRED,
                    sdxl_runtime_policy.VAE_POSTURE_GPU_RESIDENT,
                }
            if prefer_gpu_vae:
                # SDXL VAE is CPU-cached and activated opportunistically on GPU.
                # Do not map policy preference back to resident GPU placement.
                return torch.device("cpu"), "cpu_resident"
            return torch.device("cpu"), "cpu_resident"
            
    return dev, mode


def model_reconciliation_signature(model_patcher):
    target_uuid = getattr(model_patcher, "patches_uuid", None)
    if target_uuid is not None:
        return str(target_uuid)

    model_obj = getattr(model_patcher, "model", None)
    return str(getattr(model_obj, "current_weight_patches_uuid", None))
