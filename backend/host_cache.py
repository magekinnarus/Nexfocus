import ctypes
import os
import sys
import torch
import logging

logger = logging.getLogger(__name__)

_FLUSH_FN = None
_RESOLVED = False

def _resolve_flush_fn():
    global _FLUSH_FN, _RESOLVED
    if _RESOLVED:
        return _FLUSH_FN
    
    _RESOLVED = True
    if not torch.cuda.is_available():
        return None

    try:
        torch_dir = os.path.dirname(torch.__file__)
        system_name = sys.platform
        
        if system_name.startswith("win"):
            lib_path = os.path.join(torch_dir, "lib", "torch_cuda.dll")
            if not os.path.exists(lib_path):
                lib_path = os.path.join(torch_dir, "lib", "c10_cuda.dll")
            
            if os.path.exists(lib_path):
                lib = ctypes.CDLL(lib_path)
                symbol_name = "?CachingHostAllocator_emptyCache@cuda@at@@YAXXZ"
                if hasattr(lib, symbol_name):
                    fn = getattr(lib, symbol_name)
                    fn.restype = None
                    fn.argtypes = []
                    _FLUSH_FN = fn
                    logger.info("[Nex-Memory] Successfully bound Windows CachingHostAllocator_emptyCache")
                    return _FLUSH_FN
                    
        elif system_name.startswith("linux"):
            lib_path = os.path.join(torch_dir, "lib", "libtorch_cuda.so")
            if not os.path.exists(lib_path):
                lib_path = os.path.join(torch_dir, "lib", "libc10_cuda.so")
            
            if os.path.exists(lib_path):
                lib = ctypes.CDLL(lib_path)
                symbol_name = "_ZN2at4cuda31CachingHostAllocator_emptyCacheEv"
                if hasattr(lib, symbol_name):
                    fn = getattr(lib, symbol_name)
                    fn.restype = None
                    fn.argtypes = []
                    _FLUSH_FN = fn
                    logger.info("[Nex-Memory] Successfully bound Linux CachingHostAllocator_emptyCache")
                    return _FLUSH_FN
                else:
                    # Fallback just in case
                    fallback_name = "_ZN2at4cuda18CachingHostAllocator10emptyCacheEv"
                    if hasattr(lib, fallback_name):
                        fn = getattr(lib, fallback_name)
                        fn.restype = None
                        fn.argtypes = []
                        _FLUSH_FN = fn
                        logger.info("[Nex-Memory] Successfully bound Linux fallback CachingHostAllocator::emptyCache")
                        return _FLUSH_FN

    except Exception:
        logger.warning("[Nex-Memory] Failed to bind CachingHostAllocator_emptyCache symbol via ctypes", exc_info=True)
        
    return _FLUSH_FN

def flush_pinned_host_cache() -> bool:
    """Flush PyTorch's CachingHostAllocator pinned-memory cache.
    
    Returns True if the cache was successfully flushed, False otherwise.
    Safe to call even when CUDA is unavailable.
    """
    fn = _resolve_flush_fn()
    if fn is None:
        return False
    try:
        fn()
        logger.debug("[Nex-Memory] Flushed pinned host allocator cache")
        return True
    except Exception:
        logger.warning("[Nex-Memory] Failed to execute CachingHostAllocator_emptyCache", exc_info=True)
        return False
