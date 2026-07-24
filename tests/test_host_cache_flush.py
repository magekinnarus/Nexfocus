from unittest.mock import patch, MagicMock
import sys
import os
import pytest

from backend.host_cache import _resolve_flush_fn, flush_pinned_host_cache, _RESOLVED

@pytest.fixture(autouse=True)
def reset_resolved_flag():
    # Reset the global cached state between tests
    import backend.host_cache as hc
    hc._RESOLVED = False
    hc._FLUSH_FN = None


def test_resolve_flush_fn_no_cuda():
    with patch("backend.host_cache.torch.cuda.is_available", return_value=False):
        fn = _resolve_flush_fn()
        assert fn is None


def test_resolve_flush_fn_windows_success():
    mock_lib = MagicMock()
    mock_fn = MagicMock()
    setattr(mock_lib, "?CachingHostAllocator_emptyCache@cuda@at@@YAXXZ", mock_fn)

    with patch("backend.host_cache.torch.cuda.is_available", return_value=True), \
         patch("backend.host_cache.sys.platform", "win32"), \
         patch("backend.host_cache.os.path.exists", return_value=True), \
         patch("backend.host_cache.ctypes.CDLL", return_value=mock_lib):
        
        fn = _resolve_flush_fn()
        assert fn is mock_fn


def test_resolve_flush_fn_linux_success():
    mock_lib = MagicMock()
    mock_fn = MagicMock()
    setattr(mock_lib, "_ZN2at4cuda31CachingHostAllocator_emptyCacheEv", mock_fn)

    with patch("backend.host_cache.torch.cuda.is_available", return_value=True), \
         patch("backend.host_cache.sys.platform", "linux"), \
         patch("backend.host_cache.os.path.exists", return_value=True), \
         patch("backend.host_cache.ctypes.CDLL", return_value=mock_lib):
        
        fn = _resolve_flush_fn()
        assert fn is mock_fn


def test_resolve_flush_fn_linux_fallback_success():
    mock_lib = MagicMock()
    mock_fn = MagicMock()
    # No standard name, only fallback name
    delattr(mock_lib, "_ZN2at4cuda31CachingHostAllocator_emptyCacheEv")
    setattr(mock_lib, "_ZN2at4cuda18CachingHostAllocator10emptyCacheEv", mock_fn)

    with patch("backend.host_cache.torch.cuda.is_available", return_value=True), \
         patch("backend.host_cache.sys.platform", "linux"), \
         patch("backend.host_cache.os.path.exists", return_value=True), \
         patch("backend.host_cache.ctypes.CDLL", return_value=mock_lib):
        
        fn = _resolve_flush_fn()
        assert fn is mock_fn


def test_flush_pinned_host_cache_calls_underlying_function():
    mock_fn = MagicMock()
    with patch("backend.host_cache._resolve_flush_fn", return_value=mock_fn):
        res = flush_pinned_host_cache()
        assert res is True
        mock_fn.assert_called_once()


def test_flush_pinned_host_cache_handles_call_exception_gracefully():
    mock_fn = MagicMock(side_effect=RuntimeError("Some ctypes error"))
    with patch("backend.host_cache._resolve_flush_fn", return_value=mock_fn):
        res = flush_pinned_host_cache()
        assert res is False
        mock_fn.assert_called_once()
