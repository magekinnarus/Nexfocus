from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _clear_nex_runtime_caches():
    try:
        from backend import conditioning, sdxl_unified_runtime

        conditioning.clear_prompt_conditioning_cache()
        sdxl_unified_runtime.clear_unified_sdxl_runtime_component_cache()
    except BaseException:
        pass

    yield

    try:
        from backend import conditioning, sdxl_unified_runtime

        conditioning.clear_prompt_conditioning_cache()
        sdxl_unified_runtime.clear_unified_sdxl_runtime_component_cache()
    except BaseException:
        pass

