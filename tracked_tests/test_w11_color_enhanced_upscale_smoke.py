"""Tracked W11c local-smoke scaffold for color-enhanced-upscale."""

import pytest


pytestmark = pytest.mark.skip(
    reason="W11c implementation has not landed yet; this tracked smoke module is reserved by the work order."
)


def test_w11c_color_enhanced_upscale_smoke_placeholder() -> None:
    """Reserve this module for W11c local smoke coverage."""
