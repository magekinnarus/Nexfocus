from __future__ import annotations

from typing import Any


GAN_TILE_SIZE_MIN = 256
GAN_TILE_SIZE_MAX = 1024
GAN_TILE_SIZE_STEP = 64


def normalize_gan_tile_size(value: Any) -> int:
    """Normalize explicit GAN tile requests onto the supported ladder."""
    try:
        tile_size = int(value)
    except (TypeError, ValueError):
        return GAN_TILE_SIZE_MIN

    tile_size = max(GAN_TILE_SIZE_MIN, min(GAN_TILE_SIZE_MAX, tile_size))
    normalized_offset = (tile_size - GAN_TILE_SIZE_MIN) // GAN_TILE_SIZE_STEP
    return GAN_TILE_SIZE_MIN + (normalized_offset * GAN_TILE_SIZE_STEP)
