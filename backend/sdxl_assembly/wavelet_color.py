"""Stateless low-frequency color transplant utilities.

The transplant follows the reconstruction used by SUPIR's color-fix path:
an undecimated (same-resolution) wavelet pyramid made from progressively
dilated, normalized binomial blurs.  Keeping every level at the input
resolution is important here.  A critically sampled Haar pyramid introduces a
sampling phase and can make a replaced coarse band visible as square blocks.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _validate_image(image: torch.Tensor, name: str) -> None:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"{name} expects a torch.Tensor, got {type(image).__name__}.")
    if image.ndim != 4:
        raise ValueError(f"{name} expects BCHW input, got {tuple(image.shape)}.")
    if not torch.is_floating_point(image):
        raise TypeError(f"{name} expects a floating-point tensor, got {image.dtype}.")
    if image.shape[1] < 1 or image.shape[2] < 1 or image.shape[3] < 1:
        raise ValueError(f"{name} expects non-empty BCHW dimensions, got {tuple(image.shape)}.")


def _validate_levels(levels: int) -> None:
    if not isinstance(levels, int) or isinstance(levels, bool) or levels < 1:
        raise ValueError(f"Wavelet levels must be a positive integer, got {levels}.")


def _replicate_pad(image: torch.Tensor, radius: int) -> torch.Tensor:
    """Replicate-pad an image, including when the pad exceeds its size.

    ``F.pad(..., mode='replicate')`` rejects a pad larger than a spatial
    dimension on some PyTorch versions.  Index-based extension has the same
    boundary semantics and keeps the utility valid for small test images and
    narrow aspect-ratio buckets.
    """

    height, width = image.shape[-2:]
    y = torch.arange(-radius, height + radius, device=image.device).clamp(0, height - 1)
    x = torch.arange(-radius, width + radius, device=image.device).clamp(0, width - 1)
    return image.index_select(-2, y).index_select(-1, x)


def _wavelet_blur(image: torch.Tensor, radius: int) -> torch.Tensor:
    """Apply a normalized binomial low-pass filter at an a-trous scale."""

    # The 3x3 binomial kernel is a separable Gaussian approximation.  Dilation
    # 1, 2, 4, ... produces the undecimated wavelet scales without introducing
    # a downsampling grid.
    kernel = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).reshape(1, 1, 3, 3) / 16.0
    kernel = kernel.expand(image.shape[1], 1, 3, 3)
    padded = _replicate_pad(image, radius)
    return F.conv2d(padded, kernel, dilation=radius, groups=image.shape[1])


def wavelet_decomposition(
    image: torch.Tensor,
    levels: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return same-shaped high- and low-frequency components for ``image``.

    This is an undecimated residual pyramid.  At each scale, the detail band
    is the difference between the current image and its dilated binomial
    low-pass version.  Therefore ``high + low`` reconstructs ``image`` without
    any decimation, interpolation, or phase-dependent block grid.
    """

    _validate_image(image, "wavelet_decomposition")
    _validate_levels(levels)

    current = image
    high = torch.zeros_like(image)
    for level in range(levels):
        low = _wavelet_blur(current, radius=1 << level)
        high = high + (current - low)
        current = low
    return high, current


def wavelet_reconstruction(
    content: torch.Tensor,
    color_ref: torch.Tensor,
    levels: int = 5,
) -> torch.Tensor:
    """Combine GAN detail from ``content`` with SDXL color from ``color_ref``.

    The color reference is resized before decomposition, as required by the
    route contract.  No state is retained between calls and the result keeps
    the content donor's BCHW shape, device, and dtype.
    """

    _validate_image(content, "wavelet_reconstruction content")
    _validate_image(color_ref, "wavelet_reconstruction color_ref")
    _validate_levels(levels)
    if color_ref.shape[:2] != content.shape[:2]:
        raise ValueError(
            "wavelet_reconstruction requires matching batch and channel dimensions: "
            f"content={tuple(content.shape)}, color_ref={tuple(color_ref.shape)}."
        )

    color_ref = color_ref.to(device=content.device, dtype=content.dtype)
    if color_ref.shape[-2:] != content.shape[-2:]:
        color_ref = F.interpolate(
            color_ref,
            size=content.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    content_high, _ = wavelet_decomposition(content, levels=levels)
    _, color_low = wavelet_decomposition(color_ref, levels=levels)

    # Donor direction is intentional: GAN supplies structure/detail, SDXL
    # supplies only the coarse color field.
    return torch.clamp(content_high + color_low, 0.0, 1.0)
