from __future__ import annotations

import pytest

from modules.creative_document import AffineTransform, BBox, TransformError, map_pixel_center, unmap_pixel_center


def test_affine_forward_inverse_and_half_open_pixel_centers() -> None:
    transform = AffineTransform.translation(12, -4).compose(AffineTransform.scale(2, 3))
    point = transform.apply_point(3.25, 5.5)
    assert transform.inverse().apply_point(*point) == pytest.approx((3.25, 5.5))

    bbox = BBox(10, 20, 30, 50)
    assert bbox.width == 20 and bbox.height == 10
    assert map_pixel_center(0, 30, 2.0) == pytest.approx(-59.5)
    assert unmap_pixel_center(0, 30, 2.0) == pytest.approx(29.75)


def test_singular_transform_and_invalid_bbox_are_rejected() -> None:
    with pytest.raises(TransformError):
        AffineTransform.scale(0, 1)
    with pytest.raises(TransformError):
        BBox(4, 4, 0, 3).validate()
