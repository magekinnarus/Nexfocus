import numpy as np
import pytest

from modules.pipeline.outpaint import OutpaintPipeline


def test_outpaint_prepare_normalizes_missing_mask_to_keep_canvas():
    image = np.zeros((32, 48, 3), dtype=np.uint8)

    context = OutpaintPipeline().prepare(
        image,
        None,
        outpaint_direction=None,
        generate_blend_mask=False,
    )

    assert context.original_mask.shape == image.shape[:2]
    assert context.original_mask.dtype == np.uint8
    assert not context.original_mask.any()


def test_outpaint_prepare_rejects_mismatched_mask_with_actionable_error():
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    mask = np.zeros((16, 24), dtype=np.uint8)

    with pytest.raises(ValueError, match="mask must match the source image dimensions"):
        OutpaintPipeline().prepare(image, mask)
