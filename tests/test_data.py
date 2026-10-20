import numpy as np
import pytest
from PIL import Image

from src.brain_tumor.data import DEFAULT_MASK_THRESHOLD, binarize_mask


def test_binarize_mask_ignores_low_intensity_background_artifacts():
    raw = np.array([[0, 1, 7, 127, 128, 248, 254, 255]], dtype=np.uint8)

    result = binarize_mask(raw)

    assert result.dtype == np.float32
    np.testing.assert_array_equal(
        result,
        np.array([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=np.float32),
    )


def test_binarize_mask_supports_pil_images():
    raw = Image.fromarray(np.array([[0, 127, 128, 255]], dtype=np.uint8), mode="L")

    result = binarize_mask(raw)

    np.testing.assert_array_equal(
        result,
        np.array([[0, 0, 1, 1]], dtype=np.float32),
    )


def test_binarize_mask_allows_explicit_threshold():
    raw = np.array([[10, 11, 12]], dtype=np.uint8)

    result = binarize_mask(raw, threshold=11)

    np.testing.assert_array_equal(
        result,
        np.array([[0, 1, 1]], dtype=np.float32),
    )


@pytest.mark.parametrize("threshold", [-1, 256])
def test_binarize_mask_rejects_out_of_range_threshold(threshold):
    with pytest.raises(ValueError, match="threshold must be between 0 and 255"):
        binarize_mask(np.zeros((2, 2), dtype=np.uint8), threshold=threshold)


def test_default_mask_threshold_is_128():
    assert DEFAULT_MASK_THRESHOLD == 128
