"""Tests for adjacency checking functions."""

import numpy as np
import pytest

from src.imaging.adjacency import check_category_adjacency, check_green_touches_red
from src.imaging.categorical import CategoricalImage


def test_green_touches_red_adjacent():
    """Test that adjacent green and red voxels are detected."""
    # Create image where green and red are adjacent
    labels = np.array(
        [
            [
                ["green", "red", "background"],
                ["background", "background", "background"],
                ["background", "background", "background"],
            ],
            [
                ["background", "background", "background"],
                ["background", "background", "background"],
                ["background", "background", "background"],
            ],
            [
                ["background", "background", "background"],
                ["background", "background", "background"],
                ["background", "background", "background"],
            ],
        ]
    )

    image = CategoricalImage.from_labels(labels)

    assert check_green_touches_red(image) is True


def test_green_touches_red_diagonal():
    """Test that diagonally adjacent green and red voxels are detected."""
    # Green and red touch diagonally (26-connectivity)
    labels = np.array(
        [
            [
                ["green", "background", "background"],
                ["background", "red", "background"],
                ["background", "background", "background"],
            ],
            [
                ["background", "background", "background"],
                ["background", "background", "background"],
                ["background", "background", "background"],
            ],
            [
                ["background", "background", "background"],
                ["background", "background", "background"],
                ["background", "background", "background"],
            ],
        ]
    )

    image = CategoricalImage.from_labels(labels)

    assert check_green_touches_red(image) is True


def test_green_touches_red_3d_diagonal():
    """Test that green and red touching in 3D diagonal are detected."""
    # Green and red touch in 3D space (corner neighbors)
    labels = np.array(
        [
            [
                ["green", "background", "background"],
                ["background", "background", "background"],
                ["background", "background", "background"],
            ],
            [
                ["background", "red", "background"],
                ["background", "background", "background"],
                ["background", "background", "background"],
            ],
            [
                ["background", "background", "background"],
                ["background", "background", "background"],
                ["background", "background", "background"],
            ],
        ]
    )

    image = CategoricalImage.from_labels(labels)

    assert check_green_touches_red(image) is True


def test_green_does_not_touch_red():
    """Test that separated green and red voxels are not detected as touching."""
    # Green and red are separated by at least one voxel
    labels = np.array(
        [
            [
                ["green", "background", "red"],
                ["background", "background", "background"],
                ["background", "background", "background"],
            ],
            [
                ["background", "background", "background"],
                ["background", "background", "background"],
                ["background", "background", "background"],
            ],
            [
                ["background", "background", "background"],
                ["background", "background", "background"],
                ["background", "background", "background"],
            ],
        ]
    )

    image = CategoricalImage.from_labels(labels)

    assert check_green_touches_red(image) is False


def test_no_green_category():
    """Test when green category doesn't exist in the image."""
    labels = np.array(
        [
            [
                ["background", "red", "background"],
                ["background", "red", "background"],
                ["background", "background", "background"],
            ],
            [
                ["background", "background", "background"],
                ["background", "background", "background"],
                ["background", "background", "background"],
            ],
            [
                ["background", "background", "background"],
                ["background", "background", "background"],
                ["background", "background", "background"],
            ],
        ]
    )

    image = CategoricalImage.from_labels(labels)

    # Green doesn't exist, so check should fail
    with pytest.raises(ValueError, match="Category 'green' not found"):
        check_green_touches_red(image)


def test_no_red_category():
    """Test when red category doesn't exist in the image."""
    labels = np.array(
        [
            [
                ["green", "green", "background"],
                ["green", "background", "background"],
                ["background", "background", "background"],
            ],
            [
                ["background", "background", "background"],
                ["background", "background", "background"],
                ["background", "background", "background"],
            ],
            [
                ["background", "background", "background"],
                ["background", "background", "background"],
                ["background", "background", "background"],
            ],
        ]
    )

    image = CategoricalImage.from_labels(labels)

    # Red doesn't exist, so check should fail
    with pytest.raises(ValueError, match="Category 'red' not found"):
        check_green_touches_red(image)


def test_general_category_adjacency():
    """Test the general check_category_adjacency function."""
    labels = np.array(
        [
            [
                ["green", "blue", "background"],
                ["background", "background", "background"],
                ["background", "background", "background"],
            ],
            [
                ["background", "background", "background"],
                ["background", "background", "background"],
                ["background", "background", "background"],
            ],
            [
                ["background", "background", "background"],
                ["background", "background", "background"],
                ["background", "background", "background"],
            ],
        ]
    )

    image = CategoricalImage.from_labels(labels)

    # Green and blue are adjacent
    assert check_category_adjacency(image, "green", "blue") is True

    # Green and background are adjacent
    assert check_category_adjacency(image, "green", "background") is True

    # Blue and background are adjacent
    assert check_category_adjacency(image, "blue", "background") is True


def test_large_image_performance():
    """Test with a larger realistic medical image size."""
    # Create a 100x100x50 image
    data = np.zeros((100, 100, 50), dtype=np.int32)

    # Place green region in one corner
    data[10:20, 10:20, 10:20] = 1

    # Place red region adjacent to green
    data[20:30, 10:20, 10:20] = 2

    categories = ["background", "green", "red"]
    image = CategoricalImage(data, categories)

    assert check_green_touches_red(image) is True


def test_green_red_defined_but_not_present():
    """Test when green and red are in categories but not in the image."""
    data = np.zeros((3, 3, 3), dtype=np.int32)  # All background
    categories = ["background", "green", "red"]

    image = CategoricalImage(data, categories)

    # Both categories defined but not present in image
    assert check_green_touches_red(image) is False


# Legacy numpy array tests (backward compatibility)


def test_legacy_numpy_array_green_touches_red():
    """Test legacy API with raw numpy array."""
    image = np.array(
        [
            [[1, 2, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ]
    )

    category_map = {0: "background", 1: "green", 2: "red"}

    assert check_green_touches_red(image, category_map) is True


def test_legacy_numpy_array_missing_category_map():
    """Test that legacy API requires category_map."""
    image = np.array([[[1, 2, 0]]])

    with pytest.raises(ValueError, match="category_map is required"):
        check_green_touches_red(image)


def test_legacy_numpy_array_invalid_2d():
    """Test that legacy API validates 2D arrays."""
    image = np.array([[1, 2], [3, 4]])
    category_map = {1: "green", 2: "red"}

    with pytest.raises(ValueError, match="Expected 3D image, got 2D"):
        check_green_touches_red(image, category_map)


def test_legacy_category_adjacency():
    """Test legacy general category adjacency."""
    image = np.array(
        [
            [[1, 3, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ]
    )

    category_map = {0: "background", 1: "green", 2: "red", 3: "blue"}

    # Green and blue are adjacent
    assert check_category_adjacency(image, "green", "blue", category_map) is True

    # Green and red are not adjacent
    assert check_category_adjacency(image, "green", "red", category_map) is False


def test_multiple_green_red_regions():
    """Test with multiple disconnected regions of green and red."""
    labels = np.array(
        [
            [
                ["green", "background", "red"],
                ["background", "background", "background"],
                ["green", "background", "red"],
            ],
            [
                ["background", "background", "background"],
                ["background", "green", "red"],
                ["background", "background", "background"],
            ],
            [
                ["background", "background", "background"],
                ["background", "background", "background"],
                ["background", "background", "background"],
            ],
        ]
    )

    image = CategoricalImage.from_labels(labels)

    # At [1, 1, 1] green is adjacent to [1, 1, 2] red
    assert check_green_touches_red(image) is True
