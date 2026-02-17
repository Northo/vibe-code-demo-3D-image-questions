"""Tests for adjacency checking functions."""

import numpy as np
import pytest

from src.imaging.adjacency import check_category_adjacency, check_green_touches_red


def test_green_touches_red_adjacent():
    """Test that adjacent green and red voxels are detected."""
    # Create a simple 3D image where green (1) and red (2) are adjacent
    image = np.array(
        [
            [[1, 2, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ]
    )

    category_map = {0: "background", 1: "green", 2: "red"}

    assert check_green_touches_red(image, category_map) is True


def test_green_touches_red_diagonal():
    """Test that diagonally adjacent green and red voxels are detected."""
    # Green and red touch diagonally (26-connectivity)
    image = np.array(
        [
            [[1, 0, 0], [0, 2, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ]
    )

    category_map = {0: "background", 1: "green", 2: "red"}

    assert check_green_touches_red(image, category_map) is True


def test_green_touches_red_3d_diagonal():
    """Test that green and red touching in 3D diagonal are detected."""
    # Green and red touch in 3D space (corner neighbors)
    image = np.array(
        [
            [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 2, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ]
    )

    category_map = {0: "background", 1: "green", 2: "red"}

    assert check_green_touches_red(image, category_map) is True


def test_green_does_not_touch_red():
    """Test that separated green and red voxels are not detected as touching."""
    # Green and red are separated by at least one voxel
    image = np.array(
        [
            [[1, 0, 2], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ]
    )

    category_map = {0: "background", 1: "green", 2: "red"}

    assert check_green_touches_red(image, category_map) is False


def test_no_green_category():
    """Test when green category doesn't exist in the image."""
    image = np.array(
        [
            [[0, 2, 0], [0, 2, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ]
    )

    category_map = {0: "background", 1: "green", 2: "red"}

    assert check_green_touches_red(image, category_map) is False


def test_no_red_category():
    """Test when red category doesn't exist in the image."""
    image = np.array(
        [
            [[1, 1, 0], [1, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ]
    )

    category_map = {0: "background", 1: "green", 2: "red"}

    assert check_green_touches_red(image, category_map) is False


def test_category_not_in_map():
    """Test that missing category in map raises ValueError."""
    image = np.zeros((3, 3, 3), dtype=int)
    category_map = {0: "background"}

    with pytest.raises(ValueError, match="Category 'green' not found"):
        check_green_touches_red(image, category_map)


def test_invalid_2d_image():
    """Test that 2D image raises ValueError."""
    image = np.array([[1, 2], [3, 4]])
    category_map = {1: "green", 2: "red"}

    with pytest.raises(ValueError, match="Expected 3D image, got 2D"):
        check_green_touches_red(image, category_map)


def test_empty_image():
    """Test that empty image raises ValueError."""
    image = np.array([[[]]]).reshape(0, 0, 0)
    category_map = {1: "green", 2: "red"}

    with pytest.raises(ValueError, match="Image cannot be empty"):
        check_green_touches_red(image, category_map)


def test_general_category_adjacency():
    """Test the general check_category_adjacency function."""
    image = np.array(
        [
            [[1, 3, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ]
    )

    category_map = {0: "background", 1: "green", 2: "red", 3: "blue"}

    # Green and blue are adjacent
    assert check_category_adjacency(image, category_map, "green", "blue") is True

    # Green and red are not adjacent
    assert check_category_adjacency(image, category_map, "green", "red") is False


def test_large_image_performance():
    """Test with a larger realistic medical image size."""
    # Create a 100x100x50 image
    image = np.zeros((100, 100, 50), dtype=np.int32)

    # Place green region in one corner
    image[10:20, 10:20, 10:20] = 1

    # Place red region adjacent to green
    image[20:30, 10:20, 10:20] = 2

    category_map = {0: "background", 1: "green", 2: "red"}

    assert check_green_touches_red(image, category_map) is True
