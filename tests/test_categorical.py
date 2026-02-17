"""Tests for CategoricalImage data structure."""

import numpy as np
import pytest

from src.imaging.categorical import CategoricalImage


def test_create_from_encoded_data():
    """Test creating CategoricalImage from integer-encoded data."""
    data = np.array([[[0, 1, 2], [1, 2, 0], [2, 0, 1]]])
    categories = ["background", "liver", "kidney"]

    image = CategoricalImage(data, categories)

    assert image.shape == (1, 3, 3)
    assert image.categories == tuple(categories)


def test_create_from_labels():
    """Test creating CategoricalImage from string labels."""
    labels = np.array([[["liver", "kidney", "liver"], ["kidney", "liver", "kidney"]]])

    image = CategoricalImage.from_labels(labels)

    assert image.shape == (1, 2, 3)
    assert set(image.categories) == {"liver", "kidney"}


def test_get_mask():
    """Test getting binary mask for a category."""
    data = np.array([[[0, 1, 2], [1, 2, 0], [2, 0, 1]]])
    categories = ["background", "liver", "kidney"]

    image = CategoricalImage(data, categories)

    liver_mask = image.get_mask("liver")
    expected = np.array(
        [[[False, True, False], [True, False, False], [False, False, True]]]
    )

    assert np.array_equal(liver_mask, expected)


def test_has_category():
    """Test checking if category exists in image."""
    data = np.array([[[0, 1, 0], [0, 1, 0]]])
    categories = ["background", "liver", "kidney"]

    image = CategoricalImage(data, categories)

    assert image.has_category("liver") is True
    assert image.has_category("background") is True
    assert image.has_category("kidney") is False  # Defined but not present


def test_count_voxels():
    """Test counting voxels per category."""
    data = np.array([[[0, 1, 1], [1, 2, 0], [2, 0, 1]]])
    categories = ["background", "liver", "kidney"]

    image = CategoricalImage(data, categories)

    assert image.count_voxels("background") == 3
    assert image.count_voxels("liver") == 4
    assert image.count_voxels("kidney") == 2


def test_get_category_stats():
    """Test getting voxel counts for all categories."""
    data = np.array([[[0, 1, 1], [1, 2, 0], [2, 0, 1]]])
    categories = ["background", "liver", "kidney"]

    image = CategoricalImage(data, categories)
    stats = image.get_category_stats()

    assert stats == {"background": 3, "liver": 4, "kidney": 2}


def test_get_labels():
    """Test converting back to string labels."""
    data = np.array([[[0, 1, 2]]])
    categories = ["background", "liver", "kidney"]

    image = CategoricalImage(data, categories)
    labels = image.get_labels()

    expected = np.array([[["background", "liver", "kidney"]]])
    assert np.array_equal(labels, expected)


def test_get_labels_roundtrip():
    """Test creating from labels and converting back."""
    original_labels = np.array(
        [[["liver", "kidney", "liver"], ["kidney", "liver", "kidney"]]]
    )

    image = CategoricalImage.from_labels(original_labels)
    recovered_labels = image.get_labels()

    assert np.array_equal(original_labels, recovered_labels)


def test_data_is_copied():
    """Test that data is copied to prevent external modification."""
    data = np.array([[[0, 1, 2]]])
    categories = ["background", "liver", "kidney"]

    image = CategoricalImage(data, categories)

    # Modify original
    data[0, 0, 0] = 99

    # Image data should be unchanged
    assert image.data[0, 0, 0] == 0


def test_categories_is_copied():
    """Test that categories list is copied."""
    data = np.array([[[0, 1, 2]]])
    categories = ["background", "liver", "kidney"]

    image = CategoricalImage(data, categories)

    # Modify original
    categories[0] = "modified"

    # Image categories should be unchanged
    assert image.categories[0] == "background"


def test_invalid_2d_data():
    """Test that 2D data raises ValueError."""
    data = np.array([[0, 1], [2, 3]])
    categories = ["a", "b", "c", "d"]

    with pytest.raises(ValueError, match="Expected 3D image, got 2D"):
        CategoricalImage(data, categories)


def test_invalid_4d_data():
    """Test that 4D data raises ValueError."""
    data = np.zeros((2, 3, 4, 5), dtype=int)
    categories = ["a"]

    with pytest.raises(ValueError, match="Expected 3D image, got 4D"):
        CategoricalImage(data, categories)


def test_empty_data():
    """Test that empty data raises ValueError."""
    data = np.array([[[]]]).reshape(0, 0, 0)
    categories = ["a"]

    with pytest.raises(ValueError, match="Image cannot be empty"):
        CategoricalImage(data, categories)


def test_empty_categories():
    """Test that empty categories list raises ValueError."""
    data = np.array([[[0]]])
    categories = []

    with pytest.raises(ValueError, match="Categories list cannot be empty"):
        CategoricalImage(data, categories)


def test_duplicate_categories():
    """Test that duplicate category names raise ValueError."""
    data = np.array([[[0, 1]]])
    categories = ["liver", "liver"]

    with pytest.raises(ValueError, match="Category names must be unique"):
        CategoricalImage(data, categories)


def test_data_value_out_of_range():
    """Test that data values outside category range raise ValueError."""
    data = np.array([[[0, 1, 5]]])  # 5 is out of range
    categories = ["a", "b"]  # Only indices 0-1 valid

    with pytest.raises(ValueError, match="Data contains value 5 but only 2 categories"):
        CategoricalImage(data, categories)


def test_negative_data_values():
    """Test that negative data values raise ValueError."""
    data = np.array([[[0, -1, 1]]])
    categories = ["a", "b"]

    with pytest.raises(ValueError, match="Data contains negative values"):
        CategoricalImage(data, categories)


def test_get_mask_nonexistent_category():
    """Test that getting mask for undefined category raises ValueError."""
    data = np.array([[[0, 1]]])
    categories = ["a", "b"]

    image = CategoricalImage(data, categories)

    with pytest.raises(ValueError, match="Category 'nonexistent' not found"):
        image.get_mask("nonexistent")


def test_count_voxels_nonexistent_category():
    """Test that counting nonexistent category raises ValueError."""
    data = np.array([[[0, 1]]])
    categories = ["a", "b"]

    image = CategoricalImage(data, categories)

    with pytest.raises(ValueError, match="Category 'nonexistent' not found"):
        image.count_voxels("nonexistent")


def test_repr():
    """Test string representation of CategoricalImage."""
    data = np.array([[[0, 1, 2]]])
    categories = ["background", "liver", "kidney"]

    image = CategoricalImage(data, categories)
    repr_str = repr(image)

    assert "CategoricalImage" in repr_str
    assert "shape=(1, 1, 3)" in repr_str
    assert "background" in repr_str
    assert "liver" in repr_str
    assert "kidney" in repr_str


def test_from_labels_empty():
    """Test that from_labels with empty array raises ValueError."""
    labels = np.array([[[]]]).reshape(0, 0, 0)

    with pytest.raises(ValueError, match="Labels array cannot be empty"):
        CategoricalImage.from_labels(labels)


def test_from_labels_2d():
    """Test that from_labels with 2D array raises ValueError."""
    labels = np.array([["a", "b"], ["c", "d"]])

    with pytest.raises(ValueError, match="Expected 3D labels array, got 2D"):
        CategoricalImage.from_labels(labels)


def test_large_image():
    """Test with realistic medical image size."""
    # Create a 100x100x50 image
    data = np.zeros((100, 100, 50), dtype=np.int32)
    data[10:20, 10:20, 10:20] = 1
    data[30:40, 30:40, 30:40] = 2

    categories = ["background", "liver", "kidney"]

    image = CategoricalImage(data, categories)

    assert image.shape == (100, 100, 50)
    assert image.count_voxels("liver") == 10 * 10 * 10
    assert image.count_voxels("kidney") == 10 * 10 * 10
