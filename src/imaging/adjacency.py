"""Functions for analyzing adjacency relationships in categorical 3D medical images."""

import numpy as np
from scipy import ndimage


def check_category_adjacency(
    image: np.ndarray, category_map: dict[int, str], category1: str, category2: str
) -> bool:
    """Check if any voxels of category1 are adjacent to voxels of category2.

    Two voxels are considered adjacent if they are neighbors in 3D space
    (26-connectivity: face, edge, or corner neighbors).

    Args:
        image: 3D numpy array where each voxel contains an integer representing
            a categorical value
        category_map: Dictionary mapping integer values to category names
        category1: Name of the first category (e.g., "green")
        category2: Name of the second category (e.g., "red")

    Returns:
        True if any voxel labeled as category1 touches any voxel labeled as
        category2, False otherwise

    Raises:
        ValueError: If the image is not 3D, if category names are not found
            in the category_map, or if the image is empty

    Example:
        >>> image = np.array([[[1, 1, 2], [1, 3, 2], [3, 3, 2]]])
        >>> category_map = {1: "green", 2: "red", 3: "blue"}
        >>> check_category_adjacency(image, category_map, "green", "red")
        True
    """
    # Validate input
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image, got {image.ndim}D")

    if image.size == 0:
        raise ValueError("Image cannot be empty")

    # Create reverse mapping from category names to integer values
    reverse_map = {name: value for value, name in category_map.items()}

    if category1 not in reverse_map:
        raise ValueError(f"Category '{category1}' not found in category_map")

    if category2 not in reverse_map:
        raise ValueError(f"Category '{category2}' not found in category_map")

    value1 = reverse_map[category1]
    value2 = reverse_map[category2]

    # Create binary masks for each category
    mask1 = image == value1
    mask2 = image == value2

    # If either category doesn't exist in the image, they can't touch
    if not np.any(mask1) or not np.any(mask2):
        return False

    # Dilate the first category mask to include all neighboring positions
    # Using 26-connectivity (3x3x3 structuring element)
    structure = ndimage.generate_binary_structure(3, 3)
    dilated_mask1 = ndimage.binary_dilation(mask1, structure=structure)

    # Check if any dilated voxels from category1 overlap with category2
    return bool(np.any(dilated_mask1 & mask2))


def check_green_touches_red(image: np.ndarray, category_map: dict[int, str]) -> bool:
    """Check if any voxels labeled 'green' are adjacent to voxels labeled 'red'.

    This is a convenience wrapper around check_category_adjacency specifically
    for checking green-red adjacency.

    Args:
        image: 3D numpy array where each voxel contains an integer representing
            a categorical value
        category_map: Dictionary mapping integer values to category names.
            Must include "green" and "red" as category names.

    Returns:
        True if any green voxel touches any red voxel, False otherwise

    Raises:
        ValueError: If the image is not 3D, if "green" or "red" categories
            are not found in the category_map, or if the image is empty

    Example:
        >>> image = np.array([[[1, 1, 2], [1, 3, 2], [3, 3, 2]]])
        >>> category_map = {1: "green", 2: "red", 3: "blue"}
        >>> check_green_touches_red(image, category_map)
        True
    """
    return check_category_adjacency(image, category_map, "green", "red")
