"""Example usage of the adjacency checking functions with CategoricalImage."""

import numpy as np

from src.imaging import (
    CategoricalImage,
    check_category_adjacency,
    check_green_touches_red,
)


def main():
    """Demonstrate checking if green touches red in a 3D medical image."""
    print("=" * 70)
    print("CategoricalImage: Adjacency Checking Examples")
    print("=" * 70)
    print()

    # Example 1: Create image from string labels (most intuitive)
    print("Example 1: Creating image from category labels")
    print("-" * 70)

    labels1 = np.array(
        [
            [
                ["green", "red", "blue"],
                ["green", "blue", "red"],
                ["blue", "blue", "red"],
            ],
            [
                ["background", "background", "background"],
                ["background", "green", "background"],
                ["background", "background", "background"],
            ],
            [
                ["background", "background", "background"],
                ["background", "background", "background"],
                ["red", "red", "red"],
            ],
        ]
    )

    image1 = CategoricalImage.from_labels(labels1)
    print(f"  Image: {image1}")
    print(f"  Categories found: {image1.categories}")
    print(f"  Green voxels: {image1.count_voxels('green')}")
    print(f"  Red voxels: {image1.count_voxels('red')}")
    print()

    result1 = check_green_touches_red(image1)
    print(f"  Do green and red touch? {result1}")
    print()

    # Example 2: Separated green and red
    print("Example 2: Separated regions")
    print("-" * 70)

    labels2 = np.array(
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

    image2 = CategoricalImage.from_labels(labels2)
    print(f"  Image: {image2}")

    result2 = check_green_touches_red(image2)
    print(f"  Do green and red touch? {result2}")
    print()

    # Example 3: Diagonal touching in 3D (26-connectivity)
    print("Example 3: 3D diagonal adjacency")
    print("-" * 70)

    labels3 = np.array(
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

    image3 = CategoricalImage.from_labels(labels3)
    print(f"  Image: {image3}")
    print(f"  Green is at position [0, 0, 0]")
    print(f"  Red is at position [1, 0, 1]")
    print(f"  These are diagonal neighbors in 3D space")
    print()

    result3 = check_green_touches_red(image3)
    print(f"  Do green and red touch? {result3}")
    print()

    # Example 4: Check arbitrary category pairs
    print("Example 4: Checking other category pairs")
    print("-" * 70)

    labels4 = np.array(
        [
            [
                ["liver", "kidney", "background"],
                ["liver", "background", "kidney"],
                ["background", "background", "background"],
            ],
            [
                ["spleen", "background", "background"],
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

    image4 = CategoricalImage.from_labels(labels4)
    print(f"  Image: {image4}")
    print(f"  Category statistics: {image4.get_category_stats()}")
    print()

    liver_kidney = check_category_adjacency(image4, "liver", "kidney")
    print(f"  Do liver and kidney touch? {liver_kidney}")

    liver_spleen = check_category_adjacency(image4, "liver", "spleen")
    print(f"  Do liver and spleen touch? {liver_spleen}")

    kidney_spleen = check_category_adjacency(image4, "kidney", "spleen")
    print(f"  Do kidney and spleen touch? {kidney_spleen}")
    print()

    # Example 5: Working with encoded data (when you have integer arrays)
    print("Example 5: Creating from integer-encoded data")
    print("-" * 70)

    data5 = np.array(
        [
            [[0, 1, 2], [1, 2, 0], [2, 0, 1]],
            [[1, 1, 1], [2, 2, 2], [0, 0, 0]],
            [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        ]
    )

    categories5 = ["background", "green", "red"]
    image5 = CategoricalImage(data5, categories5)

    print(f"  Image: {image5}")
    print(f"  Created from integer array with explicit category mapping")
    print()

    result5 = check_green_touches_red(image5)
    print(f"  Do green and red touch? {result5}")
    print()

    # Example 6: Using masks for analysis
    print("Example 6: Working with category masks")
    print("-" * 70)

    image6 = CategoricalImage.from_labels(labels1)

    green_mask = image6.get_mask("green")
    red_mask = image6.get_mask("red")

    print(f"  Total voxels: {np.prod(image6.shape)}")
    print(f"  Green voxels: {np.sum(green_mask)}")
    print(f"  Red voxels: {np.sum(red_mask)}")
    print(f"  Green mask shape: {green_mask.shape}")
    print(f"  Mask is boolean array: {green_mask.dtype == bool}")
    print()

    print("=" * 70)


if __name__ == "__main__":
    main()
