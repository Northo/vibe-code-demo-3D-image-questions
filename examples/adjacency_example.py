"""Example usage of the adjacency checking functions."""

import numpy as np

from src.imaging.adjacency import check_green_touches_red


def main():
    """Demonstrate checking if green touches red in a 3D medical image."""
    # Example 1: Green and red are adjacent
    print("Example 1: Adjacent green and red")
    image1 = np.array(
        [
            [[1, 2, 3], [1, 3, 2], [3, 3, 2]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [2, 2, 2]],
        ]
    )

    category_map1 = {0: "background", 1: "green", 2: "red", 3: "blue"}

    result1 = check_green_touches_red(image1, category_map1)
    print(f"  Do green and red touch? {result1}")
    print()

    # Example 2: Green and red are separated
    print("Example 2: Separated green and red")
    image2 = np.array(
        [
            [[1, 0, 2], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ]
    )

    category_map2 = {0: "background", 1: "green", 2: "red"}

    result2 = check_green_touches_red(image2, category_map2)
    print(f"  Do green and red touch? {result2}")
    print()

    # Example 3: Diagonal touching (26-connectivity)
    print("Example 3: Diagonal touching in 3D")
    image3 = np.array(
        [
            [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 2, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ]
    )

    category_map3 = {0: "background", 1: "green", 2: "red"}

    result3 = check_green_touches_red(image3, category_map3)
    print(f"  Do green and red touch? {result3}")
    print()


if __name__ == "__main__":
    main()
