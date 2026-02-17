"""Data structures for categorical medical images."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CategoricalImage:
    """A 3D medical image with categorical voxel values.

    This class encapsulates a 3D image where each voxel represents a categorical
    value (e.g., tissue types, organ labels, etc.). Category names are treated
    as first-class citizens, with integer encoding handled internally.

    Attributes:
        data: 3D numpy array with integer-encoded categorical values
        categories: Tuple of category names in encoding order

    Example:
        >>> # Create from category labels directly
        >>> labels = np.array([[["background", "liver", "kidney"]]])
        >>> image = CategoricalImage.from_labels(labels)
        >>>
        >>> # Create from encoded array with category mapping
        >>> data = np.array([[[0, 1, 2]]])
        >>> categories = ["background", "liver", "kidney"]
        >>> image = CategoricalImage(data, categories)
        >>>
        >>> # Access category data
        >>> image.get_mask("liver")
        >>> image.has_category("kidney")
    """

    data: np.ndarray
    categories: tuple[str, ...] | list[str]
    _category_to_index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self):
        """Validate and process data after initialization."""
        # Validate data dimensions
        if self.data.ndim != 3:
            raise ValueError(f"Expected 3D image, got {self.data.ndim}D")

        if self.data.size == 0:
            raise ValueError("Image cannot be empty")

        # Validate categories
        if len(self.categories) == 0:
            raise ValueError("Categories list cannot be empty")

        if len(self.categories) != len(set(self.categories)):
            raise ValueError("Category names must be unique")

        # Validate data values are within valid range
        max_value = np.max(self.data)
        min_value = np.min(self.data)

        if min_value < 0:
            raise ValueError(f"Data contains negative values: {min_value}")

        if max_value >= len(self.categories):
            raise ValueError(
                f"Data contains value {max_value} but only {len(self.categories)} "
                f"categories provided (valid range: 0-{len(self.categories) - 1})"
            )

        # Convert data to int32 and make immutable (copy to avoid external mutation)
        object.__setattr__(self, "data", self.data.astype(np.int32, copy=True))

        # Convert categories to tuple if it's a list
        if isinstance(self.categories, list):
            object.__setattr__(self, "categories", tuple(self.categories))

        # Build category to index mapping
        category_to_index = {cat: idx for idx, cat in enumerate(self.categories)}
        object.__setattr__(self, "_category_to_index", category_to_index)

    @classmethod
    def from_labels(cls, labels: np.ndarray) -> "CategoricalImage":
        """Create a CategoricalImage from an array of string labels.

        Args:
            labels: 3D numpy array of strings, where each element is a category name

        Returns:
            A new CategoricalImage instance

        Example:
            >>> labels = np.array([[["background", "liver"], ["kidney", "liver"]]])
            >>> image = CategoricalImage.from_labels(labels)
        """
        if labels.ndim != 3:
            raise ValueError(f"Expected 3D labels array, got {labels.ndim}D")

        if labels.size == 0:
            raise ValueError("Labels array cannot be empty")

        # Find unique categories and create encoding
        unique_categories = np.unique(labels)
        categories = tuple(unique_categories)

        # Encode labels as integers
        data = np.zeros(labels.shape, dtype=np.int32)
        for idx, category in enumerate(categories):
            data[labels == category] = idx

        return cls(data, categories)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Get the shape of the image (height, width, depth)."""
        return self.data.shape

    def has_category(self, category: str) -> bool:
        """Check if a category exists in this image.

        Args:
            category: Category name to check

        Returns:
            True if the category is defined and exists in the image
        """
        if category not in self._category_to_index:
            return False

        category_index = self._category_to_index[category]
        return bool(np.any(self.data == category_index))

    def get_mask(self, category: str) -> np.ndarray:
        """Get a binary mask for voxels belonging to a specific category.

        Args:
            category: Name of the category

        Returns:
            Boolean numpy array of the same shape as the image, where True
            indicates voxels belonging to the specified category

        Raises:
            ValueError: If the category is not defined for this image

        Example:
            >>> image = CategoricalImage.from_labels(
            ...     np.array([[["liver", "kidney", "liver"]]])
            ... )
            >>> mask = image.get_mask("liver")
            >>> print(mask)
            [[[True False True]]]
        """
        if category not in self._category_to_index:
            raise ValueError(
                f"Category '{category}' not found. "
                f"Available categories: {', '.join(self.categories)}"
            )

        category_index = self._category_to_index[category]
        return self.data == category_index

    def get_labels(self) -> np.ndarray:
        """Get the image as a 3D array of category name strings.

        Returns:
            3D numpy array of strings with category names

        Example:
            >>> data = np.array([[[0, 1, 0]]])
            >>> image = CategoricalImage(data, ["background", "liver"])
            >>> labels = image.get_labels()
            >>> print(labels)
            [[['background' 'liver' 'background']]]
        """
        labels = np.empty(self.data.shape, dtype=object)
        for idx, category in enumerate(self.categories):
            labels[self.data == idx] = category
        return labels

    def count_voxels(self, category: str) -> int:
        """Count the number of voxels belonging to a category.

        Args:
            category: Name of the category to count

        Returns:
            Number of voxels with the specified category

        Raises:
            ValueError: If the category is not defined for this image
        """
        if category not in self._category_to_index:
            raise ValueError(
                f"Category '{category}' not found. "
                f"Available categories: {', '.join(self.categories)}"
            )

        category_index = self._category_to_index[category]
        return int(np.sum(self.data == category_index))

    def get_category_stats(self) -> dict[str, int]:
        """Get voxel counts for all categories.

        Returns:
            Dictionary mapping category names to voxel counts
        """
        return {cat: self.count_voxels(cat) for cat in self.categories}

    def _get_category_index(self, category: str) -> int:
        """Internal method to get the integer index for a category.

        Args:
            category: Category name

        Returns:
            Integer index for the category

        Raises:
            ValueError: If category not found
        """
        if category not in self._category_to_index:
            raise ValueError(
                f"Category '{category}' not found. "
                f"Available categories: {', '.join(self.categories)}"
            )
        return self._category_to_index[category]
