"""Image processing and analysis functions for medical imaging."""

from .adjacency import check_category_adjacency, check_green_touches_red
from .categorical import CategoricalImage

__all__ = [
    "CategoricalImage",
    "check_category_adjacency",
    "check_green_touches_red",
]
