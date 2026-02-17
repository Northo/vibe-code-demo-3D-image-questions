# Demo of how code assistant may solve medical imaging tasks

A demonstration project showing how to work with categorical 3D medical images in Python, featuring clean APIs for image manipulation and adjacency analysis.

## Installation

```bash
# Install dependencies using uv (recommended)
uv sync

# Or with pip
pip install -e .

# Install with development dependencies (for testing)
uv sync --extra dev
```

## Features

### CategoricalImage: Work with Category Names Directly

The `CategoricalImage` is a **frozen dataclass** that provides a clean, immutable interface for working with categorical medical images where voxel values represent categories (e.g., tissue types, organ labels). **Category names are first-class citizens** - you work directly with meaningful names like "liver" and "kidney" rather than remembering integer codes.

```python
import numpy as np
from src.imaging import CategoricalImage

# Create from string labels (most intuitive)
labels = np.array([[["liver", "kidney", "background"]]])
image = CategoricalImage.from_labels(labels)

# Or create from integer-encoded data with explicit mapping
data = np.array([[[0, 1, 2]]])
categories = ["background", "liver", "kidney"]
image = CategoricalImage(data, categories)

# Work with categories by name
liver_mask = image.get_mask("liver")
liver_count = image.count_voxels("liver")
stats = image.get_category_stats()  # {"background": 1, "liver": 1, "kidney": 1}

# Check if category exists
if image.has_category("spleen"):
    print("Spleen found!")
```

### Category Adjacency Checking

Check if voxels of one category are adjacent to voxels of another category in 3D space (26-connectivity: includes face, edge, and corner neighbors).

```python
from src.imaging import CategoricalImage, check_green_touches_red

# Create a 3D medical image with category labels
labels = np.array([
    [["green", "red", "background"],
     ["background", "background", "background"]],
    
    [["background", "background", "background"],
     ["background", "background", "background"]]
])

image = CategoricalImage.from_labels(labels)

# Check if green touches red - clean and simple!
result = check_green_touches_red(image)
print(f"Green touches red: {result}")  # True

# Check any category pair
from src.imaging import check_category_adjacency

# Create image with organ labels
organ_labels = np.array([
    [["liver", "kidney", "background"],
     ["liver", "background", "kidney"]],
    
    [["spleen", "background", "background"],
     ["background", "background", "background"]]
])

organ_image = CategoricalImage.from_labels(organ_labels)

# Check if liver and kidney are adjacent
liver_kidney_touch = check_category_adjacency(organ_image, "liver", "kidney")
print(f"Liver and kidney touch: {liver_kidney_touch}")  # True
```

### Key Features

- **Frozen dataclass**: Immutable, hashable, and type-safe by design
- **Category names as first-class citizens**: Work with meaningful names instead of integer codes
- **Multiple creation methods**: Create from string labels or integer-encoded arrays
- **Type-safe operations**: Comprehensive validation and clear error messages
- **Efficient algorithms**: Uses scipy's binary dilation for fast neighbor checking
- **26-connectivity**: Detects adjacency including face, edge, and corner neighbors
- **Rich API**: Get masks, count voxels, check existence, and more

## Examples

Run the comprehensive example to see all features:

```bash
python examples/adjacency_example.py
```

This demonstrates:
- Creating images from labels vs encoded data
- Checking green-red adjacency
- Testing 3D diagonal neighbors
- Working with arbitrary category pairs (organs)
- Using category masks for analysis
- Getting category statistics

## Running Tests

```bash
# Install dev dependencies
uv sync --extra dev

# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_categorical.py
pytest tests/test_adjacency.py

# Run with verbose output
pytest -v tests/
```

The test suite includes:
- **test_categorical.py**: 20+ tests for CategoricalImage data structure
- **test_adjacency.py**: 15+ tests for adjacency checking algorithms

## API Reference

### CategoricalImage

```python
@dataclass(frozen=True)
class CategoricalImage:
    """A 3D medical image with categorical voxel values (immutable)."""
    
    # Fields
    data: np.ndarray                    # Integer-encoded 3D array
    categories: tuple[str, ...]         # Category names in encoding order
    
    # Creation
    def __init__(data: np.ndarray, categories: list[str] | tuple[str, ...])
    @classmethod
    def from_labels(labels: np.ndarray) -> CategoricalImage
    
    # Properties
    @property
    def shape -> tuple[int, int, int]
    
    # Category operations
    def has_category(category: str) -> bool
    def get_mask(category: str) -> np.ndarray  # Boolean mask
    def count_voxels(category: str) -> int
    def get_category_stats() -> dict[str, int]
    def get_labels() -> np.ndarray  # Convert back to string labels
```

### Adjacency Functions

```python
def check_green_touches_red(
    image: CategoricalImage | np.ndarray,
    category_map: dict[int, str] | None = None
) -> bool:
    """Check if green voxels are adjacent to red voxels."""

def check_category_adjacency(
    image: CategoricalImage | np.ndarray,
    category1: str,
    category2: str,
    category_map: dict[int, str] | None = None
) -> bool:
    """Check if any voxels of category1 are adjacent to category2."""
```

## Design Philosophy

This project demonstrates best practices for medical imaging software:

1. **Domain-driven design**: Category names (like "liver", "kidney") are the primary interface, not implementation details like integer encoding
2. **Clean abstractions**: `CategoricalImage` is a frozen dataclass that encapsulates data and metadata together immutably
3. **Type safety**: Full type annotations and comprehensive validation with dataclass benefits
4. **Immutability**: Frozen dataclass prevents accidental modifications
5. **Backward compatibility**: Legacy numpy array API still supported
6. **Performance**: Efficient numpy/scipy operations for large 3D volumes
7. **Testability**: Comprehensive test coverage with clear examples

## Project Structure

```
vibe-code-demo-rognes/
├── src/
│   └── imaging/
│       ├── categorical.py      # CategoricalImage frozen dataclass
│       ├── adjacency.py        # Adjacency checking functions
│       └── __init__.py
├── tests/
│   ├── test_categorical.py     # Tests for CategoricalImage
│   └── test_adjacency.py       # Tests for adjacency functions
├── examples/
│   └── adjacency_example.py    # Comprehensive usage examples
└── pyproject.toml              # Dependencies and configuration
```

## Contributing

This is a demonstration project. For questions or improvements, please refer to the AGENTS.md file for code style guidelines.
