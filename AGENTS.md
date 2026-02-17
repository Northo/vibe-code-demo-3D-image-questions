# Agent Guidelines for vibe-code-demo-rognes

This document provides essential information for AI coding agents working on this medical imaging demonstration project.

## Project Overview

- **Purpose**: Demonstration of code assistant solving medical imaging tasks
- **Language**: Python 3.14+
- **Build System**: uv (pyproject.toml-based)
- **Domain**: Medical imaging and healthcare applications

## Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync
# or
pip install -e .
```


## Code Style Guidelines

### Import Organization

Organize imports in the following order (separated by blank lines):
1. Standard library imports
2. Third-party imports
3. Local application imports

```python
# Standard library
import os
from pathlib import Path
from typing import Optional

# Third-party
import numpy as np
import SimpleITK as sitk

# Local
from src.imaging.loader import load_dicom
from src.utils.validation import validate_image
```

### Formatting

- **Line length**: Maximum 100 characters
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Prefer double quotes for strings
- **Trailing commas**: Use in multi-line collections

### Type Annotations

Always use type annotations for function signatures:

```python
def process_image(
    image_path: Path,
    normalize: bool = True,
    output_format: str = "nifti"
) -> np.ndarray:
    """Process medical image with optional normalization."""
    ...
```

Use modern Python 3.10+ type syntax:
- `list[str]` instead of `List[str]`
- `dict[str, int]` instead of `Dict[str, int]`
- `str | None` instead of `Optional[str]`

### Naming Conventions

- **Functions/variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: Prefix with single underscore `_private_method`
- **Module names**: Short, lowercase, no underscores if possible

### Error Handling

Use specific exception types and provide meaningful error messages:

```python
def load_medical_image(path: Path) -> sitk.Image:
    """Load medical image from file."""
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    
    try:
        return sitk.ReadImage(str(path))
    except RuntimeError as e:
        raise ValueError(f"Failed to read image from {path}: {e}") from e
```

### Documentation

Use Google-style docstrings:

```python
def segment_organ(
    image: np.ndarray,
    organ_type: str,
    confidence_threshold: float = 0.5
) -> tuple[np.ndarray, float]:
    """Segment organ from medical image.
    
    Args:
        image: 3D medical image array (H, W, D)
        organ_type: Type of organ to segment (e.g., "liver", "kidney")
        confidence_threshold: Minimum confidence for segmentation
        
    Returns:
        Tuple of (segmentation_mask, confidence_score)
        
    Raises:
        ValueError: If organ_type is not supported
    """
    ...
```

## Medical Imaging Specific Guidelines

### Data description

- Our data are 3D images consisting of voxels. They are represented as numpy arrays.
- Each voxel has an integer value, which represents a categorical value. The values are not to be interpreted numerically.
- When we say any 1 touch 2, we mean if any voxel with the value 1 has any neighbour with the value 2.


### Data Handling

- Use `pathlib.Path` for file paths, not string manipulation

### Performance

- Use numpy vectorized operations over loops
- Consider memory usage for large 3D volumes
- Implement lazy loading for image datasets
- Profile code for bottlenecks in image processing pipelines

