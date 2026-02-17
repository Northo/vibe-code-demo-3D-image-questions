# Demo of how code assistant may solve medical imaging tasks

## Installation

```bash
# Install dependencies
uv sync

# Or with pip
pip install -e .

# Install with development dependencies (for testing)
uv sync --extra dev
```

## Features

### Category Adjacency Checking

Check if voxels of one category are adjacent to voxels of another category in 3D medical images.

```python
import numpy as np
from src.imaging.adjacency import check_green_touches_red

# Create a 3D image with categorical values
image = np.array([
    [[1, 2, 0],
     [0, 0, 0],
     [0, 0, 0]],
    # ... more slices
])

# Map integer values to category names
category_map = {0: "background", 1: "green", 2: "red"}

# Check if green touches red
result = check_green_touches_red(image, category_map)
print(f"Green touches red: {result}")
```

For more examples, see `examples/adjacency_example.py`.

## Running Tests

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
pytest tests/
```

