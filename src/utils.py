import numpy as np

def serialize_grid(grid):
    """Converts a 2D list grid into a text representation for model prompts."""
    if not grid or not grid[0]:
        return ""
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)

def deserialize_grid(text_str):
    """Parses model text generation back into a 2D list grid."""
    grid = []
    for line in text_str.strip().split("\n"):
        row = [int(x) for x in line.strip().split() if x.isdigit()]
        if row:
            grid.append(row)
    return grid if grid else [[0]]

def is_valid_grid(grid):
    """Validates if a grid is a proper 2D ARC matrix within size and color constraints."""
    try:
        arr = np.array(grid)
        if arr.ndim != 2 or arr.size == 0 or any(not 0 < s <= 30 for s in arr.shape):
            return False
        if arr.min() < 0 or arr.max() > 9:
            return False
        return True
    except Exception:
        return False