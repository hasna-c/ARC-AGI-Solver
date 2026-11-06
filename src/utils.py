import json
import re
import numpy as np

def serialize_grid(grid):
    """Converts a 2D list grid into a text representation for model prompts."""
    if not grid or not grid[0]:
        return ""
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)

def deserialize_grid(text_str):
    """Extracts the last valid ARC grid from model text, ignoring malformed trailing text."""
    if not isinstance(text_str, str) or not text_str.strip():
        return []

    marker = re.search(
        r"(?:###\s*)?(?:Predicted Output Grid|Predicted Output)\s*:\s*",
        text_str,
        flags=re.IGNORECASE,
    )
    if marker:
        text_str = text_str[marker.end():]

    # Try JSON parsing first
    decoder = json.JSONDecoder()
    json_grids = []
    for match in re.finditer(r"\[", text_str):
        try:
            candidate, _ = decoder.raw_decode(text_str[match.start():])
        except json.JSONDecodeError:
            continue
        if is_valid_grid(candidate):
            json_grids.append(candidate)
    if json_grids:
        return json_grids[-1]

    # Row-by-row line parsing with strict length filtering
    rows = []
    candidates = []
    for line in text_str.splitlines():
        line = line.strip()
        if not line or not re.fullmatch(r"[\s\[\]\(\)\{\},;|0-9]+", line):
            if rows:
                candidates.append(rows)
                rows = []
            continue

        values = [int(value) for value in re.findall(r"\d+", line)]
        
        # ARC grids are typically between 1 and 30 elements wide. 
        # If a line has too many values (like repeating token glitches), skip it!
        if values and all(0 <= value <= 9 for value in values) and len(values) <= 30:
            rows.append(values)
        else:
            if rows:
                candidates.append(rows)
                rows = []

    if rows:
        candidates.append(rows)

    valid_candidates = [grid for grid in candidates if is_valid_grid(grid)]
    return valid_candidates[-1] if valid_candidates else []

def is_valid_grid(grid):
    """Validates if a grid is a proper 2D ARC matrix within size and color constraints."""
    try:
        arr = np.array(grid)
        if arr.ndim != 2 or arr.size == 0 or any(not 0 < s <= 30 for s in arr.shape):
            return False
        if not np.issubdtype(arr.dtype, np.integer):
            return False
        if arr.min() < 0 or arr.max() > 9:
            return False
        return True
    except Exception:
        return False