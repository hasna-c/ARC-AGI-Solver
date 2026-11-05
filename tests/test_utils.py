import pytest
import numpy as np
from src.utils import serialize_grid, deserialize_grid, is_valid_grid

def test_serialize_grid():
    grid = [[1, 2], [3, 4]]
    expected = "1 2\n3 4"
    assert serialize_grid(grid) == expected

def test_deserialize_grid():
    text_str = "1 2\n3 4"
    expected = [[1, 2], [3, 4]]
    assert deserialize_grid(text_str) == expected

def test_is_valid_grid():
    valid_grid = [[0, 1], [2, 3]]
    invalid_size_grid = [[1] * 35] 
    invalid_color_grid = [[10, 1]] 

    assert is_valid_grid(valid_grid) is True
    assert is_valid_grid(invalid_size_grid) is False
    assert is_valid_grid(invalid_color_grid) is False