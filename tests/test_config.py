import pytest

from grid_motion.config import GridConfig


def test_historical_cell_layout_is_supported():
    config = GridConfig.from_mapping({"cell_ids": [7, 1, 4, 8, 2, 5, 9, 3, 6]})

    assert config.cell_ids == (7, 1, 4, 8, 2, 5, 9, 3, 6)


def test_duplicate_cell_ids_are_rejected():
    with pytest.raises(ValueError, match="must be unique"):
        GridConfig.from_mapping({"cell_ids": [1] * 9})
