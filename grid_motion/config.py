"""Validated optical-flow and grid configuration."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class GridConfig:
    grid_rows: int = 3
    grid_cols: int = 3
    cell_ids: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9)
    canonical_width: int = 600
    canonical_height: int = 600
    grid_min_area_ratio: float = 0.25

    pixel_motion_threshold: float = 0.35
    cell_active_pixel_fraction: float = 0.03
    translation_velocity_threshold: float = 0.22
    rotation_velocity_threshold: float = 0.18
    rotation_coherence_threshold: float = 0.35

    interval_seconds: float = 1.0
    interval_active_frame_fraction: float = 0.35
    frame_stride: int = 1

    flow_pyr_scale: float = 0.5
    flow_levels: int = 3
    flow_window_size: int = 15
    flow_iterations: int = 3
    flow_poly_n: int = 5
    flow_poly_sigma: float = 1.2

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> GridConfig:
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {', '.join(unknown)}")
        prepared = dict(values)
        if "cell_ids" in prepared:
            if not isinstance(prepared["cell_ids"], (list, tuple)):
                raise ValueError("cell_ids must be a JSON array")
            prepared["cell_ids"] = tuple(int(value) for value in prepared["cell_ids"])
        config = cls(**prepared)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> GridConfig:
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("Configuration root must be a JSON object")
        return cls.from_mapping(payload)

    def validate(self) -> None:
        if self.grid_rows < 1 or self.grid_cols < 1:
            raise ValueError("grid_rows and grid_cols must be positive")
        expected_cells = self.grid_rows * self.grid_cols
        if len(self.cell_ids) != expected_cells:
            raise ValueError(f"cell_ids must contain {expected_cells} values")
        if len(set(self.cell_ids)) != len(self.cell_ids):
            raise ValueError("cell_ids must be unique")
        if self.canonical_width < self.grid_cols * 32:
            raise ValueError("canonical_width is too small for the configured columns")
        if self.canonical_height < self.grid_rows * 32:
            raise ValueError("canonical_height is too small for the configured rows")

        unit_fields = (
            "grid_min_area_ratio",
            "cell_active_pixel_fraction",
            "rotation_coherence_threshold",
            "interval_active_frame_fraction",
            "flow_pyr_scale",
        )
        for name in unit_fields:
            value = float(getattr(self, name))
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be greater than 0 and at most 1")
        positive_fields = (
            "pixel_motion_threshold",
            "translation_velocity_threshold",
            "rotation_velocity_threshold",
            "interval_seconds",
            "flow_poly_sigma",
        )
        for name in positive_fields:
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        integer_fields = (
            "frame_stride",
            "flow_levels",
            "flow_window_size",
            "flow_iterations",
            "flow_poly_n",
        )
        for name in integer_fields:
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["cell_ids"] = list(self.cell_ids)
        return result
