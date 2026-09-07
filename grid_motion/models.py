"""Serializable models for grid geometry and cell motion evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _round(value: float) -> float:
    return round(float(value), 5)


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    def to_dict(self) -> dict[str, float]:
        return {"x": _round(self.x), "y": _round(self.y)}


@dataclass(frozen=True, slots=True)
class GridEstimate:
    source: str
    confidence: float
    corners: tuple[Point, ...]
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "confidence": _round(self.confidence),
            "corners": [point.to_dict() for point in self.corners],
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class CellMotion:
    cell_id: int
    row: int
    column: int
    state: str
    moving: bool
    mean_magnitude: float
    active_pixel_fraction: float
    translation_magnitude: float
    mean_tangential_velocity: float
    mean_absolute_tangential_velocity: float
    tangential_coherence: float
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "row": self.row,
            "column": self.column,
            "state": self.state,
            "moving": self.moving,
            "mean_magnitude": _round(self.mean_magnitude),
            "active_pixel_fraction": _round(self.active_pixel_fraction),
            "translation_magnitude": _round(self.translation_magnitude),
            "mean_tangential_velocity": _round(self.mean_tangential_velocity),
            "mean_absolute_tangential_velocity": _round(self.mean_absolute_tangential_velocity),
            "tangential_coherence": _round(self.tangential_coherence),
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class FrameResult:
    frame_index: int
    timestamp_ms: float
    grid: GridEstimate
    cells: tuple[CellMotion, ...]
    processing_ms: float
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "frame_index": self.frame_index,
            "timestamp_ms": _round(self.timestamp_ms),
            "processing_ms": _round(self.processing_ms),
            "grid": self.grid.to_dict(),
            "cells": [cell.to_dict() for cell in self.cells],
        }
