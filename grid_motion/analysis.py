"""Dense-flow decomposition into per-cell translation and rotation evidence."""

from __future__ import annotations

import time
from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from grid_motion.config import GridConfig
from grid_motion.geometry import detect_grid, rectify
from grid_motion.models import CellMotion, FrameResult, GridEstimate

STATE_COLORS = {
    "warmup": (150, 150, 150),
    "stationary": (60, 60, 220),
    "translating": (50, 210, 240),
    "rotating_clockwise": (60, 220, 60),
    "rotating_counterclockwise": (230, 130, 40),
    "complex_motion": (220, 60, 220),
}

STATE_LABELS = {
    "warmup": "warmup",
    "stationary": "still",
    "translating": "translate",
    "rotating_clockwise": "rotate CW",
    "rotating_counterclockwise": "rotate CCW",
    "complex_motion": "complex",
}


def flow_to_color(flow: NDArray[np.float32]) -> NDArray[np.uint8]:
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = np.mod(angle * 90 / np.pi, 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class GridMotionAnalyzer:
    def __init__(self, config: GridConfig | None = None) -> None:
        self.config = config or GridConfig()
        self.config.validate()
        self.grid: GridEstimate | None = None
        self.previous_gray: NDArray[np.uint8] | None = None

    def process(
        self, frame: NDArray[np.uint8], frame_index: int, fps: float
    ) -> tuple[FrameResult, NDArray[np.uint8]]:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a non-empty BGR image with shape (height, width, 3)")
        started = time.perf_counter()
        if self.grid is None:
            self.grid = detect_grid(frame, self.config)
        rectified = rectify(frame, self.grid, self.config)
        gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)

        if self.previous_gray is None:
            cells = self._warmup_cells()
            flow = np.zeros((*gray.shape, 2), dtype=np.float32)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                self.previous_gray,
                gray,
                None,
                self.config.flow_pyr_scale,
                self.config.flow_levels,
                self.config.flow_window_size,
                self.config.flow_iterations,
                self.config.flow_poly_n,
                self.config.flow_poly_sigma,
                0,
            )
            cells = self._analyze_cells(flow)
        self.previous_gray = gray

        result = FrameResult(
            frame_index=frame_index,
            timestamp_ms=frame_index * 1000 / max(fps, 0.001),
            grid=self.grid,
            cells=tuple(cells),
            processing_ms=(time.perf_counter() - started) * 1000,
        )
        return result, self._annotate(rectified, flow, cells, frame_index)

    def _warmup_cells(self) -> list[CellMotion]:
        return [
            CellMotion(
                cell_id=cell_id,
                row=index // self.config.grid_cols,
                column=index % self.config.grid_cols,
                state="warmup",
                moving=False,
                mean_magnitude=0.0,
                active_pixel_fraction=0.0,
                translation_magnitude=0.0,
                mean_tangential_velocity=0.0,
                mean_absolute_tangential_velocity=0.0,
                tangential_coherence=0.0,
                evidence={"decision_rule": "no_previous_frame"},
            )
            for index, cell_id in enumerate(self.config.cell_ids)
        ]

    def _analyze_cells(self, flow: NDArray[np.float32]) -> list[CellMotion]:
        cell_width = self.config.canonical_width // self.config.grid_cols
        cell_height = self.config.canonical_height // self.config.grid_rows
        cells: list[CellMotion] = []
        for index, cell_id in enumerate(self.config.cell_ids):
            row, column = divmod(index, self.config.grid_cols)
            x_start = column * cell_width
            x_end = (
                self.config.canonical_width
                if column == self.config.grid_cols - 1
                else (column + 1) * cell_width
            )
            y_start = row * cell_height
            y_end = (
                self.config.canonical_height
                if row == self.config.grid_rows - 1
                else (row + 1) * cell_height
            )
            cell_flow = flow[y_start:y_end, x_start:x_end]
            cells.append(self._cell_metrics(cell_id, row, column, cell_flow))
        return cells

    def _cell_metrics(
        self,
        cell_id: int,
        row: int,
        column: int,
        flow: NDArray[np.float32],
    ) -> CellMotion:
        velocity_x = flow[..., 0]
        velocity_y = flow[..., 1]
        magnitude = np.hypot(velocity_x, velocity_y)
        active = magnitude >= self.config.pixel_motion_threshold
        active_fraction = float(active.mean())
        mean_magnitude = float(magnitude.mean())
        moving = active_fraction >= self.config.cell_active_pixel_fraction

        if moving and active.any():
            translation_x = float(velocity_x[active].mean())
            translation_y = float(velocity_y[active].mean())
            translation_magnitude = float(np.hypot(translation_x, translation_y))

            y_coordinates, x_coordinates = np.indices(magnitude.shape, dtype=np.float32)
            relative_x = x_coordinates - (magnitude.shape[1] - 1) / 2
            relative_y = y_coordinates - (magnitude.shape[0] - 1) / 2
            radius = np.hypot(relative_x, relative_y)
            valid = active & (radius > 3)
            if valid.any():
                clockwise_x = -relative_y[valid] / radius[valid]
                clockwise_y = relative_x[valid] / radius[valid]
                tangent = velocity_x[valid] * clockwise_x + velocity_y[valid] * clockwise_y
                mean_tangent = float(tangent.mean())
                mean_absolute_tangent = float(np.abs(tangent).mean())
                coherence = abs(mean_tangent) / max(mean_absolute_tangent, 1e-6)
            else:
                mean_tangent = mean_absolute_tangent = coherence = 0.0
        else:
            translation_magnitude = 0.0
            mean_tangent = mean_absolute_tangent = coherence = 0.0

        if not moving:
            state = "stationary"
            decision_rule = "insufficient_active_pixels"
        elif (
            abs(mean_tangent) >= self.config.rotation_velocity_threshold
            and coherence >= self.config.rotation_coherence_threshold
        ):
            state = "rotating_clockwise" if mean_tangent > 0 else "rotating_counterclockwise"
            decision_rule = "coherent_tangential_flow"
        elif translation_magnitude >= self.config.translation_velocity_threshold:
            state = "translating"
            decision_rule = "coherent_mean_flow"
        else:
            state = "complex_motion"
            decision_rule = "active_noncoherent_flow"

        return CellMotion(
            cell_id=cell_id,
            row=row,
            column=column,
            state=state,
            moving=moving,
            mean_magnitude=mean_magnitude,
            active_pixel_fraction=active_fraction,
            translation_magnitude=translation_magnitude,
            mean_tangential_velocity=mean_tangent,
            mean_absolute_tangential_velocity=mean_absolute_tangent,
            tangential_coherence=coherence,
            evidence={
                "decision_rule": decision_rule,
                "pixel_motion_threshold": self.config.pixel_motion_threshold,
                "minimum_active_pixel_fraction": self.config.cell_active_pixel_fraction,
                "translation_velocity_threshold": self.config.translation_velocity_threshold,
                "rotation_velocity_threshold": self.config.rotation_velocity_threshold,
                "rotation_coherence_threshold": self.config.rotation_coherence_threshold,
                "tangential_sign_convention": "positive_is_clockwise_in_image_coordinates",
            },
        )

    def _annotate(
        self,
        rectified: NDArray[np.uint8],
        flow: NDArray[np.float32],
        cells: Sequence[CellMotion],
        frame_index: int,
    ) -> NDArray[np.uint8]:
        canvas = rectified.copy()
        cell_width = self.config.canonical_width // self.config.grid_cols
        cell_height = self.config.canonical_height // self.config.grid_rows
        for cell in cells:
            x_start = cell.column * cell_width
            x_end = (
                self.config.canonical_width
                if cell.column == self.config.grid_cols - 1
                else (cell.column + 1) * cell_width
            )
            y_start = cell.row * cell_height
            y_end = (
                self.config.canonical_height
                if cell.row == self.config.grid_rows - 1
                else (cell.row + 1) * cell_height
            )
            color = STATE_COLORS[cell.state]
            cv2.rectangle(canvas, (x_start, y_start), (x_end - 1, y_end - 1), color, 3)
            cv2.putText(
                canvas,
                f"C{cell.cell_id} {STATE_LABELS[cell.state]}",
                (x_start + 7, y_start + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"active={cell.active_pixel_fraction:.2f} tan={cell.mean_tangential_velocity:.2f}",
                (x_start + 7, y_start + 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
        flow_view = flow_to_color(flow)
        cv2.putText(
            canvas,
            f"frame={frame_index} grid={self.grid.source if self.grid else 'unknown'}",
            (10, self.config.canonical_height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return np.hstack((canvas, flow_view))
