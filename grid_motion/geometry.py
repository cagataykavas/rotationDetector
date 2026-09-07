"""Grid-board discovery and perspective rectification."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from grid_motion.config import GridConfig
from grid_motion.models import GridEstimate, Point


def order_quad(points: NDArray[np.generic]) -> NDArray[np.float32]:
    quad = np.asarray(points, dtype=np.float32).reshape(4, 2)
    result = np.zeros((4, 2), dtype=np.float32)
    coordinate_sum = quad.sum(axis=1)
    coordinate_delta = np.diff(quad, axis=1).reshape(-1)
    result[0] = quad[np.argmin(coordinate_sum)]
    result[2] = quad[np.argmax(coordinate_sum)]
    result[1] = quad[np.argmin(coordinate_delta)]
    result[3] = quad[np.argmax(coordinate_delta)]
    if len({tuple(float(value) for value in point) for point in result}) != 4:
        raise ValueError("Quadrilateral points are ambiguous")
    return result


def detect_grid(frame: NDArray[np.uint8], config: GridConfig) -> GridEstimate:
    """Find the largest high-contrast quadrilateral or explicitly fall back."""

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(height * width)

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = float(cv2.contourArea(contour))
        area_ratio = area / frame_area
        if area_ratio < config.grid_min_area_ratio:
            break
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        try:
            ordered = order_quad(polygon.reshape(4, 2))
        except ValueError:
            continue
        hull_area = max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)
        solidity = min(area / hull_area, 1.0)
        confidence = min(0.99, 0.5 + 0.25 * solidity + 0.25 * min(area_ratio / 0.6, 1))
        return GridEstimate(
            source="observed_quadrilateral",
            confidence=confidence,
            corners=tuple(Point(float(x), float(y)) for x, y in ordered),
            evidence={
                "decision_rule": "largest_otsu_quadrilateral",
                "area_ratio": round(area_ratio, 5),
                "minimum_area_ratio": config.grid_min_area_ratio,
                "solidity": round(solidity, 5),
            },
        )

    fallback = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    return GridEstimate(
        source="full_frame_fallback",
        confidence=0.0,
        corners=tuple(Point(float(x), float(y)) for x, y in fallback),
        evidence={
            "decision_rule": "no_quadrilateral_above_threshold",
            "minimum_area_ratio": config.grid_min_area_ratio,
        },
    )


def rectify(
    frame: NDArray[np.uint8], estimate: GridEstimate, config: GridConfig
) -> NDArray[np.uint8]:
    source = np.array([[point.x, point.y] for point in estimate.corners], dtype=np.float32)
    destination = np.array(
        [
            [0, 0],
            [config.canonical_width - 1, 0],
            [config.canonical_width - 1, config.canonical_height - 1],
            [0, config.canonical_height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(frame, transform, (config.canonical_width, config.canonical_height))
