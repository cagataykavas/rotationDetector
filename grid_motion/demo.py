"""Deterministic perspective-grid generator for integration testing."""

from __future__ import annotations

import math
from collections.abc import Iterator

import cv2
import numpy as np
from numpy.typing import NDArray

BOARD_SIZE = 600
CELL_SIZE = 200
BOARD_COLOR = (218, 220, 222)


def _rotor_tile(angle: float, seed: int) -> NDArray[np.uint8]:
    tile = np.full((CELL_SIZE, CELL_SIZE, 3), BOARD_COLOR, dtype=np.uint8)
    center = (CELL_SIZE // 2, CELL_SIZE // 2)
    cv2.circle(tile, center, 66, (160, 165, 170), -1, cv2.LINE_AA)
    cv2.circle(tile, center, 66, (30, 30, 30), 3, cv2.LINE_AA)
    for spoke in range(8):
        theta = spoke * math.tau / 8
        endpoint = (
            int(center[0] + 58 * math.cos(theta)),
            int(center[1] + 58 * math.sin(theta)),
        )
        cv2.line(tile, center, endpoint, (35, 35, 35), 5, cv2.LINE_AA)
    rng = np.random.default_rng(seed)
    for _ in range(18):
        theta = float(rng.uniform(0, math.tau))
        radius = float(rng.uniform(18, 58))
        point = (
            int(center[0] + radius * math.cos(theta)),
            int(center[1] + radius * math.sin(theta)),
        )
        color = (25, 60, 220) if rng.random() > 0.5 else (220, 90, 25)
        cv2.circle(tile, point, 4, color, -1, cv2.LINE_AA)
    transform = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        tile,
        transform,
        (CELL_SIZE, CELL_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=BOARD_COLOR,
    )


def generate_synthetic_frames(
    frame_count: int = 48, *, width: int = 800, height: int = 720
) -> Iterator[NDArray[np.uint8]]:
    """Yield a board where cell 1/5 rotate and cell 9 translates."""

    if frame_count < 6:
        raise ValueError("Synthetic demo requires at least six frames")
    if width < 640 or height < 640:
        raise ValueError("Synthetic frame dimensions must be at least 640 x 640")

    source_quad = np.float32(
        [[95, 42], [width - 96, 68], [width - 48, height - 42], [48, height - 68]]
    )
    destination_quad = np.float32(
        [[0, 0], [BOARD_SIZE - 1, 0], [BOARD_SIZE - 1, BOARD_SIZE - 1], [0, BOARD_SIZE - 1]]
    )
    board_to_source = cv2.getPerspectiveTransform(destination_quad, source_quad)
    board_mask = np.full((BOARD_SIZE, BOARD_SIZE), 255, dtype=np.uint8)

    for frame_index in range(frame_count):
        board = np.full((BOARD_SIZE, BOARD_SIZE, 3), BOARD_COLOR, dtype=np.uint8)
        for cell_index in range(9):
            row, column = divmod(cell_index, 3)
            y_start = row * CELL_SIZE
            x_start = column * CELL_SIZE
            if cell_index == 0:
                tile = _rotor_tile(frame_index * 5.0, seed=11)
            elif cell_index == 4:
                tile = _rotor_tile(-frame_index * 4.0, seed=29)
            elif cell_index == 8:
                tile = np.full((CELL_SIZE, CELL_SIZE, 3), BOARD_COLOR, dtype=np.uint8)
                offset = int(52 * math.sin(frame_index * math.tau / 18))
                center = (CELL_SIZE // 2 + offset, CELL_SIZE // 2)
                cv2.rectangle(
                    tile,
                    (center[0] - 26, center[1] - 26),
                    (center[0] + 26, center[1] + 26),
                    (45, 55, 65),
                    -1,
                )
                cv2.line(
                    tile,
                    (center[0] - 22, center[1]),
                    (center[0] + 22, center[1]),
                    (230, 230, 230),
                    4,
                )
            else:
                tile = _rotor_tile(0.0, seed=100 + cell_index)
            board[y_start : y_start + CELL_SIZE, x_start : x_start + CELL_SIZE] = tile

        for position in (0, CELL_SIZE, CELL_SIZE * 2, BOARD_SIZE - 1):
            cv2.line(board, (position, 0), (position, BOARD_SIZE - 1), (15, 15, 15), 4)
            cv2.line(board, (0, position), (BOARD_SIZE - 1, position), (15, 15, 15), 4)
        frame = np.full((height, width, 3), (34, 40, 46), dtype=np.uint8)
        warped_board = cv2.warpPerspective(board, board_to_source, (width, height))
        warped_mask = cv2.warpPerspective(board_mask, board_to_source, (width, height))
        frame[warped_mask > 0] = warped_board[warped_mask > 0]
        cv2.putText(
            frame,
            "SYNTHETIC GRID - C1/C5 ROTATE, C9 TRANSLATES",
            (20, height - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        yield frame
