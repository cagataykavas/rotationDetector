"""Interval aggregation and reference scoring."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from grid_motion.config import GridConfig
from grid_motion.models import CellMotion, FrameResult


class IntervalAggregator:
    def __init__(self, config: GridConfig) -> None:
        self.config = config
        self._cells: dict[int, dict[int, list[CellMotion]]] = defaultdict(lambda: defaultdict(list))

    def add(self, result: FrameResult) -> None:
        interval_index = int(result.timestamp_ms / 1000 / self.config.interval_seconds)
        for cell in result.cells:
            self._cells[interval_index][cell.cell_id].append(cell)

    def to_dict(self) -> dict[str, Any]:
        intervals: list[dict[str, Any]] = []
        for interval_index in sorted(self._cells):
            cell_results: list[dict[str, Any]] = []
            for cell_id in self.config.cell_ids:
                observations = self._cells[interval_index].get(cell_id, [])
                if not observations:
                    continue
                moving_observations = [item for item in observations if item.moving]
                moving_fraction = len(moving_observations) / len(observations)
                moving = moving_fraction >= self.config.interval_active_frame_fraction
                if moving_observations:
                    dominant_state = Counter(
                        item.state for item in moving_observations
                    ).most_common(1)[0][0]
                else:
                    dominant_state = "stationary"
                cell_results.append(
                    {
                        "cell_id": cell_id,
                        "moving": moving,
                        "dominant_state": dominant_state if moving else "stationary",
                        "frames_observed": len(observations),
                        "moving_frame_fraction": round(moving_fraction, 5),
                        "mean_magnitude": round(
                            fmean(item.mean_magnitude for item in observations), 5
                        ),
                        "mean_translation_magnitude": round(
                            fmean(item.translation_magnitude for item in observations), 5
                        ),
                        "mean_tangential_velocity": round(
                            fmean(item.mean_tangential_velocity for item in observations), 5
                        ),
                        "decision_rule": "moving_frame_fraction_at_or_above_threshold",
                        "active_frame_fraction_threshold": (
                            self.config.interval_active_frame_fraction
                        ),
                    }
                )
            intervals.append(
                {
                    "interval_index": interval_index,
                    "start_seconds": round(interval_index * self.config.interval_seconds, 5),
                    "end_seconds": round((interval_index + 1) * self.config.interval_seconds, 5),
                    "cells": cell_results,
                }
            )
        return {
            "schema_version": "1.0",
            "interval_seconds": self.config.interval_seconds,
            "intervals": intervals,
        }


def load_interval_file(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("intervals"), list):
        raise ValueError("Interval file must contain an 'intervals' JSON array")
    return payload


def evaluate_intervals(predictions: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Score moving/not-moving decisions on keys present in the reference file."""

    prediction_map = _moving_map(predictions, "predictions")
    reference_map = _moving_map(reference, "reference")
    if not reference_map:
        raise ValueError("Reference contains no cell decisions")

    true_positive = false_positive = false_negative = true_negative = 0
    missing: list[dict[str, int]] = []
    per_cell: dict[int, Counter[str]] = defaultdict(Counter)
    for key, expected in sorted(reference_map.items()):
        predicted = prediction_map.get(key)
        if predicted is None:
            missing.append({"interval_index": key[0], "cell_id": key[1]})
            predicted = False
        if expected and predicted:
            outcome = "true_positive"
            true_positive += 1
        elif not expected and predicted:
            outcome = "false_positive"
            false_positive += 1
        elif expected and not predicted:
            outcome = "false_negative"
            false_negative += 1
        else:
            outcome = "true_negative"
            true_negative += 1
        per_cell[key[1]][outcome] += 1

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (true_positive + true_negative) / len(reference_map)
    return {
        "schema_version": "1.0",
        "reference_decisions": len(reference_map),
        "confusion": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
        },
        "precision": round(precision, 5),
        "recall": round(recall, 5),
        "f1": round(f1, 5),
        "accuracy": round(accuracy, 5),
        "missing_prediction_keys": missing,
        "per_cell_confusion": {
            str(cell_id): dict(sorted(counter.items()))
            for cell_id, counter in sorted(per_cell.items())
        },
        "metric_note": "Scores only moving/not-moving decisions present in the reference.",
    }


def _moving_map(payload: dict[str, Any], name: str) -> dict[tuple[int, int], bool]:
    result: dict[tuple[int, int], bool] = {}
    for interval in payload.get("intervals", []):
        if not isinstance(interval, dict) or "interval_index" not in interval:
            raise ValueError(f"Every {name} interval must contain interval_index")
        interval_index = int(interval["interval_index"])
        cells = interval.get("cells")
        if not isinstance(cells, list):
            raise ValueError(f"Every {name} interval must contain a cells array")
        for cell in cells:
            if not isinstance(cell, dict) or "cell_id" not in cell or "moving" not in cell:
                raise ValueError(f"Every {name} cell must contain cell_id and moving")
            key = (interval_index, int(cell["cell_id"]))
            if key in result:
                raise ValueError(f"Duplicate {name} decision for interval {key[0]}, cell {key[1]}")
            if not isinstance(cell["moving"], bool):
                raise ValueError(f"{name} moving values must be booleans")
            result[key] = cell["moving"]
    return result
