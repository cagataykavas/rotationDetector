"""Video I/O and reproducible grid-motion artifact generation."""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from grid_motion.aggregation import IntervalAggregator
from grid_motion.analysis import GridMotionAnalyzer
from grid_motion.config import GridConfig
from grid_motion.models import FrameResult


class FrameProcessor:
    def __init__(self, config: GridConfig | None = None) -> None:
        self.config = config or GridConfig()
        self.config.validate()
        self.analyzer = GridMotionAnalyzer(self.config)
        self.aggregator = IntervalAggregator(self.config)

    def process(
        self, frame: NDArray[np.uint8], frame_index: int, fps: float
    ) -> tuple[FrameResult, NDArray[np.uint8]]:
        result, annotated = self.analyzer.process(frame, frame_index, fps)
        self.aggregator.add(result)
        return result, annotated


def analyze_frames(
    frames: Iterable[NDArray[np.uint8]],
    *,
    fps: float,
    output_dir: str | Path,
    config: GridConfig | None = None,
    max_frames: int | None = None,
    write_video: bool = True,
    input_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided")

    processor = FrameProcessor(config)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    events_path = destination / "events.jsonl"
    intervals_path = destination / "intervals.json"
    summary_path = destination / "summary.json"
    preview_path = destination / "preview.jpg"
    video_path = destination / "annotated.mp4"

    source_frames_seen = 0
    frames_processed = 0
    processing_total_ms = 0.0
    state_counts: Counter[str] = Counter()
    grid_source_counts: Counter[str] = Counter()
    writer: cv2.VideoWriter | None = None
    last_annotated: NDArray[np.uint8] | None = None
    wall_started = time.perf_counter()

    with events_path.open("w", encoding="utf-8") as handle:
        for frame_index, frame in enumerate(frames):
            if max_frames is not None and source_frames_seen >= max_frames:
                break
            source_frames_seen += 1
            if frame_index % processor.config.frame_stride != 0:
                continue
            result, annotated = processor.process(frame, frame_index, fps)
            handle.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
            if write_video and writer is None:
                height, width = annotated.shape[:2]
                candidate = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps / processor.config.frame_stride,
                    (width, height),
                )
                if candidate.isOpened():
                    writer = candidate
                else:
                    candidate.release()
            if writer is not None:
                writer.write(annotated)
            for cell in result.cells:
                state_counts[cell.state] += 1
            grid_source_counts[result.grid.source] += 1
            processing_total_ms += result.processing_ms
            frames_processed += 1
            last_annotated = annotated

    if writer is not None:
        writer.release()
    elif video_path.exists():
        video_path.unlink()
    if frames_processed == 0 or last_annotated is None:
        raise ValueError("No frames were processed")
    if not cv2.imwrite(str(preview_path), last_annotated):
        raise OSError(f"Could not write preview image: {preview_path}")

    intervals = processor.aggregator.to_dict()
    with intervals_path.open("w", encoding="utf-8") as handle:
        json.dump(intervals, handle, indent=2, sort_keys=True)
        handle.write("\n")

    wall_seconds = time.perf_counter() - wall_started
    video_written = video_path.exists() and video_path.stat().st_size > 0
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "source_frames_seen": source_frames_seen,
        "frames_processed": frames_processed,
        "fps_reported": round(float(fps), 5),
        "interval_count": len(intervals["intervals"]),
        "state_observations": dict(sorted(state_counts.items())),
        "grid_source_observations": dict(sorted(grid_source_counts.items())),
        "mean_processing_ms": round(processing_total_ms / frames_processed, 5),
        "wall_seconds": round(wall_seconds, 5),
        "throughput_fps": round(frames_processed / max(wall_seconds, 0.0001), 5),
        "configuration": processor.config.to_dict(),
        "input": input_metadata or {"kind": "frame_iterable"},
        "artifacts": {
            "events": events_path.name,
            "intervals": intervals_path.name,
            "summary": summary_path.name,
            "preview": preview_path.name,
            "video": video_path.name if video_written else None,
        },
        "metric_note": (
            "Flow states are heuristic observations; use the evaluation command with "
            "held-out labels before reporting accuracy."
        ),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def _capture_frames(capture: cv2.VideoCapture) -> Iterator[NDArray[np.uint8]]:
    while True:
        ok, frame = capture.read()
        if not ok:
            return
        yield frame


def analyze_video(
    input_path: str | Path,
    *,
    output_dir: str | Path,
    config: GridConfig | None = None,
    max_frames: int | None = None,
    write_video: bool = True,
) -> dict[str, Any]:
    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(f"Input video does not exist: {source}")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"OpenCV could not open input video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    metadata = {
        "kind": "video",
        "path": str(source),
        "reported_frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    try:
        return analyze_frames(
            _capture_frames(capture),
            fps=fps,
            output_dir=output_dir,
            config=config,
            max_frames=max_frames,
            write_video=write_video,
            input_metadata=metadata,
        )
    finally:
        capture.release()
