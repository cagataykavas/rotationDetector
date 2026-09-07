"""Command-line entry points for analysis, demo, evaluation, and schema inspection."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from grid_motion.aggregation import evaluate_intervals, load_interval_file
from grid_motion.config import GridConfig
from grid_motion.demo import generate_synthetic_frames
from grid_motion.pipeline import analyze_frames, analyze_video


def json_contract() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "frame_event": {
            "frame_index": "original source-frame index",
            "timestamp_ms": "source-frame timestamp",
            "grid": "observed quadrilateral or explicit full-frame fallback",
            "cells": [
                {
                    "cell_id": "configured identifier",
                    "state": (
                        "warmup | stationary | translating | rotating_clockwise | "
                        "rotating_counterclockwise | complex_motion"
                    ),
                    "moving": "frame-level boolean",
                    "mean_magnitude": "mean dense-flow magnitude",
                    "active_pixel_fraction": "fraction above pixel threshold",
                    "translation_magnitude": "magnitude of mean active flow vector",
                    "mean_tangential_velocity": "positive means clockwise in image coordinates",
                    "tangential_coherence": "absolute signed mean / mean absolute tangent",
                    "evidence": "decision rule and exact thresholds",
                }
            ],
        },
        "intervals": "moving-frame aggregation by interval and cell",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grid-motion",
        description="Explainable grid motion and image-plane rotation analysis",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="analyze a real video")
    analyze.add_argument("--input", required=True, type=Path)
    analyze.add_argument("--output", required=True, type=Path)
    analyze.add_argument("--config", type=Path)
    analyze.add_argument("--max-frames", type=int)
    analyze.add_argument("--no-video", action="store_true")

    demo = subparsers.add_parser("demo", help="run the generated grid scene")
    demo.add_argument("--output", required=True, type=Path)
    demo.add_argument("--frames", type=int, default=48)
    demo.add_argument("--fps", type=float, default=12.0)
    demo.add_argument("--config", type=Path)
    demo.add_argument("--no-video", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="score interval predictions")
    evaluate.add_argument("--predictions", required=True, type=Path)
    evaluate.add_argument("--reference", required=True, type=Path)
    evaluate.add_argument("--output", type=Path)

    subparsers.add_parser("explain-schema", help="print the JSON contracts")
    return parser


def _load_config(path: Path | None) -> GridConfig:
    config = GridConfig.from_json(path) if path else GridConfig()
    config.validate()
    return config


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "explain-schema":
            print(json.dumps(json_contract(), indent=2, sort_keys=True))
            return 0
        if args.command == "evaluate":
            metrics = evaluate_intervals(
                load_interval_file(args.predictions), load_interval_file(args.reference)
            )
            output = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(output, encoding="utf-8")
            print(output, end="")
            return 0

        config = _load_config(args.config)
        if args.command == "analyze":
            summary = analyze_video(
                args.input,
                output_dir=args.output,
                config=config,
                max_frames=args.max_frames,
                write_video=not args.no_video,
            )
        else:
            summary = analyze_frames(
                generate_synthetic_frames(args.frames),
                fps=args.fps,
                output_dir=args.output,
                config=config,
                write_video=not args.no_video,
                input_metadata={
                    "kind": "synthetic",
                    "generator": "grid_motion.demo.generate_synthetic_frames",
                    "frames_requested": args.frames,
                    "moving_cells": [1, 5, 9],
                    "ground_truth_metrics": False,
                },
            )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
