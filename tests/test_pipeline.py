import json

from grid_motion.demo import generate_synthetic_frames
from grid_motion.pipeline import analyze_frames


def test_demo_writes_frame_and_interval_contracts(tmp_path):
    output = tmp_path / "demo"

    summary = analyze_frames(
        generate_synthetic_frames(12),
        fps=12.0,
        output_dir=output,
        write_video=False,
        input_metadata={"kind": "synthetic-test"},
    )
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    intervals = json.loads((output / "intervals.json").read_text())

    assert summary == json.loads((output / "summary.json").read_text())
    assert summary["frames_processed"] == 12
    assert summary["interval_count"] == 1
    assert summary["artifacts"]["video"] is None
    assert (output / "preview.jpg").stat().st_size > 0
    assert len(events) == 12
    assert len(events[0]["cells"]) == 9
    assert len(intervals["intervals"][0]["cells"]) == 9
