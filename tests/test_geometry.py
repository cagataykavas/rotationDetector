from grid_motion.config import GridConfig
from grid_motion.demo import generate_synthetic_frames
from grid_motion.geometry import detect_grid


def test_generated_board_is_detected_as_quadrilateral():
    frame = next(generate_synthetic_frames(6))

    estimate = detect_grid(frame, GridConfig())

    assert estimate.source == "observed_quadrilateral"
    assert len(estimate.corners) == 4
    assert estimate.evidence["area_ratio"] >= GridConfig().grid_min_area_ratio
