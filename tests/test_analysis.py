from grid_motion.analysis import GridMotionAnalyzer
from grid_motion.config import GridConfig
from grid_motion.demo import generate_synthetic_frames


def test_generated_motion_activates_intended_cells():
    analyzer = GridMotionAnalyzer(GridConfig())
    frames = generate_synthetic_frames(6)
    analyzer.process(next(frames), 0, 12.0)
    result, _ = analyzer.process(next(frames), 1, 12.0)
    cells = {cell.cell_id: cell for cell in result.cells}

    assert cells[1].moving is True
    assert cells[5].moving is True
    assert cells[9].moving is True
    assert cells[1].evidence["tangential_sign_convention"].startswith("positive")
