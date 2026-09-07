"""Explainable grid motion and image-plane rotation analysis."""

from grid_motion.config import GridConfig
from grid_motion.pipeline import FrameProcessor, analyze_frames, analyze_video

__all__ = ["FrameProcessor", "GridConfig", "analyze_frames", "analyze_video"]
__version__ = "1.0.0"
