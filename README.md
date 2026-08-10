# Grid Motion & Rotation Analysis with Optical Flow

A classical computer-vision project for detecting a drop event and subsequently analysing motion across a **3×3 spatial grid** using dense optical flow.

The program rectifies the observed surface with a homography, divides it into labelled cells, estimates motion using Farnebäck optical flow, detects dark rover-like blobs, and can compare the estimated cell activity against optional reference / ground-truth files.

> This repository is an academic computer-vision project and is preserved as a transparent classical-CV implementation rather than presented as a general-purpose production tracker.

## Pipeline overview

The processing sequence is split into two main stages.

### 1. Drop-event detection

The video is first scanned using dense Farnebäck optical flow. Average flow magnitude is monitored using start/end thresholds to determine when a large motion event begins and settles.

The resulting frame is treated as the reference point for the subsequent motion analysis.

### 2. Grid motion analysis

After a configurable delay, the program:

1. Detects corner candidates using the Harris corner response.
2. Estimates the four outer corners of the observed surface.
3. Computes a homography and rectifies the surface to a top-down view.
4. Divides the rectified image into a 3×3 grid.
5. Computes dense optical flow between consecutive rectified frames.
6. Measures optical-flow magnitude independently for each cell.
7. Marks cells according to detected motion.
8. Detects dark blobs / rover-like objects using quantisation, morphology and contours.
9. Optionally overlays reference values loaded from a text file.
10. Records debug information for analysing threshold behaviour.

## Techniques demonstrated

- OpenCV video processing
- Dense Farnebäck optical flow
- Optical-flow magnitude / direction visualisation
- Event detection from temporal motion
- Harris corner detection
- Homography estimation
- Perspective rectification
- Spatial grid analysis
- Morphological image processing
- Contour-based blob detection
- Per-cell temporal aggregation
- Optional ground-truth comparison
- Matplotlib/OpenCV visualisation

## Repository structure

```text
rotationDetector/
├── HW2.py              # Main motion-analysis pipeline
├── requirements.txt    # Python dependencies
├── ReadMe.txt           # Original assignment instructions
└── README.md            # Project documentation
```

The original assignment expects the video dataset outside the repository under a nested directory structure similar to:

```text
videolar/
└── videolar/
    ├── test1.mp4
    ├── test2.mp4
    └── ...
```

Optional reference files can be placed beside the script:

```text
referans1.txt
referans2.txt
...
```

## Installation

Python 3 is required.

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

## Running

```bash
python HW2.py
```

The current script selects the input through the `file_index` variable inside `main()`:

```python
file_index = 1
```

For example, index `1` selects `test1.mp4` and, when present, `referans1.txt`.

## Main configuration

Important thresholds are intentionally exposed near the beginning of `HW2.py`:

```python
drop_wait_seconds = 1.1
move_threshold = 0.35
flow_mag_thresh = 0.017
debug_mode = False
```

They control the delay after the detected drop, motion classification, per-cell optical-flow sensitivity, and visual/debug behaviour.

Additional thresholds inside the drop detector determine when the initial large-motion event starts and ends.

## Perspective rectification

At the detected drop frame, the code computes a Harris corner response and uses extreme combinations of the detected coordinates to approximate the top-left, top-right, bottom-right and bottom-left corners.

A homography maps these points to a rectangular destination image. Subsequent analysis is performed in this rectified coordinate system so that the surface can be divided consistently into nine cells.

The cell IDs follow the assignment-specific arrangement:

```text
7  1  4
8  2  5
9  3  6
```

## Motion detection

Dense optical flow is computed between consecutive rectified grayscale frames. Each grid cell is evaluated independently from its local flow field.

The implementation records quantities such as mean flow magnitude and the number of pixels exceeding the configured magnitude threshold. This makes the detector easier to inspect than a single black-box classification result.

## Rover / blob detection

The rectified grayscale frame is quantised and very dark regions are isolated. Morphological closing reduces fragmentation, and contours above a minimum size are converted into bounding boxes.

These detections are visualised alongside the grid-level motion estimates.

## Reference data

If a matching `referans{index}.txt` file exists, the program reads reference values and overlays them on the corresponding cells. This provides a convenient visual comparison between predicted motion states and supplied ground truth.

The reference file is optional; the motion pipeline can operate without it.

## Limitations

The method is deliberately handcrafted and tuned to the assignment footage:

- homography estimation assumes the relevant surface can be inferred from strong corner responses;
- fixed optical-flow thresholds are sensitive to camera motion, resolution and frame rate;
- dark-object detection assumes useful intensity separation from the background;
- the 3×3 geometry and cell numbering are task-specific;
- abrupt illumination changes can appear as motion;
- Farnebäck flow is computationally more expensive than simple frame differencing;
- configuration is currently performed in the source rather than through CLI arguments.

## Possible extensions

Natural improvements would include automatic dataset discovery, command-line arguments, quantitative accuracy metrics, more robust corner estimation, camera-motion compensation, and unit tests for the temporal aggregation logic.

## Why this project is useful

Unlike a model that simply returns a label, this project exposes the full motion-analysis pipeline: event detection, optical-flow estimation, geometric rectification, spatial reasoning, blob detection, temporal aggregation and optional ground-truth comparison. It is a compact demonstration of classical video-analysis techniques implemented directly with NumPy and OpenCV.
