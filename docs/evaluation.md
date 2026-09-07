# Evaluation guide

The generated demo validates integration and makes failures reproducible. It is not a
substitute for held-out labeled video.

## Data split

Split by recording session or physical setup. Random frame splitting leaks nearly
identical neighboring frames across train and test. Record camera, lens, resolution,
FPS, lighting, grid geometry, robot type, and motion regime.

## Labels

For the baseline moving/not-moving evaluation, label every required interval and cell
using the reference JSON contract. For state evaluation, add a reviewed `state` field
with stationary, translating, clockwise, counter-clockwise, or complex motion.

Use at least two reviewers for ambiguous transitions. Document whether a partially
active interval counts as moving and keep that policy fixed.

## Metrics

Report:

- moving/not-moving precision, recall, F1, and confusion matrix;
- per-cell metrics to expose calibration or perspective bias;
- rotation-direction accuracy on intervals labeled as rotation;
- false-motion rate on stationary footage;
- median and p95 processing time on named hardware; and
- performance by blur, lighting, occlusion, and camera-motion bucket.

Save the exact configuration and commit SHA beside results. Tune thresholds on a
validation split only, then run the held-out test once.

## Camera-motion warning

Global camera motion can make every cell active. A fixed mount or explicit video
stabilization is required before interpreting cell flow. Include camera-jolt negative
cases in the held-out set.
