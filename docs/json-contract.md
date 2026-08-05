# JSON contracts

## Frame events

`events.jsonl` contains one complete event for each processed source frame.

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string | Currently `1.0` |
| `frame_index` | integer | Original source index, even with frame stride |
| `timestamp_ms` | number | Source timestamp from reported FPS |
| `processing_ms` | number | Rectification, flow, and cell-analysis wall time |
| `grid` | object | Board source, score, corners, and evidence |
| `cells` | array | One motion result per configured cell ID |

### Cell result

| Field | Meaning |
|---|---|
| `cell_id`, `row`, `column` | Configured identity and zero-based grid location |
| `state` | Warmup, stationary, translation, rotation direction, or complex motion |
| `moving` | Whether enough pixels moved in this frame |
| `mean_magnitude` | Mean dense-flow magnitude across the cell |
| `active_pixel_fraction` | Fraction above the pixel-motion threshold |
| `translation_magnitude` | Magnitude of the mean active-flow vector |
| `mean_tangential_velocity` | Signed tangent; positive is clockwise in image coordinates |
| `mean_absolute_tangential_velocity` | Unsigned tangent strength |
| `tangential_coherence` | Directional consistency from 0 to 1 |
| `evidence` | Decision rule, thresholds, and sign convention |

## Interval predictions

`intervals.json` contains:

```json
{
  "schema_version": "1.0",
  "interval_seconds": 1.0,
  "intervals": [
    {
      "interval_index": 0,
      "start_seconds": 0.0,
      "end_seconds": 1.0,
      "cells": [
        {
          "cell_id": 1,
          "moving": true,
          "dominant_state": "rotating_clockwise",
          "frames_observed": 12,
          "moving_frame_fraction": 0.91667,
          "mean_magnitude": 1.42,
          "mean_translation_magnitude": 0.08,
          "mean_tangential_velocity": 1.17,
          "decision_rule": "moving_frame_fraction_at_or_above_threshold",
          "active_frame_fraction_threshold": 0.35
        }
      ]
    }
  ]
}
```

Numbers are illustrative, not benchmark results.

## Reference input

The evaluation command requires only `interval_index`, `cell_id`, and boolean `moving`
fields. See [`reference.example.json`](../reference.example.json). Extra fields are
allowed. Duplicate interval/cell keys, missing arrays, and non-boolean labels fail.

## Evaluation output

Evaluation reports the confusion matrix, precision, recall, F1, accuracy, missing
prediction keys, and per-cell confusion counts. A missing prediction is recorded and
scored as not moving; it is never silently dropped.

## Versioning

Consumers should check `schema_version` and ignore unknown additive fields. Removing a
field or changing its meaning requires a new major schema version.
