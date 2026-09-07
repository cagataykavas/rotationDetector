# Legacy migration

The old script expected `videolar/videolar/test1.mp4`, optionally read
`referans1.txt`, waited for a global-flow drop event, analyzed exactly 60 seconds, and
wrote `ogr.txt`, `debug-output.txt`, nine cell logs, and a Matplotlib plot beside the
source code.

The repaired workflow is explicit:

```bash
grid-motion analyze --input path/to/test1.mp4 --output artifacts/test1
```

| Legacy behavior | Repaired equivalent |
|---|---|
| hard-coded input folder | required `--input` |
| implicit debug mode | headless operation by default |
| fixed 60 seconds | `--max-frames` or complete video |
| pixel/file globals | validated JSON configuration |
| `ogr.txt` movement matrix | versioned `intervals.json` |
| unstructured debug text | per-frame `events.jsonl` |
| fixed cell permutation | configurable `cell_ids` |
| optional `referans1.txt` | documented reference JSON plus evaluator |
| forced plots/windows | annotated MP4 and preview artifact |

To preserve the historical robot layout, put this in a config file:

```json
{
  "cell_ids": [7, 1, 4, 8, 2, 5, 9, 3, 6]
}
```

The global drop detector was not carried into the default path because it could loop
forever and dereference missing frames. Trim the source video before analysis or use a
known start point in preprocessing. A future event-gating module should be independently
tested and expose a timeout/fallback state.
