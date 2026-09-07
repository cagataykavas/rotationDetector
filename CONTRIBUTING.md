# Contributing

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
pytest
grid-motion demo --output artifacts/smoke --frames 12 --no-video
```

Keep video I/O and GUI side effects out of imports. Add thresholds to `GridConfig`, add
tests for decision changes, preserve the JSON contracts, and label synthetic results as
synthetic. Do not commit private recordings, generated videos, credentials, or personal
data.

Pull requests should state the commands run, configuration used, and any expected
change to cell states or interval decisions.
