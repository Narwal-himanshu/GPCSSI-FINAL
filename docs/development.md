# Development

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
uvicorn src.app:app --reload
```

`--reload` restarts the server when Python files change. Frontend changes only need a browser refresh.

## Tests and linting

```bash
pytest                       # unit and API tests; no Ollama or Gemini needed
ruff check src tests scripts # lint
ruff check --fix ...         # auto-fix
```

The suite covers:

- `tests/test_prefilter.py`: SOM training, anomaly detection on injected lines, reproducibility, and vectorised versus reference scoring
- `tests/test_llm_utils.py`: tolerant parsing of LLM JSON output
- `tests/test_api.py`: endpoints, input validation, upload limits, security headers, the full ML Only pipeline, and the LLM pipeline with a stubbed detector

CI (`.github/workflows/ci.yml`) runs lint and tests on Python 3.10–3.13 and builds the Docker image on every push and pull request.

## Generating test logs

`scripts/genlog.py` writes synthetic Apache access logs with a chosen number of error and unauthorised responses:

```bash
python scripts/genlog.py --total 5000 --errors 200 --unauthorized 100 -o my_access.log.txt --seed 42
```

## Project conventions

- The backend is the `src` package; run it as `uvicorn src.app:app` from the repository root.
- Every setting goes through `src/config.py`. Document new settings in `docs/configuration.md` and `.env.example`.
- Both detectors share one interface: `analyze_batch(logs, mode, progress_callback, chunk_percentage, total_lines)` and `explain_batch(logs, mode, progress_callback, chunk_percentage)`. Each takes `[(line_number, text), ...]` and returns dicts with `line`, `content`, `anomaly` and `reason`. Put shared parsing in `src/llm_utils.py`.
- The frontend must never put log content or server messages into `innerHTML`; use `textContent`.

## Adding a new LLM provider

1. Create `src/<provider>_detector.py` with a class implementing `analyze_batch` and `explain_batch` (use `llm_utils` for parsing).
2. In `src/app.py`, pick the detector in `run_analysis()` from the model name, for example by a prefix.
3. Add speed-to-concurrency mappings, configuration and docs.
4. Add tests with the network call stubbed.
