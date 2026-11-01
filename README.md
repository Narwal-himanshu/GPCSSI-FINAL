# Log Anomaly Detector (GPCSSI)

[![CI](https://github.com/Narwal-himanshu/GPCSSI-FINAL/actions/workflows/ci.yml/badge.svg)](https://github.com/Narwal-himanshu/GPCSSI-FINAL/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

A web application that finds anomalies in system logs in two stages:

1. **ML pre-filter.** An unsupervised Self-Organizing Map (SOM) learns what "normal" looks like in *your* log file and flags the most unusual lines, with no labelled data needed.
2. **LLM analysis.** A local model via [Ollama](https://ollama.com) or a cloud model via Google Gemini explains why each flagged line is suspicious. It can also scan the whole file directly.

Upload a log in the browser and watch progress stream live. You get a table of anomalous lines with scores and plain-English explanations.

Built for the GPCSSI program.

---

## Features

- **Two-stage pipeline.** The ML pre-filter cuts thousands of lines down to the few hundred worth an LLM's time.
- **ML Only mode.** Instant anomaly detection with no LLM and no GPU required.
- **Local or cloud LLMs.** Ollama models (Llama, Qwen, Gemma, …) or Gemini models through your own API key.
- **Four analysis modes:** Auto, Line-by-Line, Intensive, and Chunking (percentage-based).
- **Live progress** over Server-Sent Events.
- **Production hardening:** upload size and line limits, input validation, concurrency limits, security headers and CSP, health and readiness probes, Docker image, and CI.

## Quick start

### Requirements

- Python **3.10+**
- [Ollama](https://ollama.com/download) for local LLM analysis (optional if you use ML Only mode or Gemini)
- A Google Gemini API key (optional)

### Run locally

```bash
git clone https://github.com/Narwal-himanshu/GPCSSI-FINAL.git
cd GPCSSI-FINAL

python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

pip install -r requirements.txt

# Optional: install Ollama and pull the default models
./scripts/setup_ollama.sh          # macOS / Linux
./scripts/setup_ollama.ps1         # Windows

uvicorn src.app:app --host 127.0.0.1 --port 8000
```

Open <http://localhost:8000>, pick a model, upload a log file (try `data/Linux.log` or `examplelog/my_access.log.txt`), and click **Analyze Logs**.

### Run with Docker

```bash
docker compose up -d --build
docker compose exec ollama ollama pull llama3.2:1b
```

The app is served on <http://localhost:8000>. See [docs/deployment.md](docs/deployment.md) for GPU support, reverse proxies and other production settings.

## Configuration

All settings are environment variables (or a `.env` file). Copy [`.env.example`](.env.example) to get started. The most common ones:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Where the Ollama server is |
| `OLLAMA_MODELS` | `llama3.2:1b,gemma4:12b,qwen2.5:0.5b` | Models listed in the UI |
| `GOOGLE_API_KEY` | – | Optional server-side Gemini key |
| `MAX_UPLOAD_MB` | `50` | Upload size limit |
| `MAX_CONCURRENT_JOBS` | `2` | Analyses allowed at once |
| `CORS_ORIGINS` | *(empty)* | Cross-origin callers allowed (same-origin only by default) |

The full list is in [docs/configuration.md](docs/configuration.md).

## Documentation

| Document | What's in it |
|---|---|
| [Architecture](docs/architecture.md) | How the pipeline, modules and frontend fit together |
| [ML pre-filter](docs/ml-prefilter.md) | The SOM model, features, scoring and evaluation |
| [API reference](docs/api.md) | HTTP endpoints and the streaming event format |
| [Configuration](docs/configuration.md) | Every environment variable |
| [Deployment](docs/deployment.md) | Docker, reverse proxies, GPUs, production checklist |
| [Development](docs/development.md) | Local setup, tests, linting, project layout |
| [Troubleshooting](docs/troubleshooting.md) | Common problems and fixes |

Background material from the original project lives in [`docs/reference/`](docs/reference/): the mathematical write-up of the ML core, the pre-filter evaluation report, and the project guide (HTML and PDF).

## Project structure

```
├── src/
│   ├── app.py               # FastAPI app: routes, validation, the two-stage pipeline
│   ├── config.py            # Settings loaded from environment variables
│   ├── prefilter.py         # SOM-based ML pre-filter
│   ├── ollama_detector.py   # Local LLM detector (Ollama)
│   ├── gemini_detector.py   # Cloud LLM detector (Gemini) with rate limiting & caching
│   └── llm_utils.py         # Shared parsing of LLM responses
├── frontend/                # Static UI (HTML/CSS/JS), served by the app
├── tests/                   # pytest suite
├── scripts/
│   ├── genlog.py            # Synthetic Apache access-log generator
│   ├── setup_ollama.sh      # Ollama setup (macOS/Linux)
│   └── setup_ollama.ps1     # Ollama setup (Windows)
├── data/                    # Sample datasets (Apache, Linux) from loghub
├── examplelog/              # Synthetic access log produced by genlog.py
├── docs/                    # Documentation
├── Dockerfile
└── docker-compose.yml
```

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest
ruff check src tests scripts
```

The tests need neither Ollama nor a Gemini key: LLM calls are stubbed.

## Acknowledgements

This project stands on the work of others, and we're grateful to them.

- **[AICoE/log-anomaly-detector](https://github.com/AICoE/log-anomaly-detector)** by the AI Center of Excellence at Red Hat. Our ML pre-filter's approach comes from this project: learning normal log behaviour with a Self-Organizing Map and scoring lines by their distance to the map. Thank you to its authors and contributors for publishing that work openly under the GPL-3.0. This project is released under the same license.
- **[Loghub](https://github.com/logpai/loghub)** by the LogPAI team. The `data/Apache.log` and `data/Linux.log` sample datasets come from Loghub, a collection of system logs released for AI-driven log analytics research. If you use these datasets in research, please cite the Loghub paper:
  Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, Michael R. Lyu. *Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics.* IEEE ISSRE, 2023.
- **[Ollama](https://ollama.com)** and **[Google Gemini](https://ai.google.dev)** power the LLM stage. **[FastAPI](https://fastapi.tiangolo.com)** and **[NumPy](https://numpy.org)** power the rest.

## License

This project is licensed under the **GNU General Public License v3.0**. See [LICENSE](LICENSE).

The ML pre-filter is derived from the approach of [AICoE/log-anomaly-detector](https://github.com/AICoE/log-anomaly-detector), which is also GPL-3.0. The sample datasets in `data/` belong to their original authors (see Loghub's terms) and are included for research and demonstration only.
