# Configuration

All settings are read from environment variables. A `.env` file in the working directory is also loaded; copy [`.env.example`](../.env.example) to start. Variable names are case-insensitive.

## General

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `FRONTEND_DIR` | `<repo>/frontend` | Directory of static files served at `/` |
| `CORS_ORIGINS` | *(empty)* | Comma-separated origins allowed to call the API from another site. Empty means same-origin only, which is all the bundled UI needs. |
| `ENABLE_GPU_STATUS` | `true` | Enables `/api/gpu-status` (runs `nvidia-smi` if present) |

## Limits

| Variable | Default | Description |
|---|---|---|
| `MAX_UPLOAD_MB` | `50` | Maximum upload size in megabytes |
| `MAX_LINES` | `200000` | Maximum number of lines per upload |
| `MAX_CONCURRENT_JOBS` | `2` | Analyses that may run at once (per process); extra requests get `429` |

## Ollama

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | URL of the Ollama server |
| `OLLAMA_TIMEOUT` | `120` | Seconds to wait for one Ollama response |
| `OLLAMA_MODELS` | `llama3.2:1b,gemma4:12b,qwen2.5:0.5b` | Models listed in the UI. Other pulled models are listed too. |
| `DEFAULT_MODEL` | `llama3.2:1b` | Model used when a request doesn't specify one; preselected in the UI |

GPU settings such as `OLLAMA_VULKAN` or `CUDA_VISIBLE_DEVICES` must be set on the **Ollama server** process, not on this app. See [troubleshooting.md](troubleshooting.md#ollama-is-using-the-cpu-instead-of-my-gpu).

## Gemini

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | – | Server-side key, used when the user doesn't supply one in the UI |
| `GEMINI_MAX_RPM` | `15` | Requests per minute allowed per API key |
| `GEMINI_BURST` | `1` | Token-bucket burst size |
| `GEMINI_COOLDOWN_SECONDS` | `15` | Pause for all requests on a key after an HTTP 429 |
| `GEMINI_MAX_RETRIES` | `5` | Attempts per request for rate-limit and server errors |
| `GEMINI_BASE_DELAY` | `2.0` | Base of the exponential backoff, in seconds |
| `GEMINI_MAX_DELAY` | `30.0` | Maximum backoff delay, in seconds |
| `GEMINI_CACHE_TTL` | `300` | Seconds identical prompts are served from cache |
| `GEMINI_TIMEOUT` | `60` | Seconds to wait for one Gemini response |
| `GEMINI_MAX_CONCURRENCY` | `2` | Default parallelism (overridden by the UI speed setting) |

The free Gemini tier is heavily rate-limited. Keep `GEMINI_MAX_RPM` at or below your quota.

## ML pre-filter

| Variable | Default | Description |
|---|---|---|
| `PREFILTER_MIN_LINES` | `50` | Files with this many non-empty lines or fewer skip the pre-filter |
| `PREFILTER_PERCENTILE` | `85` | Score percentile used as the anomaly threshold (85 flags about 15% of lines) |
| `PREFILTER_MAP_SIZE` | `12` | SOM grid side length |
| `PREFILTER_SEED` | `42` | RNG seed for reproducible results; set it empty for random |

See [ml-prefilter.md](ml-prefilter.md) for how these affect detection.
