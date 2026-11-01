# API reference

Interactive OpenAPI docs are available at `/docs` (Swagger UI) and `/redoc` while the server is running.

## Health

### `GET /healthz`

Liveness probe. Always returns `200` if the process is serving requests.

```json
{"status": "ok", "version": "1.0.0"}
```

### `GET /readyz`

Readiness information. Returns `200` even when Ollama is down, because Gemini and ML Only mode still work, so check the fields.

```json
{"status": "ok", "version": "1.0.0", "ollama_reachable": true, "gemini_server_key": false}
```

## Models

### `GET /api/models`

The Ollama models configured in `OLLAMA_MODELS`, plus any other models pulled into Ollama.

```json
{
  "default_model": "llama3.2:1b",
  "ollama_reachable": true,
  "gemini_server_key": false,
  "models": [
    {"name": "llama3.2:1b", "provider": "ollama", "installed": true},
    {"name": "qwen2.5:0.5b", "provider": "ollama", "installed": false}
  ]
}
```

`installed` is `null` when Ollama cannot be reached.

### `GET /api/gpu-status`

Reports whether an NVIDIA GPU is present (via `nvidia-smi`) and whether Ollama's loaded models sit in VRAM.

```json
{
  "nvidia_gpu": {"name": "NVIDIA GeForce RTX 3060", "driver": "560.94", "vram": "12288 MiB"},
  "ollama_reachable": true,
  "gpu_active": true,
  "loaded_models": [{"name": "llama3.2:1b", "size": 2019393189, "size_vram": 2019393189}],
  "message": "GPU active: NVIDIA GeForce RTX 3060"
}
```

### `POST /api/validate-api-key`

Form field: `google_api_key`. If it is empty, the server's `GOOGLE_API_KEY` is used.

- `200`: `{"status": "success", "models": ["models/gemini-2.0-flash", ...]}`
- `400`: `{"detail": "<reason>"}` for an invalid or missing key, or when the request is rate-limited

Results are cached per key for 5 minutes.

## Analysis

### `POST /upload-log/`

`multipart/form-data` fields:

| Field | Type | Default | Notes |
|---|---|---|---|
| `file` | file | *required* | UTF-8 text; invalid bytes are replaced |
| `model` | string | `DEFAULT_MODEL` | Ollama model name, or `gemini-*` |
| `mode` | string | `auto` | `auto`, `line-by-line`, `intensive`, `chunking` |
| `speed` | string | `balanced` | `balanced`, `fast`, `max` |
| `chunk_percentage` | int | `10` | 1–100, used by `chunking` mode |
| `use_prefilter` | bool | `false` | Run the ML pre-filter first |
| `ml_only` | bool | `false` | Skip the LLM; requires `use_prefilter` |
| `google_api_key` | string | – | Gemini key; falls back to `GOOGLE_API_KEY` |

Error responses (before streaming starts):

| Status | Cause |
|---|---|
| `413` | File larger than `MAX_UPLOAD_MB`, or more than `MAX_LINES` lines |
| `422` | Invalid `model`, `mode`, `speed` or `chunk_percentage` |
| `429` | `MAX_CONCURRENT_JOBS` analyses already running |

On success, the response is `text/event-stream`. Each message is one `data: <json>` line followed by a blank line.

#### Event types

**`prefilter`**: Stage 1 status changes.

```json
{"type": "prefilter", "status": "training", "total": 25567, "stage": "1/2"}
{"type": "prefilter", "status": "done", "stage": "1/2", "total": 25567, "flagged": 3835, "filtered_out": 21732, "ml_only": false}
{"type": "prefilter", "status": "skipped", "stage": "1/2", "message": "File too small for ML prefilter (needs more than 50 lines)"}
```

**`prefilter_progress`**: SOM training progress (0–100).

```json
{"type": "prefilter_progress", "current": 45, "total": 100, "stage": "1/2", "message": "SOM training 2000/5000"}
```

**`progress`**: Stage 2 progress, in completed chunks.

```json
{"type": "progress", "current": 3, "total": 16, "stage": "2/2"}
```

**`result`**: the final event on success.

```json
{
  "type": "result",
  "stage": "complete",
  "total_lines": 25567,
  "anomalies": [
    {"line": 812, "content": "...", "score": "ML Score: 22.4217", "prediction": "Repeated authentication failure from ..."}
  ],
  "failed_lines": 0,
  "prefilter_enabled": true,
  "ml_only": false,
  "prefilter_flagged": 3835,
  "prefilter_filtered_out": 21732,
  "prefilter_remaining": 3835
}
```

- `score` is `"ML Score: <float>"` when the line was flagged by the pre-filter, otherwise `"LLM Detected"`.
- `prediction` is the LLM's explanation (empty in ML Only mode).
- `failed_lines` counts lines whose LLM call failed after retries. Their `prediction` starts with `Analysis failed:` or `Explanation failed:`.

**`error`**: the final event on failure (Ollama unreachable, model not pulled, invalid Gemini key, and so on).

```json
{"type": "error", "message": "Model 'llama3.2:1b' is not installed in Ollama. Run `ollama pull llama3.2:1b` and try again."}
```

#### Example

```bash
curl -N -F file=@data/Linux.log -F use_prefilter=true -F ml_only=true \
     http://localhost:8000/upload-log/
```
