# Troubleshooting

## "Cannot reach Ollama at http://localhost:11434"

Ollama is not running or is on another address.

- Start it: `ollama serve` (or launch the Ollama app on Windows/macOS).
- Check it: `curl http://localhost:11434/api/tags`.
- If it runs elsewhere, set `OLLAMA_HOST`.
- You can still use **ML Only** mode or a Gemini model without Ollama.

## "Model 'xyz' is not installed in Ollama"

Pull it: `ollama pull xyz`. The model dropdown marks unpulled models "(not pulled)".

## Ollama is using the CPU instead of my GPU

The GPU indicator at the top of the page reads `/api/gpu-status`, which asks Ollama how much of each loaded model sits in VRAM. Run one analysis first, because models are only loaded on demand.

If a GPU is detected but not used:

1. GPU variables must be set for the **Ollama server process**, not for this app. On hybrid-GPU laptops where Ollama picks the wrong GPU (for example the Vulkan backend on an integrated GPU), set them as user environment variables:
   ```powershell
   [Environment]::SetEnvironmentVariable('OLLAMA_VULKAN', '0', 'User')
   ```
   On Linux, add `Environment="OLLAMA_VULKAN=0"` to the Ollama systemd unit.
2. Fully quit Ollama (system tray → Quit) and start it again.
3. Make sure the model fits in VRAM. Larger models are split between GPU and CPU.
4. Update your NVIDIA driver.

## Analysis is slow

- Turn on the **ML Pre-filter** so the LLM only sees about 15% of the lines.
- Use a smaller model (`qwen2.5:0.5b`, `llama3.2:1b`).
- Raise the **Speed** setting if your GPU has spare capacity.
- Avoid **Line-by-Line** mode on large files: it makes one request per line.

## Gemini: "quota exhausted" / rate limited

The free tier allows only a few requests per minute and per day. The app already rate-limits, backs off and caches, but a large file can still exhaust a daily quota. Turn on the ML pre-filter to cut the number of requests, use a paid key, or switch to a local model.

## "The server is busy with other analyses" (HTTP 429)

`MAX_CONCURRENT_JOBS` analyses are already running. Wait, or raise the limit if your hardware allows.

## Upload rejected (HTTP 413)

The file is larger than `MAX_UPLOAD_MB` or has more than `MAX_LINES` lines. Raise the limits, or split the file. Behind a reverse proxy, raise its body-size limit too (`client_max_body_size` in nginx).

## Progress bar stalls behind a reverse proxy

The proxy is buffering the event stream. Turn off buffering (`proxy_buffering off;` in nginx) and raise the read timeout. See [deployment.md](deployment.md#reverse-proxy).

## Some lines say "Analysis failed" / "Explanation failed"

The LLM call for that chunk failed after retries (timeout, invalid JSON, quota). The UI shows a warning with the count, and the server log has the cause. Smaller models sometimes return malformed JSON for big chunks; try a lower chunk percentage or a larger model.

## Port 8000 already in use

`uvicorn src.app:app --port 8001`
