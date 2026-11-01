# Deployment

## Docker Compose (recommended)

`docker-compose.yml` runs the app and an Ollama server side by side:

```bash
cp .env.example .env          # optional: adjust settings
docker compose up -d --build
docker compose exec ollama ollama pull llama3.2:1b
```

- The app listens on port `8000` and reaches Ollama at `http://ollama:11434`.
- Pulled models persist in the `ollama` volume.
- The app container runs as a non-root user and has a `HEALTHCHECK` on `/healthz`.

### GPU

To give Ollama an NVIDIA GPU, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and uncomment the `deploy:` block of the `ollama` service in `docker-compose.yml`. Check the result with `curl localhost:8000/api/gpu-status` after running one analysis.

### Using an existing Ollama

Remove the `ollama` service and point the app at your server:

```bash
docker build -t log-anomaly-detector .
docker run -d -p 8000:8000 -e OLLAMA_HOST=http://host.docker.internal:11434 log-anomaly-detector
```

## Bare metal / VM

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn src.app:app --host 0.0.0.0 --port 8000 --proxy-headers
```

Run **a single worker** (`--workers 1`). The concurrency limit (`MAX_CONCURRENT_JOBS`) and the Gemini rate limiter live in process memory, so several workers would each enforce their own limits. The pre-filter runs in a thread pool, so one worker handles concurrent uploads fine.

### systemd example

```ini
# /etc/systemd/system/log-anomaly-detector.service
[Unit]
Description=Log Anomaly Detector
After=network.target

[Service]
User=logad
WorkingDirectory=/opt/log-anomaly-detector
EnvironmentFile=/opt/log-anomaly-detector/.env
ExecStart=/opt/log-anomaly-detector/.venv/bin/uvicorn src.app:app --host 127.0.0.1 --port 8000 --proxy-headers
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

## Reverse proxy

Put the app behind a reverse proxy for TLS and, since the app has no login of its own, authentication. The analysis endpoint streams Server-Sent Events, so **turn off response buffering** and allow long requests. The app already sends `X-Accel-Buffering: no`, which nginx honours.

```nginx
server {
    listen 443 ssl;
    server_name logs.example.com;

    client_max_body_size 50m;          # match MAX_UPLOAD_MB

    auth_basic "Log Anomaly Detector";
    auth_basic_user_file /etc/nginx/htpasswd;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_http_version 1.1;
        proxy_buffering off;
        proxy_read_timeout 3600s;      # LLM analysis of big files can take a while
    }
}
```

## Production checklist

- [ ] Served over HTTPS behind a reverse proxy with authentication
- [ ] `MAX_UPLOAD_MB` and the proxy's body-size limit agree
- [ ] `MAX_CONCURRENT_JOBS` sized for your CPU and GPU (the SOM uses one CPU core per job; LLM jobs share the GPU)
- [ ] Required Ollama models pulled (`/api/models` shows `installed: true`)
- [ ] `CORS_ORIGINS` left empty unless another site must call the API
- [ ] If Gemini is enabled: users know log contents are sent to Google
- [ ] Liveness probe on `/healthz`; alert on `/readyz` → `ollama_reachable: false`
- [ ] Logs collected from stdout (`LOG_LEVEL=INFO`)
