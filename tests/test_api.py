import src.app as app_module
from tests.conftest import parse_sse


def _upload(client, text, **form):
    data = {"use_prefilter": "true", "ml_only": "true", **form}
    return client.post("/upload-log/", files={"file": ("test.log", text.encode(), "text/plain")}, data=data)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_security_headers(client):
    r = client.get("/healthz")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert "default-src 'self'" in r.headers["Content-Security-Policy"]


def test_frontend_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Log Anomaly Detector" in r.text


def test_models_endpoint_without_ollama(client, monkeypatch):
    async def unreachable():
        return None

    monkeypatch.setattr(app_module, "_installed_ollama_models", unreachable)
    body = client.get("/api/models").json()
    assert body["ollama_reachable"] is False
    assert body["models"]
    assert all(m["installed"] is None for m in body["models"])


def test_models_endpoint_marks_installed(client, monkeypatch):
    async def installed():
        return ["llama3.2:1b", "mistral:latest"]

    monkeypatch.setattr(app_module, "_installed_ollama_models", installed)
    models = {m["name"]: m["installed"] for m in client.get("/api/models").json()["models"]}
    assert models["llama3.2:1b"] is True
    assert models["qwen2.5:0.5b"] is False
    assert models["mistral:latest"] is True


def test_validate_key_requires_key(client):
    r = client.post("/api/validate-api-key", data={"google_api_key": ""})
    assert r.status_code == 400


def test_rejects_invalid_mode(client):
    assert _upload(client, "a\n", mode="bogus").status_code == 422


def test_rejects_invalid_speed(client):
    assert _upload(client, "a\n", speed="ludicrous").status_code == 422


def test_rejects_invalid_chunk_percentage(client):
    assert _upload(client, "a\n", chunk_percentage="0").status_code == 422


def test_rejects_invalid_model_name(client):
    assert _upload(client, "a\n", model="bad model; rm -rf /").status_code == 422


def test_rejects_oversized_upload(client, monkeypatch):
    monkeypatch.setattr(app_module.settings, "max_upload_mb", 0.001)
    assert _upload(client, "x" * 5000).status_code == 413


def test_empty_file_returns_empty_result(client):
    events = parse_sse(_upload(client, "\n\n").text)
    assert events == [{"type": "result", "total_lines": 0, "anomalies": []}]


def test_ml_only_pipeline(client, access_log_lines):
    lines = access_log_lines[:600]
    lines[123] = "CRITICAL kernel panic: unable to mount root fs on unknown-block(0,0) !!!"
    r = _upload(client, "\n".join(lines))
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(r.text)
    types = [e["type"] for e in events]
    assert types[0] == "prefilter" and events[0]["status"] == "training"
    assert "prefilter_progress" in types

    result = events[-1]
    assert result["type"] == "result"
    assert result["ml_only"] is True
    assert result["total_lines"] == 600
    flagged = {a["line"] for a in result["anomalies"]}
    assert 124 in flagged
    assert all(a["score"].startswith("ML Score:") for a in result["anomalies"])


def test_ollama_unreachable_reports_error(client, monkeypatch):
    async def unreachable():
        return None

    monkeypatch.setattr(app_module, "_installed_ollama_models", unreachable)
    events = parse_sse(_upload(client, "one\ntwo\n", use_prefilter="false", ml_only="false").text)
    assert events[-1]["type"] == "error"
    assert "Cannot reach Ollama" in events[-1]["message"]


def test_missing_ollama_model_reports_error(client, monkeypatch):
    async def installed():
        return ["qwen2.5:0.5b"]

    monkeypatch.setattr(app_module, "_installed_ollama_models", installed)
    events = parse_sse(_upload(client, "one\n", model="llama3.2:1b", use_prefilter="false", ml_only="false").text)
    assert events[-1]["type"] == "error"
    assert "ollama pull llama3.2:1b" in events[-1]["message"]


def test_llm_pipeline_with_stubbed_detector(client, monkeypatch, access_log_lines):
    async def installed():
        return ["llama3.2:1b"]

    async def fake_explain(self, logs, mode="auto", progress_callback=None, chunk_percentage=10):
        if progress_callback:
            await progress_callback(1, 1)
        return [{"line": ln, "content": c, "anomaly": True, "reason": "stub reason"} for ln, c in logs]

    monkeypatch.setattr(app_module, "_installed_ollama_models", installed)
    monkeypatch.setattr(app_module.OllamaDetector, "explain_batch", fake_explain)

    events = parse_sse(_upload(client, "\n".join(access_log_lines[:300]), ml_only="false").text)
    assert any(e["type"] == "progress" and e["stage"] == "2/2" for e in events)
    result = events[-1]
    assert result["type"] == "result"
    assert result["anomalies"]
    assert all(a["prediction"] == "stub reason" for a in result["anomalies"])
    assert result["prefilter_remaining"] == len(result["anomalies"])
    assert result["failed_lines"] == 0
