import asyncio
import json
import logging
import queue as qmod
import re
import shutil
import subprocess
from contextlib import asynccontextmanager

import ollama
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import get_settings
from .gemini_detector import GeminiDetector
from .gemini_detector import validate_api_key as _validate_gemini_key
from .ollama_detector import OllamaDetector
from .prefilter import LogPrefilter

settings = get_settings()

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("app")

VALID_MODES = {"auto", "line-by-line", "intensive", "chunking"}
OLLAMA_SPEED_MAP = {"balanced": 2, "fast": 4, "max": 6}
GEMINI_SPEED_MAP = {"balanced": 1, "fast": 2, "max": 3}
MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9._:/\-]{1,128}$")

_job_slots = asyncio.Semaphore(settings.max_concurrent_jobs)


def sse(payload: dict) -> str:
    """Format a dict as a single Server-Sent Events message."""
    return f"data: {json.dumps(payload)}\n\n"


def _is_true(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _ollama_client() -> ollama.AsyncClient:
    return ollama.AsyncClient(host=settings.ollama_host, timeout=10)


async def _installed_ollama_models() -> list[str] | None:
    """Names of models pulled into Ollama, or None if Ollama is unreachable."""
    try:
        response = await _ollama_client().list()
    except Exception as e:
        logger.debug("Ollama unreachable: %s", e)
        return None
    return [m.model for m in response.models]


def _model_installed(model: str, installed: list[str]) -> bool:
    candidates = {model, model if ":" in model else f"{model}:latest"}
    return any(name in candidates for name in installed)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Starting Log Anomaly Detector v%s (ollama=%s, max_upload=%.0fMB, max_jobs=%d)",
        __version__, settings.ollama_host, settings.max_upload_mb, settings.max_concurrent_jobs,
    )
    yield


app = FastAPI(
    title="Log Anomaly Detector",
    version=__version__,
    description="Two-stage log anomaly detection: an ML pre-filter (SOM) followed by LLM analysis.",
    lifespan=lifespan,
)

if settings.cors_origin_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",
    )
    return response


# ---------------------------------------------------------------------------
# Health & metadata
# ---------------------------------------------------------------------------
@app.get("/healthz", tags=["health"])
async def healthz():
    """Liveness probe: the process is up and serving requests."""
    return {"status": "ok", "version": __version__}


@app.get("/readyz", tags=["health"])
async def readyz():
    """Readiness probe: reports whether the Ollama backend is reachable."""
    installed = await _installed_ollama_models()
    return {
        "status": "ok",
        "version": __version__,
        "ollama_reachable": installed is not None,
        "gemini_server_key": settings.google_api_key is not None,
    }


@app.get("/api/models", tags=["models"])
async def list_models():
    """Ollama models offered in the UI, with whether each one is pulled locally."""
    installed = await _installed_ollama_models()
    configured = settings.ollama_model_list
    models = [
        {"name": m, "provider": "ollama", "installed": None if installed is None else _model_installed(m, installed)}
        for m in configured
    ]
    # Also offer any other locally pulled models that are not in the configured list.
    for name in installed or []:
        if not any(_model_installed(m, [name]) for m in configured):
            models.append({"name": name, "provider": "ollama", "installed": True})
    return {
        "default_model": settings.default_model,
        "ollama_reachable": installed is not None,
        "gemini_server_key": settings.google_api_key is not None,
        "models": models,
    }


@app.get("/api/gpu-status", tags=["models"])
async def gpu_status():
    result = {
        "nvidia_gpu": None,
        "ollama_reachable": False,
        "gpu_active": False,
        "loaded_models": [],
        "message": "",
    }
    if not settings.enable_gpu_status:
        result["message"] = "GPU status check disabled."
        return result

    # 1. Check for an NVIDIA GPU on this host.
    if shutil.which("nvidia-smi"):
        try:
            nv_out = await asyncio.to_thread(
                subprocess.run,
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if nv_out.returncode == 0 and nv_out.stdout.strip():
                parts = [p.strip() for p in nv_out.stdout.strip().splitlines()[0].split(",")]
                result["nvidia_gpu"] = {
                    "name": parts[0] if len(parts) > 0 else "Unknown",
                    "driver": parts[1] if len(parts) > 1 else "Unknown",
                    "vram": parts[2] if len(parts) > 2 else "Unknown",
                }
        except Exception as e:
            logger.debug("nvidia-smi failed: %s", e)

    # 2. Ask Ollama which models are loaded and how much of each sits in VRAM.
    try:
        ps = await _ollama_client().ps()
        result["ollama_reachable"] = True
        for m in ps.models:
            in_vram = (m.size_vram or 0) > 0
            result["loaded_models"].append({"name": m.model, "size": m.size, "size_vram": m.size_vram})
            result["gpu_active"] = result["gpu_active"] or in_vram
    except Exception as e:
        logger.debug("Ollama ps failed: %s", e)

    gpu_name = result["nvidia_gpu"]["name"] if result["nvidia_gpu"] else None
    if not result["ollama_reachable"]:
        result["message"] = f"Ollama is not reachable at {settings.ollama_host}."
    elif result["gpu_active"]:
        result["message"] = f"GPU active{': ' + gpu_name if gpu_name else ''}"
    elif not result["loaded_models"]:
        result["message"] = "Ollama is running. No model loaded yet; GPU use is shown after the first analysis."
    elif gpu_name:
        result["message"] = f"{gpu_name} detected but the loaded model is running on CPU."
    else:
        result["message"] = "Ollama is running on CPU."
    return result


# ---------------------------------------------------------------------------
# Gemini key validation
# ---------------------------------------------------------------------------
@app.post("/api/validate-api-key", tags=["models"])
async def validate_api_key(google_api_key: str = Form("")):
    key = google_api_key.strip() or settings.google_api_key
    if not key:
        raise HTTPException(status_code=400, detail="No API key provided.")
    try:
        return await _validate_gemini_key(key)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
async def _read_upload(file: UploadFile) -> bytes:
    """Read an upload, rejecting it once it exceeds the configured size limit."""
    limit = settings.max_upload_bytes
    data = bytearray()
    while chunk := await file.read(1024 * 1024):
        data.extend(chunk)
        if len(data) > limit:
            raise HTTPException(status_code=413, detail=f"File exceeds the {settings.max_upload_mb:g} MB upload limit.")
    return bytes(data)


@app.post("/upload-log/", tags=["analysis"])
async def upload_log(
    file: UploadFile = File(...),
    model: str = Form(None),
    mode: str = Form("auto"),
    speed: str = Form("balanced"),
    google_api_key: str = Form(None),
    chunk_percentage: int = Form(10),
    use_prefilter: str = Form("false"),
    ml_only: str = Form("false"),
):
    """Analyse an uploaded log file. Responds with a `text/event-stream` of progress and result events."""
    model = (model or settings.default_model).strip()
    if not MODEL_NAME_RE.match(model):
        raise HTTPException(status_code=422, detail="Invalid model name.")
    if mode not in VALID_MODES:
        raise HTTPException(status_code=422, detail=f"Invalid mode. Choose one of: {', '.join(sorted(VALID_MODES))}.")
    if speed not in OLLAMA_SPEED_MAP:
        raise HTTPException(status_code=422, detail="Invalid speed. Choose one of: balanced, fast, max.")
    if not 1 <= chunk_percentage <= 100:
        raise HTTPException(status_code=422, detail="chunk_percentage must be between 1 and 100.")

    prefilter_enabled = _is_true(use_prefilter)
    ml_only_enabled = _is_true(ml_only) and prefilter_enabled
    is_gemini = model.startswith("gemini-")
    max_concurrency = (GEMINI_SPEED_MAP if is_gemini else OLLAMA_SPEED_MAP)[speed]
    api_key = (google_api_key or "").strip() or settings.google_api_key

    contents = await _read_upload(file)
    lines = contents.decode("utf-8", errors="replace").splitlines()
    if len(lines) > settings.max_lines:
        raise HTTPException(status_code=413, detail=f"File has {len(lines)} lines; the limit is {settings.max_lines}.")

    if _job_slots.locked():
        raise HTTPException(status_code=429, detail="The server is busy with other analyses. Try again shortly.")

    return StreamingResponse(
        event_generator(lines, model, mode, max_concurrency, api_key, chunk_percentage,
                        prefilter_enabled, ml_only_enabled),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def event_generator(lines: list[str], model: str, mode: str, max_concurrency: int = 2,
                          google_api_key: str | None = None, chunk_percentage: int = 10,
                          use_prefilter: bool = False, ml_only: bool = False):
    async with _job_slots:
        async for event in _run_pipeline(lines, model, mode, max_concurrency, google_api_key,
                                         chunk_percentage, use_prefilter, ml_only):
            yield event


async def _run_pipeline(lines, model, mode, max_concurrency, google_api_key,
                        chunk_percentage, use_prefilter, ml_only):
    logs_to_analyze = [(line_num, line) for line_num, line in enumerate(lines, 1) if line.strip()]

    if not logs_to_analyze:
        yield sse({"type": "result", "total_lines": 0, "anomalies": []})
        return

    original_count = len(logs_to_analyze)
    ml_scores = {}
    run_prefilter = use_prefilter and original_count > settings.prefilter_min_lines

    # Fail fast, before the ML stage, if the selected Ollama model cannot be used.
    if not (run_prefilter and ml_only) and not model.startswith("gemini-"):
        installed = await _installed_ollama_models()
        if installed is None:
            yield sse({"type": "error", "message": f"Cannot reach Ollama at {settings.ollama_host}. "
                                                   "Start it with `ollama serve`, or enable ML Only mode."})
            return
        if not _model_installed(model, installed):
            yield sse({"type": "error", "message": f"Model '{model}' is not installed in Ollama. "
                                                   f"Run `ollama pull {model}` and try again."})
            return

    # ===== STAGE 1: ML Pre-filter (inspired by AICoE/log-anomaly-detector) =====
    if run_prefilter:
        logger.info("STAGE 1: ML prefilter, scanning %d lines", original_count)
        prefilter = LogPrefilter(
            map_size=settings.prefilter_map_size,
            flag_percentile=settings.prefilter_percentile,
            seed=settings.prefilter_seed,
        )
        progress_queue = qmod.Queue()

        def pf_cb(current, total, message=None):
            progress_queue.put((current, total, message))

        def drain_progress():
            events = []
            while True:
                try:
                    current, total, msg = progress_queue.get_nowait()
                except qmod.Empty:
                    return events
                events.append(sse({"type": "prefilter_progress", "current": current, "total": total,
                                   "stage": "1/2", "message": msg or ""}))

        yield sse({"type": "prefilter", "status": "training", "total": original_count, "stage": "1/2"})

        def fit_and_filter():
            prefilter.fit([c for _, c in logs_to_analyze], verbose=False, progress_callback=pf_cb)
            return prefilter.filter(logs_to_analyze)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, fit_and_filter)
        while not future.done():
            for event in drain_progress():
                yield event
            await asyncio.sleep(0.1)
        for event in drain_progress():
            yield event

        try:
            results = future.result()
        except Exception as exc:
            logger.exception("ML prefilter failed")
            yield sse({"type": "error", "message": f"ML Prefilter failed: {exc}"})
            return

        ml_scores = {line_num: score for line_num, _, score in results}
        prefilter_flagged = len(results)
        prefilter_filtered_out = original_count - prefilter_flagged
        logger.info("STAGE 1 complete: flagged %d/%d lines", prefilter_flagged, original_count)

        yield sse({
            "type": "prefilter",
            "status": "done",
            "stage": "1/2",
            "total": original_count,
            "flagged": prefilter_flagged,
            "filtered_out": prefilter_filtered_out,
            "ml_only": ml_only,
        })

        logs_to_analyze = [(ln, c) for ln, c, _ in results]

        if not logs_to_analyze:
            yield sse({"type": "result", "total_lines": len(lines), "anomalies": [], "prefilter_enabled": True,
                       "ml_only": ml_only, "prefilter_flagged": 0, "prefilter_remaining": 0,
                       "prefilter_filtered_out": original_count})
            return

        if ml_only:
            anomalies = [
                {"line": ln, "content": c, "score": f"ML Score: {ml_scores[ln]:.4f}", "prediction": ""}
                for ln, c in logs_to_analyze
            ]
            yield sse({
                "type": "result",
                "total_lines": len(lines),
                "anomalies": anomalies,
                "prefilter_enabled": True,
                "ml_only": True,
                "prefilter_flagged": prefilter_flagged,
                "prefilter_filtered_out": prefilter_filtered_out,
                "prefilter_remaining": 0,
            })
            return
    elif use_prefilter:
        logger.info("STAGE 1 skipped: only %d lines", len(logs_to_analyze))
        yield sse({
            "type": "prefilter", "status": "skipped", "stage": "1/2",
            "message": f"File too small for ML prefilter (needs more than {settings.prefilter_min_lines} lines)",
        })

    # ===== STAGE 2: LLM — explanation only (if prefilter detected) or full analysis =====
    explain_only = bool(ml_scores)
    logger.info("STAGE 2: %s, %d lines with %s",
                "explanation" if explain_only else "analysis", len(logs_to_analyze), model)

    queue = asyncio.Queue()

    async def progress_callback(current, total):
        await queue.put({"type": "progress", "current": current, "total": total, "stage": "2/2"})

    async def run_analysis():
        try:
            if model.startswith("gemini-"):
                detector = GeminiDetector(model_name=model, api_key=google_api_key, max_concurrency=max_concurrency)
            else:
                detector = OllamaDetector(model_name=model, max_concurrency=max_concurrency,
                                          host=settings.ollama_host, timeout=settings.ollama_timeout)

            if explain_only:
                results = await detector.explain_batch(
                    logs_to_analyze, mode=mode, progress_callback=progress_callback,
                    chunk_percentage=chunk_percentage,
                )
            else:
                results = await detector.analyze_batch(
                    logs_to_analyze, mode=mode, progress_callback=progress_callback,
                    chunk_percentage=chunk_percentage, total_lines=len(logs_to_analyze),
                )

            anomalies = []
            failed_lines = 0
            for res in results:
                if str(res.get("reason", "")).startswith(("Analysis failed", "Explanation failed")):
                    failed_lines += 1
                if explain_only:
                    ms = ml_scores.get(res["line"])
                    anomalies.append({
                        "line": res["line"],
                        "content": res["content"],
                        "score": f"ML Score: {ms:.4f}" if ms is not None else "ML Detected",
                        "prediction": res["reason"],
                    })
                elif res["anomaly"]:
                    anomalies.append({
                        "line": res["line"],
                        "content": res["content"],
                        "score": "LLM Detected",
                        "prediction": res["reason"],
                    })

            prefilter_flagged = prefilter_filtered_out = prefilter_remaining = 0
            if explain_only:
                prefilter_flagged = len(logs_to_analyze)
                prefilter_filtered_out = original_count - len(logs_to_analyze)
                prefilter_remaining = len(logs_to_analyze)

            await queue.put({
                "type": "result",
                "stage": "complete",
                "total_lines": len(lines),
                "anomalies": anomalies,
                "failed_lines": failed_lines,
                "prefilter_enabled": use_prefilter,
                "ml_only": ml_only,
                "prefilter_flagged": prefilter_flagged,
                "prefilter_filtered_out": prefilter_filtered_out,
                "prefilter_remaining": prefilter_remaining,
            })
            logger.info("STAGE 2 complete: %d anomalies in %d lines (%d failed)",
                        len(anomalies), len(logs_to_analyze), failed_lines)
        except Exception as e:
            err_msg = f"{'Explanation' if explain_only else 'Analysis'} failed: {e}"
            logger.exception("STAGE 2 error")
            await queue.put({"type": "error", "message": err_msg})
        finally:
            await queue.put(None)

    task = asyncio.create_task(run_analysis())
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield sse(item)
        await task
    finally:
        # Stop LLM work if the client disconnects mid-stream.
        if not task.done():
            task.cancel()


# Mount static files last so API routes take precedence.
app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
