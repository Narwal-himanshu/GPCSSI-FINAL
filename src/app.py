from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import json
import asyncio
import subprocess

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ollama_detector import OllamaDetector
from gemini_detector import GeminiDetector
from prefilter import LogPrefilter

# Force Ollama to use CUDA backend instead of Vulkan for this process.
# This fixes hybrid GPU setups where Vulkan picks the wrong GPU.
os.environ["OLLAMA_VULKAN"] = "0"
os.environ["OLLAMA_LLM_LIBRARY"] = "cuda_v13"

app = FastAPI()

@app.get("/api/gpu-status")
async def gpu_status():
    result = {
        "nvidia_gpu": None,
        "ollama_gpu_env": {},
        "ollama_reachable": False,
        "gpu_active": False,
        "message": ""
    }

    # 1. Check nvidia-smi
    try:
        nv_out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,temperature.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if nv_out.returncode == 0 and nv_out.stdout.strip():
            parts = nv_out.stdout.strip().split(", ")
            result["nvidia_gpu"] = {
                "name": parts[0] if len(parts) > 0 else "Unknown",
                "driver": parts[1] if len(parts) > 1 else "Unknown",
                "vram": parts[2] if len(parts) > 2 else "Unknown",
            }
    except Exception:
        result["nvidia_gpu"] = None

    # 2. Check Ollama env vars
    for var in ["OLLAMA_VULKAN", "OLLAMA_LLM_LIBRARY", "CUDA_VISIBLE_DEVICES"]:
        val = os.environ.get(var)
        if val is not None:
            result["ollama_gpu_env"][var] = val

    # 3. Check Ollama is reachable via its API
    try:
        proc = await asyncio.create_subprocess_exec(
            "ollama", "ps",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
        result["ollama_reachable"] = True

        # Parse ollama ps output for GPU info
        output = stdout.decode().strip()
        if output:
            lines = output.split("\n")
            if len(lines) > 1:
                header = lines[0]
                for row in lines[1:]:
                    if "GPU" in row or "%" in row:
                        result["gpu_active"] = True
                        break
                    # Also check: if any non-CPU processor is listed
                    if "CPU" not in row.split()[-1] if row.split() else True:
                        result["gpu_active"] = True
                        break
    except Exception:
        result["ollama_reachable"] = False

    # Summary message
    if result["nvidia_gpu"]:
        if result["gpu_active"]:
            result["message"] = f"GPU active: {result['nvidia_gpu']['name']}"
        elif result["ollama_reachable"]:
            has_vulkan_off = result["ollama_gpu_env"].get("OLLAMA_VULKAN") == "0"
            has_cuda_lib = "cuda" in result["ollama_gpu_env"].get("OLLAMA_LLM_LIBRARY", "")
            if has_vulkan_off and has_cuda_lib:
                result["message"] = (
                    f"NVIDIA {result['nvidia_gpu']['name']} detected. Env vars set for this project "
                    "- restart Ollama from the system tray to pick them up."
                )
            else:
                result["message"] = (
                    f"NVIDIA {result['nvidia_gpu']['name']} detected but Ollama may not be using it. "
                    "GPU env vars are set in the app - restart Ollama from the system tray."
                )
        else:
            result["message"] = (
                f"NVIDIA {result['nvidia_gpu']['name']} detected. "
                "Ollama is not running. Start it from the Start Menu and try again."
            )
    else:
        result["message"] = "No NVIDIA GPU detected. Running on CPU."

    return result

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/validate-api-key")
async def validate_api_key(google_api_key: str = Form(...)):
    try:
        from gemini_detector import validate_api_key as _validate
        return await _validate(google_api_key)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/upload-log/")
async def upload_log(
    file: UploadFile = File(...),
    model: str = Form("llama3.2:1b"),
    mode: str = Form("auto"),
    speed: str = Form("balanced"),
    google_api_key: str = Form(None),
    chunk_percentage: int = Form(10),
    use_prefilter: str = Form("false")
):
    if model.startswith("gemini-"):
        speed_map = {"balanced": 1, "fast": 2, "max": 3}
    else:
        speed_map = {"balanced": 2, "fast": 4, "max": 6}
    max_concurrency = speed_map.get(speed, 2)
    prefilter_enabled = use_prefilter.lower() == "true"
    return StreamingResponse(
        event_generator(file, model, mode, max_concurrency, google_api_key, chunk_percentage, prefilter_enabled),
        media_type="text/event-stream"
    )

async def event_generator(file: UploadFile, model: str, mode: str, max_concurrency: int = 3, google_api_key: str = None, chunk_percentage: int = 10, use_prefilter: bool = False):
    try:
        contents = await file.read()
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': f'Failed to read file: {str(e)}'})}\n\n"
        return

    lines = contents.decode("utf-8", errors="replace").splitlines()

    logs_to_analyze = []
    for line_num, line in enumerate(lines, 1):
        if not line.strip():
            continue
        logs_to_analyze.append((line_num, line))

    if not logs_to_analyze:
        yield f"data: {json.dumps({'type': 'result', 'total_lines': 0, 'anomalies': []})}\n\n"
        return

    original_count = len(logs_to_analyze)
    ml_scores = {}

    # ===== STAGE 1: ML Pre-filter (inspired by AICoE/log-anomaly-detector) =====
    if use_prefilter and len(logs_to_analyze) > 50:
        import threading
        import queue as qmod

        print(f"[PIPELINE] STAGE 1: ML Prefilter — scanning {original_count} lines...")
        prefilter = LogPrefilter()
        progress_queue = qmod.Queue()

        def pf_cb(current, total, message=None):
            progress_queue.put(("progress", current, total, message))

        yield f"data: {json.dumps({'type': 'prefilter', 'status': 'training', 'total': original_count, 'stage': '1/2'})}\n\n"

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, lambda: prefilter.fit(
            [c for _, c in logs_to_analyze],
            progress_callback=pf_cb
        ))

        while not future.done():
            try:
                typ, current, total, msg = progress_queue.get_nowait()
                if typ == "progress":
                    yield f"data: {json.dumps({
                        'type': 'prefilter_progress',
                        'current': current,
                        'total': total,
                        'stage': '1/2',
                        'message': msg or ''
                    })}\n\n"
            except qmod.Empty:
                pass
            await asyncio.sleep(0.1)

        while True:
            try:
                typ, current, total, msg = progress_queue.get_nowait()
                if typ == "progress":
                    yield f"data: {json.dumps({
                        'type': 'prefilter_progress',
                        'current': current,
                        'total': total,
                        'stage': '1/2',
                        'message': msg or ''
                    })}\n\n"
            except qmod.Empty:
                break

        exc = future.exception()
        if exc:
            yield f"data: {json.dumps({'type': 'error', 'message': f'ML Prefilter failed: {exc}'})}\n\n"
            return

        results = prefilter.filter(logs_to_analyze)
        # ML scores map: line_num → score
        ml_scores = {line_num: score for line_num, content, score in results}
        prefilter_flagged = len(results)
        prefilter_filtered_out = original_count - prefilter_flagged

        print(f"[PIPELINE] STAGE 1 complete: flagged {prefilter_flagged}/{original_count} lines ({prefilter_filtered_out} filtered out)")
        print(f"[PIPELINE] → ML detected {prefilter_flagged} anomalies; LLM will explain them")

        yield f"data: {json.dumps({
            'type': 'prefilter',
            'status': 'done',
            'stage': '1/2',
            'total': original_count,
            'flagged': prefilter_flagged,
            'filtered_out': prefilter_filtered_out
        })}\n\n"

        logs_to_analyze = [(ln, c) for ln, c, _ in results]

        if not logs_to_analyze:
            yield f"data: {json.dumps({'type': 'result', 'total_lines': original_count, 'anomalies': [], 'prefilter_enabled': True, 'prefilter_flagged': 0, 'prefilter_remaining': 0, 'prefilter_filtered_out': original_count})}\n\n"
            return
    elif use_prefilter:
        print(f"[PIPELINE] STAGE 1: skipped (only {len(logs_to_analyze)} lines, need > 50)")
        yield f"data: {json.dumps({
            'type': 'prefilter', 'status': 'skipped', 'stage': '1/2',
            'message': 'File too small for ML prefilter (less than 50 lines)'
        })}\n\n"

    # ===== STAGE 2: LLM — explanation only (if prefilter detected) or full analysis =====
    explain_only = bool(ml_scores)
    print(f"[PIPELINE] STAGE 2: {'Explanation' if explain_only else 'Analysis'} — "
          f"{len(logs_to_analyze)} lines with {model}")

    queue = asyncio.Queue()

    async def progress_callback(current, total):
        await queue.put({"type": "progress", "current": current, "total": total, "stage": "2/2"})

    async def run_analysis():
        nonlocal ml_scores
        try:
            if model.startswith("gemini-"):
                detector = GeminiDetector(model_name=model, api_key=google_api_key, max_concurrency=max_concurrency)
            else:
                detector = OllamaDetector(model_name=model, max_concurrency=max_concurrency)

            if explain_only:
                results = await detector.explain_batch(
                    logs_to_analyze,
                    mode=mode,
                    progress_callback=progress_callback,
                    chunk_percentage=chunk_percentage,
                )
            else:
                results = await detector.analyze_batch(
                    logs_to_analyze,
                    mode=mode,
                    progress_callback=progress_callback,
                    chunk_percentage=chunk_percentage,
                    total_lines=len(logs_to_analyze)
                )

            anomalies = []
            for res in results:
                if explain_only:
                    ms = ml_scores.get(res["line"])
                    score_display = f"ML Score: {ms:.4f}" if ms is not None else "ML Detected"
                    anomalies.append({
                        "line": res["line"],
                        "content": res["content"],
                        "score": score_display,
                        "prediction": res["reason"]
                    })
                elif res["anomaly"]:
                    anomalies.append({
                        "line": res["line"],
                        "content": res["content"],
                        "score": "LLM Detected",
                        "prediction": res["reason"]
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
                "prefilter_enabled": use_prefilter,
                "prefilter_flagged": prefilter_flagged,
                "prefilter_filtered_out": prefilter_filtered_out,
                "prefilter_remaining": prefilter_remaining
            })
            print(f"[PIPELINE] STAGE 2 complete: {len(anomalies)} anomalies explained in {len(logs_to_analyze)} lines")
        except Exception as e:
            err_msg = f"{'Explanation' if explain_only else 'Analysis'} failed: {str(e)}"
            print(f"[PIPELINE] STAGE 2 error: {err_msg}")
            await queue.put({"type": "error", "message": err_msg})
        finally:
            await queue.put(None)

    task = asyncio.create_task(run_analysis())

    while True:
        item = await queue.get()
        if item is None:
            break
        yield f"data: {json.dumps(item)}\n\n"

    await task

# Mount static files
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
