# Ollama Setup and GPU Configuration Script for Windows
# Run this in PowerShell as Administrator for best results.

$ErrorActionPreference = "Stop"

Write-Host "========================================"
Write-Host "  Ollama + GPU Setup for Windows"
Write-Host "========================================"
Write-Host ""

# ---- Step 1: Check / Install Ollama ----
if (!(Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "[1/5] Ollama not found. Installing via winget..."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install -e --id Ollama.Ollama
        Write-Host "  -> Installed. Restart your terminal, then re-run this script."
        exit
    } else {
        Write-Host "  -> winget not found. Download manually from: https://ollama.com/download/windows"
        exit
    }
} else {
    Write-Host "[1/5] Ollama is installed."
}

# ---- Step 2: Detect GPU ----
Write-Host "[2/5] Detecting GPU..."
$hasNvidia = $false
$hasAmd = $false
$gpuName = ""

try {
    $nvidiaSmi = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null
    if ($nvidiaSmi) {
        $hasNvidia = $true
        $gpuName = $nvidiaSmi[0].Split(',')[0].Trim()
        Write-Host "  -> NVIDIA GPU detected: $gpuName"
    }
} catch {}

try {
    $amdInfo = & rocminfo 2>$null
    if ($LASTEXITCODE -eq 0) {
        $hasAmd = $true
        Write-Host "  -> AMD GPU detected via ROCm"
    }
} catch {}

if (-not $hasNvidia -and -not $hasAmd) {
    Write-Host "  -> No supported GPU detected. Ollama will run on CPU (slow)."
}

# ---- Step 3: Pull Models ----
Write-Host "[3/5] Pulling models (this may take a while)..."
ollama pull llama3.2:1b
ollama pull qwen2.5:0.5b

# ---- Step 4: Verify Ollama is ready ----
Write-Host "[4/5] Verifying Ollama is running..."

$ollamaProc = Get-Process ollama -ErrorAction SilentlyContinue
if (-not $ollamaProc) {
    Write-Host "  -> Starting Ollama..."
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5
} else {
    Write-Host "  -> Ollama is running (PID: $($ollamaProc.Id))"
}

# ---- Step 5: GPU configuration note ----
Write-Host "[5/5] GPU configuration..."

if ($hasNvidia) {
    Write-Host ""
    Write-Host "============================================"
    Write-Host "  GPU READY"
    Write-Host "============================================"
    Write-Host ""
    Write-Host "  GPU detected: $gpuName"
    Write-Host ""
    Write-Host "  The app will automatically force CUDA GPU usage"
    Write-Host "  when you start it with:"
    Write-Host "    uvicorn src.app:app --reload"
    Write-Host ""
    Write-Host "  Environment variables are set inside the Python app"
    Write-Host "  (OLLAMA_VULKAN=0, OLLAMA_LLM_LIBRARY=cuda_v13)"
    Write-Host "  so they only affect this project, not your whole system."
    Write-Host ""
    Write-Host "  IMPORTANT: For the env vars to take effect, you MUST"
    Write-Host "  restart Ollama from the system tray BEFORE starting the app."
    Write-Host "  (Right-click Ollama icon in tray -> Quit, then start again)"
    Write-Host "============================================"
} else {
    Write-Host "  -> No NVIDIA GPU detected. Ollama will run on CPU."
}

Write-Host ""
Write-Host "Setup complete!"
