let validatedGeminiModels = [];
let ollamaModels = [{ name: "llama3.2:1b", installed: null }];
let defaultModel = "llama3.2:1b";

function showError(message, level = 'error') {
    const banner = document.getElementById('errorBanner');
    document.getElementById('errorText').textContent = message;
    banner.className = level === 'warning' ? 'error-banner warning' : 'error-banner';
}

function hideError() {
    document.getElementById('errorBanner').classList.add('hidden');
}

async function errorFromResponse(response) {
    try {
        const data = await response.json();
        if (typeof data.detail === 'string') return data.detail;
        if (Array.isArray(data.detail)) return data.detail.map(d => d.msg).join('; ');
    } catch (e) {
        // fall through to the status text
    }
    return `Server error: ${response.status} ${response.statusText}`;
}

async function loadModels() {
    try {
        const res = await fetch('/api/models');
        if (!res.ok) return;
        const data = await res.json();
        if (Array.isArray(data.models) && data.models.length) {
            ollamaModels = data.models;
        }
        defaultModel = data.default_model || defaultModel;
    } catch (e) {
        console.warn('Could not load model list:', e);
    }
    updateModelDropdown(defaultModel);
}

async function validateAndSaveKey() {
    const apiKeyInput = document.getElementById('apiKeyInput');
    const statusMsg = document.getElementById('apiKeyStatus');
    const validateBtn = document.getElementById('validateKeyBtn');
    const apiKey = apiKeyInput.value.trim();

    if (!apiKey) {
        statusMsg.textContent = "Please enter an API key.";
        statusMsg.className = "status-msg error";
        return;
    }

    validateBtn.disabled = true;
    statusMsg.textContent = "Validating...";
    statusMsg.className = "status-msg";

    try {
        const formData = new FormData();
        formData.append("google_api_key", apiKey);

        const response = await fetch("/api/validate-api-key", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(await errorFromResponse(response));
        }
        const data = await response.json();

        statusMsg.textContent = "API Key validated and saved in this browser.";
        statusMsg.className = "status-msg success";
        localStorage.setItem("google_api_key", apiKey);
        validatedGeminiModels = data.models;
        updateModelDropdown();
    } catch (error) {
        statusMsg.textContent = "Error: " + error.message;
        statusMsg.className = "status-msg error";
        localStorage.removeItem("google_api_key");
        validatedGeminiModels = [];
        updateModelDropdown();
    } finally {
        validateBtn.disabled = false;
    }
}

function updateModelDropdown(preferred) {
    const modelSelect = document.getElementById('modelSelect');
    const currentValue = preferred || modelSelect.value;
    modelSelect.innerHTML = '';

    ollamaModels.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.name;
        opt.textContent = m.installed === false
            ? `${m.name} (Ollama, not pulled)`
            : `${m.name} (Ollama)`;
        modelSelect.appendChild(opt);
    });

    validatedGeminiModels.forEach(modelName => {
        const opt = document.createElement('option');
        // The API returns names like "models/gemini-1.5-flash"; the backend expects "gemini-1.5-flash"
        opt.value = modelName.replace('models/', '');
        opt.textContent = "Gemini " + opt.value.replace('gemini-', '').toUpperCase();
        modelSelect.appendChild(opt);
    });

    // Try to restore previous selection if it still exists
    if ([...modelSelect.options].some(opt => opt.value === currentValue)) {
        modelSelect.value = currentValue;
    }

    handleModelChange();
}

function handleModelChange() {
    const modelSelect = document.getElementById('modelSelect');
    const modeSelect = document.getElementById('modeSelect');
    const isGemini = modelSelect.value.startsWith('gemini-');

    // Disable Line-by-Line if an API model is selected
    for (let i = 0; i < modeSelect.options.length; i++) {
        if (modeSelect.options[i].value === 'line-by-line') {
            modeSelect.options[i].disabled = isGemini;
            if (isGemini && modeSelect.value === 'line-by-line') {
                modeSelect.value = 'chunking';
            }
            break;
        }
    }

    toggleChunkPercentage();
}

function toggleChunkPercentage() {
    const modeSelect = document.getElementById('modeSelect');
    const chunkPercentageContainer = document.getElementById('chunkPercentageContainer');

    if (modeSelect.value === 'chunking') {
        chunkPercentageContainer.classList.remove('hidden');
    } else {
        chunkPercentageContainer.classList.add('hidden');
    }
}

function updateMLOnlyToggle() {
    const prefilterToggle = document.getElementById('prefilterToggle');
    const mlOnlyToggle = document.getElementById('mlOnlyToggle');
    if (prefilterToggle && mlOnlyToggle) {
        mlOnlyToggle.disabled = !prefilterToggle.checked;
        if (!prefilterToggle.checked) {
            mlOnlyToggle.checked = false;
        }
    }
}

// Build "<span class=cls>text</span>" safely
function span(cls, text) {
    const el = document.createElement('span');
    el.className = cls;
    el.textContent = text;
    return el;
}

// Replace an element's children with a mix of strings and nodes (no innerHTML)
function setContent(el, parts) {
    el.replaceChildren(...parts.map(p => typeof p === 'string' ? document.createTextNode(p) : p));
}

function setNoResultsRow(colSpan) {
    const tbody = document.getElementById('resultsTableBody');
    const row = document.createElement('tr');
    row.id = 'noResultsRow';
    const cell = document.createElement('td');
    cell.id = 'noResultsCell';
    cell.colSpan = colSpan;
    cell.style.textAlign = 'center';
    cell.style.fontStyle = 'italic';
    cell.textContent = 'No anomalies detected.';
    row.appendChild(cell);
    tbody.replaceChildren(row);
}

async function analyzeLogs() {
    const fileInput = document.getElementById('logFileInput');
    const modelSelect = document.getElementById('modelSelect');
    const modeSelect = document.getElementById('modeSelect');
    const speedSelect = document.getElementById('speedSelect');
    const apiKeyInput = document.getElementById('apiKeyInput');
    const analyzeBtn = document.getElementById('analyzeBtn');

    hideError();
    if (fileInput.files.length === 0) {
        showError("Please select a log file to analyze.");
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);
    formData.append("model", modelSelect.value);
    formData.append("mode", modeSelect.value);
    formData.append("speed", speedSelect.value);
    if (modelSelect.value.startsWith('gemini-')) {
        formData.append("google_api_key", apiKeyInput.value.trim());
    }

    const prefilterToggle = document.getElementById('prefilterToggle');
    formData.append("use_prefilter", prefilterToggle.checked ? "true" : "false");

    const mlOnlyToggle = document.getElementById('mlOnlyToggle');
    formData.append("ml_only", mlOnlyToggle.checked ? "true" : "false");

    const chunkPercentageInput = document.getElementById('chunkPercentageInput');
    if (chunkPercentageInput && modeSelect.value === 'chunking') {
        formData.append("chunk_percentage", chunkPercentageInput.value);
    } else {
        formData.append("chunk_percentage", 10);
    }

    // Show loading state
    analyzeBtn.disabled = true;
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('resultsSection').classList.add('hidden');
    document.getElementById('prefilterInfo').classList.add('hidden');
    setNoResultsRow(4);

    // Reset both progress bars
    document.getElementById('mlProgressContainer').classList.add('hidden');
    document.getElementById('llmProgressContainer').classList.add('hidden');
    document.getElementById('mlProgressBar').style.width = '0%';
    document.getElementById('mlProgressBar').classList.remove('complete');
    document.getElementById('progressBar').style.width = '0%';
    document.getElementById('progressText').textContent = '0%';
    document.getElementById('loadingStageText').textContent = 'Starting analysis...';

    let gotResult = false;
    try {
        const response = await fetch("/upload-log/", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(await errorFromResponse(response));
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const parts = buffer.split('\n');
            buffer = parts.pop() || '';

            for (const part of parts) {
                const trimmed = part.trim();
                if (!trimmed.startsWith('data: ')) continue;

                let data;
                try {
                    data = JSON.parse(trimmed.slice(6));
                } catch (e) {
                    console.warn('Failed to parse SSE data:', e);
                    continue;
                }

                if (data.type === 'prefilter_progress') {
                    const mlBar = document.getElementById('mlProgressBar');
                    const mlText = document.getElementById('mlProgressText');
                    const pct = Math.round((data.current / data.total) * 100);
                    mlBar.style.width = pct + '%';
                    mlText.textContent = data.message || `ML Pre-filter: ${pct}%`;
                } else if (data.type === 'progress') {
                    const llmContainer = document.getElementById('llmProgressContainer');
                    if (llmContainer.classList.contains('hidden')) {
                        document.getElementById('mlProgressContainer').classList.add('hidden');
                        llmContainer.classList.remove('hidden');
                        document.getElementById('loadingStageText').textContent = 'Stage 2/2: LLM Analysis';
                    }
                    const percent = Math.round((data.current / data.total) * 100);
                    document.getElementById('progressBar').style.width = percent + '%';
                    document.getElementById('progressText').textContent = percent + '%';
                } else if (data.type === 'prefilter') {
                    handlePrefilterEvent(data);
                } else if (data.type === 'result') {
                    gotResult = true;
                    document.getElementById('loading').classList.add('hidden');
                    displayResults(data);
                } else if (data.type === 'error') {
                    gotResult = true;
                    document.getElementById('loading').classList.add('hidden');
                    showError(data.message);
                }
            }
        }
        if (!gotResult) {
            throw new Error("The connection closed before the analysis finished.");
        }
    } catch (error) {
        document.getElementById('loading').classList.add('hidden');
        showError("Error analyzing log file: " + error.message);
    } finally {
        analyzeBtn.disabled = false;
    }
}

function handlePrefilterEvent(data) {
    const mlContainer = document.getElementById('mlProgressContainer');
    const llmContainer = document.getElementById('llmProgressContainer');
    const prefilterInfo = document.getElementById('prefilterInfo');
    const stageText = document.getElementById('loadingStageText');

    if (data.status === 'training') {
        mlContainer.classList.remove('hidden');
        llmContainer.classList.add('hidden');
        stageText.textContent = 'Stage 1/2: ML Pre-filter';
        prefilterInfo.className = 'prefilter-info';
        prefilterInfo.textContent = `ML Pre-filter: Training on ${data.total} log lines...`;
    } else if (data.status === 'done') {
        // Mark ML bar as complete
        const mlBar = document.getElementById('mlProgressBar');
        mlBar.style.width = '100%';
        mlBar.classList.add('complete');
        document.getElementById('mlProgressText').textContent = 'ML Pre-filter complete ✓';

        const summary = [
            'ML Pre-filter flagged ', span('highlight', String(data.flagged)),
            ' lines out of ', span('highlight', String(data.total)), ' total. ',
            span('filtered-out', `${data.filtered_out} lines filtered out.`), ' ',
        ];
        if (data.ml_only) {
            setContent(prefilterInfo, [...summary, span('remaining', 'LLM analysis skipped (ML Only mode).')]);
            stageText.textContent = 'ML Only — generating results...';
        } else {
            setContent(prefilterInfo, [...summary, span('remaining', `Proceeding with LLM analysis on ${data.flagged} lines.`)]);

            // Transition to LLM bar
            stageText.textContent = 'Stage 2/2: LLM Analysis';
            setTimeout(() => {
                mlContainer.classList.add('hidden');
                llmContainer.classList.remove('hidden');
            }, 1500);
        }
    } else if (data.status === 'skipped') {
        prefilterInfo.className = 'prefilter-info hidden';
        mlContainer.classList.add('hidden');
        llmContainer.classList.remove('hidden');
        stageText.textContent = 'Stage 2/2: LLM Analysis';
    }
}

function displayResults(data) {
    // Reveal the results section
    document.getElementById('resultsSection').classList.remove('hidden');

    document.getElementById('totalLogs').textContent = data.total_lines;
    document.getElementById('anomaliesDetected').textContent = data.anomalies.length;

    // Hide prediction column in ML Only mode
    const mlOnly = data.ml_only || false;
    const predHeader = document.getElementById('predictionHeader');
    if (predHeader) {
        predHeader.style.display = mlOnly ? 'none' : '';
    }

    if (data.failed_lines) {
        showError(`${data.failed_lines} line(s) could not be analysed by the LLM. Check the server logs for details.`, 'warning');
    }

    // Show prefilter stats if enabled
    const prefilterInfo = document.getElementById('prefilterInfo');
    if (data.prefilter_enabled) {
        const head = document.createElement('strong');
        head.textContent = 'ML Pre-filter:';
        if (mlOnly) {
            setContent(prefilterInfo, [
                head, ` Scanned ${data.total_lines} lines, flagged `,
                span('highlight', String(data.prefilter_flagged)),
                ' anomalies. LLM analysis skipped (ML Only mode).',
            ]);
        } else {
            setContent(prefilterInfo, [
                head, ` Scanned ${data.total_lines} lines, flagged `,
                span('highlight', String(data.prefilter_flagged)),
                ' potential anomalies. LLM analyzed ',
                span('remaining', String(data.prefilter_remaining)), ' lines in detail.',
            ]);
        }
        prefilterInfo.className = 'prefilter-info';
    }

    const tbody = document.getElementById('resultsTableBody');

    if (data.anomalies.length === 0) {
        setNoResultsRow(mlOnly ? 3 : 4);
        return;
    }

    tbody.replaceChildren();
    data.anomalies.forEach(anomaly => {
        const row = document.createElement('tr');
        row.className = 'anomaly-row';

        const lineCell = document.createElement('td');
        lineCell.textContent = anomaly.line;

        const contentCell = document.createElement('td');
        const code = document.createElement('code');
        code.textContent = anomaly.content;
        contentCell.appendChild(code);

        const scoreCell = document.createElement('td');
        scoreCell.textContent = anomaly.score;

        const predictionCell = document.createElement('td');
        predictionCell.textContent = anomaly.prediction;
        if (mlOnly) predictionCell.style.display = 'none';

        row.append(lineCell, contentCell, scoreCell, predictionCell);
        tbody.appendChild(row);
    });
}

// GPU Status Check
async function checkGpuStatus() {
    const el = document.getElementById('gpuStatus');
    const text = el.querySelector('.gpu-text');

    try {
        const res = await fetch('/api/gpu-status');
        const data = await res.json();

        el.classList.remove('loading');

        if (data.gpu_active) {
            el.classList.add('gpu-ok');
            const vram = data.nvidia_gpu?.vram ? ` (${data.nvidia_gpu.vram} VRAM)` : '';
            text.textContent = `GPU active: ${data.nvidia_gpu?.name || 'GPU'}${vram}`;
        } else if (data.nvidia_gpu && data.loaded_models?.length) {
            el.classList.add('gpu-warn');
            text.textContent = `GPU detected (${data.nvidia_gpu.name}) but Ollama is not using it. `;
            const action = span('gpu-action', 'How to fix');
            action.addEventListener('click', () => {
                showError(
                    'To make Ollama use your NVIDIA GPU: quit Ollama (system tray → Quit), set the GPU ' +
                    'environment variables for the Ollama server (e.g. OLLAMA_VULKAN=0 on hybrid-GPU ' +
                    'laptops), start Ollama again, then refresh this page. See docs/troubleshooting.md.',
                    'warning'
                );
            });
            text.appendChild(action);
        } else if (data.ollama_reachable) {
            el.classList.add(data.nvidia_gpu ? 'gpu-warn' : 'gpu-err');
            text.textContent = data.message;
        } else {
            el.classList.add('gpu-err');
            text.textContent = data.message || 'GPU status unknown.';
        }
    } catch (e) {
        el.classList.remove('loading');
        el.classList.add('gpu-err');
        text.textContent = 'Could not check GPU status.';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('modelSelect').addEventListener('change', handleModelChange);
    document.getElementById('modeSelect').addEventListener('change', toggleChunkPercentage);
    document.getElementById('validateKeyBtn').addEventListener('click', validateAndSaveKey);
    document.getElementById('analyzeBtn').addEventListener('click', analyzeLogs);
    document.getElementById('errorDismiss').addEventListener('click', hideError);

    const prefilterToggle = document.getElementById('prefilterToggle');
    prefilterToggle.addEventListener('change', updateMLOnlyToggle);
    updateMLOnlyToggle();

    checkGpuStatus();
    loadModels();

    // Load saved API key
    const savedKey = localStorage.getItem("google_api_key");
    if (savedKey) {
        document.getElementById('apiKeyInput').value = savedKey;
        validateAndSaveKey();
    }
});
