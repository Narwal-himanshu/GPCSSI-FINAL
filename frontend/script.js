let validatedGeminiModels = [];

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

        const data = await response.json();

        if (response.ok && data.status === "success") {
            statusMsg.textContent = "API Key validated and saved!";
            statusMsg.className = "status-msg success";
            localStorage.setItem("google_api_key", apiKey);
            validatedGeminiModels = data.models;
            updateModelDropdown();
        } else {
            throw new Error(data.detail || "Validation failed");
        }
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

function updateModelDropdown() {
    const modelSelect = document.getElementById('modelSelect');
    // Keep local models
    const localModels = [
        { value: "llama3.2:1b", text: "Llama 3.2 (1B)" },
        { value: "qwen2.5:0.5b", text: "Qwen 2.5 (0.5B)" }
    ];

    const currentValue = modelSelect.value;
    modelSelect.innerHTML = '';

    localModels.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.value;
        opt.textContent = m.text;
        modelSelect.appendChild(opt);
    });

    validatedGeminiModels.forEach(modelName => {
        const opt = document.createElement('option');
        // The API returns names like "models/gemini-1.5-flash"
        // Our detector expects "gemini-1.5-flash" or the full name
        // We'll keep the full name but show a friendly text
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

// Add event listener for model change
document.addEventListener('DOMContentLoaded', () => {
    const modelSelect = document.getElementById('modelSelect');
    if (modelSelect) {
        modelSelect.addEventListener('change', handleModelChange);
    }
});

async function analyzeLogs() {
    const fileInput = document.getElementById('logFileInput');
    const modelSelect = document.getElementById('modelSelect');
    const modeSelect = document.getElementById('modeSelect');
    const speedSelect = document.getElementById('speedSelect');
    const apiKeyInput = document.getElementById('apiKeyInput');

    if (fileInput.files.length === 0) {
        alert("Please select a log file to analyze.");
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);
    formData.append("model", modelSelect.value);
    formData.append("mode", modeSelect.value);
    formData.append("speed", speedSelect.value);
    formData.append("google_api_key", apiKeyInput.value);

    const chunkPercentageInput = document.getElementById('chunkPercentageInput');
    if (chunkPercentageInput && modeSelect.value === 'chunking') {
        formData.append("chunk_percentage", chunkPercentageInput.value);
    } else {
        formData.append("chunk_percentage", 10);
    }

    // Show loading state
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('resultsSection').classList.add('hidden');
    document.getElementById('resultsTableBody').innerHTML = `
        <tr id="noResultsRow">
            <td colspan="4" style="text-align: center; font-style: italic;">No anomalies detected.</td>
        </tr>
    `;
    document.getElementById('progressBar').style.width = '0%';
    document.getElementById('progressText').textContent = '0%';

    try {
        const response = await fetch("/upload-log/", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
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
                if (trimmed.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(trimmed.slice(6));

                        if (data.type === 'progress') {
                            const percent = Math.round((data.current / data.total) * 100);
                            document.getElementById('progressBar').style.width = percent + '%';
                            document.getElementById('progressText').textContent = percent + '%';
                        } else if (data.type === 'result') {
                            document.getElementById('loading').classList.add('hidden');
                            displayResults(data);
                        } else if (data.type === 'error') {
                            document.getElementById('loading').classList.add('hidden');
                            alert("Error: " + data.message);
                        }
                    } catch (e) {
                        console.warn('Failed to parse SSE data:', e);
                    }
                }
            }
        }
    } catch (error) {
        document.getElementById('loading').classList.add('hidden');
        alert("Error analyzing log file: " + error.message);
    }
}

function displayResults(data) {
    // Reveal the results section
    document.getElementById('resultsSection').classList.remove('hidden');

    document.getElementById('totalLogs').textContent = data.total_lines;
    document.getElementById('anomaliesDetected').textContent = data.anomalies.length;

    const tbody = document.getElementById('resultsTableBody');
    tbody.innerHTML = '';

    if (data.anomalies.length === 0) {
        tbody.innerHTML = `
            <tr id="noResultsRow">
                <td colspan="4" style="text-align: center; font-style: italic;">No anomalies detected.</td>
            </tr>
        `;
        return;
    }

    data.anomalies.forEach(anomaly => {
        const row = document.createElement('tr');
        row.className = 'anomaly-row';

        row.innerHTML = `
            <td>${anomaly.line}</td>
            <td><code>${anomaly.content}</code></td>
            <td>${anomaly.score}</td>
            <td>${anomaly.prediction}</td>
        `;

        tbody.appendChild(row);
    });
}

// GPU Status Check
async function checkGpuStatus() {
    const el = document.getElementById('gpuStatus');
    const indicator = el.querySelector('.gpu-indicator');
    const text = el.querySelector('.gpu-text');

    try {
        const res = await fetch('/api/gpu-status');
        const data = await res.json();

        el.classList.remove('loading');

        if (data.gpu_active) {
            el.classList.add('gpu-ok');
            text.textContent = `GPU active: ${data.nvidia_gpu?.name || 'GPU'} (${data.nvidia_gpu?.vram || '?'} VRAM)`;
        } else if (data.nvidia_gpu) {
            el.classList.add('gpu-warn');
            text.innerHTML = `GPU detected (${data.nvidia_gpu.name}) but Ollama is not using it. `;
            const action = document.createElement('span');
            action.className = 'gpu-action';
            action.textContent = 'How to fix';
            action.onclick = () => {
                alert(
                    'To force Ollama to use your NVIDIA GPU:\n\n' +
                    '1. Right-click the Ollama icon in the system tray\n' +
                    '   and select "Quit" to stop Ollama completely.\n\n' +
                    '2. Start the app with:\n' +
                    '   uvicorn src.app:app --reload\n\n' +
                    '   The app sets OLLAMA_VULKAN=0 and\n' +
                    '   OLLAMA_LLM_LIBRARY=cuda_v13 automatically\n' +
                    '   (project-level only, not system-wide).\n\n' +
                    '3. Start Ollama from the Start Menu.\n\n' +
                    '4. Refresh this page.\n\n' +
                    'The env vars are set inside the Python app so\n' +
                    'they only affect this project.'
                );
            };
            text.appendChild(action);
        } else if (!data.nvidia_gpu) {
            el.classList.add('gpu-err');
            text.textContent = 'No NVIDIA GPU detected. Running on CPU.';
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
    checkGpuStatus();

    // Load saved API key
    const savedKey = localStorage.getItem("google_api_key");
    if (savedKey) {
        document.getElementById('apiKeyInput').value = savedKey;
        validateAndSaveKey();
    }
});
