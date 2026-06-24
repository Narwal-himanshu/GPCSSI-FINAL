# GPCSSI-FINAL

A FastAPI-powered web application built for the GPCSSI program that lets you interact with local and cloud-based LLMs through a clean browser interface. It runs a Python backend with Uvicorn, uses Ollama for local model inference, and also supports Google's Generative AI (Gemini) for cloud-based responses.

---

## What it does

You get a web UI (served from the `frontend/` folder) that talks to a FastAPI backend. The backend handles requests, processes data, and routes them to either a locally running Ollama model or the Google Generative AI API. There's also a `data/` directory for datasets and an `examplelog/` folder with sample logs — probably from previous runs or for testing purposes.

---

## Tech Stack

- **Backend**: Python, FastAPI, Uvicorn
- **LLMs**: Ollama (local), Google Generative AI (`google-genai`)
- **Frontend**: HTML, CSS, JavaScript
- **Data handling**: NumPy, Pandas
- **File uploads**: python-multipart

---

## Prerequisites

Make sure you have the following before running anything:

- **Python 3.9+** installed
- **pip** available in your terminal
- **Ollama** installed and running (the setup script handles this — see below)
- A Google Generative AI API key if you want to use Gemini (optional)

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Narwal-himanshu/GPCSSI-FINAL.git
cd GPCSSI-FINAL
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up Ollama

Ollama is what powers the local model inference. Run the appropriate setup script for your OS:

**macOS / Linux:**
```bash
./scripts/setup_ollama.sh
```

**Windows (PowerShell):**
```powershell
./scripts/setup_ollama.ps1
```

> If you run into a permissions error on macOS/Linux, try `chmod +x scripts/setup_ollama.sh` first.

---

## Running the App

Once dependencies are installed and Ollama is set up, start the server:

```bash
uvicorn src.app:app --reload
```

Then open your browser and go to:

```
http://localhost:8000
```

The `--reload` flag means the server will automatically restart whenever you change a file, which is handy during development.

---

## Project Structure

```
GPCSSI-FINAL/
├── data/               # Datasets and input files
├── examplelog/         # Sample logs from previous runs
├── frontend/           # HTML, CSS, JS for the web UI
├── scripts/
│   ├── setup_ollama.sh     # Ollama setup for macOS/Linux
│   └── setup_ollama.ps1    # Ollama setup for Windows
├── src/
│   └── app.py          # FastAPI application entry point
├── requirements.txt
└── uvicorn_output.log  # Server output log
```

---

## Environment Variables

If you're using the Google Generative AI integration, you'll need to set your API key:

```bash
export GOOGLE_API_KEY=your_api_key_here
```

On Windows:
```powershell
$env:GOOGLE_API_KEY = "your_api_key_here"
```

---

## Roadmap / Known TODOs

- [ ] Add custom model support — particularly Gemma 4 and Gemini 2 Pro
- [ ] Better error handling and user-facing messages when a model isn't available
- [ ] Proper environment variable / config file support

---

## Troubleshooting

**Ollama isn't connecting?**
Make sure Ollama is actually running in the background. You can start it manually with `ollama serve` and check it's up at `http://localhost:11434`.

**Port 8000 already in use?**
Run on a different port: `uvicorn src.app:app --reload --port 8001`

**Module not found errors?**
Double-check you're in the right directory and that `pip install -r requirements.txt` completed without errors.

---

## Contributing

This project was built as part of the GPCSSI program. Feel free to open issues or PRs if you run into something broken or have ideas for improvements.
