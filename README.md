# NBLM Small

FastAPI RAG workspace with PDF/TXT upload, ChromaDB indexing, chat, quizzes, summaries, podcast generation, and a cinematic slide preview/export engine.

## Requirements

- Python 3.10 or newer
- Git
- At least one LLM API key for AI generation features
- Windows PowerShell if you want to use `run_app.ps1`

The app can start without API keys, but chat/slides/quiz/summary generation will fail until at least one provider key is configured.

## Setup

Clone the repository:

```powershell
git clone https://github.com/Mphuc310771/NBLM-small.git
cd NBLM-small
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional, for Playwright browser automation:

```powershell
playwright install chromium
```

Create your local environment file:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and add one or more API keys:

```env
GROQ_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
SAMBANOVA_API_KEY=your_key_here
MISTRAL_API_KEY=your_key_here
```

## Run

Windows PowerShell:

```powershell
.\run_app.ps1
```

Manual run:

```powershell
python app/workers/vision_grpc_server.py
python -m uvicorn app.main:app --port 8000
```

Open:

```text
http://localhost:8000
```

## Common Workflow

1. Open the web app.
2. Upload a `.txt` or `.pdf` document.
3. Ask questions against the uploaded document.
4. Generate summaries, quizzes, podcasts, or slides.
5. Open the slide preview and use `Export to PDF`.

## Local Data

The app creates local runtime data that is intentionally not committed:

- `.env`
- `.venv/`, `venv/`, `venv_win/`
- `chroma_db/`
- `scratch/`
- `app/static/outputs/`

Delete `chroma_db/` if you want to reset all indexed documents.

## Notes

- First startup can be slow because `sentence-transformers` downloads the embedding model.
- The gRPC vision worker has a fallback path if the native C++ vision library is not compiled.
- Never commit real API keys. Keep them only in `.env`.
