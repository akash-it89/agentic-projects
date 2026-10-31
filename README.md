# Agentic AI capstone projects

Three end-to-end AI projects built from real-world briefs, each runnable on a CPU-only laptop (Windows, macOS or Linux) with local models, and switchable to AWS where it helps.

| Project | What it does | Key techniques |
|---|---|---|
| [p1-avatar](p1-avatar/) | Multilingual airport kiosk: ask by voice in English, Hindi or Spanish; a talking avatar answers only from the airport's data and refuses otherwise | RAG with reranking and a refusal threshold, speech to text (Whisper), text to speech (Kokoro), Ditto talking head, LiveKit real-time agent, Amazon Nova Sonic |
| [p2-newspaper](p2-newspaper/) | Digitizes old newspaper scans on a budget: one ID, text and bounding box per article | PaddleOCR, layout analysis, a rules → small-LLM decider cascade, union-find grouping, IoU/F1 evaluation, cost model |
| [p3-docintel](p3-docintel/) | Upload any document: extract every field with its location, classify, summarise, and chat with clickable citations | Grounded extraction (the LLM cites line IDs; code computes boxes), PDF text layers, OCR routing, BM25 or embedding retrieval, FastAPI + viewer |

Each project has its own README with setup, run steps, evaluation and design notes.

## Quick start (Windows PowerShell)

For the full Windows checklist (hardware, installs, every project, troubleshooting), see [WINDOWS.md](WINDOWS.md).

```powershell
winget install -e --id Python.Python.3.12
winget install -e --id astral-sh.uv
winget install -e --id Ollama.Ollama
# reopen PowerShell, then:
ollama pull qwen2.5:3b
cd p3-docintel
Copy-Item .env.example .env
uv sync --group ocr
uv run pytest
uv run python -m p3docintel samples
uv run python -m p3docintel serve     # http://localhost:8000
```

On macOS or Linux, use `cp` instead of `Copy-Item`; everything else is the same.

## Conventions

- `uv` manages each project's environment. `uv.lock` pins exact versions, so `uv sync` reproduces the setup on any machine.
- Settings live in `.env` (copy it from `.env.example`). Secrets never go in the repo.
- Every project has tests that run without GPUs, cloud accounts or model servers, using fakes. Each project also ships sample data with gold labels and an `eval` command that scores the real pipeline.
- All sample documents and data are invented and marked as such.
