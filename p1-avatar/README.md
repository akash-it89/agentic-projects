# Project 1: Multilingual airport talking avatar

A kiosk where a visitor asks a question out loud in English, Hindi or Spanish, and a lip-synced avatar answers in the same language, using **only** the airport Markdown files. If the data doesn't cover the question, the kiosk refuses instead of guessing.

> The files in `data/airport/` are **invented sample data** so you can test the pipeline. Replace them with the trainer's folders (keep one top-level folder per language: `en/`, `hi/`, `es/`, or names starting with those letters).

## How it fits together

```
MVP (turn-based):  browser mic → /turn → faster-whisper → RAG (Qdrant + reranker) → LLM or refusal
                   → Kokoro TTS → Ditto (WSL2) → mp4 back to the kiosk page
Real time:         LiveKit room → agent (Speaches → Ollama → Kokoro, or Nova Sonic) → RAG as a tool
```

| Path | What it is |
|---|---|
| `src/p1avatar/rag/` | Markdown chunking, ingest into Qdrant, search + rerank + refusal threshold |
| `src/p1avatar/answer.py` | Grounded answer. Refusal is enforced in code, not only in the prompt |
| `src/p1avatar/llm/` | Ollama (default) or Bedrock, chosen with `LLM_PROVIDER` |
| `src/p1avatar/voice/` | Speech to text (faster-whisper) and text to speech (Kokoro) |
| `src/p1avatar/avatar/client.py` | Calls the Ditto service; falls back to audio only if it's down |
| `src/p1avatar/api/main.py` + `web/` | Kiosk API and page |
| `src/p1avatar/agent/livekit_agent.py` | Real-time agent (cascaded or Nova Sonic) |
| `ditto_service/` | Ditto wrapper that runs in WSL2. See `ditto_service/README_WSL.md` |
| `eval/` | Question set + eval runner |
| `tests/` | Unit tests; no GPU, models or network needed |

## Setup

Install the shared tools first (Python 3.12, uv, Docker Desktop with WSL2, Ollama); see the Setup tab in the build guide.

**Windows (PowerShell)**

```powershell
cd capstone\p1-avatar
Copy-Item .env.example .env
uv sync --group rag --group voice          # add --group realtime --group nova for phases 5-6
uv run pytest                              # quick check: should pass with no services running
docker compose --profile gpu up -d         # Qdrant + Kokoro + Speaches (use --profile cpu without a GPU)
ollama pull qwen2.5:7b
```

**macOS / Linux**

```bash
cd capstone/p1-avatar
cp .env.example .env
uv sync --group rag --group voice
uv run pytest
docker compose --profile cpu up -d         # --profile gpu on a Linux box with an NVIDIA GPU
ollama pull qwen2.5:7b
```

PyTorch with CUDA on Windows: if `uv run python -c "import torch; print(torch.cuda.is_available())"` prints `False` on a GPU machine, install the CUDA build of torch with the command from pytorch.org's selector (`uv pip install torch --index-url https://download.pytorch.org/whl/cu1xx`).

## Phase by phase

### 1. Index the data and test answers as text

```bash
uv run python -m p1avatar.rag.ingest
uv run uvicorn p1avatar.api.main:app --port 8080
```

Then, in a second terminal:

```powershell
# Windows
Invoke-RestMethod -Method Post -Uri http://localhost:8080/ask -ContentType "application/json" -Body '{"question":"Where is gate C22?"}'
Invoke-RestMethod -Method Post -Uri http://localhost:8080/ask -ContentType "application/json" -Body '{"question":"Is there a pharmacy?"}'
```

```bash
# macOS / Linux
curl -s localhost:8080/ask -H 'content-type: application/json' -d '{"question":"Where is gate C22?"}'
curl -s localhost:8080/ask -H 'content-type: application/json' -d '{"question":"Is there a pharmacy?"}'
```

The second question should come back with `"refused": true`.

### 2. Evaluate and tune the refusal threshold

```bash
uv run python eval/run_eval.py
```

Open `runs/eval-*/report.md`. If good questions are refused, lower `MIN_SCORE` in `.env`. If unanswerable ones get answers, raise it. Write your own `eval/questions.jsonl` for the trainer's data, with about 20 questions per language including 5 the data can't answer.

### 3. Voice kiosk (no avatar yet)

With the server running, open http://localhost:8080, hold the button and speak. The first request downloads the Whisper model (about 3 GB). Set `AVATAR_ENABLED=false` until phase 4 to skip the avatar check.

### 4. Avatar

Set up `ditto_service` in WSL2 (`ditto_service/README_WSL.md`), check `curl http://localhost:8090/health`, then set `AVATAR_ENABLED=true`. Answers now come back as lip-synced video.

### 5. Real time with LiveKit

```powershell
# Windows: download livekit-server from https://github.com/livekit/livekit/releases
.\livekit-server.exe --dev
uv sync --group rag --group voice --group realtime
uv run python -m p1avatar.agent.livekit_agent dev
```

```bash
# macOS: brew install livekit    Linux: curl -sSL https://get.livekit.io | bash
livekit-server --dev
uv sync --group rag --group voice --group realtime
uv run python -m p1avatar.agent.livekit_agent dev
```

The cascaded agent uses the Speaches container for speech to text. Download its model once:
`curl -X POST http://localhost:8000/v1/models/Systran/faster-whisper-large-v3`.
Connect to the room with the LiveKit Agents Playground or the `agent-starter-react` template (see the LiveKit docs).

### 6. Nova Sonic

Run `aws configure` (personal or course account), then:

```bash
uv sync --group rag --group realtime --group nova
# .env: AGENT_MODE=nova
uv run python -m p1avatar.agent.livekit_agent dev
```

Confirm the Nova Sonic model ID, region and voices in the Bedrock console. To use Bedrock for the text answers as well, set `LLM_PROVIDER=bedrock` and `BEDROCK_MODEL_ID`.

## Things to verify for your installed versions

These third-party names change between releases. Each is marked in the code:

- LiveKit Agents plugin arguments (`openai.STT(detect_language=...)`, `aws.realtime.RealtimeModel(...)`, the `user_input_transcribed` event)
- Ditto checkpoint folder and config names (`ditto_service/server.py`)
- Kokoro voice IDs (`src/p1avatar/lang.py`)
- Docker image tags (`docker-compose.yml`): pin them once everything works
