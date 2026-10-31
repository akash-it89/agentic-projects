# Running on Windows: checklist

Everything here runs on CPU, so an Intel or AMD GPU is fine. Only project 1's talking avatar (Ditto) needs an NVIDIA GPU; see "Project 1" below.

## 1. Check the machine

| Need | Minimum | Notes |
|---|---|---|
| Windows | 10 or 11, 64-bit | |
| RAM | 8 GB | 16 GB if you use the 7B model or run OCR and the LLM together |
| Free disk | ~15 GB | Mostly models and Python environments, downloaded on first use (sizes below) |
| Internet | First run | Models download on first use; after that, everything runs offline |
| Admin rights | Only for Docker Desktop | uv and Ollama install per user. Docker is needed for project 1 only |
| Virtualization | Enabled in BIOS | Docker only. Check in Task Manager → Performance → CPU → "Virtualization: Enabled" |

Approximate first-run downloads:

| Download | Size | Used by |
|---|---|---|
| `qwen2.5:3b` | ~2 GB | all projects |
| `qwen2.5:7b` | ~4.7 GB | optional, better answers |
| PaddleOCR models | a few hundred MB | projects 2, 3 |
| BGE-M3 + reranker | ~4.5 GB | project 1 |
| Whisper `small` | ~0.5 GB | project 1 |
| Kokoro Docker image | several GB | project 1 |

## 2. Install the tools (PowerShell)

```powershell
winget install -e --id Git.Git
winget install -e --id astral-sh.uv
winget install -e --id Ollama.Ollama
winget install -e --id Docker.DockerDesktop     # project 1 only; needs admin and a reboot
```

You don't need to install Python yourself: uv downloads Python 3.12 the first time you run `uv sync`. To use your own, `winget install -e --id Python.Python.3.12` works too.

**Close PowerShell and open a new window**, then check:

```powershell
git --version
uv --version
ollama --version
ollama pull qwen2.5:3b
git config --global core.longpaths true
```

Ollama runs in the background after install, with an icon in the system tray. If the projects say the LLM connection is refused, start Ollama from the Start menu.

For every new PowerShell window you use for these projects:

```powershell
$env:PYTHONUTF8 = "1"                                   # Hindi, Tamil and Arabic print correctly
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"     # skips a slow network check in PaddleOCR
```

## 3. Get the code

```powershell
mkdir C:\work -Force
cd C:\work
git clone https://github.com/akash-it89/agentic-projects.git
cd agentic-projects
```

Use a short path like `C:\work`. Avoid OneDrive folders, because syncing slows installs down a lot.

## 4. Project 3: Document intelligence (start here)

```powershell
cd C:\work\agentic-projects\p3-docintel
Copy-Item .env.example .env
notepad .env                                  # set OLLAMA_MODEL=qwen2.5:3b if you only pulled the 3B model
uv sync --group ocr
uv run pytest                                 # expect: 36 passed
uv run python -m p3docintel samples
uv run python -m p3docintel serve
```

Open http://localhost:8000. Upload `samples\docs\utility_bill.pdf` (no OCR needed), then `samples\docs\utility_bill_scan.png` (OCR; the first one downloads the OCR models). You should see fields on the left; click one and it lights up on the page.

Score it from a second window: `uv run python -m p3docintel eval`.

## 5. Project 2: Newspaper digitization

```powershell
cd C:\work\agentic-projects\p2-newspaper
Copy-Item .env.example .env
uv sync --group ocr
uv run pytest                                 # expect: 24 passed
uv run python -m p2news synth --pages 3
uv run python -m p2news run --input data\synthetic\pages --out runs\ocr
uv run python -m p2news eval --pred runs\ocr\pages --gold data\synthetic\gold
uv run python -m p2news cost --run runs\ocr --pages 1000000 --price 0.5
```

Look at `runs\ocr\overlays\*.png` for the article boxes. Expect about a minute per page on CPU. To test the layout logic alone in seconds, add `--gold-lines data\synthetic\gold` to the `run` command.

## 6. Project 1: Voice avatar (audio answers; no avatar without an NVIDIA GPU)

Open Docker Desktop and wait until it says it's running. Then:

```powershell
cd C:\work\agentic-projects\p1-avatar
Copy-Item .env.example .env
notepad .env
```

Change these lines in `.env`:

```
DEVICE=cpu
WHISPER_MODEL=small
OLLAMA_MODEL=qwen2.5:3b
AVATAR_ENABLED=false
```

```powershell
uv sync --group rag --group voice
uv run pytest                                 # expect: 16 passed
docker compose --profile cpu up -d            # Qdrant + Kokoro text to speech
uv run python -m p1avatar.rag.ingest          # first run downloads ~4.5 GB of retrieval models
uv run uvicorn p1avatar.api.main:app --port 8080
```

Open http://localhost:8080. Type a question first ("Where is gate C22?"), then try the microphone. A question the data doesn't cover should get a polite refusal. For the talking avatar, run `ditto_service` on a rented NVIDIA GPU and point `DITTO_URL` at it (see `p1-avatar/ditto_service/README_WSL.md`).

Projects 1 and 2 both use port 8080: project 1's kiosk and project 2's optional Label Studio. Run one at a time, or change a port.

## 7. If something goes wrong

| Symptom | Fix |
|---|---|
| `uv`, `git` or `ollama` "is not recognized" | Open a new PowerShell window after installing |
| "Connection refused" or "Connection error" from the LLM, or no fields extracted | Ollama isn't running: start it from the Start menu |
| `uv sync --group ocr` fails on paddlepaddle | Install PaddlePaddle with the CPU command from https://www.paddlepaddle.org.cn/en/install/quick, then run `uv sync --group ocr` again |
| "Scans and photos need OCR, which isn't installed" | `uv sync --group ocr`, then restart the app |
| `docker compose` fails | Docker Desktop isn't running yet, or virtualization is off in the BIOS |
| "Address already in use" | Something else uses the port: `netstat -ano \| findstr :8000`, or start on another port (`serve --port 8001`) |
| Windows Firewall prompt | Allow on Private networks |
| `????` instead of Tamil, Hindi or Arabic in the console | `$env:PYTHONUTF8 = "1"` (the web pages are unaffected) |
| Very slow | Use `qwen2.5:3b` and `WHISPER_MODEL=small`; close other heavy apps. The first run is always slower (downloads, antivirus scanning new files) |
| Microphone doesn't work in project 1 | Use `http://localhost:8080`, not the machine's IP, and allow microphone access when the browser asks |

## Updating

```powershell
cd C:\work\agentic-projects
git pull
cd p3-docintel
uv sync --group ocr                           # repeat for any project whose uv.lock changed
```
