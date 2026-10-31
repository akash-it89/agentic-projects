"""HTTP wrapper around Ditto (antgroup/ditto-talkinghead). Runs inside WSL2 or on a Linux GPU box.

    cd ditto_service
    DITTO_REPO=~/ditto-talkinghead DITTO_SOURCE_IMAGE=~/avatar.png \
        uvicorn server:app --host 0.0.0.0 --port 8090

POST /render  (multipart: audio=<wav>, optional image=<png/jpg>)  ->  video/mp4 with the audio muxed in
GET  /health

Ditto's CLI and checkpoint names come from its README at the time of writing. If inference.py changes,
update build_command() only.
"""

import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

REPO = Path(os.environ.get("DITTO_REPO", "~/ditto-talkinghead")).expanduser()
PYTHON = os.environ.get("DITTO_PYTHON", "python")  # python of the env where Ditto's requirements are installed
DATA_ROOT = os.environ.get("DITTO_DATA_ROOT", "./checkpoints/ditto_pytorch")
CFG_PKL = os.environ.get("DITTO_CFG_PKL", "./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl")
SOURCE_IMAGE = Path(os.environ.get("DITTO_SOURCE_IMAGE", "~/avatar.png")).expanduser()
TIMEOUT = int(os.environ.get("DITTO_TIMEOUT", "300"))
WORK = Path(__file__).resolve().parent / "tmp"
WORK.mkdir(exist_ok=True)

app = FastAPI(title="Ditto avatar service")


def build_command(audio: Path, image: Path, out: Path) -> list[str]:
    return [PYTHON, "inference.py", "--data_root", DATA_ROOT, "--cfg_pkl", CFG_PKL,
            "--audio_path", str(audio), "--source_path", str(image), "--output_path", str(out)]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    r = subprocess.run(cmd, check=False, cwd=cwd, capture_output=True, text=True, timeout=TIMEOUT)
    if r.returncode != 0:
        raise HTTPException(500, f"{Path(cmd[0]).name} failed: {r.stderr[-1500:]}")


@app.get("/health")
def health():
    problems = [msg for ok, msg in [
        (REPO.joinpath("inference.py").exists(), f"inference.py not found in {REPO}"),
        (SOURCE_IMAGE.exists(), f"avatar image not found: {SOURCE_IMAGE}"),
        (shutil.which("ffmpeg") is not None, "ffmpeg not installed"),
    ] if not ok]
    if problems:
        raise HTTPException(503, "; ".join(problems))
    return {"status": "ok"}


@app.post("/render")
def render(audio: UploadFile = File(...), image: UploadFile | None = File(None)):
    job = WORK / uuid.uuid4().hex[:12]
    job.mkdir()
    raw = job / "input_audio"
    raw.write_bytes(audio.file.read())

    # Ditto's audio encoder expects 16 kHz mono; Kokoro outputs 24 kHz.
    wav16 = job / "audio16k.wav"
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-ac", "1", "-ar", "16000", str(wav16)])

    src = SOURCE_IMAGE
    if image is not None:
        src = job / f"source{Path(image.filename or 'x.png').suffix}"
        src.write_bytes(image.file.read())

    silent = job / "ditto.mp4"
    run(build_command(wav16, src, silent), cwd=REPO)

    # Always mux the original audio, so playback quality doesn't depend on what Ditto wrote.
    final = Path(tempfile.gettempdir()) / f"{job.name}.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-i", str(raw), "-map", "0:v", "-map", "1:a",
         "-c:v", "copy", "-c:a", "aac", "-shortest", str(final)])
    shutil.rmtree(job, ignore_errors=True)
    return FileResponse(final, media_type="video/mp4", filename="answer.mp4")
