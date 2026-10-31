"""Turn-based kiosk API (the MVP). One request = one conversational turn.

    uv run uvicorn p1avatar.api.main:app --port 8080
    open http://localhost:8080
"""

import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..answer import answer
from ..config import get_settings
from ..lang import KOKORO_VOICES, NAMES, detect_lang

WEB_DIR = Path(__file__).resolve().parents[1] / "web"

settings = get_settings()
settings.media_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Airport kiosk")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
app.mount("/media", StaticFiles(directory=settings.media_dir), name="media")


class AskRequest(BaseModel):
    question: str
    lang: str | None = None


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health():
    from ..avatar.client import avatar_available

    return {"status": "ok", "llm_provider": settings.llm_provider, "asr_backend": settings.asr_backend,
            "avatar": avatar_available()}


@app.get("/voices")
def voices():
    return {"languages": NAMES, "voices": KOKORO_VOICES}


# Plain `def` endpoints run in FastAPI's thread pool, so slow model calls don't block the server.
@app.post("/ask")
def ask(req: AskRequest):
    """Text in, text out. Useful for testing RAG and refusals without any audio services."""
    t0 = time.perf_counter()
    lang = req.lang or detect_lang(req.question) or "en"
    a = answer(req.question, lang)
    return {"question": req.question, "lang": a.lang, "answer": a.text, "refused": a.refused, "reason": a.reason,
            "passages": a.passages, "llm": a.llm, "ms": round((time.perf_counter() - t0) * 1000)}


@app.post("/turn")
def turn(audio: UploadFile = File(...), voice: str | None = Form(None), lang: str | None = Form(None)):
    """Audio in → transcript, answer, answer audio and (if the avatar is up) a lip-synced video."""
    from ..avatar.client import render_avatar
    from ..voice.asr import get_asr
    from ..voice.tts import get_tts

    turn_id = uuid.uuid4().hex[:12]
    turn_dir = settings.media_dir / turn_id
    turn_dir.mkdir(parents=True)
    suffix = Path(audio.filename or "question.webm").suffix or ".webm"
    question_path = turn_dir / f"question{suffix}"
    question_path.write_bytes(audio.file.read())

    timings: dict[str, int] = {}
    t = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal t
        now = time.perf_counter()
        timings[name] = round((now - t) * 1000)
        t = now

    tr = get_asr().transcribe(question_path)
    lap("asr")
    if not tr.text:
        raise HTTPException(422, "No speech detected. Hold the button while you speak.")

    a = answer(tr.text, lang or tr.lang)
    lap("answer")

    wav = get_tts().synthesize(a.text, a.lang, turn_dir / "answer.wav", voice=voice)
    lap("tts")

    video = render_avatar(wav, turn_dir / "answer.mp4")
    lap("avatar")

    return {
        "turn_id": turn_id,
        "transcript": tr.text,
        "lang": a.lang,
        "lang_prob": round(tr.lang_prob, 3),
        "answer": a.text,
        "refused": a.refused,
        "reason": a.reason,
        "passages": a.passages,
        "audio_url": f"/media/{turn_id}/answer.wav",
        "video_url": f"/media/{turn_id}/answer.mp4" if video else None,
        "timings_ms": timings,
    }
