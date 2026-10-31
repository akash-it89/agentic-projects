"""Web API + viewer.

    uv run python -m p3docintel serve        # then open http://localhost:8000
"""

import logging
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import pipeline, store
from .chat import answer
from .config import get_settings
from .ingest import supported
from .llm import get_llm
from .schema import ChatAnswer, DocResult

log = logging.getLogger(__name__)
WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_UPLOAD_MB = 25

app = FastAPI(title="Document intelligence")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class ChatRequest(BaseModel):
    question: str


def _load(doc_id: str) -> DocResult:
    try:
        return store.load(doc_id)
    except KeyError:
        raise HTTPException(404, "Document not found") from None


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health():
    s = get_settings()
    return {"status": "ok", "llm_provider": s.llm_provider, "ocr_engine": s.ocr_engine, "retriever": s.retriever}


@app.post("/api/documents", status_code=202)
def upload(background: BackgroundTasks, file: UploadFile = File(...), language: str = Form("auto")):
    if not file.filename or not supported(file.filename):
        raise HTTPException(415, "Upload a PDF or an image (PNG, JPG, TIFF).")
    data = file.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB.")
    result = pipeline.create(file.filename, data, language)
    background.add_task(pipeline.process, result.doc_id)  # runs after the response is sent
    return {"doc_id": result.doc_id, "status": result.status}


@app.get("/api/documents")
def list_documents():
    return [{"doc_id": d.doc_id, "filename": d.filename, "status": d.status, "doc_type": d.doc_type}
            for d in store.list_docs()]


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str) -> DocResult:
    return _load(doc_id)


@app.get("/api/documents/{doc_id}/pages/{number}")
def page_image(doc_id: str, number: int):
    _load(doc_id)
    path = store.pages_dir(doc_id) / f"page-{number}.png"
    if not path.exists():
        raise HTTPException(404, "Page not found")
    return FileResponse(path, media_type="image/png")


@app.post("/api/documents/{doc_id}/chat")
def chat(doc_id: str, req: ChatRequest) -> ChatAnswer:
    doc = _load(doc_id)
    if doc.status != "done":
        raise HTTPException(409, "The document is still being processed.")
    try:
        return answer(req.question, doc.lines, get_llm())
    except Exception as e:  # model server down or model not pulled: say so instead of a bare 500
        log.exception("Chat failed for %s", doc_id)
        s = get_settings()
        model = s.ollama_model if s.llm_provider == "ollama" else s.bedrock_model_id
        hint = (f"Is Ollama running, and is '{model}' pulled? (ollama pull {model})"
                if s.llm_provider == "ollama" else f"Check your AWS credentials and access to '{model}'.")
        raise HTTPException(503, f"The language model didn't respond ({type(e).__name__}). {hint}") from None
