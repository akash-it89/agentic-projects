# Project 3: Document intelligence

Upload any document (a licence, a bill, a lease, a pamphlet) in English, Tamil or Arabic. The app extracts every piece of information **with the exact region it came from**, says what kind of document it is, summarises it, and answers questions about it with citations you can click. Everything runs on CPU with a local LLM; AWS (Bedrock, Textract) is optional.

```
upload → page PNGs ─┬─ born-digital PDF: text layer (exact, free) ──────────┐
                    └─ scan / photo: PaddleOCR (or Textract for English) ───┤
                                       [optional: vision model re-reads low-confidence lines]
                                                                             ▼
                      lines with IDs + boxes  [p1-L7] "Due date: 25/10/2026"  (0.08, 0.27, 0.45, 0.29)
                                                                             │
             ┌────────────────────┬────────────────────┬────────────────────┤
             ▼                    ▼                    ▼                    ▼
   fields (LLM cites IDs,   type (small LLM,    summary (LLM)      chat (BM25 or embeddings
   code computes boxes)     fixed list)                            over the doc; answers cite IDs)
             └────────────────────┴──────────── viewer: click a field or a citation → highlight on the page
```

## The one idea to understand

**The LLM never produces coordinates.** OCR gives every line an ID and a box. The LLM is asked to return fields that cite line IDs, and `extract.py` builds each field's box from the cited lines. That has three consequences:

- Every highlight points at real text on the page.
- A field citing a line that doesn't exist (a hallucination) is dropped.
- The code checks that the value really appears in the cited lines. If it doesn't (the model reformatted a date, say), the field is kept but marked "check this" with confidence 0.5.

Chat works the same way: citations must come from the retrieved excerpts, otherwise the answer is "That isn't in this document."

## Setup

**Windows (PowerShell)**

```powershell
cd capstone\p3-docintel
Copy-Item .env.example .env
uv sync                      # app + born-digital PDFs + tests
uv run pytest                # 36 tests; no Ollama, OCR or AWS needed
uv sync --group ocr          # PaddleOCR, for scans and photos
ollama pull qwen2.5:7b       # extraction, summary, chat
ollama pull qwen2.5:3b       # classification (small and cheap)
uv run python -m p3docintel samples
uv run python -m p3docintel serve
```

**macOS / Linux**

```bash
cd capstone/p3-docintel
cp .env.example .env
uv sync && uv run pytest
uv sync --group ocr
ollama pull qwen2.5:7b && ollama pull qwen2.5:3b
uv run python -m p3docintel samples
uv run python -m p3docintel serve
```

Open http://localhost:8000 and upload a file from `samples/docs/`. With uv you can combine groups, for example `uv sync --group ocr --group aws`.

## Commands

| Command | What it does |
|---|---|
| `python -m p3docintel serve` | Web app on http://localhost:8000 |
| `python -m p3docintel samples` | Writes 4 invented SPECIMEN documents plus gold labels to `samples/` |
| `python -m p3docintel process FILE [--lang ta]` | Processes one file and prints the fields, type and summary |
| `python -m p3docintel eval` | Scores every gold-labelled sample: field recall, box accuracy, type, chat |

Each document is stored in `runs/docs/<doc_id>/`: the original file, the page PNGs, and `result.json`.

## Choosing engines and models

| Setting | Options | Notes |
|---|---|---|
| `OCR_ENGINE` | `paddle` (default), `textract` | Born-digital PDFs never use OCR. Textract reads Latin scripts only, so Tamil and Arabic always use PaddleOCR. Textract responses are cached by file hash, so a page is never paid for twice |
| `DET_MODEL` | `PP-OCRv5_mobile_det`, `PP-OCRv5_server_det` | Mobile is much faster on CPU. Compare on your own documents |
| `LLM_PROVIDER` | `ollama`, `bedrock` | Bedrock needs `uv sync --group aws`, `aws configure`, and `BEDROCK_MODEL_ID` |
| `CLASSIFY_MODEL` | any Ollama model | Small is fine for picking one label from a list |
| `RETRIEVER` | `bm25`, `ollama` | BM25 needs no model and works with any script. `ollama` uses embeddings (`ollama pull bge-m3`) and handles paraphrased questions better |
| `CORRECTION` | `none`, `vlm` | Re-reads lines below `KEEP_SCORE` with a local vision model (`ollama pull qwen2.5vl:7b`) |

## Tamil and Arabic

- For **scans and photos**, choose the language when uploading. Auto-detect only works for PDFs that contain text; for images it assumes English. Getting this wrong means the wrong OCR model reads the page.
- PaddleOCR's boxes are usually good for both scripts, but the text is weaker than for English. Try `CORRECTION=vlm` and compare with `eval`.
- Arabic is shown right to left automatically (`dir="auto"`). The summary includes a version in the document's language.
- No Tamil or Arabic samples are generated, because Pillow can't shape these scripts without extra system libraries. Use real documents: public government forms and pamphlets are a good source. Label 5–10 in the same gold format (see `samples/gold/*.json` after running `samples`) and put them in a folder with `docs/` and `gold/` subfolders, then run `eval --samples that-folder`.

## Evaluation

`eval` processes each gold-labelled document and reports:

- **Field recall**: the share of gold fields whose value was extracted. Field names may differ, so matching is by value.
- **Box accuracy**: of the found fields, the share whose box overlaps the gold box with IoU ≥ 0.5. Field boxes cover the whole line, for example "Due date: 25/10/2026", while gold boxes cover only the value, so expect IoU well below 1 even when the box is right.
- **Classification accuracy**, and **chat accuracy**, which includes correctly refusing questions the document can't answer.

Precision isn't reported, because the gold files list key fields only and a good extractor finds more. To measure precision, label every field on a few documents.

## Code map

| File | Responsibility |
|---|---|
| `ingest.py`, `text_layer.py` | Upload → page PNGs; exact text and boxes from PDF text layers |
| `ocr.py` | PaddleOCR and Textract → `Line(id, page, text, bbox, confidence)` |
| `correction.py` | Optional vision-model re-read of low-confidence lines |
| `extract.py` | Fields, with grounding checks |
| `classify.py` | Document type and summary |
| `chat.py` | Chunks, BM25 or embedding retrieval, cited answers |
| `pipeline.py`, `store.py` | Runs the stages, saves progress for the UI, and reports errors instead of hanging |
| `api.py`, `web/` | FastAPI and a plain HTML/JS viewer with no build step |
| `samples.py`, `evaluate.py` | Sample documents with gold labels; scoring |

## Versions this was tested with

PaddleOCR 3.4.1 with PaddlePaddle 3.3.1 (CPU), and PyMuPDF 2.0. Two details matter in `ocr.py`. First, once a detection model is named, PaddleOCR ignores `lang`, so the recognition model per language is named explicitly. Second, PyMuPDF 2.0 invalidates earlier `Page` objects when a page is added (handled in `samples.py`). If a newer version errors on start-up, check those two places first.
