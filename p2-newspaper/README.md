# Project 2: Newspaper digitization on a budget

Turns scanned newspaper pages into one record per article: an ID, the text, and **one bounding box**. It's built for millions of pages without flagship models: open-source OCR, cheap layout rules, and a small local LLM only for the pairs the rules can't settle. Everything runs on CPU.

```
page scan → clean-up → PaddleOCR lines → printed rules → lines into blocks → columns + reading order
          → decider (rules → small LLM → fallback) → union-find → articles → JSON + overlay PNG
                                                                           → eval (IoU F1, grouping F1, CER) → cost model
```

| File | What it does |
|---|---|
| `preprocess.py` | Deskew and contrast clean-up (OpenCV) |
| `ocr.py` | PaddleOCR adapter → `Line(box, text, score)`. `GoldEngine` replays gold lines instead |
| `separators.py` | Finds printed horizontal and vertical rules: the strongest "different article" signal |
| `blocks.py` | Merges lines into blocks by gap, overlap and type size; marks headlines |
| `layout.py` | Columns, reading order (right to left for Arabic), features for each neighbouring pair |
| `decider.py` | "Same article?" decisions, cheapest tier first; counts what each tier decided |
| `grouping.py` | Union-find → articles; box = union of the article's blocks; rejoins hyphenated words |
| `correction.py` | Optional: re-reads low-confidence blocks with a local vision model |
| `pipeline.py` | One page, or a resumable batch with per-page error isolation and timings |
| `evaluate.py` | Box P/R/F1 at IoU 0.5 and 0.75, pairwise grouping F1, character error rate |
| `costmodel.py` | Extrapolates a pilot run to the full archive |
| `synth.py` | Synthetic pages with gold labels, for testing before you have labelled scans |
| `labelstudio.py` | Converts a Label Studio export into gold files |

## Setup

**Windows (PowerShell)**

```powershell
cd capstone\p2-newspaper
Copy-Item .env.example .env
uv sync                      # core + tests; runs everything except real OCR
uv run pytest
uv sync --group ocr          # PaddleOCR + PaddlePaddle (CPU build)
ollama pull qwen2.5:3b       # the decider LLM (optional: DECIDER=rules works without it)
```

**macOS / Linux**

```bash
cd capstone/p2-newspaper
cp .env.example .env
uv sync && uv run pytest
uv sync --group ocr
ollama pull qwen2.5:3b
```

If `uv sync --group ocr` fails for PaddlePaddle, install it with the command from the PaddlePaddle install selector for your OS (CPU), then `uv sync --group ocr` again.

## Workflow

### 1. Check the layout logic on synthetic pages (no OCR needed)

```bash
uv run python -m p2news synth --pages 5
uv run python -m p2news run --input data/synthetic/pages --out runs/synth-gold --gold-lines data/synthetic/gold
uv run python -m p2news eval --pred runs/synth-gold/pages --gold data/synthetic/gold
```

`--gold-lines` skips OCR and feeds the pipeline perfect lines, so every error you see is a layout or grouping error. Open `runs/synth-gold/overlays/*.png` to see the article boxes.

### 2. Add real OCR

```bash
uv run python -m p2news run --input data/synthetic/pages --out runs/synth-ocr
uv run python -m p2news eval --pred runs/synth-ocr/pages --gold data/synthetic/gold
```

The gap between run 1 and run 2 is what OCR costs you. Add `--noisy` to `synth` for scan-like blur and grain.

### 3. Real pages and labels

1. Put page images in `data/real/pages/` (see `data/README.md` for sources).
2. `docker compose up -d` → http://localhost:8080. Create a project with a rectangle label called `Article`, and optionally a per-region text area for the article text (needed for CER).
3. Label about 50 pages: one rectangle per article. Export as JSON.
4. Convert and evaluate:

```bash
uv run python -m p2news labelstudio export.json data/real/gold
uv run python -m p2news run --input data/real/pages --out runs/real-v1
uv run python -m p2news eval --pred runs/real-v1/pages --gold data/real/gold
```

### 4. Cost at archive scale

Run a pilot of a few hundred pages, then:

```bash
uv run python -m p2news cost --run runs/real-v1 --pages 2000000 --price 0.50 --workers 8
```

`--price` is what one machine costs per hour. Add `--llm-cost` and `--vlm-cost` if those calls aren't free (local Ollama is free per call; you pay for the machine time).

## Tuning: symptom → setting

| What you see in the overlays | Change in `.env` |
|---|---|
| One article split into many boxes inside a column | Raise `MERGE_GAP_K` (0.8 → 1.2) |
| Neighbouring articles merged | Lower `MERGE_GAP_K`; check `separators` counts in the page JSON `stats` |
| Small type missing entirely | `DET_LIMIT_SIDE_LEN` too low, or scan resolution too low |
| Boxes don't quite touch, so lines don't merge | Raise `DET_UNCLIP_RATIO` |
| Headlines treated as body text | Lower `HEADLINE_RATIO` |
| Columns mixed up | Adjust `COLUMN_GAP_RATIO` |
| Articles broken at column tops | `DECIDER=cascade` with Ollama running; check `decisions_by_source` |

Every run writes `pairs.jsonl`: every pair decision with its features. Label a few hundred by hand and you can train a tiny classifier (for example LightGBM on the features) to replace the LLM tier at archive scale. This is the weak-supervision idea from the guide.

## Tamil and Arabic

Set `OCR_LANG=ta` or `OCR_LANG=ar`. Arabic also switches reading order to right to left. The boxes are usually still good when the text isn't, so turn on `CORRECTION=vlm` (after `ollama pull qwen2.5vl:7b`) and compare CER with and without it. The trainer said Arabic is optional.

## Open questions for the trainer

- When an article wraps around another item, its single rectangle swallows that item. Is one rectangle what they want, or the list of regions? Both are saved (`bbox` and `regions`).
- Classified ads and stock tables produce many tiny "articles". Keep them, merge them, or drop them?

## Detection model: speed vs text quality

Measured on three synthetic 2400×3300 pages (two clean, one noisy), CPU only, with the English recognizer `en_PP-OCRv5_mobile_rec` in both runs. This was an Apple laptop; your Windows machine will differ:

| `DET_MODEL` | OCR seconds per page | Box F1 @0.5 | Character error rate |
|---|---|---|---|
| `PP-OCRv5_mobile_det` (default) | ~45 | 0.98 | 0.2% |
| `PP-OCRv5_server_det` | ~186 | 0.97 | 1.3% |

On these pages the mobile model is about 4× faster and no less accurate. Synthetic pages are far cleaner than 1950s newsprint, though, so measure both on your labelled real pages before deciding: the heavier model may earn its cost on faded or damaged scans. Put both cost and quality pairs in front of the client using the cost model. `CORRECTION=vlm` is the other lever: mobile detection everywhere, and only low-confidence blocks re-read.

## Versions this was tested with

PaddleOCR 3.4.1 with PaddlePaddle 3.3.1 (CPU). The argument names and result keys used in `src/p2news/ocr.py` work with these versions. Note that once a detection model is named, PaddleOCR ignores `lang`, so `ocr.py` also names the recognition model for each language (`PADDLE_REC_MODEL`). Without that, Tamil and Arabic would silently be read by a generic model. If a later PaddleOCR release errors on start-up, check its docs for renamed arguments. The first run downloads the models (a few hundred MB) into `~/.paddlex`.
