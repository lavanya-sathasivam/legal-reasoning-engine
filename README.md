# Legal Reasoning Platform

A lawyer-focused web platform for mapping case descriptions to applicable legal sections across BNS, BNSS, BSA, and the Constitution of India.

This project has moved beyond the original BNS-only starter. The current V1 is a small-firm legal reasoning workspace with a chatbot-style case analyzer, law-book browsing, matter storage, doubt clearance, settings, and a deterministic legal-graph reasoning engine.

## Core Idea

The platform is intentionally not RAG-first and not keyword-match-first.

Instead, it uses a hybrid legal reasoning flow:

1. Extract structured facts from the lawyer's case description.
2. Classify the issue domain: criminal offence, procedure, evidence, or constitutional.
3. Match those facts against structured legal ingredients in the corpus.
4. Track satisfied elements, missing elements, citations, confidence, and reasoning trace.
5. Use the AI adapter only for explanation and doubt-clearance support, not as the final legal decision-maker.

Embeddings are no longer the core applicability engine.

## Current Law Corpus

The processed corpus is generated into `data/processed/legal_corpus.json`.

Current imported corpus:

- BNS: Bharatiya Nyaya Sanhita
- BNSS: Bharatiya Nagarik Suraksha Sanhita
- BSA: Bharatiya Sakshya Adhiniyam
- Constitution of India

## Features

- Chat-based case analysis.
- Deterministic legal-element matching.
- Applicable section recommendations with citations.
- Missing fact and clarification prompts.
- Reasoning trace for matched legal ingredients.
- Original law text preview.
- Law-book status and section APIs.
- Doubt-clearance endpoint.
- Matter and message persistence using local SQLite.
- Settings API for model/provider and platform preferences.
- Backward-compatible `/chat` and `/analyze` endpoints.

## Project Structure

```text
src/
  ai/                 Provider-agnostic AI adapter layer
  api/                FastAPI app and static web workspace
  preprocessing/      Corpus import and schema-v2 transformation
  reasoning/          Structured fact extraction and legal graph matching
  platform_store.py   SQLite persistence for matters, messages, settings
  pipeline.py         Main orchestration layer

data/
  corpus_manifest.json
  schemas/
  processed/
  indexes/
```

## Setup

Use Python 3.11 for the pinned dependency set:

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you created `.venv` with Python 3.13, recreate it with Python 3.11. `numpy==1.26.4` does not provide wheels for Python 3.13, so pip tries to compile NumPy from source and fails unless a C/C++ build toolchain is installed.

```bash
python -m src.preprocessing.ai_transformer
python -m src.api.app
```

Then open:
http://127.0.0.1:8000

If port `8000` is already in use:

```bash
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8001
```

Or run the app entrypoint on a different port:

```bash
set PORT=8010
python -m src.api.app
```

## API

### Case Chat

```http
POST /api/chat/case
```

```json
{
  "message": "The accused intentionally hit the victim with an iron rod and caused a fracture. Which sections apply?",
  "matter_id": null,
  "selected_laws": []
}
```

Returns:

- assistant message
- applicable sections
- extracted facts
- reasoning trace
- missing facts
- follow-up questions
- excluded sections

### Reasoning Analysis

```http
POST /api/reason/analyze
```

```json
{
  "description": "A witness statement and video recording are being challenged for admissibility."
}
```

### Law Books

```http
GET /api/laws
GET /api/laws/{law}/sections/{section}
```

Example:

```http
GET /api/laws/BNS/sections/118
```

### Doubt Clearance

```http
POST /api/doubts
```

```json
{
  "question": "How should I understand this section?",
  "law": "BNS",
  "section": "118",
  "matter_id": null
}
```

### Matters

```http
GET /api/matters
POST /api/matters
POST /api/matters/{id}/messages
```

### Settings

```http
GET /api/settings
POST /api/settings
```

## Legacy Endpoints

These remain available for compatibility:

```http
POST /chat
POST /analyze
```

## Regenerating The Corpus

Run:

```bash
python -m src.preprocessing.ai_transformer
```

This reads `data/corpus_manifest.json`, imports the configured raw law sources, writes per-law processed files, updates `data/processed/legal_corpus.json`, and rebuilds `data/indexes/section_id_map.json`.

Raw legal source files should be treated as source material and not destructively rewritten.

## Optional OCR Correction Script

The standalone `parser.py` script can use Gemini to clean OCR issues in raw law JSON.

```bash
set GOOGLE_API_KEY=your_key_here
python parser.py --input bns_en.json --output corrected_bns.json
```

Notes:

- `GOOGLE_API_KEY` must be provided through the environment.
- The parser is optional and separate from the legal reasoning engine.

## Tests

```bash
python -m unittest
```

The current suite covers preprocessing, pipeline behavior, and API compatibility.

## Important Notes

- The engine provides legal research assistance, not legal advice.
- AI output must remain grounded in imported legal text and citations.
- Section applicability should be decided by structured legal reasoning, not by keyword overlap or embeddings.
- `data/platform.db` is local runtime data and should not be committed.
