# BNS Legal Intelligence Starter System

This project is a modular legal intelligence starter system for analyzing case descriptions against the Bharatiya Nyaya Sanhita (BNS) 2023.

## Features

- Transforms raw BNS JSON into an AI-ready knowledge base
- Extracts entities, actions, intent, and harm indicators from case text
- Applies explicit, explainable rule-based reasoning to map case facts to BNS sections
- Retrieves semantically similar BNS sections using embeddings
- Exposes a `POST /analyze` API with FastAPI

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m src.preprocessing.ai_transformer
python -m src.api.app
```

## Optional OCR Correction Script

The standalone `parser.py` script uses Gemini to clean OCR issues in the raw BNS JSON.

```bash
set GOOGLE_API_KEY=your_key_here
python parser.py --input bns_en.json --output corrected_bns.json
```

Notes:

- `GOOGLE_API_KEY` must be provided through the environment. It is not stored in source code.
- Importing `parser.py` is now side-effect free; the correction pipeline only runs via the CLI entrypoint.

## API

### `POST /analyze`

```json
{
  "description": "The accused intentionally hit the victim with an iron rod and caused a fracture."
}
```

## Notes

- `similar_cases` returns semantically similar BNS sections in v1, not judicial precedents.
- Rule-based scoring drives legal applicability. Embeddings provide contextual retrieval support only, and low-confidence semantic matches are filtered out.
- The system includes lightweight fallbacks when spaCy models or sentence-transformers are not installed, so the starter pipeline remains runnable.
