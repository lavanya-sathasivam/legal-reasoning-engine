from functools import lru_cache
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.pipeline import (
    CaseAnalysisRequest,
    CaseAnalysisResponse,
    ChatRequest,
    ChatResponse,
    LegalAnalysisPipeline,
)


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="Legal RAG Assistant API",
    version="0.2.0",
    description="Chat-style legal assistant for grounded section recommendations across multiple laws.",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@lru_cache(maxsize=1)
def get_pipeline() -> LegalAnalysisPipeline:
    return LegalAnalysisPipeline()


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    result = get_pipeline().chat(payload.message)
    return ChatResponse(**result)


@app.post("/analyze", response_model=CaseAnalysisResponse)
def analyze_case(payload: CaseAnalysisRequest) -> CaseAnalysisResponse:
    result = get_pipeline().analyze_case(payload.description)
    return CaseAnalysisResponse(**result)


if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=False)
