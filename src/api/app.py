from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.pipeline import (
    CaseAnalysisRequest,
    CaseAnalysisResponse,
    ChatRequest,
    ChatResponse,
    DoubtRequest,
    DoubtResponse,
    LegalAnalysisPipeline,
    ReasoningAnalysisRequest,
)
from src.platform_store import PlatformStore


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="Legal Reasoning Platform API",
    version="1.0.0",
    description="Lawyer-focused legal reasoning platform with legal-graph section recommendations.",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class MatterCreateRequest(BaseModel):
    title: str = Field(..., min_length=1)
    description: str = ""


class MessageCreateRequest(BaseModel):
    role: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    analysis: dict[str, Any] | None = None


class SettingsUpdateRequest(BaseModel):
    settings: dict[str, Any]


@lru_cache(maxsize=1)
def get_pipeline() -> LegalAnalysisPipeline:
    return LegalAnalysisPipeline()


@lru_cache(maxsize=1)
def get_store() -> PlatformStore:
    return PlatformStore()


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    result = get_pipeline().chat(payload.message, selected_laws=payload.selected_laws)
    if payload.matter_id:
        get_store().save_message("user", payload.message, payload.matter_id)
        get_store().save_message("assistant", result["message"], payload.matter_id, result)
        get_store().save_analysis(payload.message, result["extracted_facts"], result["applicable_sections"], payload.matter_id)
    return ChatResponse(**result)


@app.post("/analyze", response_model=CaseAnalysisResponse)
def analyze_case(payload: CaseAnalysisRequest) -> CaseAnalysisResponse:
    result = get_pipeline().analyze_case(payload.description)
    return CaseAnalysisResponse(**result)


@app.post("/api/chat/case", response_model=ChatResponse)
def api_case_chat(payload: ChatRequest) -> ChatResponse:
    return chat(payload)


@app.post("/api/reason/analyze")
def api_reason_analyze(payload: ReasoningAnalysisRequest) -> dict[str, Any]:
    result = get_pipeline().analyze_reasoning(payload.description, selected_laws=payload.selected_laws)
    if payload.matter_id:
        get_store().save_analysis(payload.description, result["extracted_facts"], result["applicable_sections"], payload.matter_id)
    return result


@app.get("/api/laws")
def api_laws() -> list[dict[str, Any]]:
    return get_pipeline().list_laws()


@app.get("/api/laws/{law}/sections/{section}")
def api_law_section(law: str, section: str) -> dict[str, Any]:
    record = get_pipeline().get_section(law, section)
    if record is None:
        raise HTTPException(status_code=404, detail="Section not found in the imported corpus.")
    return record


@app.post("/api/doubts", response_model=DoubtResponse)
def api_doubts(payload: DoubtRequest) -> DoubtResponse:
    result = get_pipeline().answer_doubt(payload.question, payload.law, payload.section, payload.matter_id)
    return DoubtResponse(**result)


@app.get("/api/matters")
def api_list_matters() -> list[dict[str, Any]]:
    return get_store().list_matters()


@app.post("/api/matters")
def api_create_matter(payload: MatterCreateRequest) -> dict[str, Any]:
    return get_store().create_matter(payload.title, payload.description)


@app.post("/api/matters/{matter_id}/messages")
def api_create_message(matter_id: int, payload: MessageCreateRequest) -> dict[str, Any]:
    return get_store().save_message(payload.role, payload.content, matter_id, payload.analysis)


@app.get("/api/settings")
def api_get_settings() -> dict[str, Any]:
    return get_store().get_settings()


@app.post("/api/settings")
def api_update_settings(payload: SettingsUpdateRequest) -> dict[str, Any]:
    return get_store().update_settings(payload.settings)


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("src.api.app:app", host=host, port=port, reload=False)
