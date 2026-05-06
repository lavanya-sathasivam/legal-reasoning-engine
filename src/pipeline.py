import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.ai.adapter import get_ai_provider
from src.config import COMBINED_CORPUS_PATH, DEFAULT_TOP_K_RESULTS, MIN_LEGAL_CONFIDENCE
from src.preprocessing.ai_transformer import build_corpus
from src.reasoning.fact_extractor import StructuredFactExtractor
from src.reasoning.legal_graph import LegalGraphReasoner


class LegacySimilaritySearchControl:
    def search(self, *args: Any, **kwargs: Any) -> None:
        return None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=2, description="User chat message")
    matter_id: int | None = None
    selected_laws: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    law: str
    section: str
    label: str
    title: str | None = None


class ElementTraceModel(BaseModel):
    element_id: str
    label: str
    satisfied: bool
    fact_slots: list[str]
    matched_facts: list[str] = Field(default_factory=list)
    source_text: str = ""


class LegalSectionResult(BaseModel):
    law: str
    section: str
    title: str
    why_it_applies: str
    explanation: str
    original_text: str | None = None
    score: float
    confidence: float | None = None
    source_id: str | None = None
    provision_type: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    matched_elements: list[ElementTraceModel] = Field(default_factory=list)
    missing_elements: list[str] = Field(default_factory=list)
    applicability_notes: str | None = None
    review_status: str | None = None


class ChatResponse(BaseModel):
    mode: Literal["casual", "legal_answer", "needs_clarification"]
    message: str
    follow_up_questions: list[str] = Field(default_factory=list)
    applicable_sections: list[LegalSectionResult] = Field(default_factory=list)
    extracted_facts: dict[str, Any] = Field(default_factory=dict)
    reasoning_trace: list[dict[str, Any]] = Field(default_factory=list)
    missing_facts: list[str] = Field(default_factory=list)
    excluded_sections: list[dict[str, Any]] = Field(default_factory=list)


class CaseAnalysisRequest(BaseModel):
    description: str = Field(..., min_length=2, description="Natural language case description")
    matter_id: int | None = None
    selected_laws: list[str] = Field(default_factory=list)


class SectionMatch(BaseModel):
    section: str
    title: str
    score: float
    matched_keywords: list[str]
    matched_conditions: list[str]


class SimilarCaseResult(BaseModel):
    section: str
    title: str
    similarity: float
    source_type: str


class CaseAnalysisResponse(BaseModel):
    sections: list[SectionMatch]
    reason: str
    similar_cases: list[SimilarCaseResult]


class ReasoningAnalysisRequest(BaseModel):
    description: str = Field(..., min_length=2)
    matter_id: int | None = None
    selected_laws: list[str] = Field(default_factory=list)


class DoubtRequest(BaseModel):
    question: str = Field(..., min_length=2)
    law: str | None = None
    section: str | None = None
    matter_id: int | None = None


class DoubtResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)


class LegalAnalysisPipeline:
    def __init__(self) -> None:
        self.records = self._load_records()
        self.fact_extractor = StructuredFactExtractor()
        self.reasoner = LegalGraphReasoner(self.records)
        self.ai_provider = get_ai_provider()
        self.similarity_search = LegacySimilaritySearchControl()

    def _load_records(self) -> list[dict[str, Any]]:
        if not COMBINED_CORPUS_PATH.exists():
            records = build_corpus()
        else:
            with COMBINED_CORPUS_PATH.open("r", encoding="utf-8") as handle:
                records = json.load(handle)
            if not records or "schema_version" not in records[0]:
                records = build_corpus()
        return records

    @staticmethod
    def _format_casual_reply(message: str) -> str:
        lowered = message.lower()
        if any(token in lowered for token in ("hello", "hi", "hey")):
            return "Hello. Describe the case facts and I will map them to structured legal ingredients and cited sections."
        if "thank" in lowered:
            return "You're welcome. You can save the matter or ask a doubt about any provision."
        return "I can analyze case facts, surface applicable legal sections, and explain the reasoning with citations."

    @staticmethod
    def _slot_question(slot: str) -> str:
        questions = {
            "conduct": "What exact act or omission is alleged?",
            "intent_or_knowledge": "Was the act intentional, knowing, dishonest, fraudulent, or accidental?",
            "harm": "What injury, loss, fear, or death resulted?",
            "procedure_stage": "Is the issue about arrest, bail, investigation, charge, trial, or another procedure stage?",
            "evidence": "What evidence is involved, such as documents, witness testimony, confession, or electronic records?",
            "rights_or_state_action": "Which right, article, or state action is being challenged?",
        }
        return questions.get(slot, f"Please clarify the missing fact: {slot}.")

    @staticmethod
    def _is_casual(message: str, facts: dict[str, Any]) -> bool:
        lowered = message.lower().strip()
        casual = any(token in lowered for token in ("hello", "hi", "hey", "thanks", "thank you", "what can you do"))
        return casual and not facts.get("domains")

    @staticmethod
    def _to_applicable_section(item: dict[str, Any]) -> dict[str, Any]:
        confidence = float(item.get("confidence", item.get("score", 0.0)))
        return {
            "law": item["law"],
            "section": item["section"],
            "title": item["title"],
            "why_it_applies": item.get("why_it_applies", ""),
            "explanation": item.get("why_it_applies", ""),
            "original_text": item.get("original_text"),
            "score": confidence,
            "confidence": confidence,
            "source_id": item.get("source_id"),
            "provision_type": item.get("provision_type"),
            "citations": item.get("citations", []),
            "matched_elements": item.get("matched_elements", []),
            "missing_elements": item.get("missing_elements", []),
            "applicability_notes": item.get("applicability_notes"),
            "review_status": item.get("review_status"),
        }

    def _format_legal_message(self, recommendations: list[dict[str, Any]], missing_facts: list[str]) -> str:
        if not recommendations:
            if missing_facts:
                questions = " ".join(self._slot_question(slot) for slot in missing_facts[:3])
                return f"I need more facts before recommending sections confidently. {questions}"
            return (
                "Applicable Legal Sections: none confidently identified. "
                "Why it applies: the imported corpus does not contain enough matched legal ingredients for this query. "
                "Explanation: I am not sufficiently confident to recommend specific sections from the available corpus for these facts."
            )
        lines = ["Applicable Legal Sections based on matched legal ingredients:"]
        for item in recommendations:
            citation = item.get("citations", [{}])[0].get("label", f"{item['law']} Section {item['section']}")
            lines.append(f"{citation}: {item['title']} - confidence {item['confidence']:.2f}.")
            lines.append(f"Why it applies: {item['why_it_applies']}")
            lines.append(f"Explanation: {item.get('applicability_notes') or 'The recommendation is based on satisfied structured legal elements and the cited provision text.'}")
            if item.get("missing_elements"):
                lines.append(f"Facts still to confirm: {', '.join(item['missing_elements'][:3])}.")
        return "\n".join(lines)

    def chat(self, message: str, top_k: int = DEFAULT_TOP_K_RESULTS, selected_laws: list[str] | None = None) -> dict[str, Any]:
        facts = self.fact_extractor.extract(message)
        facts_dict = facts.to_dict()
        legacy_override = self.similarity_search.search()
        if legacy_override == []:
            return {
                "mode": "legal_answer",
                "message": "I am not sufficiently confident to recommend specific sections from the available corpus for these facts.",
                "follow_up_questions": [],
                "applicable_sections": [],
                "extracted_facts": facts_dict,
                "reasoning_trace": [],
                "missing_facts": [],
                "excluded_sections": [],
            }
        if self._is_casual(message, facts_dict):
            return {
                "mode": "casual",
                "message": self._format_casual_reply(message),
                "follow_up_questions": [],
                "applicable_sections": [],
                "extracted_facts": facts_dict,
                "reasoning_trace": [],
                "missing_facts": [],
                "excluded_sections": [],
            }

        original_records = self.reasoner.records
        if selected_laws:
            self.reasoner.records = [record for record in original_records if record.get("law") in selected_laws]
        try:
            analysis = self.reasoner.analyze(facts, top_k=min(top_k, DEFAULT_TOP_K_RESULTS))
        finally:
            self.reasoner.records = original_records

        recommendations = analysis["recommendations"]
        applicable_sections = [self._to_applicable_section(item) for item in recommendations]
        follow_ups = [self._slot_question(slot) for slot in analysis["missing_facts"][:4]]
        confident = bool(applicable_sections and applicable_sections[0]["score"] >= MIN_LEGAL_CONFIDENCE)
        mode: Literal["legal_answer", "needs_clarification"] = "legal_answer" if confident else "needs_clarification"
        if recommendations and mode == "needs_clarification":
            mode = "legal_answer"
        if not recommendations and facts.domains and not analysis["missing_facts"]:
            mode = "legal_answer"
        return {
            "mode": mode,
            "message": self._format_legal_message(recommendations, analysis["missing_facts"]),
            "follow_up_questions": follow_ups if mode == "needs_clarification" or analysis["missing_facts"] else [],
            "applicable_sections": applicable_sections,
            "extracted_facts": facts_dict,
            "reasoning_trace": [
                {
                    "source_id": item.get("source_id"),
                    "matched_elements": item.get("matched_elements", []),
                    "missing_elements": item.get("missing_elements", []),
                    "confidence": item.get("confidence"),
                }
                for item in recommendations
            ],
            "missing_facts": analysis["missing_facts"],
            "excluded_sections": analysis["excluded_sections"],
        }

    def analyze_reasoning(self, description: str, selected_laws: list[str] | None = None) -> dict[str, Any]:
        response = self.chat(description, selected_laws=selected_laws)
        return {
            "extracted_facts": response["extracted_facts"],
            "issue_classification": response["extracted_facts"].get("domains", []),
            "applicable_sections": response["applicable_sections"],
            "excluded_sections": response["excluded_sections"],
            "missing_elements": response["missing_facts"],
            "reasoning_trace": response["reasoning_trace"],
        }

    def analyze_case(self, description: str, top_k_sections: int = DEFAULT_TOP_K_RESULTS, top_k_similar: int = DEFAULT_TOP_K_RESULTS) -> dict[str, Any]:
        response = self.chat(description, top_k=top_k_sections)
        applicable_sections = response.get("applicable_sections", [])
        sections = [
            {
                "section": item["section"],
                "title": item["title"],
                "score": item["score"],
                "matched_keywords": [fact for trace in item.get("matched_elements", []) for fact in trace.get("matched_facts", [])],
                "matched_conditions": [trace.get("label", "") for trace in item.get("matched_elements", []) if trace.get("satisfied")],
            }
            for item in applicable_sections[:top_k_sections]
        ]
        similar_cases = [
            {
                "section": item["section"],
                "title": item["title"],
                "similarity": item["score"],
                "source_type": f"{item['law']}_section",
            }
            for item in applicable_sections[:top_k_similar]
        ]
        return {"sections": sections, "reason": response["message"], "similar_cases": similar_cases}

    def list_laws(self) -> list[dict[str, Any]]:
        by_law: dict[str, dict[str, Any]] = {}
        for record in self.records:
            law = record.get("law", "")
            by_law.setdefault(law, {"law": law, "sections": 0, "status": "processed", "provision_types": set()})
            by_law[law]["sections"] += 1
            by_law[law]["provision_types"].add(record.get("provision_type", "general"))
        for law in ("BNSS", "BSA", "Constitution"):
            by_law.setdefault(law, {"law": law, "sections": 0, "status": "missing_source", "provision_types": set()})
        return [
            {**value, "provision_types": sorted(value["provision_types"])}
            for value in sorted(by_law.values(), key=lambda item: item["law"])
        ]

    def get_section(self, law: str, section: str) -> dict[str, Any] | None:
        for record in self.records:
            if record.get("law", "").lower() == law.lower() and record.get("section", "").lower() == section.lower():
                return record
        return None

    def answer_doubt(self, question: str, law: str | None = None, section: str | None = None, matter_id: int | None = None) -> dict[str, Any]:
        context = {"law": law, "section": section, "matter_id": matter_id}
        record = self.get_section(law, section) if law and section else None
        if record:
            context["original_text"] = record.get("original_text", "")
            citations = record.get("citations", [])
        else:
            citations = []
        return {"answer": self.ai_provider.answer_doubt(question, context), "citations": citations, "context": context}
