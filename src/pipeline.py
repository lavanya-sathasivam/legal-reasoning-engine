import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.config import COMBINED_CORPUS_PATH, DEFAULT_TOP_K_RESULTS, MIN_LEGAL_CONFIDENCE
from src.nlp.entity_extractor import EntityExtractor
from src.preprocessing.ai_transformer import build_corpus
from src.retrieval.embedding_model import EmbeddingModel
from src.retrieval.similarity_search import SimilaritySearch


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=2, description="User chat message")


class LegalSectionResult(BaseModel):
    law: str
    section: str
    title: str
    why_it_applies: str
    explanation: str
    original_text: str | None = None
    score: float


class ChatResponse(BaseModel):
    mode: Literal["casual", "legal_answer", "needs_clarification"]
    message: str
    follow_up_questions: list[str] = Field(default_factory=list)
    applicable_sections: list[LegalSectionResult] = Field(default_factory=list)


class CaseAnalysisRequest(BaseModel):
    description: str = Field(..., min_length=2, description="Natural language case description")


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


COMMON_QUERY_TERMS = {"which", "what", "apply", "applies", "section", "sections", "law", "legal", "case", "accused", "victim", "person", "issue", "involves", "under", "this", "that", "with", "from", "into", "being", "challenged", "need", "know", "may"}
TERM_EXPANSIONS = {
    "fracture": {"grievous", "hurt", "injury"},
    "injury": {"hurt"},
    "hurt": {"injury"},
    "rod": {"weapon", "dangerous", "hurt"},
    "knife": {"weapon", "dangerous"},
    "gun": {"weapon", "dangerous"},
    "poison": {"dangerous", "weapon"},
    "witness": {"evidence", "testimony"},
    "statement": {"evidence", "declaration"},
    "admissibility": {"evidence", "admissible"},
    "document": {"evidence"},
    "video": {"electronic", "evidence"},
    "arrest": {"procedure", "procedural"},
    "bail": {"procedure", "procedural"},
    "investigation": {"procedure", "procedural"},
    "theft": {"property", "dishonest"},
    "cheating": {"deception", "fraud"},
    "robbery": {"theft", "violence"},
    "rights": {"constitutional", "article"},
    "liberty": {"constitutional", "article"},
}


class LegalAnalysisPipeline:
    def __init__(self) -> None:
        self.records = self._load_records()
        self.entity_extractor = EntityExtractor()
        self.embedding_model = EmbeddingModel()
        self.similarity_search = SimilaritySearch(self.records, self.embedding_model)

    def _load_records(self) -> list[dict[str, Any]]:
        if not COMBINED_CORPUS_PATH.exists():
            records = build_corpus()
        else:
            with COMBINED_CORPUS_PATH.open("r", encoding="utf-8") as handle:
                records = json.load(handle)
            if not records:
                records = build_corpus()
        return records

    @staticmethod
    def _format_casual_reply(message: str) -> str:
        lowered = message.lower()
        if any(token in lowered for token in ("hello", "hi", "hey")):
            return "Hello. I can help with legal section lookups or general conversation. Share your facts when you want a legal analysis."
        if "thank" in lowered:
            return "You're welcome. If you'd like, I can also help map a fact pattern to the most relevant legal sections."
        return "I can chat normally or help analyze legal fact patterns across multiple laws. Share the facts when you want a legal section recommendation."

    @staticmethod
    def _compact_law_label(law_code: str) -> str:
        labels = {
            "BNS": "Bharatiya Nyaya Sanhita",
            "BNSS": "Bharatiya Nagarik Suraksha Sanhita",
            "BSA": "Bharatiya Sakshya Adhiniyam",
            "Constitution": "Constitution of India",
        }
        return labels.get(law_code, law_code)

    @staticmethod
    def _build_query_text(message: str, analysis: dict[str, Any]) -> str:
        fact_terms = []
        for values in analysis["facts"].values():
            fact_terms.extend(values)
        issue_terms = analysis.get("issue_types", [])
        return " ".join([message, *fact_terms, *issue_terms]).strip()

    @staticmethod
    def _build_query_terms(analysis: dict[str, Any]) -> set[str]:
        query_terms = set()
        for values in analysis.get("facts", {}).values():
            for value in values:
                cleaned = value.lower().strip()
                if cleaned and cleaned not in COMMON_QUERY_TERMS:
                    query_terms.add(cleaned)
        for issue_type in analysis.get("issue_types", []):
            query_terms.add(issue_type.lower())
        for term in analysis.get("extracted_terms", []):
            cleaned = term.lower().strip()
            if len(cleaned) >= 4 and cleaned not in COMMON_QUERY_TERMS:
                query_terms.add(cleaned)
                query_terms.update(TERM_EXPANSIONS.get(cleaned, set()))
        expanded = set()
        for term in list(query_terms):
            expanded.update(TERM_EXPANSIONS.get(term, set()))
        query_terms.update(expanded)
        return query_terms

    def _build_why_it_applies(self, record: dict[str, Any], analysis: dict[str, Any]) -> str:
        query_terms = set(analysis.get("extracted_terms", []))
        matched_tags = [tag for tag in record.get("tags", []) if tag.lower() in query_terms][:3]
        matched_facts = []
        for values in analysis.get("facts", {}).values():
            for value in values:
                if value in record["retrieval_text"].lower() and value not in matched_facts:
                    matched_facts.append(value)
        reasons = []
        if matched_facts:
            reasons.append(f"The query facts line up with this provision through {', '.join(matched_facts[:3])}.")
        if matched_tags:
            reasons.append(f"Its indexed legal themes include {', '.join(matched_tags)}.")
        if analysis.get("issue_types"):
            reasons.append(f"It aligns with the detected issue type: {', '.join(analysis['issue_types'])}.")
        if not reasons:
            reasons.append("This provision was retrieved because its indexed summary and text closely match the facts provided.")
        return " ".join(reasons)

    @staticmethod
    def _build_explanation(record: dict[str, Any]) -> str:
        key_points = record.get("key_points", [])[:2]
        if key_points:
            return " ".join(key_points)
        return record.get("summary", "No summary available.")

    def _format_legal_message(self, sections: list[dict[str, Any]], uncertainty_note: str | None = None) -> str:
        lines = []
        if uncertainty_note:
            lines.append(uncertainty_note)
            lines.append("")
        lines.append("????? Applicable Legal Sections:")
        lines.append("")
        for entry in sections:
            lines.append(f"[{self._compact_law_label(entry['law'])}] - Section {entry['section']} - {entry['title']}")
            lines.append("")
            lines.append("? Why it applies:")
            lines.append(entry["why_it_applies"])
            lines.append("")
            lines.append("?? Explanation:")
            lines.append(entry["explanation"])
            if entry.get("original_text"):
                lines.append("")
                lines.append("?? Original Law:")
                lines.append(entry["original_text"])
            lines.append("")
            lines.append("---")
            lines.append("")
        return "\n".join(lines).strip()

    def chat(self, message: str, top_k: int = DEFAULT_TOP_K_RESULTS) -> dict[str, Any]:
        analysis = self.entity_extractor.analyze_query(message).to_dict()

        if analysis["mode"] == "casual":
            return {
                "mode": "casual",
                "message": self._format_casual_reply(message),
                "follow_up_questions": [],
                "applicable_sections": [],
            }

        if analysis["mode"] == "legal_ambiguous":
            return {
                "mode": "needs_clarification",
                "message": "I need a few more facts before I can recommend legal sections with confidence.",
                "follow_up_questions": analysis["follow_up_questions"],
                "applicable_sections": [],
            }

        query_text = self._build_query_text(message, analysis)
        results = self.similarity_search.search(
            query_text=query_text,
            issue_types=analysis["issue_types"],
            query_terms=self._build_query_terms(analysis),
            top_k=min(top_k, DEFAULT_TOP_K_RESULTS),
        )

        if not results or results[0]["score"] < MIN_LEGAL_CONFIDENCE:
            return {
                "mode": "legal_answer",
                "message": "I am not sufficiently confident to recommend specific legal sections from the available corpus for these facts.",
                "follow_up_questions": [],
                "applicable_sections": [],
            }

        applicable_sections = []
        for record in results[: min(top_k, DEFAULT_TOP_K_RESULTS)]:
            applicable_sections.append(
                {
                    "law": record["law"],
                    "section": record["section"],
                    "title": record["title"],
                    "why_it_applies": self._build_why_it_applies(record, analysis),
                    "explanation": self._build_explanation(record),
                    "original_text": record["original_text"],
                    "score": record["score"],
                }
            )

        uncertainty_note = None
        if results[0]["score"] < 0.56:
            uncertainty_note = "I found potentially relevant sections, but the factual match is still somewhat uncertain based on the current query."

        return {
            "mode": "legal_answer",
            "message": self._format_legal_message(applicable_sections, uncertainty_note=uncertainty_note),
            "follow_up_questions": [],
            "applicable_sections": applicable_sections,
        }

    def analyze_case(self, description: str, top_k_sections: int = DEFAULT_TOP_K_RESULTS, top_k_similar: int = DEFAULT_TOP_K_RESULTS) -> dict[str, Any]:
        response = self.chat(description, top_k=top_k_sections)
        applicable_sections = response.get("applicable_sections", [])

        sections = [
            {
                "section": item["section"],
                "title": item["title"],
                "score": item["score"],
                "matched_keywords": [],
                "matched_conditions": [],
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

        reason = response["message"]
        if response["mode"] == "needs_clarification":
            reason = f"{response['message']} {' '.join(response['follow_up_questions'])}".strip()

        return {
            "sections": sections,
            "reason": reason,
            "similar_cases": similar_cases,
        }
