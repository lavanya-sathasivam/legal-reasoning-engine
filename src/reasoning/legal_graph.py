from dataclasses import asdict, dataclass, field
from typing import Any

from src.config import DEFAULT_TOP_K_RESULTS
from src.reasoning.fact_extractor import StructuredFacts


LAW_DOMAIN_MAP = {
    "BNS": "criminal",
    "BNSS": "procedural",
    "BSA": "evidence",
    "Constitution": "constitutional",
}

SUPPORTING_TYPES = {"definition", "exception", "general", "interpretation", "punishment"}


@dataclass
class ElementTrace:
    element_id: str
    label: str
    satisfied: bool
    fact_slots: list[str]
    matched_facts: list[str] = field(default_factory=list)
    source_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SectionRecommendation:
    law: str
    section: str
    title: str
    source_id: str
    provision_type: str
    confidence: float
    why_it_applies: str
    missing_elements: list[str]
    matched_elements: list[ElementTrace]
    citations: list[dict[str, Any]]
    original_text: str
    applicability_notes: str
    review_status: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["score"] = self.confidence
        data["explanation"] = self.why_it_applies
        return data


class LegalGraphReasoner:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    @staticmethod
    def _domain_matches(record: dict[str, Any], facts: StructuredFacts) -> bool:
        if not facts.domains:
            return True
        law_domain = LAW_DOMAIN_MAP.get(record.get("law", ""))
        record_issues = set(record.get("issue_types", []))
        return bool(law_domain in facts.domains or record_issues.intersection(facts.domains))

    @staticmethod
    def _trace_element(element: dict[str, Any], facts: StructuredFacts) -> ElementTrace:
        fact_slots = element.get("fact_slots", [])
        matched = []
        for slot in fact_slots:
            matched.extend(facts.facts.get(slot, []))
        return ElementTrace(
            element_id=element.get("id", ""),
            label=element.get("label", ""),
            satisfied=bool(matched),
            fact_slots=fact_slots,
            matched_facts=sorted(set(matched)),
            source_text=element.get("source_text", ""),
        )

    @staticmethod
    def _confidence(record: dict[str, Any], traces: list[ElementTrace], facts: StructuredFacts) -> float:
        required = [trace for trace, element in zip(traces, record.get("legal_elements", [])) if element.get("required", True)]
        if required:
            satisfied = sum(1 for trace in required if trace.satisfied)
            base = satisfied / len(required)
        else:
            base = 0.2
        domain_bonus = 0.16 if record.get("law") in LAW_DOMAIN_MAP and LAW_DOMAIN_MAP[record["law"]] in facts.domains else 0.0
        type_penalty = 0.18 if record.get("provision_type") in SUPPORTING_TYPES else 0.0
        general_penalty = 0.12 if record.get("is_general_section") else 0.0
        confidence = max(0.0, min(base * 0.78 + domain_bonus - type_penalty - general_penalty, 0.98))
        return round(confidence, 3)

    @staticmethod
    def _why(record: dict[str, Any], traces: list[ElementTrace], missing: list[str]) -> str:
        matched = [trace for trace in traces if trace.satisfied]
        if matched:
            parts = [
                f"{trace.label} is supported by: {', '.join(trace.matched_facts[:4])}"
                for trace in matched[:3]
            ]
            if missing:
                parts.append(f"Missing facts remain for: {', '.join(missing[:3])}.")
            return " ".join(parts)
        return record.get("applicability_notes") or "This provision may be relevant, but the required legal ingredients need confirmation."

    def analyze(self, facts: StructuredFacts, top_k: int = DEFAULT_TOP_K_RESULTS) -> dict[str, Any]:
        recommendations: list[SectionRecommendation] = []
        excluded_sections = []
        for record in self.records:
            if not self._domain_matches(record, facts):
                continue
            traces = [self._trace_element(element, facts) for element in record.get("legal_elements", [])]
            missing = [trace.label for trace in traces if not trace.satisfied and trace.label]
            confidence = self._confidence(record, traces, facts)
            policy = record.get("confidence_policy", {})
            minimum = policy.get("minimum_required_elements", 1)
            satisfied_count = sum(1 for trace in traces if trace.satisfied)
            if confidence < 0.28 or satisfied_count < minimum:
                excluded_sections.append(
                    {
                        "law": record.get("law"),
                        "section": record.get("section"),
                        "title": record.get("title"),
                        "reason": "Required legal elements were not sufficiently supported by the facts.",
                        "missing_elements": missing,
                    }
                )
                continue
            recommendations.append(
                SectionRecommendation(
                    law=record["law"],
                    section=record["section"],
                    title=record["title"],
                    source_id=record["source_id"],
                    provision_type=record.get("provision_type", "general"),
                    confidence=confidence,
                    why_it_applies=self._why(record, traces, missing),
                    missing_elements=missing,
                    matched_elements=traces,
                    citations=record.get("citations", []),
                    original_text=record.get("original_text", ""),
                    applicability_notes=record.get("applicability_notes", ""),
                    review_status=record.get("review_status", "needs_human_review"),
                )
            )
        recommendations.sort(key=lambda item: (item.provision_type in SUPPORTING_TYPES, -item.confidence, item.law, item.section))
        return {
            "recommendations": [item.to_dict() for item in recommendations[:top_k]],
            "excluded_sections": excluded_sections[:25],
            "missing_facts": sorted(set(facts.unclear_facts)),
        }
