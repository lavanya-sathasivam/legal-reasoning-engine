import re
from dataclasses import asdict, dataclass
from typing import Any


ISSUE_TYPE_PATTERNS = {
    "criminal": [
        r"\bassault\b", r"\btheft\b", r"\brobbery\b", r"\bcheating\b", r"\bextortion\b",
        r"\bhurt\b", r"\bfracture\b", r"\bmurder\b", r"\boffence\b", r"\baccused\b",
    ],
    "procedural": [
        r"\barrest\b", r"\bbail\b", r"\bfir\b", r"\bcharge sheet\b", r"\btrial\b",
        r"\bprocedure\b", r"\binvestigation\b", r"\bmagistrate\b", r"\bjurisdiction\b",
    ],
    "evidence": [
        r"\bevidence\b", r"\badmissib\w*\b", r"\bwitness\b", r"\bdocument\b", r"\bproof\b",
        r"\bconfession\b", r"\bstatement\b", r"\btestimony\b",
    ],
    "constitutional": [
        r"\bconstitution\b", r"\barticle\b", r"\bfundamental right\b", r"\bwrit\b",
        r"\bequality\b", r"\bliberty\b", r"\barticle 14\b", r"\barticle 21\b",
    ],
}

LEGAL_TERMS = [
    r"\bsection\b", r"\blaw\b", r"\billegal\b", r"\bcrime\b", r"\bcase\b", r"\bpunishment\b",
    r"\bpolice\b", r"\bcourt\b", r"\blegal\b", r"\bright\b", r"\bconstitution\b", r"\bfir\b",
]

CASUAL_PATTERNS = [
    r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bhow are you\b", r"\bthank you\b", r"\bthanks\b",
    r"\bwhat can you do\b", r"\bwho are you\b",
]

FACT_PATTERNS = {
    "actors": [r"\baccused\b", r"\bcomplainant\b", r"\bvictim\b", r"\bpolice\b", r"\bmagistrate\b"],
    "harm": [r"\bfracture\b", r"\binjury\b", r"\bhurt\b", r"\bdeath\b", r"\bloss\b"],
    "conduct": [r"\bhit\b", r"\bthreaten\w*\b", r"\bsteal\w*\b", r"\bdeceiv\w*\b", r"\barrest\b"],
    "evidence": [r"\bdocument\b", r"\bwitness\b", r"\bvideo\b", r"\bstatement\b", r"\bproof\b"],
}

QUESTION_PATTERNS = [r"\bwhat section\b", r"\bwhich law\b", r"\bcan i\b", r"\bdoes this\b", r"\bapplicable\b"]


@dataclass
class QueryAnalysis:
    mode: str
    normalized_text: str
    issue_types: list[str]
    facts: dict[str, list[str]]
    extracted_terms: list[str]
    confidence: float
    follow_up_questions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EntityExtractor:
    def __init__(self) -> None:
        self.nlp = self._load_spacy_pipeline()

    @staticmethod
    def _load_spacy_pipeline():
        try:
            import spacy

            try:
                return spacy.load("en_core_web_sm")
            except OSError:
                return spacy.blank("en")
        except ImportError:
            return None

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @staticmethod
    def _match_labels(text: str, pattern_map: dict[str, list[str]]) -> list[str]:
        labels = []
        for label, patterns in pattern_map.items():
            if any(re.search(pattern, text) for pattern in patterns):
                labels.append(label)
        return labels

    def _extract_terms(self, text: str) -> list[str]:
        terms = set(re.findall(r"[a-zA-Z][a-zA-Z0-9']+", text))
        if self.nlp is not None:
            doc = self.nlp(text)
            for ent in getattr(doc, "ents", []):
                entity_text = ent.text.strip().lower()
                if entity_text:
                    terms.add(entity_text)
        return sorted(term.lower() for term in terms if len(term) > 2)

    def _extract_facts(self, text: str) -> dict[str, list[str]]:
        facts: dict[str, list[str]] = {}
        for group, patterns in FACT_PATTERNS.items():
            matches = []
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    token = match.group(0).strip().lower()
                    if token not in matches:
                        matches.append(token)
            facts[group] = matches
        return facts

    def analyze_query(self, message: str) -> QueryAnalysis:
        normalized_text = self._normalize_text(message)
        issue_types = self._match_labels(normalized_text, ISSUE_TYPE_PATTERNS)
        facts = self._extract_facts(normalized_text)
        extracted_terms = self._extract_terms(normalized_text)

        legal_signal_count = sum(bool(re.search(pattern, normalized_text)) for pattern in LEGAL_TERMS)
        issue_signal_count = len(issue_types)
        fact_signal_count = sum(1 for values in facts.values() if values)
        is_question = any(re.search(pattern, normalized_text) for pattern in QUESTION_PATTERNS)
        is_casual = any(re.search(pattern, normalized_text) for pattern in CASUAL_PATTERNS)

        follow_up_questions: list[str] = []
        mode = "casual"
        confidence = 0.1

        if issue_signal_count or legal_signal_count >= 2 or is_question:
            mode = "legal_clear"
            confidence = min(0.35 + issue_signal_count * 0.18 + fact_signal_count * 0.12 + legal_signal_count * 0.05, 0.95)

            if fact_signal_count == 0 and not any(issue in issue_types for issue in ("evidence", "constitutional", "procedural")):
                mode = "legal_ambiguous"
                follow_up_questions.append("What happened, and who did what to whom?")
            elif "criminal" in issue_types and not facts.get("conduct"):
                mode = "legal_ambiguous"
                follow_up_questions.append("What specific act is involved, such as assault, theft, cheating, or threat?")
            elif "procedural" in issue_types and not re.search(r"\barrest\b|\bbail\b|\bfir\b|\btrial\b|\binvestigation\b", normalized_text):
                mode = "legal_ambiguous"
                follow_up_questions.append("Is your issue about arrest, bail, investigation, trial, or another court procedure?")
            elif "evidence" in issue_types and not facts.get("evidence"):
                mode = "legal_ambiguous"
                follow_up_questions.append("What kind of evidence is involved, such as documents, witness testimony, or electronic records?")
            elif "constitutional" in issue_types and not re.search(r"\barticle\b|\bright\b|\bliberty\b|\bequality\b|\bwrit\b", normalized_text):
                mode = "legal_ambiguous"
                follow_up_questions.append("Which constitutional right, article, or state action is involved?")

        if is_casual and legal_signal_count == 0 and not issue_types:
            mode = "casual"
            confidence = 0.92
            follow_up_questions = []

        return QueryAnalysis(
            mode=mode,
            normalized_text=normalized_text,
            issue_types=issue_types or (["criminal"] if mode.startswith("legal") else []),
            facts=facts,
            extracted_terms=extracted_terms,
            confidence=round(confidence, 3),
            follow_up_questions=follow_up_questions,
        )
