import re
from dataclasses import asdict, dataclass, field
from typing import Any


DOMAIN_PATTERNS = {
    "criminal": [
        r"\bassault\b", r"\bhurt\b", r"\bfracture\b", r"\bkill\w*\b", r"\bmurder\b",
        r"\bdeath\b", r"\btheft\b", r"\bsteal\w*\b", r"\brob\w*\b", r"\bcheat\w*\b",
        r"\bextort\w*\b", r"\baccused\b", r"\bvictim\b", r"\boffence\b",
    ],
    "procedural": [
        r"\barrest\b", r"\bbail\b", r"\bfir\b", r"\binvestigation\b", r"\btrial\b",
        r"\bcharge.?sheet\b", r"\bmagistrate\b", r"\bpolice custody\b", r"\bpolice\b", r"\btook him away\b",
    ],
    "evidence": [
        r"\bevidence\b", r"\badmissib\w*\b", r"\bwitness\b", r"\bdocument\b",
        r"\bconfession\b", r"\bstatement\b", r"\belectronic\b", r"\bvideo\b",
    ],
    "constitutional": [
        r"\barticle\b", r"\bfundamental right\b", r"\bwrit\b", r"\bliberty\b",
        r"\bequality\b", r"\bstate action\b", r"\bconstitution\b",
    ],
}

FACT_PATTERNS = {
    "actors": [r"\baccused\b", r"\bvictim\b", r"\bcomplainant\b", r"\bpolice\b", r"\bmagistrate\b", r"\bcourt\b"],
    "conduct": [r"\bhit\b", r"\bbeaten?\b", r"\battack\w*\b", r"\bthreaten\w*\b", r"\bsteal\w*\b", r"\btake\w*\b", r"\bdeceiv\w*\b", r"\barrest\w*\b", r"\bkill\w*\b"],
    "intent_or_knowledge": [r"\bintentional\w*\b", r"\bknow\w*\b", r"\bdishonest\w*\b", r"\bfraudulent\w*\b", r"\bvoluntar\w*\b"],
    "harm": [r"\bfracture\b", r"\binjur\w*\b", r"\bhurt\b", r"\bgrievous\b", r"\bdeath\b", r"\bloss\b"],
    "property": [r"\bproperty\b", r"\bmoney\b", r"\bmovable\b", r"\bstolen\b", r"\btheft\b"],
    "weapon_or_method": [r"\bknife\b", r"\brod\b", r"\bstick\b", r"\bgun\b", r"\bweapon\b", r"\bpoison\b", r"\bfire\b", r"\bdangerous\b"],
    "evidence": [r"\bevidence\b", r"\bwitness\b", r"\bdocument\b", r"\bstatement\b", r"\bconfession\b", r"\bvideo\b", r"\belectronic\b"],
    "procedure_stage": [r"\barrest\b", r"\bbail\b", r"\bfir\b", r"\binvestigation\b", r"\btrial\b", r"\bcharge.?sheet\b"],
    "rights_or_state_action": [r"\bright\b", r"\bliberty\b", r"\bequality\b", r"\bwrit\b", r"\barticle\b", r"\bstate\b"],
}


@dataclass
class StructuredFacts:
    original_text: str
    normalized_text: str
    domains: list[str] = field(default_factory=list)
    facts: dict[str, list[str]] = field(default_factory=dict)
    unclear_facts: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StructuredFactExtractor:
    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @staticmethod
    def _matches(text: str, patterns: list[str]) -> list[str]:
        found = []
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                value = match.group(0).lower()
                if value not in found:
                    found.append(value)
        return found

    def extract(self, text: str) -> StructuredFacts:
        normalized = self._normalize(text)
        domains = [
            domain
            for domain, patterns in DOMAIN_PATTERNS.items()
            if any(re.search(pattern, normalized) for pattern in patterns)
        ]
        facts = {slot: self._matches(normalized, patterns) for slot, patterns in FACT_PATTERNS.items()}
        populated = [slot for slot, values in facts.items() if values]
        unclear = []
        if "criminal" in domains:
            for slot in ("conduct", "intent_or_knowledge", "harm"):
                if not facts.get(slot):
                    unclear.append(slot)
        if "procedural" in domains and not facts.get("procedure_stage"):
            unclear.append("procedure_stage")
        if "evidence" in domains and not facts.get("evidence"):
            unclear.append("evidence")
        if "constitutional" in domains and not facts.get("rights_or_state_action"):
            unclear.append("rights_or_state_action")
        if not domains and any(word in normalized for word in ("section", "law", "case", "legal")):
            domains = ["criminal"]
        confidence = min(0.25 + len(domains) * 0.12 + len(populated) * 0.08, 0.95)
        return StructuredFacts(
            original_text=text,
            normalized_text=normalized,
            domains=domains,
            facts=facts,
            unclear_facts=unclear,
            confidence=round(confidence, 3),
        )
