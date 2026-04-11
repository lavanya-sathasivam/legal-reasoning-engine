from dataclasses import dataclass, field
from typing import Any

from src.config import MIN_RULE_MATCH_SCORE
from src.nlp.entity_extractor import CaseSignals


OVERRIDE_RULES = [
    {
        "name": "fracture_implies_grievous_hurt",
        "triggers": ["fracture"],
        "target_section": "116",
        "score": 4.0,
        "reason": "Fracture or dislocation is expressly enumerated under grievous hurt.",
        "matched_condition": "fracture or dislocation",
    },
    {
        "name": "weapon_and_hurt",
        "triggers": ["weapon_use", "hurt"],
        "target_section": "118",
        "score": 3.5,
        "reason": "Voluntary hurt using dangerous weapons aligns with Section 118.",
        "matched_condition": "use of dangerous weapon",
    },
    {
        "name": "violent_theft_implies_robbery",
        "triggers": ["steal", "weapon_use"],
        "target_section": "309",
        "score": 9.0,
        "reason": "Theft accompanied by threat or dangerous means points to robbery.",
        "matched_condition": "fear-induced delivery of property",
    },
    {
        "name": "wrongful_confinement_primary_section",
        "triggers": ["wrongful_confinement"],
        "target_section": "127",
        "score": 3.0,
        "reason": "Confinement facts should surface the primary wrongful confinement section.",
        "matched_condition": "wrongful restraint or confinement",
    },
    {
        "name": "death_implies_homicide",
        "triggers": ["kill", "death"],
        "target_section": "100",
        "score": 3.0,
        "reason": "Intentional fatal conduct should surface culpable homicide as a primary section.",
        "matched_condition": "death caused",
    },
]

SYNONYM_MATCHES = {
    "injure": ["hurt", "grievous"],
    "steal": ["theft"],
    "rob": ["robbery"],
    "extort": ["extortion"],
    "cheat": ["cheating"],
    "confine": ["wrongful_confinement"],
    "kill": ["death"],
}

SEVERITY_SCORES = {"low": 0.2, "medium": 0.6, "high": 1.1}
GENERIC_SECTION_PENALTY = 2.5
DIRECT_EVIDENCE_KEYWORDS = {"theft", "robbery", "extortion", "cheating", "hurt", "grievous", "wrongful_confinement", "death"}
DIRECT_EVIDENCE_CONDITIONS = {
    "fracture or dislocation",
    "grievous hurt",
    "death caused",
    "dishonest taking of property",
    "fear-induced delivery of property",
    "use of dangerous weapon",
    "wrongful restraint or confinement",
    "deception causing delivery",
}
CONDITION_SIGNAL_MAP = {
    "fracture or dislocation": {"fracture"},
    "grievous hurt": {"grievous", "grievous_hurt", "injure", "hurt"},
    "death caused": {"death", "kill"},
    "dishonest taking of property": {"theft", "steal", "robbery", "rob"},
    "fear-induced delivery of property": {"extortion", "extort", "violent", "robbery", "rob"},
    "use of dangerous weapon": {"weapon_use", "violent"},
    "wrongful restraint or confinement": {"wrongful_confinement", "confine"},
    "deception causing delivery": {"cheating", "cheat"},
}
CONTEXTUAL_SECTION_PATTERNS = {"is_general_section", "is_contextual_section"}
TITLE_SIGNAL_PATTERNS = {
    "robbery": ["robbery", "extortion", "theft"],
    "extortion": ["extortion"],
    "cheating": ["cheating"],
    "wrongful confinement": ["wrongful_confinement", "confine"],
    "culpable homicide": ["death", "kill"],
    "murder": ["death", "kill"],
    "grievous hurt": ["grievous", "fracture", "injure", "hurt"],
    "hurt": ["hurt", "injure"],
}


@dataclass
class RuleMatch:
    section: str
    title: str
    score: float = 0.0
    matched_keywords: list[str] = field(default_factory=list)
    matched_conditions: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    chapter: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "section": self.section,
            "title": self.title,
            "score": round(self.score, 3),
            "matched_keywords": sorted(set(self.matched_keywords)),
            "matched_conditions": sorted(set(self.matched_conditions)),
            "evidence": self.evidence,
            "chapter": self.chapter,
        }


class RuleEngine:
    def evaluate(self, case_signals: CaseSignals, sections: list[dict[str, Any]]) -> list[RuleMatch]:
        matches: dict[str, RuleMatch] = {}

        for section in sections:
            match = self._score_section(case_signals, section)
            if match.score >= MIN_RULE_MATCH_SCORE:
                matches[match.section] = match

        self._apply_override_rules(case_signals, sections, matches)
        filtered = [match for match in matches.values() if match.score >= MIN_RULE_MATCH_SCORE]
        return sorted(filtered, key=lambda item: item.score, reverse=True)

    def _score_section(self, case_signals: CaseSignals, section: dict[str, Any]) -> RuleMatch:
        match = RuleMatch(
            section=section["section"],
            title=section["title"],
            chapter=section.get("chapter", {}),
        )

        section_keywords = set(section.get("keywords", []))
        section_conditions = set(section.get("conditions", []))
        lowered_title = section["title"].lower()
        available_signals = set(
            case_signals.harm_indicators
            + case_signals.keywords
            + case_signals.actions
            + case_signals.intent_indicators
        )

        for keyword in case_signals.keywords:
            if keyword in section_keywords:
                match.score += 1.6
                match.matched_keywords.append(keyword)
                match.evidence.append(f"Keyword '{keyword}' matched section metadata.")

        for action in case_signals.actions:
            for mapped_keyword in SYNONYM_MATCHES.get(action, []):
                if mapped_keyword in section_keywords:
                    match.score += 1.3
                    match.matched_keywords.append(mapped_keyword)
                    match.evidence.append(f"Action '{action}' aligned with legal keyword '{mapped_keyword}'.")

        for condition, triggers in CONDITION_SIGNAL_MAP.items():
            if condition not in section_conditions or not triggers.intersection(available_signals):
                continue

            match.matched_conditions.append(condition)
            if condition == "fracture or dislocation":
                match.score += 2.4
                match.evidence.append("Case facts mention fracture/dislocation.")
            elif condition == "grievous hurt":
                match.score += 2.0
                match.evidence.append("Case facts indicate serious injury.")
            elif condition == "death caused":
                match.score += 2.8
                match.evidence.append("Case facts indicate death or fatal outcome.")
            elif condition == "use of dangerous weapon":
                match.score += 2.0
                match.evidence.append("Case facts mention use of a dangerous weapon.")
            else:
                match.score += 1.8
                match.evidence.append(f"Case facts aligned with condition '{condition}'.")

        if case_signals.intent_indicators and section.get("intent_required"):
            match.score += 1.0
            match.evidence.append("Intent indicators matched a section that requires intent.")

        for title_phrase, triggers in TITLE_SIGNAL_PATTERNS.items():
            if lowered_title.startswith(title_phrase) and set(triggers).intersection(available_signals):
                match.score += 1.6
                match.evidence.append(f"Section title aligned directly with '{title_phrase}'.")

        severity_bonus = SEVERITY_SCORES.get(section.get("severity", "low"), 0.0)
        if case_signals.harm_indicators and severity_bonus:
            match.score += severity_bonus

        if any(section.get(flag) for flag in CONTEXTUAL_SECTION_PATTERNS):
            if not self._has_direct_offence_evidence(match):
                match.evidence.append("Generic section filtered due to lack of offence-specific evidence.")
                match.score = 0.0
                return match
            match.score = max(match.score - GENERIC_SECTION_PENALTY, 0.0)
            match.evidence.append("Contextual section penalty applied.")

        return match

    @staticmethod
    def _has_direct_offence_evidence(match: RuleMatch) -> bool:
        return bool(
            DIRECT_EVIDENCE_KEYWORDS.intersection(match.matched_keywords)
            or DIRECT_EVIDENCE_CONDITIONS.intersection(match.matched_conditions)
        )

    def _apply_override_rules(
        self,
        case_signals: CaseSignals,
        sections: list[dict[str, Any]],
        matches: dict[str, RuleMatch],
    ) -> None:
        section_lookup = {section["section"]: section for section in sections}
        available_signals = set(case_signals.harm_indicators + case_signals.keywords + case_signals.actions)

        for rule in OVERRIDE_RULES:
            if not set(rule["triggers"]).issubset(available_signals):
                continue

            section = section_lookup.get(rule["target_section"])
            if section is None:
                continue

            existing = matches.get(section["section"])
            if existing is None:
                existing = RuleMatch(
                    section=section["section"],
                    title=section["title"],
                    chapter=section.get("chapter", {}),
                )
                matches[existing.section] = existing

            existing.score += rule["score"]
            existing.matched_conditions.append(rule["matched_condition"])
            existing.evidence.append(rule["reason"])
