from typing import Any

from src.legal_engine.rule_engine import RuleEngine
from src.nlp.entity_extractor import EntityExtractor


class SectionMapper:
    def __init__(self, sections: list[dict[str, Any]]) -> None:
        self.sections = sections
        self.entity_extractor = EntityExtractor()
        self.rule_engine = RuleEngine()

    def map_case(self, description: str, top_k: int = 5) -> dict[str, Any]:
        case_signals = self.entity_extractor.analyze(description)
        matches = self.rule_engine.evaluate(case_signals, self.sections)

        return {
            "case_signals": case_signals.to_dict(),
            "section_matches": [match.to_dict() for match in matches[:top_k]],
        }
