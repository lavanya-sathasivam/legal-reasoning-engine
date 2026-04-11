from typing import Any


class ExplanationGenerator:
    def generate(
        self,
        description: str,
        case_signals: dict[str, Any],
        section_matches: list[dict[str, Any]],
        similar_sections: list[dict[str, Any]],
    ) -> str:
        if not section_matches:
            if similar_sections:
                top = similar_sections[0]
                return (
                    "The rule engine was inconclusive for the provided facts. "
                    f"The closest semantic support came from BNS Section {top['section']} "
                    f"('{top['title']}') with similarity {top['similarity']}."
                )
            return "The rule engine was inconclusive and no high-confidence semantic support was found."

        top_match = section_matches[0]
        actions = ", ".join(case_signals.get("actions", [])) or "no clear action markers"
        intents = ", ".join(case_signals.get("intent_indicators", [])) or "no explicit intent markers"
        harms = ", ".join(case_signals.get("harm_indicators", [])) or "no explicit harm markers"
        conditions = ", ".join(top_match.get("matched_conditions", [])) or "general keyword overlap"
        evidence = "; ".join(top_match.get("evidence", [])[:3])

        retrieval_note = ""
        if similar_sections:
            top_similar = similar_sections[0]
            retrieval_note = (
                f" Semantic retrieval also surfaced Section {top_similar['section']} "
                f"with similarity {top_similar['similarity']}."
            )

        return (
            f"Section {top_match['section']} ('{top_match['title']}') was selected because the case description "
            f"contains action markers ({actions}), intent indicators ({intents}), and harm indicators ({harms}). "
            f"The strongest matched condition was {conditions}. {evidence}.{retrieval_note}"
        )
