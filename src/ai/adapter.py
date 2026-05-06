from typing import Any, Protocol


class LegalAIProvider(Protocol):
    name: str

    def explain_recommendations(self, facts: dict[str, Any], recommendations: list[dict[str, Any]]) -> str:
        ...

    def answer_doubt(self, question: str, context: dict[str, Any] | None = None) -> str:
        ...


class DeterministicLegalAI:
    name = "deterministic"

    def explain_recommendations(self, facts: dict[str, Any], recommendations: list[dict[str, Any]]) -> str:
        if not recommendations:
            return "I do not have enough matched legal ingredients to recommend a section confidently. Please add more facts."
        lines = ["I matched the case facts against structured legal ingredients and found these likely provisions:"]
        for item in recommendations[:3]:
            lines.append(f"{item['law']} Section {item['section']}: {item['title']} ({item['confidence']:.2f} confidence).")
        return " ".join(lines)

    def answer_doubt(self, question: str, context: dict[str, Any] | None = None) -> str:
        section = context.get("section") if context else None
        law = context.get("law") if context else None
        if law and section:
            return f"For {law} Section {section}, read the original text and matched ingredients together. The safe view is to confirm each required fact before relying on the provision."
        return "Ask the doubt with a selected law section or matter context, and I will ground the answer in that material."


def get_ai_provider(provider_name: str | None = None) -> LegalAIProvider:
    return DeterministicLegalAI()
