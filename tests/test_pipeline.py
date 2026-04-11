import unittest
from unittest.mock import patch

from src.pipeline import LegalAnalysisPipeline


class PipelineBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = LegalAnalysisPipeline()

    def test_casual_input_returns_casual_mode(self) -> None:
        result = self.pipeline.chat("Hello there")
        self.assertEqual(result["mode"], "casual")
        self.assertEqual(result["applicable_sections"], [])

    def test_criminal_query_returns_grounded_bns_sections(self) -> None:
        result = self.pipeline.chat(
            "The accused intentionally hit the victim with an iron rod and caused a fracture. Which sections may apply?"
        )
        self.assertEqual(result["mode"], "legal_answer")
        self.assertTrue(result["applicable_sections"])
        self.assertLessEqual(len(result["applicable_sections"]), 5)
        self.assertTrue(all(item["law"] == "BNS" for item in result["applicable_sections"]))
        self.assertIn("Applicable Legal Sections", result["message"])

    def test_ambiguous_query_requests_clarification(self) -> None:
        result = self.pipeline.chat("The police took him away and I need to know what procedure applies.")
        self.assertEqual(result["mode"], "needs_clarification")
        self.assertTrue(result["follow_up_questions"])
        self.assertEqual(result["applicable_sections"], [])

    def test_legal_response_uses_required_format_markers(self) -> None:
        result = self.pipeline.chat("Which law applies if a witness statement is being challenged for admissibility?")
        self.assertEqual(result["mode"], "legal_answer")
        self.assertIn("Applicable Legal Sections", result["message"])
        self.assertIn("Why it applies", result["message"])
        self.assertIn("Explanation", result["message"])

    def test_no_match_does_not_hallucinate_sections(self) -> None:
        with patch.object(self.pipeline.similarity_search, "search", return_value=[]):
            result = self.pipeline.chat("Which law applies if a witness statement is being challenged for admissibility?")
        self.assertEqual(result["mode"], "legal_answer")
        self.assertEqual(result["applicable_sections"], [])
        self.assertIn("not sufficiently confident", result["message"])

    def test_legacy_analyze_adapter_returns_compatible_shape(self) -> None:
        result = self.pipeline.analyze_case("Which law applies if a witness statement is being challenged for admissibility?")
        self.assertEqual(set(result.keys()), {"sections", "reason", "similar_cases"})
        self.assertIsInstance(result["sections"], list)
        self.assertIsInstance(result["similar_cases"], list)
        self.assertIsInstance(result["reason"], str)


if __name__ == "__main__":
    unittest.main()
