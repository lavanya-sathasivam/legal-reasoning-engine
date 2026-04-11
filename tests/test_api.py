import unittest

try:
    from fastapi.testclient import TestClient
    from src.api.app import app
    FASTAPI_AVAILABLE = True
except ModuleNotFoundError:
    FASTAPI_AVAILABLE = False
    TestClient = None
    app = None


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed in the current environment")
class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_homepage_serves_updated_html(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("Legal RAG Assistant", response.text)

    def test_chat_endpoint_supports_casual_and_legal_modes(self) -> None:
        casual_response = self.client.post("/chat", json={"message": "Hello"})
        self.assertEqual(casual_response.status_code, 200)
        self.assertEqual(casual_response.json()["mode"], "casual")

        legal_response = self.client.post(
            "/chat",
            json={"message": "Which law applies if a witness statement is being challenged for admissibility?"},
        )
        self.assertEqual(legal_response.status_code, 200)
        payload = legal_response.json()
        self.assertEqual(payload["mode"], "legal_answer")
        self.assertIn("applicable_sections", payload)
        self.assertLessEqual(len(payload["applicable_sections"]), 5)

    def test_analyze_endpoint_remains_available(self) -> None:
        response = self.client.post(
            "/analyze",
            json={"description": "Which law applies if a witness statement is being challenged for admissibility?"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(set(payload.keys()), {"sections", "reason", "similar_cases"})


if __name__ == "__main__":
    unittest.main()
