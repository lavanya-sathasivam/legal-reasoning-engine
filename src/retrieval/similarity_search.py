import re
from typing import Any

from src.config import DEFAULT_TOP_K_RESULTS, MIN_RETRIEVAL_SCORE
from src.retrieval.embedding_model import EmbeddingModel


class SimilaritySearch:
    def __init__(self, records: list[dict[str, Any]], embedding_model: EmbeddingModel) -> None:
        self.records = records
        self.embedding_model = embedding_model
        self.record_embeddings = self.embedding_model.load_or_build_embeddings(records)

    @staticmethod
    def _cosine_similarity(query_vector, matrix) -> list[float]:
        def norm(vector) -> float:
            return sum(value * value for value in vector) ** 0.5

        query_norm = norm(query_vector)
        similarities = []
        for row in matrix:
            denominator = max(query_norm * norm(row), 1e-9)
            numerator = sum(left * right for left, right in zip(row, query_vector))
            similarities.append(numerator / denominator)
        return similarities

    @staticmethod
    def _keyword_overlap_score(query_terms: set[str], record: dict[str, Any]) -> float:
        record_text = record.get("retrieval_text", "").lower()
        record_tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9'\-]{2,}", record_text))
        exact_tags = {
            record["law"].lower(),
            record["title"].lower(),
            record["section"].lower(),
            *[tag.lower() for tag in record.get("tags", [])],
            *[issue.lower() for issue in record.get("issue_types", [])],
        }
        overlap = 0.0
        for term in query_terms:
            if len(term) < 3:
                continue
            if term in exact_tags:
                overlap += 0.18
            elif term in record_tokens:
                overlap += 0.12
        return min(overlap, 0.6)

    @staticmethod
    def _issue_type_boost(issue_types: list[str], record: dict[str, Any]) -> float:
        record_issue_types = {issue.lower() for issue in record.get("issue_types", [])}
        return 0.12 if record_issue_types.intersection({issue.lower() for issue in issue_types}) else 0.0

    @staticmethod
    def _contradiction_penalty(query_terms: set[str], record: dict[str, Any]) -> float:
        record_text = record.get("retrieval_text", "").lower()
        penalty = 0.0
        if any(term in record_text for term in ("murder", "homicide", "death")) and not query_terms.intersection({"death", "murder", "kill", "homicide", "fatal"}):
            penalty += 0.28
        if "rape" in record_text and "rape" not in query_terms:
            penalty += 0.24
        if "evidence" in record_text and not query_terms.intersection({"evidence", "witness", "statement", "admissibility", "admissible", "document", "proof", "testimony", "declaration"}):
            penalty += 0.08
        return penalty

    def search(
        self,
        query_text: str,
        issue_types: list[str] | None = None,
        query_terms: set[str] | None = None,
        top_k: int = DEFAULT_TOP_K_RESULTS,
    ) -> list[dict[str, Any]]:
        issue_types = issue_types or []
        query_terms = query_terms or set()
        query_embedding = self.embedding_model.encode([query_text])[0]
        similarities = self._cosine_similarity(query_embedding, self.record_embeddings)

        results = []
        for index, semantic_score in enumerate(similarities):
            record = self.records[index]
            lexical_score = self._keyword_overlap_score(query_terms, record)
            if self.embedding_model.backend == "hashing":
                final_score = semantic_score * 0.15 + lexical_score
            else:
                final_score = semantic_score * 0.58 + lexical_score
            final_score += self._issue_type_boost(issue_types, record)
            final_score -= self._contradiction_penalty(query_terms, record)
            if record.get("is_general_section"):
                final_score -= 0.22
            if record.get("is_contextual_section"):
                final_score -= 0.12

            if final_score < MIN_RETRIEVAL_SCORE:
                continue

            results.append(
                {
                    **record,
                    "semantic_score": round(float(semantic_score), 4),
                    "score": round(float(final_score), 4),
                }
            )

        results.sort(
            key=lambda item: (
                item.get("is_general_section", False),
                item.get("is_contextual_section", False),
                -item["score"],
            )
        )
        return results[:top_k]
