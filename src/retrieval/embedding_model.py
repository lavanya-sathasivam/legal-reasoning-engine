import hashlib
import json
import math
from pathlib import Path
from typing import Any

from src.config import COMBINED_CORPUS_PATH, DEFAULT_EMBEDDING_MODEL, EMBEDDINGS_PATH

try:
    import numpy as np
except ImportError:
    np = None


class EmbeddingModel:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, cache_path: Path = EMBEDDINGS_PATH) -> None:
        self.model_name = model_name
        self.backend = "hashing"
        self.model = None
        self.vector_size = 256
        self.cache_path = cache_path
        self.cache_path_fallback = cache_path.with_suffix(".json")
        self._load_backend()

    def _load_backend(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
            self.backend = "sentence_transformers"
        except Exception:
            self.model = None
            self.backend = "hashing"

    def encode(self, texts: list[str]):
        if self.backend == "sentence_transformers" and self.model is not None:
            embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            if np is not None:
                return np.asarray(embeddings, dtype=np.float32)
            return [embedding.tolist() for embedding in embeddings]

        embeddings = [self._hashed_embedding(text) for text in texts]
        if np is not None:
            return np.vstack(embeddings).astype(np.float32)
        return embeddings

    def _hashed_embedding(self, text: str):
        if np is not None:
            vector = np.zeros(self.vector_size, dtype=np.float32)
        else:
            vector = [0.0] * self.vector_size
        tokens = [token for token in text.lower().split() if token]
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest, 16) % self.vector_size
            vector[index] += 1.0

        if np is not None:
            norm = np.linalg.norm(vector)
            return vector if norm == 0 else vector / norm

        norm = math.sqrt(sum(value * value for value in vector))
        return vector if norm == 0 else [value / norm for value in vector]

    def build_and_cache_embeddings(self, records: list[dict[str, Any]]):
        texts = [record["retrieval_text"] for record in records]
        embeddings = self.encode(texts)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if np is not None:
            np.save(self.cache_path, embeddings)
        else:
            with self.cache_path_fallback.open("w", encoding="utf-8") as handle:
                json.dump(embeddings, handle)
        return embeddings

    def load_or_build_embeddings(self, records: list[dict[str, Any]] | None = None):
        if np is not None and self.cache_path.exists():
            return np.load(self.cache_path)
        if np is None and self.cache_path_fallback.exists():
            with self.cache_path_fallback.open("r", encoding="utf-8") as handle:
                return json.load(handle)

        if records is None:
            with COMBINED_CORPUS_PATH.open("r", encoding="utf-8") as handle:
                records = json.load(handle)
        return self.build_and_cache_embeddings(records)
