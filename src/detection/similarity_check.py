"""
Semantic Similarity signal for hallucination detection.

Uses sentence-level embeddings to measure how semantically grounded a generated
answer is in the provided context documents. This signal is primarily effective at:
- Detecting off-topic or unrelated answers
- Identifying major semantic drift from the context
- Providing a coarse grounding signal for aggregation

The signal outputs both a raw similarity score and a derived hallucination score
(1 - similarity) intended for downstream aggregation.

"""

import re
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer


class SimilarityChecker:
    """
    Semantic similarity signal for hallucination detection.
    Embeds answer and context chunks, computes cosine similarity,
    and returns the maximum similarity score.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    @staticmethod
    def _clean_answer(answer: str) -> str:
        """
        Remove citation markers like [doc1], [1], etc.
        """
        return re.sub(r"\[[^\]]+\]", "", answer).strip()

    def _embed(self, text: str) -> np.ndarray:
        """
        Embed text into a normalized vector.
        """
        if not text or not text.strip():
            return np.zeros(self.model.get_sentence_embedding_dimension(), dtype=np.float32)

        vec = self.model.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)

    def check(self, answer: str, context: List[Dict[str, str]]) -> Dict[str, object]:
        if not context:
            return {"max_similarity": 0.0,"hallucination_score": 1.0,"is_hallucination": True,"per_context_similarity": []}

        clean_answer = self._clean_answer(answer)
        answer_vec = self._embed(clean_answer)

        texts = [doc["text"] for doc in context]
        ctx_vecs = self.model.encode(texts, normalize_embeddings=True)

        per_context = []
        for doc, vec in zip(context, ctx_vecs):
            similarity = float(np.dot(answer_vec, vec))
            per_context.append(
                {"doc_id": doc["doc_id"], "similarity": similarity}
            )

        max_similarity = max(item["similarity"] for item in per_context)
        hallucination_score = 1.0 - max_similarity
        is_hallucination = max_similarity < 0.7

        return {
            "max_similarity": max_similarity,
            "hallucination_score": hallucination_score,
            "is_hallucination": is_hallucination,
            "per_context_similarity": per_context,
        }

    
