from sentence_transformers import CrossEncoder
import numpy as np


class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a cross-encoder model.

    Cross-encoders directly score query-document pairs, often more accurate
    than embedding-based similarity for ranking relevant passages.
    """

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder for reranking.

        Args:
            model_name: HuggingFace cross-encoder model ID
                - Default: ms-marco-MiniLM (80M params, ~50ms per ranking)
                - Alternatives: cross-encoder/qnli-distilroberta-base (faster, lower quality)
        """
        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidates, top_k=None):
        """
        Rerank candidates based on relevance to query.

        Args:
            query: Query string
            candidates: List of candidate chunk strings
            top_k: If provided, return only top-k results (default: return all)

        Returns:
            List of candidates sorted by relevance (highest first)
        """
        if not candidates:
            return candidates

        # Score all query-candidate pairs
        pairs = [[query, candidate] for candidate in candidates]
        scores = self.model.predict(pairs)

        # Sort by scores (descending)
        ranked_indices = np.argsort(scores)[::-1]
        ranked = [candidates[i] for i in ranked_indices]

        return ranked[:top_k] if top_k else ranked
