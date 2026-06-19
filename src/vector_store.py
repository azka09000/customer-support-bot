import numpy as np


class VectorStore:
    def __init__(self):
        self.chunks = []
        self.vectors = None
        self.metadata = []  # Track source info for each chunk

    def add_chunks(self, chunks, vectors, metadata=None):
        self.chunks = chunks
        self.vectors = np.array(vectors)
        self.metadata = metadata if metadata else [None] * len(chunks)

    def search(self, query_vector, k=3):
        query_vector = np.array(query_vector)

        doc_norms = np.linalg.norm(self.vectors, axis=1)
        query_norm = np.linalg.norm(query_vector)

        scores = np.dot(self.vectors, query_vector) / (doc_norms * query_norm + 1e-8)

        top_k_idx = np.argsort(scores)[::-1][:k]

        return [self.chunks[i] for i in top_k_idx]

    def search_with_metadata(self, query_vector, k=3):
        """Search and return results with metadata."""
        query_vector = np.array(query_vector)

        doc_norms = np.linalg.norm(self.vectors, axis=1)
        query_norm = np.linalg.norm(query_vector)

        scores = np.dot(self.vectors, query_vector) / (doc_norms * query_norm + 1e-8)

        top_k_idx = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_k_idx:
            results.append({
                "chunk": self.chunks[idx],
                "metadata": self.metadata[idx],
                "score": scores[idx]
            })

        return results