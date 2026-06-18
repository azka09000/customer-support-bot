import numpy as np


class VectorStore:
    def __init__(self):
        self.chunks = []
        self.vectors = None

    def add_chunks(self, chunks, vectors):
        self.chunks = chunks
        self.vectors = np.array(vectors)

    def search(self, query_vector, k=3):
        query_vector = np.array(query_vector)

        # normalize
        doc_norms = np.linalg.norm(self.vectors, axis=1)
        query_norm = np.linalg.norm(query_vector)

        scores = np.dot(self.vectors, query_vector) / (doc_norms * query_norm + 1e-8)

        top_k_idx = np.argsort(scores)[::-1][:k]

        return [self.chunks[i] for i in top_k_idx]