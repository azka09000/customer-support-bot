import numpy as np


class VectorStore:
    def __init__(self):
        self.chunks = []
        self.vectors = None

    def add_chunks(self, chunks, vectors):
        self.chunks = chunks
        self.vectors = vectors  # ❗ keep sparse matrix as-is

    def search(self, query_vector, k=3):
        # TF-IDF query is (1, n_features)
        query_vector = query_vector

        # cosine similarity using sparse matrix multiplication
        scores = (self.vectors @ query_vector.T).toarray().ravel()

        # normalize (cosine similarity)
        doc_norms = np.linalg.norm(self.vectors.toarray(), axis=1)
        query_norm = np.linalg.norm(query_vector.toarray())

        scores = scores / (doc_norms * query_norm + 1e-10)

        top_k_idx = np.argsort(scores)[::-1][:k]

        return [self.chunks[i] for i in top_k_idx]