import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:
    def __init__(self):
        self.vectors = None
        self.chunks = None

    def add_chunks(self, chunks, vectors):
        self.chunks = chunks
        self.vectors = vectors

    def search(self, query_vector, k=3):
        scores = cosine_similarity(query_vector, self.vectors)[0]

        top_k_idx = np.argsort(scores)[::-1][:k]

        return [self.chunks[i] for i in top_k_idx]