import numpy as np


class FAISSVectorStore:
    """
    FAISS-based vector store for efficient similarity search.
    Use this for production systems with large-scale data.
    """

    def __init__(self, dimension=None):
        """
        Initialize FAISS vector store.

        Args:
            dimension: Embedding dimension (auto-detected from first add)
        """
        try:
            import faiss
            self.faiss = faiss
            self.index = None
            self.dimension = dimension
        except ImportError:
            raise ImportError(
                "FAISS not installed. Install with: pip install faiss-cpu or faiss-gpu"
            )

        self.chunks = []

    def add_chunks(self, chunks, vectors):
        """
        Add chunks and their embeddings to the FAISS index.

        Args:
            chunks: List of text strings
            vectors: numpy array of embeddings (N, D)
        """
        vectors = np.array(vectors, dtype=np.float32)

        if self.index is None:
            self.dimension = vectors.shape[1]
            self.index = self.faiss.IndexFlatL2(self.dimension)

        self.chunks = chunks
        self.index.add(vectors)

    def search(self, query_vector, k=3):
        """
        Search for k nearest neighbors using FAISS.

        Args:
            query_vector: Query embedding
            k: Number of results to return

        Returns:
            List of most similar chunks
        """
        query_vector = np.array([query_vector], dtype=np.float32)
        distances, indices = self.index.search(query_vector, k)

        return [self.chunks[i] for i in indices[0]]
