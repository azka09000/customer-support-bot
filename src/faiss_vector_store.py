import numpy as np
import os


class FAISSVectorStore:
    """
    FAISS-based vector store for efficient similarity search.
    Use for production systems with large document sets.
    """

    def __init__(self, dimension=None, index_path=None):
        """
        Initialize FAISS vector store.

        Args:
            dimension: Embedding dimension (auto-detected from first add)
            index_path: Path to save/load index (optional)
        """
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError(
                "FAISS not installed. Install with: pip install faiss-cpu"
            )

        self.index = None
        self.dimension = dimension
        self.index_path = index_path
        self.chunks = []

    def _normalize_vectors(self, vectors):
        """L2-normalize vectors for cosine similarity via inner product."""
        norm = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norm + 1e-8)

    def add_chunks(self, chunks, vectors):
        """
        Add chunks and their embeddings to the FAISS index.

        Args:
            chunks: List of text strings
            vectors: numpy array of embeddings (N, D)
        """
        vectors = np.array(vectors, dtype=np.float32)

        # L2-normalize for cosine similarity
        vectors = self._normalize_vectors(vectors)

        if self.index is None:
            self.dimension = vectors.shape[1]
            # IndexFlatIP: inner product on normalized vectors = cosine similarity
            self.index = self.faiss.IndexFlatIP(self.dimension)

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
        # L2-normalize query for cosine similarity
        query_vector = self._normalize_vectors(query_vector)

        distances, indices = self.index.search(query_vector, k)

        return [self.chunks[i] for i in indices[0]]

    def save(self, path=None):
        """Save index to disk for persistence."""
        if self.index is None:
            raise ValueError("No index to save")

        save_path = path or self.index_path
        if not save_path:
            raise ValueError("No save path specified")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.faiss.write_index(self.index, save_path)

    def load(self, path=None):
        """Load index from disk."""
        load_path = path or self.index_path
        if not load_path or not os.path.exists(load_path):
            raise ValueError(f"Index path not found: {load_path}")

        self.index = self.faiss.read_index(load_path)
