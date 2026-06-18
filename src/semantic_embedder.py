from sentence_transformers import SentenceTransformer
import numpy as np


class SemanticEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize with a SentenceTransformer model for semantic embeddings.

        Args:
            model_name: HuggingFace model ID (default: all-MiniLM-L6-v2)
                Other options: all-mpnet-base-v2 (better quality),
                             paraphrase-MiniLM-L6-v2, etc.
        """
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        """
        Convert multiple text chunks into dense semantic embeddings.

        Args:
            texts: List of text strings

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        return self.model.encode(texts, convert_to_numpy=True)

    def embed_query(self, text):
        """
        Convert a single query text into a semantic embedding.

        Args:
            text: Query string

        Returns:
            numpy array of shape (embedding_dim,)
        """
        return self.model.encode(text, convert_to_numpy=True)
