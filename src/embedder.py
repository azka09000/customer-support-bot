from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        """
        Converts list of text chunks into embeddings.
        Returns: list of vectors
        """
        embeddings = self.model.encode(texts)
        return embeddings