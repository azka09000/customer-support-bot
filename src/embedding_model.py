from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self):
        # lightweight but strong model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, texts):
        return self.model.encode(texts, show_progress_bar=False)

    def embed_query(self, text):
        return self.model.encode([text])[0]