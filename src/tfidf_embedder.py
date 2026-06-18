from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFEmbedder:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, texts):
        """
        Learn vocabulary from the document chunks and
        return their TF-IDF vectors.
        """
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        """
        Transform new text (queries) using the
        vocabulary learned from the chunks.
        """
        return self.vectorizer.transform(texts)