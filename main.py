from src.pdf_loader import PDFLoader
from src.text_chunker import TextChunker
from src.tfidf_embedder import TFIDFEmbedder
from src.vector_store import VectorStore


def main():
    file_path = "data/pdfs/academic_regulations.pdf"

    # Step 1: extract text
    loader = PDFLoader(file_path)
    text = loader.extract_text()

    # Step 2: chunk text
    chunker = TextChunker(chunk_size=500, overlap=100)
    chunks = chunker.chunk_text(text)

    print("\nTOTAL CHUNKS:", len(chunks))

    # Step 3: TF-IDF embeddings
    embedder = TFIDFEmbedder()
    chunk_vectors = embedder.fit_transform(chunks)

    # Step 4: store vectors
    store = VectorStore()
    store.add_chunks(chunks, chunk_vectors)

    # Step 5: query test
    question = "What are the rules for examinations?"
    query_vector = embedder.transform([question])

    results = store.search(query_vector, k=3)

    print("\nTOP RESULTS:\n")

    for i, r in enumerate(results, 1):
        print(f"\nResult {i}")
        print("-" * 40)
        print(r)


if __name__ == "__main__":
    main()