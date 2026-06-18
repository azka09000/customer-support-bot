from src.pdf_loader import PDFLoader
from src.text_chunker import TextChunker
from src.tfidf_embedder import TFIDFEmbedder
from src.vector_store import VectorStore
from src.llm import GeminiLLM


def main():
    file_path = "data/pdfs/academic_regulations.pdf"

    # Step 1: Load and extract text from PDF
    loader = PDFLoader(file_path)
    text = loader.extract_text()

    # Step 2: Split text into chunks
    chunker = TextChunker(
        chunk_size=500,
        overlap=100
    )
    chunks = chunker.chunk_text(text)

    print("\nTOTAL CHUNKS:", len(chunks))

    # Step 3: Create TF-IDF embeddings
    embedder = TFIDFEmbedder()
    chunk_vectors = embedder.fit_transform(chunks)

    # Step 4: Store chunks and vectors
    store = VectorStore()
    store.add_chunks(chunks, chunk_vectors)

    # Step 5: User query
    question = "What are the rules for examinations?"
    print("\nQUESTION:")
    print(question)

    # Convert question into TF-IDF vector
    query_vector = embedder.transform([question])

    # Retrieve top-k relevant chunks
    results = store.search(query_vector, k=3)

    if not results:
        print("\nNo relevant information found.")
        return

    print("\nTOP CHUNKS:")

    for i, chunk in enumerate(results, start=1):
        print(f"\nChunk {i}")
        print("-" * 40)
        print(chunk)

    # Step 6: Generate final answer using Gemini
    llm = GeminiLLM()
    answer = llm.generate_answer(question, results)

    print("\nFINAL ANSWER:")
    print(answer)


if __name__ == "__main__":
    main()