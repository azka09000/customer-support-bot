from src.pdf_loader import PDFLoader
from src.text_chunker import TextChunker
from src.tfidf_embedder import TFIDFEmbedder
from src.vector_store import VectorStore
from src.llm import GeminiLLM


def main():
    file_path = "data/pdfs/academic_regulations.pdf"

    # Day 1: Load PDF
    print("Day 1: Loading PDF...")
    loader = PDFLoader(file_path)
    text = loader.extract_text()
    print(f"✓ PDF loaded successfully")

    # Day 2: Chunk text
    print("\nDay 2: Chunking text...")
    chunker = TextChunker(chunk_size=500, overlap=100)
    chunks = chunker.chunk_text(text)
    print(f"✓ Created {len(chunks)} chunks")

    # Day 3: TF-IDF embeddings (no external APIs)
    print("\nDay 3: Computing TF-IDF embeddings...")
    embedder = TFIDFEmbedder()
    chunk_vectors = embedder.fit_transform(chunks).toarray()
    print(f"✓ Generated TF-IDF vectors (shape: {chunk_vectors.shape})")

    # Day 4: Store vectors and build retriever
    print("\nDay 4: Building vector store...")
    store = VectorStore()
    store.add_chunks(chunks, chunk_vectors)
    print(f"✓ Vector store initialized")

    # Day 6: End-to-end pipeline
    print("\nDay 6: Running RAG pipeline...")
    question = "What are the rules for examinations?"
    query_vector = embedder.transform([question]).toarray()[0]

    results = store.search(query_vector, k=3)
    print(f"✓ Retrieved {len(results)} relevant chunks")

    print("\n" + "="*50)
    print("TOP CHUNKS:")
    print("="*50)
    for i, chunk in enumerate(results, 1):
        print(f"\nChunk {i}:")
        print("-" * 40)
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)

    # Day 5: LLM answer generation
    print("\n" + "="*50)
    print("Day 5: Generating answer with Gemini...")
    print("="*50)
    llm = GeminiLLM()
    answer = llm.generate_answer(question, results)

    print("\nFINAL ANSWER:")
    print("-" * 40)
    print(answer)


if __name__ == "__main__":
    main()