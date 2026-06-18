from src.pdf_loader import PDFLoader
from src.text_chunker import TextChunker
from src.semantic_embedder import SemanticEmbedder
from src.vector_store import VectorStore
from src.llm import GeminiLLM


def main():
    file_path = "data/pdfs/academic_regulations.pdf"

    # Step 1: Load PDF
    print("Step 1: Loading PDF...")
    loader = PDFLoader(file_path)
    text = loader.extract_text()
    print(f"✓ PDF loaded successfully")

    # Step 2: Chunk text
    print("\nStep 2: Chunking text...")
    chunker = TextChunker(chunk_size=500, overlap=100)
    chunks = chunker.chunk_text(text)
    print(f"✓ Created {len(chunks)} chunks")

    # Step 3: Semantic embeddings (SentenceTransformers)
    print("\nStep 3: Computing semantic embeddings...")
    embedder = SemanticEmbedder(model_name="all-MiniLM-L6-v2")
    chunk_vectors = embedder.embed(chunks)
    print(f"✓ Generated semantic embeddings (shape: {chunk_vectors.shape})")

    # Step 4: Vector store with cosine similarity
    print("\nStep 4: Building vector store...")
    store = VectorStore()
    store.add_chunks(chunks, chunk_vectors)
    print(f"✓ Vector store initialized with {len(chunks)} chunks")

    # Step 5 & 6: Semantic search + LLM generation
    print("\nStep 5-6: Running modern RAG pipeline...")
    question = "What are the rules for examinations?"
    query_vector = embedder.embed_query(question)

    results = store.search(query_vector, k=3)
    print(f"✓ Retrieved {len(results)} semantically relevant chunks")

    print("\n" + "="*50)
    print("TOP CHUNKS:")
    print("="*50)
    for i, chunk in enumerate(results, 1):
        print(f"\nChunk {i}:")
        print("-" * 40)
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)

    # Final step: LLM answer generation
    print("\n" + "="*50)
    print("Generating answer with Gemini...")
    print("="*50)
    llm = GeminiLLM()
    answer = llm.generate_answer(question, results)

    print("\nFINAL ANSWER:")
    print("-" * 40)
    print(answer)


if __name__ == "__main__":
    main()