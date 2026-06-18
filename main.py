from src.pdf_loader import PDFLoader
from src.text_chunker import TextChunker
from src.semantic_embedder import SemanticEmbedder
from src.vector_store import VectorStore
from src.faiss_vector_store import FAISSVectorStore
from src.reranker import CrossEncoderReranker
from src.llm import GeminiLLM


CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
RETRIEVAL_K = 10  # Initial retrieval (before reranking)
USE_FAISS = True  # Use FAISS for faster retrieval (requires: pip install faiss-cpu)


def main():
    file_path = "data/pdfs/academic_regulations.pdf"

    # Step 1: Load PDF
    print("Step 1: Loading PDF...")
    loader = PDFLoader(file_path)
    text = loader.extract_text()
    print(f"✓ PDF loaded successfully")

    # Step 2: Chunk text
    print("\nStep 2: Chunking text...")
    chunker = TextChunker(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    chunks = chunker.chunk_text(text)
    print(f"✓ Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # Step 3: Semantic embeddings (SentenceTransformers)
    print("\nStep 3: Computing semantic embeddings...")
    embedder = SemanticEmbedder(model_name="all-MiniLM-L6-v2")
    chunk_vectors = embedder.embed(chunks)
    print(f"✓ Generated semantic embeddings (shape: {chunk_vectors.shape})")

    # Step 4: Vector store with cosine similarity
    print("\nStep 4: Building vector store...")
    if USE_FAISS:
        try:
            store = FAISSVectorStore()
            store.add_chunks(chunks, chunk_vectors)
            print(f"✓ FAISS vector store initialized (faster retrieval)")
        except ImportError:
            print("⚠ FAISS not installed, falling back to NumPy VectorStore")
            print("  Install with: pip install faiss-cpu")
            store = VectorStore()
            store.add_chunks(chunks, chunk_vectors)
    else:
        store = VectorStore()
        store.add_chunks(chunks, chunk_vectors)
    print(f"✓ Vector store ready with {len(chunks)} chunks")

    # Step 5-6: Semantic search + LLM generation
    print("\nStep 5-6: Running enhanced RAG pipeline...")
    question = "What are the rules for examinations?"
    query_vector = embedder.embed_query(question)

    results = store.search(query_vector, k=RETRIEVAL_K)
    print(f"✓ Retrieved {len(results)} candidate chunks")

    # Step 7: Rerank candidates using cross-encoder
    print("\nStep 7: Reranking with cross-encoder...")
    reranker = CrossEncoderReranker()
    results = reranker.rerank(question, results, top_k=3)
    print(f"✓ Reranked to top-3 results")

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