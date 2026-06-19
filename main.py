from src.multi_pdf_loader import MultiPDFLoader
from src.multi_pdf_chunker import MultiPDFChunker
from src.semantic_embedder import SemanticEmbedder
from src.vector_store import VectorStore
from src.faiss_vector_store import FAISSVectorStore
from src.reranker import CrossEncoderReranker
from src.llm import GeminiLLM


CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
RETRIEVAL_K = 3
USE_FAISS = True
USE_RERANKING = False


def main():
    # Step 1: Load multiple PDFs
    print("Step 1: Loading multiple PDFs...")
    pdf_loader = MultiPDFLoader(pdf_dir="data/pdfs")
    documents = pdf_loader.load_all()
    print(f"✓ Loaded {len(documents)} documents")

    # Step 2: Chunk all documents
    print("\nStep 2: Chunking documents...")
    chunker = MultiPDFChunker(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    chunks, metadata = chunker.chunk_documents(documents)
    print(f"✓ Created {len(chunks)} chunks from all documents")

    # Step 3: Compute semantic embeddings
    print("\nStep 3: Computing semantic embeddings...")
    embedder = SemanticEmbedder(model_name="all-MiniLM-L6-v2")
    chunk_vectors = embedder.embed(chunks)
    print(f"✓ Generated semantic embeddings (shape: {chunk_vectors.shape})")

    # Step 4: Build vector store
    print("\nStep 4: Building vector store...")
    if USE_FAISS:
        try:
            store = FAISSVectorStore()
            store.add_chunks(chunks, chunk_vectors)
            print(f"✓ FAISS vector store initialized (faster retrieval)")
        except ImportError:
            print("⚠ FAISS not installed, falling back to NumPy VectorStore")
            store = VectorStore()
            store.add_chunks(chunks, chunk_vectors, metadata=metadata)
    else:
        store = VectorStore()
        store.add_chunks(chunks, chunk_vectors, metadata=metadata)
    print(f"✓ Vector store ready with {len(chunks)} chunks")

    # Step 5-6: Semantic search
    print("\nStep 5-6: Running enhanced RAG pipeline...")
    question = "What are the examination rules?"
    query_vector = embedder.embed_query(question)

    # Get results with metadata if using NumPy store
    if isinstance(store, VectorStore):
        results_with_meta = store.search_with_metadata(query_vector, k=RETRIEVAL_K)
    else:
        # FAISS doesn't have metadata support yet, get basic results
        results_with_meta = [{"chunk": c, "metadata": metadata[i], "score": 0}
                            for i, c in enumerate(store.search(query_vector, k=RETRIEVAL_K))]

    print(f"✓ Retrieved {len(results_with_meta)} results")

    print("\n" + "="*60)
    print("RETRIEVED CHUNKS WITH SOURCES:")
    print("="*60)
    results = []
    for i, result in enumerate(results_with_meta, 1):
        chunk = result["chunk"]
        meta = result["metadata"]
        source = meta["source"] if meta else "Unknown"

        print(f"\n[Result {i}] Source: {source}")
        print("-" * 60)
        print(chunk[:250] + "..." if len(chunk) > 250 else chunk)
        results.append(chunk)

    # Step 7: Optional reranking
    if USE_RERANKING:
        print("\n" + "="*60)
        print("Step 7: Reranking with cross-encoder...")
        reranker = CrossEncoderReranker()
        results = reranker.rerank(question, results, top_k=3)
        print(f"✓ Reranked to top-3 results")

    # Step 8: Generate answer with LLM
    print("\n" + "="*60)
    print("Generating answer with Gemini...")
    print("="*60)
    llm = GeminiLLM()
    answer = llm.generate_answer(question, results)

    print("\nFINAL ANSWER:")
    print("-" * 60)
    print(answer)


if __name__ == "__main__":
    main()