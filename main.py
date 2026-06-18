from src.pdf_loader import PDFLoader
from src.text_chunker import TextChunker
from src.tfidf_embedder import TFIDFEmbedder
from src.vector_store import VectorStore
from src.llm import GeminiLLM


def build_system():
    file_path = "data/pdfs/academic_regulations.pdf"

    # Step 1: extract text
    loader = PDFLoader(file_path)
    text = loader.extract_text()

    # Step 2: chunk text
    chunker = TextChunker(chunk_size=500, overlap=100)
    chunks = chunker.chunk_text(text)

    print("\nTOTAL CHUNKS:", len(chunks))

    # Step 3: TF-IDF
    embedder = TFIDFEmbedder()
    chunk_vectors = embedder.fit_transform(chunks)

    # Step 4: store
    store = VectorStore()
    store.add_chunks(chunks, chunk_vectors)

    return embedder, store


def main():
    embedder, store = build_system()
    llm = GeminiLLM()

    history = []

    print("\n📄 PDF Chatbot Ready! Type 'exit' to quit.\n")

    while True:
        question = input("You: ")

        if question.lower().strip() == "exit":
            print("Goodbye 👋")
            break

        # retrieve
        query_vector = embedder.transform([question])
        results = store.search(query_vector, k=3)

        print("\nTOP CHUNKS:\n")
        for i, r in enumerate(results, 1):
            print(f"\nChunk {i}")
            print("-" * 40)
            print(r)

        # build history string (simple version)
        history_text = ""
        for hq, ha in history[-3:]:
            history_text += f"Q: {hq}\nA: {ha}\n\n"

        # generate answer
        answer = llm.generate_answer(
            question=question,
            context_chunks=results,
            history=history_text
        )

        print("\nAI:", answer)

        # save memory
        history.append((question, answer))


if __name__ == "__main__":
    main()