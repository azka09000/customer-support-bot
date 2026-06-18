from src.pdf_loader import PDFLoader
from src.text_chunker import TextChunker
from src.embedder import Embedder


def main():
    file_path = "data/pdfs/academic_regulations.pdf"

    # Step 1: extract text
    loader = PDFLoader(file_path)
    text = loader.extract_text()

    # Step 2: chunk text
    chunker = TextChunker(chunk_size=500, overlap=100)
    chunks = chunker.chunk_text(text)

    print("TOTAL CHUNKS:", len(chunks))

    # Step 3: embeddings
    embedder = Embedder()
    embeddings = embedder.embed(chunks)

    print("\nEMBEDDING SHAPE:")
    print(len(embeddings), len(embeddings[0]))

    print("\nSAMPLE EMBEDDING (first chunk):")
    print(embeddings[0][:10])  # first 10 values


if __name__ == "__main__":
    main()