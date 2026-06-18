from src.pdf_loader import PDFLoader
from src.text_chunker import TextChunker


def main():
    file_path = "data/pdfs/academic_regulations.pdf"

    # Step 1: extract text
    loader = PDFLoader(file_path)
    text = loader.extract_text()

    # Step 2: chunk text
    chunker = TextChunker(chunk_size=500, overlap=100)
    chunks = chunker.chunk_text(text)

    # Step 3: results
    print("\nTOTAL CHUNKS:", len(chunks))

    print("\nSAMPLE CHUNK:\n")
    print(chunks[0])

    print("\nANOTHER CHUNK:\n")
    print(chunks[1])


if __name__ == "__main__":
    main()