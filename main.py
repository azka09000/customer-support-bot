from src.pdf_loader import PDFLoader


def main():
    file_path = "data/pdfs/academic_regulations.pdf"  

    loader = PDFLoader(file_path)
    text = loader.extract_text()

    print("\n===== EXTRACTED TEXT =====\n")
    print(text[:1500])  # preview first part


if __name__ == "__main__":
    main()