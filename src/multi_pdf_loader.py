import os
from pathlib import Path
from src.pdf_loader import PDFLoader


class MultiPDFLoader:
    """Load and process multiple PDFs from a directory."""

    def __init__(self, pdf_dir="data/pdfs"):
        self.pdf_dir = pdf_dir
        self.documents = {}  # {filename: text}

    def load_all(self):
        """Load all PDFs from directory."""
        if not os.path.exists(self.pdf_dir):
            raise FileNotFoundError(f"PDF directory not found: {self.pdf_dir}")

        pdf_files = sorted(Path(self.pdf_dir).glob("*.pdf"))

        if not pdf_files:
            raise FileNotFoundError(f"No PDFs found in {self.pdf_dir}")

        print(f"\nLoading {len(pdf_files)} PDFs from {self.pdf_dir}...")

        for pdf_path in pdf_files:
            loader = PDFLoader(str(pdf_path))
            text = loader.extract_text()
            filename = pdf_path.name
            self.documents[filename] = text
            print(f"  ✓ {filename} ({len(text)} chars)")

        return self.documents

    def get_document(self, filename):
        """Get text for a specific document."""
        return self.documents.get(filename)

    def get_all_text(self):
        """Get all text combined."""
        return "\n\n---\n\n".join(self.documents.values())
