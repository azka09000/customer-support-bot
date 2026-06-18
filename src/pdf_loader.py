import pdfplumber


class PDFLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def extract_text(self):
        """
        Extracts text from a PDF file page by page.
        Returns full text as a single string.
        """
        full_text = ""

        with pdfplumber.open(self.file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()

                # Some pages may return None
                if text:
                    full_text += text + "\n"

        return full_text