import PyPDF2
from pathlib import Path

class DocumentReader:
    def __init__(self):
        self.supported_extensions = {".pdf", ".txt"}

    def read_document(self, file_path: Path) -> dict:
        ext = file_path.suffix.lower()
        if ext not in self.supported_extensions:
            raise ValueError(f"Unsupported file extension: {ext}")
        
        if ext == ".pdf":
            return self.read_pdf(file_path)
        elif ext == ".txt":
            return self.read_txt(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        
    def read_pdf(self, file_path) -> dict:
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = []
                for page in reader.pages:
                    text.append(page.extract_text() or "")
                content = "\n".join(text)
            return {"title": file_path.name, "content": content}
        except Exception as e:
            raise Exception("PDF text extraction failed.", e)

    def read_txt(self, file_path) -> dict:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return {"title": file_path.name, "content": content}
        except Exception as e:
            raise Exception("Text extraction failed.", e)