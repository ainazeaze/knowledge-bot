from dataclasses import dataclass
import fitz
from pathlib import Path

@dataclass
class ParsedPDF:
    text: str
    title: str
    page_count : int

    @property
    def is_valid(self) -> bool:
        return len(self.text.strip()) > 50

def parse_pdf(file_path: str) -> ParsedPDF:
    with fitz.open(file_path) as doc:
        pages = [str(page.get_text("text")) for page in doc]
        metadata = doc.metadata or {}
        title = metadata.get("title") or Path(file_path).stem

    return ParsedPDF(
        text = "\n".join(pages),
        title = title,
        page_count = len(pages)
    )
