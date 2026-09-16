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

def parse_pdf(file_path: str, filename_hint: str | None = None) -> ParsedPDF:
    with fitz.open(file_path) as doc:
        pages = [str(page.get_text("text")) for page in doc]
        metadata = doc.metadata or {}
        fallback = Path(filename_hint).stem if filename_hint else Path(file_path).stem
        title = metadata.get("title") or fallback

    return ParsedPDF(
        text = "\n".join(pages),
        title = title,
        page_count = len(pages)
    )
