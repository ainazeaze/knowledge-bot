import fitz
import pytest
from knowledge_bot.pdf_parser import parse_pdf


def _make_pdf(tmp_path, text="Hello world " * 20, title="Test PDF", pages=1):
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_text((50, 50), text)
    if title:
        doc.set_metadata({"title": title})
    path = tmp_path / "test.pdf"
    doc.save(str(path))
    doc.close()
    return str(path)


def test_is_valid(tmp_path):
    path = _make_pdf(tmp_path)
    assert parse_pdf(path).is_valid is True


def test_page_count(tmp_path):
    path = _make_pdf(tmp_path, pages=3)
    assert parse_pdf(path).page_count == 3


def test_title_from_metadata(tmp_path):
    path = _make_pdf(tmp_path, title="My Title")
    assert parse_pdf(path).title == "My Title"


def test_title_falls_back_to_filename(tmp_path):
    doc = fitz.open()
    doc.new_page()
    path = tmp_path / "my_document.pdf"
    doc.save(str(path))
    doc.close()
    assert parse_pdf(str(path)).title == "my_document"


def test_text_content(tmp_path):
    path = _make_pdf(tmp_path, text="Quantum entanglement " * 10)
    assert "Quantum" in parse_pdf(path).text
