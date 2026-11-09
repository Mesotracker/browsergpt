from __future__ import annotations

import io

import pytest
from docx import Document as DocxDocument

from chainforge.ingest import IngestError, dedupe_documents, extract_bytes


def test_plain_text_and_markdown():
    text = extract_bytes(b"# Title\n\nHello **world**.\n", "notes.md")
    assert "Hello" in text.text
    assert text.name == "notes.md"


def test_unknown_extension_decodes_with_warning():
    doc = extract_bytes(b"some odd but readable content " * 10, "artifact.weird")
    assert "readable content" in doc.text
    assert any("unknown extension" in warning for warning in doc.warnings)


def test_docx_paragraphs_and_tables():
    buffer = io.BytesIO()
    document = DocxDocument()
    document.add_paragraph("A paragraph from Word.")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "cell one"
    table.cell(0, 1).text = "cell two"
    document.save(buffer)
    doc = extract_bytes(buffer.getvalue(), "report.docx")
    assert "A paragraph from Word." in doc.text
    assert "cell one | cell two" in doc.text


def test_binary_is_rejected():
    with pytest.raises(IngestError):
        extract_bytes(b"\x89PNG\r\n\x1a\n" + bytes(range(1, 200)), "image.png")


def test_legacy_doc_is_rejected_with_guidance():
    with pytest.raises(IngestError, match="docx"):
        extract_bytes(b"\xd0\xcf\x11\xe0", "old.doc")


def test_duplicate_documents_are_removed():
    first = extract_bytes(b"same text " * 20, "a.txt")
    second = extract_bytes(b"same text " * 20, "b.txt")
    unique, skipped = dedupe_documents([first, second])
    assert len(unique) == 1
    assert skipped
