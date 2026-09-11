"""Tests for the DOCX loader.

Test DOCX files are generated programmatically using python-docx so
no binary fixtures need to be committed.
"""

from pathlib import Path

import docx as python_docx
import pytest

from rag.ingestion.docx_loader import load_docx
from rag.ingestion.exceptions import InvalidDocumentError


def _create_docx(
    path: Path,
    paragraphs: list[str] | None = None,
    table_data: list[list[str]] | None = None,
) -> Path:
    """Create a minimal DOCX with the given paragraphs and optional table."""
    doc = python_docx.Document()
    for text in paragraphs or []:
        doc.add_paragraph(text)
    if table_data:
        rows = len(table_data)
        cols = len(table_data[0]) if table_data else 0
        table = doc.add_table(rows=rows, cols=cols)
        for r_idx, row_data in enumerate(table_data):
            for c_idx, cell_text in enumerate(row_data):
                table.rows[r_idx].cells[c_idx].text = cell_text
    doc.save(str(path))
    return path


# --- valid DOCX tests ---


def test_load_valid_docx(tmp_path: Path):
    """A valid DOCX loads successfully."""
    docx_file = _create_docx(tmp_path / "simple.docx", paragraphs=["Hello World"])
    doc = load_docx(docx_file)
    assert doc.source_type == "docx"
    assert len(doc.pages) == 1


def test_paragraph_content_extracted(tmp_path: Path):
    """Paragraph text is extracted."""
    docx_file = _create_docx(
        tmp_path / "paras.docx",
        paragraphs=["First paragraph", "Second paragraph"],
    )
    doc = load_docx(docx_file)
    assert "First paragraph" in doc.pages[0].content
    assert "Second paragraph" in doc.pages[0].content


def test_table_content_extracted(tmp_path: Path):
    """Table cell text is extracted."""
    docx_file = _create_docx(
        tmp_path / "table.docx",
        paragraphs=["Intro"],
        table_data=[["A1", "B1"], ["A2", "B2"]],
    )
    doc = load_docx(docx_file)
    content = doc.pages[0].content
    assert "A1" in content
    assert "B2" in content


def test_source_type_is_docx(tmp_path: Path):
    """Source type is set to 'docx'."""
    docx_file = _create_docx(tmp_path / "type.docx", paragraphs=["text"])
    doc = load_docx(docx_file)
    assert doc.source_type == "docx"


def test_document_id_exists(tmp_path: Path):
    """The returned document has a non-empty document_id."""
    docx_file = _create_docx(tmp_path / "id.docx", paragraphs=["text"])
    doc = load_docx(docx_file)
    assert doc.document_id
    assert len(doc.document_id) > 0


def test_document_name_preserved(tmp_path: Path):
    """The document_name matches the provided name."""
    docx_file = _create_docx(tmp_path / "named.docx", paragraphs=["text"])
    doc = load_docx(docx_file, document_name="My Report.docx")
    assert doc.document_name == "My Report.docx"


def test_document_name_falls_back_to_filename(tmp_path: Path):
    """Without an explicit name, the filename is used."""
    docx_file = _create_docx(tmp_path / "fallback.docx", paragraphs=["text"])
    doc = load_docx(docx_file)
    assert doc.document_name == "fallback.docx"


def test_page_number_is_one(tmp_path: Path):
    """DOCX documents have a single page with page_number=1."""
    docx_file = _create_docx(tmp_path / "page.docx", paragraphs=["text"])
    doc = load_docx(docx_file)
    assert doc.pages[0].page_number == 1


def test_metadata_includes_total_pages(tmp_path: Path):
    """Document metadata includes total_pages=1 for DOCX."""
    docx_file = _create_docx(tmp_path / "meta.docx", paragraphs=["text"])
    doc = load_docx(docx_file)
    assert doc.metadata["total_pages"] == 1


def test_metadata_includes_file_size(tmp_path: Path):
    """Document metadata includes file_size_bytes."""
    docx_file = _create_docx(tmp_path / "size.docx", paragraphs=["text"])
    doc = load_docx(docx_file)
    assert doc.metadata["file_size_bytes"] > 0


def test_empty_docx_loads(tmp_path: Path):
    """A DOCX with no paragraphs or tables loads with empty content."""
    docx_file = _create_docx(tmp_path / "empty.docx", paragraphs=[])
    doc = load_docx(docx_file)
    assert doc.pages[0].content == ""


# --- invalid DOCX tests ---


def test_invalid_docx_raises(tmp_path: Path):
    """A corrupted/non-DOCX file raises InvalidDocumentError."""
    bad = tmp_path / "bad.docx"
    bad.write_bytes(b"this is not a docx at all")
    with pytest.raises(InvalidDocumentError):
        load_docx(bad)


def test_unicode_content_extracted(tmp_path: Path):
    """Unicode characters in paragraphs and tables are extracted intact."""
    unicode_para = "Résumé: Élève a réussi l'examen avec €100 de bourse."
    unicode_table = [["Métrique", "Valeur"], ["Température", "25°C"]]
    docx_file = _create_docx(
        tmp_path / "unicode.docx",
        paragraphs=[unicode_para],
        table_data=unicode_table,
    )
    doc = load_docx(docx_file)
    content = doc.pages[0].content
    assert unicode_para in content
    assert "Température | 25°C" in content


def test_non_ascii_filename_handled(tmp_path: Path):
    """DOCX with non-ASCII filename is loaded successfully."""
    docx_file = _create_docx(tmp_path / "rapport_été_2024.docx", paragraphs=["Texte"])
    doc = load_docx(docx_file)
    assert doc.document_name == "rapport_été_2024.docx"
    assert "Texte" in doc.pages[0].content


def test_empty_table_rows_ignored(tmp_path: Path):
    """Table rows where all cells are whitespace/empty are not extracted as ' | '."""
    docx_file = _create_docx(
        tmp_path / "empty_rows.docx",
        paragraphs=["Header"],
        table_data=[["", ""], ["Valid", "Data"], [" ", "  "]],
    )
    doc = load_docx(docx_file)
    content = doc.pages[0].content
    # Should not contain orphaned pipes
    other_lines = [
        line.strip() for line in content.split("\n") if line.strip() != "Valid | Data"
    ]
    assert " | " not in other_lines


def test_zip_file_not_docx_raises_invalid_document(tmp_path: Path):
    """A valid ZIP file that is not a DOCX document raises InvalidDocumentError."""
    import zipfile

    fake_docx = tmp_path / "not_a_docx.docx"
    with zipfile.ZipFile(fake_docx, "w") as zf:
        zf.writestr("test.txt", "just plain text in a zip")

    with pytest.raises(InvalidDocumentError):
        load_docx(fake_docx)
