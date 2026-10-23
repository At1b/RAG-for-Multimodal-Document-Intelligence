"""Deterministic synthetic fixture generator for Phase 10 OCR evaluation.

Generates reproducible PDF, PNG, and JPEG fixtures using PyMuPDF and Pillow,
along with exact matching ground-truth reference texts.
Target file size for each fixture is <= 100 KB, total < 1 MB.
"""

import io
from pathlib import Path

import pymupdf
from PIL import Image

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
GROUND_TRUTH_DIR = Path(__file__).resolve().parent / "ground_truth"

# --- Text Definitions ---

CLEAN_SINGLE_TEXT = (
    "Document intelligence enables automated extraction and comprehension "
    "of content from diverse file formats. Modern retrieval-augmented generation "
    "systems integrate multimodal components to process both native text layers "
    "and scanned document images reliably."
)

MULTI_PARAGRAPH_TEXT = (
    "The retrieval pipeline begins by indexing chunked representations into a "
    "vector store. Each chunk maintains explicit provenance including document "
    "identity and page numbers.\n\n"
    "When queries arrive at the system, hybrid retrieval combines dense vector "
    "embeddings with sparse keyword matching. Reciprocal rank fusion merges the "
    "ranked candidates into an initial candidate set.\n\n"
    "Cross-encoder reranking provides fine-grained relevance scoring across the "
    "candidate pool. The top-ranked passages are presented to the generation "
    "engine with verifiable citations."
)

HEADINGS_PUNCT_TEXT = (
    "SECTION 1.0: EXECUTIVE SUMMARY\n\n"
    "The evaluation recorded 1,250 queries (sample size: 95% confidence interval; "
    "margin of error: +/-2.5%). Key findings include:\n"
    "- System throughput reached 45.8 requests/second;\n"
    '- Accuracy improved by 18.4% ("high-precision mode");\n'
    "- Latency remained under 350ms across all trials."
)

SCAN_3PAGE_TEXTS = [
    (
        "Page one introduces the fundamental principles of multimodal document "
        "ingestion. Text layers are analyzed to identify scanned content requiring "
        "optical character recognition."
    ),
    (
        "Page two examines layout reconstruction and bounding box coordinates. "
        "Words are grouped into lines and blocks based on spatial coordinates "
        "returned by the optical engine."
    ),
    (
        "Page three concludes with provenance preservation and citation integrity. "
        "Metadata tracks the exact source document, page number, and extraction "
        "method throughout the indexing workflow."
    ),
]

MIXED_PDF_TEXTS = [
    (
        "Native page one contains standard digital text extracted directly "
        "from the document stream without optical processing."
    ),
    (
        "Scanned page two requires optical character recognition due to the "
        "absence of an underlying digital text stream."
    ),
    (
        "Native page three provides another digital text section confirming "
        "that mixed-mode documents preserve original text streams."
    ),
    (
        "Scanned page four concludes the mixed-mode document demonstration with "
        "a second scanned page processed via optical recognition."
    ),
]

IMAGE_PNG_TEXT = (
    "Direct image ingestion allows standalone graphics and photographs of text "
    "documents to enter the multi-modal retrieval pipeline. Images undergo "
    "preprocessing before optical character recognition."
)

IMAGE_JPG_TEXT = (
    "Standard JPEG images are commonly produced by mobile cameras and document "
    "scanners. The ingestion pipeline normalizes orientation and converts images "
    "to grayscale for optical processing."
)

DEGRADED_JPG_TEXT = IMAGE_JPG_TEXT

TABLE_ROWS = [
    ["Metric", "Baseline", "Proposed"],
    ["Precision", "0.82", "0.94"],
    ["Recall", "0.76", "0.89"],
    ["F1 Score", "0.79", "0.91"],
]

TABLE_GROUND_TRUTH = "\n".join(" | ".join(row) for row in TABLE_ROWS)


def _render_text_to_jpeg_bytes(
    text: str,
    *,
    dpi: int = 200,
    width: float = 595.0,
    height: float = 842.0,
    fontsize: float = 12.0,
    quality: int = 85,
) -> tuple[bytes, int, int]:
    """Render text onto an A4 page and return compressed grayscale
    JPEG bytes and dimensions.
    """
    doc = pymupdf.open()
    page = doc.new_page(width=width, height=height)
    margin = 54.0  # 0.75 in
    rect = pymupdf.Rect(margin, margin, width - margin, height - margin)
    page.insert_textbox(rect, text, fontsize=fontsize, fontname="helv")
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    w, h = pix.width, pix.height
    img = Image.frombytes("L", (w, h), pix.samples)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    doc.close()
    return buf.getvalue(), w, h


def _create_scanned_pdf(image_data_list: list[bytes], output_path: Path) -> None:
    """Create a compact PDF where each page embeds an image with deflated streams."""
    doc = pymupdf.open()
    for img_bytes in image_data_list:
        page = doc.new_page(width=595.0, height=842.0)
        page.insert_image(page.rect, stream=img_bytes)
    doc.save(str(output_path), deflate=True, garbage=4)
    doc.close()


def build_clean_single() -> None:
    img_bytes, _, _ = _render_text_to_jpeg_bytes(CLEAN_SINGLE_TEXT, dpi=200)
    _create_scanned_pdf([img_bytes], FIXTURES_DIR / "clean_single.pdf")
    (GROUND_TRUTH_DIR / "clean_single.txt").write_text(
        CLEAN_SINGLE_TEXT, encoding="utf-8"
    )


def build_multi_paragraph() -> None:
    img_bytes, _, _ = _render_text_to_jpeg_bytes(MULTI_PARAGRAPH_TEXT, dpi=200)
    _create_scanned_pdf([img_bytes], FIXTURES_DIR / "multi_paragraph.pdf")
    (GROUND_TRUTH_DIR / "multi_paragraph.txt").write_text(
        MULTI_PARAGRAPH_TEXT, encoding="utf-8"
    )


def build_headings_punct() -> None:
    img_bytes, _, _ = _render_text_to_jpeg_bytes(HEADINGS_PUNCT_TEXT, dpi=200)
    _create_scanned_pdf([img_bytes], FIXTURES_DIR / "headings_punct.pdf")
    (GROUND_TRUTH_DIR / "headings_punct.txt").write_text(
        HEADINGS_PUNCT_TEXT, encoding="utf-8"
    )


def build_scan_pdf_3page() -> None:
    img_list = [_render_text_to_jpeg_bytes(txt, dpi=200)[0] for txt in SCAN_3PAGE_TEXTS]
    _create_scanned_pdf(img_list, FIXTURES_DIR / "scan_pdf_3page.pdf")
    (GROUND_TRUTH_DIR / "scan_pdf_3page.txt").write_text(
        "\n\n".join(SCAN_3PAGE_TEXTS), encoding="utf-8"
    )


def build_mixed_pdf() -> None:
    doc = pymupdf.open()
    # Page 1: Native text
    p1 = doc.new_page(width=595.0, height=842.0)
    p1.insert_textbox(
        pymupdf.Rect(54, 54, 541, 788), MIXED_PDF_TEXTS[0], fontsize=12, fontname="helv"
    )

    # Page 2: Scanned image
    img2, _, _ = _render_text_to_jpeg_bytes(MIXED_PDF_TEXTS[1], dpi=200)
    p2 = doc.new_page(width=595.0, height=842.0)
    p2.insert_image(p2.rect, stream=img2)

    # Page 3: Native text
    p3 = doc.new_page(width=595.0, height=842.0)
    p3.insert_textbox(
        pymupdf.Rect(54, 54, 541, 788), MIXED_PDF_TEXTS[2], fontsize=12, fontname="helv"
    )

    # Page 4: Scanned image
    img4, _, _ = _render_text_to_jpeg_bytes(MIXED_PDF_TEXTS[3], dpi=200)
    p4 = doc.new_page(width=595.0, height=842.0)
    p4.insert_image(p4.rect, stream=img4)

    doc.save(str(FIXTURES_DIR / "mixed_pdf.pdf"), deflate=True, garbage=4)
    doc.close()

    (GROUND_TRUTH_DIR / "mixed_pdf.txt").write_text(
        "\n\n".join(MIXED_PDF_TEXTS), encoding="utf-8"
    )


def build_image_png() -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=595.0, height=842.0)
    page.insert_textbox(
        pymupdf.Rect(54, 54, 541, 788), IMAGE_PNG_TEXT, fontsize=12, fontname="helv"
    )
    pix = page.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
    pix.save(str(FIXTURES_DIR / "image_png.png"))
    doc.close()
    (GROUND_TRUTH_DIR / "image_png.txt").write_text(IMAGE_PNG_TEXT, encoding="utf-8")


def build_image_jpg() -> None:
    img_bytes, _, _ = _render_text_to_jpeg_bytes(IMAGE_JPG_TEXT, dpi=200, quality=75)
    (FIXTURES_DIR / "image_jpg.jpg").write_bytes(img_bytes)
    (GROUND_TRUTH_DIR / "image_jpg.txt").write_text(IMAGE_JPG_TEXT, encoding="utf-8")


def build_degraded_jpg() -> None:
    # 150 DPI and quality 35 to test quality degradation
    img_bytes, _, _ = _render_text_to_jpeg_bytes(DEGRADED_JPG_TEXT, dpi=150, quality=35)
    (FIXTURES_DIR / "degraded_jpg.jpg").write_bytes(img_bytes)
    (GROUND_TRUTH_DIR / "degraded_jpg.txt").write_text(
        DEGRADED_JPG_TEXT, encoding="utf-8"
    )


def build_table_fixtures() -> None:
    """Build simple ruled and borderless table fixtures."""
    for borderless in (False, True):
        doc = pymupdf.open()
        page = doc.new_page(width=595.0, height=842.0)

        # Draw 4x3 table
        start_x, start_y = 70.0, 100.0
        col_widths = [140.0, 110.0, 110.0]
        row_height = 30.0

        for r_idx, row in enumerate(TABLE_ROWS):
            y = start_y + r_idx * row_height
            curr_x = start_x
            for c_idx, cell in enumerate(row):
                w = col_widths[c_idx]
                rect = pymupdf.Rect(curr_x, y, curr_x + w, y + row_height)
                if not borderless:
                    page.draw_rect(rect, color=(0.2, 0.2, 0.2), width=1.0)
                # Centered vertically in cell
                text_point = pymupdf.Point(curr_x + 10, y + 20)
                page.insert_text(text_point, cell, fontsize=12, fontname="helv")
                curr_x += w

        pix = page.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
        img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        doc.close()

        suffix = "borderless" if borderless else "simple"
        _create_scanned_pdf([buf.getvalue()], FIXTURES_DIR / f"table_{suffix}.pdf")
        (GROUND_TRUTH_DIR / f"table_{suffix}.txt").write_text(
            TABLE_GROUND_TRUTH, encoding="utf-8"
        )


def build_all() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)

    build_clean_single()
    build_multi_paragraph()
    build_headings_punct()
    build_scan_pdf_3page()
    build_mixed_pdf()
    build_image_png()
    build_image_jpg()
    build_degraded_jpg()
    build_table_fixtures()
    print(f"Built all fixtures in {FIXTURES_DIR}")


if __name__ == "__main__":
    build_all()
