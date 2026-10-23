# Phase 10 — OCR and Multimodal Processing Architecture & Evaluation

## 1. Overview & Architecture

Phase 10 introduces optical character recognition (OCR) and standalone image processing into the MM-RAG ingestion pipeline. The system can process scanned PDF documents, mixed-mode PDFs (containing both native digital text and scanned pages), and direct image uploads (`.png`, `.jpg`, `.jpeg`).

The implementation adheres to strict process isolation, CPU-practical execution, and explicit provenance tracking without introducing vision-language models or heavy runtime model downloads.

```
rag/ingestion/
├── ocr/
│   ├── __init__.py          # Exports OcrProcessor and TesseractEngine
│   ├── engine.py            # OcrEngine protocol & TesseractEngine subprocess adapter
│   ├── text_layout.py       # TSV parser (blocks, paragraphs, lines, words, confidence)
│   ├── page_analysis.py     # Heuristic: image coverage >= 30% and native text < 25 chars
│   └── processor.py         # OcrProcessor coordinating rendering bounds and execution
├── image_loader.py          # Standalone image loader (EXIF orientation, grayscale, bounds)
├── pdf_loader.py            # Enhanced PDF loader with selective per-page OCR fallback
├── detection.py             # MIME/magic detection extended for PNG and JPEG
├── validation.py            # File validation extended for .png, .jpg, .jpeg
└── service.py               # IngestionService orchestrating document loaders
```

---

## 2. Engine Design & Subprocess Adapter

- **Engine Choice**: Tesseract 5 (`tesseract v5.5.3.20260724` via UB-Mannheim Windows build).
- **Subprocess Isolation**: Tesseract is invoked as an isolated subprocess with explicit argument lists (no shell execution). Input PNG bytes are piped over `stdin`, and tab-separated values (TSV) are captured via `stdout`.
- **Layout Reconstruction**: Pure function `parse_tsv_to_text` reconstructs paragraphs and lines based on TSV `(block_num, par_num, line_num, word_num)` indices, skipping non-word entries (`conf == -1`) and calculating real mean confidence.
- **Hard Timeout**: `subprocess.run(timeout=...)` kills any hanging or long-running engine process, raising `OcrPageError` mapped to `NoExtractableTextError`.

---

## 3. Page Detection & Mixed-PDF Policy

### 3.1 Scanned Page Heuristic (`needs_ocr`)
A PDF page is identified as requiring OCR if and only if both conditions are met:
1. **Low Native Character Count**: The page text layer contains fewer than `OCR_MIN_NATIVE_CHARS` (default: 25) non-whitespace characters.
2. **Substantial Image Coverage**: The cumulative bounding area of raster images on the page covers at least 30% (`_MIN_IMAGE_COVERAGE = 0.30`) of the total page dimensions.

Blank pages (no text, no images) and sparse digital pages (e.g. "Page intentionally left blank") have 0% image coverage and do not trigger OCR. Pages with hidden digital text layers (e.g. pre-OCR'd PDFs) exceed the character threshold and bypass OCR.

### 3.2 Mixed-Mode Document Preservation
When a PDF contains both native digital text pages and scanned image pages:
- Native pages retain their exact digital text stream without optical processing (`extraction_method="native_text"`).
- Scanned pages are rendered to grayscale PNG and processed via Tesseract (`extraction_method="ocr"`).
- Digital and optical texts are never conflated or duplicated across pages.
- Per-page provenance (`page_number`, `extraction_method`, and `ocr_mean_confidence`) is stored in document and chunk metadata.

---

## 4. Error Handling & Outcome Matrix

| Condition | Detected Where | Behavior | HTTP Status |
|---|---|---|---|
| OCR disabled, image upload | `load_image` | `OcrUnavailableError` | 503 |
| Tesseract missing / language missing | `TesseractEngine` | `OcrUnavailableError` | 503 |
| Page needs OCR count > `OCR_MAX_PAGES` | `load_pdf` pre-scan | `OcrLimitExceededError` | 422 |
| Image pixels > `OCR_MAX_IMAGE_PIXELS` | `load_image` | `OcrLimitExceededError` | 422 |
| Corrupt / truncated image | Pillow in `load_image` | `InvalidDocumentError` | 422 |
| Unsupported format / magic mismatch | `detection` / `validation` | `UnsupportedFormatError` | 415 |
| All OCR pages fail or produce zero text | `load_pdf` / `load_image` | `NoExtractableTextError` | 422 |
| Partial OCR failure (>=1 usable page) | `load_pdf` | Partial success; failed pages in `ocr_failed_pages` | 200 |
| Blank native PDF (0 chunks, no OCR) | `IndexingService` | `DocumentIndexingError` (preserved legacy behavior) | 500 |

---

## 5. Metadata Schema

Downstream vector indexing and citation formatting store the following metadata keys:

- `source_type`: Set to `"pdf"`, `"docx"`, or `"image"`.
- `page_number`: 1-indexed page provenance.
- `extraction_method`: `"native_text"` for digital text layers, `"ocr"` for optically processed pages.
- `ocr_engine`: Engine identifier string (e.g. `"tesseract v5.5.3.20260724"`).
- `ocr_language`: Configured language code (e.g. `"eng"`).
- `ocr_dpi`: Render resolution used (e.g. `200` or `300`).
- `ocr_mean_confidence`: Mean word confidence score (0.0 to 100.0) from TSV output. Never fabricated or defaulted to 0 if absent.
- `ocr_page_count`: Number of pages processed via OCR in the document.
- `ocr_failed_pages`: List of 1-indexed page numbers where OCR failed.

---

## 6. Evaluation Framework & Measured Results

The evaluation suite (`evaluation/ocr_eval.py`) benchmarks quality and performance on reproducible synthetic fixtures in `evaluation/ocr/fixtures/` with exact matching ground truth in `evaluation/ocr/ground_truth/`.

> [!NOTE]
> **Synthetic Benchmark Caveat**: These benchmarks evaluate OCR quality and latency using clean, reproducible synthetic document and image fixtures (`evaluation/ocr/fixtures/`). They establish baseline engine behavior, layout parsing correctness, and relative latency trade-offs, but do not establish or represent accuracy on real-world, degraded, skewed, or low-contrast physical paper scans.

### 6.1 Evaluation Environment
- **Tesseract Version**: `tesseract v5.5.3.20260724`
- **PyMuPDF Version**: `1.28.2`
- **Pillow Version**: `13.0.0`
- **Platform**: `Windows 11 (AMD64)` / `Python 3.14.6`
- **Evaluation Date**: `2026-10-23`

### 6.2 Quality Benchmark (CER & WER)
Character Error Rate (CER) and Word Error Rate (WER) are computed using Levenshtein edit distance with normalization (NFC Unicode, space collapse, line-break normalization). Character case and punctuation are preserved.

| Fixture | 200 DPI CER | 200 DPI WER | 300 DPI CER | 300 DPI WER | Description |
|---|---|---|---|---|---|
| `clean_single` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Clean 12pt single paragraph (baseline) |
| `multi_paragraph` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3 paragraphs reading order |
| `headings_punct` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Headings, punctuation, numbers |
| `scan_pdf_3page` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3-page image-only PDF |
| `mixed_pdf` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Mixed native + scanned pages |
| `image_png` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Direct PNG image |
| `image_jpg` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Direct JPEG image (q=75) |
| `degraded_jpg` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Degraded JPEG (150 DPI, q=35) |
| `table_simple` | 0.1667 | 0.3810 | 0.1667 | 0.3810 | Ruled 4x3 table |
| `table_borderless` | 0.1667 | 0.3810 | 0.1667 | 0.3810 | Borderless 4x3 table |
| **Macro Average** | **0.0333** | **0.0762** | **0.0333** | **0.0762** | All 10 fixtures |

*Note on table CER & token presence*: The 0.1667 CER on table fixtures reflects the absence of Markdown pipe (`|`) delimiters in plain OCR text. 100% of the actual table cell words and numerical values were recognized without errors (`token_presence_rate = 1.0`, 13/13 tokens present across all 12 cells).

### 6.3 Per-Page Latency Benchmark (`clean_single`, 5 trials)

| Method | Render Mean | Engine Mean | Total Mean | Total Median |
|---|---|---|---|---|
| **Native text** | — | — | **0.14 ms** | **0.11 ms** |
| **OCR 200 DPI** | 96.83 ms | 330.32 ms | **427.16 ms** | **428.04 ms** |
| **OCR 300 DPI** | 242.42 ms | 545.55 ms | **787.98 ms** | **790.72 ms** |

### 6.4 DPI Selection Analysis
- **Quality**: Across all clean printed text fixtures (including degraded 150 DPI JPEG), 200 DPI and 300 DPI both achieved **0.0000 CER and 0.0000 WER**.
- **Latency**: 200 DPI is **1.84x faster** than 300 DPI (427 ms vs 788 ms total per page).
- **Trade-off**: 300 DPI is retained as the safe default for documents with small fonts (<9pt) or noisy scans. However, 200 DPI (`OCR_DPI=200`) is recommended for production environments requiring high ingestion throughput.

---

## 7. Table Extraction Assessment

Table extraction readability was assessed using the 4×3 ruled table (`table_simple.pdf`) and borderless table (`table_borderless.pdf`):
- **Page Segmentation Modes (PSM)**:
  - **PSM 3 (Fully automatic segmentation, adopted default)**: Preserved all 4 table rows on distinct lines with a 100% token-presence match rate (`token_presence_rate = 1.0`, all 13 constituent tokens found across 12 cells).
  - **PSM 4 (Single column of variable sizes)**: Performed identically to PSM 3 on table fixtures.
  - **PSM 6 (Single uniform block)**: Completely failed on the ruled table (`CER = 1.0`, 0 lines returned) due to gridline interference. PSM 3 is demonstrably more robust.
- **Word Gap Heuristic**: An experimental heuristic inserting ` | ` between words whose horizontal gap exceeded 2× median gap was tested. It did not reliably distinguish intra-cell word spaces from inter-column gaps on small tables and risked corrupting plain text. The heuristic was **not adopted**.
- **Conclusion**: Standard PSM 3 text layout preserves row structure with space-separated cells suitable for character chunking. Advanced structured table retrieval remains designated for future scope.

---

## 8. Limitations & Future Scope

1. **Language Scope**: Default deployment includes English (`eng`) language data. Multilingual documents require installing the corresponding `*.traineddata` files and configuring `OCR_LANGUAGE`.
2. **Layout Complexity**: Multi-column magazine layouts, complex forms, and rotated tables are processed in reading order determined by Tesseract's layout analyzer. No visual document understanding (VLM) or bounding-box graph modeling is performed.
3. **Hardware Acceleration**: OCR runs on the CPU via Tesseract. High-volume document scanning can be bound by CPU core limits; per-document page limits (`OCR_MAX_PAGES`) and timeouts (`OCR_TIMEOUT_SECONDS`) protect system responsiveness.
