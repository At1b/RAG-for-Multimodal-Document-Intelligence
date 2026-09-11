# MM-RAG — Project Memory

## Current Phase

**Phase 1 — Document Ingestion**

Status: **COMPLETED**

---

## Completed Work

### Phase 0 — Project Foundation

- Git repository initialized
- GitHub remote configured (origin → At1b/RAG-for-Multimodal-Document-Intelligence)
- Project directory structure created (backend/, frontend/, rag/, evaluation/, tests/, docs/)
- Backend skeleton created (FastAPI entry point with /health endpoint)
- Backend configuration module (pydantic-settings, environment variables)
- Backend dependency file (requirements.txt)
- Frontend skeleton created (React via Vite)
- Frontend builds successfully
- Test structure created (pytest + 3 foundation tests)
- All tests pass (3/3)
- Formatting/linting configured (ruff)
- All code passes lint and format checks
- CI workflow created (.github/workflows/ci.yml)
- .gitignore, .env.example, README.md, LICENSE created
- PRD.md, Architecture.md, Rules.md, Phases.md present

### Phase 1 — Document Ingestion

- Normalized Document model (Pydantic): Document, PageContent
- Document ID generation (UUID4, isolated in document_id module)
- File validation: exists, not empty, size limit, extension check
- Format detection: magic bytes + extension (PDF %PDF, DOCX PK zip)
- PDF loader (PyMuPDF): page-by-page text extraction, page number preservation
- DOCX loader (python-docx): paragraph + table extraction
- IngestionService orchestrator: validate → detect → load → Document
- API endpoint: POST /documents/upload
- Error handling: custom exception hierarchy mapped to HTTP status codes
- Configurable max upload size (default 50 MB)
- Hardening & Edge-Case Protection:
  - Stream/BytesIO loaders preventing OS file descriptor leaks and Windows file locking (`PermissionError`)
  - Early encrypted/password-protected PDF rejection (`InvalidDocumentError`)
  - Zero-page / truncated PDF validation
  - Empty table row filtering in DOCX loader (prevents orphaned `" | "` lines)
  - Symmetric format detection preventing cross-format spoofing
  - Memory-safe chunked upload streaming with early size limit abort
  - Early empty/whitespace filename validation (HTTP 400)
- 80 tests pass (27 new edge-case tests + 50 Phase 1 + 3 Phase 0)
- Ruff lint and format checks pass (100% clean)

---

## Technology Stack (Implemented)

| Component | Technology | Version |
|---|---|---|
| Backend Language | Python | 3.14.6 |
| API Framework | FastAPI | 0.141.x |
| ASGI Server | Uvicorn | 0.52.x |
| Configuration | pydantic-settings | 2.15.x |
| PDF Processing | PyMuPDF | 1.28.x |
| DOCX Processing | python-docx | 1.2.x |
| File Upload | python-multipart | 0.0.32 |
| Frontend | React (Vite) | Vite 8.x |
| Node.js | Node.js | 24.17.0 |
| Testing | Pytest | 9.1.x |
| Linting/Formatting | Ruff | 0.16.x |
| CI | GitHub Actions | — |

---

## Important Technical Decisions

| Decision | Reason |
|---|---|
| FastAPI for backend | Documented in PRD and Architecture as the API framework |
| React via Vite for frontend | React documented in PRD; Vite chosen as simplest modern scaffold |
| Ruff for linting/formatting | Single tool replaces flake8+isort+black; fast, simple, CI-friendly |
| pydantic-settings for config | Type-safe env config, integrates naturally with FastAPI/Pydantic |
| MIT License | Common for open-source student projects; can be changed by team |
| Pytest for testing | Documented in PRD as the test framework |
| Pydantic for Document model | Already a dependency; provides validation, serialization, clean interfaces |
| UUID4 for document IDs | Simple, unique, no filesystem path exposure; isolated for future change |
| PyMuPDF for PDF processing | Lightweight, fast, no Java dependency; recommended in PRD |
| python-docx for DOCX | Standard library for DOCX reading; recommended in PRD |
| Magic bytes + extension for format detection | More reliable than extension alone; no external dependencies |
| DOCX as single logical page | DOCX lacks native page boundaries; chunking handles splitting in Phase 2 |
| 50 MB default upload limit | Reasonable for document processing; configurable via Settings |
| Stream / BytesIO loader reading | Prevents C library / zipfile OS handle locks on Windows during temp cleanup |
| Chunked upload streaming | Avoids reading massive uploads into memory before size validation |

---

## Known Issues

- None at present

---

## Known Limitations

- Frontend is the default Vite/React template; no MM-RAG-specific UI yet
- OCR is not implemented (deferred to Phase 10)
- Scanned PDFs will extract no text (text extraction only, no image-based OCR)
- DOCX does not preserve page boundaries (entire content as single page)
- Tables in DOCX are extracted as pipe-separated plain text
- No document persistence/storage — in-memory processing only
- No database — documents are processed and returned, not stored

---

## Not Started

- Text normalization and chunking (Phase 2)
- Embeddings and vector store (Phase 3)
- Semantic retrieval (Phase 4)
- LLM generation (Phase 5)
- End-to-end baseline RAG (Phase 6)
- Multi-document support and citations (Phase 7)
- Hybrid retrieval (Phase 8)
- Reranking (Phase 9)
- OCR and multimodal processing (Phase 10)
- Evaluation framework (Phase 11)

---

## Next Immediate Tasks

1. Begin Phase 2 — Normalization and Chunking
2. Implement text cleaning/normalization
3. Implement chunking strategy with configurable size and overlap
4. Preserve metadata through chunking
5. Generate unique chunk IDs

---

Last Updated: 2026-09-11
