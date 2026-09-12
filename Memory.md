# MM-RAG — Project Memory

## Current Phase

**Phase 2 — Normalization and Chunking**

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

### Phase 2 — Normalization and Chunking

- Chunk model (Pydantic): `Chunk` with chunk_id, document_id, document_name, source_type, content, page_number, chunk_index, metadata
- Chunk ID generation (UUID4, isolated in `chunk_id` module — mirrors `document_id` pattern)
- Text cleaning / normalization:
  - Line ending normalization (CRLF, CR → LF)
  - Tab-to-space conversion
  - Per-line whitespace stripping
  - Horizontal whitespace collapsing (2+ → single space)
  - Excessive blank line collapsing (3+ newlines → double newline, preserving paragraph breaks)
  - Full-text leading/trailing whitespace stripping
  - Conservative: no lowercasing, no punctuation removal, no stop-word removal
- Chunking strategy: fixed-size character chunking with configurable overlap
  - Defaults: chunk_size=1000, chunk_overlap=200
  - Validation: chunk_size > 0, overlap >= 0, overlap < chunk_size
  - Deterministic behavior
  - Infinite-loop guard (step always >= 1)
  - Handles: empty text, whitespace-only, short text, exact boundary, very long text
- Page-number tracking: proportional mapping from cleaned-text offset to raw page boundaries
- Metadata preservation: document_id, document_name, source_type, page_number carried through
- Chunk metadata includes: char_count, chunk_size config, chunk_overlap config
- Configuration: CHUNK_SIZE and CHUNK_OVERLAP added to backend Settings and .env.example
- 59 new Phase 2 tests + 80 existing = 139 total tests passing
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
| UUID4 for chunk IDs | Same strategy as document IDs; isolated in `chunk_id` module for future swap |
| Fixed-size character chunking | Simple, deterministic, no external NLP dependencies; replaceable later |
| 1000 char chunk_size default | ~200-250 words; fits typical embedding model context windows |
| 200 char chunk_overlap default | Enough to avoid mid-sentence breaks at boundaries |
| Conservative text cleaning | Preserves semantic content; only normalizes whitespace artifacts |
| Proportional page-number mapping | Handles cleaned-text offset drift; assigns chunk to starting page |
| No new dependencies for Phase 2 | Only uses Pydantic (already installed) and Python stdlib |

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
- Chunking is character-based only; no sentence-aware or semantic chunking
- Page-number assignment uses proportional offset mapping, which may be slightly approximate when cleaning changes text length significantly

---

## Not Started

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

1. Begin Phase 3 — Embeddings and Vector Store
2. Define embedding interface
3. Select initial embedding model
4. Implement embedding generation for chunks
5. Select and integrate vector store

---

Last Updated: 2026-09-12
