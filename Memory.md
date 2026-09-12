# MM-RAG — Project Memory

## Current Phase

**Phase 3 — Embeddings and Vector Store**

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
- Page-number tracking: exact boundary tracking in cleaned text (replaces proportional offset mapping, guaranteeing 100% provenance accuracy across multi-page documents)
- Metadata preservation: document_id, document_name, source_type, page_number carried through; original document metadata and page metadata preserved in chunk metadata
- Chunk metadata includes: char_count, chunk_size config, chunk_overlap config, and inherited document/page metadata
- Configuration: CHUNK_SIZE and CHUNK_OVERLAP added to backend Settings and .env.example with strict validation
- Hardening & Edge-Case Protection:
  - Exact page boundary tracking per cleaned page eliminates provenance misattribution caused by uneven whitespace collapse
  - Null bytes (`\x00`), byte-order mark (`\ufeff`), and zero-width spaces (`\u200b`) stripped in `clean_text` to protect downstream stores
  - Non-breaking spaces (`\u00a0`) normalized to standard spaces
  - Full metadata passthrough from `Document.metadata` and `PageContent.metadata` into `Chunk.metadata`
  - Strict model validation on `Chunk` (`min_length=1` on strings, `ge=1` on `page_number`, `ge=0` on `chunk_index`)
  - Strict field and cross-field validation in backend `Settings` for `chunk_size`, `chunk_overlap`, and `max_upload_size_mb`
- 184 total tests passing (104 Phase 2 tests, including 45 new hardening tests + 80 Phase 1 + Phase 0)
- Ruff lint and format checks pass (100% clean)

### Phase 3 — Embeddings and Vector Store

- **Embedding model**: `all-MiniLM-L6-v2` via `sentence-transformers`
  - 384 dimensions, ~80 MB, Apache 2.0
  - CPU-practical, fast inference, no GPU required
  - Selected for quality/size tradeoff; ideal for student project scope
- **Embedding interface**: Abstract `EmbeddingService` (ABC) in `rag/embeddings/base.py`
  - `embed_documents(texts)` — batch embedding
  - `embed_query(text)` — single query embedding
  - `dimension` property — embedding dimensionality
  - `model_name` property — model identifier
  - Replaceable: swap concrete class + config value, no downstream changes
- **Concrete implementation**: `SentenceTransformerEmbeddingService` in `rag/embeddings/sentence_transformer.py`
  - Lazy model loading (no startup cost)
  - Configurable model name and batch size
  - Input validation: empty list, non-string items, empty/whitespace strings
  - Error handling: model initialization failures → `RuntimeError`
  - Normalized embeddings (L2-normalized via sentence-transformers)
- **Vector store**: ChromaDB in `rag/vectorstore/chroma_store.py`
  - Native metadata storage per embedding
  - Built-in local persistence (directory-based)
  - Cosine similarity search
  - Deterministic collection naming
  - ID-based upsert (duplicate chunk IDs overwritten)
- **Vector store interface**: Abstract `VectorStore` (ABC) in `rag/vectorstore/base.py`
  - `add_chunks(chunks, embeddings)` — store with metadata
  - `query(query_vector, top_k)` → `list[VectorSearchResult]`
  - `delete_document(document_id)` — document-level deletion
  - `count()` — total stored vectors
  - `reset()` — clear all vectors
  - Replaceable: swap concrete class, no downstream changes
- **VectorSearchResult** model: chunk_id, document_id, document_name, content, score, metadata
- **Re-indexing strategy**: Document-level replacement
  - Delete all existing chunks for `document_id` before inserting new version
  - No stale chunks remain after re-processing
  - Deterministic behavior
- **IndexingService**: Chunk → Embed → Store orchestrator in `rag/embeddings/indexing.py`
  - Consumes Phase 2 `Chunk` model directly
  - Configurable re-indexing (`reindex=True` by default)
  - Batch embedding via EmbeddingService
- **Metadata preservation**: Every stored vector retains chunk_id, document_id, document_name, source_type, page_number, chunk_index, and custom chunk metadata
- **Configuration added**:
  - `EMBEDDING_MODEL=all-MiniLM-L6-v2`
  - `EMBEDDING_BATCH_SIZE=64`
  - `VECTOR_STORE_PATH=data/vectorstore`
  - `VECTOR_STORE_COLLECTION=mmrag_chunks`
  - `VECTOR_SEARCH_TOP_K=10`
  - Field validators for `embedding_batch_size` (>= 1) and `vector_search_top_k` (>= 1)
- **Security**: `data/` directory added to `.gitignore` (prevents committing vector store data, model caches)
- **Dependencies added**: `sentence-transformers>=3.0.0`, `chromadb>=0.5.0`
- **Hardening & Edge-Case Protection**:
  - Lazy model loading avoids import-time cost and failure
  - Empty/non-string input validation on all embedding methods
  - ChromaDB `PersistentClient` file lock handling in tests (`ignore_cleanup_errors=True` for Windows compatibility)
  - Score conversion: ChromaDB cosine distance → similarity (1.0 - distance)
  - `top_k` clamped to available count to avoid ChromaDB errors on empty/small stores
  - Flat metadata serialization for ChromaDB compatibility (only str/int/float/bool values)
- **257 total tests passing** (73 new Phase 3 tests + 184 Phase 1+2):
  - 28 embedding tests (interface, init, dimension, batch, query, unicode, invalid model)
  - 37 vector store tests (interface, init, add, query, metadata, delete, reindex, count, reset, persistence, collections)
  - 8 indexing integration tests (pipeline, reindex, metadata round-trip, semantic similarity)
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
| Embeddings | sentence-transformers | 6.0.x |
| Embedding Model | all-MiniLM-L6-v2 | — |
| Vector Store | ChromaDB | 1.5.x |
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
| Conservative text cleaning | Preserves semantic content; normalizes whitespace artifacts and sanitizes null bytes/BOM/zero-width spaces |
| Exact page-boundary tracking | Replaces proportional mapping to eliminate provenance drift across pages with uneven whitespace |
| Full metadata passthrough | Preserves document-level and page-level metadata in Chunk.metadata for downstream citation and retrieval |
| Backend Settings validation | Validates chunk size and overlap at config load time via Pydantic v2 model_validator |
| No new dependencies for Phase 2 | Only uses Pydantic (already installed) and Python stdlib |
| all-MiniLM-L6-v2 for embeddings | 384-dim, ~80 MB, CPU-friendly, Apache 2.0; best quality/size tradeoff for student project |
| sentence-transformers library | Clean API for batch/query embedding, normalization, well-maintained |
| ChromaDB for vector store | Native metadata, built-in persistence, ID-based ops, no infrastructure needed |
| Abstract EmbeddingService ABC | Allows model swap without downstream changes; future-proof for multilingual or larger models |
| Abstract VectorStore ABC | Allows backend swap (e.g. FAISS, Qdrant) without pipeline changes |
| Document-level re-indexing | Delete all chunks for document_id before insert; prevents stale data; deterministic |
| Lazy model loading | Avoids import-time cost; model loaded on first embed call |
| Cosine similarity in ChromaDB | Standard metric for normalized sentence embeddings |
| data/ in .gitignore | Prevents committing vector store data and model caches |

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
- Embedding model is English-optimized; multilingual support requires model swap
- Embedding model max sequence length is 256 tokens; chunks exceeding this are truncated
- ChromaDB not designed for massive scale (millions of vectors); acceptable for project scope

---

## Not Started

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

1. Begin Phase 4 — Basic Semantic Retrieval
2. Define retriever interface
3. Embed user query
4. Search vector store
5. Return top-K results with metadata

---

Last Updated: 2026-09-13
