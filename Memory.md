# MM-RAG — Project Memory

## Current Phase

**Phase 7 — Multi-Document Support and Citations**

Status: **NOT_STARTED** (Phase 6 is COMPLETED)

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
  - Embeddings generated *before* deletion to guarantee atomicity and prevent data loss on embedding failure
  - No stale chunks remain after re-processing
  - Deterministic behavior
- **IndexingService**: Chunk → Embed → Store orchestrator in `rag/embeddings/indexing.py`
  - Consumes Phase 2 `Chunk` model directly
  - Configurable re-indexing (`reindex=True` by default)
  - Batch embedding via EmbeddingService
- **Metadata preservation**: Every stored vector retains chunk_id, document_id, document_name, source_type, page_number, chunk_index, and custom chunk metadata
- **Configuration added & hardened**:
  - `EMBEDDING_MODEL=all-MiniLM-L6-v2` (non-empty string validation)
  - `EMBEDDING_BATCH_SIZE=64` (>= 1 validation)
  - `VECTOR_STORE_PATH=data/vectorstore` (non-empty path validation)
  - `VECTOR_STORE_COLLECTION=mmrag_chunks` (3-512 chars, alphanumeric boundary validation)
  - `VECTOR_SEARCH_TOP_K=10` (>= 1 validation)
- **Security**: `data/`, `*.sqlite3`, `*.sqlite` added to `.gitignore` (prevents committing vector store data and model caches)
- **Dependencies added**: `sentence-transformers>=3.0.0`, `chromadb>=0.5.0`
- **Hardening & Edge-Case Protection**:
  - Lazy model loading avoids import-time cost and failure
  - Empty/non-string and empty/whitespace string validation on all embedding methods
  - Strict type checking on `model_name` and `batch_size`
  - Order preservation and determinism across multiple batch encoding passes verified
  - ChromaDB `PersistentClient` file lock handling in tests (`ignore_cleanup_errors=True` for Windows compatibility)
  - In-batch duplicate chunk ID deduplication in `add_chunks` (latest occurrence supersedes earlier ones, order preserved, warning logged, preventing ChromaDB `DuplicateIDError`)
  - Batch embedding dimension mismatch and empty vector validation in `add_chunks`
  - Chunk ID explicitly preserved in `VectorSearchResult.metadata["chunk_id"]`
  - Non-finite float metadata (`math.isnan`, `math.isinf`) filtered out to prevent index corruption
  - Query similarity score clamped to `[-1.0, 1.0]`
  - Query results safely unpack `None` values in Chroma response dictionaries
  - Atomic re-indexing ordering: `embed_documents` executed before `delete_document` in `IndexingService.index_chunks` so failures never wipe existing collections
  - Idempotent collection reset protecting against `NotFoundError`
  - Direct primary-key deletion via `ids=existing["ids"]` in `delete_document`
- **298 total tests passing** (114 Phase 3 tests + 184 Phase 1+2):
  - 34 embedding tests (interface, init, dimension, batch, query, unicode, invalid model, empty/whitespace strings, multi-batch determinism)
  - 50 vector store tests (interface, init, add, in-batch dupes, dimension mismatch, query, metadata, delete, reindex, count, reset, persistence, collections, nan/inf safety, score clamping)
  - 11 indexing integration tests (pipeline, atomic reindex safety, metadata round-trip, semantic similarity, type validation)
  - 19 configuration tests (defaults, custom, field validators for model, path, collection naming rules, batch size, top_k)
- Ruff lint and format checks pass (100% clean, 41 files formatted)

### Phase 4 — Basic Semantic Retrieval (Hardened)

- **Status**: Completed & Hardened
- **Retriever interface**: Abstract `Retriever` (ABC) in `rag/retrieval/base.py`
  - `retrieve(query, top_k)` → `list[VectorSearchResult]`
  - `top_k` optional; defaults to configurable value when `None`
  - Replaceable: swap concrete class for future retrieval strategies (hybrid, reranking)
- **Concrete implementation**: `SemanticRetriever` in `rag/retrieval/semantic.py`
  - Thin composable layer delegating to existing Phase 3 services
  - Flow: Query Validation & Sanitization → `EmbeddingService.embed_query()` → `VectorStore.query()` → Top-K `VectorSearchResult`
  - No embedding or vector-store logic duplicated
- **Result model decision**: Reuses Phase 3 `VectorSearchResult` directly
  - `VectorSearchResult` already preserves: chunk_id, document_id, document_name, content, score, metadata
  - Metadata contains: source_type, page_number, chunk_index, and custom metadata
  - No new result model needed — avoids unnecessary duplication
- **Query validation & sanitization**:
  - Empty string → `InvalidQueryError`
  - Whitespace-only → `InvalidQueryError`
  - Non-string types → `InvalidQueryError`
  - Invisible characters / null-byte only inputs (`\x00`, `\ufeff`, `\u200b`) → `InvalidQueryError`
  - Exceeds `MAX_QUERY_LENGTH` (10,000 chars) → `InvalidQueryError`
  - Unicode text → fully supported (CJK, Arabic, emojis, accented characters)
  - Null bytes in valid queries stripped before embedding
- **top_k behavior & resource safety**:
  - Caller-provided `top_k` overrides default
  - Default from `default_top_k` constructor parameter (maps to `VECTOR_SEARCH_TOP_K` setting)
  - Validated: must be int ≥ 1, rejects bool/float/zero/negative/string
  - Upper bound: `MAX_TOP_K = 1000` enforced at runtime, constructor, and Settings level
  - Invalid top_k raises `InvalidTopKError` (subclass of `InvalidQueryError`)
  - Fewer results returned naturally when collection is smaller than top_k
  - Empty vector store returns empty list (no fabricated results)
- **Ordering and scores**: Results preserve vector store ordering (cosine similarity, descending); scores passed through unchanged
- **Exception hierarchy**: `rag/retrieval/exceptions.py`
  - `RetrievalError` (base)
  - `InvalidQueryError` — invalid user query
  - `InvalidTopKError(InvalidQueryError)` — invalid top_k parameter
  - `EmbeddingError` — embedding service failure or malformed vector (wraps cause)
  - `VectorStoreError` — vector store search failure or malformed results (wraps cause)
  - Follows same pattern as `rag/ingestion/exceptions.py`
- **Hardening improvements (Secondary Review)**:
  - Fixed unbounded `default_top_k` vulnerability: enforced `1 <= default_top_k <= MAX_TOP_K` in `SemanticRetriever.__init__`
  - Fixed unbounded `Settings.vector_search_top_k`: added `v <= 1000` validator in `backend/config.py`
  - Added query sanitization against null bytes, BOM, and zero-width spaces; rejects invisible-only inputs
  - Added component boundary vector validation in `_embed_query` (validates non-empty list of finite numbers, correctly attributes embedding anomalies to `EmbeddingError`)
  - Added component boundary result validation in `_search` (verifies vector store returned a list)
  - Introduced `InvalidTopKError` inheriting from `InvalidQueryError` for parameter-specific error handling while preserving backward compatibility
- **No new dependencies added**
- **No new configuration values**: Reuses existing `VECTOR_SEARCH_TOP_K` from Phase 3 with upper bound enforced
- **No API endpoint added**: Phase 4 is retrieval layer only (API deferred to Phase 6)
- **383 total tests passing** (83 Phase 4 tests + 184 Phase 1–2 + 116 Phase 3):
  - 3 interface tests (ABC contract, subclass, method existence)
  - 8 constructor validation tests (type checks, default_top_k zero/negative/bool, upper bound rejection, max bound acceptance)
  - 14 query validation tests (empty, whitespace, non-string, unicode, long, max-length, valid, null-only, zero-width-only, BOM-only, null-sanitized)
  - 11 top_k tests (default, override, one, zero, negative, bool, float, max, at-max, fewer-results, invalid_top_k_error)
  - 4 basic retrieval tests (returns list, result types, query vector passing, multiple results)
  - 9 metadata preservation tests (chunk_id, document_id, document_name, content, score, source_type, page_number, chunk_index, custom metadata)
  - 3 ordering/scores tests (preserve ordering, floats, near-equal scores)
  - 2 empty vector store tests (empty list, no fabrication)
  - 11 dependency failure tests (embedding error, vector store error, cause preservation, empty vector, none vector, nan vector, inf vector, non-list vector store result)
  - 1 multiple documents test (cross-document retrieval)
  - 6 exception hierarchy tests (subclass relationships including InvalidTopKError)
  - 11 integration tests (real SentenceTransformer + ChromaDB: relevant query, unrelated query scores, top_k limits, top_k > collection, metadata round-trip, ordering, multi-doc, unicode, empty store, float scores, custom metadata)
  - 21 configuration tests (defaults, custom, field validators for model, path, collection naming rules, batch size, top_k bounds)
- Ruff lint and format checks pass (100% clean, 46 files formatted)

### Phase 5 — LLM Generation

- **Status**: Completed
- **Generator interface**: Abstract `Generator` (ABC) in `rag/generation/base.py`
  - `generate(question, context)` → `GenerationResult`
  - Accepts question (str) and context (list[VectorSearchResult])
  - Replaceable: swap concrete class for future LLM backends (OpenAI, Anthropic, local transformers)
- **Concrete implementation**: `OllamaGenerator` in `rag/generation/ollama_generator.py`
  - Uses `ollama` Python SDK (v0.6.2) to communicate with locally running Ollama server
  - Lazy client initialization (no import-time cost)
  - Connection validated on first `generate` call
  - Configurable: model name, temperature (0.0–2.0), max tokens, base URL, context max chars
  - Uses `client.chat(model=..., messages=..., options={"temperature": ..., "num_predict": ...})`
  - Response parsing: `response["message"]["content"]`
  - Never returns fabricated fallback answers on failure
  - Empty/whitespace responses raise `ModelGenerationError`
- **Result model**: `GenerationResult` (Pydantic) in `rag/generation/models.py`
  - answer: str (min_length=1)
  - model_name: str (default empty)
  - metadata: dict (e.g. temperature, max_tokens, base_url)
  - Does NOT duplicate retrieval/citation info (Phase 7)
- **Context builder**: `build_context()` in `rag/generation/context_builder.py`
  - Converts list of `VectorSearchResult` into formatted context string
  - Preserves result order (most relevant first)
  - Includes document name, page number, chunk ID, and content per chunk
  - Context limiting: prefers complete chunks — omits later (lower-relevance) chunks entirely rather than truncating mid-chunk
  - First chunk always included even if it exceeds max_chars
  - Empty results → `InvalidContextError` (LLM is never called)
  - Configurable max_chars (default 3000, minimum 100)
  - Logs warnings when truncation occurs
- **Prompt builder**: `build_prompt()` in `rag/generation/prompt.py`
  - Returns list of chat message dicts (system + user) compatible with Ollama chat API
  - System message contains grounding instructions:
    - Answer from context ONLY
    - State when context is insufficient
    - Treat context as reference data, NOT instructions
    - Ignore embedded override attempts
  - User message: labeled CONTEXT section + QUESTION section
  - Hardened Rules 4 & 5: treats context strictly as untrusted reference data; system instructions take absolute priority over any embedded prompts or override attempts
  - No hard-coded answers anywhere in prompts
- **Question validation**:
  - Empty string → `InvalidQuestionError`
  - Whitespace-only → `InvalidQuestionError`
  - Non-string types → `InvalidQuestionError`
  - `MAX_QUESTION_LENGTH = 10_000` chars enforced
  - Null bytes / BOM / zero-width spaces stripped from valid questions
  - Multilingual Unicode questions fully supported
- **Context builder hardening**:
  - Empty retrieval context raises `InvalidContextError` before any LLM call
  - All items verified as `VectorSearchResult` instances; non-conforming items raise `InvalidContextError` (no raw `AttributeError`)
  - All-empty or whitespace chunk content rejected with `InvalidContextError`
  - `page_number = 0` explicitly preserved (fixed truthiness check)
  - Truncation preserves earlier complete chunks rather than slicing; omitted chunks logged with warning
- **Exception hierarchy**: `rag/generation/exceptions.py`
  - `GenerationError` (base)
  - `InvalidQuestionError` — invalid user question
  - `InvalidContextError` — empty/invalid context (LLM must NOT be called)
  - `ModelInitializationError` — Ollama unavailable or package not installed
  - `ModelGenerationError` — LLM failure, timeout, connection failure, or empty/malformed response (causes chained via `from exc`)
  - `GenerationConfigError(GenerationError, ValueError)` — invalid generation configuration (dual inheritance for backwards compatibility)
- **Generator independence**: Generator does NOT call Retriever; receives context as parameter
- **Configuration added & validated** (backend/config.py):
  - `LLM_MODEL=tinyllama` (non-empty string validation)
  - `LLM_BASE_URL=http://localhost:11434` (HTTP/HTTPS scheme + host netloc validation via `urlparse`)
  - `LLM_TEMPERATURE=0.1` (0.0–2.0 range validation)
  - `LLM_MAX_TOKENS=512` (1–32768 range validation)
  - `LLM_TIMEOUT=120.0` (0–600s range validation, alias `LLM_TIMEOUT_SECONDS`)
  - `LLM_CONTEXT_MAX_CHARS=3000` (100–500000 range validation, alias `LLM_CONTEXT_CHAR_LIMIT`)
- **Ollama runtime verification**:
  - Ollama runtime (v0.34.2) verified running on local Windows environment
  - `tinyllama:latest` (1.1B parameters, 637 MB) verified installed and ready
  - Ollama Python SDK 0.6.2 installed and verified
  - `Client(host=..., timeout=...)` constructor verified
  - `client.chat(model, messages, options)` API verified with dict and object response parsing
  - Real generation smoke test executed against local Ollama runtime and passed
- **Ollama setup documented in README.md**: install, serve, pull instructions, and environment variables
- **Dependencies added**: `ollama>=0.4.0`
- **Security**: Prompt injection attempts in retrieved context strictly contained in user context; no API keys required (local LLM); no secrets in configuration
- **467 total tests passing** (84 Phase 5 tests + 383 Phase 0–4):
  - 3 interface tests (ABC contract, subclass, method existence)
  - 10 prompt builder tests (question/context inclusion, system/user roles, grounding instructions, no hardcoded answers, override prevention, context/question labels, insufficient context instruction)
  - 12 context builder tests (content, document name, page number, chunk ID, multiple docs, empty/none results, ordering, context limit truncation, first chunk kept, max_chars minimum)
  - 15 OllamaGenerator tests (successful generation, init failure, generation failure, empty response, invalid question/whitespace/non-string, empty context no LLM call, metadata, invalid model/temperature/max_tokens/context_max_chars/base_url)
  - 2 integration tests (full pipeline mock, empty context blocks generation)
  - 3 exception hierarchy tests (inheritance, base class, chaining)
  - 8 configuration tests (defaults, custom values, invalid base_url/empty base_url/temperature/negative temp/max_tokens/context_max_chars/empty model)
  - 3 prompt injection security tests (system override attempt, persona hijacking, grounding preservation)
  - 4 timeout & connection handling tests (configured timeout, invalid timeout, timeout wrapping with cause, connection refused with cause)
  - 6 malformed response tests (missing message, non-dict message, missing content, non-string content, whitespace content, ChatResponse object support)
  - 5 context builder hardening tests (non-VectorSearchResult items, all-empty chunks, page 0 preservation, multi-chunk truncation logging, long context bounding)
  - 4 question validation hardening tests (max length, Unicode multilingual, BOM/zero-width space sanitization, only zero-width space rejection)
  - 6 extended configuration tests (llm_timeout setting, timeout bounds, context limit alias, missing host URL rejection, generator base URL validation, config error hierarchy)
  - 2 generator independence tests (no retriever/vectorstore coupling, custom generator implementation)
  - 1 real Ollama generation smoke test (live TinyLlama generation against local server)
- Ruff lint and format checks pass (100% clean, 54 files formatted)

### Phase 6 — End-to-End Baseline RAG (Completed)

- **Status**: Completed (Parts A & B)
- **Part A — Core RAG Orchestration**:
  - Orchestration package: `rag/orchestration/` — application-level service layer
  - Document indexing service: `DocumentIndexingService` in `rag/orchestration/indexing_service.py`
    - Orchestrates: `IngestionService.ingest()` → `chunk_document()` → `IndexingService.index_chunks()`
    - Constructor-injected dependencies: `IngestionService`, `ChunkingConfig`, `IndexingService`
    - Method: `index_document(file_path, document_name=None)` → `IndexingResult`
    - `IndexingResult` model: `document_id`, `document_name`, `num_pages`, `num_chunks`
    - Error handling: all pipeline failures wrapped in `DocumentIndexingError` with cause preserved
    - Zero chunks after chunking raises `DocumentIndexingError`
    - No duplicated processing — delegates entirely to existing Phase 1–3 abstractions
  - RAG query service: `RAGQueryService` in `rag/orchestration/query_service.py`
    - Orchestrates: `Retriever.retrieve()` → `Generator.generate()` → `QueryResult`
    - Constructor-injected dependencies: `Retriever` (abstract), `Generator` (abstract)
    - Method: `query(question, top_k=None)` → `QueryResult`
    - `QueryResult` model: `answer`, `model_name`, `num_chunks_retrieved`, `metadata`
    - Empty retrieval → `EmptyRetrievalError` (LLM is NOT called)
    - Invalid question errors pass through directly from retriever
    - Retrieval/generation failures wrapped in `QueryError` with cause preserved
    - Independent of FastAPI / HTTP request-response objects
  - Exception hierarchy: `rag/orchestration/exceptions.py`
    - `OrchestrationError` (base)
    - `DocumentIndexingError` — indexing pipeline failure
    - `QueryError` — query pipeline failure
    - `EmptyRetrievalError(QueryError)` — no usable retrieval context
- **Part B — API Integration & Verification**:
  - **POST /documents/upload** updated to full indexing:
    - Delegates to `DocumentIndexingService`
    - Document is ingested, chunked, embedded, and indexed in ChromaDB in a single operation
    - Returns `document_id`, `document_name`, `num_pages`, `num_chunks`
    - `_resolve_indexing_error` helper unwraps `DocumentIndexingError.__cause__` to preserve correct HTTP status codes (`EmptyFileError` → 400, `FileTooLargeError` → 413, `UnsupportedFormatError` → 415, `InvalidDocumentError` → 422, generic → 500)
  - **POST /query** endpoint implemented in `backend/query.py`:
    - Thin route handler: delegates to `RAGQueryService`
    - Accepts `question` (min_length=1) and optional `top_k` (ge=1, le=1000)
    - `_resolve_query_error` helper unwraps `QueryError.__cause__` to map validation errors to HTTP 400 (`InvalidQueryError`, `InvalidQuestionError` → 400, `EmptyRetrievalError` → 404, generic `QueryError` → 500)
    - Returns `QueryResponse(answer, model_name, num_chunks_retrieved)`
  - **Thin route architecture**:
    - Route handlers only construct service dependencies from settings, call orchestration services, and format responses/errors
    - No business logic, embedding math, ChromaDB queries, or LLM chat calls inside route handlers
  - **Real sample document verification**:
    - Fixture created: `tests/fixtures/sample_ai_overview.pdf` (multi-paragraph AI/RAG/Transformers overview)
    - Integration tests in `tests/test_sample_document.py` verify full flow: ingestion → multi-chunk creation → embedding → ChromaDB storage → semantic query retrieval → live LLM answer generation
    - Optional/skip-aware live LLM test against local Ollama runtime (`@pytest.mark.skipif`)
  - **End-to-end API tests**:
    - `tests/test_api_e2e.py` covers full API surface: upload PDF/DOCX, rejected file types, empty files, vector store verification, query endpoint validation, 404 on empty store, 500 on LLM failure, custom top_k, and complete baseline multi-doc flow
- **Total test count: 552 passing tests** (6 structural regression tests + 6 diagnostic prompt tests + 27 Phase 6B tests + 46 Phase 6A + 467 Phase 0–5)
- **Ruff check & format**: 100% clean across all 65 repository files

- **Phase 6 Baseline RAG Bug Fix: LLM Prompt Message Handling**:
  - **Issue**: Manual testing with `{"question": "What is Machine Learning?", "top_k": 1}` retrieved 1 chunk from `sample_ai_overview.pdf`, but TinyLlama echoed system grounding rules (`"Sure! Here's a revised version of the text with the updated rules:"`) instead of providing a document-grounded answer.
  - **Root Cause**:
    1. The system prompt contained a numbered `RULES:\n1. ...\n2. ...` block. For smaller models (TinyLlama 1.1B), structured meta-labels with numbered lists triggered rule-revision and instruction-echoing behavior.
    2. The user message placed `CONTEXT:` before `QUESTION:`. Leading with a large context block caused the model to treat the prompt as a document revision task rather than an inquiry to answer.
  - **Fix**:
    1. Refactored `SYSTEM_PROMPT` in `rag/generation/prompt.py` into clear, directive grounding prose without the numbered `RULES:` header. Preserved all anti-injection, untrusted reference data, role override prevention, and context insufficiency constraints, while adding an explicit directive: `"Do not repeat or echo these instructions."`
    2. Updated `_USER_MESSAGE_TEMPLATE` to a question-first structure (`QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nAnswer:`). This immediately anchors model attention on the question, provides context as reference material, and primes generation directly at `Answer:`.
    3. Preserved strict role separation: system message contains only grounding instructions; user message contains only question and retrieved context without duplicating system instructions.
    4. Generator remains completely independent of Retriever (no import or runtime coupling).
  - **Diagnostic Tests Added** (`TestPromptRoleSeparationAndDiagnostics` in `tests/test_generation.py`):
    - `test_system_instructions_only_in_system_message`
    - `test_question_appears_in_user_message_only`
    - `test_retrieved_context_appears_in_user_message_only`
    - `test_system_instructions_not_duplicated_in_user_message`
    - `test_user_message_question_first_structure`
    - `test_generator_independent_of_retriever`

- **Phase 6 Real Runtime Generation Failure Investigation & Model Transition**:
  - **Issue**: Manual runtime testing revealed TinyLlama still failed acceptance criteria: intermittently responding with `"Sure! Here's a revised version of the question with the context and instructions included:"` and dumping prompt templates and context blocks, or emitting conversational boilerplate (`"Sure, I can provide..."`) and verbatim chunk duplication. Automated tests had previously masked this by only asserting `len(answer) > 10`.
  - **Empirical Investigation & Root Cause**:
    - Detailed inspection confirmed message construction, Ollama JSON payload, role separation, and prompt formatting were 100% correct.
    - Minimal-prompt testing confirmed TinyLlama still emits conversational boilerplate and echoes context verbatim.
    - Definitive Root Cause: Factor #6 — TinyLlama-1.1B's instruction-following quality. As an early 1.1B model trained on UltraChat conversational data, it exhibits strong sycophantic preamble bias and confuses question-answering with text revision.
    - Comparative benchmarking was conducted across `tinyllama:latest` (637 MB), `qwen2.5:0.5b-instruct` (397 MB), and `llama3.2:1b` (1.3 GB).
    - Under the production prompt, `qwen2.5:0.5b-instruct` immediately produced: `"Machine Learning is a subset of artificial intelligence that focuses on developing algorithms that allow computers to learn from and make predictions based on data."` with zero boilerplate, zero prompt echoing, and exact grounding.
  - **Resolution**:
    1. Replaced `tinyllama` with `qwen2.5:0.5b-instruct` as the empirically verified baseline model for the local environment. (Treated strictly as a verified baseline replacement based on local empirical results, not universally superior).
    2. Maintained single source of truth for the default model via `DEFAULT_MODEL = "qwen2.5:0.5b-instruct"` in `rag/generation/ollama_generator.py`, imported into `backend/config.py` to prevent duplicate defaults.
    3. Kept `LLM_MODEL` fully configurable via environment variables and `.env`.
    4. Kept the `Generator` ABC abstraction completely unchanged.
    5. Documented `tinyllama`'s failure mode in `README.md`, `Memory.md`, and module docstrings.
  - **Regression Tests Added**:
    - `TestGenerationStructuralRegression` in `tests/test_generation.py`: guards against conversational preambles (`"Sure, here's a revised version..."`), prompt template echoing (`"QUESTION:"`, `"CONTEXT:"`), chunk header leaks (`"[Chunk N]"`, `"Chunk ID:"`), and system instruction reproduction (`"untrusted reference data"`, `"System instructions"`).
    - `TestRealOllamaGeneration.test_real_generation_smoke`: dynamically tests `DEFAULT_MODEL` with structural quality assertions.
    - `tests/test_sample_document.py::TestSampleLLMGeneration`: dynamically uses configured `settings.llm_model` and verifies complete structural output integrity against `sample_ai_overview.pdf`.
  - **Verification**:
    - 552/552 tests passing (`pytest tests/ -v`).
    - Ruff lint and formatting checks 100% clean.
    - Live manual `/query` endpoint test against running FastAPI backend returned HTTP 200 with model `qwen2.5:0.5b-instruct`, 1 chunk retrieved, and clean, concise answer with zero prompt repetition or conversational filler.

- **Phase 6 Grounding Failure: Prevention of Answers to Unrelated Queries (Relevance Gate)**:
  - **Issue**: Manual testing on `sample_ai_overview.pdf` revealed that querying `"What is the capital of France?"` returned HTTP 200 with `"The capital of France is Paris."`, failing the baseline RAG grounding requirement because the answer was drawn entirely from the LLM's pretrained knowledge rather than the uploaded document.
  - **Root Cause**: ChromaDB's vector search unconditionally returns the top-$k$ nearest neighbors regardless of distance. When querying an unrelated topic, low-similarity chunks (e.g. cosine similarity ~0.04) were returned to `RAGQueryService` and passed to the LLM generator, which ignored grounding refusal rules and answered from its pretrained weights.
  - **Empirical Score Measurements** (`all-MiniLM-L6-v2` on `sample_ai_overview.pdf`):
    - Relevant queries: top similarity scores between **0.4244** and **0.7005** (e.g., *"What is Machine Learning?"* scored 0.6465 with 1000-char chunks, 0.7005 with 500-char chunks; *"What applications of deep learning..."* scored 0.5747 / 0.6614).
    - Unrelated queries: top similarity scores between **-0.0127** and **0.1432** (e.g., *"What is the capital of France?"* scored 0.0396 / 0.0545; cooking scored 0.0814 / 0.1072; history scored 0.0728 / 0.0791; out-of-domain astrophysics scored 0.1355 / 0.1432).
    - Separation Gap: A wide, unambiguous gap exists between the highest unrelated score (0.1432) and the lowest relevant top score (0.4244).
    - Threshold Selection: `RETRIEVAL_MIN_SCORE = 0.30` was chosen as a balanced baseline heuristic (more than $2\times$ higher than unrelated scores, and 0.12 below relevant scores).
  - **Implementation**:
    1. **Configuration**: Added `retrieval_min_score: float = 0.3` to `Settings` in `backend/config.py` with alias `RETRIEVAL_MIN_SCORE`, validated in `[-1.0, 1.0]` (rejecting booleans and non-numerics). Added to `.env.example`.
    2. **Retrieval Layer**: `SemanticRetriever` owns primary relevance score filtering. Accepts optional `min_score` in `__init__` and `retrieve()`. Discards chunks with similarity score strictly below threshold, returning an empty list if no chunks qualify.
    3. **Orchestration Layer**: Introduced `InsufficientContextError(EmptyRetrievalError)` in `rag/orchestration/exceptions.py`. `RAGQueryService` applies defensive score filtering and raises `InsufficientContextError` when context is empty or all chunks fall below threshold. The LLM generator is **never** called.
    4. **API Layer**: `backend/query.py` catches `EmptyRetrievalError` (which includes `InsufficientContextError`) and returns HTTP 404 with descriptive detail, preserving existing API contract behavior.
  - **Tests Added**:
    - `TestRelevanceGate` in `tests/test_retrieval.py`: `min_score` validation (bounds, bool, nan/inf), threshold boundary tests (`score == min_score` kept, `score < min_score` dropped), partial relevance filtering, per-call override, and real embedding integration test.
    - `TestRelevanceGateInQueryService` in `tests/test_query_orchestration.py`: `InsufficientContextError` inheritance, `min_score` validation, generator `assert_not_called()` on unrelated queries, partial relevance chunk forwarding.
    - `TestRetrievalMinScoreValidation` in `tests/test_phase3_config.py`: defaults, custom values, bounds, bool rejection.
    - `TestQueryEndpoint` in `tests/test_api_e2e.py`: verified `/query` returns HTTP 404 for unrelated query without calling generator.
  - **Verification**:
    - 580/580 tests passing (`pytest tests/ -v`).
    - Ruff check & format 100% clean across 65 repository files.
    - `git diff --check` clean.
    - Real Ollama live `/query` smoke tests with `qwen2.5:0.5b-instruct`:
      1. `"What is Machine Learning?"` $\to$ HTTP 200, grounded answer.
      2. `"What applications of deep learning are mentioned in the document?"` $\to$ HTTP 200, grounded answer.
      3. `"What is the capital of France?"` $\to$ HTTP 404, insufficient context error detail; **no LLM call, no pretrained Paris answer**.

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
| LLM Runtime | Ollama | — |
| LLM Model | Qwen 2.5 Instruct (Baseline) | 0.5B (397 MB) |
| LLM Model (Configurable) | Configurable via LLM_MODEL | — |
| LLM SDK | ollama (Python) | 0.6.2 |
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
| Abstract Retriever ABC | Allows swapping retrieval strategy (semantic, hybrid, reranking) without changing callers |
| Reuse VectorSearchResult as retrieval result | Already contains chunk_id, document_id, document_name, content, score, metadata; avoids unnecessary duplication |
| MAX_QUERY_LENGTH = 10,000 chars | Prevents excessive embedding compute; well above practical query lengths |
| MAX_TOP_K = 1,000 | Prevents unbounded resource usage; far above practical retrieval needs |
| Exception wrapping in retriever | Downstream exceptions wrapped into RetrievalError hierarchy; preserves cause chain for debugging |
| InvalidTopKError(InvalidQueryError) | Differentiates top_k errors from query text errors while maintaining full backward compatibility |
| Component boundary validation | Validates embedding vectors and vector store results before passing downstream; prevents misleading error attribution |
| default_top_k and Settings upper bound | Enforces MAX_TOP_K = 1000 across Settings, __init__, and runtime to prevent resource abuse |
| No new config values for Phase 4 | Reuses VECTOR_SEARCH_TOP_K; avoids duplicate settings |
| No API endpoint in Phase 4 | Retrieval is an internal service; API integration deferred to Phase 6 (end-to-end baseline) |
| Ollama for LLM runtime | Local-only, no API keys, lightweight; ideal for student project scope |
| TinyLlama as default model | 1.1B params, ~638 MB, 2K context; smallest practical model for development/testing |
| ollama Python SDK | Clean chat API, local client, well-maintained; no heavyweight frameworks needed |
| Abstract Generator ABC | Allows LLM backend swap (OpenAI, Anthropic, local) without downstream changes |
| Lazy Ollama client init | Avoids import-time cost; client created on first generate() call |
| num_predict for max tokens | Ollama uses `num_predict` instead of `max_tokens` in options dict |
| Context limiting: complete chunks only | Omits later chunks entirely rather than truncating mid-chunk; preserves semantic coherence |
| First chunk always included | Even if it exceeds max_chars, ensures at least one context chunk reaches the LLM |
| Empty context → InvalidContextError | Prevents LLM call without supporting evidence; explicit enforcement with tests |
| Generator does not call Retriever | Clean separation of concerns; context passed as parameter |
| HTTP/HTTPS URL validation for LLM_BASE_URL | Prevents invalid protocols; validated at Settings level |
| GenerationResult model | Pydantic model with answer, model_name, metadata; no citation info (Phase 7) |
| No API endpoint in Phase 5 | Generation is an internal service; API integration deferred to Phase 6 |
| GenerationConfigError(GenerationError, ValueError) | Dual inheritance allows catching as ValueError for backward compatibility and GenerationError in pipeline |
| LLM_TIMEOUT = 120.0 default | Configurable timeout passed to Ollama Client, ensuring CPU inference doesn't prematurely fail |
| urlparse netloc validation for base URL | Prevents incomplete URLs like `http://` missing host from passing validation |
| MAX_QUESTION_LENGTH = 10,000 chars | Prevents excessive prompt construction compute; matches Phase 4 MAX_QUERY_LENGTH |
| Explicit page None check in context builder | `page is None` check preserves `page_number = 0` avoiding Python truthiness falsy drop |
| Strict VectorSearchResult item validation | Prevents raw AttributeError on malformed context input; maps cleanly to InvalidContextError |
| Document indexing API integration | POST /documents/upload delegates to DocumentIndexingService; ingests, chunks, embeds, and stores in one step |
| Query API integration | POST /query delegates to RAGQueryService; retrieves context chunks and generates answer |
| Error cause unwrapping | _resolve_indexing_error and _resolve_query_error inspect __cause__ to map inner validation errors to 400/413/415/422 without hiding under generic 500 |
| Thin route handlers | FastAPI routes contain zero RAG business logic; services handle all pipeline operations |
| Sample document test fixture | tests/fixtures/sample_ai_overview.pdf enables repeatable end-to-end verification of ingestion, chunking, storage, and retrieval |
| Skip-aware live LLM test | Real Ollama generation tested when server is reachable, gracefully skipped in CI |
| Question-first prompt structure | QUESTION placed before CONTEXT followed by Answer: in user message; avoids instruction/context echo with small LLMs (TinyLlama) |
| Directive prose grounding prompt | Avoids numbered RULES headers in system message; prevents TinyLlama from interpreting prompt as rule-revision task |
| RETRIEVAL_MIN_SCORE = 0.30 baseline heuristic | Empirically chosen separation threshold based on all-MiniLM-L6-v2 measurements (unrelated <= 0.143, relevant >= 0.424) |
| SemanticRetriever owns relevance filtering | Filters out chunks below min_score at retrieval boundary; RAGQueryService handles resulting empty context and applies defensive check |
| InsufficientContextError(EmptyRetrievalError) | Subclasses EmptyRetrievalError for backward compatibility while providing explicit semantic typing for relevance gate rejections |
| Preserve HTTP 404 for insufficient context | Keeps existing API contract for queries with no supporting document evidence, avoiding breaking API changes |

---

## Known Issues

- None at present

---

## Known Limitations

- Baseline limitations observed in Phase 6:
  - Single-turn baseline only (no conversation history or chat memory)
  - Retrieval is semantic-only (no BM25 keyword search or hybrid retrieval yet — deferred to Phase 8)
  - No cross-encoder reranking of retrieved candidates (deferred to Phase 9)
  - No source citations or page attribution in API response (only answer text and chunk count; deferred to Phase 7)
  - Multi-document retrieval occurs across a shared flat vector space without document filtering or provenance grouping
  - LLM generation relies on local Ollama availability; cold-start or low-spec CPU inference may experience latency
  - `RETRIEVAL_MIN_SCORE = 0.30` is an empirically chosen Phase 6 baseline heuristic measured on `all-MiniLM-L6-v2` and `sample_ai_overview.pdf`, not a universally valid semantic threshold across all domains, models, or languages. Must be rigorously re-evaluated in Phase 11 evaluation framework.
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

- Multi-document support and citations (Phase 7)
- Hybrid retrieval (Phase 8)
- Reranking (Phase 9)
- OCR and multimodal processing (Phase 10)
- Evaluation framework (Phase 11)
- Optimization and reliability (Phase 12)
- Frontend and full integration (Phase 13)
- Documentation and open-source release (Phase 14)

---

## Next Immediate Tasks

1. Begin Phase 7 — Multi-Document Support and Citations
2. Support indexing and querying across multiple identified documents
3. Implement citation formatter (document name, page number, chunk identifier)
4. Add citation metadata to query API response
5. Add cross-document and multi-source attribution tests

---

Last Updated: 2026-09-19
