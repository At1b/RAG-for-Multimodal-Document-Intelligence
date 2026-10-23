# MM-RAG

**Multi-Modal Multi-Document Retrieval-Augmented Generation**

> Upload multiple documents → ask questions in natural language → retrieve relevant evidence → generate a grounded answer → show the sources.

## Project Overview

MM-RAG is a modular RAG pipeline built from scratch as a final-year ROSP/open-source project. It processes heterogeneous documents (PDF, DOCX, images, scanned documents), indexes their content, retrieves relevant information for user queries, and generates evidence-grounded answers with source citations.

**Current Status:** Phase 10 — OCR and Multimodal Document Processing (Completed)

## Architecture

```text
Document Upload (PDF / DOCX / Images) → Ingestion & OCR → Chunking → Embeddings → Vector Store
                                                                                        ↓
User Question → Query Processing → Hybrid Retrieval (Vector + BM25) → Reranking → LLM → Answer + Citations
```

For the full architecture, see [Architecture.md](Architecture.md) and [docs/ocr.md](docs/ocr.md).

## Repository Structure

```text
MM-RAG/
├── PRD.md              # Product requirements
├── Architecture.md     # System architecture
├── Rules.md            # Development rules
├── Phases.md           # Development roadmap
├── Memory.md           # Living project state
├── README.md           # This file
├── LICENSE             # MIT License
├── .gitignore
├── .env.example        # Environment variable template
├── pyproject.toml      # Python tooling config (ruff, pytest)
│
├── backend/            # FastAPI backend
│   ├── __init__.py
│   ├── main.py         # Entry point + health endpoint
│   ├── config.py       # Environment configuration
│   ├── documents.py    # Document upload & ingestion endpoint
│   ├── query.py        # Query & RAG generation endpoint
│   └── requirements.txt
│
├── frontend/           # React frontend (Vite)
│
├── rag/                # RAG pipeline modules
│   ├── ingestion/      # Document loaders (PDF, DOCX, Image) and OCR engine
│   ├── chunking/       # Character & token text chunking
│   ├── embeddings/     # Sentence-transformers embedding & ChromaDB indexing
│   ├── retrieval/      # Dense vector, BM25 sparse, and hybrid reciprocal rank fusion
│   ├── reranking/      # Cross-encoder semantic reranking
│   ├── generation/     # Ollama LLM client and prompt orchestration
│   └── citations/      # Verifiable source citation extraction & formatting
│
├── evaluation/         # Evaluation framework (retrieval & OCR benchmarks)
├── tests/              # Test suite (919 tests)
├── docs/               # Technical documentation (docs/ocr.md)
└── .github/workflows/  # CI pipeline
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 20+
- Git
- [Ollama](https://ollama.com/download) (for LLM generation)
- [Tesseract OCR 5.x](https://github.com/UB-Mannheim/tesseract/wiki) (for scanned PDFs and images)

### OCR Setup (Tesseract 5)

MM-RAG uses Tesseract 5 via subprocess execution for scanned PDF pages and standalone image uploads (`.png`, `.jpg`, `.jpeg`).

#### Windows Installation:
Install the UB-Mannheim Tesseract OCR package via Winget or direct installer:
```powershell
winget install UB-Mannheim.TesseractOCR
```
Ensure English (`eng`) language data is selected. Verify installation:
```powershell
tesseract --version
tesseract --list-langs
```
If Tesseract is installed to the default directory (`C:\Program Files\Tesseract-OCR\tesseract.exe`), the backend will detect it automatically even if not added to the system `PATH`.

Configurable parameters in `.env`:
- `OCR_ENABLED` (default: `true`)
- `OCR_LANGUAGE` (default: `eng`)
- `OCR_DPI` (default: `300`; set to `200` for 1.9x faster ingestion with equivalent printed accuracy)
- `OCR_MIN_NATIVE_CHARS` (default: `25`)
- `OCR_TIMEOUT_SECONDS` (default: `60.0`)
- `OCR_MAX_PAGES` (default: `50`)
- `OCR_MAX_IMAGE_PIXELS` (default: `40000000`)

For detailed architecture, DPI benchmarks, and failure policies, see [docs/ocr.md](docs/ocr.md).

### Backend Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows

# Install dependencies
pip install -r backend/requirements.txt

# Copy environment config
cp .env.example .env

# Start the backend
uvicorn backend.main:app --reload
```

The API will be available at `http://localhost:8000`.

Verify it's running:

```bash
curl http://localhost:8000/health
```

### LLM Setup (Ollama)

The generation layer uses [Ollama](https://ollama.com) to run a local LLM. No API key is required.

```bash
# 1. Install Ollama from https://ollama.com/download

# 2. Start the Ollama server
ollama serve

# 3. Pull the verified baseline model (~397 MB)
ollama pull qwen2.5:0.5b-instruct
```

Ollama runs on `http://localhost:11434` by default. You can change this in your `.env` file via `LLM_BASE_URL`.

Configurable parameters in `.env`:
- `LLM_MODEL` (default: `qwen2.5:0.5b-instruct`)
- `LLM_BASE_URL` (default: `http://localhost:11434`)
- `LLM_TEMPERATURE` (default: `0.1`)
- `LLM_MAX_TOKENS` (default: `512`)
- `LLM_TIMEOUT` (default: `120.0`)
- `LLM_CONTEXT_MAX_CHARS` (default: `3000`)

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

## Development Commands

| Command | Description |
|---|---|
| `uvicorn backend.main:app --reload` | Start backend dev server |
| `cd frontend && npm run dev` | Start frontend dev server |
| `pytest tests/ -v` | Run full test suite |
| `pytest tests/ -m "not ocr_live"` | Run tests excluding live OCR engine |
| `python -m evaluation.ocr_eval` | Run OCR CER/WER and latency evaluation |
| `ruff check .` | Lint Python codebase |
| `ruff format --check .` | Check Python code formatting |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=rag --cov=backend tests/
```

## Documentation

- [PRD.md](PRD.md) — Product requirements and scope
- [Architecture.md](Architecture.md) — System design and data flow
- [Rules.md](Rules.md) — Development and coding rules
- [Phases.md](Phases.md) — Implementation roadmap
- [Memory.md](Memory.md) — Current project state
- [docs/ocr.md](docs/ocr.md) — OCR architecture, evaluation metrics, and benchmarks

## Current Phase

**Phase 10 — OCR & Multimodal Document Processing (Completed)**

The pipeline processes native and scanned documents (PDF, DOCX, PNG, JPEG), chunks content with exact page provenance, embeds into ChromaDB, executes hybrid retrieval with cross-encoder reranking, and generates grounded answers with verifiable citations.

## Implemented Features

- [x] PDF and DOCX digital document ingestion
- [x] Tesseract 5 OCR for scanned PDF pages and standalone image files (`.png`, `.jpg`, `.jpeg`)
- [x] Mixed-mode PDF processing (digital text layer preserved, scanned pages OCR'd)
- [x] Recursive character chunking with provenance preservation
- [x] Local Sentence-Transformers embeddings (`all-MiniLM-L6-v2`)
- [x] ChromaDB persistent vector storage with cosine similarity search
- [x] Pure-Python BM25 keyword search and Reciprocal Rank Fusion (RRF) hybrid retrieval
- [x] Sentence-Transformers cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- [x] Relevance gating to suppress hallucination on out-of-context queries
- [x] Ollama local LLM integration (`qwen2.5:0.5b-instruct`)
- [x] Verifiable source citation formatting (`[Doc, p. X]`)
- [x] End-to-end React web UI with document upload and grounded chat

## Planned Features (Upcoming Phases)

- [ ] Phase 11 — Quantitative RAG Evaluation Framework (RAG Triad, Hit Rate, MRR)
- [ ] Phase 12 — Final Polish, Hardening, and Deployment Packaging

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

---

*This project is under active development as a final-year ROSP project.*
