# MM-RAG

**Multi-Modal Multi-Document Retrieval-Augmented Generation**

> Upload multiple documents → ask questions in natural language → retrieve relevant evidence → generate a grounded answer → show the sources.

## Project Overview

MM-RAG is a modular RAG pipeline built from scratch as a final-year ROSP/open-source project. It processes heterogeneous documents (PDF, DOCX, images, scanned documents), indexes their content, retrieves relevant information for user queries, and generates evidence-grounded answers with source citations.

**Current Status:** Phase 0 — Project Foundation (under active development)

## Architecture

```text
Document Upload → Processing → Chunking → Embeddings → Vector Store
                                                            ↓
User Question → Query Processing → Hybrid Retrieval → Reranking → LLM → Answer + Citations
```

For the full architecture, see [Architecture.md](Architecture.md).

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
│   └── requirements.txt
│
├── frontend/           # React frontend (Vite)
│
├── rag/                # RAG pipeline modules (future)
├── evaluation/         # Evaluation framework (future)
├── tests/              # Test suite
├── docs/               # Additional documentation
└── .github/workflows/  # CI pipeline
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 20+
- Git

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
| `pytest tests/ -v` | Run tests |
| `ruff check backend/ rag/ tests/` | Lint Python code |
| `ruff format backend/ rag/ tests/` | Format Python code |

## Testing

```bash
# Install test dependencies
pip install pytest httpx

# Run all tests
pytest tests/ -v
```

## Documentation

- [PRD.md](PRD.md) — Product requirements and scope
- [Architecture.md](Architecture.md) — System design and data flow
- [Rules.md](Rules.md) — Development and coding rules
- [Phases.md](Phases.md) — Implementation roadmap
- [Memory.md](Memory.md) — Current project state

## Current Phase

**Phase 0 — Project Foundation**

The project is establishing its development foundation. RAG pipeline features (document ingestion, embeddings, retrieval, generation) will be implemented in subsequent phases.

## Planned Features

> ⚠️ The following features are **planned** and not yet implemented.

- PDF/DOCX document ingestion
- OCR for scanned documents and images
- Document chunking with metadata preservation
- Embedding generation
- Vector-based semantic search
- Keyword/hybrid retrieval
- Reranking
- LLM-based answer generation
- Source citations
- Evaluation framework

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

---

*This project is under active development as a final-year ROSP project.*
