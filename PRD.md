# Product Requirements Document (PRD)

## Multi-Modal Multi-Document RAG System

**Project Name:** MM-RAG  
**Project Type:** Final-Year ROSP Project  
**Status:** In Development  
**Version:** 1.0  
**Team Size:** 4

---

## 1. Project Overview

The project aims to develop a **Retrieval-Augmented Generation (RAG) system from the ground up** that can answer questions using information contained in a user's collection of documents.

The system will process multiple heterogeneous document types such as PDF, DOCX, images, scanned documents, and tables. It will extract and transform their content into searchable representations, retrieve relevant information for a user's query, and provide the retrieved context to an LLM for answer generation.

The final response will include **source references** so users can verify where the information came from.

The project will be developed as an **open-source project on GitHub**, including source code, documentation, setup instructions, tests, and evaluation results.

---

## 2. Problem Statement

Large Language Models can generate highly capable answers, but they do not automatically have access to information contained in a user's private or newly provided documents.

Traditional document-question-answering systems can also struggle when information is distributed across multiple documents, scanned pages, images, tables, and different document formats.

This project addresses these challenges by building a RAG pipeline capable of processing heterogeneous documents, retrieving relevant information across multiple sources, and generating answers grounded in retrieved evidence.

---

## 3. Project Vision

> **Upload multiple documents → ask questions in natural language → retrieve relevant evidence → generate a grounded answer → show the sources.**

### Example

A user uploads:

- `Annual_Report_2023.pdf`
- `Annual_Report_2024.pdf`
- `Financial_Table.png`
- `Research_Paper.pdf`
- `Scanned_Report.pdf`

The user asks:

> "How did the company's revenue change between 2023 and 2024?"

The system retrieves relevant information from the appropriate documents and generates an answer with supporting sources.

---

## 4. Objectives

### Primary Objective

Develop a complete **multi-document and multimodal RAG pipeline from scratch** using modular open-source technologies.

### Specific Objectives

1. Build a document ingestion pipeline.
2. Support multiple document formats.
3. Extract text and visual information.
4. Implement OCR for scanned documents and images.
5. Implement document chunking and metadata extraction.
6. Generate embeddings for searchable content.
7. Build a vector-based retrieval system.
8. Implement semantic and keyword/hybrid retrieval.
9. Implement reranking of retrieved results.
10. Support cross-document question answering.
11. Integrate an LLM for answer generation.
12. Provide source-level citations.
13. Evaluate retrieval and generation quality.
14. Publish the project as an open-source GitHub repository.

---

## 5. Target Users

- Students
- Researchers
- Developers
- Businesses
- Analysts
- Organizations with internal documents
- Users working with large document collections

### Example Use Cases

**Research:** Compare methodologies across multiple papers.

**Business:** Compare information across annual reports.

**Education:** Ask questions about lecture notes.

**Document Analysis:** Find and explain clauses across multiple documents.

---

## 6. Scope

### In Scope — MVP

#### Document Ingestion
- PDF
- DOCX
- Images
- Scanned PDF

#### Processing
- Text extraction
- OCR
- Chunking
- Metadata extraction

#### Retrieval
- Embeddings
- Vector search
- Semantic search
- Keyword search
- Hybrid retrieval
- Reranking
- Multi-document retrieval

#### Generation
- LLM integration
- Context-aware responses
- Grounded answers

#### Sources
- Document name
- Page number
- Section where available
- Retrieved context

#### Evaluation
- Retrieval quality
- Answer relevance
- Faithfulness
- Response latency

#### Open Source
- GitHub repository
- README
- Documentation
- Setup instructions
- Tests
- Evaluation results

---

## 7. Future Scope

The following features are outside the initial MVP but may be considered later:

- PPTX support
- XLSX/CSV understanding
- Video/audio documents
- Advanced vision-language models
- Knowledge graph integration
- Agentic RAG
- Query rewriting
- Conversation memory
- Distributed vector search
- Authentication and user accounts
- Cloud deployment

---

## 8. High-Level Architecture

```text
                 ┌─────────────────────┐
                 │   Document Upload   │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Document Processing │
                 │ Parsing + OCR       │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Chunking + Metadata │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Embedding Generation│
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │    Vector Store     │
                 └──────────┬──────────┘
                            │
                     User Question
                            ↓
                 ┌─────────────────────┐
                 │ Query Processing    │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Hybrid Retrieval    │
                 │ Semantic + Keyword  │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │     Reranking       │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Relevant Context    │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │        LLM          │
                 └──────────┬──────────┘
                            ↓
              ┌──────────────────────────┐
              │ Answer + Source Citations│
              └──────────────────────────┘
```

---

## 9. System Modules

### Module 1 — Document Ingestion

Accept and identify uploaded PDF, DOCX, image, and scanned-PDF files.

**Output:** A standardized internal document representation.

### Module 2 — Document Processing

- PDF text extraction
- DOCX paragraph/table extraction
- OCR for images and scanned PDFs

**Goal:** Convert different inputs into content the RAG pipeline can process.

### Module 3 — Multimodal Processing

Process information contained in:

- Text
- Tables
- Images
- Scanned pages

Visual content may use OCR, structured extraction, or a vision-language model depending on the final implementation.

### Module 4 — Chunking & Metadata

Split large documents into meaningful chunks while preserving:

```text
document_id
document_name
page_number
section
chunk_id
```

This metadata supports source attribution.

### Module 5 — Embedding Generation

Convert document chunks and user queries into vector representations for semantic retrieval.

### Module 6 — Vector Store

Store embeddings in a vector database/index.

Potential technologies:

- FAISS
- Chroma
- Qdrant

### Module 7 — Hybrid Retrieval

Combine semantic retrieval with keyword retrieval.

```text
                  Query
                    │
             ┌──────┴──────┐
             ↓             ↓
      Semantic Search   Keyword Search
             ↓             ↓
             └──────┬──────┘
                    ↓
              Combined Results
                    ↓
                 Reranker
```

### Module 8 — Reranking

Reorder retrieved candidates to identify the most relevant chunks before passing context to the LLM.

### Module 9 — Multi-Document Reasoning

Retrieve and synthesize information from different documents for a single query.

### Module 10 — LLM Generation

Provide the user question and retrieved context to an existing LLM through an API or locally hosted model.

**The project will not train an LLM from scratch.**

### Module 11 — Source Attribution

Display supporting information such as:

- Document name
- Page number
- Section where available

Example:

> **Source:** `Annual_Report_2024.pdf`, Page 38

### Module 12 — Evaluation

Evaluate retrieval, generation, and system performance.

Potential metrics:

- Recall@K
- Precision@K
- MRR
- Faithfulness
- Answer relevance
- Context relevance
- Response latency

---

## 10. Proposed Technology Stack

| Component | Proposed Technology |
|---|---|
| Backend Language | Python |
| API Framework | FastAPI |
| Document Processing | PyMuPDF, python-docx, Unstructured |
| OCR | Tesseract / PaddleOCR |
| Embeddings | Sentence Transformers / BGE embeddings |
| Vector Store | FAISS / Chroma / Qdrant |
| Retrieval | Semantic + Keyword / Hybrid Search |
| Reranking | Cross-Encoder / BGE Reranker |
| LLM | Open-source local model or API-based model |
| Frontend | React |
| Testing | Pytest |
| Version Control | Git + GitHub |
| Documentation | Markdown + GitHub |

The final libraries will be selected based on performance, compatibility, resource requirements, licensing, and project requirements.

---

## 11. API Design

### Upload Documents

```http
POST /documents/upload
```

### List Documents

```http
GET /documents
```

### Ask Question

```http
POST /query
```

Example request:

```json
{
  "question": "Compare revenue between 2023 and 2024",
  "document_ids": ["doc_1", "doc_2"]
}
```

Example response:

```json
{
  "answer": "Revenue increased by 25%.",
  "sources": [
    {
      "document": "annual_report_2023.pdf",
      "page": 42
    },
    {
      "document": "annual_report_2024.pdf",
      "page": 38
    }
  ]
}
```

These API definitions are initial design targets and may change during implementation.

---

## 12. Frontend Requirements

The frontend should provide:

1. Document upload
2. Indexed document list
3. Natural-language query interface
4. Generated answer display
5. Supporting source display

### Basic Flow

```text
Upload Documents
       ↓
Documents Indexed
       ↓
Ask Question
       ↓
Retrieval
       ↓
Answer
       ↓
Sources
```

---

## 13. Open-Source Strategy

The project itself will be developed as an open-source project and hosted on GitHub.

### Planned Repository Structure

```text
MM-RAG/
│
├── backend/
├── frontend/
│
├── rag/
│   ├── ingestion/
│   ├── processing/
│   ├── embeddings/
│   ├── retrieval/
│   ├── reranking/
│   └── generation/
│
├── evaluation/
├── tests/
├── docs/
│
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── requirements.txt
└── .gitignore
```

The structure may be adjusted as implementation evolves.

---

## 14. Team Responsibilities

### Member 1 — Backend & Architecture

- FastAPI
- API design
- System architecture
- Backend integration
- Overall pipeline integration

### Member 2 — Document Processing

- PDF/DOCX processing
- OCR
- Images/tables
- Chunking
- Metadata extraction

### Member 3 — RAG & Retrieval

- Embeddings
- Vector store
- Semantic search
- Hybrid retrieval
- Reranking
- LLM integration

### Member 4 — Frontend & Evaluation

- React interface
- Source display
- Testing
- Evaluation
- Documentation

All members should understand the complete end-to-end RAG architecture.

---

## 15. Development Roadmap

### Phase 1 — Foundation
**Week 1**

- Create GitHub repository
- Establish project structure
- Set up Python environment
- Set up FastAPI
- Create initial frontend
- Establish Git workflow

### Phase 2 — Document Pipeline
**Week 2**

- PDF ingestion
- DOCX ingestion
- Image ingestion
- OCR
- Chunking
- Metadata extraction

### Phase 3 — Basic RAG
**Week 3**

Build:

```text
Document
 ↓
Chunk
 ↓
Embedding
 ↓
Vector Store
 ↓
Retrieve
 ↓
LLM
 ↓
Answer
```

This becomes the baseline implementation.

### Phase 4 — Improved Retrieval
**Week 4**

- Semantic search
- Keyword search
- Hybrid retrieval
- Reranking

### Phase 5 — Multimodal & Multi-Document
**Weeks 5–6**

- Multiple document collections
- Cross-document retrieval
- OCR improvements
- Tables/images
- Multimodal experiments

### Phase 6 — Sources & UI
**Week 7**

- Source citations
- Page references
- Frontend improvements
- Document management

### Phase 7 — Evaluation
**Week 8**

- Build evaluation dataset
- Baseline evaluation
- Enhanced evaluation
- Performance comparison

### Phase 8 — Open-Source Release
**Week 9**

- Documentation
- README
- Installation guide
- Architecture documentation
- Tests
- License
- GitHub release

---

## 16. MVP Definition

The first working version should be:

> **Upload multiple PDFs → process → chunk → embed → store → retrieve → ask questions → generate answers → show sources.**

### Development Progression

```text
LEVEL 1
Single-PDF RAG
       ↓
LEVEL 2
Multi-PDF RAG
       ↓
LEVEL 3
Source Citations
       ↓
LEVEL 4
Hybrid Retrieval
       ↓
LEVEL 5
Reranking
       ↓
LEVEL 6
OCR / Multimodal Processing
       ↓
LEVEL 7
Evaluation
       ↓
LEVEL 8
Open-Source Release
```

The team should complete the basic end-to-end pipeline before implementing advanced functionality.

---

## 17. Evaluation Plan

Example evaluation structure:

```text
Documents
    +
Questions
    ↓
Baseline RAG
    ↓
Enhanced RAG
    ↓
Compare Results
```

| Area | Example Metrics |
|---|---|
| Retrieval | Recall@K, Precision@K, MRR |
| Generation | Faithfulness, Answer Relevance |
| Context | Context Relevance |
| Performance | Response Latency, Processing Time |

The exact dataset size and metrics may be adjusted during implementation.

---

## 18. Non-Functional Requirements

### Performance
The system should provide acceptable retrieval and response times for the target document collection.

### Accuracy
Retrieved information should be relevant and generated responses should be grounded in retrieved context.

### Scalability
The architecture should support multiple documents within practical resource limits.

### Maintainability
The codebase should use modular components with clear responsibilities.

### Reproducibility
The repository should include installation, environment, dependency, usage, and evaluation instructions.

### Security
API keys and secrets must never be committed to the repository. Uploaded document contents should not be unnecessarily logged or exposed.

### Open Source
The project should include an appropriate open-source license and clear contribution/documentation guidelines.

---

## 19. Risks and Mitigation

| Risk | Mitigation |
|---|---|
| Large documents require significant processing time | Efficient chunking and indexing |
| OCR errors | Evaluate OCR quality and use fallback processing |
| Poor retrieval quality | Hybrid retrieval and reranking |
| LLM hallucination | Ground answers in retrieved context and provide citations |
| High local hardware requirements | Use lightweight models or APIs where necessary |
| Multimodal complexity | Implement incrementally |
| Scope becoming too large | Prioritize MVP |
| Weak evaluation dataset | Create representative questions across document types |
| Model/API availability | Keep model integration modular |
| Sensitive documents | Prefer local processing where practical and avoid logging content |

---

## 20. Definition of Done

- [ ] PDF ingestion works
- [ ] Multiple documents can be indexed
- [ ] Documents are chunked
- [ ] Embeddings are generated
- [ ] Vector retrieval works
- [ ] Users can ask questions
- [ ] LLM generates answers from retrieved context
- [ ] Sources are displayed
- [ ] Hybrid retrieval is implemented and tested
- [ ] Reranking is implemented and tested
- [ ] At least one multimodal capability works
- [ ] Cross-document queries work
- [ ] Evaluation dataset exists
- [ ] Baseline evaluation is completed
- [ ] Enhanced evaluation is completed
- [ ] Tests are written
- [ ] Documentation is complete
- [ ] GitHub repository is publicly available

---

## 21. Success Criteria

The project will be considered successful if it:

1. Provides a working end-to-end RAG pipeline.
2. Supports querying multiple documents.
3. Demonstrates at least one meaningful multimodal capability.
4. Provides relevant retrieved context.
5. Generates evidence-grounded answers.
6. Displays supporting sources where available.
7. Demonstrates measurable performance through evaluation.
8. Maintains a modular and understandable codebase.
9. Is documented and published as an open-source GitHub project.

---

## 22. Future Enhancements

Potential post-MVP extensions include:

- Advanced vision-language models
- Table-aware retrieval
- Knowledge graphs
- Agentic retrieval
- Query decomposition
- Query rewriting
- Conversational memory
- User authentication
- Cloud deployment
- Distributed vector databases
- Advanced evaluation dashboards

---

## 23. Project Principle

> **Simple → Working → Measurable → Improved**

The development strategy is to first build a reliable basic RAG pipeline, then incrementally add multi-document support, source attribution, improved retrieval, reranking, multimodal processing, and evaluation.

The project should prioritize **one reliable and measurable system over a large number of incomplete features**.

---

## 24. Final Project Goal

The final system will be an **open-source, modular, multi-modal, multi-document RAG application** that allows users to upload heterogeneous documents, ask natural-language questions, retrieve relevant evidence, and receive grounded answers with supporting sources.

The project will demonstrate the complete development lifecycle:

```text
Requirements
     ↓
Architecture
     ↓
Implementation
     ↓
Testing
     ↓
Evaluation
     ↓
Documentation
     ↓
Open-Source Release
```

---

**Document Status:** Draft / Development v1.0  
**Last Updated:** 2026
