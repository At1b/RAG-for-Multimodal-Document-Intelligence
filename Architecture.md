# System Architecture

## Multi-Modal Multi-Document RAG System (MM-RAG)

**Project:** Final-Year ROSP Project  
**Document:** System Architecture  
**Version:** 1.0  
**Status:** Initial Architecture / In Development

---

## 1. Architecture Overview

MM-RAG is designed as a modular Retrieval-Augmented Generation system that processes heterogeneous documents, indexes their content, retrieves relevant information for a user's question, and generates an evidence-grounded answer using an LLM.

The architecture is divided into two major flows:

1. **Ingestion and Indexing Flow** — converts uploaded documents into searchable representations.
2. **Query and Generation Flow** — retrieves relevant information and generates the final answer.

### High-Level Flow

```text
                         ┌─────────────────────┐
                         │        USER         │
                         └──────────┬──────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     │                             │
                     ▼                             ▼
             Upload Documents               Ask Question
                     │                             │
                     ▼                             ▼
          ┌────────────────────┐        ┌────────────────────┐
          │ Document Ingestion │        │   Query Processing │
          └─────────┬──────────┘        └──────────┬─────────┘
                    │                              │
                    ▼                              │
          ┌────────────────────┐                   │
          │ Document Processing│                   │
          │ Parsing + OCR      │                   │
          └─────────┬──────────┘                   │
                    │                              │
                    ▼                              │
          ┌────────────────────┐                   │
          │ Chunking + Metadata│                   │
          └─────────┬──────────┘                   │
                    │                              │
                    ▼                              │
          ┌────────────────────┐                   │
          │ Embedding Generation│                  │
          └─────────┬──────────┘                   │
                    │                              │
                    ▼                              │
          ┌────────────────────┐                   │
          │    Vector Store    │◄──────────────────┘
          └─────────┬──────────┘
                    │
                    ▼
          ┌────────────────────┐
          │  Hybrid Retrieval  │
          │ Semantic + Keyword │
          └─────────┬──────────┘
                    │
                    ▼
          ┌────────────────────┐
          │     Reranking      │
          └─────────┬──────────┘
                    │
                    ▼
          ┌────────────────────┐
          │ Relevant Context   │
          └─────────┬──────────┘
                    │
                    ▼
          ┌────────────────────┐
          │        LLM         │
          └─────────┬──────────┘
                    │
                    ▼
          ┌────────────────────┐
          │ Answer + Citations │
          └────────────────────┘
```

---

# 2. Architectural Layers

The system is organized into the following logical layers:

```text
┌─────────────────────────────────────────────┐
│              Presentation Layer             │
│              React Web Interface            │
└──────────────────────┬──────────────────────┘
                       │
┌──────────────────────▼──────────────────────┐
│                 API Layer                   │
│                   FastAPI                   │
└──────────────────────┬──────────────────────┘
                       │
┌──────────────────────▼──────────────────────┐
│              Application Layer              │
│     Document Management + Query Service     │
└──────────────────────┬──────────────────────┘
                       │
┌──────────────────────▼──────────────────────┐
│                  RAG Layer                  │
│ Ingestion | Retrieval | Reranking | LLM     │
└──────────────────────┬──────────────────────┘
                       │
┌──────────────────────▼──────────────────────┐
│               Data / Storage Layer          │
│ Vector Store | Metadata | Uploaded Files    │
└─────────────────────────────────────────────┘
```

---

# 3. Component Architecture

## 3.1 Frontend

### Responsibility

The frontend provides the user interface for:

- Uploading documents
- Viewing indexed documents
- Entering questions
- Displaying answers
- Displaying sources and citations
- Showing processing/retrieval status

### Proposed Technology

**React**

### Communication

The frontend communicates with the backend through REST APIs.

```text
React
  │
  │ HTTP / JSON
  ▼
FastAPI
```

---

# 4. API Layer

## FastAPI Backend

The API layer exposes operations required by the frontend.

### Initial API Endpoints

```text
POST   /documents/upload
GET    /documents
DELETE /documents/{document_id}
POST   /query
GET    /health
```

### Responsibilities

- Request validation
- File upload handling
- Query handling
- Calling application services
- Returning structured responses
- Error handling

The API layer should not contain the core retrieval or generation logic. Those responsibilities belong to the RAG/application modules.

---

# 5. Document Ingestion Architecture

The ingestion pipeline starts when the user uploads one or more documents.

```text
                Uploaded File
                     │
                     ▼
            ┌─────────────────┐
            │ File Validation │
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ File Type Router│
            └────────┬────────┘
                     │
          ┌──────────┼───────────┐
          ▼          ▼           ▼
        PDF        DOCX       Image/Scan
          │          │           │
          ▼          ▼           ▼
      PDF Parser  DOCX Parser    OCR
          │          │           │
          └──────────┼───────────┘
                     ▼
             Normalized Content
```

---

# 6. File Validation

Before processing, the system should validate:

- File type
- File extension
- MIME type where available
- File size
- Basic file integrity

Invalid files should be rejected before entering the processing pipeline.

---

# 7. Document Processing

Different document types use different processing strategies.

## PDF

```text
PDF
 ↓
Page Extraction
 ↓
Text / Table / Metadata
```

## DOCX

```text
DOCX
 ↓
Paragraph Extraction
 ↓
Table Extraction
 ↓
Metadata
```

## Image / Scanned PDF

```text
Image / Scan
 ↓
OCR
 ↓
Extracted Text
 ↓
Page / Position Metadata
```

The implementation may later introduce specialized processors for tables, charts, and visual content.

---

# 8. Normalized Document Representation

Different source formats should be converted into a common internal representation before chunking.

Conceptually:

```text
Document
│
├── document_id
├── document_name
├── source_type
├── pages
│   ├── page_number
│   ├── content
│   └── metadata
└── metadata
```

A normalized representation allows downstream RAG components to work independently of the original file format.

---

# 9. Chunking Architecture

Large documents cannot always be passed directly to the embedding model or LLM.

Therefore, the normalized content is divided into smaller chunks.

```text
Normalized Document
        │
        ▼
   Chunking Strategy
        │
        ├──── Chunk 1
        ├──── Chunk 2
        ├──── Chunk 3
        ├──── ...
        └──── Chunk N
```

Each chunk should preserve metadata.

### Example

```json
{
  "chunk_id": "chunk_001",
  "document_id": "doc_001",
  "document_name": "annual_report.pdf",
  "page_number": 42,
  "section": "Financial Performance",
  "content": "..."
}
```

### Design Requirement

The chunking strategy should avoid losing important context and should preserve enough metadata to generate citations.

---

# 10. Embedding Pipeline

After chunking, each chunk is converted into a vector representation.

```text
Text Chunk
    │
    ▼
Embedding Model
    │
    ▼
Vector
    │
    ▼
Vector Store
```

The query will use the same embedding space:

```text
User Question
     │
     ▼
Embedding Model
     │
     ▼
Query Vector
```

The query vector is then used to find semantically similar document chunks.

---

# 11. Vector Storage

The vector store maintains the searchable representation of indexed chunks.

### Stored Information

Conceptually:

```text
Vector
+
Chunk Content
+
Metadata
```

### Candidate Technologies

- FAISS
- Chroma
- Qdrant

The final technology will be selected after evaluating:

- Search performance
- Persistence
- Filtering capabilities
- Deployment complexity
- Hardware/resource requirements
- Project requirements

---

# 12. Query Processing Architecture

When a user asks a question:

```text
User Question
      │
      ▼
Query Validation
      │
      ▼
Query Processing
      │
      ▼
Query Embedding
      │
      ▼
Retrieval
```

The query processor may later support:

- Query normalization
- Query rewriting
- Document filtering
- Query decomposition

These are considered future enhancements unless required by the MVP.

---

# 13. Retrieval Architecture

The retrieval layer is responsible for finding the most relevant information.

The proposed architecture combines semantic and keyword retrieval.

```text
                         User Query
                             │
                 ┌───────────┴───────────┐
                 │                       │
                 ▼                       ▼
          Semantic Retrieval       Keyword Retrieval
                 │                       │
                 ▼                       ▼
           Semantic Results         Keyword Results
                 │                       │
                 └───────────┬───────────┘
                             ▼
                    Result Combination
                             │
                             ▼
                         Reranking
                             │
                             ▼
                     Top-K Context
```

---

# 14. Semantic Retrieval

Semantic retrieval uses embeddings to identify chunks that are conceptually similar to the query.

Example:

```text
Query:
"How did company revenue grow?"

Potential retrieved text:
"Annual revenue increased by 25% during the fiscal year."
```

The exact words do not need to match for the content to be retrieved.

---

# 15. Keyword Retrieval

Keyword retrieval helps when exact terms, names, identifiers, or numbers are important.

Example:

```text
Query:
"Project Alpha 2024 revenue"

Keyword retrieval can prioritize:
"Project Alpha"
"2024"
"revenue"
```

A BM25-style implementation may be evaluated for this component.

---

# 16. Hybrid Retrieval

Hybrid retrieval combines semantic and keyword results.

Conceptually:

```text
Semantic Score
       +
Keyword Score
       ↓
Combined Retrieval Score
       ↓
Candidate Results
```

The exact score-combination method will be determined during implementation and experimentation.

---

# 17. Reranking Architecture

The initial retrieval stage may produce more candidates than the LLM should receive.

Example:

```text
Initial Retrieval
     │
     ▼
Top 20 Candidates
     │
     ▼
Reranker
     │
     ▼
Top 5 Relevant Chunks
```

A cross-encoder or suitable reranking model may be evaluated.

### Goal

Improve the relevance of the final context supplied to the LLM.

---

# 18. Context Assembly

After reranking, the system constructs the final context.

```text
Top-K Chunks
     │
     ▼
Context Builder
     │
     ├── Chunk Content
     ├── Document Name
     ├── Page Number
     ├── Section
     └── Other Metadata
     │
     ▼
LLM Context
```

The context builder is responsible for maintaining source information.

---

# 19. LLM Generation Architecture

The generation stage combines:

```text
System Instructions
       +
User Question
       +
Retrieved Context
       ↓
      LLM
       ↓
Generated Answer
```

The system should instruct the LLM to prioritize the retrieved evidence.

### Grounding Principle

The LLM should not be treated as the source of truth for document-specific questions. Retrieved document context should be the primary evidence used for the response.

---

# 20. Source Attribution Architecture

Because each chunk retains metadata, the final response can associate claims with source documents.

```text
Retrieved Chunk
      │
      ├── Document Name
      ├── Page Number
      ├── Section
      └── Chunk ID
             │
             ▼
      Source Formatter
             │
             ▼
       Final Response
```

Example:

```text
Answer:
Revenue increased by approximately 25%.

Sources:
[1] Annual_Report_2023.pdf — Page 42
[2] Annual_Report_2024.pdf — Page 38
```

The exact citation format will be finalized during implementation.

---

# 21. Multi-Document Retrieval

The system should not assume that the answer exists in a single document.

```text
                    User Query
                        │
                        ▼
                   Retrieval
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
      Document A    Document B    Document C
          │             │             │
          └─────────────┼─────────────┘
                        ▼
                  Relevant Context
                        │
                        ▼
                       LLM
```

This enables questions such as:

> "Compare the revenue reported in the 2023 and 2024 annual reports."

---

# 22. Multimodal Processing Architecture

The project will progressively support information that is not represented purely as machine-readable text.

```text
                Document
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
       Text      Tables     Images
        │          │          │
        │          │          ▼
        │          │         OCR /
        │          │     Vision Processing
        │          │          │
        └──────────┼──────────┘
                   ▼
            Normalized Content
                   │
                   ▼
                Chunking
                   │
                   ▼
               Indexing
```

### MVP Multimodal Strategy

The initial implementation should prioritize:

1. OCR for images/scanned documents.
2. Extraction of text from tables where practical.
3. Preservation of page/source metadata.

Advanced image reasoning through vision-language models can be added after the basic pipeline is stable.

---

# 23. Storage Architecture

The system may use three logical storage areas.

```text
                 Storage
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
 Uploaded Files   Metadata   Vector Store
```

### Uploaded Files

Original user documents.

### Metadata

Information such as:

- Document ID
- File name
- File type
- Upload time
- Processing status

### Vector Store

- Embeddings
- Chunk references
- Search metadata

The exact persistence technologies will be finalized during implementation.

---

# 24. End-to-End Data Flow

## Ingestion Flow

```text
User
 │
 ▼
Upload Document
 │
 ▼
Validation
 │
 ▼
File-Type Detection
 │
 ▼
Parser / OCR
 │
 ▼
Normalized Content
 │
 ▼
Chunking
 │
 ▼
Metadata Attachment
 │
 ▼
Embedding Generation
 │
 ▼
Vector Store
```

## Query Flow

```text
User
 │
 ▼
Question
 │
 ▼
Query Processing
 │
 ▼
Query Embedding
 │
 ▼
Hybrid Retrieval
 │
 ▼
Candidate Chunks
 │
 ▼
Reranking
 │
 ▼
Top-K Context
 │
 ▼
Prompt / Context Assembly
 │
 ▼
LLM
 │
 ▼
Answer Generation
 │
 ▼
Source Attribution
 │
 ▼
Frontend
```

---

# 25. Error Handling

The architecture should handle failures at each stage.

### Examples

```text
Invalid File
    ↓
Return Validation Error

Unsupported Format
    ↓
Return Unsupported-Type Error

OCR Failure
    ↓
Return Processing Error / Fallback

Embedding Failure
    ↓
Mark Document Processing as Failed

Vector Store Failure
    ↓
Return Retrieval Service Error

LLM Failure
    ↓
Return Generation Error
```

Errors should be logged for debugging without unnecessarily logging sensitive document content.

---

# 26. Processing States

Documents should have explicit processing states.

```text
UPLOADED
   ↓
PROCESSING
   ↓
CHUNKED
   ↓
INDEXING
   ↓
READY
```

If processing fails:

```text
PROCESSING
     ↓
   FAILED
```

This state model will allow the frontend to communicate document status to the user.

---

# 27. Security and Privacy Architecture

The initial system should follow these principles:

- Never commit API keys to GitHub.
- Store secrets in environment variables.
- Validate uploaded files.
- Restrict accepted file types.
- Limit upload sizes.
- Avoid unnecessary document-content logging.
- Avoid exposing internal storage paths.
- Sanitize user-controlled metadata.
- Clearly document whether external APIs receive uploaded content.

If the system uses external LLM APIs, the privacy implications should be documented in the README.

---

# 28. Modularity Principles

Each major RAG component should be replaceable without rewriting the entire system.

For example:

```text
Embedding Interface
       │
       ├── SentenceTransformer
       ├── BGE
       └── Future Model
```

Similarly:

```text
Vector Store Interface
       │
       ├── FAISS
       ├── Chroma
       └── Qdrant
```

And:

```text
LLM Interface
       │
       ├── Local Model
       ├── API Model
       └── Future Provider
```

This makes the system easier to test, extend, and maintain.

---

# 29. Proposed Code Architecture

The following is the initial logical structure:

```text
MM-RAG/
│
├── backend/
│   ├── api/
│   ├── services/
│   └── main.py
│
├── rag/
│   ├── ingestion/
│   │   ├── loaders/
│   │   └── validators/
│   │
│   ├── processing/
│   │   ├── parsers/
│   │   ├── ocr/
│   │   ├── chunking/
│   │   └── metadata/
│   │
│   ├── embeddings/
│   │
│   ├── retrieval/
│   │   ├── semantic/
│   │   ├── keyword/
│   │   └── hybrid/
│   │
│   ├── reranking/
│   │
│   ├── generation/
│   │
│   └── citations/
│
├── frontend/
│
├── evaluation/
│
├── tests/
│
├── docs/
│
├── README.md
├── PRD.md
├── LICENSE
├── requirements.txt
└── .gitignore
```

This structure is a proposed starting point and may be simplified or reorganized during implementation.

---

# 30. Technology Decision Matrix

The project should not select technologies solely because they are popular. Each major component should be evaluated.

| Component | Candidate Options | Selection Criteria |
|---|---|---|
| PDF Processing | PyMuPDF / Unstructured | Accuracy, speed, metadata |
| OCR | Tesseract / PaddleOCR | Accuracy, speed, local resources |
| Embeddings | Sentence Transformers / BGE | Retrieval quality, model size |
| Vector Store | FAISS / Chroma / Qdrant | Performance, persistence, filtering |
| Keyword Search | BM25 / equivalent | Exact-match retrieval |
| Reranker | Cross-Encoder / BGE | Relevance improvement |
| LLM | Local / API | Quality, cost, latency, hardware |
| Frontend | React | Usability and integration |
| Backend | FastAPI | API performance and simplicity |

Final choices should be documented in a technical decision record after experimentation.

---

# 31. Architecture Evolution

The architecture will evolve incrementally.

### Stage 1 — Basic RAG

```text
PDF
 ↓
Chunk
 ↓
Embedding
 ↓
Vector Search
 ↓
LLM
 ↓
Answer
```

### Stage 2 — Multi-Document

```text
Multiple Documents
 ↓
Unified Index
 ↓
Retrieval
 ↓
LLM
```

### Stage 3 — Better Retrieval

```text
Query
 ↓
Semantic + Keyword
 ↓
Hybrid Results
 ↓
Reranker
 ↓
Context
 ↓
LLM
```

### Stage 4 — Multimodal

```text
PDF + DOCX + Images + Scans
             ↓
       Processing/OCR
             ↓
      Unified Representation
             ↓
           Index
```

### Stage 5 — Evaluation

```text
Dataset
  ↓
Baseline
  ↓
Enhanced System
  ↓
Metrics
  ↓
Analysis
```

---

# 32. Performance Considerations

The main performance-sensitive stages are:

1. Document parsing
2. OCR
3. Embedding generation
4. Vector retrieval
5. Reranking
6. LLM generation

Potential optimization strategies include:

- Batch embedding generation
- Caching embeddings
- Efficient chunk sizes
- Limiting top-K candidates
- Reranking only a small candidate set
- Reusing indexed documents
- Streaming LLM responses where supported
- Avoiding unnecessary repeated OCR

Performance optimization should be driven by measurements rather than assumptions.

---

# 33. Testing Architecture

Testing will exist at multiple levels.

### Unit Tests

Test individual components:

```text
Parser
Chunker
Embedding Module
Retriever
Reranker
Citation Formatter
```

### Integration Tests

Test component interactions:

```text
Upload
 ↓
Process
 ↓
Index
 ↓
Retrieve
```

### End-to-End Tests

Test the complete user flow:

```text
Upload Documents
 ↓
Ask Question
 ↓
Retrieve
 ↓
Generate
 ↓
Display Answer + Sources
```

---

# 34. Evaluation Architecture

The evaluation system should operate separately from the main application.

```text
                  Evaluation Dataset
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
         Baseline RAG          Enhanced RAG
              │                     │
              └──────────┬──────────┘
                         ▼
                    Evaluation
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
          Retrieval   Generation  Latency
           Metrics     Metrics     Metrics
```

This allows the team to demonstrate measurable improvements as the RAG pipeline evolves.

---

# 35. Architecture Goals

The architecture should satisfy the following principles:

### Modular

Components can be replaced independently.

### Extensible

New document types, embedding models, vector stores, and LLMs can be added.

### Testable

Individual RAG components can be tested independently.

### Observable

Processing states, errors, latency, and evaluation metrics can be measured.

### Reproducible

Another developer should be able to install and run the system using the repository documentation.

### Open Source Friendly

The architecture should remain understandable to external contributors.

---

# 36. Current Architectural Decisions

| Decision | Status |
|---|---|
| Build RAG from scratch | Confirmed |
| Open-source project | Confirmed |
| Python backend | Proposed |
| FastAPI | Proposed |
| React frontend | Proposed |
| Multi-document support | Confirmed |
| Multimodal support | Confirmed as project goal |
| PDF support | Confirmed |
| DOCX support | Confirmed |
| OCR | Confirmed |
| Vector retrieval | Confirmed |
| Hybrid retrieval | Planned |
| Reranking | Planned |
| Source citations | Confirmed |
| LLM training from scratch | Out of scope |
| Advanced vision reasoning | Future / experimental |

---

# 37. Architecture Decision Principles

When choosing or changing a technology, the team should ask:

1. Does it solve a clearly identified project requirement?
2. Can it run within our available hardware/resources?
3. Is it compatible with the rest of the pipeline?
4. Is its license suitable for our open-source project?
5. Can the team understand and maintain it?
6. Can we evaluate its performance?
7. Does it increase project complexity unnecessarily?

The simplest technology that satisfies the requirement should be preferred.

---

# 38. Final Architecture

The intended final architecture is:

```text
┌─────────────────────────────────────────────────────────────┐
│                         USER / FRONTEND                     │
│                  React Web Application                      │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                         FASTAPI API                         │
│        Upload • Documents • Query • Health • Errors         │
└───────────────┬─────────────────────────────┬───────────────┘
                │                             │
                ▼                             ▼
┌─────────────────────────────┐    ┌─────────────────────────┐
│      INGESTION PIPELINE     │    │     QUERY PIPELINE      │
│                             │    │                         │
│ File Validation             │    │ Query Processing        │
│ File Type Detection         │    │ Query Embedding         │
│ PDF/DOCX Parsing            │    │ Semantic Retrieval      │
│ OCR                         │    │ Keyword Retrieval       │
│ Multimodal Processing       │    │ Hybrid Retrieval        │
│ Chunking                    │    │ Reranking               │
│ Metadata                    │    │ Context Assembly        │
└──────────────┬──────────────┘    └────────────┬────────────┘
               │                                │
               ▼                                │
┌─────────────────────────────┐                 │
│     EMBEDDING SERVICE       │                 │
└──────────────┬──────────────┘                 │
               │                                │
               ▼                                │
┌─────────────────────────────┐◄────────────────┘
│        VECTOR STORE         │
│  Embeddings + Chunk Metadata│
└─────────────────────────────┘

                             Query Context
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │     GENERATION LAYER   │
                     │                        │
                     │ Prompt Construction    │
                     │ LLM                    │
                     │ Answer Generation      │
                     └───────────┬────────────┘
                                 │
                                 ▼
                     ┌────────────────────────┐
                     │   CITATION / RESPONSE  │
                     │                        │
                     │ Answer                 │
                     │ Sources                │
                     │ Page References        │
                     └───────────┬────────────┘
                                 │
                                 ▼
                         React Frontend
```

---

## 39. Final Architectural Principle

The system should follow:

> **Ingest → Normalize → Chunk → Embed → Index → Retrieve → Rerank → Generate → Cite → Evaluate**

This pipeline represents the core architecture of MM-RAG.

Advanced capabilities should be added incrementally without compromising the reliability of the core RAG pipeline.

---

**Document Status:** Initial Architecture / Development v1.0  
**Last Updated:** 2026
