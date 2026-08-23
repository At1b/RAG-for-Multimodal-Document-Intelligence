# MM-RAG Development Phases

**Project:** Multi-Modal Multi-Document Retrieval-Augmented Generation System (MM-RAG)  
**Purpose:** Defines the implementation roadmap, milestones, dependencies, deliverables, and completion criteria.  
**Audience:** Human developers, AI coding agents, reviewers, and project supervisors.  
**Version:** 1.0  
**Status:** Development Roadmap

---

## 1. How to Use This Document

This document is the execution roadmap for MM-RAG.

Agents and developers must use this document together with:

```text
PRD.md
Architecture.md
Rules.md
Phases.md
Memory.md
```

Use the documents for different purposes:

```text
PRD.md
→ What the project must achieve

Architecture.md
→ How the system is designed

Rules.md
→ How the system must be developed

Phases.md
→ What should be implemented, and in what order
```

Memory.md
→ Record of implementation decisions, trade-offs, and workarounds
```

### Execution rule

Do not skip ahead to advanced phases unless the required previous phase is working and verified.

A phase may be marked complete only when its acceptance criteria are satisfied.

---

# 2. Overall Roadmap

MM-RAG will be developed incrementally:

```text
PHASE 0
Project Foundation
      ↓
PHASE 1
Document Ingestion
      ↓
PHASE 2
Text Normalization + Chunking
      ↓
PHASE 3
Embeddings + Vector Store
      ↓
PHASE 4
Basic Semantic Retrieval
      ↓
PHASE 5
LLM Generation
      ↓
PHASE 6
End-to-End Baseline RAG
      ↓
PHASE 7
Multi-Document Support + Citations
      ↓
PHASE 8
Hybrid Retrieval
      ↓
PHASE 9
Reranking
      ↓
PHASE 10
OCR + Multimodal Processing
      ↓
PHASE 11
Evaluation
      ↓
PHASE 12
Optimization + Reliability
      ↓
PHASE 13
Frontend + Full Integration
      ↓
PHASE 14
Documentation + Open-Source Release
```

---

# 3. Development Strategy

The project has three broad stages.

## Stage A — Baseline RAG

```text
Documents
 ↓
Parsing
 ↓
Chunking
 ↓
Embeddings
 ↓
Vector Store
 ↓
Semantic Retrieval
 ↓
LLM
 ↓
Answer
```

Goal:

> Build a genuinely working RAG pipeline before adding advanced features.

---

## Stage B — Enhanced RAG

```text
Multi-Document
      ↓
Citations
      ↓
Hybrid Retrieval
      ↓
Reranking
      ↓
OCR / Multimodal
```

Goal:

> Improve retrieval quality, source traceability, and document coverage.

---

## Stage C — Evaluation and Release

```text
Evaluation Dataset
      ↓
Baseline Measurement
      ↓
Enhanced Measurement
      ↓
Optimization
      ↓
Testing
      ↓
Documentation
      ↓
Open-Source Release
```

Goal:

> Demonstrate measurable improvements and make the project reproducible for external contributors.

---

# 4. Phase 0 — Project Foundation

## Objective

Create the repository foundation and development environment.

## Tasks

- [ ] Initialize Git repository.
- [ ] Create project directory structure.
- [ ] Add `PRD.md`.
- [ ] Add `Architecture.md`.
- [ ] Add `Rules.md`.
- [ ] Add `Phases.md`.
- [ ] Create `README.md`.
- [ ] Create `.gitignore`.
- [ ] Create `.env.example`.
- [ ] Create dependency files.
- [ ] Create backend entry point.
- [ ] Create frontend skeleton.
- [ ] Create test structure.
- [ ] Establish formatting/linting strategy.
- [ ] Establish basic CI workflow if practical.

## Expected Structure

```text
MM-RAG/
├── PRD.md
├── Architecture.md
├── Rules.md
├── Phases.md
├── Memory.md
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── backend/
├── frontend/
├── rag/
├── evaluation/
├── tests/
└── docs/
```

## Acceptance Criteria

- Repository initializes successfully.
- Backend starts.
- Frontend starts.
- Tests can be executed.
- No secrets are committed.
- Documentation files are present.

---

# 5. Phase 1 — Document Ingestion

## Objective

Accept supported documents and convert them into a normalized internal representation.

## Initial Supported Formats

Priority:

```text
PDF
 ↓
DOCX
 ↓
Images / Scanned PDFs
```

OCR may initially be implemented as a separate phase.

## Tasks

- [ ] Implement file validation.
- [ ] Implement file-type detection.
- [ ] Implement PDF loader.
- [ ] Implement DOCX loader.
- [ ] Define normalized `Document` representation.
- [ ] Preserve document ID.
- [ ] Preserve document name.
- [ ] Preserve page number where available.
- [ ] Preserve source type.
- [ ] Handle invalid files.
- [ ] Handle unsupported formats.
- [ ] Add unit tests.

## Expected Flow

```text
Uploaded File
      ↓
Validation
      ↓
Format Detection
      ↓
Parser
      ↓
Normalized Document
```

## Acceptance Criteria

- A valid PDF can be processed.
- A valid DOCX can be processed.
- Invalid files are rejected.
- Unsupported files return clear errors.
- Page/source metadata is preserved where available.
- Unit tests pass.

---

# 6. Phase 2 — Normalization and Chunking

## Objective

Convert parsed documents into retrieval-ready chunks while preserving source metadata.

## Tasks

- [ ] Define normalized document schema.
- [ ] Implement text cleaning.
- [ ] Implement chunking strategy.
- [ ] Support configurable chunk size.
- [ ] Support configurable overlap.
- [ ] Preserve metadata.
- [ ] Generate unique chunk IDs.
- [ ] Test edge cases.
- [ ] Test empty/very short documents.
- [ ] Test long documents.

## Expected Flow

```text
Normalized Document
       ↓
Text Cleaning
       ↓
Chunking
       ↓
Chunk + Metadata
```

## Example Chunk

```json
{
  "chunk_id": "chunk_001",
  "document_id": "doc_001",
  "document_name": "report.pdf",
  "page_number": 12,
  "content": "..."
}
```

## Acceptance Criteria

- Documents are split into meaningful chunks.
- Chunk overlap works as configured.
- Metadata survives chunking.
- Every chunk has a unique ID.
- Tests pass.

---

# 7. Phase 3 — Embeddings and Vector Store

## Objective

Convert chunks into vectors and persist them for similarity search.

## Tasks

- [ ] Define embedding interface.
- [ ] Select initial embedding model.
- [ ] Implement embedding generation.
- [ ] Support batch embedding.
- [ ] Select initial vector store.
- [ ] Implement indexing.
- [ ] Store chunk metadata.
- [ ] Implement persistence where required.
- [ ] Implement basic vector lookup.
- [ ] Add tests.

## Expected Flow

```text
Chunks
  ↓
Embedding Model
  ↓
Vectors
  ↓
Vector Store
```

## Acceptance Criteria

- Chunks can be embedded.
- Embeddings can be stored.
- Stored vectors can be queried.
- Results retain chunk metadata.
- Re-indexing behavior is defined.
- Tests pass.

---

# 8. Phase 4 — Basic Semantic Retrieval

## Objective

Retrieve relevant document chunks for a user query using semantic similarity.

## Tasks

- [ ] Define retriever interface.
- [ ] Embed user query.
- [ ] Search vector store.
- [ ] Return top-K results.
- [ ] Return similarity scores where supported.
- [ ] Preserve source metadata.
- [ ] Add retrieval tests.
- [ ] Test relevant and irrelevant queries.

## Expected Flow

```text
User Query
    ↓
Query Embedding
    ↓
Vector Search
    ↓
Top-K Chunks
```

## Acceptance Criteria

- Query can be converted into an embedding.
- Relevant chunks are returned.
- Results include metadata.
- `top_k` is configurable.
- Retrieval can be tested independently.

---

# 9. Phase 5 — LLM Generation

## Objective

Generate an answer from a question and retrieved context.

## Tasks

- [ ] Define generator interface.
- [ ] Select initial LLM.
- [ ] Implement LLM adapter.
- [ ] Create prompt template.
- [ ] Build context input.
- [ ] Add grounding instructions.
- [ ] Handle LLM errors.
- [ ] Add generation tests/mocks.

## Expected Flow

```text
Question
   +
Retrieved Context
   ↓
Prompt Builder
   ↓
LLM
   ↓
Answer
```

## Acceptance Criteria

- Generator accepts a question and context.
- LLM returns a response.
- Context is included in generation.
- LLM failures are handled.
- No hard-coded answers exist in production paths.

---

# 10. Phase 6 — End-to-End Baseline RAG

## Objective

Connect ingestion, chunking, embeddings, retrieval, and generation into one working pipeline.

## Full Flow

```text
Document
   ↓
Parse
   ↓
Normalize
   ↓
Chunk
   ↓
Embed
   ↓
Index
   ↓
User Question
   ↓
Retrieve
   ↓
Context
   ↓
LLM
   ↓
Answer
```

## Tasks

- [ ] Integrate ingestion.
- [ ] Integrate chunking.
- [ ] Integrate embeddings.
- [ ] Integrate vector store.
- [ ] Integrate retrieval.
- [ ] Integrate generation.
- [ ] Implement query service.
- [ ] Add end-to-end tests.
- [ ] Test with real sample documents.
- [ ] Record baseline limitations.

## Acceptance Criteria

A user can:

```text
Upload a document
       ↓
Index it
       ↓
Ask a question
       ↓
Receive a document-grounded answer
```

The baseline must work before Phase 7 begins.

---

# 11. Phase 7 — Multi-Document Support and Citations

## Objective

Allow the system to retrieve information across multiple documents while preserving source attribution.

## Tasks

- [ ] Support multiple indexed documents.
- [ ] Maintain document-level metadata.
- [ ] Filter or search across document collections.
- [ ] Preserve document boundaries.
- [ ] Build citation formatter.
- [ ] Return document name.
- [ ] Return page number where available.
- [ ] Return chunk/source identifier.
- [ ] Test cross-document questions.
- [ ] Test conflicting information between documents.

## Expected Flow

```text
Document A ─┐
Document B ─┼→ Shared Index
Document C ─┘       ↓
                 Retrieval
                    ↓
               Relevant Chunks
                    ↓
                    LLM
                    ↓
              Answer + Sources
```

## Acceptance Criteria

- Multiple documents can be indexed.
- A query can retrieve from multiple documents.
- Sources remain identifiable.
- Citations are based on actual metadata.
- No citation metadata is fabricated.

---

# 12. Phase 8 — Hybrid Retrieval

## Objective

Improve retrieval by combining semantic similarity with keyword-based retrieval.

## Tasks

- [ ] Implement keyword retrieval.
- [ ] Evaluate BM25 or equivalent.
- [ ] Run semantic retrieval.
- [ ] Combine result sets.
- [ ] Define score normalization/combination.
- [ ] Make weights configurable.
- [ ] Add tests.
- [ ] Compare against semantic-only baseline.

## Expected Flow

```text
                 Query
                   │
          ┌────────┴────────┐
          ↓                 ↓
     Semantic Search    Keyword Search
          ↓                 ↓
     Results A          Results B
          └────────┬────────┘
                   ↓
             Hybrid Ranking
                   ↓
              Candidates
```

## Acceptance Criteria

- Both retrieval methods work independently.
- Hybrid retrieval combines them.
- Configuration is possible.
- Results can be compared against the baseline.
- Measurements are recorded before claiming improvement.

---

# 13. Phase 9 — Reranking

## Objective

Improve the relevance ordering of retrieved candidates.

## Tasks

- [ ] Define reranker interface.
- [ ] Select candidate reranking model.
- [ ] Retrieve initial candidate set.
- [ ] Apply reranker.
- [ ] Return final top-K.
- [ ] Preserve metadata.
- [ ] Measure latency.
- [ ] Compare against non-reranked retrieval.

## Expected Flow

```text
Hybrid Retrieval
      ↓
Top-N Candidates
      ↓
Reranker
      ↓
Top-K Context
```

## Acceptance Criteria

- Reranker works independently.
- Final ordering is deterministic enough for testing.
- Source metadata remains intact.
- Retrieval quality is evaluated.
- Latency impact is measured.

---

# 14. Phase 10 — OCR and Multimodal Processing

## Objective

Extend the ingestion pipeline to handle scanned documents and image-based information.

## Initial Scope

```text
Image / Scanned PDF
        ↓
OCR
        ↓
Text
        ↓
Chunking
        ↓
Embedding
        ↓
Retrieval
```

## Tasks

- [ ] Select OCR library/tool.
- [ ] Implement OCR processor.
- [ ] Support scanned PDF pages.
- [ ] Preserve page metadata.
- [ ] Handle OCR failures.
- [ ] Evaluate OCR quality.
- [ ] Add image-document tests.
- [ ] Evaluate table extraction where practical.

## Future Scope

Potential future extensions:

- Vision-language models
- Image embeddings
- Visual question answering
- Chart understanding
- Table-aware retrieval

These should not be implemented automatically unless they are approved as project scope.

## Acceptance Criteria

- A scanned document can be processed.
- OCR text can enter the existing RAG pipeline.
- Source/page metadata remains available.
- OCR failures are handled.
- The implementation is clearly documented.

---

# 15. Phase 11 — Evaluation Framework

## Objective

Create a reproducible way to measure RAG quality.

## Tasks

- [ ] Create evaluation dataset.
- [ ] Define questions.
- [ ] Define expected/reference answers where practical.
- [ ] Define relevant source chunks.
- [ ] Implement retrieval evaluation.
- [ ] Implement generation evaluation.
- [ ] Measure latency.
- [ ] Compare baseline and enhanced systems.
- [ ] Store experiment results.
- [ ] Document methodology.

## Potential Metrics

### Retrieval

```text
Recall@K
Precision@K
MRR
```

### Generation

```text
Faithfulness
Answer Relevance
Context Relevance
```

### System

```text
Latency
Processing Time
Indexing Time
```

## Acceptance Criteria

- Evaluation can be repeated.
- Baseline results exist.
- Enhanced-system results exist.
- Metrics are calculated from actual experiments.
- No fabricated results are used.

---

# 16. Phase 12 — Optimization and Reliability

## Objective

Improve performance, reliability and maintainability based on measured bottlenecks.

## Possible Areas

```text
Embedding speed
Retrieval latency
Reranking latency
LLM latency
Memory usage
Document processing time
```

## Possible Improvements

- Batch embeddings.
- Cache embeddings.
- Optimize chunking.
- Limit candidate count.
- Optimize reranking.
- Stream responses where supported.
- Improve error recovery.
- Improve logging.
- Add health checks.
- Improve configuration management.

## Rule

Do not optimize based only on assumptions.

Use measurements.

## Acceptance Criteria

- Bottlenecks are identified.
- Improvements are measured.
- Regression tests pass.
- No quality degradation is introduced without documentation.

---

# 17. Phase 13 — Frontend and Full Integration

## Objective

Provide a usable interface for the complete system.

## Frontend Features

### Document Management

- [ ] Upload documents.
- [ ] Display documents.
- [ ] Display processing state.
- [ ] Delete documents if supported.

### Query Interface

- [ ] Question input.
- [ ] Submit query.
- [ ] Display answer.
- [ ] Display sources.
- [ ] Display errors.
- [ ] Loading/processing states.

### Future UI

- [ ] Multiple-document selection.
- [ ] Source expansion.
- [ ] Retrieval details.
- [ ] Evaluation dashboard.

## Acceptance Criteria

A user can interact with the system without manually calling backend APIs.

---

# 18. Phase 14 — Documentation and Open-Source Release

## Objective

Prepare MM-RAG for external users and contributors.

## Tasks

- [ ] Complete README.
- [ ] Document installation.
- [ ] Document configuration.
- [ ] Document backend setup.
- [ ] Document frontend setup.
- [ ] Document model requirements.
- [ ] Document vector-store setup.
- [ ] Document testing.
- [ ] Document evaluation.
- [ ] Document architecture.
- [ ] Add contribution guide.
- [ ] Add issue templates if appropriate.
- [ ] Add pull request template if appropriate.
- [ ] Confirm license.
- [ ] Remove secrets and temporary files.
- [ ] Run final tests.
- [ ] Verify clean installation.

## Acceptance Criteria

A new developer can:

```text
Clone
 ↓
Install
 ↓
Configure
 ↓
Run
 ↓
Test
 ↓
Understand
 ↓
Contribute
```

---

# 19. Phase Dependencies

The following dependencies must be respected:

```text
Phase 0
  ↓
Phase 1
  ↓
Phase 2
  ↓
Phase 3
  ↓
Phase 4
  ↓
Phase 5
  ↓
Phase 6
```

After Phase 6:

```text
Phase 6
  ├──→ Phase 7
  │       ↓
  │    Phase 8
  │       ↓
  │    Phase 9
  │
  └──→ Phase 10
          ↓
       Phase 11
          ↓
       Phase 12
          ↓
       Phase 13
          ↓
       Phase 14
```

Some work may occur in parallel after the baseline is stable, but dependencies must not be ignored.

---

# 20. Recommended AI Agent Allocation

For a four-person team, responsibilities can be grouped as follows.

## Member / Agent Group A — Ingestion

```text
Phase 1
Phase 2
Phase 10
```

Focus:

- Parsers
- Normalization
- Chunking
- OCR

---

## Member / Agent Group B — Retrieval

```text
Phase 3
Phase 4
Phase 8
Phase 9
```

Focus:

- Embeddings
- Vector store
- Semantic retrieval
- Hybrid retrieval
- Reranking

---

## Member / Agent Group C — Generation / Backend

```text
Phase 5
Phase 6
Backend APIs
```

Focus:

- LLM abstraction
- Prompting
- Context assembly
- Query service
- API integration

---

## Member / Agent Group D — Frontend / Evaluation

```text
Phase 11
Phase 13
Phase 14
```

Focus:

- Evaluation
- Frontend
- Documentation
- Open-source readiness

### Important

These are responsibility groupings, not isolated systems.

All components must integrate through the interfaces defined in `Architecture.md`.

---

# 21. AI Agent Execution Protocol

Every AI agent should follow this process:

```text
1. Read PRD.md
        ↓
2. Read Architecture.md
        ↓
3. Read Rules.md
        ↓
4. Read Phases.md
        ↓
5. Identify current phase
        ↓
6. Inspect existing repository
        ↓
7. Identify task dependencies
        ↓
8. Implement focused change
        ↓
9. Run tests
        ↓
10. Review changes
        ↓
11. Update documentation if needed
        ↓
12. Report completion + limitations
```

Agents must not assume that a phase is complete merely because code exists.

---

# 22. Phase Completion Template

When completing a phase, record:

```text
Phase:
Status:

Implemented:
- ...
- ...
- ...

Tests:
- ...
- ...

Verification:
- ...

Known Limitations:
- ...

Files Changed:
- ...

Dependencies Added:
- ...

Next Phase:
- ...
```

---

# 23. Status Labels

Use these statuses:

```text
NOT_STARTED
IN_PROGRESS
BLOCKED
IMPLEMENTED
TESTING
COMPLETED
DEFERRED
```

Do not mark a phase `COMPLETED` until its acceptance criteria are satisfied.

---

# 24. Blocked Phase Rules

If a phase is blocked:

1. Identify the exact blocker.
2. Do not hide the blocker.
3. Record the reason.
4. Determine whether another independent task can continue.
5. Avoid implementing unrelated features merely to appear productive.

Example:

```text
Phase 3: BLOCKED

Reason:
Embedding model cannot run within available memory.

Next Action:
Evaluate a smaller embedding model.
```

---

# 25. Change Management

If a new requirement appears:

```text
New Requirement
      ↓
Check PRD
      ↓
Check Architecture
      ↓
Determine Phase
      ↓
Update Documentation
      ↓
Implement
```

Do not silently add major requirements to an existing phase.

---

# 26. Out-of-Scope Unless Explicitly Approved

The following are not required for the initial MVP:

- Training an LLM from scratch.
- Building a foundation model from scratch.
- Fully autonomous AI agents inside the RAG product.
- Advanced distributed infrastructure.
- Large-scale cloud deployment.
- Enterprise authentication/authorization.
- Real-time collaborative editing.
- Advanced visual reasoning.
- Fine-tuning large models without evaluation.

These may be considered later only if project scope, resources and evaluation justify them.

---

# 27. Minimum Viable Completion

The project reaches its first major milestone when the following works:

```text
PDF Upload
    ↓
Text Extraction
    ↓
Chunking
    ↓
Embeddings
    ↓
Vector Index
    ↓
Question
    ↓
Semantic Retrieval
    ↓
Relevant Context
    ↓
LLM
    ↓
Grounded Answer
```

This is the **Baseline RAG**.

Everything after this point should be treated as an enhancement of the baseline.

---

# 28. Final Target

The intended final system is:

```text
             ┌──────────────────────┐
             │       Documents      │
             │ PDF / DOCX / Images  │
             └──────────┬───────────┘
                        ↓
              ┌───────────────────┐
              │ Ingestion + OCR   │
              └─────────┬─────────┘
                        ↓
              ┌───────────────────┐
              │ Normalize + Chunk │
              └─────────┬─────────┘
                        ↓
              ┌───────────────────┐
              │    Embeddings     │
              └─────────┬─────────┘
                        ↓
              ┌───────────────────┐
              │    Vector Store   │
              └─────────┬─────────┘
                        ↑
                        │
User Query → Processing
                        ↓
              ┌───────────────────┐
              │ Hybrid Retrieval  │
              └─────────┬─────────┘
                        ↓
              ┌───────────────────┐
              │     Reranking     │
              └─────────┬─────────┘
                        ↓
              ┌───────────────────┐
              │ Context Assembly  │
              └─────────┬─────────┘
                        ↓
              ┌───────────────────┐
              │       LLM         │
              └─────────┬─────────┘
                        ↓
              ┌───────────────────┐
              │ Answer + Citations│
              └───────────────────┘
                        ↓
                     Frontend
```

---

# 29. Project Completion Checklist

Before final release:

## Documentation

- [ ] PRD complete
- [ ] Architecture complete
- [ ] Rules complete
- [ ] Phases complete
- [ ] README complete
- [ ] Contribution guide complete
- [ ] License added

## Core RAG

- [ ] Document ingestion
- [ ] Chunking
- [ ] Embeddings
- [ ] Vector store
- [ ] Semantic retrieval
- [ ] LLM generation

## Advanced RAG

- [ ] Multi-document
- [ ] Citations
- [ ] Hybrid retrieval
- [ ] Reranking
- [ ] OCR
- [ ] Multimodal support where approved

## Quality

- [ ] Unit tests
- [ ] Integration tests
- [ ] End-to-end tests
- [ ] Evaluation dataset
- [ ] Baseline measurements
- [ ] Enhanced measurements
- [ ] Performance measurements

## Release

- [ ] Secrets removed
- [ ] Installation verified
- [ ] Clean setup verified
- [ ] Documentation verified
- [ ] Open-source license confirmed
- [ ] Contribution process documented

---

# 30. Final Development Rule

> **Complete the baseline before optimizing the baseline. Complete and verify each phase before depending on it. Every improvement must be measurable when possible, and every feature must remain explainable by the team.**

MM-RAG is intended to be a real, maintainable open-source project—not merely a generated demo.

---

**Document Status:** Development Roadmap v1.0  
**Last Updated:** 2026
