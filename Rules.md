# MM-RAG Development Rules

**Project:** Multi-Modal Multi-Document RAG System (MM-RAG)  
**Purpose:** Rules for human developers, AI coding assistants, and AI agents  
**Version:** 1.0

---

## 1. Core Principle

> **Simple → Working → Measurable → Improved**

Build the simplest reliable version first. Do not implement advanced features before the baseline RAG pipeline works.

Development order:

```text
Requirements → Architecture → Implementation → Testing → Measurement → Improvement
```

---

## 2. Source of Truth

Project documentation follows this priority:

1. `PRD.md` — requirements, scope, MVP and goals.
2. `Architecture.md` — system design, components and data flow.
3. `Rules.md` — development, coding, AI-agent and Git rules.
4. Implementation — must follow the above unless a documented decision changes them.

If implementation conflicts with the documentation, do not silently change requirements. Explain the conflict and propose the change.

---

## 3. AI Agent Rules

AI agents are development assistants, not autonomous project owners.

Before modifying code, an agent must:

1. Read the relevant project documentation.
2. Inspect the existing implementation.
3. Identify affected modules.
4. Make a focused plan.
5. Implement the smallest appropriate change.
6. Run relevant tests.
7. Report what changed and what was verified.

Agents must not:

- Rewrite working code unnecessarily.
- Invent requirements.
- Add unrelated features.
- Fabricate test results.
- Claim functionality is complete without verification.
- Add dependencies without justification.
- Commit secrets.
- Replace architecture without documenting the reason.

---

## 4. MVP-First Development

The first target is a working baseline:

```text
PDF
 ↓
Document Parsing
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

After this works, add features incrementally:

```text
Basic RAG
 ↓
Multi-document
 ↓
Source citations
 ↓
Hybrid retrieval
 ↓
Reranking
 ↓
OCR / images
 ↓
Evaluation
 ↓
Optimization
```

---

## 5. No Blind Code Generation

Never generate a large amount of code without understanding the repository.

Use:

```text
Read docs
 ↓
Inspect repository
 ↓
Identify affected code
 ↓
Plan
 ↓
Implement
 ↓
Test
 ↓
Review
```

Prefer small, reversible changes.

---

## 6. No Unnecessary Rewrites

Do not rewrite a complete module simply because another implementation looks cleaner.

A rewrite is justified only when:

- The existing design prevents required functionality.
- There is a serious architectural problem.
- The current implementation is demonstrably incorrect.
- The team explicitly approves it.

---

## 7. Modular Architecture

Keep major responsibilities separate:

```text
Ingestion
Processing
Chunking
Embeddings
Storage
Retrieval
Reranking
Generation
Citations
Evaluation
```

A module should not silently take over another module's responsibility.

---

## 8. Interface-Based Design

Major components should expose clear interfaces.

Example:

```python
class DocumentProcessor:
    def process(self, file_path):
        ...
```

```python
class Retriever:
    def retrieve(self, query, top_k=5):
        ...
```

```python
class Generator:
    def generate(self, query, context):
        ...
```

Interfaces should allow implementations to be replaced without rewriting the entire system.

---

## 9. Technology Selection

Do not add a library just because it is popular.

Before adding a dependency, ask:

1. Does it solve a real requirement?
2. Is it compatible with the architecture?
3. Is it maintained?
4. Is its license appropriate?
5. Can the team understand it?
6. Does it introduce unnecessary complexity?
7. Can it run with available resources?

Prefer the simplest suitable technology.

---

## 10. Dependency Rules

Before adding a package:

- Check whether the functionality already exists.
- Check whether an existing dependency can provide it.
- Prefer established libraries.
- Record important dependency choices.
- Avoid multiple libraries solving the same problem unless they are being evaluated.

---

## 11. Configuration and Secrets

Do not hard-code:

- API keys
- Passwords
- Tokens
- Database credentials
- Model secrets

Use environment variables:

```text
.env
.env.example
```

Never commit `.env`.

The example file should contain placeholders:

```text
LLM_API_KEY=your_api_key_here
```

---

## 12. Document Processing

The ingestion pipeline should follow:

```text
Input File
 ↓
Validation
 ↓
Format Detection
 ↓
Parser / OCR
 ↓
Normalized Document
 ↓
Chunking
 ↓
Metadata
```

Document-specific parsing must remain separate from retrieval and generation logic.

---

## 13. Chunking

Chunks must preserve enough context to remain useful for retrieval.

Where available, preserve:

```text
document_id
document_name
page_number
chunk_id
section
source_type
```

Never discard source information required for citations.

---

## 14. Embeddings

Embedding generation should be isolated behind an embedding service/interface:

```text
Text
 ↓
Embedding Service
 ↓
Vector
```

The embedding implementation should not be tightly coupled to the vector store.

---

## 15. Vector Store

The vector store should maintain or reference:

```text
Embedding
+
Chunk
+
Metadata
```

Every vector must remain traceable to its source chunk.

Candidate technologies include:

- FAISS
- Chroma
- Qdrant

The final choice must be based on actual project requirements and testing.

---

## 16. Retrieval

Initial retrieval should prioritize semantic relevance.

Later:

```text
Semantic Search
      +
Keyword Search
      ↓
Hybrid Retrieval
```

Do not increase `top_k` simply to hide poor retrieval quality. Investigate chunking, embeddings, query processing, retrieval and reranking instead.

---

## 17. Reranking

Reranking should operate on a manageable candidate set:

```text
Initial Retrieval
 ↓
Top 20 Candidates
 ↓
Reranker
 ↓
Top 5 Context Chunks
```

Do not send unnecessarily large context to the LLM.

---

## 18. LLM Rules

The LLM is a generation component, not the document database.

For document-specific questions:

```text
Question + Retrieved Evidence
              ↓
             LLM
              ↓
            Answer
```

The project will **not train an LLM from scratch**.

The LLM should be abstracted so local and API-based models can be changed without rewriting the RAG pipeline.

---

## 19. Hallucination Control

Reduce unsupported answers by:

- Retrieving relevant evidence.
- Providing retrieved context to the LLM.
- Using grounding instructions.
- Providing source references.
- Evaluating faithfulness.

If the context is insufficient, prefer an uncertainty/insufficient-information response rather than inventing information.

---

## 20. Citation Rules

Where available, retrieved chunks must preserve:

```text
Document
Page
Section
Chunk ID
```

Example:

```text
Answer:
Revenue increased by approximately 25%.

Sources:
[1] Annual_Report_2024.pdf — Page 38
```

Never invent document names, page numbers or sections.

---

## 21. Multimodal Rules

Multimodal development must be incremental.

Initial priority:

```text
Image / Scanned PDF
        ↓
       OCR
        ↓
   Extracted Text
        ↓
     RAG Pipeline
```

Advanced vision-language processing should be added only after the basic pipeline is stable.

Do not claim the system understands images unless actual visual processing has been implemented.

---

## 22. Multi-Document Rules

Documents must remain identifiable throughout the pipeline.

```text
Document A
Document B
Document C
```

must not be merged in a way that makes source attribution impossible.

---

## 23. API Rules

API routes should remain thin:

```text
Request
 ↓
Validation
 ↓
Service Call
 ↓
Response
```

Core RAG logic should reside in services/modules rather than route handlers.

Example:

```text
POST /query
 ↓
QueryService
 ↓
Retriever
 ↓
Reranker
 ↓
Generator
 ↓
Citation Formatter
```

---

## 24. Error Handling

Do not silently ignore failures.

Handle cases such as:

```text
Invalid File
Unsupported Format
OCR Failure
Embedding Failure
Vector Store Failure
LLM Failure
```

Errors should provide useful debugging information without exposing sensitive document content.

---

## 25. Logging

Logs may record:

- Processing failures
- Retrieval failures
- API failures
- Latency
- System state

Do not log unnecessarily:

- API keys
- Passwords
- Full private documents
- Sensitive user content

Prefer identifiers such as:

```text
document_id
request_id
chunk_id
```

---

## 26. Testing

Important modules must have tests.

### Unit tests

```text
Parser
Chunker
Embedding Service
Retriever
Reranker
Citation Formatter
```

### Integration tests

```text
Upload
 ↓
Process
 ↓
Chunk
 ↓
Index
 ↓
Retrieve
```

### End-to-end tests

```text
Upload Documents
 ↓
Ask Question
 ↓
Retrieve
 ↓
Generate
 ↓
Answer + Sources
```

---

## 27. Never Fake Test Results

An AI agent must never claim tests passed unless it actually ran them.

Good:

```text
Implemented PDF loader.
Tests: 8 passed, 0 failed.
```

If tests were not run:

```text
Implemented PDF loader.
Tests were not run because <reason>.
```

Never fabricate test output.

---

## 28. Evaluation

The project should eventually demonstrate measurable performance.

Possible retrieval metrics:

- Recall@K
- Precision@K
- MRR

Possible generation metrics:

- Faithfulness
- Answer relevance
- Context relevance

Possible system metrics:

- Response latency
- Processing time
- Indexing time

Do not claim an approach is better without measurements or clearly labeled qualitative reasoning.

---

## 29. Baseline vs Enhanced RAG

Maintain a baseline for comparison.

### Baseline

```text
Chunking
 ↓
Embedding
 ↓
Vector Search
 ↓
LLM
```

### Enhanced

```text
Chunking
 ↓
Embedding
 ↓
Semantic + Keyword Retrieval
 ↓
Reranking
 ↓
LLM
```

This lets the team demonstrate whether improvements actually work.

---

## 30. Code Quality

Code should be:

- Readable
- Modular
- Consistent
- Testable
- Maintainable

Avoid:

- Giant functions
- Giant files
- Magic numbers
- Duplicate logic
- Dead code
- Unused imports
- Excessive abstraction

Do not over-engineer features before they are needed.

---

## 31. Naming

Use descriptive names.

Prefer:

```python
document_processor
embedding_service
retrieval_service
citation_formatter
```

over:

```python
dp
es
rs
cf
```

Use consistent naming conventions throughout the repository.

---

## 32. Comments and Documentation

Comments should explain **why**, not merely repeat what the code does.

Document non-obvious architectural decisions.

Keep documentation synchronized with actual behavior.

---

## 33. Git Rules

Use Git throughout development.

Recommended structure:

```text
main
 └── develop
      ├── feature/document-ingestion
      ├── feature/retrieval
      ├── feature/reranking
      └── feature/frontend
```

The exact workflow can be simplified if the team prefers.

---

## 34. Commit Rules

Use meaningful commits.

Good:

```text
feat: add PDF document loader
feat: implement semantic retrieval
fix: preserve page metadata during chunking
test: add retriever unit tests
docs: update architecture
```

Avoid:

```text
update
changes
final
final2
working
```

Keep unrelated changes out of the same commit where practical.

---

## 35. Pull Request Rules

A PR should explain:

1. What changed?
2. Why?
3. Which modules were affected?
4. How was it tested?
5. What limitations remain?

Example:

```text
## Summary
Implemented PDF ingestion and page-level metadata.

## Testing
8 unit tests passed.

## Limitations
Tables are currently extracted as plain text.
```

---

## 36. AI-Generated Code

AI-generated code is allowed.

However, generated code must:

1. Be reviewed by a team member.
2. Be understood before acceptance.
3. Follow the architecture.
4. Have appropriate tests.
5. Not introduce unexplained dependencies.
6. Not contain copied proprietary code.
7. Not expose secrets.
8. Not be accepted merely because it compiles.

The team remains responsible for all committed code.

---

## 37. AI Agent Task Context

When assigning an AI agent a task, provide:

```text
Project Goal
Relevant Documentation
Target Module
Expected Behavior
Constraints
Testing Requirements
```

Example:

```text
Read:
- PRD.md
- Architecture.md
- Rules.md

Task:
Implement the PDF loader.

Constraints:
- Preserve page numbers.
- Return normalized Document objects.
- Add unit tests.
- Do not modify retrieval code.
```

---

## 38. Agent Handoff

Before an agent hands work to another agent:

```text
Code complete
 ↓
Tests run
 ↓
Documentation updated
 ↓
Known limitations recorded
 ↓
Repository ready for next agent
```

Do not leave unexplained temporary code or generated files.

---

## 39. No Fake Features

Do not create hardcoded implementations that appear functional.

For example, this is not acceptable in production:

```python
def retrieve(query):
    return ["hardcoded answer"]
```

Mocks are allowed only when clearly isolated for tests.

---

## 40. No Fake Evaluation

Never invent:

- Accuracy percentages
- Retrieval scores
- Latency numbers
- Benchmark results
- User-study results
- Dataset statistics

If an experiment has not been run, state:

```text
Not evaluated yet.
```

---

## 41. No Fake Citations

Never fabricate:

- Documents
- Page numbers
- Sections
- URLs
- References

Citation metadata must originate from the actual document processing pipeline.

---

## 42. Reproducibility

The repository should eventually document:

```text
Installation
Environment Setup
Model Setup
Vector Store Setup
Backend Setup
Frontend Setup
Testing
Evaluation
```

Another developer should be able to understand how to run the project from the repository.

---

## 43. Scope Control

Do not automatically implement suggestions that are outside the current milestone.

Record future ideas rather than continuously expanding the current task.

```text
Current Task
 ↓
Complete
 ↓
Test
 ↓
Document
 ↓
Next Task
```

---

## 44. Definition of Done

A feature is complete only when:

- [ ] Requirements are clear.
- [ ] Implementation is complete.
- [ ] Relevant tests exist.
- [ ] Tests pass or limitations are documented.
- [ ] Error handling is considered.
- [ ] Documentation is updated where required.
- [ ] No secrets are committed.
- [ ] Code follows the architecture.
- [ ] Known limitations are recorded.

---

## 45. Current Development Priority

### Baseline

```text
1. Project Setup
2. PDF Ingestion
3. Text Extraction
4. Chunking
5. Embeddings
6. Vector Store
7. Semantic Retrieval
8. LLM Generation
9. Basic End-to-End RAG
10. Testing
```

### After Baseline

```text
11. Multi-Document
12. Source Citations
13. Hybrid Retrieval
14. Reranking
15. OCR
16. Multimodal Processing
17. Evaluation
18. Optimization
19. Documentation
20. Open-Source Release
```

---

## 46. Final Rule

> **Never optimize what has not been measured, never add complexity without a requirement, and never claim functionality that has not been tested.**

The goal is not to produce the largest RAG system.

The goal is to produce a **well-designed, understandable, testable, measurable, and genuinely open-source RAG system** that the team can explain, maintain, and improve.

---

**Document Status:** Development Rules v1.0  
**Last Updated:** 2026
