"""Phase 8 — Retrieval Evaluation.

Deterministic evaluation dataset and measurement script to compare:
    - Semantic-only retrieval
    - Keyword-only (BM25) retrieval
    - Hybrid (RRF) retrieval

Metrics:
    - Hit@K: Whether an expected document appears in top-K results.
    - Recall@K: Fraction of expected documents found in top-K results.
    - Precision@K: Fraction of retrieved results that are relevant.
    - MRR (Mean Reciprocal Rank): Reciprocal rank of the first relevant result.

This is a Phase 8 baseline experiment.  The full formal evaluation
framework is deferred to Phase 11.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from rag.chunking.chunker import ChunkingConfig
from rag.embeddings.indexing import IndexingService
from rag.embeddings.sentence_transformer import SentenceTransformerEmbeddingService
from rag.ingestion.service import IngestionService
from rag.orchestration.indexing_service import DocumentIndexingService
from rag.retrieval.base import Retriever
from rag.retrieval.bm25 import BM25Retriever
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.index_builder import build_bm25_index_from_chroma
from rag.retrieval.semantic import SemanticRetriever
from rag.vectorstore.chroma_store import ChromaVectorStore
from rag.vectorstore.models import VectorSearchResult

# Repository root path for locating fixtures
_REPO_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


@dataclass
class EvalQuery:
    """A single evaluation query with expected results."""

    query: str
    description: str
    expected_doc_names: list[str] = field(default_factory=list)
    expected_chunk_ids: list[str] = field(default_factory=list)
    expected_doc_ids: list[str] = field(default_factory=list)
    category: str = "general"  # semantic, keyword, paraphrase, multi_doc, irrelevant


@dataclass
class EvalResult:
    """Result of evaluating a single query."""

    query: str
    category: str
    hit_at_k: bool  # Expected document/chunk found in top-K
    recall_at_k: float  # Fraction of expected documents found
    precision_at_k: float  # Fraction of retrieved results that are relevant
    reciprocal_rank: float  # 1 / rank of first relevant result
    retrieved_chunk_ids: list[str]
    retrieved_doc_names: list[str]
    num_results: int


@dataclass
class EvalSummary:
    """Summary of evaluation across all queries."""

    mode: str
    top_k: int
    total_queries: int
    hit_rate: float
    avg_recall: float
    avg_precision: float
    mrr: float
    per_category: dict[str, dict[str, float]]
    details: list[EvalResult]


def build_sample_eval_dataset() -> list[EvalQuery]:
    """Build a deterministic evaluation dataset for the sample fixture documents.

    Covers 5 distinct categories:
        1. semantic: Queries requiring conceptual understanding where keywords differ
        2. keyword: Queries with exact identifiers, acronyms, or proper nouns
        3. paraphrase: Rephrased inquiries testing semantic generalization
        4. multi_doc: Queries requiring evidence from more than one document
        5. irrelevant: Out-of-domain queries where relevance gate must block results
    """
    return [
        # --- 1. Semantic queries ---
        EvalQuery(
            query=(
                "What tasks typically require human intelligence in computer systems?"
            ),
            description="High-level definition of AI tasks from overview",
            expected_doc_names=["sample_ai_overview.pdf"],
            category="semantic",
        ),
        EvalQuery(
            query="How do artificial neural networks learn representations from data?",
            description="Deep learning representation learning",
            expected_doc_names=["sample_ai_overview.pdf"],
            category="semantic",
        ),
        # --- 2. Keyword queries ---
        EvalQuery(
            query="Transformers attention mechanism",
            description="Specific technical terms in AI document",
            expected_doc_names=["sample_ai_overview.pdf"],
            category="keyword",
        ),
        EvalQuery(
            query="City One population 1850",
            description="Specific numbers and founding year in Country Alpha",
            expected_doc_names=["country_alpha.pdf"],
            category="keyword",
        ),
        EvalQuery(
            query="City Two 3.5 million southern coast 1920",
            description="Specific entities and figures in Country Beta",
            expected_doc_names=["country_beta.pdf"],
            category="keyword",
        ),
        # --- 3. Paraphrase queries ---
        EvalQuery(
            query="What algorithms enable computers to improve through experience?",
            description="Paraphrase of machine learning definition",
            expected_doc_names=["sample_ai_overview.pdf"],
            category="paraphrase",
        ),
        EvalQuery(
            query=(
                "Where is the capital of Alpha located and how many people live there?"
            ),
            description="Paraphrase of Country Alpha capital and demographics",
            expected_doc_names=["country_alpha.pdf"],
            category="paraphrase",
        ),
        # --- 4. Multi-document queries ---
        EvalQuery(
            query=(
                "What are the capitals and populations of Country Alpha and"
                " Country Beta?"
            ),
            description=(
                "Cross-document comparison requiring evidence from both nations"
            ),
            expected_doc_names=["country_alpha.pdf", "country_beta.pdf"],
            category="multi_doc",
        ),
        EvalQuery(
            query="Compare the founding years and official languages of Alpha and Beta",
            description="Cross-document historical and linguistic comparison",
            expected_doc_names=["country_alpha.pdf", "country_beta.pdf"],
            category="multi_doc",
        ),
        # --- 5. Irrelevant queries (completely out-of-domain across all documents) ---
        EvalQuery(
            query="What is the best recipe for baking chocolate cake?",
            description="Culinary query entirely outside document corpus",
            expected_doc_names=[],
            category="irrelevant",
        ),
        EvalQuery(
            query="Who won the 2024 Olympic marathon gold medal?",
            description="Sports query absent from all documents",
            expected_doc_names=[],
            category="irrelevant",
        ),
        EvalQuery(
            query="What causes volcanic eruptions on Jupiter's moon Io?",
            description=(
                "Astrophysics query with zero overlap across all indexed documents"
            ),
            expected_doc_names=[],
            category="irrelevant",
        ),
    ]


def evaluate_retriever(
    retriever: Retriever,
    eval_queries: list[EvalQuery],
    top_k: int = 5,
    mode: str = "unknown",
) -> EvalSummary:
    """Evaluate a retriever against an evaluation dataset.

    Args:
        retriever: The retriever to evaluate.
        eval_queries: List of evaluation queries.
        top_k: Number of results to retrieve per query.
        mode: Label for this evaluation run.

    Returns:
        EvalSummary containing overall and per-category metrics.
    """
    results: list[EvalResult] = []

    for eq in eval_queries:
        try:
            retrieved: list[VectorSearchResult] = retriever.retrieve(
                eq.query, top_k=top_k
            )
        except Exception as exc:
            logger.warning("Retrieval failed for query '%s': %s", eq.query, exc)
            retrieved = []

        retrieved_chunk_ids = [r.chunk_id for r in retrieved]
        retrieved_doc_names = [r.document_name for r in retrieved]

        is_irrelevant_query = (
            len(eq.expected_doc_names) == 0 and len(eq.expected_chunk_ids) == 0
        )

        if is_irrelevant_query:
            # For irrelevant queries, goal is returning NO chunks.
            if len(retrieved) == 0:
                hit = True
                recall = 1.0
                precision = 1.0
                rr = 1.0
            else:
                hit = False
                recall = 0.0
                precision = 0.0
                rr = 0.0
        else:
            # For relevant queries, verify expected documents appear
            expected_set = set(eq.expected_doc_names)
            hits = [name for name in retrieved_doc_names if name in expected_set]
            unique_hits = set(hits)

            hit = len(unique_hits) > 0
            recall = len(unique_hits) / len(expected_set) if expected_set else 0.0
            precision = len(hits) / len(retrieved) if retrieved else 0.0

            # Reciprocal rank of first relevant result
            rr = 0.0
            for rank_0, name in enumerate(retrieved_doc_names):
                if name in expected_set:
                    rr = 1.0 / (rank_0 + 1)
                    break

        results.append(
            EvalResult(
                query=eq.query,
                category=eq.category,
                hit_at_k=hit,
                recall_at_k=recall,
                precision_at_k=precision,
                reciprocal_rank=rr,
                retrieved_chunk_ids=retrieved_chunk_ids,
                retrieved_doc_names=retrieved_doc_names,
                num_results=len(retrieved),
            )
        )

    total = len(results)
    hit_rate = sum(1 for r in results if r.hit_at_k) / total if total > 0 else 0.0
    avg_recall = sum(r.recall_at_k for r in results) / total if total > 0 else 0.0
    avg_precision = sum(r.precision_at_k for r in results) / total if total > 0 else 0.0
    mrr = sum(r.reciprocal_rank for r in results) / total if total > 0 else 0.0

    # Per-category metrics
    categories: dict[str, list[EvalResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    per_category: dict[str, dict[str, float]] = {}
    for cat, cat_results in categories.items():
        n = len(cat_results)
        per_category[cat] = {
            "hit_rate": sum(1 for r in cat_results if r.hit_at_k) / n,
            "avg_recall": sum(r.recall_at_k for r in cat_results) / n,
            "avg_precision": sum(r.precision_at_k for r in cat_results) / n,
            "mrr": sum(r.reciprocal_rank for r in cat_results) / n,
            "count": float(n),
        }

    return EvalSummary(
        mode=mode,
        top_k=top_k,
        total_queries=total,
        hit_rate=hit_rate,
        avg_recall=avg_recall,
        avg_precision=avg_precision,
        mrr=mrr,
        per_category=per_category,
        details=results,
    )


def format_eval_report(summaries: list[EvalSummary]) -> str:
    """Format evaluation summaries into a human-readable comparison report."""
    lines = [
        "=" * 70,
        "Phase 8 — Retrieval Benchmark Comparison Report",
        "=" * 70,
        "",
    ]

    for s in summaries:
        lines.append(f"--- Mode: {s.mode.upper()} ---")
        lines.append(f"  Top-K:          {s.top_k}")
        lines.append(f"  Total Queries:  {s.total_queries}")
        lines.append(f"  Hit@{s.top_k}:         {s.hit_rate:.2%}")
        lines.append(f"  Recall@{s.top_k}:      {s.avg_recall:.2%}")
        lines.append(f"  Precision@{s.top_k}:   {s.avg_precision:.2%}")
        lines.append(f"  MRR:            {s.mrr:.4f}")
        lines.append("")
        lines.append("  Per-Category Performance:")
        for cat, metrics in sorted(s.per_category.items()):
            lines.append(
                f"    {cat:<12s} (n={int(metrics['count'])}): "
                f"Hit={metrics['hit_rate']:>6.1%}  "
                f"Recall={metrics['avg_recall']:>6.1%}  "
                f"Prec={metrics['avg_precision']:>6.1%}  "
                f"MRR={metrics['mrr']:>6.4f}"
            )
        lines.append("")

    if len(summaries) > 1:
        lines.append("=" * 70)
        lines.append("Comparative Summary Table")
        lines.append("=" * 70)
        header = f"{'Metric':<20s}"
        for s in summaries:
            header += f"  {s.mode.upper():<14s}"
        lines.append(header)
        lines.append("-" * 70)

        for metric_name, attr in [
            (f"Hit@{summaries[0].top_k}", "hit_rate"),
            (f"Recall@{summaries[0].top_k}", "avg_recall"),
            (f"Precision@{summaries[0].top_k}", "avg_precision"),
            ("MRR", "mrr"),
        ]:
            row = f"{metric_name:<20s}"
            for s in summaries:
                val = getattr(s, attr)
                if attr == "mrr":
                    row += f"  {val:<14.4f}"
                else:
                    row += f"  {val:<14.2%}"
            lines.append(row)
        lines.append("-" * 70)

    return "\n".join(lines)


def run_benchmark(top_k: int = 5) -> list[EvalSummary]:
    """Execute the full deterministic retrieval benchmark across all three modes.

    Indexes sample fixtures into an isolated temporary vector store,
    builds the three retriever variants, evaluates them across the standard
    queries, and returns the EvalSummary list.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        vector_store = ChromaVectorStore(
            persist_directory=temp_dir,
            collection_name="eval_benchmark",
        )
        embedding_service = SentenceTransformerEmbeddingService(
            model_name="all-MiniLM-L6-v2",
            batch_size=32,
        )
        indexing_service = IndexingService(
            embedding_service=embedding_service,
            vector_store=vector_store,
        )
        doc_indexing_service = DocumentIndexingService(
            ingestion_service=IngestionService(max_size_bytes=50 * 1024 * 1024),
            chunking_config=ChunkingConfig(chunk_size=1000, chunk_overlap=200),
            indexing_service=indexing_service,
        )

        fixtures = [
            _REPO_ROOT / "tests" / "fixtures" / "sample_ai_overview.pdf",
            _REPO_ROOT / "tests" / "fixtures" / "country_alpha.pdf",
            _REPO_ROOT / "tests" / "fixtures" / "country_beta.pdf",
        ]

        logger.info(
            "Indexing %d fixture documents for evaluation benchmark...", len(fixtures)
        )
        for fix in fixtures:
            doc_indexing_service.index_document(fix, document_name=fix.name)

        # 1. Semantic-only retriever
        semantic_retriever = SemanticRetriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
            default_top_k=top_k,
            min_score=0.30,
        )

        # 2. Sparse-only BM25 retriever
        bm25_index = build_bm25_index_from_chroma(vector_store)
        keyword_retriever = BM25Retriever(
            bm25_index=bm25_index,
            default_top_k=top_k,
        )

        # 3. Hybrid retriever (semantic min_score=0.30 preserves relevance gate)
        hybrid_semantic = SemanticRetriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
            default_top_k=20,
            min_score=0.30,
        )
        hybrid_retriever = HybridRetriever(
            semantic_retriever=hybrid_semantic,
            keyword_retriever=keyword_retriever,
            semantic_weight=1.0,
            keyword_weight=1.0,
            rrf_k=60,
            default_top_k=top_k,
            semantic_top_k=20,
            keyword_top_k=20,
        )

        eval_queries = build_sample_eval_dataset()

        summary_sem = evaluate_retriever(
            semantic_retriever, eval_queries, top_k=top_k, mode="semantic"
        )
        summary_bm25 = evaluate_retriever(
            keyword_retriever, eval_queries, top_k=top_k, mode="keyword"
        )
        summary_hybrid = evaluate_retriever(
            hybrid_retriever, eval_queries, top_k=top_k, mode="hybrid"
        )

        return [summary_sem, summary_bm25, summary_hybrid]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("Running Phase 8 Retrieval Benchmark...")
    summaries = run_benchmark(top_k=5)
    report = format_eval_report(summaries)
    print("\n" + report)
