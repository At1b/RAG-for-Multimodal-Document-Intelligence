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
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

# Ensure repository root is on sys.path when script is executed directly
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rag.chunking.chunker import ChunkingConfig
from rag.embeddings.indexing import IndexingService
from rag.embeddings.sentence_transformer import SentenceTransformerEmbeddingService
from rag.ingestion.service import IngestionService
from rag.orchestration.indexing_service import DocumentIndexingService
from rag.reranking.cross_encoder import CrossEncoderReranker
from rag.retrieval.base import Retriever
from rag.retrieval.bm25 import BM25Retriever
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.index_builder import build_bm25_index_from_chroma
from rag.retrieval.reranking import RerankingRetriever
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
    is_irrelevant: bool
    hit_at_k: bool  # Expected document found in top-K (False for irrelevant)
    recall_at_k: float  # Fraction of expected documents found
    precision_at_k: float  # Fraction of retrieved results that are relevant
    reciprocal_rank: float  # 1 / rank of first relevant result
    rejected: bool  # True if 0 chunks retrieved (gate rejection for irrelevant)
    retrieved_chunk_ids: list[str]
    retrieved_doc_names: list[str]
    num_results: int
    latency_ms: float = 0.0


@dataclass
class EvalSummary:
    """Summary of evaluation across all queries."""

    mode: str
    top_k: int
    total_queries: int
    relevant_queries: int
    irrelevant_queries: int
    hit_rate: float  # Aggregated strictly over relevant queries
    avg_recall: float  # Aggregated strictly over relevant queries
    avg_precision: float  # Aggregated strictly over relevant queries
    mrr: float  # Aggregated strictly over relevant queries
    rejection_rate: float  # Rejection rate over irrelevant queries (0 chunks)
    median_latency_ms: float
    p95_latency_ms: float
    per_category: dict[str, dict[str, float]]
    details: list[EvalResult]
    llm_prevented: bool = True


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
        t0 = time.perf_counter()
        try:
            retrieved: list[VectorSearchResult] = retriever.retrieve(
                eq.query, top_k=top_k
            )
        except Exception as exc:
            logger.warning("Retrieval failed for query '%s': %s", eq.query, exc)
            retrieved = []
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        retrieved_chunk_ids = [r.chunk_id for r in retrieved]
        retrieved_doc_names = [r.document_name for r in retrieved]

        is_irrelevant_query = (
            len(eq.expected_doc_names) == 0 and len(eq.expected_chunk_ids) == 0
        )

        if is_irrelevant_query:
            # For irrelevant queries, empty retrieval represents correct rejection
            # by the relevance gate, not a positive retrieval "hit".
            rejected = len(retrieved) == 0
            hit = False
            recall = 0.0
            precision = 0.0
            rr = 0.0
        else:
            rejected = len(retrieved) == 0
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
                is_irrelevant=is_irrelevant_query,
                hit_at_k=hit,
                recall_at_k=recall,
                precision_at_k=precision,
                reciprocal_rank=rr,
                rejected=rejected,
                retrieved_chunk_ids=retrieved_chunk_ids,
                retrieved_doc_names=retrieved_doc_names,
                num_results=len(retrieved),
                latency_ms=elapsed_ms,
            )
        )

    # Separate relevant and irrelevant queries
    relevant_results = [r for r in results if not r.is_irrelevant]
    irrelevant_results = [r for r in results if r.is_irrelevant]

    num_rel = len(relevant_results)
    num_irrel = len(irrelevant_results)

    # Ranking-quality metrics aggregated strictly across relevant queries
    hit_rate = (
        sum(1 for r in relevant_results if r.hit_at_k) / num_rel if num_rel > 0 else 0.0
    )
    avg_recall = (
        sum(r.recall_at_k for r in relevant_results) / num_rel if num_rel > 0 else 0.0
    )
    avg_precision = (
        sum(r.precision_at_k for r in relevant_results) / num_rel
        if num_rel > 0
        else 0.0
    )
    mrr = (
        sum(r.reciprocal_rank for r in relevant_results) / num_rel
        if num_rel > 0
        else 0.0
    )

    # Relevance-gate rejection rate aggregated strictly across irrelevant queries
    rejection_rate = (
        sum(1 for r in irrelevant_results if r.rejected) / num_irrel
        if num_irrel > 0
        else 1.0
    )

    latencies = [r.latency_ms for r in results]
    latencies.sort()
    mid = len(latencies) // 2
    if len(latencies) == 0:
        median_lat = 0.0
        p95_lat = 0.0
    elif len(latencies) % 2 == 0:
        median_lat = (latencies[mid - 1] + latencies[mid]) / 2.0
        p95_lat = latencies[min(int(0.95 * len(latencies)), len(latencies) - 1)]
    else:
        median_lat = latencies[mid]
        p95_lat = latencies[min(int(0.95 * len(latencies)), len(latencies) - 1)]

    # Per-category metrics
    categories: dict[str, list[EvalResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    per_category: dict[str, dict[str, float]] = {}
    for cat, cat_results in categories.items():
        n = len(cat_results)
        if cat == "irrelevant":
            per_category[cat] = {
                "count": float(n),
                "rejection_rate": sum(1 for r in cat_results if r.rejected) / n,
                "avg_chunks_returned": sum(r.num_results for r in cat_results) / n,
            }
        else:
            per_category[cat] = {
                "count": float(n),
                "hit_rate": sum(1 for r in cat_results if r.hit_at_k) / n,
                "avg_recall": sum(r.recall_at_k for r in cat_results) / n,
                "avg_precision": sum(r.precision_at_k for r in cat_results) / n,
                "mrr": sum(r.reciprocal_rank for r in cat_results) / n,
            }

    return EvalSummary(
        mode=mode,
        top_k=top_k,
        total_queries=len(results),
        relevant_queries=num_rel,
        irrelevant_queries=num_irrel,
        hit_rate=hit_rate,
        avg_recall=avg_recall,
        avg_precision=avg_precision,
        mrr=mrr,
        rejection_rate=rejection_rate,
        median_latency_ms=median_lat,
        p95_latency_ms=p95_lat,
        per_category=per_category,
        details=results,
    )


def format_eval_report(
    summaries: list[EvalSummary], cold_start_s: float | None = None
) -> str:
    """Format evaluation summaries into a human-readable comparison report."""
    lines = [
        "=" * 120,
        "Phase 9 — Retrieval & Reranking Benchmark Comparison Report",
        "=" * 120,
    ]
    if cold_start_s is not None:
        lines.append(f"Model Cold-Start Initialization: {cold_start_s:.3f} s")
    if summaries and hasattr(summaries[0], "llm_prevented"):
        gate_status = (
            "VERIFIED (100% of irrelevant queries blocked before LLM across all modes)"
            if all(getattr(s, "llm_prevented", False) for s in summaries)
            else "FAILED"
        )
        lines.append(f"Relevance Gate Orchestration:   {gate_status}")
    lines.append("")

    for s in summaries:
        lines.append(f"--- Mode: {s.mode.upper()} ---")
        lines.append(f"  Top-K:                     {s.top_k}")
        lines.append(f"  Total Queries:             {s.total_queries}")
        lines.append(f"  Relevant Queries:          {s.relevant_queries}")
        lines.append(f"  Hit@{s.top_k} (Relevant):         {s.hit_rate:.2%}")
        lines.append(f"  Recall@{s.top_k} (Relevant):      {s.avg_recall:.2%}")
        lines.append(f"  Precision@{s.top_k} (Relevant):   {s.avg_precision:.2%}")
        lines.append(f"  MRR (Relevant):            {s.mrr:.4f}")
        lines.append(f"  Irrelevant Queries:        {s.irrelevant_queries}")
        lines.append(f"  Rejection Rate (Irrel):    {s.rejection_rate:.2%}")
        lines.append(f"  Latency (Median):          {s.median_latency_ms:.2f} ms")
        lines.append(f"  Latency (p95):             {s.p95_latency_ms:.2f} ms")
        lines.append("")
        lines.append("  Per-Category Performance:")
        for cat, metrics in sorted(s.per_category.items()):
            if cat == "irrelevant":
                lines.append(
                    f"    {cat:<12s} (n={int(metrics['count'])}): "
                    f"Rejection Rate={metrics['rejection_rate']:>6.1%}  "
                    f"Avg Chunks={metrics['avg_chunks_returned']:>4.1f}  "
                    f"(Excluded from ranking metrics)"
                )
            else:
                lines.append(
                    f"    {cat:<12s} (n={int(metrics['count'])}): "
                    f"Hit={metrics['hit_rate']:>6.1%}  "
                    f"Recall={metrics['avg_recall']:>6.1%}  "
                    f"Prec={metrics['avg_precision']:>6.1%}  "
                    f"MRR={metrics['mrr']:>6.4f}"
                )
        lines.append("")

    if len(summaries) > 1:
        lines.append("=" * 120)
        lines.append("Comparative Summary Table")
        lines.append("=" * 120)
        col_w = 17
        header = f"{'Metric':<25s}"
        for s in summaries:
            header += f"  {s.mode.upper():<{col_w}s}"
        lines.append(header)
        lines.append("-" * 120)

        # Dynamic overhead calculation against corresponding baselines in the same run
        mode_map = {s.mode.lower(): s for s in summaries}

        def get_overhead(summary: EvalSummary, attr: str) -> str:
            val = getattr(summary, attr)
            mode = summary.mode.lower()
            if mode == "semantic":
                return "Baseline"
            elif mode == "semantic+rerank" and "semantic" in mode_map:
                diff = val - getattr(mode_map["semantic"], attr)
                sign = "+" if diff >= 0 else ""
                return f"{sign}{diff:.2f} ms"
            elif mode == "keyword":
                return "N/A"
            elif mode == "hybrid":
                return "Baseline"
            elif mode == "hybrid+rerank" and "hybrid" in mode_map:
                diff = val - getattr(mode_map["hybrid"], attr)
                sign = "+" if diff >= 0 else ""
                return f"{sign}{diff:.2f} ms"
            return "N/A"

        rows = [
            ("Relevant Queries (n)", lambda s: f"{s.relevant_queries}"),
            (f"Hit@{summaries[0].top_k} (Relevant)", lambda s: f"{s.hit_rate:.2%}"),
            (
                f"Recall@{summaries[0].top_k} (Relevant)",
                lambda s: f"{s.avg_recall:.2%}",
            ),
            (
                f"Precision@{summaries[0].top_k} (Relevant)",
                lambda s: f"{s.avg_precision:.2%}",
            ),
            ("MRR (Relevant)", lambda s: f"{s.mrr:.4f}"),
            ("Irrelevant Queries (n)", lambda s: f"{s.irrelevant_queries}"),
            ("Rejection Rate (Irrel)", lambda s: f"{s.rejection_rate:.2%}"),
            ("Median Latency", lambda s: f"{s.median_latency_ms:.2f} ms"),
            ("Median Overhead", lambda s: get_overhead(s, "median_latency_ms")),
            ("p95 Latency", lambda s: f"{s.p95_latency_ms:.2f} ms"),
            ("p95 Overhead", lambda s: get_overhead(s, "p95_latency_ms")),
        ]

        for label, fmt in rows:
            row = f"{label:<25s}"
            for s in summaries:
                row += f"  {fmt(s):<{col_w}s}"
            lines.append(row)
        lines.append("-" * 120)

    return "\n".join(lines)


def run_benchmark(
    top_k: int = 5, return_cold_start: bool = False
) -> list[EvalSummary] | tuple[list[EvalSummary], float]:
    """Execute the retrieval benchmark comparing baseline and reranked modes.

    Indexes sample fixtures into an isolated temporary vector store,
    evaluates semantic, keyword, hybrid, and reranked variants across
    the standard queries, and returns the EvalSummary list (or with cold-start latency).
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

        # 1. Semantic-only retriever (baseline)
        semantic_retriever = SemanticRetriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
            default_top_k=top_k,
            min_score=0.30,
        )

        # 2. Sparse-only BM25 retriever (baseline)
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

        # Cross-encoder candidate reranker (Phase 9)
        t_cold_0 = time.perf_counter()
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            default_top_k=top_k,
        )
        # Preload model to measure cold start
        _ = reranker._get_model()
        cold_start_s = time.perf_counter() - t_cold_0
        logger.info(
            "CrossEncoder model cold-start initialization: %.3f s", cold_start_s
        )

        # 4. Semantic + Rerank (candidate_top_k=20 -> final top_k=5)
        semantic_candidate_retriever = SemanticRetriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
            default_top_k=20,
            min_score=0.30,
        )
        semantic_reranked = RerankingRetriever(
            base_retriever=semantic_candidate_retriever,
            reranker=reranker,
            candidate_top_k=20,
            default_top_k=top_k,
        )

        # 5. Hybrid + Rerank (candidate_top_k=20 -> final top_k=5)
        hybrid_candidate_retriever = HybridRetriever(
            semantic_retriever=hybrid_semantic,
            keyword_retriever=keyword_retriever,
            semantic_weight=1.0,
            keyword_weight=1.0,
            rrf_k=60,
            default_top_k=20,
            semantic_top_k=20,
            keyword_top_k=20,
        )
        hybrid_reranked = RerankingRetriever(
            base_retriever=hybrid_candidate_retriever,
            reranker=reranker,
            candidate_top_k=20,
            default_top_k=top_k,
        )

        eval_queries = build_sample_eval_dataset()

        summary_sem = evaluate_retriever(
            semantic_retriever, eval_queries, top_k=top_k, mode="semantic"
        )
        summary_sem_rerank = evaluate_retriever(
            semantic_reranked, eval_queries, top_k=top_k, mode="semantic+rerank"
        )
        summary_bm25 = evaluate_retriever(
            keyword_retriever, eval_queries, top_k=top_k, mode="keyword"
        )
        summary_hybrid = evaluate_retriever(
            hybrid_retriever, eval_queries, top_k=top_k, mode="hybrid"
        )
        summary_hybrid_rerank = evaluate_retriever(
            hybrid_reranked, eval_queries, top_k=top_k, mode="hybrid+rerank"
        )

        summaries = [
            summary_sem,
            summary_sem_rerank,
            summary_bm25,
            summary_hybrid,
            summary_hybrid_rerank,
        ]

        # Verify that relevance gate rejection prevents LLM invocation in orchestration
        from unittest.mock import MagicMock

        from rag.generation.base import Generator
        from rag.orchestration.exceptions import EmptyRetrievalError
        from rag.orchestration.query_service import RAGQueryService

        irrelevant_query_strings = [
            eq.query for eq in eval_queries if len(eq.expected_doc_names) == 0
        ]
        mock_generator = MagicMock(spec=Generator)

        gate_prevented_all = True
        for ret in [
            semantic_retriever,
            semantic_reranked,
            keyword_retriever,
            hybrid_retriever,
            hybrid_reranked,
        ]:
            qs = RAGQueryService(retriever=ret, generator=mock_generator)
            for q_str in irrelevant_query_strings:
                try:
                    qs.query(q_str)
                    gate_prevented_all = False
                except EmptyRetrievalError:
                    pass

        llm_prevented = (mock_generator.generate.call_count == 0) and gate_prevented_all
        for s in summaries:
            s.llm_prevented = llm_prevented

        if llm_prevented:
            logger.info(
                "Relevance gate verified: LLM generator was never invoked "
                "across any mode for irrelevant queries."
            )
        else:
            logger.error(
                "Relevance gate failure: LLM generator was invoked %d times "
                "during irrelevant query check!",
                mock_generator.generate.call_count,
            )

        if return_cold_start:
            return summaries, cold_start_s
        return summaries


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("Running Phase 9 Retrieval & Reranking Benchmark...")
    summaries, cold_start_s = run_benchmark(top_k=5, return_cold_start=True)
    report = format_eval_report(summaries, cold_start_s=cold_start_s)
    print("\n" + report)
