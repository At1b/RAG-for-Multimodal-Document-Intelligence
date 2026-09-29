"""BM25 sparse keyword retriever — Phase 8.

Implements the Okapi BM25 ranking function for keyword-based retrieval,
independent of embeddings or any vector store.

The index is built from ``VectorSearchResult``-compatible data (chunk
text + metadata) and can be reconstructed from ChromaDB persisted data
at startup.  The index itself is in-memory; it is rebuilt from the
authoritative data already persisted in ChromaDB.

Design decisions:
    - Zero external dependencies: BM25 implemented using only Python
      stdlib (``math``, ``re``, ``collections``).
    - Returns ``VectorSearchResult`` for compatibility with the existing
      retrieval pipeline, citation formatter, and relevance gate.
    - BM25 scores are **not** cosine similarity — they are raw relevance
      scores whose magnitude depends on corpus statistics.  Callers must
      not compare them directly with semantic similarity scores.
    - Thread-safe for reads after index construction.
    - Supports multi-document indexing with full metadata preservation.

BM25 Parameters:
    k1 (float): Term frequency saturation.  Default 1.5.
    b (float): Length normalization factor.  Default 0.75.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter

from rag.retrieval.base import Retriever
from rag.retrieval.exceptions import InvalidQueryError, InvalidTopKError
from rag.vectorstore.models import VectorSearchResult

logger = logging.getLogger(__name__)

# Absolute upper bound for top_k to prevent resource abuse.
MAX_TOP_K = 1000

# Maximum query length in characters (same as SemanticRetriever).
MAX_QUERY_LENGTH = 10_000

# Default top_k when neither the caller nor configuration specifies one.
_FALLBACK_DEFAULT_TOP_K = 10

# Simple tokenization pattern: word characters and digits.
_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

# Standard English stop words for IR retrieval filtering.
# Prevents common question words and grammatical particles from dominating BM25 scores
# and causing false-positive matches on out-of-domain queries.
DEFAULT_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "did",
        "do",
        "does",
        "doing",
        "don",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "s",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "t",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "ve",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "d",
        "ll",
        "m",
        "re",
    }
)


def tokenize(
    text: str, stopwords: frozenset[str] | set[str] | None = None
) -> list[str]:
    """Tokenize text into lowercase word tokens, optionally filtering stopwords.

    Uses a simple regex-based tokenizer that extracts word characters.
    Handles Unicode text correctly.

    Args:
        text: Input text to tokenize.
        stopwords: Optional set of stopwords to filter out. If None,
            all tokens are returned.

    Returns:
        List of lowercase tokens.
    """
    tokens = _TOKEN_PATTERN.findall(text.lower())
    if stopwords:
        return [t for t in tokens if t not in stopwords]
    return tokens


class BM25Index:
    """In-memory BM25 index over a corpus of documents.

    This is the core data structure that stores term frequencies,
    document frequencies, and corpus statistics needed for BM25 scoring.

    Args:
        k1: Term frequency saturation parameter.  Default 1.5.
        b: Length normalization parameter.  Default 0.75.
        stopwords: Optional set of stopwords to filter out during indexing
            and search. Defaults to DEFAULT_STOPWORDS. Pass None or empty
            set to disable stopword filtering.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        stopwords: frozenset[str] | set[str] | None = DEFAULT_STOPWORDS,
    ) -> None:
        if not isinstance(k1, (int, float)) or isinstance(k1, bool) or k1 < 0:
            raise ValueError(f"k1 must be a non-negative number, got {k1}")
        if not isinstance(b, (int, float)) or isinstance(b, bool) or b < 0 or b > 1:
            raise ValueError(f"b must be a number between 0 and 1, got {b}")

        self._k1 = float(k1)
        self._b = float(b)
        self._stopwords = frozenset(stopwords) if stopwords is not None else None

        # Corpus data
        self._doc_count: int = 0
        self._avgdl: float = 0.0
        self._doc_lengths: list[int] = []
        self._doc_term_freqs: list[Counter] = []
        self._df: Counter = Counter()  # Document frequency per term

        # Stored document data for result construction
        self._entries: list[VectorSearchResult] = []

    @property
    def doc_count(self) -> int:
        """Number of documents (chunks) in the index."""
        return self._doc_count

    @property
    def stopwords(self) -> frozenset[str] | None:
        """Stopwords filtered by this index, or None if disabled."""
        return self._stopwords

    def build(self, entries: list[VectorSearchResult]) -> None:
        """Build the BM25 index from a list of search result entries.

        Each entry represents a chunk with its text content and metadata.
        The index is built in-place, replacing any previous index.

        Args:
            entries: List of ``VectorSearchResult`` objects to index.
                The ``content`` field is tokenized for BM25 scoring.
        """
        self._entries = list(entries)
        self._doc_count = len(entries)
        self._doc_lengths = []
        self._doc_term_freqs = []
        self._df = Counter()

        if self._doc_count == 0:
            self._avgdl = 0.0
            return

        total_length = 0

        for entry in entries:
            tokens = tokenize(entry.content, stopwords=self._stopwords)
            tf = Counter(tokens)

            self._doc_term_freqs.append(tf)
            self._doc_lengths.append(len(tokens))
            total_length += len(tokens)

            # Update document frequency (count each term once per doc)
            for term in tf:
                self._df[term] += 1

        self._avgdl = total_length / self._doc_count if self._doc_count > 0 else 0.0

        logger.info(
            "BM25 index built: %d documents, %.1f avg doc length, %d unique terms.",
            self._doc_count,
            self._avgdl,
            len(self._df),
        )

    def score(self, query_tokens: list[str]) -> list[float]:
        """Compute BM25 scores for all indexed documents against query tokens.

        Args:
            query_tokens: Tokenized query terms.

        Returns:
            List of BM25 scores, one per indexed document, in index order.
        """
        scores = [0.0] * self._doc_count

        for token in query_tokens:
            if token not in self._df:
                continue

            df = self._df[token]
            # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            # Using the standard BM25 IDF formula with +1 to avoid
            # negative values when df > N/2.
            idf = math.log((self._doc_count - df + 0.5) / (df + 0.5) + 1.0)

            for i in range(self._doc_count):
                tf = self._doc_term_freqs[i].get(token, 0)
                if tf == 0:
                    continue

                dl = self._doc_lengths[i]
                # BM25 TF component
                numerator = tf * (self._k1 + 1)
                denominator = tf + self._k1 * (
                    1 - self._b + self._b * dl / self._avgdl if self._avgdl > 0 else 1.0
                )
                scores[i] += idf * (numerator / denominator)

        return scores

    def search(
        self, query: str, top_k: int = 10
    ) -> list[tuple[VectorSearchResult, float]]:
        """Search the index and return the top-K results with BM25 scores.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return.

        Returns:
            List of (VectorSearchResult, bm25_score) tuples, sorted by
            descending BM25 score.  Only entries with score > 0 are returned.
        """
        if self._doc_count == 0:
            return []

        query_tokens = tokenize(query, stopwords=self._stopwords)
        if not query_tokens:
            return []

        scores = self.score(query_tokens)

        # Build (index, score) pairs, filter to positive scores, sort descending.
        scored = [(i, s) for i, s in enumerate(scores) if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:top_k]

        return [(self._entries[i], s) for i, s in scored]

    def remove_document(self, document_id: str) -> int:
        """Remove all entries for a document and rebuild.

        Args:
            document_id: The document ID whose chunks should be removed.

        Returns:
            Number of entries removed.
        """
        original_count = self._doc_count
        remaining = [e for e in self._entries if e.document_id != document_id]
        removed = original_count - len(remaining)

        if removed > 0:
            self.build(remaining)
            logger.info(
                "Removed %d entries for document_id='%s' from BM25 index.",
                removed,
                document_id,
            )

        return removed


class BM25Retriever(Retriever):
    """Sparse keyword retriever using BM25 ranking.

    Implements the ``Retriever`` interface for keyword-based retrieval.
    The BM25 index must be built before retrieval can be performed.

    Args:
        bm25_index: A pre-built ``BM25Index`` instance.
        default_top_k: Default number of results when the caller does
            not specify ``top_k``.
    """

    def __init__(
        self,
        bm25_index: BM25Index,
        default_top_k: int = _FALLBACK_DEFAULT_TOP_K,
    ) -> None:
        if not isinstance(bm25_index, BM25Index):
            raise TypeError(
                f"bm25_index must be a BM25Index instance, "
                f"got {type(bm25_index).__name__}"
            )
        if (
            not isinstance(default_top_k, int)
            or isinstance(default_top_k, bool)
            or default_top_k < 1
            or default_top_k > MAX_TOP_K
        ):
            raise ValueError(
                f"default_top_k must be an integer between 1 and {MAX_TOP_K}, "
                f"got {default_top_k}"
            )

        self._bm25_index = bm25_index
        self._default_top_k = default_top_k

    @property
    def bm25_index(self) -> BM25Index:
        """The underlying BM25 index."""
        return self._bm25_index

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[VectorSearchResult]:
        """Retrieve the most relevant chunks using BM25 keyword matching.

        Args:
            query: The user's natural-language question.
            top_k: Maximum number of results to return.
            min_score: Optional minimum BM25 score threshold. When provided,
                chunks with a score strictly below this value are discarded.

        Returns:
            List of ``VectorSearchResult`` ordered by descending BM25
            score.  The ``score`` field contains the raw BM25 score
            (NOT cosine similarity).

        Raises:
            InvalidQueryError: If the query is invalid.
            InvalidTopKError: If top_k is invalid.
        """
        self._validate_query(query)
        effective_top_k = self._resolve_top_k(top_k)

        # Sanitize query text
        clean_query = (
            query.replace("\x00", "")
            .replace("\ufeff", "")
            .replace("\u200b", "")
            .strip()
        )

        results_with_scores = self._bm25_index.search(
            clean_query, top_k=effective_top_k
        )

        # Construct VectorSearchResult with BM25 scores
        results = []
        for entry, bm25_score in results_with_scores:
            if min_score is not None and bm25_score < min_score:
                continue
            results.append(
                VectorSearchResult(
                    chunk_id=entry.chunk_id,
                    document_id=entry.document_id,
                    document_name=entry.document_name,
                    content=entry.content,
                    score=bm25_score,
                    metadata=entry.metadata.copy(),
                )
            )

        logger.info(
            "BM25 retrieved %d results for query (top_k=%d).",
            len(results),
            effective_top_k,
        )

        return results

    @staticmethod
    def _validate_query(query: str) -> None:
        """Validate the user query (same rules as SemanticRetriever)."""
        if not isinstance(query, str):
            raise InvalidQueryError(
                f"query must be a string, got {type(query).__name__}"
            )
        sanitized = (
            query.replace("\x00", "")
            .replace("\ufeff", "")
            .replace("\u200b", "")
            .strip()
        )
        if not sanitized:
            raise InvalidQueryError("query must be a non-empty string")
        if len(query) > MAX_QUERY_LENGTH:
            raise InvalidQueryError(
                f"query exceeds maximum length of {MAX_QUERY_LENGTH} characters "
                f"(got {len(query)})"
            )

    def _resolve_top_k(self, top_k: int | None) -> int:
        """Resolve and validate the effective top_k value."""
        candidate = self._default_top_k if top_k is None else top_k

        if not isinstance(candidate, int) or isinstance(candidate, bool):
            raise InvalidTopKError(
                f"top_k must be a positive integer, got {type(candidate).__name__}"
            )
        if candidate < 1:
            raise InvalidTopKError(f"top_k must be >= 1, got {candidate}")
        if candidate > MAX_TOP_K:
            raise InvalidTopKError(
                f"top_k exceeds maximum of {MAX_TOP_K}, got {candidate}"
            )

        return candidate
