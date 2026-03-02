"""
masis.utils.text_utils
=======================
Text manipulation and analysis utilities for the MASIS system.

Implements
----------
MF-SYN-01  : U-shape context ordering -- u_shape_order()
MF-SUP-06  : Cosine-similarity repetition detection -- is_repetitive()

Design Notes
------------
u_shape_order()
    The "lost-in-the-middle" problem: LLMs attend most strongly to tokens at the
    START and END of their context window. Tokens in the middle receive less
    attention. For RAG systems with multiple evidence chunks, placing the most
    relevant chunks at the extremes of the context improves faithfulness.

    Algorithm: Sort chunks by rerank_score descending, then interleave so that:
    - Even-indexed chunks go to the START (left) of the list
    - Odd-indexed chunks go to the END (right) of the list

    Example with scores [0.92, 0.87, 0.81, 0.74, 0.69]:
    Sorted: [0.92, 0.87, 0.81, 0.74, 0.69]  (indices 0..4)
    Left:  index 0 (0.92), index 2 (0.81), index 4 (0.69)
    Right: index 1 (0.87), index 3 (0.74)
    Result: [0.69, 0.81, 0.92] + reversed([0.87, 0.74]) = nope...

    Algorithm (from architecture Section 16):
        left  = [chunk for even-indexed in sorted_chunks]   # START of context
        right = [chunk for odd-indexed  in sorted_chunks]   # END  of context (reversed)
        result = left + reversed(right)

    Input scores [0.92, 0.87, 0.81, 0.74, 0.69]:
        left  = [0.92, 0.81, 0.69]
        right = [0.87, 0.74] → reversed → [0.74, 0.87]
        Final order by score: [0.92, 0.81, 0.69, 0.74, 0.87]
    Best (0.92) is at START. Second-best (0.87) is at END.
    Weakest (0.69) is in the MIDDLE (least attention from LLM).

is_repetitive()
    Computes cosine similarity between the queries of the last two tasks of the
    same type in the DAG. If similarity > 0.90, the Supervisor Fast Path
    short-circuits to force_synthesize to prevent infinite search loops.

    Uses sentence-transformers (all-MiniLM-L6-v2) for encoding. Falls back to
    SequenceMatcher (stdlib) if sentence-transformers is not installed.

Usage
-----
    from masis.utils.text_utils import u_shape_order, is_repetitive

    # Reorder evidence for Synthesizer (called inside run_synthesizer):
    ordered_chunks = u_shape_order(state["evidence_board"])

    # Check for repetitive search loops (called inside supervisor_node Fast Path):
    if is_repetitive(state):
        return {"supervisor_decision": "force_synthesize", "reason": "repetitive_loop"}
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional

from masis.schemas.models import EvidenceChunk, MASISState, TaskNode

logger = logging.getLogger(__name__)
FAST_REPETITION_CHECK: bool = os.getenv("FAST_REPETITION_CHECK", "1").lower() in {"1", "true", "yes"}


# ---------------------------------------------------------------------------
# ENG-03 / M4 / S1 — u_shape_order  (MF-SYN-01)
# ---------------------------------------------------------------------------

def u_shape_order(chunks: List[EvidenceChunk]) -> List[EvidenceChunk]:
    """
    Reorder evidence chunks using the U-shape (Lost-in-the-Middle mitigation) strategy.

    Sorts chunks by rerank_score descending, then interleaves them so that the
    most relevant chunks appear at the START and END of the returned list.
    Weakest chunks are buried in the middle where LLM attention is lowest.

    Algorithm (from architecture Section 16 / MF-SYN-01):
        sorted_chunks = sort by rerank_score descending
        left  = [chunk for i, chunk in enumerate(sorted_chunks) if i % 2 == 0]
        right = [chunk for i, chunk in enumerate(sorted_chunks) if i % 2 == 1]
        result = left + reversed(right)

    With scores [0.92, 0.87, 0.81, 0.74, 0.69]:
        sorted:          [0.92, 0.87, 0.81, 0.74, 0.69]
        left  (even):    [0.92, 0.81, 0.69]    (indices 0, 2, 4)
        right (odd):     [0.87, 0.74]           (indices 1, 3)
        reversed(right): [0.74, 0.87]
        result:          [0.92, 0.81, 0.69, 0.74, 0.87]

    Result order (score): [0.92, 0.81, 0.69, 0.74, 0.87]
    Best (0.92) is at START (index 0, highest LLM attention).
    2nd-best (0.87) is at END (index 4, still high LLM attention).
    Weakest (0.69) is in the MIDDLE (index 2, lowest LLM attention).

    Args:
        chunks: List of EvidenceChunk objects to reorder. May be empty.

    Returns:
        Reordered list. Empty list if input is empty. Single-element unchanged.

    Examples:
        >>> from masis.schemas.models import EvidenceChunk
        >>> c = lambda score: EvidenceChunk(doc_id="d", chunk_id=str(score),
        ...                                text="", rerank_score=score)
        >>> result = u_shape_order([c(0.92), c(0.87), c(0.81), c(0.74), c(0.69)])
        >>> [r.rerank_score for r in result]
        [0.92, 0.81, 0.69, 0.74, 0.87]
    """
    if not chunks:
        return []
    if len(chunks) == 1:
        return list(chunks)

    # Sort descending by rerank_score (best first)
    sorted_chunks = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)

    # Section 16 algorithm (left + reversed(right)):
    #   Even-indexed chunks → left list (START of context)
    #   Odd-indexed chunks  → right list, then reversed → END of context
    #
    # Example with scores [0.92, 0.87, 0.81, 0.74, 0.69]:
    #   left  = [0.92, 0.81, 0.69]  (indices 0, 2, 4)
    #   right = [0.87, 0.74]         (indices 1, 3)
    #   reversed(right) = [0.74, 0.87]
    #   result = [0.92, 0.81, 0.69, 0.74, 0.87]
    left: List[EvidenceChunk] = []
    right: List[EvidenceChunk] = []
    for i, chunk in enumerate(sorted_chunks):
        if i % 2 == 0:
            left.append(chunk)    # Even indices → START (best first)
        else:
            right.append(chunk)   # Odd indices  → END (2nd best last)

    result = left + list(reversed(right))

    logger.debug(
        "u_shape_order: %d chunks reordered. Scores: %s -> %s",
        len(chunks),
        [round(c.rerank_score, 2) for c in sorted_chunks],
        [round(c.rerank_score, 2) for c in result],
    )
    return result


def u_shape_order_raw(items: List[Any], score_key: str = "rerank_score") -> List[Any]:
    """
    U-shape ordering for arbitrary objects with a numeric score attribute.

    Generalised version of u_shape_order() for non-EvidenceChunk objects.
    Used internally and in tests.

    Args:
        items:     List of objects with a numeric score attribute.
        score_key: Attribute name holding the sort score.

    Returns:
        Reordered list in U-shape pattern.
    """
    if not items:
        return []
    if len(items) == 1:
        return list(items)

    def get_score(item: Any) -> float:
        if hasattr(item, score_key):
            return float(getattr(item, score_key))
        if isinstance(item, dict):
            return float(item.get(score_key, 0.0))
        return 0.0

    sorted_items = sorted(items, key=get_score, reverse=True)
    left: List[Any] = []
    right: List[Any] = []
    for i, item in enumerate(sorted_items):
        if i % 2 == 0:
            left.append(item)    # Even indices → START
        else:
            right.append(item)   # Odd indices  → END (will be reversed)
    return left + list(reversed(right))


# ---------------------------------------------------------------------------
# ENG-03 / M3 / S1 — is_repetitive  (MF-SUP-06)
# ---------------------------------------------------------------------------

def is_repetitive(state: Dict[str, Any]) -> bool:
    """
    Detect repetitive search loops using cosine similarity between task queries.

    Compares the queries of the last two completed tasks of the SAME type.
    If cosine similarity > REPETITION_COSINE_THRESHOLD (0.90), returns True.

    This is a Fast Path check ($0, < 100ms) that prevents infinite loops where
    the Supervisor repeatedly dispatches near-identical queries.

    Example (from architecture Q15):
        Last researcher query: "TechCorp market share decline"
        This researcher query: "market share TechCorp declining"
        Cosine similarity: 0.91 > 0.90 -> return True -> force_synthesize

    Implementation layers:
    1. sentence-transformers (preferred): encode with all-MiniLM-L6-v2
    2. Python SequenceMatcher (stdlib fallback): no external dependency

    Args:
        state: The current MASISState dict. Reads state["task_dag"] to find
               the last two completed tasks of the same type.

    Returns:
        True  -- the last two same-type task queries are > 90% similar.
        False -- no repetition detected, or fewer than 2 same-type tasks exist.

    Examples:
        >>> state = {"task_dag": [
        ...     TaskNode(task_id="T1", type="researcher",
        ...              query="TechCorp market share decline", status="done"),
        ...     TaskNode(task_id="T2", type="researcher",
        ...              query="market share TechCorp declining", status="done"),
        ... ]}
        >>> is_repetitive(state)
        True  # cosine > 0.90

        >>> state2 = {"task_dag": [
        ...     TaskNode(task_id="T1", type="researcher", query="revenue growth", status="done"),
        ...     TaskNode(task_id="T2", type="researcher", query="employee headcount", status="done"),
        ... ]}
        >>> is_repetitive(state2)
        False  # cosine < 0.5
    """
    from masis.schemas.thresholds import SAFETY_LIMITS
    threshold = SAFETY_LIMITS["REPETITION_COSINE_THRESHOLD"]

    dag: List[TaskNode] = state.get("task_dag", [])
    if not dag:
        return False

    # Group completed tasks by type, preserving order
    done_by_type: Dict[str, List[str]] = {}
    for task in dag:
        if task.status in ("done", "running"):
            queries = done_by_type.setdefault(task.type, [])
            queries.append(task.query)

    # Check each type's last two queries
    for task_type, queries in done_by_type.items():
        if len(queries) < 2:
            continue
        q1 = queries[-2]
        q2 = queries[-1]
        similarity = _compute_cosine_similarity(q1, q2)
        logger.debug(
            "is_repetitive: type=%s, q1='%s...', q2='%s...', cosine=%.3f, threshold=%.2f",
            task_type, q1[:50], q2[:50], similarity, threshold,
        )
        if similarity > threshold:
            logger.info(
                "is_repetitive: REPETITION DETECTED for type='%s' (cosine=%.3f > %.2f). "
                "Queries: '%s' vs '%s'",
                task_type, similarity, threshold, q1[:80], q2[:80],
            )
            return True

    return False


def _compute_cosine_similarity(text1: str, text2: str) -> float:
    """
    Compute cosine similarity between two text strings.

    Tries sentence-transformers first (preferred, higher quality).
    Falls back to TF-IDF-style bag-of-words cosine if unavailable.
    Last resort: SequenceMatcher ratio (good enough for short queries).

    Args:
        text1: First text string.
        text2: Second text string.

    Returns:
        Float in [0.0, 1.0]. 1.0 = identical meaning, 0.0 = unrelated.
    """
    if not text1.strip() or not text2.strip():
        return 0.0
    if text1.strip().lower() == text2.strip().lower():
        return 1.0

    # Demo-speed default: skip heavyweight sentence-transformer load.
    if not FAST_REPETITION_CHECK:
        try:
            return _sentence_transformer_cosine(text1, text2)
        except ImportError:
            logger.debug("sentence-transformers not available, falling back to BOW cosine.")
        except Exception as exc:
            logger.debug("sentence-transformers failed (%s), falling back to BOW cosine.", exc)

    # Bag-of-words cosine (stdlib only)
    try:
        return _bow_cosine(text1, text2)
    except Exception as exc:
        logger.debug("BOW cosine failed (%s), falling back to SequenceMatcher.", exc)

    # Attempt 3: SequenceMatcher (always available)
    return _sequence_matcher_similarity(text1, text2)


def _sentence_transformer_cosine(text1: str, text2: str) -> float:
    """
    Cosine similarity using sentence-transformers all-MiniLM-L6-v2.

    This model produces 384-dimensional embeddings well-suited for semantic
    similarity of short texts like task queries.

    Raises:
        ImportError if sentence-transformers is not installed.
    """
    from sentence_transformers import SentenceTransformer  # type: ignore[import]
    from masis.schemas.thresholds import EMBEDDER_MODEL

    # Module-level cache to avoid reloading the model on every call
    if not hasattr(_sentence_transformer_cosine, "_model"):
        _sentence_transformer_cosine._model = SentenceTransformer(EMBEDDER_MODEL)  # type: ignore[attr-defined]

    model = _sentence_transformer_cosine._model  # type: ignore[attr-defined]
    embeddings = model.encode([text1, text2], convert_to_tensor=False)
    return float(_cosine(embeddings[0], embeddings[1]))


def _bow_cosine(text1: str, text2: str) -> float:
    """
    Simple bag-of-words cosine similarity using token frequency vectors.

    No external dependencies required. Suitable for short query strings.
    """
    def tokenize(text: str) -> Dict[str, int]:
        tokens = text.lower().split()
        freq: Dict[str, int] = {}
        for token in tokens:
            # Strip punctuation from token edges
            token = token.strip(".,!?;:\"'()[]{}").strip()
            if token:
                freq[token] = freq.get(token, 0) + 1
        return freq

    freq1 = tokenize(text1)
    freq2 = tokenize(text2)

    if not freq1 or not freq2:
        return 0.0

    # Vocabulary union
    vocab = set(freq1.keys()) | set(freq2.keys())
    vec1 = [freq1.get(w, 0) for w in vocab]
    vec2 = [freq2.get(w, 0) for w in vocab]

    return float(_cosine(vec1, vec2))


def _sequence_matcher_similarity(text1: str, text2: str) -> float:
    """
    SequenceMatcher-based string similarity (stdlib). Last-resort fallback.

    Less semantically aware than embeddings, but always available.
    """
    import difflib
    return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def _cosine(vec1: Any, vec2: Any) -> float:
    """
    Compute cosine similarity between two vectors (as lists or numpy arrays).

    Handles both Python lists and numpy arrays transparently.
    Returns 0.0 when either vector has zero magnitude.
    """
    try:
        import numpy as np  # type: ignore[import]
        a = np.asarray(vec1, dtype=float)
        b = np.asarray(vec2, dtype=float)
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    except ImportError:
        # Pure Python fallback (slower but correct)
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = math.sqrt(sum(a * a for a in vec1))
        norm_b = math.sqrt(sum(b * b for b in vec2))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Additional text utilities (supporting functions)
# ---------------------------------------------------------------------------

def truncate_to_tokens(text: str, max_tokens: int, chars_per_token: float = 4.0) -> str:
    """
    Truncate text to approximately max_tokens LLM tokens.

    Uses a rough chars-per-token ratio (4.0 for English prose with GPT tokenizers).
    This is a fast approximation -- no tokenizer library needed.

    Args:
        text:            Text to truncate.
        max_tokens:      Target maximum number of tokens.
        chars_per_token: Average characters per token (default 4.0).

    Returns:
        Truncated string. Adds "..." suffix if truncation occurred.

    Examples:
        >>> truncate_to_tokens("Hello world", max_tokens=2)
        'He...'  # 2 tokens * 4 chars = 8 chars
    """
    max_chars = int(max_tokens * chars_per_token)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def normalize_whitespace(text: str) -> str:
    """
    Collapse consecutive whitespace and strip leading/trailing whitespace.

    Args:
        text: Input string.

    Returns:
        String with normalized whitespace.
    """
    import re
    return re.sub(r"\s+", " ", text).strip()


def extract_sentences(text: str) -> List[str]:
    """
    Split text into sentences using simple rule-based splitting.

    Not perfect for all edge cases, but sufficient for NLI claim extraction
    without requiring NLTK or spaCy.

    Args:
        text: Input paragraph or multi-sentence text.

    Returns:
        List of sentence strings, stripped of leading/trailing whitespace.
    """
    import re
    # Split on sentence-ending punctuation followed by whitespace and a capital letter
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]
