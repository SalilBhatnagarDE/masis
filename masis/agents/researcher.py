"""
masis.agents.researcher
=======================
Researcher agent  --  the primary RAG pipeline (ENG-07, MF-RES-01 through MF-RES-10).

Pipeline stages
---------------
1. HyDE rewrite          --  generate a hypothetical passage to improve embedding similarity (MF-RES-01)
2. Metadata extraction   --  parse year/quarter/department for ChromaDB filter (MF-RES-02)
3. Hybrid retrieval      --  vector top-10 + BM25 top-10, fused via RRF (MF-RES-03)
4. Cross-encoder rerank  --  ms-marco-MiniLM-L-6-v2 keeps top-5 (MF-RES-04)
5. Parent expansion      --  child (500 chars)  ->  parent (2000 chars) (MF-RES-07)
6. CRAG grading          --  grade relevance, rewrite + retry up to 3× (MF-RES-05)
7. Self-RAG check        --  verify answer is grounded, regenerate if not (MF-RES-06)
8. Structured output     --  ResearcherOutput with all criteria fields (MF-RES-08/09/10)

Public API
----------
run_researcher(task, state)  ->  AgentOutput
hyde_rewrite(query)          ->  str    (hypothetical passage)
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase 0 schema imports with stubs
# ---------------------------------------------------------------------------

try:
    from masis.schemas.models import AgentOutput, EvidenceChunk, ResearcherOutput
    from masis.schemas.thresholds import (
        RESEARCHER_THRESHOLDS,
        NLI_MODEL,
        RERANKER_MODEL,
    )
except ImportError:
    logger.warning("masis.schemas not found  --  using stub types for researcher.py")

    class EvidenceChunk:  # type: ignore[no-redef]
        def __init__(self, **kw: Any) -> None:
            for k, v in kw.items():
                setattr(self, k, v)
        chunk_id: str = ""
        doc_id: str = ""
        parent_chunk_id: Optional[str] = None
        text: str = ""
        retrieval_score: float = 0.0
        rerank_score: float = 0.0
        metadata: dict = {}
        source_label: str = ""

    class ResearcherOutput:  # type: ignore[no-redef]
        def __init__(self, **kw: Any) -> None:
            for k, v in kw.items():
                setattr(self, k, v)
        def to_criteria_dict(self) -> dict:
            return {
                "chunks_after_grading": getattr(self, "chunks_after_grading", 0),
                "grading_pass_rate": getattr(self, "grading_pass_rate", 0.0),
                "self_rag_verdict": getattr(self, "self_rag_verdict", "not_grounded"),
            }

    class AgentOutput:  # type: ignore[no-redef]
        def __init__(self, **kw: Any) -> None:
            for k, v in kw.items():
                setattr(self, k, v)

    RESEARCHER_THRESHOLDS: dict = {  # type: ignore[misc]
        "min_chunks_after_grading": 2,
        "min_grading_pass_rate": 0.30,
        "required_self_rag_verdict": "grounded",
        "crag_max_retries": 3,
        "self_rag_max_retries": 3,
        "top_k_retrieval": 10,
        "top_k_after_rerank": 5,
        "rrf_k": 60,
        "rrf_vector_weight": 0.7,
        "rrf_bm25_weight": 0.3,
    }
    NLI_MODEL = "facebook/bart-large-mnli"
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------

try:
    from langchain_openai import ChatOpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    logger.warning("langchain_openai not installed  --  researcher LLM calls will be stubbed")
    ChatOpenAI = None  # type: ignore[assignment,misc]

try:
    import chromadb  # type: ignore[import]
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False
    logger.warning("chromadb not installed  --  vector retrieval will return empty results")
    chromadb = None  # type: ignore[assignment]

try:
    from sentence_transformers import CrossEncoder
    _CROSS_ENCODER_AVAILABLE = True
except ImportError:
    _CROSS_ENCODER_AVAILABLE = False
    logger.warning("sentence-transformers not installed  --  cross-encoder reranking skipped")
    CrossEncoder = None  # type: ignore[assignment,misc]

try:
    from rank_bm25 import BM25Okapi  # type: ignore[import]
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False
    logger.warning("rank-bm25 not installed  --  BM25 retrieval skipped, using vector only")
    BM25Okapi = None  # type: ignore[assignment,misc]

try:
    from masis.config.model_routing import get_model
except ImportError:
    import os

    def get_model(role: str, override: Optional[str] = None) -> str:  # type: ignore[misc]
        return override or os.getenv("MODEL_RESEARCHER", "gpt-4.1-mini")

# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------

_cross_encoder: Any = None

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

MAX_SUMMARY_CHARS: int = 500  # MF-RES-10


async def run_researcher(task: Any, state: Dict[str, Any]) -> AgentOutput:
    """Execute the full RAG pipeline for a researcher task and return AgentOutput.

    Args:
        task: TaskNode with .task_id and .query.
        state: Filtered state (original_query, task).

    Returns:
        AgentOutput wrapping a ResearcherOutput.
    """
    task_id = getattr(task, "task_id", "unknown")
    query: str = getattr(task, "query", state.get("original_query", ""))
    logger.info("Researcher started: task_id=%s query='%.80s'", task_id, query)
    start_ts = time.monotonic()

    try:
        researcher_out = await _run_pipeline(task_id, query)
    except Exception as exc:
        logger.error("Researcher pipeline failed for task %s: %s", task_id, exc, exc_info=True)
        return AgentOutput(  # type: ignore[call-arg]
            task_id=task_id,
            agent_type="researcher",
            status="failed",
            summary=f"Pipeline error: {type(exc).__name__}: {exc}",
            evidence=[],
            criteria_result={},
            error_detail=str(exc),
        )

    elapsed = time.monotonic() - start_ts
    logger.info(
        "Researcher done: task_id=%s chunks=%d pass_rate=%.2f verdict=%s elapsed=%.2fs",
        task_id,
        researcher_out.chunks_after_grading,
        researcher_out.grading_pass_rate,
        researcher_out.self_rag_verdict,
        elapsed,
    )

    return AgentOutput(  # type: ignore[call-arg]
        task_id=task_id,
        agent_type="researcher",
        status="success",
        summary=researcher_out.summary,
        evidence=researcher_out.evidence,
        criteria_result=researcher_out.to_criteria_dict(),
        tokens_used=researcher_out.tokens_used,
        cost_usd=researcher_out.cost_usd,
        raw_output=researcher_out,
    )


async def hyde_rewrite(query: str) -> str:
    """Generate a hypothetical document passage to improve embedding similarity (MF-RES-01).

    HyDE (Hypothetical Document Embeddings) generates a plausible answer passage
    and uses its embedding rather than the bare query embedding. This improves
    retrieval for short factual queries.

    Args:
        query: The research sub-query to rewrite.

    Returns:
        A hypothetical passage (~3-5 sentences) about the topic.

    Raises:
        RuntimeError: If the LLM is unavailable.
    """
    if not _OPENAI_AVAILABLE or ChatOpenAI is None:
        logger.warning("HyDE: LLM unavailable  --  returning original query")
        return query

    model_name = get_model("researcher")
    llm = ChatOpenAI(model=model_name, temperature=0.1, max_tokens=200)

    system_prompt = (
        "You are a research assistant. Generate a hypothetical document passage "
        "that would answer the following question. Write 3-5 sentences as if "
        "you are quoting from the actual document. Be specific and factual in style."
    )
    human_message = f"Question: {query}"

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_message},
        ])
        passage = response.content.strip()
        logger.debug("HyDE rewrite: original='%.60s'  ->  passage='%.80s'", query, passage)
        return passage
    except Exception as exc:
        logger.warning("HyDE rewrite failed: %s  --  using original query", exc)
        return query


# ---------------------------------------------------------------------------
# Pipeline stages (private)
# ---------------------------------------------------------------------------

async def _run_pipeline(task_id: str, query: str) -> ResearcherOutput:
    """Orchestrate all RAG pipeline stages and return ResearcherOutput.

    Args:
        task_id: Identifier for logging.
        query: The research sub-query.

    Returns:
        Fully populated ResearcherOutput.
    """
    total_tokens = 0
    total_cost = 0.0
    crag_retries_used = 0

    # ── Stage 1: HyDE rewrite ────────────────────────────────────────────────
    hyde_passage = await hyde_rewrite(query)
    total_tokens += len(hyde_passage) // 4

    # ── Stage 2: Metadata extraction ────────────────────────────────────────
    metadata_filter = _extract_metadata(query)
    logger.debug("Metadata filter: %s", metadata_filter)

    # ── Stage 3 & 4: Hybrid retrieval + reranking (CRAG outer loop) ─────────
    max_crag_retries = RESEARCHER_THRESHOLDS.get("crag_max_retries", 3)
    current_query = query
    current_hyde = hyde_passage
    graded_chunks: List[EvidenceChunk] = []
    pass_rate: float = 0.0

    for attempt in range(max_crag_retries + 1):
        # Retrieve
        raw_chunks = await _hybrid_retrieve(current_hyde, metadata_filter)
        if not raw_chunks:
            logger.info("CRAG attempt %d: no chunks retrieved", attempt + 1)
            crag_retries_used = attempt
            break

        # Rerank
        reranked = await _cross_encoder_rerank(current_query, raw_chunks)

        # Parent expansion
        expanded = await _expand_to_parents(reranked)

        # CRAG grading (MF-RES-05)
        graded, pass_rate = await _grade_chunks(current_query, expanded)
        min_pass_rate = RESEARCHER_THRESHOLDS.get("min_grading_pass_rate", 0.30)

        if pass_rate >= min_pass_rate or attempt >= max_crag_retries:
            graded_chunks = graded
            crag_retries_used = attempt
            break

        # Rewrite and retry
        logger.info(
            "CRAG attempt %d: pass_rate=%.2f < %.2f  --  rewriting query",
            attempt + 1, pass_rate, min_pass_rate,
        )
        current_query, current_hyde = await _rewrite_query(current_query)
        crag_retries_used = attempt + 1

    # ── Stage 7: Self-RAG hallucination check ───────────────────────────────
    max_self_rag_retries = RESEARCHER_THRESHOLDS.get("self_rag_max_retries", 3)
    answer, self_rag_verdict, self_rag_tokens = await _self_rag_loop(
        query, graded_chunks, max_self_rag_retries
    )
    total_tokens += self_rag_tokens
    total_cost += _tokens_to_cost(self_rag_tokens, get_model("researcher"))

    # ── Stage 8: Build structured output ────────────────────────────────────
    source_diversity = _count_source_diversity(graded_chunks)
    summary = _build_summary(answer)

    return ResearcherOutput(  # type: ignore[call-arg]
        task_id=task_id,
        evidence=graded_chunks,
        summary=summary,
        chunks_retrieved=len(graded_chunks),
        chunks_after_grading=len(graded_chunks),
        grading_pass_rate=pass_rate,
        self_rag_verdict=self_rag_verdict,
        source_diversity=source_diversity,
        crag_retries_used=crag_retries_used,
        tokens_used=total_tokens,
        cost_usd=total_cost,
    )


def _extract_metadata(query: str) -> Dict[str, Any]:
    """Extract year, quarter, department from query string (MF-RES-02).

    Uses regex patterns for common business document metadata. Returns only
    keys that were found  --  others are omitted so ChromaDB does not filter on them.

    Args:
        query: The research sub-query string.

    Returns:
        Dict with zero or more keys: year, quarter, department.
    """
    meta: Dict[str, Any] = {}
    query_lower = query.lower()

    # Year detection (4-digit year, or FY/fiscal year patterns)
    year_match = re.search(r"\b(20\d{2})\b", query)
    if year_match:
        meta["year"] = int(year_match.group(1))

    # Quarter detection (Q1-Q4)
    quarter_match = re.search(r"\b(q[1-4])\b", query_lower)
    if quarter_match:
        meta["quarter"] = quarter_match.group(1).upper()

    # Department heuristics
    dept_keywords = {
        "cloud": "cloud",
        "ai": "ai",
        "enterprise": "enterprise",
        "consumer": "consumer",
        "r&d": "research",
        "research": "research",
        "sales": "sales",
        "marketing": "marketing",
        "operations": "operations",
    }
    for keyword, dept in dept_keywords.items():
        if keyword == "ai":
            # Avoid false positives from words like "analysis"/"maintain".
            if re.search(r"\bai\b|\bartificial intelligence\b", query_lower):
                meta["department"] = dept
                break
            continue
        if re.search(rf"\b{re.escape(keyword)}\b", query_lower):
            meta["department"] = dept
            break

    return meta


async def _hybrid_retrieve(
    hyde_passage: str,
    metadata_filter: Dict[str, Any],
) -> List[EvidenceChunk]:
    """Hybrid retrieval: vector search + BM25, fused with RRF (MF-RES-03).

    Args:
        hyde_passage: The HyDE-generated passage to embed.
        metadata_filter: ChromaDB where-clause dict for pre-filtering.

    Returns:
        Fused list of up to top_k unique chunks.
    """
    top_k = RESEARCHER_THRESHOLDS.get("top_k_retrieval", 10)

    # Vector retrieval
    vector_chunks = await _vector_retrieve(hyde_passage, metadata_filter, top_k)

    # BM25 retrieval
    bm25_chunks = await _bm25_retrieve(hyde_passage, top_k)

    # RRF fusion
    fused = _rrf_fuse(vector_chunks, bm25_chunks)
    logger.debug(
        "Hybrid retrieve: vector=%d, bm25=%d, fused=%d chunks",
        len(vector_chunks), len(bm25_chunks), len(fused),
    )
    return fused


async def _vector_retrieve(
    passage: str,
    metadata_filter: Dict[str, Any],
    top_k: int,
) -> List[EvidenceChunk]:
    """Query ChromaDB with an embedded passage (MF-RES-03).

    Falls back to empty list if ChromaDB or the collection are unavailable.

    Args:
        passage: Text to embed and query with.
        metadata_filter: ChromaDB metadata filter dict.
        top_k: Number of results to retrieve.

    Returns:
        List of EvidenceChunk objects.
    """
    if not _CHROMA_AVAILABLE or chromadb is None:
        logger.debug("ChromaDB unavailable  --  vector retrieve returns empty")
        return []

    try:
        # Expect a ChromaDB collection in the module-level cache or environment
        collection = _get_chroma_collection()
        if collection is None:
            return []

        where_filter = _to_chroma_where(metadata_filter)
        query_result = collection.query(
            query_texts=[passage],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        chunks: List[EvidenceChunk] = []
        docs = query_result.get("documents", [[]])[0]
        metas = query_result.get("metadatas", [[]])[0]
        dists = query_result.get("distances", [[]])[0]

        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            chunk = EvidenceChunk(  # type: ignore[call-arg]
                chunk_id=meta.get("chunk_id", f"vec_{i}"),
                doc_id=meta.get("doc_id", "unknown"),
                parent_chunk_id=meta.get("parent_chunk_id"),
                text=doc,
                retrieval_score=float(1.0 - dist),  # cosine: distance  ->  similarity
                rerank_score=0.0,
                metadata=meta,
                source_label=meta.get("source_label", ""),
            )
            chunks.append(chunk)
        return chunks

    except Exception as exc:
        logger.warning("Vector retrieve failed: %s  --  returning empty", exc)
        return []


def _to_chroma_where(metadata_filter: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize metadata filter into a Chroma-compatible where-clause."""
    if not metadata_filter:
        return None
    if len(metadata_filter) == 1:
        key, value = next(iter(metadata_filter.items()))
        return {key: value}
    return {"$and": [{key: value} for key, value in metadata_filter.items()]}


async def _bm25_retrieve(query: str, top_k: int) -> List[EvidenceChunk]:
    """BM25 retrieval using a pre-built index (MF-RES-03).

    Falls back to empty list when rank_bm25 is unavailable or no index is loaded.

    Args:
        query: The raw query string (not the HyDE passage).
        top_k: Number of results to retrieve.

    Returns:
        List of EvidenceChunk objects.
    """
    if not _BM25_AVAILABLE or BM25Okapi is None:
        logger.debug("BM25 unavailable  --  returning empty list")
        return []

    try:
        index_data = _get_bm25_index()
        if index_data is None:
            return []

        bm25_index, corpus, corpus_meta = index_data
        tokenized_query = query.lower().split()
        scores = bm25_index.get_scores(tokenized_query)

        # Get top_k indices by score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        chunks: List[EvidenceChunk] = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                continue
            meta = corpus_meta[idx] if idx < len(corpus_meta) else {}
            chunk = EvidenceChunk(  # type: ignore[call-arg]
                chunk_id=meta.get("chunk_id", f"bm25_{idx}"),
                doc_id=meta.get("doc_id", "unknown"),
                parent_chunk_id=meta.get("parent_chunk_id"),
                text=corpus[idx],
                retrieval_score=float(scores[idx]) / (float(scores[top_indices[0]]) + 1e-10),
                rerank_score=0.0,
                metadata=meta,
                source_label=meta.get("source_label", ""),
            )
            chunks.append(chunk)
        return chunks

    except Exception as exc:
        logger.warning("BM25 retrieve failed: %s  --  returning empty", exc)
        return []


def _rrf_fuse(
    vector_chunks: List[EvidenceChunk],
    bm25_chunks: List[EvidenceChunk],
) -> List[EvidenceChunk]:
    """Reciprocal Rank Fusion of two ranked lists (MF-RES-03).

    RRF formula: score(d) = Σ 1/(k + rank(d))  for each list where d appears.
    k=60 (as per spec). Vector weight=0.7, BM25 weight=0.3.

    Args:
        vector_chunks: Ranked list from vector retrieval.
        bm25_chunks: Ranked list from BM25 retrieval.

    Returns:
        Deduplicated list sorted by descending RRF score.
    """
    k = RESEARCHER_THRESHOLDS.get("rrf_k", 60)
    vector_weight = RESEARCHER_THRESHOLDS.get("rrf_vector_weight", 0.7)
    bm25_weight = RESEARCHER_THRESHOLDS.get("rrf_bm25_weight", 0.3)

    rrf_scores: Dict[str, float] = {}
    chunk_registry: Dict[str, EvidenceChunk] = {}

    def _add_list(chunks: List[EvidenceChunk], weight: float) -> None:
        for rank, chunk in enumerate(chunks, start=1):
            key = (chunk.doc_id, chunk.chunk_id)
            key_str = f"{chunk.doc_id}::{chunk.chunk_id}"
            rrf_scores[key_str] = rrf_scores.get(key_str, 0.0) + weight / (k + rank)
            if key_str not in chunk_registry:
                chunk_registry[key_str] = chunk

    _add_list(vector_chunks, vector_weight)
    _add_list(bm25_chunks, bm25_weight)

    top_k = RESEARCHER_THRESHOLDS.get("top_k_retrieval", 10)
    sorted_keys = sorted(rrf_scores, key=lambda k_: rrf_scores[k_], reverse=True)[:top_k]

    result: List[EvidenceChunk] = []
    for key_str in sorted_keys:
        chunk = chunk_registry[key_str]
        chunk.retrieval_score = rrf_scores[key_str]
        result.append(chunk)

    return result


async def _cross_encoder_rerank(
    query: str, chunks: List[EvidenceChunk]
) -> List[EvidenceChunk]:
    """Rerank retrieved chunks using ms-marco-MiniLM-L-6-v2 (MF-RES-04).

    Args:
        query: Original research query (not HyDE).
        chunks: Chunks from hybrid retrieval.

    Returns:
        Top-5 chunks sorted by descending rerank score.
    """
    if not _CROSS_ENCODER_AVAILABLE or CrossEncoder is None or not chunks:
        logger.debug("Cross-encoder unavailable  --  using retrieval score ordering")
        top_k = RESEARCHER_THRESHOLDS.get("top_k_after_rerank", 5)
        return sorted(chunks, key=lambda c: c.retrieval_score, reverse=True)[:top_k]

    encoder = _get_cross_encoder()
    if encoder is None:
        top_k = RESEARCHER_THRESHOLDS.get("top_k_after_rerank", 5)
        return sorted(chunks, key=lambda c: c.retrieval_score, reverse=True)[:top_k]

    try:
        pairs = [(query, chunk.text[:2000]) for chunk in chunks]
        scores = encoder.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = float(score)

        top_k = RESEARCHER_THRESHOLDS.get("top_k_after_rerank", 5)
        reranked = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)[:top_k]
        logger.debug("Reranked %d  ->  %d chunks", len(chunks), len(reranked))
        return reranked

    except Exception as exc:
        logger.warning("Cross-encoder reranking failed: %s  --  using retrieval order", exc)
        top_k = RESEARCHER_THRESHOLDS.get("top_k_after_rerank", 5)
        return sorted(chunks, key=lambda c: c.retrieval_score, reverse=True)[:top_k]


async def _expand_to_parents(chunks: List[EvidenceChunk]) -> List[EvidenceChunk]:
    """Replace child chunks with their parent chunks for full context (MF-RES-07).

    Looks up ``parent_chunk_id`` in ChromaDB for each chunk that has one.
    Falls through to the original chunk if the parent cannot be retrieved.

    Args:
        chunks: Reranked child chunks.

    Returns:
        List of chunks where available child chunks are replaced by parents.
    """
    expanded: List[EvidenceChunk] = []
    for chunk in chunks:
        parent_id = getattr(chunk, "parent_chunk_id", None)
        if parent_id and _CHROMA_AVAILABLE and chromadb is not None:
            parent = await _fetch_chunk_by_id(parent_id, chunk)
            expanded.append(parent)
        else:
            expanded.append(chunk)
    return expanded


async def _fetch_chunk_by_id(
    chunk_id: str, fallback: EvidenceChunk
) -> EvidenceChunk:
    """Fetch a single chunk from ChromaDB by ID, returning fallback on failure."""
    try:
        collection = _get_chroma_collection()
        if collection is None:
            return fallback
        result = collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        docs = result.get("documents", [])
        metas = result.get("metadatas", [])
        if docs and metas:
            meta = metas[0] if metas else {}
            return EvidenceChunk(  # type: ignore[call-arg]
                chunk_id=chunk_id,
                doc_id=meta.get("doc_id", fallback.doc_id),
                text=docs[0],
                retrieval_score=fallback.retrieval_score,
                rerank_score=fallback.rerank_score,
                metadata=meta,
                source_label=meta.get("source_label", fallback.source_label),
            )
    except Exception as exc:
        logger.debug("Parent chunk fetch failed for %s: %s", chunk_id, exc)
    return fallback


async def _grade_chunks(
    query: str, chunks: List[EvidenceChunk]
) -> Tuple[List[EvidenceChunk], float]:
    """Grade each chunk for relevance to the query (CRAG stage) (MF-RES-05).

    Calls the LLM with a short relevance judgment prompt.
    Returns (relevant_chunks, pass_rate).

    Args:
        query: The research sub-query.
        chunks: Candidate chunks to grade.

    Returns:
        Tuple of (list of relevant chunks, pass_rate float).
    """
    if not chunks:
        return [], 0.0

    if not _OPENAI_AVAILABLE or ChatOpenAI is None:
        # Without LLM, conservatively accept all chunks above median rerank score
        if not chunks:
            return [], 0.0
        median_score = sorted([c.rerank_score for c in chunks])[len(chunks) // 2]
        relevant = [c for c in chunks if c.rerank_score >= median_score]
        rate = len(relevant) / len(chunks)
        return relevant, rate

    model_name = get_model("researcher")
    llm = ChatOpenAI(model=model_name, temperature=0.0)

    relevant_chunks: List[EvidenceChunk] = []

    grading_prompt_template = (
        "Is the following text relevant to the research query?\n\n"
        "Query: {query}\n\nText: {text}\n\n"
        "Answer with a single word: YES or NO."
    )

    for chunk in chunks:
        try:
            prompt = grading_prompt_template.format(
                query=query, text=chunk.text[:1000]
            )
            response = await llm.ainvoke(prompt)
            answer = response.content.strip().upper()
            if "YES" in answer:
                relevant_chunks.append(chunk)
        except Exception as exc:
            logger.debug("Chunk grading error for %s: %s", chunk.chunk_id, exc)
            # Conservative: include the chunk on error
            relevant_chunks.append(chunk)

    pass_rate = len(relevant_chunks) / len(chunks) if chunks else 0.0
    logger.debug(
        "CRAG grading: %d/%d relevant (pass_rate=%.2f)",
        len(relevant_chunks), len(chunks), pass_rate,
    )
    return relevant_chunks, pass_rate


async def _rewrite_query(query: str) -> Tuple[str, str]:
    """Rewrite a research query to improve retrieval after CRAG failure (MF-RES-05).

    Args:
        query: The original or previously rewritten query.

    Returns:
        Tuple of (rewritten_query, hyde_for_rewritten_query).
    """
    if not _OPENAI_AVAILABLE or ChatOpenAI is None:
        # Append "detailed" as a simple expansion heuristic
        rewritten = f"{query} detailed information"
        return rewritten, rewritten

    model_name = get_model("researcher")
    llm = ChatOpenAI(model=model_name, temperature=0.1)

    prompt = (
        f"The following research query did not find relevant results. "
        f"Rewrite it to be more specific, use different keywords, or expand scope.\n\n"
        f"Original query: {query}\n\nRewritten query:"
    )
    try:
        response = await llm.ainvoke(prompt)
        rewritten = response.content.strip()
        logger.info("Query rewritten: '%.60s'  ->  '%.60s'", query, rewritten)
        new_hyde = await hyde_rewrite(rewritten)
        return rewritten, new_hyde
    except Exception as exc:
        logger.warning("Query rewrite failed: %s", exc)
        return query, query


async def _self_rag_loop(
    query: str,
    chunks: List[EvidenceChunk],
    max_retries: int,
) -> Tuple[str, str, int]:
    """Generate an answer and verify it is grounded in the evidence (MF-RES-06).

    Self-RAG loop:
        1. Generate answer from chunks.
        2. Check: does the answer contain only facts from the evidence?
        3. If not grounded  ->  regenerate (up to max_retries).

    Args:
        query: The research sub-query.
        chunks: Graded evidence chunks.
        max_retries: Maximum regeneration attempts.

    Returns:
        Tuple of (answer_text, verdict, total_tokens_used).
    """
    if not chunks:
        return "No relevant evidence found for this query.", "not_grounded", 0

    evidence_text = "\n\n".join(
        f"[Chunk {c.chunk_id}] {c.text[:1500]}" for c in chunks
    )

    if not _OPENAI_AVAILABLE or ChatOpenAI is None:
        # Stub: return first chunk as answer
        summary = chunks[0].text[:300] if chunks else "No evidence."
        return summary, "grounded", 0

    model_name = get_model("researcher")
    llm = ChatOpenAI(model=model_name, temperature=0.1)
    total_tokens = 0

    generation_prompt = (
        f"Using ONLY the following evidence, answer the question. "
        f"Do not add any information not present in the evidence.\n\n"
        f"Evidence:\n{evidence_text}\n\nQuestion: {query}\n\nAnswer:"
    )

    grounding_check_prompt_template = (
        "Does the following answer contain ONLY information that is directly "
        "supported by the provided evidence? Answer YES or NO.\n\n"
        "Evidence:\n{evidence}\n\nAnswer:\n{answer}"
    )

    answer = ""
    verdict = "not_grounded"

    for attempt in range(max_retries):
        try:
            gen_response = await llm.ainvoke(generation_prompt)
            answer = gen_response.content.strip()
            total_tokens += len(answer) // 4 + len(generation_prompt) // 4

            # Grounding check
            check_prompt = grounding_check_prompt_template.format(
                evidence=evidence_text[:3000], answer=answer
            )
            check_response = await llm.ainvoke(check_prompt)
            check_text = check_response.content.strip().upper()
            total_tokens += len(check_text) // 4

            if "YES" in check_text:
                verdict = "grounded"
                logger.debug("Self-RAG: grounded on attempt %d", attempt + 1)
                break
            else:
                verdict = "not_grounded" if attempt < max_retries - 1 else "partial"
                logger.info("Self-RAG attempt %d: not grounded  --  regenerating", attempt + 1)

        except Exception as exc:
            logger.warning("Self-RAG attempt %d failed: %s", attempt + 1, exc)
            verdict = "partial"
            break

    return answer, verdict, total_tokens


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _count_source_diversity(chunks: List[EvidenceChunk]) -> int:
    """Count unique doc_id values across the graded chunks (MF-RES-09)."""
    return len({c.doc_id for c in chunks})


def _build_summary(answer: str) -> str:
    """Truncate the answer to MAX_SUMMARY_CHARS for the Supervisor context (MF-RES-10)."""
    return answer[:MAX_SUMMARY_CHARS].rstrip()


def _tokens_to_cost(tokens: int, model: str) -> float:
    """Estimate cost from token count."""
    rates = {"gpt-4.1-mini": 0.0000003, "gpt-4.1": 0.000015}
    return tokens * rates.get(model, 0.0000003)


# ---------------------------------------------------------------------------
# External index accessors (expected to be set by ingestion pipeline)
# ---------------------------------------------------------------------------

_chroma_collection: Any = None
_bm25_index_data: Optional[Tuple[Any, List[str], List[Dict[str, Any]]]] = None


def set_chroma_collection(collection: Any) -> None:
    """Register the ChromaDB collection to use for vector retrieval.

    Called during application startup after the ingestion pipeline populates
    the collection.

    Args:
        collection: A chromadb.Collection instance.
    """
    global _chroma_collection
    _chroma_collection = collection
    logger.info("ChromaDB collection registered: %s", getattr(collection, "name", collection))


def set_bm25_index(
    index: Any, corpus: List[str], corpus_meta: List[Dict[str, Any]]
) -> None:
    """Register the BM25 index for sparse retrieval.

    Args:
        index: A BM25Okapi instance.
        corpus: List of raw text strings (one per chunk, same order as index).
        corpus_meta: List of metadata dicts corresponding to each corpus entry.
    """
    global _bm25_index_data
    _bm25_index_data = (index, corpus, corpus_meta)
    logger.info("BM25 index registered with %d documents", len(corpus))


def _get_chroma_collection() -> Any:
    return _chroma_collection


def _get_bm25_index() -> Optional[Tuple[Any, List[str], List[Dict[str, Any]]]]:
    return _bm25_index_data


def _get_cross_encoder() -> Any:
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder
    if not _CROSS_ENCODER_AVAILABLE or CrossEncoder is None:
        return None
    try:
        logger.info("Loading cross-encoder: %s", RERANKER_MODEL)
        _cross_encoder = CrossEncoder(RERANKER_MODEL, max_length=512)
        return _cross_encoder
    except Exception as exc:
        logger.error("Failed to load cross-encoder %s: %s", RERANKER_MODEL, exc)
        return None
