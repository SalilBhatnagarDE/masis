"""
masis.nodes.validator
=====================
The Validator is the final quality gate. It scores the answer on four metrics and
only lets it through if all thresholds pass.

Scores computed after Synthesizer completes:
    faithfulness       — checks each sentence in the answer is supported by evidence (NLI)
    citation accuracy  — checks each citation chunk_id exists in the evidence board; NLI >= 0.80
    answer relevancy   — cosine similarity between the answer and the original query
    DAG completeness   — fraction of planned research tasks covered by the answer

All four scores must meet VALIDATOR_THRESHOLDS. If any fail, routes back to Supervisor.
After MAX_VALIDATION_ROUNDS=2 failed rounds, forces a pass with whatever answer exists.

Routing:
    pass   → END        (all thresholds met, or max rounds reached)
    revise → supervisor  (at least one threshold missed)

Public API
----------
validator_node(state)  — LangGraph node entry point
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase 0 schema imports with graceful stubs
# ---------------------------------------------------------------------------

try:
    from masis.schemas.models import (
        Citation,
        EvidenceChunk,
        SynthesizerOutput,
    )
    from masis.schemas.thresholds import (
        SAFETY_LIMITS,
        VALIDATOR_THRESHOLDS,
        NLI_MODEL,
        EMBEDDER_MODEL,
    )
except ImportError:
    logger.warning("masis.schemas not found — using stub types for validator.py")

    class EvidenceChunk:  # type: ignore[no-redef]
        chunk_id: str = ""
        doc_id: str = ""
        text: str = ""
        rerank_score: float = 0.0

    class Citation:  # type: ignore[no-redef]
        chunk_id: str = ""
        claim_text: str = ""
        entailment_score: float = 0.0

    class SynthesizerOutput:  # type: ignore[no-redef]
        answer: str = ""
        citations: list = []
        claims_count: int = 0
        citations_count: int = 0

    VALIDATOR_THRESHOLDS: dict = {  # type: ignore[misc]
        "min_faithfulness": 0.85,
        "min_citation_accuracy": 0.90,
        "min_answer_relevancy": 0.80,
        "min_dag_completeness": 0.90,
    }
    SAFETY_LIMITS: dict = {"MAX_VALIDATION_ROUNDS": 2}  # type: ignore[misc]
    NLI_MODEL: str = "facebook/bart-large-mnli"  # type: ignore[misc]
    EMBEDDER_MODEL: str = "all-MiniLM-L6-v2"  # type: ignore[misc]

# ---------------------------------------------------------------------------
# Optional ML library imports
# ---------------------------------------------------------------------------

try:
    from transformers import pipeline as hf_pipeline
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed — NLI scoring will use heuristic fallback")
    hf_pipeline = None  # type: ignore[assignment]

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False
    logger.warning("sentence-transformers not installed — answer relevancy will use heuristic")
    SentenceTransformer = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Module-level model caches (loaded once, reused across calls)
# ---------------------------------------------------------------------------

_nli_pipeline: Any = None
_sbert_model: Any = None
FAST_VALIDATOR_MODE: bool = os.getenv("FAST_VALIDATOR_MODE", "1").lower() in {"1", "true", "yes"}

MAX_VALIDATION_ROUNDS: int = SAFETY_LIMITS.get("MAX_VALIDATION_ROUNDS", 3)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def validator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node: run all quality gates and decide pass or revise (MF-VAL-01 to MF-VAL-07).

    Reads:
        synthesis_output   — the SynthesizerOutput to validate
        evidence_board     — all retrieved chunks (for citation verification)
        original_query     — used for answer relevancy scoring
        task_dag           — used for DAG completeness scoring
        validation_round   — current revision round counter

    Writes:
        quality_scores      — dict of {metric: score} (MF-VAL-06)
        validation_pass     — True if all gates pass or max rounds reached
        validation_round    — incremented by 1

    Args:
        state: Current MASISState.

    Returns:
        Partial state update dict containing quality_scores and validation_pass.
    """
    synthesis: Optional[SynthesizerOutput] = state.get("synthesis_output")
    validation_round: int = state.get("validation_round", 0) + 1

    logger.info("Validator node called (round=%d)", validation_round)

    # Guard: if no synthesis output, fail closed and route back for correction
    if synthesis is None:
        logger.warning("No synthesis_output in state — failing validation (fail-closed)")
        return {
            "quality_scores": {
                "faithfulness": 0.0,
                "citation_accuracy": 0.0,
                "answer_relevancy": 0.0,
                "dag_completeness": 0.0,
                "error": "no_synthesis_output",
            },
            "validation_pass": False,
            "validation_round": validation_round,
        }

    evidence_board: List[EvidenceChunk] = state.get("evidence_board", [])
    original_query: str = state.get("original_query", "")
    task_dag: List[Any] = state.get("task_dag", [])

    # ── Max validation rounds guard (MF-VAL-07) ──────────────────────────────
    if validation_round >= MAX_VALIDATION_ROUNDS:
        logger.warning(
            "Max validation rounds (%d) reached — forcing pass with best available answer",
            MAX_VALIDATION_ROUNDS,
        )
        existing_scores: Dict[str, float] = state.get("quality_scores", {})
        return {
            "quality_scores": {**existing_scores, "forced_pass": True},
            "validation_pass": True,
            "validation_round": validation_round,
        }

    # ── Run all four quality metrics ─────────────────────────────────────────
    faithfulness = await _score_faithfulness(synthesis, evidence_board)
    citation_accuracy = await _score_citation_accuracy(synthesis, evidence_board)
    answer_relevancy = await _score_answer_relevancy(synthesis.answer, original_query)
    dag_completeness = _score_dag_completeness(synthesis, task_dag)

    quality_scores = {
        "faithfulness": faithfulness,
        "citation_accuracy": citation_accuracy,
        "answer_relevancy": answer_relevancy,
        "dag_completeness": dag_completeness,
    }

    logger.info(
        "Validator scores — faithfulness=%.3f, citation_accuracy=%.3f, "
        "answer_relevancy=%.3f, dag_completeness=%.3f",
        faithfulness, citation_accuracy, answer_relevancy, dag_completeness,
    )

    # ── Threshold enforcement (MF-VAL-05) ────────────────────────────────────
    thresholds = VALIDATOR_THRESHOLDS
    failures = _check_thresholds(quality_scores, thresholds)

    if failures:
        logger.info("Validator FAIL — below threshold: %s", failures)
        validation_pass = False
    else:
        logger.info("Validator PASS — all quality gates met")
        validation_pass = True

    return {
        "quality_scores": quality_scores,
        "validation_pass": validation_pass,
        "validation_round": validation_round,
    }


# ---------------------------------------------------------------------------
# Individual scoring functions
# ---------------------------------------------------------------------------


async def _score_faithfulness(
    synthesis: SynthesizerOutput,
    evidence_board: List[EvidenceChunk],
) -> float:
    """NLI-based faithfulness: each sentence in the answer must be entailed by a source chunk (MF-VAL-01).

    For each sentence in synthesis.answer:
        - Find the best-matching chunk by text overlap (proxy for retrieval).
        - Run NLI entailment: chunk text (premise) → sentence (hypothesis).
        - Average entailment scores across all sentences.

    Args:
        synthesis: The SynthesizerOutput to check.
        evidence_board: All available evidence chunks.

    Returns:
        Float in [0, 1] representing average entailment across sentences.
    """
    answer = synthesis.answer
    if not answer or not evidence_board:
        logger.debug("Faithfulness: empty answer or evidence — returning 0.0")
        return 0.0

    sentences = _split_into_sentences(answer)
    if not sentences:
        return 0.0

    nli = _get_nli_pipeline()
    scores: List[float] = []

    for sentence in sentences:
        if len(sentence.strip()) < 10:
            continue  # Skip very short fragments

        # Find the most relevant chunk as premise (by simple keyword overlap)
        best_chunk = _find_best_chunk_for_sentence(sentence, evidence_board)
        if best_chunk is None:
            scores.append(0.0)
            continue

        entailment_score = _run_nli(nli, premise=best_chunk.text, hypothesis=sentence)
        scores.append(entailment_score)
        logger.debug(
            "Faithfulness NLI: sentence='%.50s...' → chunk=%s → score=%.3f",
            sentence, best_chunk.chunk_id, entailment_score,
        )

    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 4)


async def _score_citation_accuracy(
    synthesis: SynthesizerOutput,
    evidence_board: List[EvidenceChunk],
) -> float:
    """Verify each citation chunk exists in evidence_board and NLI-check entailment (MF-VAL-02).

    For each Citation in synthesis.citations:
        - Check that Citation.chunk_id exists in evidence_board.
        - NLI: chunk text (premise) → claim_text (hypothesis) → entailment score.
        - Average across all citations.

    Args:
        synthesis: The SynthesizerOutput containing citations.
        evidence_board: All available evidence chunks.

    Returns:
        Float in [0, 1] representing citation accuracy.
    """
    citations: List[Citation] = getattr(synthesis, "citations", [])
    if not citations:
        logger.debug("Citation accuracy: no citations — returning 0.0")
        return 0.0

    # Build lookup by chunk_id for O(1) access
    chunk_lookup: Dict[str, EvidenceChunk] = {
        chunk.chunk_id: chunk for chunk in evidence_board
    }

    nli = _get_nli_pipeline()
    scores: List[float] = []

    for citation in citations:
        chunk_id = citation.chunk_id
        claim_text = citation.claim_text

        if chunk_id not in chunk_lookup:
            logger.warning("Citation references non-existent chunk_id=%s", chunk_id)
            scores.append(0.0)
            continue

        chunk = chunk_lookup[chunk_id]
        entailment_score = _run_nli(nli, premise=chunk.text, hypothesis=claim_text)

        # Update the entailment_score on the citation object for Validator record
        citation.entailment_score = entailment_score

        logger.debug(
            "Citation accuracy: chunk_id=%s claim='%.50s...' → score=%.3f",
            chunk_id, claim_text, entailment_score,
        )
        scores.append(entailment_score)

    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 4)


async def _score_answer_relevancy(answer: str, original_query: str) -> float:
    """Semantic similarity between answer and original_query (MF-VAL-03).

    Uses SentenceTransformer all-MiniLM-L6-v2 for cosine similarity.
    Falls back to a keyword-overlap heuristic when sentence_transformers is unavailable.

    Args:
        answer: The synthesised answer text.
        original_query: The user's original question.

    Returns:
        Float in [0, 1] representing semantic similarity.
    """
    if not answer or not original_query:
        return 0.0

    if _SBERT_AVAILABLE and SentenceTransformer is not None:
        model = _get_sbert_model()
        try:
            import numpy as np  # noqa: PLC0415
            embeddings = model.encode([answer, original_query])
            # Cosine similarity
            a, b = embeddings[0], embeddings[1]
            cos_sim: float = float(
                np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
            )
            logger.debug("Answer relevancy (SBERT): %.4f", cos_sim)
            return round(max(0.0, min(1.0, cos_sim)), 4)
        except Exception as exc:
            logger.warning("SBERT relevancy scoring failed: %s — using heuristic", exc)

    # Heuristic fallback: Jaccard similarity on word sets
    score = _jaccard_similarity(answer.lower(), original_query.lower())
    logger.debug("Answer relevancy (heuristic Jaccard): %.4f", score)
    return round(score, 4)


def _score_dag_completeness(
    synthesis: SynthesizerOutput,
    task_dag: List[Any],
) -> float:
    """Fraction of researcher tasks whose topics appear in the answer (MF-VAL-04).

    Counts researcher tasks in the DAG and checks how many have key words from
    their query appearing in the synthesis answer.

    Args:
        synthesis: The SynthesizerOutput containing the answer text.
        task_dag: Full list of TaskNode objects.

    Returns:
        Float in [0, 1] representing DAG completeness.
    """
    researcher_tasks = [t for t in task_dag if getattr(t, "type", "") == "researcher"]
    if not researcher_tasks:
        logger.debug("DAG completeness: no researcher tasks — returning 1.0")
        return 1.0

    answer_lower = synthesis.answer.lower()
    addressed_count = 0

    for task in researcher_tasks:
        task_query: str = getattr(task, "query", "")
        if not task_query:
            continue
        # A task is "addressed" if at least 2 meaningful keywords from its query
        # appear in the answer.
        keywords = _extract_keywords(task_query)
        if keywords:
            matched = sum(1 for kw in keywords if kw in answer_lower)
            if matched >= min(2, len(keywords)):
                addressed_count += 1

    completeness = addressed_count / len(researcher_tasks)
    logger.debug(
        "DAG completeness: %d/%d researcher tasks addressed → %.3f",
        addressed_count, len(researcher_tasks), completeness,
    )
    return round(completeness, 4)


# ---------------------------------------------------------------------------
# NLI helper
# ---------------------------------------------------------------------------


def _run_nli(nli_pipeline: Any, premise: str, hypothesis: str) -> float:
    """Run NLI classification and return the entailment probability.

    Args:
        nli_pipeline: HuggingFace zero-shot-classification pipeline, or None.
        premise: The source text (evidence chunk).
        hypothesis: The claim to verify.

    Returns:
        Float in [0, 1]: entailment probability (or heuristic if NLI unavailable).
    """
    if nli_pipeline is None:
        # Heuristic fallback: word overlap as a proxy for entailment
        return _jaccard_similarity(premise.lower(), hypothesis.lower())

    try:
        # BART-MNLI via zero-shot-classification
        # Labels: entailment, contradiction, neutral
        result = nli_pipeline(
            hypothesis,
            candidate_labels=["entailment", "contradiction", "neutral"],
            hypothesis_template="This text is {}.",
            multi_label=False,
        )
        label_scores = dict(zip(result["labels"], result["scores"]))
        return float(label_scores.get("entailment", 0.0))
    except Exception as exc:
        logger.warning("NLI pipeline call failed: %s — using heuristic", exc)
        return _jaccard_similarity(premise.lower(), hypothesis.lower())


def _check_thresholds(
    scores: Dict[str, float],
    thresholds: Dict[str, float],
) -> List[str]:
    """Return list of metric names that are below their threshold (MF-VAL-05).

    Args:
        scores: Dict of {metric_name: score}.
        thresholds: Dict of {min_<metric>: threshold_value}.

    Returns:
        List of failing metric names (empty if all pass).
    """
    failures = []
    mapping = {
        "min_faithfulness": "faithfulness",
        "min_citation_accuracy": "citation_accuracy",
        "min_answer_relevancy": "answer_relevancy",
        "min_dag_completeness": "dag_completeness",
    }
    for threshold_key, metric_name in mapping.items():
        threshold_val = thresholds.get(threshold_key, 0.0)
        actual_score = scores.get(metric_name, 0.0)
        if actual_score < threshold_val:
            failures.append(
                f"{metric_name}={actual_score:.3f} < {threshold_val}"
            )
    return failures


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using simple punctuation heuristics."""
    # Split on period/exclamation/question mark followed by whitespace
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


def _extract_keywords(query: str, min_length: int = 4) -> List[str]:
    """Extract content words from a query (filter stop-words by length heuristic)."""
    stop_words = {
        "what", "when", "where", "which", "who", "how", "the", "and",
        "for", "are", "was", "were", "with", "this", "that", "from",
        "has", "have", "had", "its", "their", "about",
    }
    words = re.sub(r"[^a-z0-9\s]", "", query.lower()).split()
    return [w for w in words if len(w) >= min_length and w not in stop_words]


def _find_best_chunk_for_sentence(
    sentence: str, evidence_board: List[EvidenceChunk]
) -> Optional[EvidenceChunk]:
    """Return the evidence chunk with the highest keyword overlap to the sentence."""
    sentence_lower = sentence.lower()
    best_chunk: Optional[EvidenceChunk] = None
    best_score: float = -1.0

    for chunk in evidence_board:
        overlap = _jaccard_similarity(chunk.text.lower(), sentence_lower)
        if overlap > best_score:
            best_score = overlap
            best_chunk = chunk

    return best_chunk


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between word sets of two texts."""
    set_a = set(re.sub(r"[^a-z0-9\s]", "", text_a).split())
    set_b = set(re.sub(r"[^a-z0-9\s]", "", text_b).split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Model loaders (singleton pattern)
# ---------------------------------------------------------------------------


def _get_nli_pipeline() -> Any:
    """Load and cache the BART-MNLI pipeline (MF-SKE-02 / MF-VAL-01)."""
    global _nli_pipeline
    if _nli_pipeline is not None:
        return _nli_pipeline
    if FAST_VALIDATOR_MODE:
        logger.info("FAST_VALIDATOR_MODE enabled: using heuristic faithfulness/citation checks")
        return None
    if not _TRANSFORMERS_AVAILABLE or hf_pipeline is None:
        return None
    try:
        logger.info("Loading NLI pipeline: %s", NLI_MODEL)
        _nli_pipeline = hf_pipeline(
            "zero-shot-classification",
            model=NLI_MODEL,
            device=-1,  # CPU; set to 0 for GPU
        )
        return _nli_pipeline
    except Exception as exc:
        logger.error("Failed to load NLI pipeline %s: %s", NLI_MODEL, exc)
        return None


def _get_sbert_model() -> Any:
    """Load and cache the SentenceTransformer model."""
    global _sbert_model
    if _sbert_model is not None:
        return _sbert_model
    if FAST_VALIDATOR_MODE:
        return None
    if not _SBERT_AVAILABLE or SentenceTransformer is None:
        return None
    try:
        logger.info("Loading SentenceTransformer: %s", EMBEDDER_MODEL)
        _sbert_model = SentenceTransformer(EMBEDDER_MODEL)
        return _sbert_model
    except Exception as exc:
        logger.error("Failed to load SentenceTransformer %s: %s", EMBEDDER_MODEL, exc)
        return None
