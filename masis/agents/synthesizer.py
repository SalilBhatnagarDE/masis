"""
masis.agents.synthesizer
========================
Synthesizer agent — produces the final cited answer (ENG-09, MF-SYN-01 through MF-SYN-08).

Pipeline
--------
1. U-shape context ordering  — best evidence at start and end, weakest in middle (MF-SYN-01)
2. Critique integration      — embed Skeptic's flags in the prompt (MF-SYN-02)
3. Citation-enforced generation — gpt-4.1 with SynthesizerOutput (min_length=1 on citations) (MF-SYN-03/04)
4. Post-hoc NLI verification — fill entailment_score on each Citation (MF-SYN-05)
5. Edge cases:
     partial mode    — force_synthesize adds disclaimer about missing dimensions (MF-SYN-06)
     no evidence     — honest "no evidence found" response (MF-SYN-07)
     both sides      — when Skeptic reconciled contradictions, present both perspectives (MF-SYN-08)

Public API
----------
run_synthesizer(task, state) → AgentOutput
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
FAST_NLI_MODE: bool = os.getenv("FAST_NLI_MODE", "1").lower() in {"1", "true", "yes"}

# ---------------------------------------------------------------------------
# Phase 0 schema imports with stubs
# ---------------------------------------------------------------------------

try:
    from masis.schemas.models import (
        AgentOutput,
        Citation,
        EvidenceChunk,
        SkepticOutput,
        SynthesizerOutput,
    )
    from masis.schemas.thresholds import NLI_MODEL
except ImportError:
    logger.warning("masis.schemas not found — using stub types for synthesizer.py")

    class EvidenceChunk:  # type: ignore[no-redef]
        def __init__(self, **kw: Any) -> None:
            for k, v in kw.items():
                setattr(self, k, v)
        chunk_id: str = ""
        doc_id: str = ""
        text: str = ""
        rerank_score: float = 0.0

    class Citation:  # type: ignore[no-redef]
        def __init__(self, **kw: Any) -> None:
            for k, v in kw.items():
                setattr(self, k, v)
        chunk_id: str = ""
        claim_text: str = ""
        entailment_score: float = 0.0

    class SkepticOutput:  # type: ignore[no-redef]
        def __init__(self, **kw: Any) -> None:
            for k, v in kw.items():
                setattr(self, k, v)

    class SynthesizerOutput:  # type: ignore[no-redef]
        def __init__(self, **kw: Any) -> None:
            for k, v in kw.items():
                setattr(self, k, v)
        def to_criteria_dict(self) -> dict:
            return {
                "citations_count": getattr(self, "citations_count", 0),
                "claims_count": getattr(self, "claims_count", 0),
                "all_citations_in_evidence_board": getattr(
                    self, "all_citations_in_evidence_board", False
                ),
            }

    class AgentOutput:  # type: ignore[no-redef]
        def __init__(self, **kw: Any) -> None:
            for k, v in kw.items():
                setattr(self, k, v)

    NLI_MODEL = "facebook/bart-large-mnli"

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------

try:
    from langchain_openai import ChatOpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    logger.warning("langchain_openai not installed — synthesizer will use stub response")
    ChatOpenAI = None  # type: ignore[assignment,misc]

try:
    from transformers import pipeline as hf_pipeline
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed — post-hoc NLI verification unavailable")
    hf_pipeline = None  # type: ignore[assignment]

try:
    from masis.config.model_routing import get_model
except ImportError:
    import os

    def get_model(role: str, override: Optional[str] = None) -> str:  # type: ignore[misc]
        return override or os.getenv("MODEL_SYNTHESIZER", "gpt-4.1")

# Module-level NLI pipeline cache
_nli_pipeline: Any = None

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYNTHESIZER_SYSTEM_PROMPT = """You are the Final Synthesis Agent.

Mission:
Produce a clear, consistent, and fully traceable answer using only approved evidence.

Non-negotiable rules:
1. Every factual claim must be supported by at least one citation chunk_id from evidence.
2. Do not add any fact not grounded in evidence.
3. Respect skeptic findings:
   - unresolved concerns must appear as caveats
   - reconciled contradictions should present both sides and the reconciliation logic
4. If running in partial mode, clearly disclose missing dimensions.
5. Prioritize correctness and traceability over style.

Output contract:
Return valid JSON matching SynthesizerOutput:
- answer
- citations
- claims_count
- all_citations_in_evidence_board

Few-shot synthesis patterns (compact):
- if evidence conflicts but skeptic reconciled -> include both statements + reconciliation
- if evidence is thin for a claim -> downgrade certainty and add caveat
- if requested dimension missing -> explicitly state missing dimension
"""

SYNTHESIZER_USER_TEMPLATE = """ORIGINAL QUESTION:
{original_query}

EVIDENCE BOARD (already curated and ordered):
{evidence_text}

SKEPTIC FINDINGS TO INCORPORATE:
{critique_instructions}

PARTIAL MODE INSTRUCTION:
{partial_mode_instruction}

Write the final synthesis now.
Every factual statement must map to citation chunk_id(s).
"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_synthesizer(task: Any, state: Dict[str, Any]) -> AgentOutput:
    """Execute the synthesizer pipeline and return AgentOutput.

    Args:
        task: TaskNode with .task_id.
        state: Filtered state containing evidence_board, critique_notes,
               original_query, supervisor_decision.

    Returns:
        AgentOutput wrapping a SynthesizerOutput.
    """
    task_id = getattr(task, "task_id", "unknown")
    evidence_board: List[EvidenceChunk] = state.get("evidence_board", [])
    critique_notes: Optional[SkepticOutput] = state.get("critique_notes")
    original_query: str = state.get("original_query", "")
    supervisor_decision: str = state.get("supervisor_decision", "")
    task_dag: List[Any] = state.get("task_dag", [])

    logger.info(
        "Synthesizer started: task_id=%s, evidence_chunks=%d, force=%s",
        task_id, len(evidence_board), supervisor_decision == "force_synthesize",
    )

    # ── No-evidence honest answer (MF-SYN-07) ───────────────────────────────
    if not evidence_board:
        logger.warning("Synthesizer: empty evidence board — returning honest no-evidence answer")
        no_evidence_output = _build_no_evidence_output(task_id, original_query)
        return AgentOutput(  # type: ignore[call-arg]
            task_id=task_id,
            agent_type="synthesizer",
            status="success",
            summary="No evidence was found to support an answer.",
            evidence=[],
            criteria_result=no_evidence_output.to_criteria_dict(),
            tokens_used=0,
            cost_usd=0.0,
            raw_output=no_evidence_output,
        )

    try:
        synth_output = await _run_synthesis_pipeline(
            task_id=task_id,
            original_query=original_query,
            evidence_board=evidence_board,
            critique_notes=critique_notes,
            is_partial=supervisor_decision == "force_synthesize",
            task_dag=task_dag,
        )
    except Exception as exc:
        logger.error("Synthesizer pipeline failed: %s", exc, exc_info=True)
        return AgentOutput(  # type: ignore[call-arg]
            task_id=task_id,
            agent_type="synthesizer",
            status="failed",
            summary=f"Synthesis error: {exc}",
            error_detail=str(exc),
        )

    logger.info(
        "Synthesizer done: task_id=%s citations=%d claims=%d",
        task_id, synth_output.citations_count, synth_output.claims_count,
    )

    return AgentOutput(  # type: ignore[call-arg]
        task_id=task_id,
        agent_type="synthesizer",
        status="success",
        summary=synth_output.answer,
        evidence=[],   # Synthesizer does not add to evidence board
        criteria_result=synth_output.to_criteria_dict(),
        tokens_used=synth_output.tokens_used,
        cost_usd=synth_output.cost_usd,
        raw_output=synth_output,
    )


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

async def _run_synthesis_pipeline(
    task_id: str,
    original_query: str,
    evidence_board: List[EvidenceChunk],
    critique_notes: Optional[SkepticOutput],
    is_partial: bool,
    task_dag: List[Any],
) -> SynthesizerOutput:
    """Orchestrate all synthesis stages.

    Args:
        task_id: Task identifier for logging.
        original_query: The user's original question.
        evidence_board: All available evidence chunks.
        critique_notes: SkepticOutput from the Skeptic agent, or None.
        is_partial: True when force_synthesize was triggered (budget/time limit).
        task_dag: Full task DAG for missing-dimension detection.

    Returns:
        Populated SynthesizerOutput.
    """
    # ── Stage 1: U-shape ordering (MF-SYN-01) ───────────────────────────────
    ordered_evidence = u_shape_order(evidence_board)
    logger.debug("U-shape ordered %d chunks", len(ordered_evidence))

    # ── Stage 2: Critique integration (MF-SYN-02) ───────────────────────────
    critique_instructions = _build_critique_instructions(critique_notes)

    # ── Stage 3: Partial mode instruction (MF-SYN-06) ───────────────────────
    partial_mode_instruction = ""
    missing_dimensions: List[str] = []
    if is_partial:
        missing_dimensions = _detect_missing_dimensions(task_dag, ordered_evidence)
        if missing_dimensions:
            partial_mode_instruction = (
                f"NOTE: This is a PARTIAL answer. The following dimensions could not be "
                f"fully researched and are missing from this synthesis: "
                f"{', '.join(missing_dimensions)}. Include a disclaimer at the end of your answer."
            )
        else:
            partial_mode_instruction = (
                "NOTE: This is a partial/forced synthesis due to budget constraints. "
                "Include a disclaimer noting the answer may be incomplete."
            )

    # ── Stage 4: Format evidence text ───────────────────────────────────────
    evidence_text = _format_evidence_for_synthesis(ordered_evidence)

    # ── Stage 5: LLM synthesis with Pydantic citation enforcement (MF-SYN-03) ─
    answer, citations, tokens_used, cost_usd = await _generate_synthesis(
        original_query=original_query,
        evidence_text=evidence_text,
        critique_instructions=critique_instructions,
        partial_mode_instruction=partial_mode_instruction,
        evidence_board=ordered_evidence,
    )

    # ── Stage 6: Post-hoc NLI verification (MF-SYN-05) ─────────────────────
    citations = _verify_citations_nli(citations, evidence_board)

    # ── Stage 7: Verify all citation chunk_ids exist in evidence_board ───────
    evidence_ids = {chunk.chunk_id for chunk in evidence_board}
    all_valid = all(c.chunk_id in evidence_ids for c in citations)

    claims_count = _count_claims(answer)

    return SynthesizerOutput(  # type: ignore[call-arg]
        task_id=task_id,
        answer=answer,
        citations=citations if citations else _make_fallback_citation(evidence_board),
        claims_count=claims_count,
        citations_count=len(citations),
        all_citations_in_evidence_board=all_valid,
        is_partial=is_partial,
        missing_dimensions=missing_dimensions,
        tokens_used=tokens_used,
        cost_usd=cost_usd,
    )


def u_shape_order(chunks: List[EvidenceChunk]) -> List[EvidenceChunk]:
    """Reorder evidence using U-shape (lost-in-the-middle mitigation) (MF-SYN-01).

    LLMs attend most strongly to tokens at the start and end of context.
    The weakest evidence is placed in the middle where attention is lowest.

    Algorithm:
        Sort descending by rerank_score.
        Even indices (0, 2, 4, ...) → placed at the START (left side).
        Odd indices  (1, 3, 5, ...) → placed at the END (right side, reversed).
        Result = left + reversed(right)

    Example (scores): [0.92, 0.87, 0.81, 0.74, 0.69]
        left  = [chunk(0.92), chunk(0.81), chunk(0.69)]  (indices 0, 2, 4)
        right = [chunk(0.87), chunk(0.74)]                (indices 1, 3)
        final = [0.92, 0.81, 0.69, 0.74, 0.87]

    Args:
        chunks: Evidence chunks (unsorted).

    Returns:
        Reordered list with best chunks at start and end.
    """
    if len(chunks) <= 2:
        return sorted(chunks, key=lambda c: c.rerank_score, reverse=True)

    sorted_chunks = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)
    left: List[EvidenceChunk] = []
    right: List[EvidenceChunk] = []

    for i, chunk in enumerate(sorted_chunks):
        if i % 2 == 0:
            left.append(chunk)    # best, 2nd-best-of-even, etc. → start
        else:
            right.append(chunk)   # 2nd-best, 4th-best, etc. → end (reversed)

    return left + list(reversed(right))


async def _generate_synthesis(
    original_query: str,
    evidence_text: str,
    critique_instructions: str,
    partial_mode_instruction: str,
    evidence_board: List[EvidenceChunk],
) -> Tuple[str, List[Citation], int, float]:
    """Call gpt-4.1 with structured output to generate the cited synthesis (MF-SYN-03).

    Args:
        original_query: The user's original question.
        evidence_text: Formatted evidence chunks.
        critique_instructions: Formatted Skeptic critique.
        partial_mode_instruction: Instruction for partial/force_synthesize mode.
        evidence_board: Original evidence for fallback citation building.

    Returns:
        Tuple of (answer, citations, tokens_used, cost_usd).
    """
    if not _OPENAI_AVAILABLE or ChatOpenAI is None:
        logger.warning("Synthesizer LLM unavailable — returning heuristic answer")
        return _heuristic_synthesis(original_query, evidence_board)

    model_name = get_model("synthesizer")
    llm = ChatOpenAI(model=model_name, temperature=0.1)

    user_prompt = SYNTHESIZER_USER_TEMPLATE.format(
        original_query=original_query,
        evidence_text=evidence_text,
        critique_instructions=critique_instructions,
        partial_mode_instruction=partial_mode_instruction,
    )

    try:
        structured_llm = llm.with_structured_output(SynthesizerOutput)
        synth: SynthesizerOutput = await structured_llm.ainvoke([
            {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])

        tokens_used = (len(SYNTHESIZER_SYSTEM_PROMPT) + len(user_prompt)) // 4 + len(synth.answer) // 4
        cost_usd = tokens_used * 0.000015  # gpt-4.1 rate

        citations = synth.citations
        return _dedup_inline_citations(synth.answer), citations, tokens_used, cost_usd

    except Exception as exc:
        logger.warning("Synthesis LLM call failed: %s — using heuristic fallback", exc)
        return _heuristic_synthesis(original_query, evidence_board)


def _dedup_inline_citations(answer: str) -> str:
    """Remove duplicate chunk IDs within each inline citation bracket in the answer text.

    The LLM occasionally repeats the same chunk_id multiple times within a single
    bracket group, e.g. [A, B, A, C, B] → [A, B, C].  This preserves order of first
    appearance and removes duplicates while keeping the bracket intact.
    """
    def _dedup_bracket(m: re.Match) -> str:
        ids = [x.strip() for x in m.group(1).split(",")]
        seen: dict = {}
        deduped = [seen.setdefault(i, i) for i in ids if i not in seen]
        return f"[{', '.join(deduped)}]"

    return re.sub(r"\[([^\]]+)\]", _dedup_bracket, answer)


# ---------------------------------------------------------------------------
# Evidence and critique formatting
# ---------------------------------------------------------------------------

def _format_evidence_for_synthesis(chunks: List[EvidenceChunk]) -> str:
    """Format evidence chunks for the synthesis prompt.

    Args:
        chunks: U-shape ordered evidence chunks.

    Returns:
        Formatted string for the LLM prompt.
    """
    lines: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.source_label or chunk.doc_id
        lines.append(
            f"[Evidence {i} | chunk_id={chunk.chunk_id} | source={source}]\n"
            f"{chunk.text[:1500]}"
        )
    return "\n\n".join(lines)


def _build_critique_instructions(critique_notes: Optional[SkepticOutput]) -> str:
    """Build critique instruction string from SkepticOutput for the synthesis prompt (MF-SYN-02).

    Args:
        critique_notes: SkepticOutput from the Skeptic agent, or None.

    Returns:
        Formatted critique instructions string.
    """
    if critique_notes is None:
        return "No specific critique flags. Synthesize normally."

    instructions: List[str] = []

    # Forward-looking flags → present as outlook (MF-SKE-07)
    fl_flags = getattr(critique_notes, "forward_looking_flags", [])
    if fl_flags:
        instructions.append(
            "FORWARD-LOOKING STATEMENTS: The following claims are forward-looking. "
            "Present them as OUTLOOK or EXPECTATION, not established fact:\n"
            + "\n".join(f"  - {f}" for f in fl_flags[:5])
        )

    # Single-source warnings → note reduced confidence (MF-SKE-06)
    ss_warnings = getattr(critique_notes, "single_source_warnings", [])
    if ss_warnings:
        instructions.append(
            "SINGLE-SOURCE CLAIMS: Note reduced confidence for these:\n"
            + "\n".join(f"  - {w}" for w in ss_warnings[:3])
        )

    # Reconciled contradictions → present both sides (MF-SYN-08)
    reconciliations = getattr(critique_notes, "reconciliations", [])
    if reconciliations:
        instructions.append(
            "RECONCILED CONTRADICTIONS: Present BOTH perspectives for each:\n"
            + "\n".join(
                f"  - {r.get('explanation', str(r))}" for r in reconciliations[:3]
            )
        )

    # Logical gaps → acknowledge them
    gaps = getattr(critique_notes, "logical_gaps", [])
    if gaps:
        instructions.append(
            "LOGICAL GAPS: Acknowledge these limitations in your answer:\n"
            + "\n".join(f"  - {g}" for g in gaps[:3])
        )

    if not instructions:
        conf = getattr(critique_notes, "overall_confidence", 0.0)
        return f"Evidence confidence: {conf:.2f}. No critical flags detected."

    return "\n\n".join(instructions)


def _build_no_evidence_output(task_id: str, original_query: str) -> SynthesizerOutput:
    """Build an honest 'no evidence' SynthesizerOutput (MF-SYN-07).

    Args:
        task_id: Task identifier.
        original_query: The user's question.

    Returns:
        SynthesizerOutput with an honest no-evidence answer and a placeholder citation.
    """
    answer = (
        f"No evidence was found in the available documents to answer the question: "
        f"'{original_query}'. "
        "The research pipeline retrieved no relevant chunks after CRAG grading. "
        "Consider expanding the search scope or checking if the information exists "
        "in the document collection."
    )
    # SynthesizerOutput requires at least 1 citation — use a placeholder
    placeholder_citation = Citation(  # type: ignore[call-arg]
        chunk_id="no_evidence",
        claim_text="No evidence was found.",
        entailment_score=0.0,
    )
    return SynthesizerOutput(  # type: ignore[call-arg]
        task_id=task_id,
        answer=answer,
        citations=[placeholder_citation],
        claims_count=1,
        citations_count=1,
        all_citations_in_evidence_board=False,
        is_partial=True,
        missing_dimensions=["all dimensions — no evidence retrieved"],
        tokens_used=0,
        cost_usd=0.0,
    )


def _make_fallback_citation(evidence_board: List[EvidenceChunk]) -> List[Citation]:
    """Create a fallback citation list when the LLM fails to produce citations.

    Args:
        evidence_board: Available evidence (uses the first chunk as fallback).

    Returns:
        List with one Citation pointing to the first evidence chunk.
    """
    if evidence_board:
        first_chunk = evidence_board[0]
        return [Citation(  # type: ignore[call-arg]
            chunk_id=first_chunk.chunk_id,
            claim_text="See referenced evidence for details.",
            entailment_score=0.0,
        )]
    return [Citation(  # type: ignore[call-arg]
        chunk_id="fallback",
        claim_text="Fallback citation — LLM citation generation failed.",
        entailment_score=0.0,
    )]


def _verify_citations_nli(
    citations: List[Citation],
    evidence_board: List[EvidenceChunk],
) -> List[Citation]:
    """Post-hoc NLI: verify each citation's entailment and fill entailment_score (MF-SYN-05).

    Args:
        citations: Citations produced by the LLM.
        evidence_board: All available evidence chunks for lookup.

    Returns:
        Citations with entailment_score populated.
    """
    nli = _get_nli_pipeline()
    chunk_lookup = {c.chunk_id: c for c in evidence_board}

    for citation in citations:
        chunk = chunk_lookup.get(citation.chunk_id)
        if chunk is None:
            citation.entailment_score = 0.0
            continue

        if nli is not None:
            try:
                result = nli(
                    citation.claim_text,
                    candidate_labels=["entailment", "contradiction", "neutral"],
                    hypothesis_template="This text is {}.",
                    multi_label=False,
                )
                label_scores = dict(zip(result["labels"], result["scores"]))
                citation.entailment_score = float(label_scores.get("entailment", 0.0))
            except Exception as exc:
                logger.debug("NLI citation check failed for %s: %s", citation.chunk_id, exc)
                citation.entailment_score = _jaccard_sim(
                    chunk.text.lower(), citation.claim_text.lower()
                )
        else:
            # Heuristic fallback
            citation.entailment_score = _jaccard_sim(
                chunk.text.lower(), citation.claim_text.lower()
            )

    return citations


def _detect_missing_dimensions(
    task_dag: List[Any],
    evidence_board: List[EvidenceChunk],
) -> List[str]:
    """Detect researcher task topics that are not represented in the evidence board.

    Used for partial synthesis disclaimers (MF-SYN-06).

    Args:
        task_dag: Full task DAG.
        evidence_board: Available evidence.

    Returns:
        List of query strings for unaddressed researcher tasks.
    """
    evidence_text = " ".join(c.text.lower() for c in evidence_board)
    missing: List[str] = []

    for task in task_dag:
        if getattr(task, "type", "") != "researcher":
            continue
        if getattr(task, "status", "") == "done":
            continue  # Already successfully completed

        query = getattr(task, "query", "")
        if not query:
            continue

        # Heuristic: if fewer than 2 keywords from the query appear in evidence, it's missing
        keywords = [w for w in query.lower().split() if len(w) > 4]
        if keywords and sum(1 for kw in keywords if kw in evidence_text) < 2:
            missing.append(query)

    return missing


def _heuristic_synthesis(
    original_query: str,
    evidence_board: List[EvidenceChunk],
) -> Tuple[str, List[Citation], int, float]:
    """Produce a basic synthesis without an LLM call (fallback).

    Concatenates the top evidence chunks as a structured summary.

    Args:
        original_query: The user's question.
        evidence_board: Available evidence.

    Returns:
        Tuple of (answer, citations, tokens_used, cost_usd).
    """
    if not evidence_board:
        return (
            "No evidence available for synthesis.",
            [Citation(chunk_id="none", claim_text="No evidence.", entailment_score=0.0)],  # type: ignore[call-arg]
            0, 0.0,
        )

    ordered = u_shape_order(evidence_board)[:3]  # Use top 3 chunks
    answer_parts = [f"Based on available evidence:\n"]
    citations: List[Citation] = []

    for chunk in ordered:
        excerpt = chunk.text[:400].strip()
        answer_parts.append(f"- {excerpt} [Source: {chunk.source_label or chunk.chunk_id}]")
        citations.append(Citation(  # type: ignore[call-arg]
            chunk_id=chunk.chunk_id,
            claim_text=excerpt[:200],
            entailment_score=0.9,  # Heuristic: direct quote, assume high entailment
        ))

    answer = "\n".join(answer_parts)
    return answer, citations, 0, 0.0


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _count_claims(answer: str) -> int:
    """Count the number of factual sentences in the answer as a proxy for claims.

    Args:
        answer: The synthesised answer text.

    Returns:
        Estimated number of distinct factual claims.
    """
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    # Filter out very short sentences (connectors, headings)
    return sum(1 for s in sentences if len(s.strip()) > 20)


def _jaccard_sim(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between two text strings."""
    set_a = set(re.sub(r"[^a-z0-9\s]", "", text_a).split())
    set_b = set(re.sub(r"[^a-z0-9\s]", "", text_b).split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# ---------------------------------------------------------------------------
# Model accessor (singleton)
# ---------------------------------------------------------------------------

def _get_nli_pipeline() -> Any:
    """Load and cache the BART-MNLI zero-shot classification pipeline."""
    global _nli_pipeline
    if _nli_pipeline is not None:
        return _nli_pipeline
    if FAST_NLI_MODE:
        logger.info("FAST_NLI_MODE enabled for Synthesizer: using heuristic citation checks")
        return None
    if not _TRANSFORMERS_AVAILABLE or hf_pipeline is None:
        return None
    try:
        logger.info("Loading NLI pipeline for Synthesizer: %s", NLI_MODEL)
        _nli_pipeline = hf_pipeline(
            "zero-shot-classification",
            model=NLI_MODEL,
            device=-1,
        )
        return _nli_pipeline
    except Exception as exc:
        logger.error("Failed to load NLI pipeline for Synthesizer: %s", exc)
        return None
