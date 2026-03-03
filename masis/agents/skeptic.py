"""
masis.agents.skeptic
====================
The Skeptic reads all collected evidence and looks for problems — contradictions,
unsupported claims, and logical gaps — before the Synthesizer writes the final answer.

Two-stage pipeline
------------------
Stage 1 — NLI pre-filter (BART-MNLI, local, free, <100ms per claim)
    • Extracts discrete claims from evidence chunks
    • Runs zero-shot entailment on each claim vs. its source text
    • Flags contradictions (score > 0.80) and unsupported claims (neutral, score > 0.70)
    • Detects claims backed by only one source
    • Detects forward-looking statements that aren't grounded in data

Stage 2 — LLM judge (o3-mini, adversarial)
    • Deep critique using the NLI flags as context
    • Forced to find at least 3 issues (prevents sycophantic "looks fine" responses)
    • Attempts to reconcile contradictions where possible
    • Outputs an overall confidence score

Anti-sycophancy: the Skeptic uses o3-mini, a different model family from the Synthesizer
(gpt-4.1). This prevents the critic from simply agreeing with whatever the generator said.

Public API
----------
run_skeptic(task, state)              → AgentOutput
extract_claims(evidence)              → List[Claim]
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
FAST_NLI_MODE: bool = os.getenv("FAST_NLI_MODE", "1").lower() in {"1", "true", "yes"}

# ---------------------------------------------------------------------------
# Phase 0 schema imports with stubs
# ---------------------------------------------------------------------------

try:
    from masis.schemas.models import AgentOutput, EvidenceChunk, SkepticOutput
    from masis.schemas.thresholds import (
        NLI_MODEL,
        SKEPTIC_THRESHOLDS,
    )
except ImportError:
    logger.warning("masis.schemas not found — using stub types for skeptic.py")

    class EvidenceChunk:  # type: ignore[no-redef]
        def __init__(self, **kw: Any) -> None:
            for k, v in kw.items():
                setattr(self, k, v)
        chunk_id: str = ""
        doc_id: str = ""
        text: str = ""

    class SkepticOutput:  # type: ignore[no-redef]
        def __init__(self, **kw: Any) -> None:
            for k, v in kw.items():
                setattr(self, k, v)
        def to_criteria_dict(self) -> dict:
            return {
                "claims_unsupported": getattr(self, "claims_unsupported", 0),
                "claims_contradicted": getattr(self, "claims_contradicted", 0),
                "logical_gaps_count": len(getattr(self, "logical_gaps", [])),
                "overall_confidence": getattr(self, "overall_confidence", 0.0),
            }

    class AgentOutput:  # type: ignore[no-redef]
        def __init__(self, **kw: Any) -> None:
            for k, v in kw.items():
                setattr(self, k, v)

    SKEPTIC_THRESHOLDS: dict = {  # type: ignore[misc]
        "nli_contradiction_score_threshold": 0.80,
        "nli_unsupported_score_threshold": 0.70,
        "min_issues_required": 3,
    }
    NLI_MODEL = "facebook/bart-large-mnli"

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------

try:
    from transformers import pipeline as hf_pipeline
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed — NLI pre-filter will use heuristics")
    hf_pipeline = None  # type: ignore[assignment]

try:
    from langchain_openai import ChatOpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    logger.warning("langchain_openai not installed — LLM judge will use fallback")
    ChatOpenAI = None  # type: ignore[assignment,misc]

try:
    from masis.config.model_routing import get_model
except ImportError:
    import os

    def get_model(role: str, override: Optional[str] = None) -> str:  # type: ignore[misc]
        return override or os.getenv("MODEL_SKEPTIC", "o3-mini")

# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class Claim:
    """A discrete factual claim extracted from evidence (MF-SKE-01).

    Attributes:
        claim_text: The exact claim text.
        source_chunk_id: Chunk from which this claim was extracted.
        source_chunk_text: Full text of the source chunk (for NLI comparison).
        supporting_chunk_ids: All chunk_ids that corroborate this claim.
    """
    claim_text: str
    source_chunk_id: str
    source_chunk_text: str
    supporting_chunk_ids: List[str] = field(default_factory=list)


@dataclass
class NLIResult:
    """Result of an NLI comparison between a claim and its source text."""
    claim: Claim
    label: str          # "entailment" | "contradiction" | "neutral"
    score: float        # Confidence of the label
    is_flagged: bool    # True if this result requires LLM judge attention
    flag_reason: str    # "contradiction" | "unsupported" | ""


# ---------------------------------------------------------------------------
# Forward-looking statement patterns (MF-SKE-07)
# ---------------------------------------------------------------------------

FORWARD_LOOKING_PATTERNS = [
    r"\bexpected\s+to\b",
    r"\bprojected\s+to\b",
    r"\bforecast(ed)?\b",
    r"\bwill\s+likely\b",
    r"\banticipated\s+to\b",
    r"\bplanned\s+to\b",
    r"\baims?\s+to\b",
    r"\btarget(ing)?\b",
    r"\bpotentially\b",
    r"\bin\s+the\s+coming\b",
    r"\bin\s+the\s+future\b",
    r"\bin\s+\d+\s+months\b",
]

# Module-level NLI pipeline cache
_nli_pipeline: Any = None

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

SKEPTIC_PROMPT = """You are the Skeptic Auditor in a multi-agent research system.

Mission:
Stress-test the evidence and surface reliability risks before synthesis.
Be strict, specific, and traceable.

Primary inputs:
- NLI pre-filter flags
- single-source warnings
- forward-looking warnings
- evidence blocks

Audit protocol:
1. Validate contradiction risk:
   - unresolved factual conflicts go to contradictions
   - if conflict is reconcilable, add a reconciliation item
2. Validate logical quality:
   - identify missing links, implied assumptions, and overgeneralization
3. Validate evidence strength:
   - single-source dependence
   - weakly supported claims
   - forward-looking statements presented as present fact
4. Prioritize issues that could change final conclusions.

Formatting rules:
- Keep findings concise and concrete.
- Include claim_id where available in contradiction/reconciliation entries.
- Do not invent evidence not present in inputs.

Few-shot critique patterns (compact):
- contradiction + reconcilable -> keep in reconciliations, not contradictions
- single-source critical metric -> weak_evidence_flags entry
- forecast framed as current fact -> logical_gaps or weak_evidence_flags entry

NLI PRE-FILTER RESULTS:
{nli_flags}

SINGLE-SOURCE WARNINGS:
{single_source_warnings}

FORWARD-LOOKING FLAGS:
{forward_looking_flags}

EVIDENCE TO AUDIT:
{evidence_text}

Return only structured JSON matching schema keys:
contradictions, logical_gaps, weak_evidence_flags, reconciliations, confidence.
"""


async def run_skeptic(task: Any, state: Dict[str, Any]) -> AgentOutput:
    """Execute the two-stage skeptic pipeline and return AgentOutput.

    Args:
        task: TaskNode with .task_id.
        state: Filtered state containing evidence_board and original_query.

    Returns:
        AgentOutput wrapping a SkepticOutput.
    """
    task_id = getattr(task, "task_id", "unknown")
    evidence_board: List[EvidenceChunk] = state.get("evidence_board", [])
    logger.info(
        "Skeptic started: task_id=%s, evidence_chunks=%d", task_id, len(evidence_board)
    )
    start_ts = time.monotonic()

    if not evidence_board:
        logger.warning("Skeptic: empty evidence board — returning zero confidence")
        empty_output = SkepticOutput(  # type: ignore[call-arg]
            task_id=task_id,
            claims_checked=0,
            claims_supported=0,
            claims_unsupported=0,
            claims_contradicted=0,
            overall_confidence=0.0,
            tokens_used=0,
            cost_usd=0.0,
        )
        return AgentOutput(  # type: ignore[call-arg]
            task_id=task_id,
            agent_type="skeptic",
            status="success",
            summary="No evidence available for skeptic analysis.",
            evidence=[],
            criteria_result=empty_output.to_criteria_dict(),
            raw_output=empty_output,
        )

    try:
        skeptic_out = await _run_skeptic_pipeline(task_id, evidence_board)
    except Exception as exc:
        logger.error("Skeptic pipeline failed: %s", exc, exc_info=True)
        return AgentOutput(  # type: ignore[call-arg]
            task_id=task_id,
            agent_type="skeptic",
            status="failed",
            summary=f"Skeptic error: {exc}",
            error_detail=str(exc),
        )

    elapsed = time.monotonic() - start_ts
    logger.info(
        "Skeptic done: task_id=%s claims_checked=%d contradicted=%d confidence=%.2f elapsed=%.2fs",
        task_id,
        skeptic_out.claims_checked,
        skeptic_out.claims_contradicted,
        skeptic_out.overall_confidence,
        elapsed,
    )

    return AgentOutput(  # type: ignore[call-arg]
        task_id=task_id,
        agent_type="skeptic",
        status="success",
        summary=_build_skeptic_summary(skeptic_out),
        evidence=[],   # Skeptic does not add to evidence board
        criteria_result=skeptic_out.to_criteria_dict(),
        tokens_used=skeptic_out.tokens_used,
        cost_usd=skeptic_out.cost_usd,
        raw_output=skeptic_out,
    )


def extract_claims(evidence: List[EvidenceChunk]) -> List[Claim]:
    """Extract discrete factual claims from the evidence board (MF-SKE-01).

    Each claim is tied to a source chunk for NLI verification.
    Simple heuristic: split on punctuation and filter for meaningful sentences.

    Args:
        evidence: List of EvidenceChunk objects from the evidence board.

    Returns:
        List of Claim objects, each linked to a source chunk.
    """
    claims: List[Claim] = []

    for chunk in evidence:
        text = chunk.text.strip()
        if not text:
            continue

        # Split into sentences
        raw_sentences = re.split(r"(?<=[.!?])\s+", text)
        for sentence in raw_sentences:
            sentence = sentence.strip()

            # Filter: must be at least 15 characters and contain a letter
            if len(sentence) < 15 or not re.search(r"[a-zA-Z]", sentence):
                continue

            claim = Claim(
                claim_text=sentence,
                source_chunk_id=chunk.chunk_id,
                source_chunk_text=chunk.text,
                supporting_chunk_ids=[chunk.chunk_id],
            )
            claims.append(claim)

    logger.debug("extract_claims: %d chunks → %d claims", len(evidence), len(claims))
    return claims


# ---------------------------------------------------------------------------
# Pipeline stages (private)
# ---------------------------------------------------------------------------

async def _run_skeptic_pipeline(
    task_id: str, evidence_board: List[EvidenceChunk]
) -> SkepticOutput:
    """Run the full two-stage skeptic pipeline.

    Args:
        task_id: Task identifier for logging.
        evidence_board: All evidence chunks on the whiteboard.

    Returns:
        Populated SkepticOutput.
    """
    # ── Stage 1: Claim extraction ────────────────────────────────────────────
    claims = extract_claims(evidence_board)
    if not claims:
        return _empty_skeptic_output(task_id)

    # ── Stage 1b: Single-source detection (MF-SKE-06) ───────────────────────
    single_source_warnings = _detect_single_source(claims, evidence_board)

    # ── Stage 1c: Forward-looking detection (MF-SKE-07) ─────────────────────
    forward_looking_flags = _detect_forward_looking(claims)

    # ── Stage 2a: NLI pre-filter (MF-SKE-02/03/04) ──────────────────────────
    nli_results = _run_nli_prefilter(claims)

    contradictions_raw = [r for r in nli_results if r.label == "contradiction" and r.is_flagged]
    unsupported_raw = [r for r in nli_results if r.label == "neutral" and r.is_flagged]

    nli_flag_text = _format_nli_flags(contradictions_raw, unsupported_raw)

    # ── Stage 2b: LLM judge (MF-SKE-05) ─────────────────────────────────────
    evidence_text = _format_evidence_for_judge(evidence_board)
    llm_result, tokens_used, cost_usd = await _run_llm_judge(
        evidence_text=evidence_text,
        nli_flags=nli_flag_text,
        single_source_warnings=[w for w in single_source_warnings],
        forward_looking_flags=[f for f in forward_looking_flags],
    )

    # ── Merge results ────────────────────────────────────────────────────────
    supported_count = len([r for r in nli_results if r.label == "entailment"])
    total_claims = len(claims)

    # Apply reconciliations: reduce contradiction count by reconciled items
    reconciliations = _normalize_reconciliations(llm_result.get("reconciliations", []))
    logical_gaps = _normalize_issue_list(llm_result.get("logical_gaps", []))
    weak_evidence_flags = _normalize_issue_list(llm_result.get("weak_evidence_flags", []))
    reconciled_ids = {
        r.get("claim_id", "")
        for r in reconciliations
        if r.get("resolved", False) and r.get("claim_id")
    }
    final_contradictions_count = max(0, len(contradictions_raw) - len(reconciled_ids))

    # Confidence score (MF-SKE-08)
    if total_claims > 0:
        confidence = supported_count / total_claims
    else:
        confidence = 0.0

    # Incorporate LLM judge confidence if available
    # Map categorical labels (returned by some models) to numeric scores
    _CATEGORICAL_CONFIDENCE_MAP = {
        "very_high": 0.92, "very high": 0.92,
        "high": 0.82,
        "medium_high": 0.72, "medium high": 0.72,
        "medium": 0.62,
        "medium_low": 0.48, "medium low": 0.48,
        "low": 0.35,
        "very_low": 0.20, "very low": 0.20,
    }
    llm_confidence_raw = llm_result.get("confidence", confidence)
    try:
        llm_confidence = float(llm_confidence_raw)
    except (TypeError, ValueError):
        normalized = str(llm_confidence_raw).strip().lower()
        if normalized in _CATEGORICAL_CONFIDENCE_MAP:
            llm_confidence = _CATEGORICAL_CONFIDENCE_MAP[normalized]
            logger.info(
                "LLM judge returned categorical confidence=%r -> mapped to %.2f",
                llm_confidence_raw,
                llm_confidence,
            )
        else:
            logger.warning(
                "LLM judge returned non-numeric confidence=%r; using heuristic confidence",
                llm_confidence_raw,
            )
            llm_confidence = confidence
    llm_confidence = max(0.0, min(1.0, llm_confidence))
    overall_confidence = (confidence + llm_confidence) / 2.0

    return SkepticOutput(  # type: ignore[call-arg]
        task_id=task_id,
        claims_checked=total_claims,
        claims_supported=supported_count,
        claims_unsupported=len(unsupported_raw),
        claims_contradicted=final_contradictions_count,
        weak_evidence_flags=weak_evidence_flags,
        logical_gaps=logical_gaps,
        single_source_warnings=single_source_warnings,
        forward_looking_flags=forward_looking_flags,
        reconciliations=reconciliations,
        overall_confidence=round(overall_confidence, 4),
        tokens_used=tokens_used,
        cost_usd=cost_usd,
    )


def _run_nli_prefilter(claims: List[Claim]) -> List[NLIResult]:
    """Run BART-MNLI NLI on each claim against its source text (MF-SKE-02/03/04).

    Args:
        claims: List of Claim objects with source text.

    Returns:
        List of NLIResult objects.
    """
    nli = _get_nli_pipeline()
    contradiction_threshold = SKEPTIC_THRESHOLDS.get("nli_contradiction_score_threshold", 0.80)
    unsupported_threshold = SKEPTIC_THRESHOLDS.get("nli_unsupported_score_threshold", 0.70)

    results: List[NLIResult] = []

    for claim in claims:
        premise = claim.source_chunk_text[:1500]
        hypothesis = claim.claim_text

        if nli is not None:
            try:
                nli_out = nli(
                    hypothesis,
                    candidate_labels=["entailment", "contradiction", "neutral"],
                    hypothesis_template="This text is {}.",
                    multi_label=False,
                )
                label_scores = dict(zip(nli_out["labels"], nli_out["scores"]))
                top_label = nli_out["labels"][0]
                top_score = float(nli_out["scores"][0])
            except Exception as exc:
                logger.debug("NLI error for claim '%.40s': %s", claim.claim_text, exc)
                top_label = "neutral"
                top_score = 0.5
                label_scores = {"entailment": 0.5, "neutral": 0.5, "contradiction": 0.0}
        else:
            # Heuristic fallback: use word overlap as proxy
            overlap = _jaccard_similarity(premise.lower(), hypothesis.lower())
            if overlap > 0.4:
                top_label, top_score = "entailment", overlap
            elif overlap > 0.15:
                top_label, top_score = "neutral", 1.0 - overlap
            else:
                top_label, top_score = "neutral", 0.7
            label_scores = {top_label: top_score}

        # Flag rules (MF-SKE-03/04)
        is_flagged = False
        flag_reason = ""

        if top_label == "contradiction" and top_score > contradiction_threshold:
            is_flagged = True
            flag_reason = "contradiction"
        elif top_label == "neutral" and top_score > unsupported_threshold:
            is_flagged = True
            flag_reason = "unsupported"

        results.append(NLIResult(
            claim=claim,
            label=top_label,
            score=top_score,
            is_flagged=is_flagged,
            flag_reason=flag_reason,
        ))

    contradictions = sum(1 for r in results if r.flag_reason == "contradiction")
    logger.debug(
        "NLI pre-filter: %d claims → %d contradictions, %d unsupported",
        len(claims),
        contradictions,
        sum(1 for r in results if r.flag_reason == "unsupported"),
    )
    return results


def _detect_single_source(
    claims: List[Claim], evidence_board: List[EvidenceChunk]
) -> List[str]:
    """Flag claims backed by only one evidence chunk (MF-SKE-06).

    Cross-reference all claims against the full evidence board to find additional
    supporting chunks via keyword overlap.

    Args:
        claims: Extracted claims.
        evidence_board: All available evidence chunks.

    Returns:
        List of warning strings for single-source claims.
    """
    warnings: List[str] = []

    for claim in claims:
        # Count how many chunks contain at least 2 keywords from the claim
        keywords = _extract_keywords(claim.claim_text)
        supporting_ids: List[str] = []

        for chunk in evidence_board:
            if any(kw in chunk.text.lower() for kw in keywords if len(kw) > 4):
                if chunk.chunk_id not in supporting_ids:
                    supporting_ids.append(chunk.chunk_id)

        if len(supporting_ids) <= 1:
            warnings.append(
                f"Single-source claim (chunk={claim.source_chunk_id}): "
                f"'{claim.claim_text[:80]}...'"
            )
            claim.supporting_chunk_ids = supporting_ids

    logger.debug("Single-source detection: %d warnings", len(warnings))
    return warnings


def _detect_forward_looking(claims: List[Claim]) -> List[str]:
    """Flag claims containing forward-looking language (MF-SKE-07).

    Args:
        claims: Extracted claims.

    Returns:
        List of INFO-level flag strings.
    """
    flags: List[str] = []
    patterns = [re.compile(p, re.IGNORECASE) for p in FORWARD_LOOKING_PATTERNS]

    for claim in claims:
        text = claim.claim_text
        for pattern in patterns:
            if pattern.search(text):
                flags.append(
                    f"Forward-looking statement (chunk={claim.source_chunk_id}): "
                    f"'{text[:100]}'"
                )
                break  # Only flag once per claim

    logger.debug("Forward-looking detection: %d flags", len(flags))
    return flags


async def _run_llm_judge(
    evidence_text: str,
    nli_flags: str,
    single_source_warnings: List[str],
    forward_looking_flags: List[str],
) -> Tuple[Dict[str, Any], int, float]:
    """Run the o3-mini LLM judge for deep adversarial critique (MF-SKE-05).

    Args:
        evidence_text: Formatted evidence to audit.
        nli_flags: Pre-formatted NLI flag descriptions.
        single_source_warnings: Single-source warning strings.
        forward_looking_flags: Forward-looking flag strings.

    Returns:
        Tuple of (result_dict, tokens_used, cost_usd).
    """
    if not _OPENAI_AVAILABLE or ChatOpenAI is None:
        logger.warning("LLM judge unavailable — returning minimal structured result")
        return _minimal_judge_result(single_source_warnings, forward_looking_flags), 0, 0.0

    model_name = get_model("skeptic_llm")
    # Some reasoning models reject temperature; only set it when supported.
    uses_temperature = _supports_temperature(model_name)
    llm_kwargs: Dict[str, Any] = {"model": model_name}
    if uses_temperature:
        llm_kwargs["temperature"] = 0.3
    llm = ChatOpenAI(**llm_kwargs)

    prompt = SKEPTIC_PROMPT.format(
        nli_flags=nli_flags or "None detected by pre-filter.",
        single_source_warnings="\n".join(single_source_warnings) or "None.",
        forward_looking_flags="\n".join(forward_looking_flags) or "None.",
        evidence_text=evidence_text[:4000],
    )

    # Structured output schema as JSON
    json_schema = {
        "type": "object",
        "properties": {
            "contradictions": {"type": "array", "items": {"type": "string"}},
            "logical_gaps": {"type": "array", "items": {"type": "string"}},
            "weak_evidence_flags": {"type": "array", "items": {"type": "string"}},
            "reconciliations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string"},
                        "explanation": {"type": "string"},
                        "resolved": {"type": "boolean"},
                    },
                },
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["contradictions", "logical_gaps", "weak_evidence_flags",
                     "reconciliations", "confidence"],
    }

    try:
        return await _invoke_llm_judge_once(llm, prompt, forward_looking_flags)
    except Exception as exc:
        # If backend rejects temperature, retry once without it before falling back.
        msg = str(exc)
        if uses_temperature and "Unsupported parameter" in msg and "temperature" in msg:
            logger.warning(
                "LLM judge rejected temperature for model '%s'; retrying without temperature",
                model_name,
            )
            try:
                llm_retry = ChatOpenAI(model=model_name)
                return await _invoke_llm_judge_once(
                    llm_retry, prompt, forward_looking_flags
                )
            except Exception as retry_exc:
                logger.warning(
                    "LLM judge retry failed: %s — using fallback result",
                    retry_exc,
                )
                return (
                    _minimal_judge_result(single_source_warnings, forward_looking_flags),
                    0,
                    0.0,
                )

        logger.warning("LLM judge failed: %s — using fallback result", exc)
        return _minimal_judge_result(single_source_warnings, forward_looking_flags), 0, 0.0


def _supports_temperature(model_name: str) -> bool:
    """Return True when it's safe to send a temperature parameter."""
    normalized = (model_name or "").strip().lower()
    return not (
        normalized.startswith("o1")
        or normalized.startswith("o3")
        or normalized.startswith("o4")
    )


async def _invoke_llm_judge_once(
    llm: Any,
    prompt: str,
    forward_looking_flags: List[str],
) -> Tuple[Dict[str, Any], int, float]:
    """Invoke LLM once, parse JSON, and enforce minimum issue count."""
    response = await llm.ainvoke(
        [{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    import json  # noqa: PLC0415

    raw_content = response.content
    result = json.loads(raw_content)
    if not isinstance(result, dict):
        result = {}

    # Normalize untrusted LLM JSON so downstream logic never crashes on type mismatches.
    result["contradictions"] = _normalize_issue_list(result.get("contradictions", []))
    result["logical_gaps"] = _normalize_issue_list(result.get("logical_gaps", []))
    result["weak_evidence_flags"] = _normalize_issue_list(
        result.get("weak_evidence_flags", [])
    )
    result["reconciliations"] = _normalize_reconciliations(
        result.get("reconciliations", [])
    )

    all_issues = (
        result.get("contradictions", [])
        + result.get("logical_gaps", [])
        + result.get("weak_evidence_flags", [])
    )
    if len(all_issues) < 3:
        extra = forward_looking_flags[: 3 - len(all_issues)]
        result.setdefault("weak_evidence_flags", []).extend(extra)

    tokens_used = len(prompt) // 4 + len(raw_content) // 4
    cost_usd = tokens_used * 0.000004
    return result, tokens_used, cost_usd


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_nli_flags(
    contradictions: List[NLIResult],
    unsupported: List[NLIResult],
) -> str:
    """Format NLI results for inclusion in the LLM judge prompt."""
    lines: List[str] = []
    for r in contradictions:
        lines.append(
            f"CONTRADICTION (score={r.score:.2f}): "
            f"Claim='{r.claim.claim_text[:80]}' vs chunk={r.claim.source_chunk_id}"
        )
    for r in unsupported:
        lines.append(
            f"UNSUPPORTED (score={r.score:.2f}): "
            f"Claim='{r.claim.claim_text[:80]}' from chunk={r.claim.source_chunk_id}"
        )
    return "\n".join(lines) if lines else "None detected."


def _format_evidence_for_judge(evidence_board: List[EvidenceChunk]) -> str:
    """Format evidence chunks as numbered blocks for the LLM prompt."""
    lines: List[str] = []
    for i, chunk in enumerate(evidence_board, start=1):
        lines.append(f"[Evidence {i} | chunk_id={chunk.chunk_id}]\n{chunk.text[:1000]}")
    return "\n\n".join(lines)


def _build_skeptic_summary(output: SkepticOutput) -> str:
    """Build a concise summary string for the Supervisor's context view."""
    parts = [
        f"claims_checked={output.claims_checked}",
        f"claims_supported={output.claims_supported}",
        f"claims_contradicted={output.claims_contradicted}",
        f"confidence={output.overall_confidence:.2f}",
    ]
    if output.logical_gaps:
        parts.append(f"logical_gaps={len(output.logical_gaps)}")
    if output.reconciliations:
        parts.append(f"reconciliations={len(output.reconciliations)}")
    return ", ".join(parts)


def _minimal_judge_result(
    single_source_warnings: List[str],
    forward_looking_flags: List[str],
) -> Dict[str, Any]:
    """Return a minimal judge result when the LLM is unavailable."""
    issues: List[str] = []
    issues.extend(single_source_warnings[:2])
    issues.extend(forward_looking_flags[:1])
    # Ensure at least 3 issues
    while len(issues) < 3:
        issues.append("INFO: Unable to perform deep LLM analysis — manual review recommended.")
    return {
        "contradictions": [],
        "logical_gaps": issues[:3],
        "weak_evidence_flags": issues[3:],
        "reconciliations": [],
        "confidence": 0.65,
    }


def _normalize_issue_list(raw: Any) -> List[str]:
    """Coerce LLM issue lists into plain strings expected by SkepticOutput."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if not isinstance(raw, list):
        return [str(raw)]

    normalized: List[str] = []
    for item in raw:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = (
                item.get("description")
                or item.get("text")
                or item.get("issue")
                or item.get("message")
                or item.get("explanation")
                or str(item)
            )
            text = str(text).strip()
        else:
            text = str(item).strip()

        if text:
            normalized.append(text)
    return normalized


def _normalize_reconciliations(raw: Any) -> List[Dict[str, Any]]:
    """Coerce reconciliation payloads into a stable list[dict] format."""
    if raw is None:
        return []

    items = raw if isinstance(raw, list) else [raw]
    normalized: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            claim_id = item.get("claim_id") or item.get("id") or ""
            explanation = (
                item.get("explanation")
                or item.get("reason")
                or item.get("message")
                or ""
            )
            resolved_raw = item.get("resolved", False)
            if isinstance(resolved_raw, bool):
                resolved = resolved_raw
            elif isinstance(resolved_raw, str):
                resolved = resolved_raw.strip().lower() in {"true", "yes", "1", "resolved"}
            else:
                resolved = bool(resolved_raw)

            normalized.append(
                {
                    "claim_id": str(claim_id).strip(),
                    "explanation": str(explanation).strip(),
                    "resolved": resolved,
                }
            )
        elif isinstance(item, str):
            text = item.strip()
            if text:
                normalized.append(
                    {"claim_id": "", "explanation": text, "resolved": False}
                )
    return normalized


def _empty_skeptic_output(task_id: str) -> SkepticOutput:
    """Return a zero-filled SkepticOutput when no evidence is available."""
    return SkepticOutput(  # type: ignore[call-arg]
        task_id=task_id,
        claims_checked=0,
        claims_supported=0,
        claims_unsupported=0,
        claims_contradicted=0,
        weak_evidence_flags=[],
        logical_gaps=[],
        single_source_warnings=[],
        forward_looking_flags=[],
        reconciliations=[],
        overall_confidence=0.0,
        tokens_used=0,
        cost_usd=0.0,
    )


# ---------------------------------------------------------------------------
# Text utilities (shared)
# ---------------------------------------------------------------------------

def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between word sets."""
    set_a = set(re.sub(r"[^a-z0-9\s]", "", text_a).split())
    set_b = set(re.sub(r"[^a-z0-9\s]", "", text_b).split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _extract_keywords(text: str, min_length: int = 4) -> List[str]:
    """Extract content words from text."""
    stop_words = {
        "what", "when", "where", "which", "who", "how", "the", "and",
        "for", "are", "was", "were", "with", "this", "that", "from",
        "has", "have", "had", "its", "their", "about", "than", "more",
    }
    words = re.sub(r"[^a-z0-9\s]", "", text.lower()).split()
    return [w for w in words if len(w) >= min_length and w not in stop_words]


# ---------------------------------------------------------------------------
# Model accessor (singleton)
# ---------------------------------------------------------------------------

def _get_nli_pipeline() -> Any:
    """Load and cache the BART-MNLI zero-shot classification pipeline."""
    global _nli_pipeline
    if _nli_pipeline is not None:
        return _nli_pipeline
    if FAST_NLI_MODE:
        logger.info("FAST_NLI_MODE enabled for Skeptic: using heuristic NLI fallback")
        return None
    if not _TRANSFORMERS_AVAILABLE or hf_pipeline is None:
        return None
    try:
        logger.info("Loading NLI pipeline for Skeptic: %s", NLI_MODEL)
        _nli_pipeline = hf_pipeline(
            "zero-shot-classification",
            model=NLI_MODEL,
            device=-1,
        )
        return _nli_pipeline
    except Exception as exc:
        logger.error("Failed to load NLI pipeline: %s", exc)
        return None
