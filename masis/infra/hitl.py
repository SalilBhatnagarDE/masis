"""
masis.infra.hitl
================
Human-in-the-Loop (HITL) integration for MASIS (ENG-13).

All seven MF-HITL micro-features are implemented here:

MF-HITL-01  Pre-supervisor ambiguity gate
    ``ambiguity_detector(state)``  — classifies the query as CLEAR /
    AMBIGUOUS / OUT_OF_SCOPE using gpt-4.1-mini with structured output, then
    either passes through, interrupts with options, or rejects outright.

MF-HITL-02  DAG approval pause
    ``dag_approval_interrupt(plan)``  — after the Supervisor plans the DAG,
    interrupts with the proposed plan so the user can approve / edit / cancel.

MF-HITL-03  Mid-execution evidence pause
    Integrated via the Supervisor Slow Path escalate decision.  When the
    Supervisor returns ``action="escalate"``, the caller should call
    ``mid_execution_interrupt()`` (defined here) to pause with partial results.

MF-HITL-04  Risk gate pause
    ``risk_gate(synthesis)``  — scans for financial/legal advisory language
    and interrupts for human sign-off before returning the answer.

MF-HITL-05  Resume with action handler
    ``handle_resume(resume_value, state)``  — parses the ``Command(resume=...)``
    value and routes to the appropriate continuation function, returning state
    updates for the graph.

MF-HITL-06  Graceful partial result
    ``build_partial_result(state, completed_task_ids)``  — assembles a
    best-effort answer from completed tasks with a missing-aspects disclaimer.

MF-HITL-07  Cancel support
    ``build_cancel_result(state)``  — produces a summary of work done so far
    and evidence found, so the user gets useful output even when cancelling.

LangGraph API
-------------
``interrupt()``   — saves state to checkpointer and pauses the graph.
``Command``       — value returned by the human to resume execution.

Both are imported from ``langgraph.types``; a try/except stub is provided so
the module imports cleanly even without LangGraph installed (useful for
isolated unit tests).

Architecture reference
----------------------
final_architecture_and_flow.md Section 9 (HITL Integration)
engineering_tasks.md ENG-13 M1–M6
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LangGraph HITL primitives — forward-compatible stubs
# ---------------------------------------------------------------------------
try:
    from langgraph.types import interrupt, Command  # noqa: F401
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False

    def interrupt(value: Any) -> Any:  # type: ignore[misc]
        """Stub: raises when called outside a LangGraph execution context."""
        raise RuntimeError(
            "langgraph is not installed or interrupt() called outside a graph. "
            f"Interrupt payload: {value}"
        )

    class Command:  # type: ignore[no-redef]
        """Stub Command class for environments without LangGraph."""

        def __init__(self, *, resume: Any = None, goto: Any = None) -> None:
            self.resume = resume
            self.goto = goto

        def __repr__(self) -> str:
            return f"Command(resume={self.resume!r}, goto={self.goto!r})"


# ---------------------------------------------------------------------------
# Phase 0 forward-compatible imports
# ---------------------------------------------------------------------------
try:
    from masis.config.model_routing import MODEL_ROUTING
except ImportError:
    MODEL_ROUTING: dict[str, str] = {  # type: ignore[assignment]
        "ambiguity_detector": "gpt-4.1-mini",
    }

try:
    from masis.schemas.models import MASISState, TaskNode, TaskPlan  # type: ignore[import]
except ImportError:
    MASISState = Any  # type: ignore[assignment,misc]
    TaskNode = Any  # type: ignore[assignment,misc]
    TaskPlan = Any  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Ambiguity classification enum & Pydantic model
# ---------------------------------------------------------------------------

class AmbiguityLabel(str, Enum):
    """Classification outcomes for the ambiguity detector."""
    CLEAR = "CLEAR"
    AMBIGUOUS = "AMBIGUOUS"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


# We use a plain dataclass-like class to avoid requiring pydantic at import
# time while still offering field access and a clean repr.
class AmbiguityClassification:
    """Result of the ambiguity detection LLM call.

    Attributes
    ----------
    label:
        One of ``CLEAR``, ``AMBIGUOUS``, or ``OUT_OF_SCOPE``.
    confidence:
        Float [0.0, 1.0] — how confident the classifier is in *label*.
    options:
        List of clarifying question strings (empty for CLEAR / OUT_OF_SCOPE).
    suggestion:
        Suggested rephrasing or domain explanation (empty string if not needed).
    reason:
        Short explanation of the classification (for audit logs).
    """

    __slots__ = ("label", "confidence", "options", "suggestion", "reason")

    def __init__(
        self,
        label: AmbiguityLabel | str,
        confidence: float,
        options: list[str] | None = None,
        suggestion: str = "",
        reason: str = "",
    ) -> None:
        self.label = AmbiguityLabel(label) if not isinstance(label, AmbiguityLabel) else label
        self.confidence = float(confidence)
        self.options = list(options) if options else []
        self.suggestion = suggestion
        self.reason = reason

    def __repr__(self) -> str:
        return (
            f"AmbiguityClassification(label={self.label.value!r}, "
            f"confidence={self.confidence:.2f}, options={self.options!r})"
        )


# ---------------------------------------------------------------------------
# Risk-gate keyword patterns (MF-HITL-04)
# ---------------------------------------------------------------------------

_RISK_KEYWORDS: list[re.Pattern[str]] = [
    re.compile(r"\b(invest(ment|ors?|ing)?)\b", re.IGNORECASE),
    re.compile(r"\b(buy|sell|short|long)\b.{0,30}\b(shares?|stocks?|bonds?|options?)\b", re.IGNORECASE),
    re.compile(r"\b(recommend(ation)?s?)\b", re.IGNORECASE),
    re.compile(r"\b(should\s+(you|investors?|clients?)|(you|investors?|clients?)\s+should)\b", re.IGNORECASE),
    re.compile(r"\b(legal\s+(advice|opinion|counsel|liability))\b", re.IGNORECASE),
    re.compile(r"\b(tax\s+advice)\b", re.IGNORECASE),
    re.compile(r"\b(financial\s+(advice|planning|recommendation))\b", re.IGNORECASE),
    re.compile(r"\b(sue|lawsuit|litigation|settlement)\b", re.IGNORECASE),
    re.compile(r"\b(portfolio\s+(allocation|rebalance|strategy))\b", re.IGNORECASE),
]


def _compute_risk_score(text: str) -> tuple[float, list[str]]:
    """Return (risk_score, matched_keywords) for *text*.

    risk_score is in [0.0, 1.0]; each matched unique pattern adds weight.
    """
    matched: list[str] = []
    for pat in _RISK_KEYWORDS:
        m = pat.search(text)
        if m:
            matched.append(m.group(0))
    # Normalise: 0 matches → 0.0; 3+ matches → 1.0
    score = min(1.0, len(matched) / 3.0)
    return score, matched


# ---------------------------------------------------------------------------
# MF-HITL-01: Ambiguity detector
# ---------------------------------------------------------------------------

async def ambiguity_detector(
    query: str,
    model: Optional[str] = None,
    llm_client: Any = None,
) -> AmbiguityClassification:
    """Classify *query* before it reaches the Supervisor.

    Three paths:
    - CLEAR        → returns ``AmbiguityClassification`` with label CLEAR;
                     caller proceeds to Supervisor.
    - AMBIGUOUS    → calls ``interrupt()`` with clarifying options and the
                     classification so the human can pick an interpretation.
    - OUT_OF_SCOPE → returns ``AmbiguityClassification`` with label
                     OUT_OF_SCOPE; no DAG is created; no cost beyond this call.

    Parameters
    ----------
    query:
        The user's raw query string.
    model:
        Override the model from ``MODEL_ROUTING["ambiguity_detector"]``.
        Defaults to ``gpt-4.1-mini``.
    llm_client:
        Optional pre-initialised LLM client (e.g. ``ChatOpenAI`` instance).
        If ``None``, the function builds its own client using the resolved
        model name.  This parameter exists to allow injecting mocks in tests.

    Returns
    -------
    AmbiguityClassification
        CLEAR or OUT_OF_SCOPE classifications are returned directly.
        AMBIGUOUS causes ``interrupt()`` to be called first; when the graph
        resumes the return value is the updated classification enriched with
        the human's clarification.

    Notes
    -----
    ``interrupt()`` never returns in normal LangGraph execution — the graph
    saves state and the function only continues after the human calls
    ``POST /masis/resume`` with ``Command(resume=...)``.  The resumed value
    should contain ``{"clarification": "...", "action": "clarify"}``.
    """
    resolved_model = model or MODEL_ROUTING.get("ambiguity_detector", "gpt-4.1-mini")
    logger.info(
        "ambiguity_detector: classifying query='%s' using model=%s",
        query[:120], resolved_model
    )

    classification = await _classify_ambiguity(query, resolved_model, llm_client)

    logger.info(
        "ambiguity_detector: label=%s confidence=%.2f",
        classification.label.value, classification.confidence
    )

    if classification.label is AmbiguityLabel.OUT_OF_SCOPE:
        logger.info("ambiguity_detector: query is OUT_OF_SCOPE — rejecting.")
        return classification

    if classification.label is AmbiguityLabel.AMBIGUOUS:
        logger.info(
            "ambiguity_detector: AMBIGUOUS — interrupting with options: %s",
            classification.options
        )
        # interrupt() pauses the graph and returns the resume value when the
        # user responds.  The resume value is expected to be a dict with keys
        # "action" (== "clarify") and "clarification" (the chosen option).
        resume_value = interrupt({
            "type": "ambiguity",
            "query": query,
            "options": classification.options,
            "classification": {
                "label": classification.label.value,
                "confidence": classification.confidence,
                "options": classification.options,
                "suggestion": classification.suggestion,
                "reason": classification.reason,
            },
            "message": (
                "Your query is ambiguous.  Please choose one of the options "
                "or rephrase your question."
            ),
        })
        # After resume, enrich the classification with the human's choice.
        if isinstance(resume_value, dict):
            clarification = resume_value.get("clarification", "")
            classification.suggestion = clarification
            classification.label = AmbiguityLabel.CLEAR
            classification.reason = f"Clarified by human: {clarification}"
            logger.info(
                "ambiguity_detector: resumed with clarification='%s'.",
                clarification[:120]
            )

    return classification


async def _classify_ambiguity(
    query: str,
    model: str,
    llm_client: Any,
) -> AmbiguityClassification:
    """Internal LLM call for ambiguity classification.

    Falls back to a heuristic rule-set if the LLM client is unavailable.
    This keeps the module runnable in offline / test environments.
    """
    if llm_client is not None:
        return await _llm_classify(query, model, llm_client)

    # Try to build a real LLM client.
    try:
        from langchain_openai import ChatOpenAI  # type: ignore[import]
        from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore[import]

        llm = ChatOpenAI(model=model, temperature=0.0)

        prompt = _build_ambiguity_prompt(query)
        response = await llm.ainvoke([
            SystemMessage(content=_AMBIGUITY_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        return _parse_ambiguity_response(response.content, query)

    except Exception as exc:
        logger.warning(
            "_classify_ambiguity: LLM unavailable (%s). Using heuristic classifier.",
            exc
        )
        return _heuristic_classify(query)


async def _llm_classify(
    query: str,
    model: str,
    llm_client: Any,
) -> AmbiguityClassification:
    """Use an injected *llm_client* to classify *query*."""
    try:
        from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore[import]
        response = await llm_client.ainvoke([
            SystemMessage(content=_AMBIGUITY_SYSTEM_PROMPT),
            HumanMessage(content=_build_ambiguity_prompt(query)),
        ])
        return _parse_ambiguity_response(response.content, query)
    except Exception as exc:
        logger.warning("_llm_classify failed: %s. Falling back to heuristic.", exc)
        return _heuristic_classify(query)


_AMBIGUITY_SYSTEM_PROMPT = """\
You are the Ambiguity Gate for a multi-agent enterprise research system.

Goal:
Route each query to the right path with minimal friction and high precision.

Labels:
- CLEAR: precise and directly answerable from enterprise research documents.
- AMBIGUOUS: intent is underspecified; needs user clarification.
- OUT_OF_SCOPE: unrelated to enterprise research domain.

Decision policy:
- Prefer CLEAR when intent is specific enough to plan a DAG.
- Use AMBIGUOUS when key dimensions are missing (entity, timeframe, business unit, metric scope).
- Use OUT_OF_SCOPE only for clearly unrelated requests.

If AMBIGUOUS:
- Provide 2-4 concrete clarification options.
- Make options mutually distinct and business-relevant.

Few-shot classification patterns (compact):
- "How is technology division performing?" -> AMBIGUOUS
- "Q3 FY25 revenue trend for Infosys cloud" -> CLEAR
- "Weather tomorrow in Bangalore" -> OUT_OF_SCOPE

Respond ONLY in this exact format:
LABEL: <CLEAR|AMBIGUOUS|OUT_OF_SCOPE>
CONFIDENCE: <0.00-1.00>
OPTIONS: <comma-separated list, or NONE>
SUGGESTION: <brief clarification question, or NONE>
REASON: <one sentence>
"""


def _build_ambiguity_prompt(query: str) -> str:
    return (
        "Classify this user query for orchestration routing.\n"
        "Focus on business research intent clarity and scope completeness.\n\n"
        f'Query: "{query}"'
    )


def _parse_ambiguity_response(
    raw: str, original_query: str
) -> AmbiguityClassification:
    """Parse the structured LLM response into an ``AmbiguityClassification``."""
    lines = {
        k.strip(): v.strip()
        for line in raw.strip().splitlines()
        if ":" in line
        for k, v in [line.split(":", 1)]
    }
    label_str = lines.get("LABEL", "CLEAR").upper()
    try:
        label = AmbiguityLabel(label_str)
    except ValueError:
        label = AmbiguityLabel.CLEAR

    try:
        confidence = float(lines.get("CONFIDENCE", "0.80"))
    except ValueError:
        confidence = 0.80

    options_raw = lines.get("OPTIONS", "NONE")
    options = (
        [o.strip() for o in options_raw.split(",") if o.strip() and o.strip() != "NONE"]
        if options_raw != "NONE"
        else []
    )

    suggestion_raw = lines.get("SUGGESTION", "NONE")
    suggestion = "" if suggestion_raw == "NONE" else suggestion_raw

    reason = lines.get("REASON", "")

    return AmbiguityClassification(
        label=label,
        confidence=confidence,
        options=options,
        suggestion=suggestion,
        reason=reason,
    )


def _heuristic_classify(query: str) -> AmbiguityClassification:
    """Simple keyword-based fallback classifier used when the LLM is down."""
    q_lower = query.lower().strip()

    # Out-of-scope patterns
    oos_patterns = [
        r"\b(weather|temperature|forecast|rain|sunny)\b",
        r"\b(recipe|cook|bake|ingredients)\b",
        r"\b(sports?\s+score|nfl|nba|soccer)\b",
        r"\b(movie|film|actor|celebrity|music|song)\b",
        r"\bhow\s+do\s+i\s+(lose\s+weight|get\s+fit|be\s+happy)\b",
        r"\bwhat'?s?\s+(2\s*\+\s*2|the\s+capital\s+of)\b",
    ]
    for pat in oos_patterns:
        if re.search(pat, q_lower):
            return AmbiguityClassification(
                label=AmbiguityLabel.OUT_OF_SCOPE,
                confidence=0.92,
                reason="Query matches out-of-scope heuristic pattern.",
            )

    # Vague/ambiguous patterns
    ambiguous_triggers = [
        r"^how\s+(is|are)\s+(things?|it|everything)\b",
        r"^what\s+about\b",
        r"^tell\s+me\s+(about|more)\b",
        r"^(update|status)\b$",
    ]
    for pat in ambiguous_triggers:
        if re.search(pat, q_lower):
            return AmbiguityClassification(
                label=AmbiguityLabel.AMBIGUOUS,
                confidence=0.75,
                options=[
                    "Financial performance (revenue, margins)?",
                    "Operational metrics (headcount, productivity)?",
                    "Market position (share, competitors)?",
                    "Strategic initiatives (AI, expansion)?",
                ],
                suggestion="Which business dimension are you asking about?",
                reason="Query matched ambiguity heuristic pattern.",
            )

    # Default: CLEAR
    return AmbiguityClassification(
        label=AmbiguityLabel.CLEAR,
        confidence=0.80,
        reason="No ambiguity patterns detected (heuristic).",
    )


# ---------------------------------------------------------------------------
# MF-HITL-02: DAG approval interrupt
# ---------------------------------------------------------------------------

def dag_approval_interrupt(plan: Any) -> Any:
    """Pause execution for user to review and optionally edit the task DAG.

    Parameters
    ----------
    plan:
        A ``TaskPlan`` or dict representation of the proposed DAG returned by
        the Supervisor's ``plan_dag()`` function.

    Returns
    -------
    Any
        The value returned by ``interrupt()``, which is the human's resume
        response dict.  The caller (Supervisor node) should then apply any
        DAG edits before proceeding.

    Resume response format
    ----------------------
    ::

        {"action": "approve"}
        {"action": "edit", "modifications": [{"task_id": "T2", "query": "..."}]}
        {"action": "cancel"}

    Notes
    -----
    This function calls ``interrupt()`` and therefore only returns after the
    human has responded via ``POST /masis/resume``.
    """
    # Serialise plan to a JSON-safe dict for the interrupt payload.
    if hasattr(plan, "model_dump"):
        plan_dict = plan.model_dump()
    elif hasattr(plan, "__dict__"):
        plan_dict = vars(plan)
    else:
        plan_dict = plan  # already a dict

    payload = {
        "type": "dag_approval",
        "proposed_dag": plan_dict,
        "options": ["approve", "edit", "cancel"],
        "message": (
            "The research plan below has been generated. "
            "Review it and choose: approve to proceed, edit to modify tasks, "
            "or cancel to abort."
        ),
    }
    logger.info(
        "dag_approval_interrupt: pausing for DAG review. tasks=%d",
        len(plan_dict.get("tasks", [])) if isinstance(plan_dict, dict) else -1,
    )
    resume_value = interrupt(payload)
    logger.info(
        "dag_approval_interrupt: resumed with action=%r",
        resume_value.get("action") if isinstance(resume_value, dict) else resume_value,
    )
    return resume_value


def apply_dag_edits(
    plan_dict: dict[str, Any],
    modifications: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply user-supplied modifications to a serialised DAG plan dict.

    Parameters
    ----------
    plan_dict:
        The ``TaskPlan`` serialised as a dict (from ``dag_approval_interrupt``).
    modifications:
        List of task-update dicts, each with at least ``task_id`` and one or
        more fields to overwrite (e.g. ``{"task_id": "T2", "query": "..."}``.

    Returns
    -------
    dict
        Updated ``plan_dict`` with modifications applied.
    """
    tasks = plan_dict.get("tasks", [])
    mod_map = {m["task_id"]: m for m in modifications if "task_id" in m}

    updated_tasks = []
    for task in tasks:
        tid = task.get("task_id") if isinstance(task, dict) else getattr(task, "task_id", None)
        if tid and tid in mod_map:
            if isinstance(task, dict):
                updated = {**task, **{k: v for k, v in mod_map[tid].items() if k != "task_id"}}
            else:
                updated = task
            updated_tasks.append(updated)
        else:
            updated_tasks.append(task)

    return {**plan_dict, "tasks": updated_tasks}


# ---------------------------------------------------------------------------
# MF-HITL-03: Mid-execution evidence pause
# ---------------------------------------------------------------------------

def mid_execution_interrupt(
    state: Any,
    coverage: float,
    missing_aspects: list[str],
) -> Any:
    """Pause mid-execution when evidence coverage is below threshold.

    Called by the Supervisor Slow Path when it determines the available
    evidence is insufficient and wants the user to decide how to proceed.

    Parameters
    ----------
    state:
        Current ``MASISState`` (used to extract completed task summaries).
    coverage:
        Fraction [0.0, 1.0] of DAG tasks that have succeeded.
    missing_aspects:
        Human-readable list of dimensions that lack evidence.

    Returns
    -------
    Any
        The human's resume response dict.

    Resume response format
    ----------------------
    ::

        {"action": "expand_to_web"}   → Supervisor adds web_search tasks
        {"action": "provide_data"}    → Human will supply additional context
        {"action": "accept_partial"}  → Synthesize with what is available
        {"action": "cancel"}          → Abort
    """
    completed_summaries = _extract_completed_summaries(state)

    payload = {
        "type": "evidence_insufficient",
        "coverage": coverage,
        "missing_aspects": missing_aspects,
        "completed_summaries": completed_summaries,
        "options": ["expand_to_web", "provide_data", "accept_partial", "cancel"],
        "message": (
            f"Evidence coverage is {coverage * 100:.0f}%. "
            f"Missing: {', '.join(missing_aspects)}. "
            "How would you like to proceed?"
        ),
    }
    logger.info(
        "mid_execution_interrupt: pausing. coverage=%.2f missing=%s",
        coverage, missing_aspects
    )
    return interrupt(payload)


# ---------------------------------------------------------------------------
# MF-HITL-04: Risk gate
# ---------------------------------------------------------------------------

def risk_gate(
    synthesis_text: str,
    risk_threshold: float = 0.70,
) -> tuple[bool, float, list[str]]:
    """Check whether *synthesis_text* contains financial/legal advisory language.

    If the risk score exceeds *risk_threshold*, this function calls
    ``interrupt()`` to pause and request human sign-off before the answer is
    returned to the end user.

    Parameters
    ----------
    synthesis_text:
        The synthesised answer string to evaluate.
    risk_threshold:
        Minimum score to trigger the gate (default 0.70, per MF-HITL-04).

    Returns
    -------
    (triggered: bool, risk_score: float, matched_keywords: list[str])
        If *triggered* is ``True``, ``interrupt()`` was called and the graph
        is now paused.  Execution only continues after resume.

    Resume response format
    ----------------------
    ::

        {"action": "approve"}              → Return answer as-is
        {"action": "revise"}               → Re-run Synthesizer with stricter prompt
        {"action": "add_disclaimer"}       → Append a regulatory disclaimer
    """
    risk_score, matched = _compute_risk_score(synthesis_text)
    triggered = risk_score >= risk_threshold

    if triggered:
        logger.warning(
            "risk_gate: risk_score=%.2f >= %.2f. Keywords: %s. Interrupting.",
            risk_score, risk_threshold, matched
        )
        resume_value = interrupt({
            "type": "risk_gate",
            "risk_score": risk_score,
            "matched_keywords": matched,
            "answer_preview": synthesis_text[:500] + ("..." if len(synthesis_text) > 500 else ""),
            "options": ["approve", "revise", "add_disclaimer"],
            "message": (
                "The generated answer contains financial or legal advisory "
                "language that requires human review before delivery."
            ),
        })
        logger.info(
            "risk_gate: resumed with action=%r",
            resume_value.get("action") if isinstance(resume_value, dict) else resume_value,
        )
    else:
        logger.debug(
            "risk_gate: risk_score=%.2f < %.2f — no intervention required.",
            risk_score, risk_threshold
        )

    return triggered, risk_score, matched


def add_risk_disclaimer(text: str, keywords: list[str]) -> str:
    """Append a regulatory disclaimer to *text*.

    Called when the user chooses ``"add_disclaimer"`` at the risk gate.
    """
    disclaimer = (
        "\n\n---\n"
        "DISCLAIMER: This response contains references to financial topics "
        "and is provided for informational purposes only. It does not "
        "constitute financial, legal, or investment advice. Please consult a "
        "qualified professional before making any decisions based on this "
        "information."
    )
    return text + disclaimer


# ---------------------------------------------------------------------------
# MF-HITL-05: Resume handler
# ---------------------------------------------------------------------------

_VALID_ACTIONS: frozenset[str] = frozenset({
    # Ambiguity resume
    "clarify",
    # DAG approval resume
    "approve",
    "edit",
    # Risk gate resume
    "add_disclaimer",
    "revise",
    # Mid-execution resume
    "expand_to_web",
    "provide_data",
    "accept_partial",
    # Universal
    "cancel",
})


def handle_resume(
    resume_value: Any,
    state: Any,
) -> dict[str, Any]:
    """Parse and dispatch a human resume response into MASIS state updates.

    This function is called by the graph node that was interrupted (typically
    the Supervisor) after the human calls ``POST /masis/resume`` and the graph
    restarts from its checkpoint.

    Parameters
    ----------
    resume_value:
        The value supplied in ``Command(resume=...)``.  Expected to be a dict
        with at minimum an ``"action"`` key.
    state:
        Current ``MASISState`` at the time of resume.

    Returns
    -------
    dict
        State update dict to be merged into ``MASISState`` by the graph.

    Supported actions
    -----------------
    ``"approve"``           → No state changes; graph continues normally.
    ``"edit"``              → Apply DAG modifications from resume_value["modifications"].
    ``"clarify"``           → Update the query / stop_condition from resume_value["clarification"].
    ``"expand_to_web"``     → Add a web_search task to the existing DAG.
    ``"provide_data"``      → Attach user-supplied context to the evidence board.
    ``"accept_partial"``    → Set force_synthesize flag.
    ``"add_disclaimer"``    → Mark synthesis for disclaimer injection.
    ``"revise"``            → Reset synthesis for a redo pass.
    ``"cancel"``            → Set supervisor_decision="failed" to terminate.
    """
    if not isinstance(resume_value, dict):
        logger.warning(
            "handle_resume: resume_value is not a dict (type=%s). "
            "Treating as 'approve'.", type(resume_value).__name__
        )
        return {}

    action = str(resume_value.get("action", "approve")).lower()

    if action not in _VALID_ACTIONS:
        logger.warning(
            "handle_resume: unknown action '%s'. Treating as 'approve'.", action
        )
        return {}

    logger.info("handle_resume: action='%s'", action)

    if action == "approve":
        return {}

    if action == "cancel":
        return {
            "supervisor_decision": "failed",
            "stop_reason": "Cancelled by human via HITL resume.",
        }

    if action == "clarify":
        clarification: str = resume_value.get("clarification", "")
        original_query: str = _get_state_field(state, "original_query", "")
        new_query = clarification if clarification else original_query
        return {
            "original_query": new_query,
            "stop_reason": None,
        }

    if action == "edit":
        modifications: list[dict[str, Any]] = resume_value.get("modifications", [])
        current_dag = _get_state_field(state, "task_dag", [])
        updated_dag = _apply_task_modifications(current_dag, modifications)
        return {"task_dag": updated_dag}

    if action == "expand_to_web":
        missing = resume_value.get("missing_aspects", [])
        web_query = resume_value.get(
            "web_query",
            "Additional data: " + ", ".join(missing) if missing else "supplemental data"
        )
        return _build_add_web_search_update(state, web_query)

    if action == "provide_data":
        additional_context: str = resume_value.get("data", "")
        if additional_context:
            return _build_user_evidence_update(additional_context)
        return {}

    if action == "accept_partial":
        return {
            "supervisor_decision": "force_synthesize",
            "stop_reason": "Human accepted partial result via HITL.",
        }

    if action == "add_disclaimer":
        return {"hitl_add_disclaimer": True}

    if action == "revise":
        return {
            "synthesis_output": None,
            "supervisor_decision": "continue",
        }

    return {}


# ---------------------------------------------------------------------------
# MF-HITL-06: Graceful partial result
# ---------------------------------------------------------------------------

def build_partial_result(
    state: Any,
    completed_task_ids: list[str],
) -> dict[str, Any]:
    """Assemble a partial answer from completed tasks with a missing disclaimer.

    Parameters
    ----------
    state:
        Current ``MASISState``.
    completed_task_ids:
        Task IDs that finished successfully (status == "done").

    Returns
    -------
    dict
        Partial result dict suitable for setting as ``synthesis_output`` in
        state, including ``"is_partial": True`` and ``"disclaimer"``.
    """
    task_dag = _get_state_field(state, "task_dag", [])
    all_task_ids = [
        (t.get("task_id") if isinstance(t, dict) else getattr(t, "task_id", ""))
        for t in task_dag
    ]
    missing_ids = [tid for tid in all_task_ids if tid not in completed_task_ids]
    coverage = (
        len(completed_task_ids) / len(all_task_ids) if all_task_ids else 0.0
    )

    evidence_board = _get_state_field(state, "evidence_board", [])
    evidence_summary = _summarise_evidence(evidence_board)

    disclaimer = (
        f"NOTE: This response is partial (coverage {coverage * 100:.0f}%). "
        f"Missing task results: {', '.join(missing_ids) or 'none'}. "
        "The answer may be incomplete."
    )

    logger.info(
        "build_partial_result: coverage=%.2f completed=%s missing=%s",
        coverage, completed_task_ids, missing_ids
    )

    return {
        "answer": evidence_summary,
        "citations": [],
        "claims_count": 0,
        "citations_count": 0,
        "all_citations_in_evidence_board": False,
        "is_partial": True,
        "coverage": coverage,
        "tasks_completed": completed_task_ids,
        "missing_task_ids": missing_ids,
        "disclaimer": disclaimer,
    }


# ---------------------------------------------------------------------------
# MF-HITL-07: Cancel result
# ---------------------------------------------------------------------------

def build_cancel_result(state: Any) -> dict[str, Any]:
    """Build a cancellation summary describing work done so far.

    Returns
    -------
    dict
        Structured result dict suitable for returning to the user when the
        query is cancelled.  Includes completed tasks, evidence count, and a
        plain-language summary of what was accomplished.
    """
    task_dag = _get_state_field(state, "task_dag", [])
    evidence_board = _get_state_field(state, "evidence_board", [])
    iteration_count = _get_state_field(state, "iteration_count", 0)

    completed_tasks = []
    pending_tasks = []
    for t in task_dag:
        if isinstance(t, dict):
            tid = t.get("task_id", "?")
            status = t.get("status", "unknown")
        else:
            tid = getattr(t, "task_id", "?")
            status = getattr(t, "status", "unknown")
        if status == "done":
            completed_tasks.append(tid)
        else:
            pending_tasks.append(tid)

    evidence_count = len(evidence_board)
    evidence_summary = _summarise_evidence(evidence_board)

    logger.info(
        "build_cancel_result: completed=%s pending=%s evidence=%d",
        completed_tasks, pending_tasks, evidence_count
    )

    return {
        "status": "cancelled",
        "work_completed": completed_tasks,
        "work_pending": pending_tasks,
        "evidence_found": evidence_count,
        "evidence_summary": evidence_summary,
        "iterations_run": iteration_count,
        "message": (
            f"Query cancelled by user. Completed {len(completed_tasks)} of "
            f"{len(task_dag)} planned tasks. "
            f"Gathered {evidence_count} evidence chunks."
        ),
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_state_field(state: Any, field: str, default: Any) -> Any:
    """Safely retrieve *field* from *state* (TypedDict or plain dict)."""
    if isinstance(state, dict):
        return state.get(field, default)
    return getattr(state, field, default)


def _extract_completed_summaries(state: Any) -> list[dict[str, str]]:
    """Return a list of {task_id, summary} dicts for completed tasks."""
    task_dag = _get_state_field(state, "task_dag", [])
    summaries: list[dict[str, str]] = []
    for t in task_dag:
        if isinstance(t, dict):
            if t.get("status") == "done":
                summaries.append({
                    "task_id": t.get("task_id", ""),
                    "type": t.get("type", ""),
                    "summary": t.get("result_summary", ""),
                })
        else:
            if getattr(t, "status", "") == "done":
                summaries.append({
                    "task_id": getattr(t, "task_id", ""),
                    "type": getattr(t, "type", ""),
                    "summary": getattr(t, "result_summary", ""),
                })
    return summaries


def _summarise_evidence(evidence_board: list[Any]) -> str:
    """Produce a short summary string from the evidence board."""
    if not evidence_board:
        return "No evidence gathered."
    count = len(evidence_board)
    sources = set()
    for chunk in evidence_board:
        doc_id = (
            chunk.get("doc_id") if isinstance(chunk, dict)
            else getattr(chunk, "doc_id", None)
        )
        if doc_id:
            sources.add(doc_id)
    return (
        f"{count} evidence chunk(s) from {len(sources)} unique document(s)."
    )


def _apply_task_modifications(
    task_dag: list[Any],
    modifications: list[dict[str, Any]],
) -> list[Any]:
    """Apply field-level modifications to tasks identified by task_id."""
    mod_map = {m.get("task_id"): m for m in modifications if "task_id" in m}
    updated: list[Any] = []
    for task in task_dag:
        if isinstance(task, dict):
            tid = task.get("task_id")
            if tid and tid in mod_map:
                task = {**task, **{k: v for k, v in mod_map[tid].items() if k != "task_id"}}
        updated.append(task)
    return updated


def _build_add_web_search_update(
    state: Any,
    web_query: str,
) -> dict[str, Any]:
    """Build state-update dict that adds a new web_search task to the DAG."""
    task_dag = _get_state_field(state, "task_dag", [])

    # Assign a unique task ID beyond existing ones.
    existing_ids: list[str] = []
    for t in task_dag:
        existing_ids.append(
            t.get("task_id") if isinstance(t, dict) else getattr(t, "task_id", "T0")
        )
    new_id = f"T{len(existing_ids) + 1}_web"

    new_task: dict[str, Any] = {
        "task_id": new_id,
        "type": "web_search",
        "query": web_query,
        "dependencies": [],
        "parallel_group": 1,
        "acceptance_criteria": ">=1 relevant result, no timeout",
        "status": "pending",
    }

    logger.info(
        "_build_add_web_search_update: adding task %s with query='%s'",
        new_id, web_query[:80]
    )

    return {
        "task_dag": list(task_dag) + [new_task],
        "supervisor_decision": "continue",
        "next_tasks": [new_task],
    }


def _build_user_evidence_update(additional_context: str) -> dict[str, Any]:
    """Wrap user-supplied context as a pseudo-evidence chunk in state."""
    chunk: dict[str, Any] = {
        "doc_id": "user_provided",
        "chunk_id": "user_chunk_001",
        "content": additional_context,
        "retrieval_score": 1.0,
        "source": "Human-provided via HITL",
    }
    return {"evidence_board": [chunk]}
