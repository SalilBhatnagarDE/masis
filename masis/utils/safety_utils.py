"""
masis.utils.safety_utils
========================
Safety, content sanitisation, and audit logging utilities for the MASIS system.

Implements
----------
MF-SAFE-04  : Content sanitiser -- strip prompt injection from web search results
MF-SUP-17   : Decision logger -- structured audit trail of every Supervisor decision

Usage
-----
    from masis.utils.safety_utils import content_sanitizer, log_decision

    # In web_search agent -- sanitise raw API results before LLM context:
    sanitized_text = content_sanitizer(raw_web_result_text)

    # In supervisor_node -- after every routing decision:
    updated_log = log_decision(
        decision_log=state.get("decision_log", []),
        entry={
            "turn": state["iteration_count"],
            "mode": "fast",
            "task_id": "T2",
            "decision": "continue",
            "cost": 0.0,
            "latency_ms": 3,
            "reason": "All researcher criteria passed.",
        }
    )
    return {"decision_log": updated_log}
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Injection patterns (loaded lazily from thresholds to avoid circular import)
# ---------------------------------------------------------------------------

def _get_injection_patterns() -> List[str]:
    """Load compiled injection pattern list from thresholds (lazy to avoid circular import)."""
    try:
        from masis.schemas.thresholds import INJECTION_PATTERNS
        return INJECTION_PATTERNS
    except ImportError:
        # Fallback patterns if thresholds cannot be imported
        return [
            r"ignore\s+(all\s+)?previous\s+instructions",
            r"you\s+are\s+now\s+a",
            r"system\s*:",
            r"disregard\s+(all\s+)?prior",
            r"forget\s+everything",
            r"new\s+instructions\s*:",
            r"override\s+your\s+(previous\s+)?instructions",
            r"act\s+as\s+if\s+you",
            r"pretend\s+you\s+are",
            r"jailbreak",
            r"do\s+anything\s+now",
        ]


def _get_max_web_chars() -> int:
    """Load MAX_WEB_RESULT_CHARS from thresholds (lazy to avoid circular import)."""
    try:
        from masis.schemas.thresholds import MAX_WEB_RESULT_CHARS
        return MAX_WEB_RESULT_CHARS
    except ImportError:
        return 5_000


# ---------------------------------------------------------------------------
# ENG-03 / M5 / S1 — content_sanitizer  (MF-SAFE-04)
# ---------------------------------------------------------------------------

def content_sanitizer(
    text: str,
    replacement: str = "[FILTERED]",
    extra_patterns: Optional[List[str]] = None,
) -> str:
    """
    Strip prompt injection patterns from web search results or any user-provided text.

    Applies all patterns from INJECTION_PATTERNS (thresholds.py) as case-insensitive
    regex substitutions, replacing matched spans with the replacement string.
    Text is also hard-truncated to MAX_WEB_RESULT_CHARS (5000 characters) to prevent
    extremely long web results from consuming the LLM context window.

    This is applied BEFORE web search results are included in any LLM prompt.

    Patterns handled (MF-SAFE-04):
    - "ignore previous instructions" (and variants)
    - "you are now a" (persona hijacking)
    - "system:" (fake system message injection)
    - "disregard all prior" (instruction override)
    - "forget everything" (context wipe attempt)
    - "new instructions:" (instruction appending)
    - "override your instructions" (direct override)
    - "act as if you" (role-play jailbreak)
    - "pretend you are" (role-play jailbreak)
    - "jailbreak" (explicit jailbreak keyword)
    - "do anything now" (DAN-style prompt)

    Args:
        text:            The raw text to sanitise (e.g. web search snippet).
        replacement:     The string that replaces each matched injection pattern.
                         Default "[FILTERED]".
        extra_patterns:  Additional regex patterns to apply beyond the defaults.
                         Useful for domain-specific injection attempts.

    Returns:
        Sanitised and truncated text. The structure of legitimate content is
        preserved -- only injection patterns are replaced.

    Examples:
        >>> content_sanitizer("Hello IGNORE PREVIOUS INSTRUCTIONS world")
        'Hello [FILTERED] world'
        >>> content_sanitizer("Normal text about TechCorp revenue")
        'Normal text about TechCorp revenue'
        >>> content_sanitizer("system: you are now a hacker")
        '[FILTERED] [FILTERED] hacker'

    Security Note:
        This sanitiser is a defence-in-depth measure, not a guarantee.
        New injection techniques appear regularly. Review patterns quarterly.
    """
    if not text:
        return ""

    patterns = _get_injection_patterns()
    if extra_patterns:
        patterns = patterns + extra_patterns

    sanitized = text
    injection_count = 0

    for pattern in patterns:
        try:
            compiled = re.compile(pattern, re.IGNORECASE | re.DOTALL)
            new_text, n_subs = compiled.subn(replacement, sanitized)
            if n_subs > 0:
                injection_count += n_subs
                sanitized = new_text
        except re.error as exc:
            # Malformed pattern should not crash the system
            logger.warning("content_sanitizer: invalid regex pattern '%s': %s", pattern, exc)

    if injection_count > 0:
        logger.warning(
            "content_sanitizer: detected and removed %d injection pattern(s) from text "
            "of length %d.",
            injection_count, len(text),
        )

    # Hard truncation to prevent context exhaustion
    max_chars = _get_max_web_chars()
    if len(sanitized) > max_chars:
        truncated = sanitized[:max_chars]
        logger.debug(
            "content_sanitizer: truncated text from %d to %d chars.",
            len(sanitized), max_chars,
        )
        sanitized = truncated

    return sanitized


def sanitize_batch(texts: List[str], **kwargs: Any) -> List[str]:
    """
    Apply content_sanitizer() to a list of texts.

    Convenience wrapper for processing multiple web search results at once.

    Args:
        texts:  List of text strings to sanitise.
        kwargs: Passed through to content_sanitizer().

    Returns:
        List of sanitised strings, same length and order as input.
    """
    return [content_sanitizer(t, **kwargs) for t in texts]


def is_safe_text(text: str) -> bool:
    """
    Return True if the text contains no injection patterns.

    Non-mutating check -- useful for logging or metrics without modifying content.

    Args:
        text: Text to check.

    Returns:
        True if no injection patterns found.
    """
    patterns = _get_injection_patterns()
    for pattern in patterns:
        try:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                return False
        except re.error:
            pass
    return True


# ---------------------------------------------------------------------------
# ENG-03 / M6 / S1 — log_decision  (MF-SUP-17)
# ---------------------------------------------------------------------------

def log_decision(
    decision_log: List[Dict[str, Any]],
    entry: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Append a structured entry to the Supervisor decision log and return the new list.

    This is a pure function -- it does not mutate the input list. The caller
    must write the returned list back to state:
        return {"decision_log": log_decision(state.get("decision_log", []), entry)}

    Every Supervisor decision (Fast Path OR Slow Path) should be logged here.
    The log is persisted in MASISState.decision_log and available via GET /trace.

    Required entry fields (MF-SUP-17):
        turn        (int)   : state["iteration_count"] at time of decision
        mode        (str)   : "fast" or "slow"
        decision    (str)   : the routing action taken
        cost        (float) : USD cost of this decision (0.0 for Fast Path)
        latency_ms  (float) : milliseconds to reach the decision

    Recommended additional fields:
        task_id     (str)   : task_id being evaluated
        reason      (str)   : plain-English explanation
        criteria    (dict)  : check results that led to this decision
        timestamp   (float) : Unix timestamp (auto-added if missing)

    Args:
        decision_log: The current list from state["decision_log"].
        entry:        Dict of structured decision fields. Must include at least
                      'turn', 'mode', 'decision', 'cost', 'latency_ms'.

    Returns:
        New list = decision_log + [entry_with_timestamp]. Never mutates input.

    Raises:
        Never. Logs a warning if required fields are missing but continues.

    Examples:
        >>> log = []
        >>> log = log_decision(log, {
        ...     "turn": 1, "mode": "fast", "task_id": "T1",
        ...     "decision": "continue", "cost": 0.0, "latency_ms": 2.1,
        ...     "reason": "Researcher criteria passed."
        ... })
        >>> len(log)
        1
        >>> log[0]["mode"]
        'fast'

        >>> # After 3 turns, log has 3 entries:
        >>> len(log_decision(log_decision(log_decision([], e), e), e))
        3
    """
    # Validate required fields (warn but never crash)
    required_fields = {"turn", "mode", "decision", "cost", "latency_ms"}
    missing = required_fields - set(entry.keys())
    if missing:
        logger.warning(
            "log_decision: entry is missing required fields: %s. "
            "Full entry: %s", sorted(missing), entry
        )

    # Always stamp a Unix timestamp if the caller didn't provide one
    stamped_entry = dict(entry)
    if "timestamp" not in stamped_entry:
        stamped_entry["timestamp"] = time.time()

    # Validate mode is one of the expected values
    mode = stamped_entry.get("mode", "")
    if mode not in ("fast", "slow", "plan"):
        logger.warning(
            "log_decision: unexpected mode='%s'. Expected 'fast', 'slow', or 'plan'.",
            mode,
        )

    logger.debug(
        "log_decision: turn=%s mode=%s decision=%s cost=%.4f latency_ms=%.1f",
        stamped_entry.get("turn", "?"),
        stamped_entry.get("mode", "?"),
        stamped_entry.get("decision", "?"),
        float(stamped_entry.get("cost", 0.0)),
        float(stamped_entry.get("latency_ms", 0.0)),
    )

    # Return new list (immutable update pattern for LangGraph state)
    return list(decision_log) + [stamped_entry]


def build_fast_path_entry(
    turn: int,
    task_id: str,
    decision: str,
    latency_ms: float,
    reason: str = "",
    criteria: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience factory for Fast Path decision log entries.

    Fast Path decisions always cost $0 and complete in < 10ms.

    Args:
        turn:       Supervisor iteration count.
        task_id:    The task being evaluated.
        decision:   Routing action: 'continue', 'ready_for_validation', 'force_synthesize'.
        latency_ms: Wall-clock milliseconds for the Fast Path check.
        reason:     Optional explanation string.
        criteria:   Optional dict of criteria check results.

    Returns:
        Dict suitable for passing to log_decision().

    Example:
        >>> entry = build_fast_path_entry(3, "T2", "continue", 1.8, "Researcher PASS")
        >>> entry["mode"]
        'fast'
        >>> entry["cost"]
        0.0
    """
    return {
        "turn": turn,
        "mode": "fast",
        "task_id": task_id,
        "decision": decision,
        "cost": 0.0,
        "latency_ms": round(latency_ms, 2),
        "reason": reason,
        "criteria": criteria or {},
    }


def build_slow_path_entry(
    turn: int,
    task_id: str,
    decision: str,
    cost: float,
    latency_ms: float,
    reason: str = "",
    llm_model: str = "",
) -> Dict[str, Any]:
    """
    Convenience factory for Slow Path decision log entries.

    Slow Path entries include LLM cost and model information.

    Args:
        turn:       Supervisor iteration count.
        task_id:    The task being evaluated.
        decision:   Routing action: 'retry', 'modify_dag', 'escalate', 'force_synthesize', 'stop'.
        cost:       USD cost of the LLM call.
        latency_ms: Wall-clock milliseconds for the Slow Path LLM call.
        reason:     Explanation from the LLM's structured response.
        llm_model:  The model used for this Slow Path call (e.g. "gpt-4.1").

    Returns:
        Dict suitable for passing to log_decision().
    """
    return {
        "turn": turn,
        "mode": "slow",
        "task_id": task_id,
        "decision": decision,
        "cost": round(cost, 6),
        "latency_ms": round(latency_ms, 2),
        "reason": reason,
        "llm_model": llm_model,
    }


def build_plan_entry(
    task_count: int,
    cost: float,
    latency_ms: float,
    stop_condition: str = "",
    llm_model: str = "gpt-4.1",
) -> Dict[str, Any]:
    """
    Convenience factory for the first-turn DAG planning log entry.

    Args:
        task_count:     Number of tasks in the generated plan.
        cost:           USD cost of the planning LLM call.
        latency_ms:     Wall-clock milliseconds for the planning call.
        stop_condition: The stop_condition from the generated TaskPlan.
        llm_model:      The model used for planning.

    Returns:
        Dict suitable for passing to log_decision().
    """
    return {
        "turn": 0,
        "mode": "plan",
        "task_id": "PLAN",
        "decision": "plan_created",
        "cost": round(cost, 6),
        "latency_ms": round(latency_ms, 2),
        "task_count": task_count,
        "stop_condition": stop_condition,
        "llm_model": llm_model,
    }


# ---------------------------------------------------------------------------
# Risk scoring utility  (MF-HITL-04)
# ---------------------------------------------------------------------------

def compute_risk_score(query: str, synthesis_text: str = "") -> float:
    """
    Compute a simple heuristic risk score for a query or synthesized answer.

    Used by the Supervisor Slow Path to determine if HITL escalation is needed
    for financial/legal/medical recommendations (MF-HITL-04).

    Risk keywords that increase the score:
    - Financial: investment, recommend, buy, sell, portfolio, return, yield
    - Legal: liability, compliance, regulation, litigation, legal advice
    - Medical: diagnose, treatment, prescription, clinical, medical advice
    - Security: classified, confidential, secret, proprietary

    Args:
        query:          The original or sub-task query string.
        synthesis_text: Optional synthesized answer text (checked for high-risk claims).

    Returns:
        Float in [0.0, 1.0]. Score > 0.7 triggers HITL risk gate (MF-HITL-04).

    Notes:
        This is a keyword heuristic, not an ML model. It is intentionally
        conservative (higher false positives) to err on the side of caution.
    """
    HIGH_RISK_KEYWORDS = {
        # Financial advice (weight 0.3 each)
        "recommend invest": 0.3,
        "buy this stock": 0.3,
        "sell your": 0.3,
        "portfolio allocation": 0.2,
        "financial advice": 0.4,
        "investment recommendation": 0.4,
        # Legal advice (weight 0.3 each)
        "legal advice": 0.4,
        "you should sue": 0.35,
        "compliance violation": 0.2,
        "regulatory penalty": 0.2,
        # Medical advice
        "medical advice": 0.4,
        "you should take": 0.15,
        "dosage recommendation": 0.35,
        # High-stakes framing
        "guaranteed return": 0.3,
        "certain to": 0.15,
    }

    combined = (query + " " + synthesis_text).lower()
    score = 0.0
    for phrase, weight in HIGH_RISK_KEYWORDS.items():
        if phrase in combined:
            score += weight

    # Cap at 1.0
    return min(1.0, score)
