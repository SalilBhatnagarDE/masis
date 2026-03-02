"""
masis.config.model_routing
==========================
Centralised model assignment and fallback configuration for the MASIS system.

Implements
----------
MF-API-08   : Centralised model routing config with environment variable overrides
MF-SAFE-03  : Model fallback chains -- Primary -> Fallback -> Last Resort per role
MF-SAFE-05  : Per-agent rate limits (also duplicated from thresholds for convenience)

Design
------
All model assignments are environment-variable overridable. This means:
  - In production: set MODEL_RESEARCHER=gpt-4.1-nano in .env to reduce cost
  - In testing:   set MODEL_SUPERVISOR=gpt-4.1-mini to avoid gpt-4.1 costs
  - Per-query:    pass model_overrides in graph.invoke() config for A/B testing

The fallback chain is applied by the Executor when a primary model fails
with an API error (rate limit, unavailable, or circuit breaker OPEN):
    Primary -> Fallback -> Last Resort

If Last Resort also fails, the Executor returns AgentOutput(status="failed")
and the Supervisor Slow Path decides whether to retry, escalate, or stop.

Usage
-----
    from masis.config.model_routing import get_model, get_fallback, MODEL_ROUTING

    # Get configured model for a role
    model_name = get_model("researcher")                    # "gpt-4.1-mini" by default
    model_name = get_model("researcher", "gpt-4.1-nano")   # per-query override

    # Get next fallback when primary fails
    fallback = get_fallback("researcher")  # "gpt-4.1" (second in chain)
    fallback = get_fallback("skeptic_llm") # "gpt-4.1" (second in chain)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Any

# ---------------------------------------------------------------------------
# MODEL_ROUTING  (MF-API-08)
# ---------------------------------------------------------------------------
# Every entry reads from an environment variable first, falling back to the
# listed default. Change any default by setting the env var in .env.
#
# Trade-off notes (from architecture Q14):
#   researcher:         SAFE to swap for cheaper models (more CRAG retries acceptable)
#   ambiguity_detector: SAFE to swap (slight false-positive increase)
#   skeptic_llm:        CAREFUL -- cheaper model may miss contradictions
#   supervisor_plan:    NOT recommended -- bad DAG planning cascades everywhere
#   synthesizer:        NOT recommended -- user-facing quality drops

MODEL_ROUTING: Dict[str, str] = {
    # Supervisor: always gpt-4.1 for strategic DAG planning and slow-path decisions
    "supervisor_plan": os.getenv("MODEL_SUPERVISOR", "gpt-4.1"),
    "supervisor_slow": os.getenv("MODEL_SUPERVISOR", "gpt-4.1"),

    # Researcher: gpt-4.1-mini is cost-effective for factual retrieval tasks
    "researcher": os.getenv("MODEL_RESEARCHER", "gpt-4.1-mini"),

    # Skeptic LLM judge: o3-mini ensures anti-sycophancy (different model family from Synthesizer)
    "skeptic_llm": os.getenv("MODEL_SKEPTIC", "o3-mini"),

    # Synthesizer: gpt-4.1 for user-facing answer quality
    "synthesizer": os.getenv("MODEL_SYNTHESIZER", "gpt-4.1"),

    # Ambiguity detector: gpt-4.1-mini is sufficient for query classification
    "ambiguity_detector": os.getenv("MODEL_AMBIGUITY", "gpt-4.1-mini"),

    # Embedding model for vector search (MF-RES-03)
    "embedder": os.getenv("MODEL_EMBEDDER", "text-embedding-3-small"),

    # Web search does not use an LLM -- Tavily API returns structured results
    # Included here for completeness and potential future use
    "web_search": os.getenv("MODEL_WEB_SEARCH", ""),
}

# ---------------------------------------------------------------------------
# FALLBACK_CHAINS  (MF-SAFE-03)
# ---------------------------------------------------------------------------
# Ordered list of model names per role.
# Index 0 = primary (same as MODEL_ROUTING).
# Index 1 = first fallback.
# Index 2+ = last resort.
#
# The circuit breaker (MF-SAFE-02) moves a role to OPEN state after 4 failures.
# In OPEN state, the Executor immediately skips to the next fallback model.
# After 60 seconds in OPEN, it enters HALF-OPEN and probes with the next model.

FALLBACK_CHAINS: Dict[str, List[str]] = {
    "researcher": [
        os.getenv("MODEL_RESEARCHER", "gpt-4.1-mini"),
        "gpt-4.1",                    # First fallback: more capable
        # No last resort: return partial result with disclaimer (MF-SAFE-07)
    ],
    "supervisor": [
        os.getenv("MODEL_SUPERVISOR", "gpt-4.1"),
        "gpt-4.1-mini",               # First fallback: may degrade planning quality
        # Last resort: escalate to HITL (MF-SUP-11) -- cannot synthesize without supervisor
    ],
    "skeptic_llm": [
        os.getenv("MODEL_SKEPTIC", "o3-mini"),
        "gpt-4.1",                    # First fallback: different family but acceptable
        # Last resort: skip skeptic with WARNING (MF-SAFE-07)
    ],
    "synthesizer": [
        os.getenv("MODEL_SYNTHESIZER", "gpt-4.1"),
        "gpt-4.1-mini",               # First fallback: lower quality but still usable
        # Last resort: return partial with disclaimer (MF-SAFE-07)
    ],
    "ambiguity_detector": [
        os.getenv("MODEL_AMBIGUITY", "gpt-4.1-mini"),
        "gpt-4.1",
    ],
}

# ---------------------------------------------------------------------------
# TOOL_LIMITS  (MF-SAFE-05, MF-EXE-10)
# ---------------------------------------------------------------------------
# Also defined in schemas/thresholds.py -- this copy is the canonical one that
# config/settings.py validates against, and can be overridden via env vars.

TOOL_LIMITS: Dict[str, Dict[str, Any]] = {
    "researcher": {
        "max_parallel": int(os.getenv("RESEARCHER_MAX_PARALLEL", "3")),
        "max_total": int(os.getenv("RESEARCHER_MAX_TOTAL", "8")),
        "timeout_s": int(os.getenv("RESEARCHER_TIMEOUT_S", "30")),
    },
    "web_search": {
        "max_parallel": int(os.getenv("WEB_SEARCH_MAX_PARALLEL", "2")),
        "max_total": int(os.getenv("WEB_SEARCH_MAX_TOTAL", "4")),
        "timeout_s": int(os.getenv("WEB_SEARCH_TIMEOUT_S", "15")),
    },
    "skeptic": {
        "max_parallel": int(os.getenv("SKEPTIC_MAX_PARALLEL", "1")),
        "max_total": int(os.getenv("SKEPTIC_MAX_TOTAL", "3")),
        "timeout_s": int(os.getenv("SKEPTIC_TIMEOUT_S", "45")),
    },
    "synthesizer": {
        "max_parallel": int(os.getenv("SYNTHESIZER_MAX_PARALLEL", "1")),
        "max_total": int(os.getenv("SYNTHESIZER_MAX_TOTAL", "3")),
        "timeout_s": int(os.getenv("SYNTHESIZER_TIMEOUT_S", "60")),
    },
}

# ---------------------------------------------------------------------------
# Model cost table (USD per 1000 tokens, approximate)
# Used by BudgetTracker.add() to estimate cost when agents don't report it
# ---------------------------------------------------------------------------

MODEL_COST_PER_1K: Dict[str, Dict[str, float]] = {
    "gpt-4.1": {
        "input": 0.005,   # $5.00 per 1M input tokens
        "output": 0.015,  # $15.00 per 1M output tokens
    },
    "gpt-4.1-mini": {
        "input": 0.00015,
        "output": 0.0006,
    },
    "o3-mini": {
        "input": 0.0011,
        "output": 0.0044,
    },
    "gpt-4.1-nano": {
        "input": 0.0001,
        "output": 0.0004,
    },
    "text-embedding-3-small": {
        "input": 0.00002,
        "output": 0.0,
    },
}


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------

def get_model(role: str, override: Optional[str] = None) -> str:
    """
    Return the model identifier configured for the given role.

    Priority order:
    1. override parameter (per-query A/B testing or user preference)
    2. MODEL_ROUTING[role] (env-var overridable default)
    3. Empty string if role is not found (caller should handle gracefully)

    Args:
        role:     The functional role key. Valid keys:
                    supervisor_plan, supervisor_slow, researcher, skeptic_llm,
                    synthesizer, ambiguity_detector, embedder, web_search
        override: Optional explicit model name that bypasses the routing table.
                  Useful for per-query model experiments without changing config.

    Returns:
        The model identifier string (e.g. "gpt-4.1-mini").

    Raises:
        KeyError: Never. Returns "" for unknown roles to avoid crashes.

    Examples:
        >>> get_model("researcher")
        'gpt-4.1-mini'
        >>> get_model("researcher", "gpt-4.1-nano")
        'gpt-4.1-nano'
        >>> import os; os.environ["MODEL_RESEARCHER"] = "test-model"
        >>> # Reload module, then:
        >>> get_model("researcher")
        'test-model'
    """
    if override:
        return override
    return MODEL_ROUTING.get(role, "")


def get_fallback(role: str, current_model: Optional[str] = None) -> Optional[str]:
    """
    Return the next fallback model for a role when the current model fails.

    Walks the FALLBACK_CHAINS list for the role. If current_model is provided,
    finds its position and returns the next entry. If current_model is not in
    the chain or is None, returns the second entry (first fallback).

    Args:
        role:          The functional role key (same as MODEL_ROUTING keys).
                       Use "supervisor", not "supervisor_plan" or "supervisor_slow".
        current_model: The model that just failed. If provided, the next model
                       in the chain is returned. If None, returns the first fallback.

    Returns:
        The next model identifier string, or None if there is no fallback left
        (caller should implement last-resort behaviour like returning partial result).

    Examples:
        >>> get_fallback("researcher")
        'gpt-4.1'
        >>> get_fallback("researcher", "gpt-4.1-mini")
        'gpt-4.1'
        >>> get_fallback("researcher", "gpt-4.1")
        None  # No more fallbacks -- return partial result
        >>> get_fallback("unknown_role")
        None
    """
    chain = FALLBACK_CHAINS.get(role, [])
    if not chain:
        return None

    if current_model is None:
        # Return the first fallback (index 1) if chain has at least 2 entries
        if len(chain) >= 2:
            return chain[1]
        return None

    try:
        idx = chain.index(current_model)
        next_idx = idx + 1
        if next_idx < len(chain):
            return chain[next_idx]
        return None
    except ValueError:
        # current_model not found in chain -- return first fallback
        if len(chain) >= 2:
            return chain[1]
        return None


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate USD cost for a model call based on the MODEL_COST_PER_1K table.

    Args:
        model:         Model identifier string.
        input_tokens:  Number of input (prompt) tokens consumed.
        output_tokens: Number of output (completion) tokens generated.

    Returns:
        Estimated cost in USD. Returns 0.0 for unknown models (safe default).

    Example:
        >>> estimate_cost("gpt-4.1", 1000, 500)
        0.0125  # (1000 * 0.005 + 500 * 0.015) / 1000
    """
    costs = MODEL_COST_PER_1K.get(model, {})
    if not costs:
        return 0.0
    input_cost = (input_tokens / 1000.0) * costs.get("input", 0.0)
    output_cost = (output_tokens / 1000.0) * costs.get("output", 0.0)
    return round(input_cost + output_cost, 8)


def get_rate_limit(agent_type: str, key: str) -> Any:
    """
    Return a specific rate limit value for an agent type.

    Args:
        agent_type: One of: researcher, web_search, skeptic, synthesizer.
        key:        One of: max_parallel, max_total, timeout_s.

    Returns:
        The configured value, or None if agent_type or key is not found.

    Example:
        >>> get_rate_limit("researcher", "max_parallel")
        3
        >>> get_rate_limit("skeptic", "timeout_s")
        45
    """
    limits = TOOL_LIMITS.get(agent_type, {})
    return limits.get(key)
