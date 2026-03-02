"""
masis.schemas
=============
Data models and threshold constants for the MASIS system.

This package provides all Pydantic models, the LangGraph TypedDict state schema,
the evidence_reducer function, and all numeric threshold constants.

Exports -- models
-----------------
TaskNode          -- a single node in the dynamic task DAG (MF-SUP-01, MF-SUP-02, MF-MEM-05)
TaskPlan          -- the full DAG produced by the Supervisor on first turn (MF-SUP-01)
EvidenceChunk     -- a single retrieved and scored text chunk (MF-MEM-06)
Citation          -- chunk_id -> claim_text mapping enforced by SynthesizerOutput (MF-SYN-04)
AgentOutput       -- normalised output wrapper for every agent type (MF-EXE-06)
ResearcherOutput  -- structured output for the Researcher agent (MF-RES-08)
SkepticOutput     -- structured output for the Skeptic agent (MF-SKE-08)
SynthesizerOutput -- structured output for the Synthesizer agent (MF-SYN-03)
BudgetTracker     -- tracks tokens, cost, and API calls per query (MF-MEM-04)
MASISState        -- the full LangGraph TypedDict state schema (MF-MEM-01..08)
evidence_reducer  -- Annotated reducer for evidence_board deduplication (MF-MEM-01)
SupervisorDecision -- structured slow-path decision model (MF-SUP-09)
RetrySpec          -- retry parameters for SupervisorDecision
ModifyDagSpec      -- DAG-modification parameters for SupervisorDecision (MF-SUP-10)
EscalateSpec       -- HITL escalation parameters for SupervisorDecision (MF-SUP-11)

Exports -- thresholds
---------------------
RESEARCHER_THRESHOLDS -- Fast Path accept/reject thresholds for Researcher (MF-RES-05, MF-SUP-07)
SKEPTIC_THRESHOLDS    -- Fast Path accept/reject thresholds for Skeptic (MF-SKE-02..05)
VALIDATOR_THRESHOLDS  -- Hard gates for the Validator node (MF-VAL-05)
SAFETY_LIMITS         -- Loop prevention and hard cap constants (MF-SUP-04..06, MF-SUP-16)
BUDGET_LIMITS         -- Token/cost/time ceiling per query (MF-SAFE-06)
TOOL_LIMITS           -- Per-agent rate limit configuration (MF-SAFE-05)
AMBIGUITY_THRESHOLDS  -- HITL ambiguity detection thresholds (MF-HITL-01)
INJECTION_PATTERNS    -- Regex patterns for content sanitiser (MF-SAFE-04)
NLI_MODEL             -- Model identifier for BART-MNLI
RERANKER_MODEL        -- Model identifier for cross-encoder reranker
EMBEDDER_MODEL        -- Model identifier for sentence-transformers embedder
"""

from masis.schemas.models import (
    AgentOutput,
    BudgetTracker,
    Citation,
    EscalateSpec,
    EvidenceChunk,
    ModifyDagSpec,
    ResearcherOutput,
    RetrySpec,
    SkepticOutput,
    SupervisorDecision,
    SynthesizerOutput,
    TaskNode,
    TaskPlan,
    MASISState,
    evidence_reducer,
)

from masis.schemas.thresholds import (
    AMBIGUITY_THRESHOLDS,
    BUDGET_LIMITS,
    EMBEDDER_MODEL,
    INJECTION_PATTERNS,
    MAX_WEB_RESULT_CHARS,
    NLI_MODEL,
    RERANKER_MODEL,
    RESEARCHER_THRESHOLDS,
    SAFETY_LIMITS,
    SKEPTIC_THRESHOLDS,
    TOOL_LIMITS,
    VALIDATOR_THRESHOLDS,
)

__all__ = [
    # models
    "AgentOutput",
    "BudgetTracker",
    "Citation",
    "EscalateSpec",
    "EvidenceChunk",
    "MASISState",
    "ModifyDagSpec",
    "ResearcherOutput",
    "RetrySpec",
    "SkepticOutput",
    "SupervisorDecision",
    "SynthesizerOutput",
    "TaskNode",
    "TaskPlan",
    "evidence_reducer",
    # thresholds
    "AMBIGUITY_THRESHOLDS",
    "BUDGET_LIMITS",
    "EMBEDDER_MODEL",
    "INJECTION_PATTERNS",
    "MAX_WEB_RESULT_CHARS",
    "NLI_MODEL",
    "RERANKER_MODEL",
    "RESEARCHER_THRESHOLDS",
    "SAFETY_LIMITS",
    "SKEPTIC_THRESHOLDS",
    "TOOL_LIMITS",
    "VALIDATOR_THRESHOLDS",
]
