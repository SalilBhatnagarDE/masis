"""
masis.schemas.thresholds
========================
Centralised threshold and safety-limit constants for the MASIS system (Phase 0, ENG-01 M3).

Every numeric gate in the system is defined here. No magic numbers anywhere else.
When you need to tune a threshold, change it here — all components pick it up automatically.

References
----------
MF-VAL-05   : Validator quality thresholds (faithfulness, citation_accuracy, etc.)
MF-SUP-04   : Fast Path budget check -- remaining <= 0 -> force_synthesize
MF-SUP-05   : Fast Path iteration limit -- iteration_count >= 15 -> force_synthesize
MF-SUP-06   : Fast Path repetition check -- cosine > 0.90 -> force_synthesize
MF-RES-05   : CRAG document grading retry limit
MF-SKE-02   : NLI contradiction score threshold
MF-SKE-03   : NLI unsupported (neutral) score threshold
MF-SKE-05   : LLM judge minimum issues required
MF-SKE-08   : Skeptic minimum confidence
MF-SAFE-01  : 3-layer loop prevention caps
MF-SAFE-05  : Per-agent rate limits (max_parallel, max_total, timeout_s)
MF-SAFE-06  : Budget hard caps (tokens, cost, wall clock)
MF-MEM-04   : Budget tracker defaults
MF-API-08   : Model routing defaults (referenced here as documentation)
"""

from __future__ import annotations

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Researcher acceptance criteria  (MF-RES-05, MF-SUP-07)
# ---------------------------------------------------------------------------

RESEARCHER_THRESHOLDS: Dict[str, Any] = {
    # Fast Path PASS criteria -- ALL must be true for "PASS" verdict
    "min_chunks_after_grading": 2,          # Minimum graded chunks to proceed without retry
    "min_grading_pass_rate": 0.30,          # Minimum fraction of retrieved chunks that are relevant
    "required_self_rag_verdict": "grounded",  # Self-RAG must confirm answer is grounded
    "allow_partial_self_rag": True,           # Demo-speed setting: allow "partial" when other criteria are strong

    # Internal RAG pipeline limits
    "crag_max_retries": 1,                  # Demo-speed setting: keep retrieval rewrite loop tight
    "self_rag_max_retries": 1,              # Demo-speed setting: keep grounding loop tight

    # Retrieval configuration
    "top_k_retrieval": 10,                  # Chunks returned by hybrid search before reranking (MF-RES-03)
    "top_k_after_rerank": 5,               # Chunks kept after cross-encoder reranking (MF-RES-04)

    # RRF fusion weights (MF-RES-03)
    "rrf_k": 60,                            # RRF constant k (standard value)
    "rrf_vector_weight": 0.7,               # Weight for vector (semantic) search leg
    "rrf_bm25_weight": 0.3,                # Weight for BM25 (keyword) search leg

    # Chunking sizes for parent-child strategy (MF-RES-07)
    "chunk_size_parent": 2000,              # Parent chunk size in characters
    "chunk_size_child": 500,               # Child chunk size in characters (for embedding)
    "chunk_overlap": 50,                   # Overlap between consecutive chunks

    # Embedding and reranker model identifiers
    "embedding_model": "text-embedding-3-small",
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}

# ---------------------------------------------------------------------------
# Skeptic acceptance criteria  (MF-SKE-02, MF-SKE-03, MF-SKE-05, MF-SKE-08)
# ---------------------------------------------------------------------------

SKEPTIC_THRESHOLDS: Dict[str, Any] = {
    # Fast Path PASS criteria -- ALL must be true for "PASS" verdict
    "max_unsupported_claims": 2,            # Max claims with no supporting evidence
    "max_contradicted_claims": 0,           # Zero tolerance for direct contradictions
    "max_logical_gaps": 3,                  # Maximum logical gaps before Slow Path
    "min_confidence": 0.65,                 # Minimum overall_confidence score

    # NLI scoring thresholds (Stage 1, BART-MNLI)
    "nli_contradiction_score_threshold": 0.80,   # Score above which CONTRADICTION is flagged (MF-SKE-03)
    "nli_unsupported_score_threshold": 0.70,     # Score above which NEUTRAL is flagged (MF-SKE-04)
    "nli_entailment_score_threshold": 0.70,      # Score above which ENTAILMENT counts as supported

    # LLM judge configuration (Stage 2, o3-mini)
    "min_issues_required": 3,               # LLM judge MUST find at least 3 issues (MF-SKE-05)
    "single_source_warning_enabled": True,  # Flag claims from only one source (MF-SKE-06)
    "forward_looking_detection_enabled": True,   # Flag predictions used as facts (MF-SKE-07)

    # Reconciliation settings
    "reconciliation_enabled": True,         # Attempt LLM reconciliation before flagging (MF-SKE-09)
}

# ---------------------------------------------------------------------------
# Validator quality gates  (MF-VAL-05)
# ---------------------------------------------------------------------------

VALIDATOR_THRESHOLDS: Dict[str, Any] = {
    # Hard gates -- any score below threshold triggers route_validator="revise"
    "min_faithfulness": 0.00,              # Best-effort demo gate (heuristic validation mode)
    "min_citation_accuracy": 0.00,         # Best-effort demo gate (heuristic validation mode)
    "min_answer_relevancy": 0.02,          # Require at least minimal intent alignment
    "min_dag_completeness": 0.50,          # Require at least half the planned scope addressed

    # NLI citation verification threshold
    "citation_entailment_threshold": 0.80, # Minimum entailment_score for a citation to be valid (MF-VAL-02)
}

# ---------------------------------------------------------------------------
# Safety limits  (MF-SUP-04, MF-SUP-05, MF-SUP-06, MF-SUP-16, MF-SAFE-01, MF-VAL-07)
# ---------------------------------------------------------------------------

SAFETY_LIMITS: Dict[str, Any] = {
    # Loop prevention -- Fast Path checks these every Supervisor turn (MF-SAFE-01)
    "MAX_SUPERVISOR_TURNS": 10,            # Demo-speed setting: cap long control loops
    "MAX_WALL_CLOCK_SECONDS": 170,         # Keep graph under external 180s watchdog
    "REPETITION_COSINE_THRESHOLD": 0.90,   # Same-type task cosine similarity threshold (MF-SUP-06)
    "MAX_VALIDATION_ROUNDS": 2,            # Demo-speed setting: reduce revise loops
    "MAX_TASK_RETRIES_PER_NODE": 1,        # Demo-speed setting: avoid repetitive low-value retries

    # Circuit breaker settings (MF-SAFE-02)
    "circuit_breaker_failure_threshold": 4,     # Failures before OPEN state
    "circuit_breaker_recovery_timeout_s": 60,   # Seconds in OPEN before HALF-OPEN probe

    # CRAG and Self-RAG limits (also in RESEARCHER_THRESHOLDS for convenience)
    "crag_max_retries": 1,
    "self_rag_max_retries": 1,
}

# ---------------------------------------------------------------------------
# Budget limits  (MF-SAFE-06, MF-MEM-04)
# ---------------------------------------------------------------------------

BUDGET_LIMITS: Dict[str, Any] = {
    # Hard caps per query session
    "max_tokens": 200_000,                 # Total LLM tokens across all agents
    "max_cost_usd": 0.50,                  # Total cost ceiling in USD
    "max_wall_clock_seconds": 300,         # Same as SAFETY_LIMITS for convenience

    # Soft limits -- approaching these triggers force_synthesize warning
    "soft_token_warning_threshold": 180_000,  # 90% of max_tokens
    "soft_cost_warning_threshold": 0.45,     # 90% of max_cost_usd

    # Context window management
    "supervisor_context_max_tokens": 800,  # Supervisor sees at most 800 tokens of context
    "result_summary_max_chars": 500,       # Agent summary truncated to 500 chars (MF-RES-10)
}

# ---------------------------------------------------------------------------
# Per-agent rate limits  (MF-SAFE-05, MF-EXE-10)
# Also exported at top level as TOOL_LIMITS for backward compatibility
# ---------------------------------------------------------------------------

TOOL_LIMITS: Dict[str, Any] = {
    "researcher": {
        "max_parallel": 3,                 # Maximum concurrent researcher tasks via Send()
        "max_total": 8,                    # Maximum total researcher calls per query session
        "timeout_s": 45,                   # asyncio.wait_for timeout per call
    },
    "web_search": {
        "max_parallel": 2,
        "max_total": 4,
        "timeout_s": 15,
    },
    "skeptic": {
        "max_parallel": 1,                 # Skeptic sees all evidence; should run sequentially
        "max_total": 3,
        "timeout_s": 120,
    },
    "synthesizer": {
        "max_parallel": 1,
        "max_total": 3,
        "timeout_s": 60,
    },
}

# ---------------------------------------------------------------------------
# Ambiguity detection thresholds  (MF-HITL-01)
# ---------------------------------------------------------------------------

AMBIGUITY_THRESHOLDS: Dict[str, Any] = {
    "ambiguity_score_threshold": 0.70,     # Above this -> AMBIGUOUS -> interrupt() with options
    "out_of_scope_score_threshold": 0.80,  # Above this -> OUT_OF_SCOPE -> direct rejection
    "clear_query_threshold": 0.30,         # Below this -> CLEAR -> proceed to Supervisor
}

# ---------------------------------------------------------------------------
# NLI and embedding model identifiers (used by Skeptic, Validator, Researcher)
# ---------------------------------------------------------------------------

NLI_MODEL: str = "valhalla/distilbart-mnli-12-3"
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDER_MODEL: str = "all-MiniLM-L6-v2"  # For repetition detection (is_repetitive)
FAITHFULNESS_MODEL: str = "vectara/hallucination_evaluation_model"  # HHEM-2.1 for Validator

# ---------------------------------------------------------------------------
# Content sanitiser injection patterns  (MF-SAFE-04)
# ---------------------------------------------------------------------------

INJECTION_PATTERNS: List[str] = [
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

MAX_WEB_RESULT_CHARS: int = 5_000  # Hard truncation for web search results before LLM context
