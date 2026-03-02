"""
masis.schemas.models
====================
All Pydantic data models and the LangGraph TypedDict state schema for MASIS.

Implements
----------
MF-MEM-01  evidence_reducer        -- dedup by (doc_id, chunk_id), keep highest score
MF-MEM-02  original_query          -- immutable field in MASISState
MF-MEM-03  iteration_count         -- global counter in MASISState
MF-MEM-04  BudgetTracker           -- tokens / cost / API-calls tracker
MF-MEM-05  TaskNode.status         -- pending -> running -> done / failed
MF-MEM-06  evidence_board          -- shared whiteboard in MASISState
MF-MEM-07  filtered state views    -- enforced at the node level (documented here)
MF-MEM-08  checkpoint persistence  -- enabled via PostgresSaver (documented here)
MF-SUP-02  TaskNode.acceptance_criteria -- per-task success definition
MF-EXE-06  AgentOutput             -- normalised wrapper for all agent types
MF-RES-08  ResearcherOutput        -- structured fields for Fast Path checking
MF-RES-09  source_diversity        -- unique doc_id count in ResearcherOutput
MF-RES-10  summary truncation      -- 200-token / 500-char cap on summaries
MF-SKE-06  single_source_warnings  -- SkepticOutput field
MF-SKE-07  forward_looking_flags   -- SkepticOutput field
MF-SKE-08  SkepticOutput           -- confidence + claim breakdown for Fast Path
MF-SKE-09  reconciliations         -- SkepticOutput field
MF-SYN-03  SynthesizerOutput       -- Pydantic citation enforcement (min_length=1)
MF-SYN-04  Citation                -- chunk_id -> claim_text mapping
MF-SYN-05  entailment_score        -- post-hoc NLI field in Citation
MF-SYN-06  partial_result          -- force_synthesize mode disclaimer
MF-VAL-06  quality_scores          -- written to state for Supervisor on revise
"""

from __future__ import annotations

import time
from typing import Annotated, Any, Dict, List, Literal, Optional
from typing import TypedDict

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# EvidenceChunk  (MF-MEM-06, MF-MEM-01 key)
# ---------------------------------------------------------------------------

class EvidenceChunk(BaseModel):
    """
    A single retrieved and scored text chunk on the shared evidence board.

    The composite key (doc_id, chunk_id) is used by the evidence_reducer to
    deduplicate parallel researcher writes (MF-MEM-01).

    Supports:
    - Hybrid retrieval (vector + BM25 -> retrieval_score via RRF)
    - Cross-encoder reranking (rerank_score, MF-RES-04)
    - Parent-child chunking (parent_chunk_id links child 500-char to parent 2000-char, MF-RES-07)
    - Metadata-filtered search (metadata dict with year/quarter/department/source_file)
    """

    chunk_id: str = Field(
        ...,
        description="Unique identifier for this chunk within its document.",
    )
    doc_id: str = Field(
        ...,
        description="Unique identifier for the source document.",
    )
    parent_chunk_id: Optional[str] = Field(
        default=None,
        description=(
            "ID of the 2000-char parent chunk if this is a 500-char child. "
            "None for parent chunks and standalone chunks (MF-RES-07)."
        ),
    )
    text: str = Field(
        ...,
        description="Raw text content of this chunk.",
    )
    retrieval_score: float = Field(
        default=0.0,
        ge=0.0,
        description="Fused RRF retrieval score from hybrid search. Higher is better.",
    )
    rerank_score: float = Field(
        default=0.0,
        description=(
            "Cross-encoder score from ms-marco-MiniLM-L-6-v2. "
            "Used by u_shape_order() to order evidence for the Synthesizer."
        ),
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured metadata: year, quarter, department, source_file, page_number.",
    )
    source_label: str = Field(
        default="",
        description="Human-readable label displayed in citation footnotes.",
    )

    class Config:
        frozen = False  # Mutable: reducer may update retrieval_score.

    def dedup_key(self) -> tuple:
        """Composite key for evidence board deduplication."""
        return (self.doc_id, self.chunk_id)

    def __hash__(self) -> int:
        return hash((self.doc_id, self.chunk_id))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EvidenceChunk):
            return NotImplemented
        return self.doc_id == other.doc_id and self.chunk_id == other.chunk_id


# ---------------------------------------------------------------------------
# TaskNode  (MF-SUP-01, MF-SUP-02, MF-MEM-05)
# ---------------------------------------------------------------------------

class TaskNode(BaseModel):
    """
    A single node in the Supervisor's dynamic task DAG.

    Created by the Supervisor LLM during first-turn planning (MF-SUP-01).
    Each node carries its own acceptance_criteria string (MF-SUP-02) which is
    checked by the Fast Path (MF-SUP-07) against the agent's structured output
    without any LLM call.

    Status lifecycle (MF-MEM-05):
        pending  -> running -> done
                            -> failed
    """

    task_id: str = Field(
        ...,
        min_length=1,
        description="Unique task identifier within the DAG, e.g. 'T1', 'T2b'.",
    )
    type: Literal["researcher", "web_search", "skeptic", "synthesizer"] = Field(
        ...,
        description="The agent type responsible for executing this task.",
    )
    query: str = Field(
        ...,
        min_length=1,
        description="The specific sub-question or instruction passed to the agent.",
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="task_ids that must reach status='done' before this task can run.",
    )
    parallel_group: int = Field(
        default=1,
        ge=1,
        description=(
            "Tasks sharing the same parallel_group run concurrently via LangGraph Send(). "
            "Lower group numbers execute before higher ones."
        ),
    )
    acceptance_criteria: str = Field(
        default="",
        description=(
            "Natural-language description of what 'success' looks like for this task. "
            "Written by Supervisor LLM during planning. Parsed by check_agent_criteria()."
        ),
    )
    status: Literal["pending", "running", "done", "failed"] = Field(
        default="pending",
        description="Lifecycle status, updated by the Executor node.",
    )
    result_summary: str = Field(
        default="",
        description="Short summary written back by the Executor after completion.",
    )
    retry_count: int = Field(
        default=0,
        ge=0,
        description="How many times this task has been retried (for Slow Path decisions).",
    )

    @field_validator("task_id")
    @classmethod
    def task_id_non_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("task_id must be a non-empty, non-whitespace string.")
        return stripped

    @field_validator("type")
    @classmethod
    def type_is_valid(cls, v: str) -> str:
        valid = {"researcher", "web_search", "skeptic", "synthesizer"}
        if v not in valid:
            raise ValueError(f"task type '{v}' is not valid. Must be one of: {valid}")
        return v

    def is_ready(self, done_ids: set) -> bool:
        """
        Return True if this task can be dispatched right now.

        Conditions:
          1. status must be 'pending' (not already running, done, or failed).
          2. All dependency task_ids must be in the provided completed-id set.

        Args:
            done_ids: Set of dependency task_ids considered complete for
                scheduling purposes (typically terminal tasks: done/failed).

        Returns:
            True when the task is runnable.

        Examples:
            >>> t = TaskNode(task_id="T2", type="skeptic", query="q", dependencies=["T1"])
            >>> t.is_ready(set())
            False
            >>> t.is_ready({"T1"})
            True
        """
        if self.status != "pending":
            return False
        return all(dep in done_ids for dep in self.dependencies)

    def mark_running(self) -> None:
        """Transition status to 'running'. Called by the Executor before dispatch."""
        self.status = "running"

    def mark_done(self, summary: str = "") -> None:
        """Transition status to 'done'. Called by the Executor after success."""
        self.status = "done"
        if summary:
            self.result_summary = summary[:500]

    def mark_failed(self, summary: str = "") -> None:
        """Transition status to 'failed'. Called by the Executor on error or timeout."""
        self.status = "failed"
        if summary:
            self.result_summary = summary[:500]


# ---------------------------------------------------------------------------
# TaskPlan  (MF-SUP-01)
# ---------------------------------------------------------------------------

class TaskPlan(BaseModel):
    """
    The full dynamic task DAG produced by the Supervisor on the first turn.

    Invariants:
    - At least 1 task (enforced by min_length=1 validator).
    - The final execution group must contain a 'synthesizer' task.
    - stop_condition must be non-empty so Fast Path knows when to route to Validator.
    """

    tasks: List[TaskNode] = Field(
        ...,
        min_length=1,
        description="All TaskNodes forming the research plan, ordered by parallel_group.",
    )
    stop_condition: str = Field(
        default="",
        description=(
            "Plain-English description of when the original query is fully answered. "
            "Used by the Supervisor to evaluate force_synthesize decisions."
        ),
    )

    @field_validator("tasks")
    @classmethod
    def must_have_at_least_one_task(cls, v: List[TaskNode]) -> List[TaskNode]:
        if not v:
            raise ValueError("TaskPlan must contain at least one task.")
        return v

    @model_validator(mode="after")
    def last_group_must_have_synthesizer(self) -> "TaskPlan":
        """Enforce that the final parallel_group contains a synthesizer task."""
        if not self.tasks:
            return self
        max_group = max(t.parallel_group for t in self.tasks)
        last_types = {t.type for t in self.tasks if t.parallel_group == max_group}
        if "synthesizer" not in last_types:
            raise ValueError(
                f"The final execution group (parallel_group={max_group}) must contain a "
                f"'synthesizer' task. Found types: {last_types}. "
                "Ensure the TaskPlan always ends with a synthesizer."
            )
        return self

    def get_task(self, task_id: str) -> Optional[TaskNode]:
        """Return the TaskNode with the given task_id, or None if not found."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

    def all_task_ids(self) -> set:
        """Return all task_ids in the plan as a set."""
        return {t.task_id for t in self.tasks}


# ---------------------------------------------------------------------------
# Citation  (MF-SYN-04, MF-SYN-05)
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    """
    Maps a retrieved evidence chunk to the claim it supports in the synthesis.

    MF-SYN-04: chunk_id -> claim_text mapping enforced in SynthesizerOutput.
    MF-SYN-05: entailment_score populated post-hoc by the NLI verifier.
    MF-VAL-02: Validator verifies chunk_id exists in evidence_board and score >= 0.80.
    """

    chunk_id: str = Field(
        ...,
        min_length=1,
        description="Must match an EvidenceChunk.chunk_id present in the evidence_board.",
    )
    claim_text: str = Field(
        ...,
        min_length=1,
        description="The exact factual claim in the synthesis that this citation supports.",
    )
    entailment_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "NLI entailment probability computed post-hoc by the Validator. "
            "0.0 = not yet checked. Threshold: >= 0.80 for citation to be valid."
        ),
    )


# ---------------------------------------------------------------------------
# AgentOutput  (MF-EXE-06)
# ---------------------------------------------------------------------------

class AgentOutput(BaseModel):
    """
    Normalised output envelope wrapping the result of any agent call.

    The Executor always returns an AgentOutput regardless of which underlying
    agent ran (researcher, web_search, skeptic, synthesizer). This gives the
    Supervisor a uniform interface to read from in the Fast Path.

    MF-EXE-06: Normalised structure with task_id, status, summary, evidence.
    MF-RES-10: summary auto-truncated to 500 chars (Supervisor sees only this).
    MF-SUP-14: Supervisor reads summary only, never full evidence.
    """

    task_id: str = Field(..., description="The task_id of the completed/failed task.")
    agent_type: str = Field(
        ...,
        description="Agent type: researcher | web_search | skeptic | synthesizer.",
    )
    status: Literal["success", "failed", "timeout", "rate_limited"] = Field(
        default="success",
        description="Execution outcome.",
    )
    summary: str = Field(
        default="",
        description=(
            "Up to 500-character summary for the Supervisor context window. "
            "The Supervisor NEVER sees full evidence — only this summary (MF-SUP-14, MF-RES-10)."
        ),
    )
    evidence: List[EvidenceChunk] = Field(
        default_factory=list,
        description="Evidence chunks produced by this agent (empty for non-researcher types).",
    )
    criteria_result: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Structured fields the Fast Path checks against acceptance_criteria. "
            "For researcher: {'chunks_after_grading': 3, 'grading_pass_rate': 0.6, ...}."
        ),
    )
    tokens_used: int = Field(default=0, ge=0, description="LLM tokens consumed.")
    cost_usd: float = Field(default=0.0, ge=0.0, description="Estimated USD cost.")
    error_detail: str = Field(
        default="",
        description="Human-readable error description when status='failed'.",
    )
    raw_output: Optional[Any] = Field(
        default=None,
        description=(
            "The full typed output (ResearcherOutput, SkepticOutput, etc.) "
            "preserved for downstream nodes that need the complete data."
        ),
    )

    @field_validator("summary", mode="before")
    @classmethod
    def truncate_summary(cls, v: Any) -> str:
        """
        Auto-truncate summary to 2000 characters.
        The supervisor only reads the first ~500 chars via its own context builder,
        but downstream consumers (scenario_tests, synthesis_output) need more.
        """
        if not isinstance(v, str):
            v = str(v) if v is not None else ""
        if len(v) > 2000:
            return v[:1997] + "..."
        return v


# ---------------------------------------------------------------------------
# ResearcherOutput  (MF-RES-08, MF-RES-09, MF-RES-10)
# ---------------------------------------------------------------------------

class ResearcherOutput(BaseModel):
    """
    Structured output from the Researcher agent.

    All numeric fields are directly checked by the Supervisor Fast Path (MF-SUP-07)
    against RESEARCHER_THRESHOLDS without an LLM call.

    MF-RES-08: chunks_after_grading, grading_pass_rate, self_rag_verdict
    MF-RES-09: source_diversity -- count of unique doc_ids in evidence
    MF-RES-10: summary -- truncated to ~200 tokens (~500 chars) for Supervisor
    """

    task_id: str = Field(..., description="Matches the TaskNode.task_id this output is for.")
    evidence: List[EvidenceChunk] = Field(
        default_factory=list,
        description="Chunks that survived CRAG document grading.",
    )
    summary: str = Field(
        default="",
        max_length=600,
        description="~200-token summary for the Supervisor context. Never the full evidence.",
    )
    chunks_retrieved: int = Field(
        default=0, ge=0,
        description="Total chunks from hybrid retrieval before grading.",
    )
    chunks_after_grading: int = Field(
        default=0, ge=0,
        description="Chunks surviving CRAG relevance grading. Fast Path: must be >= 2.",
    )
    grading_pass_rate: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="chunks_after_grading / chunks_retrieved. Fast Path: must be >= 0.30.",
    )
    self_rag_verdict: Literal["grounded", "partial", "not_grounded"] = Field(
        default="not_grounded",
        description="Self-RAG hallucination check outcome. Fast Path requires 'grounded'.",
    )
    source_diversity: int = Field(
        default=0, ge=0,
        description="Number of unique doc_ids across graded evidence chunks (MF-RES-09).",
    )
    crag_retries_used: int = Field(
        default=0, ge=0, le=3,
        description="CRAG query-rewrite iterations consumed (max 3 per MF-RES-05).",
    )
    tokens_used: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)

    @field_validator("summary", mode="before")
    @classmethod
    def cap_summary(cls, v: Any) -> str:
        """Enforce 500-char cap (MF-RES-10)."""
        if not isinstance(v, str):
            v = str(v) if v is not None else ""
        if len(v) > 500:
            return v[:497] + "..."
        return v

    def to_criteria_dict(self) -> Dict[str, Any]:
        """
        Return the structured dict used by check_agent_criteria() Fast Path (MF-SUP-07).

        Checks performed by Fast Path:
            chunks_after_grading >= RESEARCHER_THRESHOLDS["min_chunks_after_grading"]  (2)
            grading_pass_rate    >= RESEARCHER_THRESHOLDS["min_grading_pass_rate"]      (0.30)
            self_rag_verdict     == RESEARCHER_THRESHOLDS["required_self_rag_verdict"]  ("grounded")
        """
        return {
            "chunks_retrieved": self.chunks_retrieved,
            "chunks_after_grading": self.chunks_after_grading,
            "grading_pass_rate": self.grading_pass_rate,
            "self_rag_verdict": self.self_rag_verdict,
            "source_diversity": self.source_diversity,
            "crag_retries_used": self.crag_retries_used,
        }

    def compute_source_diversity(self) -> int:
        """Derive and update source_diversity from the current evidence list."""
        self.source_diversity = len({c.doc_id for c in self.evidence})
        return self.source_diversity


# ---------------------------------------------------------------------------
# SkepticOutput  (MF-SKE-06, MF-SKE-07, MF-SKE-08, MF-SKE-09)
# ---------------------------------------------------------------------------

class SkepticOutput(BaseModel):
    """
    Structured output from the Skeptic agent.

    Combines Stage-1 NLI results (BART-MNLI, local, free) and Stage-2 LLM
    judge output (o3-mini) into a single model.

    Fast Path (MF-SUP-07) checks to_criteria_dict() against SKEPTIC_THRESHOLDS:
        claims_unsupported  == 0
        claims_contradicted == 0
        len(logical_gaps)   == 0
        overall_confidence  >= 0.65

    MF-SKE-06: single_source_warnings
    MF-SKE-07: forward_looking_flags
    MF-SKE-08: overall_confidence (0.0-1.0)
    MF-SKE-09: reconciliations -- contradictions the LLM judge resolved
    """

    task_id: str = Field(..., description="Matches the TaskNode.task_id this output is for.")
    claims_checked: int = Field(
        default=0, ge=0,
        description="Total claims evaluated by NLI stage.",
    )
    claims_supported: int = Field(
        default=0, ge=0,
        description="Claims with NLI ENTAILMENT score > 0.70.",
    )
    claims_unsupported: int = Field(
        default=0, ge=0,
        description="Claims with NLI NEUTRAL score > 0.70 (weak/no evidence).",
    )
    claims_contradicted: int = Field(
        default=0, ge=0,
        description="Claims with NLI CONTRADICTION score > 0.80 (directly contradicted).",
    )
    weak_evidence_flags: List[str] = Field(
        default_factory=list,
        description="WARNING flags from the LLM judge (single source, outdated, etc.).",
    )
    logical_gaps: List[str] = Field(
        default_factory=list,
        description="Logical leaps or unsupported inferences identified by the LLM judge.",
    )
    single_source_warnings: List[str] = Field(
        default_factory=list,
        description=(
            "Claims backed by only one source chunk (no corroboration). MF-SKE-06. "
            "Format: 'claim_text (source: chunk_id)'."
        ),
    )
    forward_looking_flags: List[str] = Field(
        default_factory=list,
        description=(
            "Forward-looking statements incorrectly used as established facts (MF-SKE-07). "
            "Format: 'claim_text -- reason this is forward-looking'."
        ),
    )
    reconciliations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Contradictions successfully reconciled by the LLM judge (MF-SKE-09). "
            "Each entry: {'claim_a': str, 'claim_b': str, 'resolution': str, "
            "'chunk_a': str, 'chunk_b': str}."
        ),
    )
    overall_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description=(
            "Aggregate evidence quality confidence score (0.0-1.0). "
            "Computed as claims_supported / claims_checked. Fast Path threshold: >= 0.65."
        ),
    )
    tokens_used: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)

    def to_criteria_dict(self) -> Dict[str, Any]:
        """Return the dict checked by check_agent_criteria() Fast Path."""
        return {
            "claims_checked": self.claims_checked,
            "claims_supported": self.claims_supported,
            "claims_unsupported": self.claims_unsupported,
            "claims_contradicted": self.claims_contradicted,
            "logical_gaps_count": len(self.logical_gaps),
            "single_source_count": len(self.single_source_warnings),
            "forward_looking_count": len(self.forward_looking_flags),
            "overall_confidence": self.overall_confidence,
        }

    def compute_confidence(self) -> float:
        """Recompute overall_confidence from claim counts and update the field in place."""
        if self.claims_checked == 0:
            self.overall_confidence = 0.0
        else:
            self.overall_confidence = round(self.claims_supported / self.claims_checked, 4)
        return self.overall_confidence


# ---------------------------------------------------------------------------
# SynthesizerOutput  (MF-SYN-03, MF-SYN-06, MF-SYN-07)
# ---------------------------------------------------------------------------

class SynthesizerOutput(BaseModel):
    """
    Structured output from the Synthesizer agent.

    MF-SYN-03: citations: list[Citation] = Field(min_length=1) makes an uncited
                answer a Pydantic ValidationError — structurally impossible.
    MF-SYN-04: Each Citation maps chunk_id -> claim_text.
    MF-SYN-05: Citation.entailment_score populated post-hoc by NLI verifier.
    MF-SYN-06: partial_result + missing_aspects when force_synthesize is active.
    MF-SYN-07: When no evidence found, answer says so explicitly (handled in prompt).
    """

    task_id: str = Field(..., description="Matches the TaskNode.task_id this output is for.")
    answer: str = Field(
        ..., min_length=1,
        description="The final synthesized answer to the original_query.",
    )
    citations: List[Citation] = Field(
        ..., min_length=1,
        description=(
            "Every factual claim must be backed by at least one Citation. "
            "min_length=1 enforces this structurally: Pydantic rejects an empty list."
        ),
    )
    claims_count: int = Field(
        default=0, ge=0,
        description="Total factual claims in the answer.",
    )
    citations_count: int = Field(
        default=0, ge=0,
        description="Total citations. Auto-synced to len(citations) by model_validator.",
    )
    all_citations_in_evidence_board: bool = Field(
        default=False,
        description=(
            "True when every Citation.chunk_id was found in evidence_board. "
            "Set by Synthesizer; independently re-verified by Validator (MF-VAL-02)."
        ),
    )
    is_partial: bool = Field(
        default=False,
        description="True when the Supervisor issued force_synthesize (MF-SYN-06).",
    )
    missing_dimensions: List[str] = Field(
        default_factory=list,
        description=(
            "DAG task dimensions NOT addressed in this answer (MF-SYN-06). "
            "Used to generate the 'NOTE: Missing X' disclaimer."
        ),
    )
    tokens_used: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def sync_citations_count(self) -> "SynthesizerOutput":
        """Keep citations_count in sync with the actual citations list length."""
        self.citations_count = len(self.citations)
        return self

    def to_criteria_dict(self) -> Dict[str, Any]:
        """Return dict checked by check_agent_criteria() Fast Path."""
        return {
            "citations_count": self.citations_count,
            "claims_count": self.claims_count,
            "all_citations_in_evidence_board": self.all_citations_in_evidence_board,
        }


# ---------------------------------------------------------------------------
# BudgetTracker  (MF-MEM-04, MF-SAFE-06)
# ---------------------------------------------------------------------------

class BudgetTracker(BaseModel):
    """
    Tracks cumulative token usage, cost, and API calls for one query session.

    MF-MEM-04: Persisted in MASISState.token_budget.
    MF-SAFE-06: Hard caps: 100K tokens, $0.50, 300s wall clock per query.
    MF-SUP-04:  Fast Path reads remaining <= 0 -> force_synthesize.

    The add() method returns a new immutable instance to support LangGraph's
    state-update pattern (nodes return only changed keys; reducers must not mutate).
    """

    total_tokens_used: int = Field(default=0, ge=0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    remaining: int = Field(
        default=100_000,
        description="Tokens remaining out of max_tokens (100,000 per query).",
    )
    api_calls: Dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Calls per agent type. e.g. {'researcher': 3, 'skeptic': 1}. "
            "Used by rate-limit pre-check (MF-EXE-10)."
        ),
    )
    start_time: float = Field(
        default_factory=time.time,
        description="Unix timestamp at query start. Used for wall-clock enforcement (MF-SUP-16).",
    )

    def add(self, tokens: int, cost: float, agent_type: str = "") -> "BudgetTracker":
        """
        Return a new BudgetTracker with the additional usage applied.

        Args:
            tokens:     Number of LLM tokens consumed by the agent call.
            cost:       USD cost of the agent call.
            agent_type: Which agent type consumed these resources (for api_calls tracking).

        Returns:
            A new BudgetTracker instance -- never mutates self.

        Example:
            >>> bt = BudgetTracker()
            >>> bt2 = bt.add(50_000, 0.25, "researcher")
            >>> bt2.remaining
            50000
            >>> bt2.total_cost_usd
            0.25
            >>> bt.remaining  # original unchanged
            100000
        """
        new_calls = dict(self.api_calls)
        if agent_type:
            new_calls[agent_type] = new_calls.get(agent_type, 0) + 1
        new_remaining = max(0, self.remaining - tokens)
        return BudgetTracker(
            total_tokens_used=self.total_tokens_used + tokens,
            total_cost_usd=round(self.total_cost_usd + cost, 6),
            remaining=new_remaining,
            api_calls=new_calls,
            start_time=self.start_time,
        )

    def is_exhausted(self) -> bool:
        """
        Return True when any hard budget cap is hit (MF-SAFE-06).

        Caps checked:
        - Tokens remaining == 0
        - Total cost >= $0.50
        - Wall clock >= 300 seconds
        """
        # Import here to avoid circular dependency at module load time
        from masis.schemas.thresholds import BUDGET_LIMITS, SAFETY_LIMITS

        if self.remaining <= 0:
            return True
        if self.total_cost_usd >= BUDGET_LIMITS["max_cost_usd"]:
            return True
        elapsed = time.time() - self.start_time
        if elapsed >= SAFETY_LIMITS["MAX_WALL_CLOCK_SECONDS"]:
            return True
        return False

    def elapsed_seconds(self) -> float:
        """Return wall-clock seconds elapsed since query start."""
        return time.time() - self.start_time

    def calls_for(self, agent_type: str) -> int:
        """Return the number of completed API calls for the given agent type."""
        return self.api_calls.get(agent_type, 0)

    def wall_clock_seconds(self) -> float:
        """Alias for elapsed_seconds() — retained for backward compatibility."""
        return self.elapsed_seconds()


# ---------------------------------------------------------------------------
# Supervisor slow-path decision models
# ---------------------------------------------------------------------------

class RetrySpec(BaseModel):
    """Parameters for a retry action from the Supervisor Slow Path (MF-SUP-09)."""
    new_query: str = Field(default="", description="Rewritten query for the retry attempt.")
    max_retries: int = Field(default=1, ge=1, description="How many more retries to allow.")


class ModifyDagSpec(BaseModel):
    """Parameters for a DAG-modification action from the Supervisor Slow Path (MF-SUP-10).

    Note: update_deps was removed because OpenAI structured output rejects
    Dict[str, List[str]] schemas. To update dependencies, remove the old task
    and add a new one with updated deps.
    """
    add: List[TaskNode] = Field(default_factory=list, description="New tasks to insert.")
    remove: List[str] = Field(default_factory=list, description="task_ids to remove.")


class EscalateSpec(BaseModel):
    """Parameters for a HITL escalation from the Supervisor Slow Path (MF-SUP-11)."""
    reason: str = Field(..., description="Why escalation is required.")
    hitl_options: List[str] = Field(
        default_factory=list,
        description="Options presented to the human, e.g. ['expand_to_web', 'accept_partial'].",
    )


class SupervisorDecision(BaseModel):
    """
    Structured slow-path decision from the Supervisor LLM (MF-SUP-09, MF-SUP-10).

    action field drives conditional routing in route_supervisor():
        retry           -> re-run the failing task with optional query rewrite
        modify_dag      -> add/remove/update tasks in the DAG
        escalate        -> call interrupt() for human review (HITL)
        force_synthesize -> skip remaining tasks, synthesize with current evidence
        stop            -> unrecoverable failure, route to END
    """

    action: Literal["retry", "modify_dag", "escalate", "force_synthesize", "stop"] = Field(
        ..., description="The routing decision."
    )
    reason: str = Field(default="", description="Plain-English justification for this decision.")
    retry_spec: Optional[RetrySpec] = Field(default=None)
    modify_dag_spec: Optional[ModifyDagSpec] = Field(default=None)
    escalate_spec: Optional[EscalateSpec] = Field(default=None)


# ---------------------------------------------------------------------------
# Evidence deduplication reducer  (MF-MEM-01)
# ---------------------------------------------------------------------------

def evidence_reducer(
    existing: List[EvidenceChunk],
    new: List[EvidenceChunk],
) -> List[EvidenceChunk]:
    """
    LangGraph Annotated reducer for the evidence_board field.

    Deduplicates incoming evidence by the composite key (doc_id, chunk_id).
    When the same chunk is retrieved by two parallel Researcher agents, the copy
    with the higher retrieval_score is kept (best evidence wins).

    This function is registered as the reducer via Annotated in MASISState:
        evidence_board: Annotated[List[EvidenceChunk], evidence_reducer]

    LangGraph calls it automatically when any node writes to evidence_board:
        return {"evidence_board": result.evidence}  # reducer merges, not replaces

    Args:
        existing: The current list of EvidenceChunks already in state.
        new:      The list of EvidenceChunks being written by the current node.

    Returns:
        Deduplicated list. For any (doc_id, chunk_id) collision, the chunk with
        the higher retrieval_score is retained.

    Examples:
        >>> c_low  = EvidenceChunk(doc_id="d1", chunk_id="c1", text="x", retrieval_score=0.8)
        >>> c_high = EvidenceChunk(doc_id="d1", chunk_id="c1", text="x", retrieval_score=0.9)
        >>> result = evidence_reducer([c_low], [c_high])
        >>> result[0].retrieval_score
        0.9
        >>> len(result)
        1
    """
    index: Dict[tuple, EvidenceChunk] = {}
    for chunk in existing:
        key = (chunk.doc_id, chunk.chunk_id)
        index[key] = chunk
    for chunk in new:
        key = (chunk.doc_id, chunk.chunk_id)
        if key not in index or chunk.retrieval_score > index[key].retrieval_score:
            index[key] = chunk
    return list(index.values())


def batch_task_results_reducer(
    existing: List[AgentOutput],
    new: List[AgentOutput],
) -> List[AgentOutput]:
    """
    Merge AgentOutput items produced by parallel executor branches.

    Dedup key: task_id. If the same task_id appears again, keep the newest item.
    """
    merged: Dict[str, AgentOutput] = {}
    for output in existing:
        merged[output.task_id] = output
    for output in new:
        merged[output.task_id] = output
    return list(merged.values())


# ---------------------------------------------------------------------------
# MASISState TypedDict  (MF-MEM-01 through MF-MEM-08, MF-VAL-06)
# ---------------------------------------------------------------------------

class MASISState(TypedDict, total=False):
    """
    The complete LangGraph state schema for one MASIS query session.

    Used as: StateGraph(MASISState)

    All fields use total=False so LangGraph nodes can return partial dicts
    (only the keys they modify). Fields omitted from a node's return are
    carried forward from the previous state.

    Field Groups
    -----------
    Immutable (MF-MEM-02):
        original_query  -- never modified after creation; all agents reference it
        query_id        -- UUID for this query session (for checkpoint and tracing)

    Supervisor-owned:
        task_dag            -- the dynamic research plan (MF-MEM-05)
        iteration_count     -- incremented every Executor turn (MF-MEM-03)
        supervisor_decision -- routing output for conditional edges
        next_tasks          -- tasks to dispatch on the next Executor invocation
        stop_condition      -- natural-language "done" criterion from TaskPlan

    Evidence whiteboard (MF-MEM-06, shared, append-only with dedup):
        evidence_board  -- Annotated with evidence_reducer for safe parallel writes (MF-MEM-01)
        critique_notes  -- SkepticOutput list, readable by Synthesizer
        synthesis_output -- Final Synthesizer answer, readable by Validator
        quality_scores  -- Validator scores (MF-VAL-06), readable by Supervisor on revise

    Per-turn agent output:
        last_task_result -- AgentOutput from the most recent agent call (MF-EXE-06)
        batch_task_results -- AgentOutputs produced in the latest executor batch

    Budget and safety (MF-MEM-04, MF-SAFE-06):
        token_budget     -- cumulative usage tracker
        api_call_counts  -- per-agent type call counter (for rate limiting, MF-EXE-10)
        start_time       -- wall clock timestamp (MF-SUP-16)
        validation_round -- how many validator->supervisor loops (cap: MF-VAL-07)

    Checkpoint/HITL:
        force_synthesize -- flag set by Fast Path budget/iteration checks
        hitl_payload     -- interrupt() payload when graph is paused (MF-HITL-01..07)
        decision_log     -- audit trail of every Supervisor decision (MF-SUP-17)
    """

    # ── Immutable query identity ─────────────────────────────────────────────
    original_query: str
    query_id: str

    # ── Supervisor-owned ─────────────────────────────────────────────────────
    task_dag: List[TaskNode]
    stop_condition: str
    iteration_count: int
    next_tasks: List[TaskNode]
    supervisor_decision: str
    last_task_result: Optional[AgentOutput]
    batch_task_results: Annotated[List[AgentOutput], batch_task_results_reducer]
    parallel_batch_mode: bool

    # ── Evidence whiteboard (parallel-safe via reducer) ──────────────────────
    evidence_board: Annotated[List[EvidenceChunk], evidence_reducer]

    # ── Agent result storage ─────────────────────────────────────────────────
    critique_notes: Optional[SkepticOutput]
    synthesis_output: Optional[SynthesizerOutput]

    # ── Quality & validation ─────────────────────────────────────────────────
    quality_scores: Dict[str, float]        # MF-VAL-06: faithfulness, citation_accuracy, etc.
    validation_pass: bool                    # Last validator gate result (True routes to END)
    validation_round: int                   # MF-VAL-07: capped at MAX_VALIDATION_ROUNDS (2)

    # ── Budget & rate limiting ───────────────────────────────────────────────
    token_budget: BudgetTracker
    api_call_counts: Dict[str, int]

    # ── Safety & control ─────────────────────────────────────────────────────
    start_time: float
    force_synthesize: bool
    enable_ambiguity_hitl: bool
    hitl_payload: Optional[Dict[str, Any]]

    # ── Audit trail ──────────────────────────────────────────────────────────
    decision_log: List[Dict[str, Any]]      # MF-SUP-17: structured log of every decision
