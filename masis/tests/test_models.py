"""
test_models.py
==============
Unit tests for masis.schemas.models

Tests every Pydantic model, the evidence_reducer function, and BudgetTracker
according to the acceptance criteria defined in ENG-01.

Run:
    pytest masis/tests/test_models.py -v
"""

from __future__ import annotations

import time

import pytest
from pydantic import ValidationError

from masis.schemas.models import (
    AgentOutput,
    BudgetTracker,
    Citation,
    EvidenceChunk,
    MASISState,
    ResearcherOutput,
    SkepticOutput,
    SynthesizerOutput,
    TaskNode,
    TaskPlan,
    evidence_reducer,
)


# =============================================================================
# Helpers
# =============================================================================

def make_chunk(doc_id: str = "d1", chunk_id: str = "c1", score: float = 0.5) -> EvidenceChunk:
    return EvidenceChunk(
        doc_id=doc_id,
        chunk_id=chunk_id,
        text="Sample evidence text.",
        retrieval_score=score,
        rerank_score=score,
    )


def make_task(
    task_id: str = "T1",
    task_type: str = "researcher",
    deps: list | None = None,
    group: int = 1,
    status: str = "pending",
) -> TaskNode:
    return TaskNode(
        task_id=task_id,
        type=task_type,
        query=f"Test query for {task_id}",
        dependencies=deps or [],
        parallel_group=group,
        acceptance_criteria=f"criteria for {task_id}",
        status=status,
    )


def make_citation(chunk_id: str = "c1", claim: str = "Test claim.") -> Citation:
    return Citation(chunk_id=chunk_id, claim_text=claim)


# =============================================================================
# EvidenceChunk tests
# =============================================================================

class TestEvidenceChunk:
    def test_basic_instantiation(self):
        chunk = EvidenceChunk(doc_id="d1", chunk_id="c1", text="Hello world.")
        assert chunk.doc_id == "d1"
        assert chunk.chunk_id == "c1"
        assert chunk.text == "Hello world."
        assert chunk.retrieval_score == 0.0
        assert chunk.rerank_score == 0.0
        assert chunk.parent_chunk_id is None
        assert chunk.metadata == {}

    def test_equality_by_doc_and_chunk_id(self):
        c1 = EvidenceChunk(doc_id="d1", chunk_id="c1", text="Text A", retrieval_score=0.8)
        c2 = EvidenceChunk(doc_id="d1", chunk_id="c1", text="Text B", retrieval_score=0.9)
        assert c1 == c2, "Two chunks with same (doc_id, chunk_id) must be equal."

    def test_inequality_different_chunk_id(self):
        c1 = make_chunk(chunk_id="c1")
        c2 = make_chunk(chunk_id="c2")
        assert c1 != c2

    def test_hash_consistency(self):
        c1 = make_chunk(doc_id="d1", chunk_id="c1")
        c2 = make_chunk(doc_id="d1", chunk_id="c1")
        assert hash(c1) == hash(c2)

    def test_metadata_stores_arbitrary_dict(self):
        chunk = EvidenceChunk(
            doc_id="d1", chunk_id="c1", text="x",
            metadata={"year": 2026, "quarter": "Q3", "department": "cloud"},
        )
        assert chunk.metadata["year"] == 2026
        assert chunk.metadata["quarter"] == "Q3"

    def test_parent_chunk_id_optional(self):
        child = EvidenceChunk(doc_id="d1", chunk_id="child_1", text="x", parent_chunk_id="parent_1")
        parent = EvidenceChunk(doc_id="d1", chunk_id="parent_1", text="x")
        assert child.parent_chunk_id == "parent_1"
        assert parent.parent_chunk_id is None

    def test_dedup_key(self):
        chunk = make_chunk(doc_id="doc_abc", chunk_id="chunk_42")
        assert chunk.dedup_key() == ("doc_abc", "chunk_42")

    def test_json_serialization(self):
        chunk = make_chunk()
        d = chunk.model_dump()
        assert "doc_id" in d
        assert "chunk_id" in d
        assert "retrieval_score" in d


# =============================================================================
# TaskNode tests
# =============================================================================

class TestTaskNode:
    def test_basic_instantiation(self):
        task = TaskNode(
            task_id="T1",
            type="researcher",
            query="What was Q3 revenue?",
            dependencies=[],
            parallel_group=1,
            acceptance_criteria=">=2 chunks, pass_rate>=0.30",
            status="pending",
        )
        assert task.task_id == "T1"
        assert task.type == "researcher"
        assert task.status == "pending"

    def test_empty_task_id_raises(self):
        with pytest.raises(ValidationError):
            TaskNode(task_id="", type="researcher", query="q")

    def test_whitespace_task_id_raises(self):
        with pytest.raises(ValidationError):
            TaskNode(task_id="   ", type="researcher", query="q")

    def test_invalid_type_raises(self):
        with pytest.raises(ValidationError):
            TaskNode(task_id="T1", type="analyzer", query="q")

    def test_all_valid_types(self):
        for t in ("researcher", "web_search", "skeptic", "synthesizer"):
            node = TaskNode(task_id="T1", type=t, query="q")
            assert node.type == t

    def test_is_ready_no_deps(self):
        task = make_task(deps=[])
        assert task.is_ready(set()) is True

    def test_is_ready_deps_satisfied(self):
        task = make_task(task_id="T2", deps=["T1"])
        assert task.is_ready({"T1"}) is True

    def test_is_ready_deps_not_satisfied(self):
        task = make_task(task_id="T2", deps=["T1"])
        assert task.is_ready(set()) is False
        assert task.is_ready({"T3"}) is False

    def test_is_ready_multiple_deps_partial(self):
        task = make_task(task_id="T3", deps=["T1", "T2"])
        assert task.is_ready({"T1"}) is False  # T2 still missing
        assert task.is_ready({"T1", "T2"}) is True

    def test_is_ready_returns_false_when_not_pending(self):
        task = make_task(status="done")
        assert task.is_ready(set()) is False
        task.status = "running"
        assert task.is_ready(set()) is False
        task.status = "failed"
        assert task.is_ready(set()) is False

    def test_mark_done(self):
        task = make_task()
        task.mark_done(summary="Success")
        assert task.status == "done"
        assert "Success" in task.result_summary

    def test_mark_failed(self):
        task = make_task()
        task.mark_failed(summary="Timed out")
        assert task.status == "failed"

    def test_mark_running(self):
        task = make_task()
        task.mark_running()
        assert task.status == "running"

    def test_default_values(self):
        task = TaskNode(task_id="T1", type="researcher", query="q")
        assert task.dependencies == []
        assert task.parallel_group == 1
        assert task.status == "pending"
        assert task.acceptance_criteria == ""
        assert task.retry_count == 0


# =============================================================================
# TaskPlan tests
# =============================================================================

class TestTaskPlan:
    def test_basic_instantiation(self):
        t1 = make_task(task_id="T1", task_type="researcher")
        t2 = make_task(task_id="T2", task_type="synthesizer", group=2)
        plan = TaskPlan(tasks=[t1, t2], stop_condition="Done when revenue answered.")
        assert len(plan.tasks) == 2

    def test_empty_tasks_raises(self):
        with pytest.raises(ValidationError):
            TaskPlan(tasks=[], stop_condition="x")

    def test_last_group_must_be_synthesizer(self):
        # Last group is group 2 with only a researcher -- should raise
        t1 = make_task(task_id="T1", task_type="researcher", group=1)
        t2 = make_task(task_id="T2", task_type="researcher", group=2)
        with pytest.raises(ValidationError, match="synthesizer"):
            TaskPlan(tasks=[t1, t2], stop_condition="x")

    def test_last_group_synthesizer_passes(self):
        t1 = make_task(task_id="T1", task_type="researcher", group=1)
        t2 = make_task(task_id="T2", task_type="skeptic", group=2)
        t3 = make_task(task_id="T3", task_type="synthesizer", group=3)
        plan = TaskPlan(tasks=[t1, t2, t3], stop_condition="x")
        assert plan is not None

    def test_get_task_found(self):
        t1 = make_task(task_id="T1", task_type="researcher")
        t2 = make_task(task_id="T2", task_type="synthesizer", group=2)
        plan = TaskPlan(tasks=[t1, t2])
        assert plan.get_task("T1") is t1

    def test_get_task_not_found_returns_none(self):
        t1 = make_task(task_id="T1", task_type="synthesizer")
        plan = TaskPlan(tasks=[t1])
        assert plan.get_task("T99") is None


# =============================================================================
# Citation tests
# =============================================================================

class TestCitation:
    def test_basic_instantiation(self):
        c = Citation(chunk_id="c12", claim_text="Revenue grew 12%.")
        assert c.chunk_id == "c12"
        assert c.claim_text == "Revenue grew 12%."
        assert c.entailment_score == 0.0

    def test_entailment_score_range(self):
        c = Citation(chunk_id="c1", claim_text="x", entailment_score=0.94)
        assert 0.0 <= c.entailment_score <= 1.0

    def test_empty_chunk_id_raises(self):
        with pytest.raises(ValidationError):
            Citation(chunk_id="", claim_text="x")

    def test_empty_claim_text_raises(self):
        with pytest.raises(ValidationError):
            Citation(chunk_id="c1", claim_text="")


# =============================================================================
# AgentOutput tests
# =============================================================================

class TestAgentOutput:
    def test_basic_instantiation(self):
        out = AgentOutput(task_id="T1", agent_type="researcher", status="success")
        assert out.task_id == "T1"
        assert out.status == "success"

    def test_summary_auto_truncated_at_500(self):
        long_summary = "X" * 600
        out = AgentOutput(task_id="T1", agent_type="researcher", status="success",
                          summary=long_summary)
        assert len(out.summary) <= 500
        assert out.summary.endswith("...")

    def test_summary_short_is_unchanged(self):
        short = "Revenue was high."
        out = AgentOutput(task_id="T1", agent_type="researcher", status="success",
                          summary=short)
        assert out.summary == short

    def test_evidence_defaults_empty(self):
        out = AgentOutput(task_id="T1", agent_type="skeptic", status="failed")
        assert out.evidence == []

    def test_cost_and_tokens_default_zero(self):
        out = AgentOutput(task_id="T1", agent_type="synthesizer", status="success")
        assert out.tokens_used == 0
        assert out.cost_usd == 0.0


# =============================================================================
# ResearcherOutput tests
# =============================================================================

class TestResearcherOutput:
    def test_basic_instantiation(self):
        r = ResearcherOutput(
            task_id="T1",
            chunks_retrieved=10,
            chunks_after_grading=3,
            grading_pass_rate=0.3,
            self_rag_verdict="grounded",
            source_diversity=2,
            crag_retries_used=1,
        )
        assert r.chunks_after_grading == 3
        assert r.self_rag_verdict == "grounded"

    def test_to_criteria_dict_returns_all_fields(self):
        r = ResearcherOutput(
            task_id="T1",
            chunks_retrieved=10,
            chunks_after_grading=3,
            grading_pass_rate=0.6,
            self_rag_verdict="grounded",
            source_diversity=2,
        )
        d = r.to_criteria_dict()
        assert d["chunks_after_grading"] == 3
        assert d["grading_pass_rate"] == 0.6
        assert d["self_rag_verdict"] == "grounded"
        assert d["source_diversity"] == 2

    def test_summary_truncated_at_500(self):
        r = ResearcherOutput(task_id="T1", summary="A" * 600)
        assert len(r.summary) <= 500

    def test_compute_source_diversity(self):
        chunks = [
            make_chunk(doc_id="d1", chunk_id="c1"),
            make_chunk(doc_id="d1", chunk_id="c2"),
            make_chunk(doc_id="d2", chunk_id="c3"),
        ]
        r = ResearcherOutput(task_id="T1", evidence=chunks)
        diversity = r.compute_source_diversity()
        assert diversity == 2  # d1 and d2
        assert r.source_diversity == 2


# =============================================================================
# SkepticOutput tests
# =============================================================================

class TestSkepticOutput:
    def test_basic_instantiation(self):
        s = SkepticOutput(
            task_id="T2",
            claims_checked=10,
            claims_supported=8,
            claims_unsupported=1,
            claims_contradicted=0,
            overall_confidence=0.80,
        )
        assert s.overall_confidence == 0.80

    def test_to_criteria_dict(self):
        s = SkepticOutput(
            task_id="T2",
            claims_checked=5,
            claims_supported=5,
            claims_unsupported=0,
            claims_contradicted=0,
            logical_gaps=[],
            overall_confidence=1.0,
        )
        d = s.to_criteria_dict()
        assert d["claims_unsupported"] == 0
        assert d["claims_contradicted"] == 0
        assert d["logical_gaps_count"] == 0
        assert d["overall_confidence"] == 1.0

    def test_compute_confidence(self):
        s = SkepticOutput(task_id="T2", claims_checked=10, claims_supported=7)
        confidence = s.compute_confidence()
        assert abs(confidence - 0.7) < 0.001
        assert s.overall_confidence == confidence

    def test_compute_confidence_zero_checked(self):
        s = SkepticOutput(task_id="T2", claims_checked=0, claims_supported=0)
        confidence = s.compute_confidence()
        assert confidence == 0.0

    def test_list_fields_default_empty(self):
        s = SkepticOutput(task_id="T2")
        assert s.weak_evidence_flags == []
        assert s.logical_gaps == []
        assert s.single_source_warnings == []
        assert s.forward_looking_flags == []
        assert s.reconciliations == []


# =============================================================================
# SynthesizerOutput tests
# =============================================================================

class TestSynthesizerOutput:
    def test_basic_instantiation(self):
        s = SynthesizerOutput(
            task_id="T3",
            answer="Revenue was $41,764 crore.",
            citations=[make_citation("c12", "Revenue was $41,764 crore.")],
        )
        assert s.answer == "Revenue was $41,764 crore."
        assert s.citations_count == 1

    def test_empty_citations_raises(self):
        """MF-SYN-03: citations: list[Citation] = Field(min_length=1)"""
        with pytest.raises(ValidationError):
            SynthesizerOutput(task_id="T3", answer="Some answer.", citations=[])

    def test_citations_count_auto_synced(self):
        s = SynthesizerOutput(
            task_id="T3",
            answer="x",
            citations=[make_citation("c1"), make_citation("c2")],
        )
        assert s.citations_count == 2

    def test_partial_result_flag(self):
        s = SynthesizerOutput(
            task_id="T3",
            answer="Partial answer.",
            citations=[make_citation()],
            is_partial=True,
            missing_dimensions=["headcount data"],
        )
        assert s.is_partial is True
        assert "headcount data" in s.missing_dimensions

    def test_to_criteria_dict(self):
        s = SynthesizerOutput(
            task_id="T3",
            answer="x",
            citations=[make_citation()],
            claims_count=1,
            all_citations_in_evidence_board=True,
        )
        d = s.to_criteria_dict()
        assert d["citations_count"] == 1
        assert d["claims_count"] == 1
        assert d["all_citations_in_evidence_board"] is True


# =============================================================================
# BudgetTracker tests
# =============================================================================

class TestBudgetTracker:
    def test_default_values(self):
        bt = BudgetTracker()
        assert bt.total_tokens_used == 0
        assert bt.total_cost_usd == 0.0
        assert bt.remaining == 100_000
        assert bt.api_calls == {}

    def test_add_returns_new_instance(self):
        bt = BudgetTracker()
        bt2 = bt.add(50_000, 0.25, "researcher")
        # Original unchanged
        assert bt.total_tokens_used == 0
        assert bt.remaining == 100_000
        # New instance has updated values
        assert bt2.total_tokens_used == 50_000
        assert bt2.total_cost_usd == 0.25
        assert bt2.remaining == 50_000
        assert bt2.api_calls["researcher"] == 1

    def test_add_remaining_never_below_zero(self):
        bt = BudgetTracker()
        bt2 = bt.add(200_000, 1.0, "researcher")
        assert bt2.remaining == 0

    def test_add_accumulates_api_calls(self):
        bt = BudgetTracker()
        bt = bt.add(1000, 0.01, "researcher")
        bt = bt.add(1000, 0.01, "researcher")
        bt = bt.add(500, 0.005, "skeptic")
        assert bt.api_calls["researcher"] == 2
        assert bt.api_calls["skeptic"] == 1

    def test_remaining_matches_spec(self):
        """ENG-01: BudgetTracker().add(50000, 0.25).remaining == 50000"""
        bt = BudgetTracker()
        result = bt.add(50_000, 0.25)
        assert result.remaining == 50_000

    def test_calls_for(self):
        bt = BudgetTracker()
        bt = bt.add(0, 0.0, "researcher")
        assert bt.calls_for("researcher") == 1
        assert bt.calls_for("skeptic") == 0

    def test_elapsed_seconds_positive(self):
        bt = BudgetTracker()
        elapsed = bt.elapsed_seconds()
        assert elapsed >= 0.0
        assert elapsed < 5.0  # Should be nearly instant

    def test_start_time_set(self):
        before = time.time()
        bt = BudgetTracker()
        after = time.time()
        assert before <= bt.start_time <= after


# =============================================================================
# evidence_reducer tests  (MF-MEM-01)
# =============================================================================

class TestEvidenceReducer:
    def test_empty_inputs(self):
        result = evidence_reducer([], [])
        assert result == []

    def test_new_chunks_added_to_empty(self):
        new = [make_chunk("d1", "c1", 0.8), make_chunk("d1", "c2", 0.9)]
        result = evidence_reducer([], new)
        assert len(result) == 2

    def test_duplicate_kept_with_higher_score(self):
        """MF-MEM-01: Keep highest retrieval_score for same (doc_id, chunk_id)."""
        low_score = make_chunk("d1", "c1", score=0.8)
        high_score = make_chunk("d1", "c1", score=0.9)
        result = evidence_reducer([low_score], [high_score])
        assert len(result) == 1
        assert result[0].retrieval_score == 0.9

    def test_duplicate_kept_with_higher_score_reversed(self):
        high_score = make_chunk("d1", "c1", score=0.9)
        low_score = make_chunk("d1", "c1", score=0.8)
        result = evidence_reducer([high_score], [low_score])
        assert len(result) == 1
        assert result[0].retrieval_score == 0.9

    def test_different_chunks_both_kept(self):
        c1 = make_chunk("d1", "c1", 0.8)
        c2 = make_chunk("d1", "c2", 0.9)
        result = evidence_reducer([c1], [c2])
        assert len(result) == 2

    def test_parallel_researchers_deduplication(self):
        """Scenario: T1 and T2 both retrieve chunk_12 -- should be stored once."""
        from_t1 = [make_chunk("doc_annual", "chunk_12", 0.85)]
        from_t2 = [make_chunk("doc_annual", "chunk_12", 0.92), make_chunk("doc_q3", "chunk_45", 0.7)]
        combined = evidence_reducer(from_t1, from_t2)
        assert len(combined) == 2
        chunk_12 = next(c for c in combined if c.chunk_id == "chunk_12")
        assert chunk_12.retrieval_score == 0.92  # Best score kept

    def test_reducer_is_idempotent(self):
        chunks = [make_chunk("d1", "c1", 0.8), make_chunk("d1", "c2", 0.9)]
        r1 = evidence_reducer(chunks, [])
        r2 = evidence_reducer(r1, [])
        assert len(r1) == len(r2)


# =============================================================================
# MASISState structure test
# =============================================================================

class TestMASISState:
    def test_state_is_typed_dict(self):
        """MASISState should be usable as a TypedDict for LangGraph."""
        from typing import get_type_hints
        hints = get_type_hints(MASISState, include_extras=True)
        assert "original_query" in hints
        assert "task_dag" in hints
        assert "evidence_board" in hints
        assert "token_budget" in hints
        assert "iteration_count" in hints
        assert "decision_log" in hints

    def test_state_has_annotated_evidence_board(self):
        """evidence_board must be Annotated with evidence_reducer (MF-MEM-01)."""
        from typing import Annotated, get_type_hints, get_args, get_origin
        hints = get_type_hints(MASISState, include_extras=True)
        evidence_type = hints.get("evidence_board")
        # Check it's Annotated
        assert get_origin(evidence_type) is Annotated, (
            "evidence_board must be Annotated[list[EvidenceChunk], evidence_reducer]"
        )
        # Check the reducer function is the annotation
        args = get_args(evidence_type)
        assert len(args) >= 2
        assert args[1] is evidence_reducer, (
            "The Annotated metadata on evidence_board must be the evidence_reducer function."
        )
