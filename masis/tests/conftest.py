"""
conftest.py
===========
Shared pytest fixtures and configuration for the MASIS test suite.

These fixtures are automatically available to all test modules in this
directory without explicit import.

Fixture categories
------------------
State fixtures:
    empty_state         -- Minimal MASISState with only required fields
    minimal_state       -- State with query, task DAG, and empty evidence board
    full_state          -- State with all Phase 0 fields populated

Model fixtures:
    sample_evidence_chunks  -- List of EvidenceChunk objects for reordering tests
    sample_task_dag         -- Simple DAG: researcher -> synthesizer

Config fixtures:
    test_settings           -- Settings instance with safe dummy values
    patch_openai_key        -- Temporarily sets OPENAI_API_KEY env var

Helpers:
    make_chunk              -- Factory function for EvidenceChunk
    make_task               -- Factory function for TaskNode
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Environment setup -- ensure tests can import masis without OPENAI_API_KEY
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def suppress_missing_key_warnings(monkeypatch):
    """
    Ensure OPENAI_API_KEY is set to a dummy value for all tests that don't
    specifically test the missing-key validation path.

    Tests that explicitly test missing-key behaviour use their own _set_env()
    helpers and override this fixture's monkeypatch.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key-for-ci")
    yield


# ---------------------------------------------------------------------------
# Factory helpers (function fixtures for flexible use)
# ---------------------------------------------------------------------------

@pytest.fixture
def make_chunk():
    """
    Factory fixture that returns a function to create EvidenceChunk objects.

    Usage:
        def test_something(make_chunk):
            chunk = make_chunk("doc1", "chunk1", rerank_score=0.85)
    """
    from masis.schemas.models import EvidenceChunk

    def _make(
        doc_id: str = "doc1",
        chunk_id: str = "chunk1",
        text: str = "Sample evidence text about TechCorp.",
        rerank_score: float = 0.75,
        retrieval_score: float = 0.70,
        source_url: str = "https://example.com/article",
        **kwargs: Any,
    ) -> EvidenceChunk:
        return EvidenceChunk(
            doc_id=doc_id,
            chunk_id=chunk_id,
            text=text,
            rerank_score=rerank_score,
            retrieval_score=retrieval_score,
            source_url=source_url,
            **kwargs,
        )

    return _make


@pytest.fixture
def make_task():
    """
    Factory fixture that returns a function to create TaskNode objects.

    Usage:
        def test_something(make_task):
            task = make_task("T1", "researcher", "TechCorp revenue")
    """
    from masis.schemas.models import TaskNode

    def _make(
        task_id: str = "T1",
        task_type: str = "researcher",
        query: str = "TechCorp revenue analysis",
        status: str = "pending",
        dependencies: List[str] = None,
        parallel_group: int = 1,
        acceptance_criteria: str = "",
        **kwargs: Any,
    ) -> TaskNode:
        return TaskNode(
            task_id=task_id,
            type=task_type,
            query=query,
            status=status,
            dependencies=dependencies or [],
            parallel_group=parallel_group,
            acceptance_criteria=acceptance_criteria,
            **kwargs,
        )

    return _make


# ---------------------------------------------------------------------------
# State fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_state() -> Dict[str, Any]:
    """
    Minimal MASISState with only the required fields populated.
    Task DAG is empty. Evidence board is empty.
    """
    from masis.schemas.models import BudgetTracker
    return {
        "original_query": "",
        "query_id": "test-query-id-empty",
        "task_dag": [],
        "iteration_count": 0,
        "evidence_board": [],
        "token_budget": BudgetTracker(tokens_used=0, cost_usd=0.0),
        "decision_log": [],
    }


@pytest.fixture
def minimal_state(make_task) -> Dict[str, Any]:
    """
    MASISState with a simple 2-task DAG (researcher → synthesizer) and
    an empty evidence board. All tasks are in 'pending' status.

    Suitable for testing DAG traversal and Supervisor routing.
    """
    from masis.schemas.models import BudgetTracker
    researcher = make_task("T1", "researcher", "TechCorp revenue analysis",
                            status="pending", parallel_group=1)
    synthesizer = make_task("T2", "synthesizer", "Summarize TechCorp findings",
                             status="pending", dependencies=["T1"], parallel_group=2)
    return {
        "original_query": "Analyse TechCorp's financial performance.",
        "query_id": "test-query-id-minimal",
        "task_dag": [researcher, synthesizer],
        "iteration_count": 0,
        "evidence_board": [],
        "token_budget": BudgetTracker(tokens_used=0, cost_usd=0.0),
        "decision_log": [],
        "start_time": 0.0,
        "final_answer": None,
        "validation_report": None,
    }


@pytest.fixture
def full_state(make_task, make_chunk) -> Dict[str, Any]:
    """
    MASISState with a 3-task DAG (researcher, web_search → synthesizer),
    populated evidence board, and budget tracking.

    Suitable for Synthesizer, Validator, and end-to-end tests.
    """
    from masis.schemas.models import BudgetTracker, Citation, SynthesizerOutput
    researcher = make_task("T1", "researcher", "TechCorp revenue analysis Q3 2024",
                            status="done", parallel_group=1)
    web_search = make_task("T2", "web_search", "TechCorp latest news 2024",
                            status="done", parallel_group=1)
    synthesizer = make_task("T3", "synthesizer", "Summarize all TechCorp findings",
                             status="pending", dependencies=["T1", "T2"], parallel_group=2)

    chunks = [
        make_chunk("doc1", f"chunk{i}", rerank_score=round(0.9 - i * 0.05, 2))
        for i in range(5)
    ]

    citation = Citation(
        chunk_id="chunk0",
        doc_id="doc1",
        quote="TechCorp revenue grew by 15% in Q3 2024.",
        relevance=0.92,
    )
    synth = SynthesizerOutput(
        answer="TechCorp demonstrated strong growth in Q3 2024.",
        citations=[citation],
        confidence=0.88,
        status="success",
    )

    return {
        "original_query": "Analyse TechCorp financial performance Q3 2024.",
        "query_id": "test-query-id-full",
        "task_dag": [researcher, web_search, synthesizer],
        "iteration_count": 3,
        "evidence_board": chunks,
        "token_budget": BudgetTracker(tokens_used=1500, cost_usd=0.02),
        "decision_log": [
            {
                "turn": 0, "mode": "plan", "task_id": "PLAN",
                "decision": "plan_created", "cost": 0.015,
                "latency_ms": 2000.0, "task_count": 3, "timestamp": 1000.0,
            },
            {
                "turn": 1, "mode": "fast", "task_id": "T1",
                "decision": "continue", "cost": 0.0,
                "latency_ms": 2.5, "timestamp": 1001.0,
            },
        ],
        "start_time": 1000.0,
        "final_answer": synth,
        "validation_report": None,
        "supervisor_decision": "ready_for_validation",
    }


# ---------------------------------------------------------------------------
# Evidence chunk fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def five_chunks(make_chunk) -> List:
    """
    Five EvidenceChunk objects with descending rerank scores.
    Canonical test set for u_shape_order() tests.

    Scores: [0.92, 0.87, 0.81, 0.74, 0.69]
    Expected U-shape order: [0.92, 0.81, 0.69, 0.74, 0.87]
    """
    scores = [0.92, 0.87, 0.81, 0.74, 0.69]
    return [make_chunk("doc1", f"c{i}", text=f"Evidence chunk {i}",
                        rerank_score=s) for i, s in enumerate(scores)]


@pytest.fixture
def ten_chunks(make_chunk) -> List:
    """Ten EvidenceChunk objects for retrieval pipeline tests."""
    return [make_chunk(f"doc{i}", f"chunk{i}", text=f"Evidence about topic {i}.",
                        rerank_score=round(1.0 - i * 0.05, 2))
            for i in range(10)]


# ---------------------------------------------------------------------------
# Settings fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_settings(monkeypatch):
    """
    Settings instance with safe dummy values for all required keys.
    Resets the settings singleton before and after each test.
    """
    from masis.config.settings import Settings, reset_settings

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fixture-key")
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    reset_settings()
    settings = Settings()
    yield settings
    reset_settings()


# ---------------------------------------------------------------------------
# Cleanup: reset module-level singletons between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_settings_singleton():
    """
    Reset the Settings singleton between every test to prevent state leakage.
    This is especially important for tests that modify env vars.
    """
    from masis.config.settings import reset_settings
    reset_settings()
    yield
    reset_settings()
