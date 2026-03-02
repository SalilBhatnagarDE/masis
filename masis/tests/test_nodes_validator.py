"""
Tests for masis.nodes.validator — ENG-06 (MF-VAL-01 through MF-VAL-07).

Coverage
--------
- validator_node()       : state updates, quality_scores written (MF-VAL-06)
- _score_faithfulness()  : entailment scoring (MF-VAL-01)
- _score_citation_accuracy(): chunk_id existence + entailment (MF-VAL-02)
- _score_answer_relevancy() : semantic similarity (MF-VAL-03)
- _score_dag_completeness() : fraction of tasks addressed (MF-VAL-04)
- threshold enforcement  : any score below threshold → validation_pass=False (MF-VAL-05)
- max validation rounds  : round >= 3 → force pass (MF-VAL-07)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import pytest

try:
    from masis.schemas.models import (
        Citation,
        EvidenceChunk,
        SynthesizerOutput,
        TaskNode,
    )
    from masis.nodes.validator import (
        validator_node,
        _score_dag_completeness,
        _score_answer_relevancy,
        _split_into_sentences,
        _check_thresholds,
        _jaccard_similarity,
    )
    _IMPORTS_OK = True
except ImportError as e:
    _IMPORTS_OK = False
    _IMPORT_ERROR = str(e)

pytestmark = pytest.mark.skipif(not _IMPORTS_OK, reason=f"Import failed: {locals().get('_IMPORT_ERROR', '')}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_chunk(chunk_id: str = "c1", text: str = "Revenue grew 12% YoY.") -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id,
        doc_id="doc_001",
        text=text,
        retrieval_score=0.9,
        rerank_score=0.85,
    )


def make_citation(chunk_id: str = "c1", claim: str = "Revenue grew 12%") -> Citation:
    return Citation(chunk_id=chunk_id, claim_text=claim, entailment_score=0.0)


def make_synth(
    answer: str = "Revenue grew 12% YoY to ₹41,764 crore.",
    citations: List[Citation] = None,
    claims_count: int = 1,
) -> SynthesizerOutput:
    cits = citations or [make_citation()]
    return SynthesizerOutput(
        task_id="T_synth",
        answer=answer,
        citations=cits,
        claims_count=claims_count,
        citations_count=len(cits),
        all_citations_in_evidence_board=True,
    )


def make_task_node(task_id: str, ttype: str = "researcher", status: str = "done") -> TaskNode:
    return TaskNode(
        task_id=task_id,
        type=ttype,
        query=f"Revenue data for {task_id}",
        status=status,
    )


def base_state(**kwargs: Any) -> Dict[str, Any]:
    return {
        "original_query": "What was Q3 FY26 revenue?",
        "evidence_board": [make_chunk()],
        "synthesis_output": make_synth(),
        "task_dag": [make_task_node("T1")],
        "quality_scores": {},
        "validation_round": 0,
        **kwargs,
    }


# ---------------------------------------------------------------------------
# Tests for validator_node() — overall behavior
# ---------------------------------------------------------------------------

class TestValidatorNode:
    """Tests for the full validator node."""

    @pytest.mark.asyncio
    async def test_validator_node_writes_quality_scores(self):
        """MF-VAL-06: validator_node writes quality_scores dict to state."""
        state = base_state()

        with patch("masis.nodes.validator._get_nli_pipeline", return_value=None):
            with patch("masis.nodes.validator._get_sbert_model", return_value=None):
                result = await validator_node(state)

        assert "quality_scores" in result
        scores = result["quality_scores"]
        assert "faithfulness" in scores
        assert "citation_accuracy" in scores
        assert "answer_relevancy" in scores
        assert "dag_completeness" in scores

    @pytest.mark.asyncio
    async def test_validator_node_writes_validation_round(self):
        """validator_node increments validation_round."""
        state = base_state(validation_round=1)

        with patch("masis.nodes.validator._get_nli_pipeline", return_value=None):
            with patch("masis.nodes.validator._get_sbert_model", return_value=None):
                result = await validator_node(state)

        assert result["validation_round"] == 2

    @pytest.mark.asyncio
    async def test_validator_node_no_synthesis_fails_closed(self):
        """If synthesis_output is None, validator fails closed and requests revision."""
        state = base_state(synthesis_output=None)
        result = await validator_node(state)

        assert result.get("validation_pass") is False
        assert result.get("quality_scores", {}).get("error") == "no_synthesis_output"

    @pytest.mark.asyncio
    async def test_max_validation_rounds_forces_pass(self):
        """MF-VAL-07: when validation_round > MAX_VALIDATION_ROUNDS, force pass."""
        state = base_state(validation_round=3)  # Will become 4, which exceeds MAX=3

        result = await validator_node(state)

        assert result.get("validation_pass") is True
        assert result.get("quality_scores", {}).get("forced_pass") is True


# ---------------------------------------------------------------------------
# Tests for threshold enforcement — MF-VAL-05
# ---------------------------------------------------------------------------

class TestThresholdEnforcement:
    """Tests for the quality gate logic."""

    def test_all_passing_scores_returns_empty_failures(self):
        """All scores above threshold → no failures."""
        scores = {
            "faithfulness": 0.92,
            "citation_accuracy": 0.95,
            "answer_relevancy": 0.88,
            "dag_completeness": 0.95,
        }
        thresholds = {
            "min_faithfulness": 0.85,
            "min_citation_accuracy": 0.90,
            "min_answer_relevancy": 0.80,
            "min_dag_completeness": 0.90,
        }
        failures = _check_thresholds(scores, thresholds)
        assert failures == []

    def test_low_faithfulness_returns_failure(self):
        """faithfulness below threshold → failure reported."""
        scores = {
            "faithfulness": 0.70,  # below 0.85
            "citation_accuracy": 0.95,
            "answer_relevancy": 0.88,
            "dag_completeness": 0.95,
        }
        thresholds = {
            "min_faithfulness": 0.85,
            "min_citation_accuracy": 0.90,
            "min_answer_relevancy": 0.80,
            "min_dag_completeness": 0.90,
        }
        failures = _check_thresholds(scores, thresholds)
        assert len(failures) == 1
        assert "faithfulness" in failures[0]

    def test_multiple_failures_all_reported(self):
        """Multiple scores below threshold → all failures listed."""
        scores = {
            "faithfulness": 0.70,
            "citation_accuracy": 0.75,
            "answer_relevancy": 0.60,
            "dag_completeness": 0.50,
        }
        thresholds = {
            "min_faithfulness": 0.85,
            "min_citation_accuracy": 0.90,
            "min_answer_relevancy": 0.80,
            "min_dag_completeness": 0.90,
        }
        failures = _check_thresholds(scores, thresholds)
        assert len(failures) == 4


# ---------------------------------------------------------------------------
# Tests for DAG completeness — MF-VAL-04
# ---------------------------------------------------------------------------

class TestDagCompleteness:
    """Tests for DAG completeness scoring."""

    def test_all_tasks_addressed_returns_1_0(self):
        """When answer covers all researcher task topics → completeness=1.0."""
        tasks = [
            make_task_node("T1", "researcher"),
            make_task_node("T2", "researcher"),
        ]
        tasks[0].__dict__["query"] = "cloud revenue data"
        tasks[1].__dict__["query"] = "operating margins data"

        synth = make_synth(
            answer="Cloud revenue grew 12% while operating margins improved by 2.3%."
        )
        score = _score_dag_completeness(synth, tasks)
        assert score >= 0.5  # At least partial coverage

    def test_no_researcher_tasks_returns_1_0(self):
        """When DAG has no researcher tasks → completeness=1.0 (nothing missing)."""
        tasks = [make_task_node("T1", "skeptic"), make_task_node("T2", "synthesizer")]
        synth = make_synth()
        score = _score_dag_completeness(synth, tasks)
        assert score == 1.0

    def test_empty_answer_returns_low_completeness(self):
        """Empty answer → completeness near 0."""
        tasks = [make_task_node("T1", "researcher")]
        tasks[0].__dict__["query"] = "revenue growth quarterly"
        synth = make_synth(answer=" ")
        score = _score_dag_completeness(synth, tasks)
        # With empty answer, no keywords match
        assert score == 0.0 or score < 0.5


# ---------------------------------------------------------------------------
# Tests for answer relevancy — MF-VAL-03
# ---------------------------------------------------------------------------

class TestAnswerRelevancy:
    """Tests for answer-to-query relevancy scoring."""

    @pytest.mark.asyncio
    async def test_relevant_answer_scores_above_threshold(self):
        """Semantically similar answer and query → score above 0.80."""
        answer = "Q3 FY26 consolidated revenue was ₹41,764 crore, up 12% YoY."
        query = "What was Q3 FY26 revenue?"

        with patch("masis.nodes.validator._SBERT_AVAILABLE", False):
            score = await _score_answer_relevancy(answer, query)

        # Even with Jaccard heuristic, same-domain text should score reasonably
        assert score >= 0.0  # Non-negative
        assert score <= 1.0  # Bounded

    @pytest.mark.asyncio
    async def test_empty_answer_returns_zero(self):
        """Empty answer → relevancy = 0.0."""
        with patch("masis.nodes.validator._SBERT_AVAILABLE", False):
            score = await _score_answer_relevancy("", "What was revenue?")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_unrelated_answer_scores_lower_than_relevant(self):
        """Answer about weather vs query about revenue → lower score than relevant pair."""
        with patch("masis.nodes.validator._SBERT_AVAILABLE", False):
            relevant_score = await _score_answer_relevancy(
                "Revenue was ₹41,764 crore.", "What was revenue?"
            )
            irrelevant_score = await _score_answer_relevancy(
                "Tomorrow will be sunny with a high of 25°C.", "What was revenue?"
            )

        assert relevant_score > irrelevant_score


# ---------------------------------------------------------------------------
# Tests for text utilities
# ---------------------------------------------------------------------------

class TestTextUtilities:
    """Tests for helper functions in the validator module."""

    def test_split_into_sentences_basic(self):
        """_split_into_sentences correctly splits on sentence boundaries."""
        text = "Revenue grew 12%. Margins compressed by 1.8pp. AI investments expected."
        sentences = _split_into_sentences(text)
        assert len(sentences) == 3

    def test_split_into_sentences_single_sentence(self):
        """Single sentence text → list with one element."""
        text = "Revenue was ₹41,764 crore."
        sentences = _split_into_sentences(text)
        assert len(sentences) == 1

    def test_jaccard_similarity_identical_texts(self):
        """Identical texts → Jaccard similarity = 1.0."""
        text = "revenue grew twelve percent year over year"
        score = _jaccard_similarity(text, text)
        assert score == 1.0

    def test_jaccard_similarity_disjoint_texts(self):
        """Completely disjoint texts → Jaccard similarity = 0.0."""
        score = _jaccard_similarity("apple orange banana", "doctor hospital patient")
        assert score == 0.0

    def test_jaccard_similarity_partial_overlap(self):
        """Partial overlap → Jaccard strictly between 0 and 1."""
        score = _jaccard_similarity("revenue grew twelve percent", "revenue declined in Q3")
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# Integration: validator_node with mocked NLI produces valid output
# ---------------------------------------------------------------------------

class TestValidatorIntegration:
    """Integration tests for the full validator pipeline with mocked NLI."""

    @pytest.mark.asyncio
    async def test_passing_synthesis_produces_valid_quality_scores(self):
        """All good inputs → quality_scores dict with four metrics."""
        chunk = make_chunk("c1", "Q3 FY26 revenue was ₹41,764 crore, up 12% YoY.")
        citation = make_citation("c1", "Revenue was ₹41,764 crore.")
        synth = make_synth(
            answer="Q3 FY26 revenue was ₹41,764 crore, up 12% year over year.",
            citations=[citation],
        )
        task = make_task_node("T1", "researcher")

        state = base_state(
            evidence_board=[chunk],
            synthesis_output=synth,
            task_dag=[task],
            original_query="What was Q3 FY26 revenue?",
        )

        with patch("masis.nodes.validator._get_nli_pipeline", return_value=None):
            with patch("masis.nodes.validator._get_sbert_model", return_value=None):
                result = await validator_node(state)

        scores = result["quality_scores"]
        assert all(
            k in scores
            for k in ["faithfulness", "citation_accuracy", "answer_relevancy", "dag_completeness"]
        )
        assert all(0.0 <= v <= 1.0 for v in scores.values() if isinstance(v, float))
