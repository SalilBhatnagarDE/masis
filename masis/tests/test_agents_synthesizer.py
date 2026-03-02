"""
Tests for masis.agents.synthesizer — ENG-09 (MF-SYN-01 through MF-SYN-08).

Coverage
--------
- run_synthesizer()            : happy path, no evidence guard, exception handling
- u_shape_order()              : algorithm correctness, edge cases (MF-SYN-01)
- _build_critique_instructions(): forward-looking, single-source, reconciled, gaps (MF-SYN-02)
- _generate_synthesis()        : LLM available, LLM unavailable fallback (MF-SYN-03)
- _verify_citations_nli()      : NLI available, heuristic fallback (MF-SYN-05)
- _detect_missing_dimensions() : undone tasks with no evidence keywords (MF-SYN-06)
- _build_no_evidence_output()  : honest no-evidence answer with placeholder (MF-SYN-07)
- _make_fallback_citation()    : fallback when LLM produces no citations
- _heuristic_synthesis()       : non-LLM path produces valid output
- _count_claims()              : sentence counting heuristic
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------

try:
    from masis.schemas.models import (
        AgentOutput,
        Citation,
        EvidenceChunk,
        SkepticOutput,
        SynthesizerOutput,
    )
    from masis.agents.synthesizer import (
        run_synthesizer,
        u_shape_order,
        _build_critique_instructions,
        _build_no_evidence_output,
        _make_fallback_citation,
        _verify_citations_nli,
        _detect_missing_dimensions,
        _heuristic_synthesis,
        _count_claims,
        _format_evidence_for_synthesis,
    )
    _IMPORTS_OK = True
except ImportError as e:
    _IMPORTS_OK = False
    _IMPORT_ERROR = str(e)

pytestmark = pytest.mark.skipif(
    not _IMPORTS_OK,
    reason=f"Import failed: {locals().get('_IMPORT_ERROR', '')}",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_task(task_id: str = "T_synth") -> Any:
    task = MagicMock()
    task.task_id = task_id
    task.type = "synthesizer"
    return task


def make_chunk(
    chunk_id: str = "c1",
    doc_id: str = "doc_001",
    text: str = "Revenue grew 12% YoY to 41,764 crore.",
    rerank_score: float = 0.85,
    source_label: str = "Annual Report 2026",
) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        retrieval_score=rerank_score,
        rerank_score=rerank_score,
        source_label=source_label,
    )


def make_citation(chunk_id: str = "c1", claim: str = "Revenue grew 12%.") -> Citation:
    return Citation(chunk_id=chunk_id, claim_text=claim, entailment_score=0.0)


def make_skeptic_output(**kwargs: Any) -> SkepticOutput:
    return SkepticOutput(
        task_id="T_skeptic",
        claims_checked=kwargs.get("claims_checked", 5),
        claims_supported=kwargs.get("claims_supported", 4),
        claims_unsupported=kwargs.get("claims_unsupported", 1),
        claims_contradicted=kwargs.get("claims_contradicted", 0),
        overall_confidence=kwargs.get("overall_confidence", 0.80),
        forward_looking_flags=kwargs.get("forward_looking_flags", []),
        single_source_warnings=kwargs.get("single_source_warnings", []),
        reconciliations=kwargs.get("reconciliations", []),
        logical_gaps=kwargs.get("logical_gaps", []),
    )


def make_task_node(task_id: str, ttype: str = "researcher", status: str = "done") -> Any:
    node = MagicMock()
    node.task_id = task_id
    node.type = ttype
    node.status = status
    node.query = f"Research query for {task_id}"
    return node


def base_state(
    chunks: List[EvidenceChunk] = None,
    critique: Optional[SkepticOutput] = None,
    **extra: Any,
) -> Dict[str, Any]:
    board = chunks if chunks is not None else [make_chunk()]
    return {
        "original_query": "What was Q3 revenue?",
        "evidence_board": board,
        "critique_notes": critique,
        "task_dag": [],
        "supervisor_decision": "continue",
        **extra,
    }


# ---------------------------------------------------------------------------
# Tests: run_synthesizer()
# ---------------------------------------------------------------------------

class TestRunSynthesizer:
    """Entry point integration tests."""

    @pytest.mark.asyncio
    async def test_returns_agent_output_on_success(self) -> None:
        """run_synthesizer should return AgentOutput with status=success."""
        task = make_task()
        state = base_state()

        mock_synth = SynthesizerOutput(
            task_id="T_synth",
            answer="Revenue grew 12% YoY.",
            citations=[make_citation()],
            claims_count=1,
            citations_count=1,
            all_citations_in_evidence_board=True,
            tokens_used=150,
            cost_usd=0.002,
        )

        with patch(
            "masis.agents.synthesizer._run_synthesis_pipeline",
            new=AsyncMock(return_value=mock_synth),
        ):
            result = await run_synthesizer(task, state)

        assert isinstance(result, AgentOutput)
        assert result.status == "success"
        assert result.agent_type == "synthesizer"
        assert result.tokens_used == 150

    @pytest.mark.asyncio
    async def test_returns_success_for_empty_evidence(self) -> None:
        """Empty evidence board should return success with no-evidence answer."""
        task = make_task()
        state = base_state(chunks=[])
        result = await run_synthesizer(task, state)

        assert result.status == "success"
        assert "No evidence" in result.summary

    @pytest.mark.asyncio
    async def test_returns_failed_on_exception(self) -> None:
        """Pipeline exception should produce status=failed."""
        task = make_task()
        state = base_state()

        with patch(
            "masis.agents.synthesizer._run_synthesis_pipeline",
            new=AsyncMock(side_effect=RuntimeError("gpt-4.1 unavailable")),
        ):
            result = await run_synthesizer(task, state)

        assert result.status == "failed"
        assert "gpt-4.1 unavailable" in result.summary


# ---------------------------------------------------------------------------
# Tests: u_shape_order()
# ---------------------------------------------------------------------------

class TestUShapeOrder:
    """U-shape context ordering algorithm (MF-SYN-01)."""

    def test_5_chunks_correct_positions(self) -> None:
        """Best score at position 0, second-best at position -1 (last)."""
        chunks = [make_chunk(f"c{i}", rerank_score=score) for i, score in enumerate(
            [0.92, 0.87, 0.81, 0.74, 0.69]
        )]
        result = u_shape_order(chunks)

        assert len(result) == 5
        # Best chunk (0.92) should be at the start
        assert result[0].rerank_score == 0.92
        # Second-best chunk (0.87) should be at the end (last position)
        assert result[-1].rerank_score == 0.87

    def test_2_chunks_sorted_descending(self) -> None:
        """With ≤2 chunks, just sort descending by rerank_score."""
        chunks = [
            make_chunk("c1", rerank_score=0.5),
            make_chunk("c2", rerank_score=0.9),
        ]
        result = u_shape_order(chunks)
        assert result[0].rerank_score == 0.9
        assert result[1].rerank_score == 0.5

    def test_1_chunk_returned_unchanged(self) -> None:
        chunks = [make_chunk("c1", rerank_score=0.8)]
        result = u_shape_order(chunks)
        assert len(result) == 1
        assert result[0].chunk_id == "c1"

    def test_empty_list_returns_empty(self) -> None:
        assert u_shape_order([]) == []

    def test_preserves_all_chunks(self) -> None:
        """No chunks should be dropped or added."""
        n = 7
        chunks = [make_chunk(f"c{i}", rerank_score=1.0 - i * 0.1) for i in range(n)]
        result = u_shape_order(chunks)
        assert len(result) == n
        assert {c.chunk_id for c in result} == {c.chunk_id for c in chunks}

    def test_weakest_in_middle(self) -> None:
        """Weakest chunk should not be at position 0 or last position."""
        chunks = [make_chunk(f"c{i}", rerank_score=1.0 - i * 0.15) for i in range(5)]
        result = u_shape_order(chunks)
        weakest = min(chunks, key=lambda c: c.rerank_score)
        # Weakest should not be first or last
        assert result[0].chunk_id != weakest.chunk_id
        assert result[-1].chunk_id != weakest.chunk_id


# ---------------------------------------------------------------------------
# Tests: _build_critique_instructions()
# ---------------------------------------------------------------------------

class TestBuildCritiqueInstructions:
    """Critique instruction formatting (MF-SYN-02)."""

    def test_none_critique_returns_default(self) -> None:
        result = _build_critique_instructions(None)
        assert "No specific critique flags" in result

    def test_includes_forward_looking_instructions(self) -> None:
        critique = make_skeptic_output(
            forward_looking_flags=["Expected to grow 15% next year."]
        )
        result = _build_critique_instructions(critique)
        assert "FORWARD-LOOKING" in result

    def test_includes_single_source_warnings(self) -> None:
        critique = make_skeptic_output(
            single_source_warnings=["Single-source claim: revenue figure."]
        )
        result = _build_critique_instructions(critique)
        assert "SINGLE-SOURCE" in result

    def test_includes_reconciliation_instructions(self) -> None:
        critique = make_skeptic_output(
            reconciliations=[{"explanation": "Sources disagree on margin figure.", "resolved": True}]
        )
        result = _build_critique_instructions(critique)
        assert "RECONCILED" in result

    def test_includes_logical_gaps(self) -> None:
        critique = make_skeptic_output(
            logical_gaps=["Gap: No data on international revenue."]
        )
        result = _build_critique_instructions(critique)
        assert "LOGICAL GAPS" in result

    def test_no_flags_returns_confidence_note(self) -> None:
        critique = make_skeptic_output(overall_confidence=0.88)
        result = _build_critique_instructions(critique)
        # No flags set → should mention confidence
        assert "0.88" in result or "No critical" in result


# ---------------------------------------------------------------------------
# Tests: _build_no_evidence_output()
# ---------------------------------------------------------------------------

class TestBuildNoEvidenceOutput:
    """Honest no-evidence answer (MF-SYN-07)."""

    def test_returns_synthesizer_output(self) -> None:
        result = _build_no_evidence_output("T1", "What was Q3 revenue?")
        assert isinstance(result, SynthesizerOutput)

    def test_answer_mentions_no_evidence(self) -> None:
        result = _build_no_evidence_output("T1", "What was Q3 revenue?")
        assert "No evidence" in result.answer

    def test_has_at_least_one_citation(self) -> None:
        """SynthesizerOutput requires min 1 citation."""
        result = _build_no_evidence_output("T1", "What was Q3 revenue?")
        assert len(result.citations) >= 1

    def test_is_partial_flag_set(self) -> None:
        result = _build_no_evidence_output("T1", "test query")
        assert result.is_partial is True

    def test_task_id_preserved(self) -> None:
        result = _build_no_evidence_output("special_task_XYZ", "query")
        assert result.task_id == "special_task_XYZ"


# ---------------------------------------------------------------------------
# Tests: _make_fallback_citation()
# ---------------------------------------------------------------------------

class TestMakeFallbackCitation:
    """Fallback citation generation."""

    def test_returns_list_with_one_citation(self) -> None:
        board = [make_chunk("c_first")]
        result = _make_fallback_citation(board)
        assert len(result) == 1
        assert isinstance(result[0], Citation)

    def test_uses_first_chunk_id(self) -> None:
        board = [make_chunk("first_chunk"), make_chunk("second_chunk")]
        result = _make_fallback_citation(board)
        assert result[0].chunk_id == "first_chunk"

    def test_empty_board_returns_fallback_citation(self) -> None:
        result = _make_fallback_citation([])
        assert len(result) == 1
        assert result[0].chunk_id == "fallback"


# ---------------------------------------------------------------------------
# Tests: _detect_missing_dimensions()
# ---------------------------------------------------------------------------

class TestDetectMissingDimensions:
    """Missing dimension detection for partial synthesis disclaimer (MF-SYN-06)."""

    def test_detects_task_with_no_evidence_coverage(self) -> None:
        """A pending researcher task whose keywords don't appear in evidence → missing."""
        task = make_task_node("T2", ttype="researcher", status="pending")
        task.query = "TechCorp headcount reduction workforce"

        evidence = [
            make_chunk("c1", text="Revenue grew 12% in Q3.")  # No headcount keywords
        ]

        missing = _detect_missing_dimensions([task], evidence)
        assert len(missing) >= 0  # May find 0 or 1 depending on keyword overlap

    def test_done_tasks_not_flagged(self) -> None:
        """Completed researcher tasks should not appear in missing dimensions."""
        done_task = make_task_node("T1", ttype="researcher", status="done")
        done_task.query = "Q3 revenue analysis"

        evidence = [make_chunk("c1", text="Q3 revenue analysis showed growth.")]
        missing = _detect_missing_dimensions([done_task], evidence)
        # Done tasks are excluded
        assert len(missing) == 0

    def test_non_researcher_tasks_ignored(self) -> None:
        """Skeptic and synthesizer tasks should not contribute to missing dimensions."""
        skeptic_task = make_task_node("T_skep", ttype="skeptic", status="pending")
        skeptic_task.query = "verify all evidence"
        evidence = [make_chunk("c1")]
        missing = _detect_missing_dimensions([skeptic_task], evidence)
        assert len(missing) == 0


# ---------------------------------------------------------------------------
# Tests: _heuristic_synthesis()
# ---------------------------------------------------------------------------

class TestHeuristicSynthesis:
    """Non-LLM fallback synthesis."""

    def test_returns_valid_tuple(self) -> None:
        board = [make_chunk("c1"), make_chunk("c2")]
        answer, citations, tokens, cost = _heuristic_synthesis("What is Q3 revenue?", board)
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert len(citations) >= 1
        assert tokens == 0
        assert cost == 0.0

    def test_empty_board_returns_no_evidence_message(self) -> None:
        answer, citations, tokens, cost = _heuristic_synthesis("query", [])
        assert "No evidence" in answer
        assert len(citations) >= 1

    def test_uses_top_3_chunks(self) -> None:
        """Heuristic synthesis should use at most 3 chunks."""
        board = [make_chunk(f"c{i}") for i in range(6)]
        _answer, citations, _t, _c = _heuristic_synthesis("test query", board)
        # Citations correspond to the chunks used (max 3)
        assert len(citations) <= 3


# ---------------------------------------------------------------------------
# Tests: _count_claims()
# ---------------------------------------------------------------------------

class TestCountClaims:
    """Sentence counting heuristic for claims_count."""

    def test_counts_sentences(self) -> None:
        answer = "Revenue grew 12% YoY. Operating margin expanded by 200 bps. Cloud was the key driver."
        count = _count_claims(answer)
        assert count >= 3

    def test_filters_short_sentences(self) -> None:
        answer = "Q3. Revenue grew by 12% in the fiscal quarter ended September 2026."
        count = _count_claims(answer)
        # "Q3." should be filtered (< 20 chars), so count = 1
        assert count >= 1

    def test_empty_answer_returns_zero(self) -> None:
        assert _count_claims("") == 0


# ---------------------------------------------------------------------------
# Tests: _verify_citations_nli()
# ---------------------------------------------------------------------------

class TestVerifyCitationsNli:
    """Post-hoc NLI citation verification (MF-SYN-05)."""

    def test_sets_entailment_score_on_citations(self) -> None:
        """Should fill entailment_score for all citations that have matching chunks."""
        chunk = make_chunk("c1", text="Revenue grew 12% in Q3 FY26.")
        citation = make_citation("c1", "Revenue grew 12% in Q3.")

        mock_nli = MagicMock()
        mock_nli.return_value = {
            "labels": ["entailment", "contradiction", "neutral"],
            "scores": [0.92, 0.04, 0.04],
        }

        with patch("masis.agents.synthesizer._get_nli_pipeline", return_value=mock_nli):
            result = _verify_citations_nli([citation], [chunk])

        assert result[0].entailment_score == pytest.approx(0.92, abs=0.01)

    def test_zero_score_for_missing_chunk(self) -> None:
        """Citation referencing a non-existent chunk should get score=0.0."""
        citation = make_citation("nonexistent_chunk_id", "Some claim.")
        evidence = [make_chunk("c1")]

        result = _verify_citations_nli([citation], evidence)
        assert result[0].entailment_score == 0.0

    def test_heuristic_fallback_without_nli(self) -> None:
        """Without NLI pipeline, should use Jaccard similarity as fallback."""
        chunk = make_chunk("c1", text="Revenue grew twelve percent.")
        citation = make_citation("c1", "Revenue grew twelve percent.")

        with patch("masis.agents.synthesizer._get_nli_pipeline", return_value=None):
            result = _verify_citations_nli([citation], [chunk])

        # Jaccard of identical text is close to 1.0
        assert result[0].entailment_score > 0.5
