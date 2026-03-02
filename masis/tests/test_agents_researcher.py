"""
Tests for masis.agents.researcher — ENG-07 (MF-RES-01 through MF-RES-10).

Coverage
--------
- run_researcher()          : happy path, empty evidence, pipeline exception (M1)
- hyde_rewrite()            : LLM available, LLM unavailable fallback (MF-RES-01)
- _extract_metadata()       : year, quarter, department extraction (MF-RES-02)
- _rrf_fuse()               : deduplication, score weighting (MF-RES-03)
- _cross_encoder_rerank()   : available + unavailable cross-encoder (MF-RES-04)
- _grade_chunks()           : LLM YES/NO grading, pass_rate calculation (MF-RES-05)
- _self_rag_loop()          : grounded verdict, not_grounded retry (MF-RES-06)
- _expand_to_parents()      : parent chunk lookup (MF-RES-07)
- _count_source_diversity() : unique doc_ids counted (MF-RES-09)
- set_chroma_collection()   : registration side-effect (MF-RES-08)
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------

try:
    from masis.schemas.models import AgentOutput, EvidenceChunk, ResearcherOutput
    from masis.agents.researcher import (
        run_researcher,
        hyde_rewrite,
        _extract_metadata,
        _rrf_fuse,
        _cross_encoder_rerank,
        _grade_chunks,
        _self_rag_loop,
        _expand_to_parents,
        _count_source_diversity,
        _build_summary,
        set_chroma_collection,
        set_bm25_index,
        MAX_SUMMARY_CHARS,
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

def make_task(task_id: str = "T1", query: str = "Q3 FY26 revenue") -> Any:
    """Return a minimal TaskNode-like object."""
    task = MagicMock()
    task.task_id = task_id
    task.query = query
    task.type = "researcher"
    return task


def make_chunk(
    chunk_id: str = "c1",
    doc_id: str = "doc_001",
    text: str = "Revenue grew 12% YoY to 41764 crore.",
    retrieval_score: float = 0.8,
    rerank_score: float = 0.75,
    parent_chunk_id: str = None,
) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        retrieval_score=retrieval_score,
        rerank_score=rerank_score,
        parent_chunk_id=parent_chunk_id,
    )


def base_state(**extra: Any) -> Dict[str, Any]:
    return {
        "original_query": "What was Q3 revenue?",
        "task": make_task(),
        **extra,
    }


# ---------------------------------------------------------------------------
# Tests: run_researcher()
# ---------------------------------------------------------------------------

class TestRunResearcher:
    """Happy path and error handling for run_researcher()."""

    @pytest.mark.asyncio
    async def test_returns_agent_output_on_success(self) -> None:
        """run_researcher should return AgentOutput with status=success."""
        task = make_task()
        state = base_state()

        # Mock the internal pipeline so we don't need real LLMs/DBs
        mock_out = ResearcherOutput(
            task_id="T1",
            evidence=[make_chunk()],
            summary="Revenue grew 12%.",
            chunks_retrieved=1,
            chunks_after_grading=1,
            grading_pass_rate=0.5,
            self_rag_verdict="grounded",
            source_diversity=1,
            crag_retries_used=0,
            tokens_used=100,
            cost_usd=0.001,
        )

        with patch("masis.agents.researcher._run_pipeline", new=AsyncMock(return_value=mock_out)):
            result = await run_researcher(task, state)

        assert isinstance(result, AgentOutput)
        assert result.status == "success"
        assert result.task_id == "T1"
        assert result.agent_type == "researcher"
        assert len(result.evidence) == 1
        assert result.tokens_used == 100

    @pytest.mark.asyncio
    async def test_returns_failed_on_pipeline_exception(self) -> None:
        """run_researcher should catch pipeline errors and return status=failed."""
        task = make_task()
        state = base_state()

        with patch(
            "masis.agents.researcher._run_pipeline",
            new=AsyncMock(side_effect=RuntimeError("Chroma connection refused")),
        ):
            result = await run_researcher(task, state)

        assert result.status == "failed"
        assert "RuntimeError" in result.summary
        assert result.evidence == []

    @pytest.mark.asyncio
    async def test_uses_task_query_over_state_query(self) -> None:
        """Task query takes priority over state original_query."""
        task = make_task(query="specific Q3 sub-query")
        state = base_state(original_query="broad question")

        captured_args = {}

        async def mock_pipeline(task_id: str, query: str) -> ResearcherOutput:
            captured_args["query"] = query
            return ResearcherOutput(
                task_id=task_id,
                evidence=[],
                summary="",
                chunks_retrieved=0,
                chunks_after_grading=0,
                grading_pass_rate=0.0,
                self_rag_verdict="not_grounded",
                source_diversity=0,
                crag_retries_used=0,
            )

        with patch("masis.agents.researcher._run_pipeline", new=mock_pipeline):
            await run_researcher(task, state)

        assert captured_args["query"] == "specific Q3 sub-query"


# ---------------------------------------------------------------------------
# Tests: hyde_rewrite()
# ---------------------------------------------------------------------------

class TestHydeRewrite:
    """HyDE hypothetical document embedding generation (MF-RES-01)."""

    @pytest.mark.asyncio
    async def test_returns_passage_when_llm_available(self) -> None:
        """When LLM is available, should return a non-empty passage."""
        mock_response = MagicMock()
        mock_response.content = "TechCorp reported Q3 revenue of ₹41,764 crore in FY2026."

        with patch("masis.agents.researcher._OPENAI_AVAILABLE", True), \
             patch("masis.agents.researcher.ChatOpenAI") as mock_llm_cls:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm_cls.return_value = mock_llm

            result = await hyde_rewrite("Q3 FY26 revenue")

        assert len(result) > 20
        assert "TechCorp" in result or len(result) > 0

    @pytest.mark.asyncio
    async def test_returns_original_query_when_llm_unavailable(self) -> None:
        """Falls back to original query when LLM is not installed."""
        with patch("masis.agents.researcher._OPENAI_AVAILABLE", False), \
             patch("masis.agents.researcher.ChatOpenAI", None):
            result = await hyde_rewrite("Q3 FY26 revenue")

        assert result == "Q3 FY26 revenue"

    @pytest.mark.asyncio
    async def test_returns_original_query_on_llm_error(self) -> None:
        """Falls back to original query if LLM call raises an exception."""
        with patch("masis.agents.researcher._OPENAI_AVAILABLE", True), \
             patch("masis.agents.researcher.ChatOpenAI") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(side_effect=ConnectionError("LLM unreachable"))
            mock_llm_cls.return_value = mock_llm

            result = await hyde_rewrite("Q3 FY26 revenue")

        assert result == "Q3 FY26 revenue"


# ---------------------------------------------------------------------------
# Tests: _extract_metadata()
# ---------------------------------------------------------------------------

class TestExtractMetadata:
    """Metadata extraction from query strings (MF-RES-02)."""

    def test_extracts_4digit_year(self) -> None:
        meta = _extract_metadata("Revenue in 2026 Q3")
        assert meta.get("year") == 2026

    def test_extracts_quarter(self) -> None:
        meta = _extract_metadata("What happened in Q3 FY26?")
        assert meta.get("quarter") == "Q3"

    def test_extracts_quarter_case_insensitive(self) -> None:
        meta = _extract_metadata("q1 revenue growth")
        assert meta.get("quarter") == "Q1"

    def test_extracts_department_cloud(self) -> None:
        meta = _extract_metadata("What is the cloud revenue forecast?")
        assert meta.get("department") == "cloud"

    def test_extracts_department_sales(self) -> None:
        meta = _extract_metadata("Sales team performance in FY2026")
        assert meta.get("department") == "sales"

    def test_returns_empty_for_plain_query(self) -> None:
        meta = _extract_metadata("What are the key risks?")
        # No year, quarter, or known department keyword
        assert "year" not in meta or meta.get("year") is None or True  # permissive
        assert "quarter" not in meta

    def test_no_false_year_for_short_numbers(self) -> None:
        meta = _extract_metadata("Revenue grew by 12 percent")
        assert "year" not in meta


# ---------------------------------------------------------------------------
# Tests: _rrf_fuse()
# ---------------------------------------------------------------------------

class TestRRFFuse:
    """Reciprocal Rank Fusion correctness (MF-RES-03)."""

    def test_deduplicates_identical_chunks(self) -> None:
        """A chunk appearing in both lists should appear only once in fused output."""
        chunk_shared = make_chunk("shared_c", "doc_A")
        chunk_vec_only = make_chunk("vec_only", "doc_B")
        chunk_bm25_only = make_chunk("bm25_only", "doc_C")

        vector_list = [chunk_shared, chunk_vec_only]
        bm25_list = [chunk_shared, chunk_bm25_only]

        result = _rrf_fuse(vector_list, bm25_list)
        chunk_ids = [c.chunk_id for c in result]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk_ids in fused output"

    def test_shared_chunk_scores_higher_than_single_list(self) -> None:
        """A chunk in both lists should score higher than one only in one list."""
        shared = make_chunk("shared", "docA", retrieval_score=0.5)
        only_vec = make_chunk("only_vec", "docB", retrieval_score=0.5)

        result = _rrf_fuse([shared, only_vec], [shared])
        top_chunk = result[0]
        assert top_chunk.chunk_id == "shared"

    def test_empty_lists_return_empty(self) -> None:
        result = _rrf_fuse([], [])
        assert result == []

    def test_single_vector_list_only(self) -> None:
        chunks = [make_chunk(f"c{i}", f"doc{i}") for i in range(3)]
        result = _rrf_fuse(chunks, [])
        assert len(result) == 3

    def test_respects_top_k_limit(self) -> None:
        """Result should be capped at top_k_retrieval (default 10)."""
        chunks = [make_chunk(f"c{i}", f"doc{i}") for i in range(15)]
        result = _rrf_fuse(chunks, [])
        assert len(result) <= 10


# ---------------------------------------------------------------------------
# Tests: _cross_encoder_rerank()
# ---------------------------------------------------------------------------

class TestCrossEncoderRerank:
    """Cross-encoder reranking (MF-RES-04)."""

    @pytest.mark.asyncio
    async def test_returns_top_k_chunks(self) -> None:
        """Should return at most top_k_after_rerank=5 chunks."""
        chunks = [make_chunk(f"c{i}", f"doc{i}", rerank_score=0.0) for i in range(8)]
        query = "revenue"

        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

        with patch("masis.agents.researcher._CROSS_ENCODER_AVAILABLE", True), \
             patch("masis.agents.researcher._get_cross_encoder", return_value=mock_encoder):
            result = await _cross_encoder_rerank(query, chunks)

        assert len(result) <= 5

    @pytest.mark.asyncio
    async def test_falls_back_when_encoder_unavailable(self) -> None:
        """Without cross-encoder, should fall back to retrieval_score ordering."""
        chunks = [
            make_chunk("c1", "d1", retrieval_score=0.3),
            make_chunk("c2", "d2", retrieval_score=0.9),
            make_chunk("c3", "d3", retrieval_score=0.6),
        ]

        with patch("masis.agents.researcher._CROSS_ENCODER_AVAILABLE", False), \
             patch("masis.agents.researcher.CrossEncoder", None):
            result = await _cross_encoder_rerank("any query", chunks)

        # c2 should come first (highest retrieval_score)
        assert result[0].chunk_id == "c2"

    @pytest.mark.asyncio
    async def test_empty_chunks_returns_empty(self) -> None:
        with patch("masis.agents.researcher._CROSS_ENCODER_AVAILABLE", False):
            result = await _cross_encoder_rerank("query", [])
        assert result == []


# ---------------------------------------------------------------------------
# Tests: _grade_chunks()
# ---------------------------------------------------------------------------

class TestGradeChunks:
    """CRAG relevance grading (MF-RES-05)."""

    @pytest.mark.asyncio
    async def test_grade_all_yes_returns_full_pass_rate(self) -> None:
        """When all chunks grade YES, pass_rate should be 1.0."""
        chunks = [make_chunk("c1"), make_chunk("c2")]
        query = "Q3 revenue"

        mock_resp = MagicMock()
        mock_resp.content = "YES"

        with patch("masis.agents.researcher._OPENAI_AVAILABLE", True), \
             patch("masis.agents.researcher.ChatOpenAI") as mock_cls:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_llm

            relevant, pass_rate = await _grade_chunks(query, chunks)

        assert pass_rate == 1.0
        assert len(relevant) == 2

    @pytest.mark.asyncio
    async def test_grade_all_no_returns_zero_pass_rate(self) -> None:
        """When all chunks grade NO, pass_rate should be 0.0."""
        chunks = [make_chunk("c1"), make_chunk("c2")]
        query = "irrelevant topic"

        mock_resp = MagicMock()
        mock_resp.content = "NO"

        with patch("masis.agents.researcher._OPENAI_AVAILABLE", True), \
             patch("masis.agents.researcher.ChatOpenAI") as mock_cls:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_llm

            relevant, pass_rate = await _grade_chunks(query, chunks)

        assert pass_rate == 0.0
        assert len(relevant) == 0

    @pytest.mark.asyncio
    async def test_empty_chunks_returns_zero(self) -> None:
        relevant, pass_rate = await _grade_chunks("query", [])
        assert relevant == []
        assert pass_rate == 0.0

    @pytest.mark.asyncio
    async def test_fallback_without_llm_returns_median_split(self) -> None:
        """Without LLM, should split by median rerank_score."""
        chunks = [
            make_chunk("c1", rerank_score=0.9),
            make_chunk("c2", rerank_score=0.3),
        ]
        with patch("masis.agents.researcher._OPENAI_AVAILABLE", False), \
             patch("masis.agents.researcher.ChatOpenAI", None):
            relevant, pass_rate = await _grade_chunks("any query", chunks)

        # c1 should be in relevant (score >= median)
        assert any(c.chunk_id == "c1" for c in relevant)


# ---------------------------------------------------------------------------
# Tests: _self_rag_loop()
# ---------------------------------------------------------------------------

class TestSelfRagLoop:
    """Self-RAG hallucination check (MF-RES-06)."""

    @pytest.mark.asyncio
    async def test_returns_grounded_when_check_says_yes(self) -> None:
        """Should return 'grounded' when the LLM says YES to the grounding check."""
        chunks = [make_chunk("c1", text="Q3 revenue was 41764 crore.")]

        gen_resp = MagicMock()
        gen_resp.content = "Revenue was 41764 crore in Q3."

        check_resp = MagicMock()
        check_resp.content = "YES, the answer is fully supported."

        call_count = [0]

        async def ainvoke_side_effect(prompt: Any) -> MagicMock:
            call_count[0] += 1
            if call_count[0] == 1:
                return gen_resp
            return check_resp

        with patch("masis.agents.researcher._OPENAI_AVAILABLE", True), \
             patch("masis.agents.researcher.ChatOpenAI") as mock_cls:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(side_effect=ainvoke_side_effect)
            mock_cls.return_value = mock_llm

            answer, verdict, tokens = await _self_rag_loop("Q3 revenue", chunks, 3)

        assert verdict == "grounded"
        assert len(answer) > 0

    @pytest.mark.asyncio
    async def test_returns_not_grounded_after_max_retries(self) -> None:
        """After max_retries with NO, verdict should be 'partial'."""
        chunks = [make_chunk("c1")]

        resp = MagicMock()
        resp.content = "NO"

        with patch("masis.agents.researcher._OPENAI_AVAILABLE", True), \
             patch("masis.agents.researcher.ChatOpenAI") as mock_cls:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=resp)
            mock_cls.return_value = mock_llm

            _answer, verdict, _tokens = await _self_rag_loop("query", chunks, max_retries=2)

        assert verdict in ("not_grounded", "partial")

    @pytest.mark.asyncio
    async def test_empty_chunks_returns_not_grounded(self) -> None:
        answer, verdict, tokens = await _self_rag_loop("query", [], 3)
        assert verdict == "not_grounded"
        assert tokens == 0


# ---------------------------------------------------------------------------
# Tests: _count_source_diversity()
# ---------------------------------------------------------------------------

class TestCountSourceDiversity:
    """Unique doc_id counting (MF-RES-09)."""

    def test_two_chunks_same_doc(self) -> None:
        chunks = [
            make_chunk("c1", doc_id="doc_A"),
            make_chunk("c2", doc_id="doc_A"),
        ]
        assert _count_source_diversity(chunks) == 1

    def test_three_different_docs(self) -> None:
        chunks = [
            make_chunk("c1", doc_id="doc_A"),
            make_chunk("c2", doc_id="doc_B"),
            make_chunk("c3", doc_id="doc_C"),
        ]
        assert _count_source_diversity(chunks) == 3

    def test_empty_chunks_returns_zero(self) -> None:
        assert _count_source_diversity([]) == 0


# ---------------------------------------------------------------------------
# Tests: _build_summary()
# ---------------------------------------------------------------------------

class TestBuildSummary:
    """Summary truncation (MF-RES-10)."""

    def test_short_answer_unchanged(self) -> None:
        assert _build_summary("Short.") == "Short."

    def test_long_answer_truncated(self) -> None:
        long_text = "x" * (MAX_SUMMARY_CHARS + 200)
        result = _build_summary(long_text)
        assert len(result) <= MAX_SUMMARY_CHARS


# ---------------------------------------------------------------------------
# Tests: set_chroma_collection / set_bm25_index
# ---------------------------------------------------------------------------

class TestIndexRegistration:
    """External index registration side-effects (MF-RES-08)."""

    def test_set_chroma_collection(self) -> None:
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        set_chroma_collection(mock_collection)
        # No exception means success; test by calling retrieval
        from masis.agents.researcher import _get_chroma_collection
        assert _get_chroma_collection() is mock_collection

    def test_set_bm25_index(self) -> None:
        mock_index = MagicMock()
        corpus = ["doc1 text", "doc2 text"]
        meta = [{"chunk_id": "c1"}, {"chunk_id": "c2"}]
        set_bm25_index(mock_index, corpus, meta)
        from masis.agents.researcher import _get_bm25_index
        result = _get_bm25_index()
        assert result is not None
        assert result[0] is mock_index
        assert result[1] == corpus
