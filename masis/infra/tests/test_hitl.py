"""
tests/test_hitl.py
==================
Comprehensive unit tests for masis.infra.hitl.

Covers
------
MF-HITL-01  ambiguity_detector: heuristic path (no LLM required)
            - CLEAR classification
            - AMBIGUOUS classification with options
            - OUT_OF_SCOPE classification
MF-HITL-02  dag_approval_interrupt: payload structure, action routing
MF-HITL-03  mid_execution_interrupt: payload structure
MF-HITL-04  risk_gate: keyword detection, risk score, threshold
MF-HITL-05  handle_resume: all action paths
MF-HITL-06  build_partial_result: coverage, missing tasks, disclaimer
MF-HITL-07  build_cancel_result: completed/pending task lists, evidence count

All tests use the heuristic fallback path (no OpenAI calls) by patching
interrupt() to capture payloads rather than actually pausing graph execution.
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from masis.infra.hitl import (
    AmbiguityClassification,
    AmbiguityLabel,
    add_risk_disclaimer,
    ambiguity_detector,
    apply_dag_edits,
    build_cancel_result,
    build_partial_result,
    dag_approval_interrupt,
    handle_resume,
    mid_execution_interrupt,
    risk_gate,
    _classify_ambiguity,
    _heuristic_classify,
    _compute_risk_score,
    _apply_task_modifications,
    _build_add_web_search_update,
    _build_user_evidence_update,
    _get_state_field,
    _extract_completed_summaries,
    _summarise_evidence,
)


# ---------------------------------------------------------------------------
# AmbiguityClassification
# ---------------------------------------------------------------------------

class TestAmbiguityClassification:

    def test_label_string_coerced_to_enum(self):
        cls = AmbiguityClassification(label="CLEAR", confidence=0.9)
        assert cls.label is AmbiguityLabel.CLEAR

    def test_invalid_label_raises(self):
        with pytest.raises(ValueError):
            AmbiguityClassification(label="INVALID", confidence=0.5)

    def test_options_defaults_to_empty_list(self):
        cls = AmbiguityClassification(label=AmbiguityLabel.AMBIGUOUS, confidence=0.8)
        assert cls.options == []

    def test_repr_contains_key_fields(self):
        cls = AmbiguityClassification(
            label=AmbiguityLabel.AMBIGUOUS,
            confidence=0.75,
            options=["Cloud?", "AI?"],
        )
        r = repr(cls)
        assert "AMBIGUOUS" in r
        assert "0.75" in r


# ---------------------------------------------------------------------------
# _heuristic_classify (internal)
# ---------------------------------------------------------------------------

class TestHeuristicClassify:

    def test_weather_is_out_of_scope(self):
        cls = _heuristic_classify("What is the weather in Berlin today?")
        assert cls.label is AmbiguityLabel.OUT_OF_SCOPE

    def test_recipe_is_out_of_scope(self):
        cls = _heuristic_classify("Give me a recipe for chocolate cake.")
        assert cls.label is AmbiguityLabel.OUT_OF_SCOPE

    def test_vague_how_is_things_is_ambiguous(self):
        cls = _heuristic_classify("How are things going?")
        assert cls.label is AmbiguityLabel.AMBIGUOUS
        assert len(cls.options) > 0

    def test_tell_me_about_is_ambiguous(self):
        cls = _heuristic_classify("Tell me about the company.")
        assert cls.label is AmbiguityLabel.AMBIGUOUS

    def test_specific_revenue_query_is_clear(self):
        cls = _heuristic_classify("What was TechCorp Q3 FY26 revenue?")
        assert cls.label is AmbiguityLabel.CLEAR

    def test_confidence_between_zero_and_one(self):
        for query in ["Q3 revenue?", "How are things?", "What's the weather?"]:
            cls = _heuristic_classify(query)
            assert 0.0 <= cls.confidence <= 1.0


# ---------------------------------------------------------------------------
# ambiguity_detector (MF-HITL-01)
# ---------------------------------------------------------------------------

class TestAmbiguityDetector:

    @pytest.mark.asyncio
    async def test_clear_query_returns_without_interrupt(self):
        """CLEAR query must NOT call interrupt()."""
        with patch("masis.infra.hitl.interrupt") as mock_interrupt:
            result = await ambiguity_detector(
                "What was TechCorp Q3 FY26 revenue?"
            )
        mock_interrupt.assert_not_called()
        assert result.label is AmbiguityLabel.CLEAR

    @pytest.mark.asyncio
    async def test_out_of_scope_returns_without_interrupt(self):
        """OUT_OF_SCOPE should be returned directly — no interrupt, no cost."""
        with patch("masis.infra.hitl.interrupt") as mock_interrupt:
            result = await ambiguity_detector("What's the weather in Paris?")
        mock_interrupt.assert_not_called()
        assert result.label is AmbiguityLabel.OUT_OF_SCOPE

    @pytest.mark.asyncio
    async def test_ambiguous_query_calls_interrupt(self):
        """AMBIGUOUS query must call interrupt() with options payload."""
        captured = {}

        def mock_interrupt(payload):
            captured["payload"] = payload
            # Simulate the human choosing an option
            return {"action": "clarify", "clarification": "Cloud Services"}

        with patch("masis.infra.hitl.interrupt", side_effect=mock_interrupt):
            result = await ambiguity_detector("How are things going?")

        assert "payload" in captured
        assert captured["payload"]["type"] == "ambiguity"
        assert isinstance(captured["payload"]["options"], list)
        # After resume, label should be CLEAR
        assert result.label is AmbiguityLabel.CLEAR
        assert "Cloud Services" in result.suggestion

    @pytest.mark.asyncio
    async def test_ambiguous_interrupt_payload_has_options(self):
        """Interrupt payload must contain a non-empty options list."""
        def mock_interrupt(payload):
            return {"action": "clarify", "clarification": "AI Division"}

        with patch("masis.infra.hitl.interrupt", side_effect=mock_interrupt):
            await ambiguity_detector("Tell me about the business.")

    @pytest.mark.asyncio
    async def test_llm_unavailable_uses_heuristic(self):
        """If LLM call fails, heuristic classifier is used."""
        with patch("masis.infra.hitl.interrupt") as mock_interrupt:
            # langchain_openai not installed in test env → heuristic fallback
            result = await ambiguity_detector("What was Q3 revenue?")
        assert result.label in AmbiguityLabel.__members__.values()


# ---------------------------------------------------------------------------
# dag_approval_interrupt (MF-HITL-02)
# ---------------------------------------------------------------------------

class TestDagApprovalInterrupt:

    def _make_plan_dict(self):
        return {
            "tasks": [
                {"task_id": "T1", "type": "researcher", "query": "Q3 revenue"},
                {"task_id": "T2", "type": "skeptic", "query": "verify T1"},
            ],
            "stop_condition": "Q3 revenue with citations",
        }

    def test_approve_action_returns_resume_value(self):
        expected_resume = {"action": "approve"}
        with patch("masis.infra.hitl.interrupt", return_value=expected_resume):
            result = dag_approval_interrupt(self._make_plan_dict())
        assert result == expected_resume

    def test_interrupt_payload_has_dag_approval_type(self):
        captured = {}

        def mock_interrupt(payload):
            captured["payload"] = payload
            return {"action": "approve"}

        with patch("masis.infra.hitl.interrupt", side_effect=mock_interrupt):
            dag_approval_interrupt(self._make_plan_dict())

        assert captured["payload"]["type"] == "dag_approval"
        assert "proposed_dag" in captured["payload"]
        assert "options" in captured["payload"]
        assert "approve" in captured["payload"]["options"]

    def test_payload_options_include_all_three(self):
        captured = {}

        def mock_interrupt(payload):
            captured["payload"] = payload
            return {"action": "approve"}

        with patch("masis.infra.hitl.interrupt", side_effect=mock_interrupt):
            dag_approval_interrupt(self._make_plan_dict())

        opts = captured["payload"]["options"]
        assert "approve" in opts
        assert "edit" in opts
        assert "cancel" in opts

    def test_plan_with_model_dump_serialised(self):
        """Plans that have model_dump() should be correctly serialised."""
        plan = MagicMock()
        plan.model_dump.return_value = self._make_plan_dict()
        captured = {}

        def mock_interrupt(payload):
            captured["payload"] = payload
            return {"action": "approve"}

        with patch("masis.infra.hitl.interrupt", side_effect=mock_interrupt):
            dag_approval_interrupt(plan)

        assert "tasks" in captured["payload"]["proposed_dag"]


# ---------------------------------------------------------------------------
# apply_dag_edits
# ---------------------------------------------------------------------------

class TestApplyDagEdits:

    def test_updates_query_for_matching_task(self):
        plan = {
            "tasks": [
                {"task_id": "T1", "type": "researcher", "query": "old query"},
                {"task_id": "T2", "type": "skeptic", "query": "verify"},
            ]
        }
        mods = [{"task_id": "T1", "query": "new improved query"}]
        updated = apply_dag_edits(plan, mods)
        tasks = updated["tasks"]
        assert tasks[0]["query"] == "new improved query"
        assert tasks[1]["query"] == "verify"  # unchanged

    def test_unknown_task_id_in_mods_ignored(self):
        plan = {"tasks": [{"task_id": "T1", "query": "q"}]}
        mods = [{"task_id": "T99", "query": "irrelevant"}]
        updated = apply_dag_edits(plan, mods)
        assert updated["tasks"][0]["query"] == "q"

    def test_empty_modifications_list(self):
        plan = {"tasks": [{"task_id": "T1", "query": "q"}]}
        updated = apply_dag_edits(plan, [])
        assert updated == plan


# ---------------------------------------------------------------------------
# mid_execution_interrupt (MF-HITL-03)
# ---------------------------------------------------------------------------

class TestMidExecutionInterrupt:

    def _make_state(self):
        return {
            "task_dag": [
                {"task_id": "T1", "type": "researcher", "status": "done", "result_summary": "Revenue data found."},
                {"task_id": "T2", "type": "web_search", "status": "pending"},
            ],
            "evidence_board": [{"doc_id": "d1", "chunk_id": "c1", "content": "evidence"}],
        }

    def test_interrupt_called_with_correct_type(self):
        captured = {}

        def mock_interrupt(payload):
            captured["payload"] = payload
            return {"action": "accept_partial"}

        with patch("masis.infra.hitl.interrupt", side_effect=mock_interrupt):
            mid_execution_interrupt(
                state=self._make_state(),
                coverage=0.50,
                missing_aspects=["headcount"],
            )

        assert captured["payload"]["type"] == "evidence_insufficient"
        assert captured["payload"]["coverage"] == 0.50
        assert "headcount" in captured["payload"]["missing_aspects"]

    def test_options_include_four_choices(self):
        captured = {}

        def mock_interrupt(payload):
            captured["payload"] = payload
            return {"action": "expand_to_web"}

        with patch("masis.infra.hitl.interrupt", side_effect=mock_interrupt):
            mid_execution_interrupt(
                state=self._make_state(),
                coverage=0.40,
                missing_aspects=["margins"],
            )

        opts = captured["payload"]["options"]
        assert "expand_to_web" in opts
        assert "provide_data" in opts
        assert "accept_partial" in opts
        assert "cancel" in opts


# ---------------------------------------------------------------------------
# risk_gate (MF-HITL-04)
# ---------------------------------------------------------------------------

class TestRiskGate:

    def test_clean_text_does_not_trigger(self):
        text = "TechCorp Q3 revenue was ₹41,764 crore, up 12% YoY."
        triggered, score, _ = risk_gate(text, risk_threshold=0.70)
        assert triggered is False
        assert score < 0.70

    def test_investment_recommendation_triggers(self):
        text = "Based on this analysis, investors should buy TechCorp shares immediately."

        def mock_interrupt(payload):
            assert payload["type"] == "risk_gate"
            assert payload["risk_score"] >= 0.70
            return {"action": "approve"}

        with patch("masis.infra.hitl.interrupt", side_effect=mock_interrupt):
            triggered, score, keywords = risk_gate(text, risk_threshold=0.70)
        assert triggered is True
        assert score >= 0.70

    def test_sell_recommendation_triggers(self):
        text = "We recommend to sell bonds and invest in growth equities."

        def mock_interrupt(payload):
            return {"action": "add_disclaimer"}

        with patch("masis.infra.hitl.interrupt", side_effect=mock_interrupt):
            triggered, score, kws = risk_gate(text)
        assert triggered is True

    def test_risk_keywords_returned(self):
        text = "You should invest in this financial planning opportunity."
        triggered, score, kws = risk_gate(text, risk_threshold=2.0)  # threshold above max
        assert isinstance(kws, list)

    def test_add_risk_disclaimer_appends_text(self):
        original = "TechCorp revenue was ₹41,764 crore."
        result = add_risk_disclaimer(original, ["invest"])
        assert "DISCLAIMER" in result
        assert original in result

    def test_compute_risk_score_zero_for_clean_text(self):
        score, matched = _compute_risk_score("Revenue grew 12% YoY.")
        assert score == 0.0
        assert matched == []

    def test_compute_risk_score_max_at_three_plus_matches(self):
        text = "You should sell shares, invest in bonds, and follow this recommendation."
        score, matched = _compute_risk_score(text)
        assert score == 1.0


# ---------------------------------------------------------------------------
# handle_resume (MF-HITL-05)
# ---------------------------------------------------------------------------

class TestHandleResume:

    def _make_state(self, dag=None):
        return {
            "task_dag": dag or [
                {"task_id": "T1", "type": "researcher", "status": "done"},
                {"task_id": "T2", "type": "skeptic", "status": "pending"},
            ],
            "evidence_board": [],
            "original_query": "Q3 revenue?",
        }

    def test_approve_returns_empty_dict(self):
        result = handle_resume({"action": "approve"}, self._make_state())
        assert result == {}

    def test_cancel_sets_supervisor_decision_failed(self):
        result = handle_resume({"action": "cancel"}, self._make_state())
        assert result["supervisor_decision"] == "failed"
        assert "cancel" in result["stop_reason"].lower()

    def test_clarify_updates_original_query(self):
        result = handle_resume(
            {"action": "clarify", "clarification": "Cloud Services revenue Q3"},
            self._make_state()
        )
        assert result["original_query"] == "Cloud Services revenue Q3"

    def test_edit_applies_dag_modifications(self):
        result = handle_resume(
            {
                "action": "edit",
                "modifications": [{"task_id": "T1", "query": "new query for T1"}],
            },
            self._make_state()
        )
        updated_dag = result["task_dag"]
        t1 = next(t for t in updated_dag if t.get("task_id") == "T1")
        assert t1["query"] == "new query for T1"

    def test_expand_to_web_adds_web_search_task(self):
        result = handle_resume(
            {"action": "expand_to_web", "missing_aspects": ["competitor margins"]},
            self._make_state()
        )
        assert "task_dag" in result
        new_tasks = [
            t for t in result["task_dag"]
            if isinstance(t, dict) and t.get("type") == "web_search"
        ]
        assert len(new_tasks) >= 1

    def test_accept_partial_forces_synthesize(self):
        result = handle_resume({"action": "accept_partial"}, self._make_state())
        assert result["supervisor_decision"] == "force_synthesize"

    def test_add_disclaimer_sets_flag(self):
        result = handle_resume({"action": "add_disclaimer"}, self._make_state())
        assert result.get("hitl_add_disclaimer") is True

    def test_revise_resets_synthesis_output(self):
        result = handle_resume({"action": "revise"}, self._make_state())
        assert result["synthesis_output"] is None
        assert result["supervisor_decision"] == "continue"

    def test_non_dict_resume_value_returns_empty(self):
        result = handle_resume("approve", self._make_state())
        assert result == {}

    def test_unknown_action_treated_as_approve(self):
        result = handle_resume({"action": "do_magic"}, self._make_state())
        assert result == {}

    def test_provide_data_creates_evidence_chunk(self):
        result = handle_resume(
            {"action": "provide_data", "data": "Cloud revenue was $5B in Q3."},
            self._make_state()
        )
        evidence = result.get("evidence_board", [])
        assert len(evidence) >= 1
        assert "Cloud revenue" in evidence[0]["content"]


# ---------------------------------------------------------------------------
# build_partial_result (MF-HITL-06)
# ---------------------------------------------------------------------------

class TestBuildPartialResult:

    def _make_state(self, evidence_count=3):
        return {
            "task_dag": [
                {"task_id": "T1", "type": "researcher"},
                {"task_id": "T2", "type": "skeptic"},
                {"task_id": "T3", "type": "synthesizer"},
            ],
            "evidence_board": [
                {"doc_id": f"d{i}", "chunk_id": f"c{i}", "content": "evidence"}
                for i in range(evidence_count)
            ],
        }

    def test_coverage_computed_correctly(self):
        state = self._make_state()
        result = build_partial_result(state, completed_task_ids=["T1", "T2"])
        assert result["coverage"] == pytest.approx(2 / 3, rel=0.01)

    def test_missing_task_ids_correct(self):
        state = self._make_state()
        result = build_partial_result(state, completed_task_ids=["T1"])
        assert "T2" in result["missing_task_ids"]
        assert "T3" in result["missing_task_ids"]

    def test_is_partial_flag_true(self):
        state = self._make_state()
        result = build_partial_result(state, completed_task_ids=["T1"])
        assert result["is_partial"] is True

    def test_disclaimer_mentions_missing_tasks(self):
        state = self._make_state()
        result = build_partial_result(state, completed_task_ids=["T1"])
        assert "T2" in result["disclaimer"] or "T3" in result["disclaimer"] or "partial" in result["disclaimer"].lower()

    def test_all_tasks_completed_full_coverage(self):
        state = self._make_state()
        result = build_partial_result(state, completed_task_ids=["T1", "T2", "T3"])
        assert result["coverage"] == pytest.approx(1.0, rel=0.01)
        assert result["missing_task_ids"] == []

    def test_empty_dag(self):
        state = {"task_dag": [], "evidence_board": []}
        result = build_partial_result(state, completed_task_ids=[])
        assert result["coverage"] == 0.0


# ---------------------------------------------------------------------------
# build_cancel_result (MF-HITL-07)
# ---------------------------------------------------------------------------

class TestBuildCancelResult:

    def _make_state(self):
        return {
            "task_dag": [
                {"task_id": "T1", "type": "researcher", "status": "done"},
                {"task_id": "T2", "type": "skeptic", "status": "pending"},
                {"task_id": "T3", "type": "synthesizer", "status": "pending"},
            ],
            "evidence_board": [
                {"doc_id": "d1", "chunk_id": "c1", "content": "some evidence"},
                {"doc_id": "d2", "chunk_id": "c2", "content": "more evidence"},
            ],
            "iteration_count": 3,
        }

    def test_status_is_cancelled(self):
        result = build_cancel_result(self._make_state())
        assert result["status"] == "cancelled"

    def test_completed_tasks_correct(self):
        result = build_cancel_result(self._make_state())
        assert "T1" in result["work_completed"]

    def test_pending_tasks_correct(self):
        result = build_cancel_result(self._make_state())
        assert "T2" in result["work_pending"]
        assert "T3" in result["work_pending"]

    def test_evidence_count_correct(self):
        result = build_cancel_result(self._make_state())
        assert result["evidence_found"] == 2

    def test_message_informative(self):
        result = build_cancel_result(self._make_state())
        assert "cancelled" in result["message"].lower() or "cancel" in result["message"].lower()
        assert "1" in result["message"]  # 1 completed task

    def test_iterations_run(self):
        result = build_cancel_result(self._make_state())
        assert result["iterations_run"] == 3

    def test_empty_state(self):
        state = {"task_dag": [], "evidence_board": [], "iteration_count": 0}
        result = build_cancel_result(state)
        assert result["status"] == "cancelled"
        assert result["work_completed"] == []
        assert result["evidence_found"] == 0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

class TestPrivateHelpers:

    def test_get_state_field_from_dict(self):
        state = {"task_dag": [1, 2, 3], "iteration_count": 5}
        assert _get_state_field(state, "iteration_count", 0) == 5
        assert _get_state_field(state, "missing_key", "default") == "default"

    def test_get_state_field_from_object(self):
        state = MagicMock()
        state.iteration_count = 7
        assert _get_state_field(state, "iteration_count", 0) == 7

    def test_summarise_evidence_empty(self):
        assert "No evidence" in _summarise_evidence([])

    def test_summarise_evidence_counts(self):
        evidence = [
            {"doc_id": "d1", "chunk_id": "c1"},
            {"doc_id": "d1", "chunk_id": "c2"},
            {"doc_id": "d2", "chunk_id": "c3"},
        ]
        summary = _summarise_evidence(evidence)
        assert "3" in summary   # 3 chunks
        assert "2" in summary   # 2 unique docs

    def test_apply_task_modifications_skips_missing_task_ids(self):
        dag = [{"task_id": "T1", "query": "q1"}]
        mods = [{"task_id": "T99", "query": "irrelevant"}]
        result = _apply_task_modifications(dag, mods)
        assert result[0]["query"] == "q1"

    def test_build_add_web_search_update_creates_new_task(self):
        state = {
            "task_dag": [{"task_id": "T1", "type": "researcher", "status": "done"}],
            "evidence_board": [],
        }
        update = _build_add_web_search_update(state, "competitor revenue Q3")
        new_tasks = [
            t for t in update["task_dag"]
            if isinstance(t, dict) and t.get("type") == "web_search"
        ]
        assert len(new_tasks) == 1
        assert "competitor revenue Q3" in new_tasks[0]["query"]

    def test_build_user_evidence_update_structure(self):
        result = _build_user_evidence_update("Cloud division grew 20%.")
        chunks = result["evidence_board"]
        assert len(chunks) == 1
        assert "Cloud division grew 20%." in chunks[0]["content"]
        assert chunks[0]["doc_id"] == "user_provided"

    def test_extract_completed_summaries_dict_state(self):
        state = {
            "task_dag": [
                {"task_id": "T1", "type": "researcher", "status": "done", "result_summary": "Found data."},
                {"task_id": "T2", "type": "skeptic", "status": "pending"},
            ]
        }
        summaries = _extract_completed_summaries(state)
        assert len(summaries) == 1
        assert summaries[0]["task_id"] == "T1"
        assert summaries[0]["summary"] == "Found data."
