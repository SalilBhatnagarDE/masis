"""
test_thresholds.py
==================
Unit tests for masis.schemas.thresholds

Verifies all required constants exist, have correct types, and match the
values specified in the engineering tasks (ENG-01 / M3).

Run:
    pytest masis/tests/test_thresholds.py -v
"""

from __future__ import annotations

import pytest


class TestResearcherThresholds:
    def test_importable(self):
        from masis.schemas.thresholds import RESEARCHER_THRESHOLDS
        assert isinstance(RESEARCHER_THRESHOLDS, dict)

    def test_min_chunks_after_grading_equals_2(self):
        """ENG-01 M3 S1a: min_chunks_after_grading must be 2."""
        from masis.schemas.thresholds import RESEARCHER_THRESHOLDS
        assert RESEARCHER_THRESHOLDS["min_chunks_after_grading"] == 2

    def test_min_grading_pass_rate(self):
        from masis.schemas.thresholds import RESEARCHER_THRESHOLDS
        assert RESEARCHER_THRESHOLDS["min_grading_pass_rate"] == 0.30

    def test_required_self_rag_verdict(self):
        from masis.schemas.thresholds import RESEARCHER_THRESHOLDS
        assert RESEARCHER_THRESHOLDS["required_self_rag_verdict"] == "grounded"

    def test_crag_max_retries_is_1(self):
        from masis.schemas.thresholds import RESEARCHER_THRESHOLDS
        assert RESEARCHER_THRESHOLDS["crag_max_retries"] == 1

    def test_top_k_retrieval(self):
        from masis.schemas.thresholds import RESEARCHER_THRESHOLDS
        assert RESEARCHER_THRESHOLDS["top_k_retrieval"] == 10

    def test_top_k_after_rerank(self):
        from masis.schemas.thresholds import RESEARCHER_THRESHOLDS
        assert RESEARCHER_THRESHOLDS["top_k_after_rerank"] == 5


class TestSkepticThresholds:
    def test_importable(self):
        from masis.schemas.thresholds import SKEPTIC_THRESHOLDS
        assert isinstance(SKEPTIC_THRESHOLDS, dict)

    def test_max_unsupported_claims(self):
        """ENG-01 M3 S1b: max_unsupported_claims must be 2."""
        from masis.schemas.thresholds import SKEPTIC_THRESHOLDS
        assert SKEPTIC_THRESHOLDS["max_unsupported_claims"] == 2

    def test_max_contradicted_claims(self):
        from masis.schemas.thresholds import SKEPTIC_THRESHOLDS
        assert SKEPTIC_THRESHOLDS["max_contradicted_claims"] == 0

    def test_max_logical_gaps(self):
        from masis.schemas.thresholds import SKEPTIC_THRESHOLDS
        assert SKEPTIC_THRESHOLDS["max_logical_gaps"] == 3

    def test_min_confidence(self):
        from masis.schemas.thresholds import SKEPTIC_THRESHOLDS
        assert SKEPTIC_THRESHOLDS["min_confidence"] == 0.65

    def test_nli_contradiction_threshold(self):
        from masis.schemas.thresholds import SKEPTIC_THRESHOLDS
        assert SKEPTIC_THRESHOLDS["nli_contradiction_score_threshold"] == 0.80

    def test_min_issues_required(self):
        from masis.schemas.thresholds import SKEPTIC_THRESHOLDS
        assert SKEPTIC_THRESHOLDS["min_issues_required"] == 3


class TestValidatorThresholds:
    def test_importable(self):
        from masis.schemas.thresholds import VALIDATOR_THRESHOLDS
        assert isinstance(VALIDATOR_THRESHOLDS, dict)

    def test_min_faithfulness(self):
        """Validator threshold should match tuned config."""
        from masis.schemas.thresholds import VALIDATOR_THRESHOLDS
        assert VALIDATOR_THRESHOLDS["min_faithfulness"] == 0.00

    def test_min_citation_accuracy(self):
        from masis.schemas.thresholds import VALIDATOR_THRESHOLDS
        assert VALIDATOR_THRESHOLDS["min_citation_accuracy"] == 0.00

    def test_min_answer_relevancy(self):
        from masis.schemas.thresholds import VALIDATOR_THRESHOLDS
        assert VALIDATOR_THRESHOLDS["min_answer_relevancy"] == 0.02

    def test_min_dag_completeness(self):
        from masis.schemas.thresholds import VALIDATOR_THRESHOLDS
        assert VALIDATOR_THRESHOLDS["min_dag_completeness"] == 0.50


class TestSafetyLimits:
    def test_importable(self):
        from masis.schemas.thresholds import SAFETY_LIMITS
        assert isinstance(SAFETY_LIMITS, dict)

    def test_max_supervisor_turns(self):
        """Demo-speed profile: MAX_SUPERVISOR_TURNS is tightened to 10."""
        from masis.schemas.thresholds import SAFETY_LIMITS
        assert SAFETY_LIMITS["MAX_SUPERVISOR_TURNS"] == 10

    def test_max_wall_clock_seconds(self):
        from masis.schemas.thresholds import SAFETY_LIMITS
        assert SAFETY_LIMITS["MAX_WALL_CLOCK_SECONDS"] == 170

    def test_repetition_cosine_threshold(self):
        from masis.schemas.thresholds import SAFETY_LIMITS
        assert SAFETY_LIMITS["REPETITION_COSINE_THRESHOLD"] == 0.90

    def test_max_validation_rounds(self):
        from masis.schemas.thresholds import SAFETY_LIMITS
        assert SAFETY_LIMITS["MAX_VALIDATION_ROUNDS"] == 2


class TestBudgetLimits:
    def test_importable(self):
        from masis.schemas.thresholds import BUDGET_LIMITS
        assert isinstance(BUDGET_LIMITS, dict)

    def test_max_tokens(self):
        """ENG-01 M3 S1e: max_tokens must be 100,000."""
        from masis.schemas.thresholds import BUDGET_LIMITS
        assert BUDGET_LIMITS["max_tokens"] == 100_000

    def test_max_cost_usd(self):
        from masis.schemas.thresholds import BUDGET_LIMITS
        assert BUDGET_LIMITS["max_cost_usd"] == 0.50


class TestToolLimits:
    def test_importable(self):
        from masis.schemas.thresholds import TOOL_LIMITS
        assert isinstance(TOOL_LIMITS, dict)

    def test_covers_all_agent_types(self):
        from masis.schemas.thresholds import TOOL_LIMITS
        for agent_type in ("researcher", "web_search", "skeptic", "synthesizer"):
            assert agent_type in TOOL_LIMITS, f"TOOL_LIMITS missing '{agent_type}'"

    def test_researcher_limits(self):
        from masis.schemas.thresholds import TOOL_LIMITS
        r = TOOL_LIMITS["researcher"]
        assert r["max_parallel"] == 3
        assert r["max_total"] == 8
        assert r["timeout_s"] == 45

    def test_skeptic_limits(self):
        from masis.schemas.thresholds import TOOL_LIMITS
        s = TOOL_LIMITS["skeptic"]
        assert s["max_parallel"] == 1
        assert s["timeout_s"] == 120

    def test_synthesizer_limits(self):
        from masis.schemas.thresholds import TOOL_LIMITS
        s = TOOL_LIMITS["synthesizer"]
        assert s["max_parallel"] == 1
        assert s["timeout_s"] == 60

    def test_web_search_limits(self):
        from masis.schemas.thresholds import TOOL_LIMITS
        w = TOOL_LIMITS["web_search"]
        assert w["max_parallel"] == 2
        assert w["timeout_s"] == 15


class TestInjectionPatterns:
    def test_importable(self):
        from masis.schemas.thresholds import INJECTION_PATTERNS
        assert isinstance(INJECTION_PATTERNS, list)

    def test_at_least_8_patterns(self):
        from masis.schemas.thresholds import INJECTION_PATTERNS
        assert len(INJECTION_PATTERNS) >= 8, (
            f"Expected >= 8 injection patterns, got {len(INJECTION_PATTERNS)}. "
            "ENG-03 M5 S1c requires detection of at least 8 patterns."
        )

    def test_all_patterns_are_strings(self):
        from masis.schemas.thresholds import INJECTION_PATTERNS
        for pattern in INJECTION_PATTERNS:
            assert isinstance(pattern, str), f"Pattern {pattern!r} is not a string."

    def test_patterns_are_valid_regex(self):
        import re
        from masis.schemas.thresholds import INJECTION_PATTERNS
        for pattern in INJECTION_PATTERNS:
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                pytest.fail(f"Pattern {pattern!r} is not valid regex: {exc}")
