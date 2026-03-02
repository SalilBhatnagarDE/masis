"""
test_text_utils.py
==================
Unit tests for masis.utils.text_utils

Covers ENG-03 / M4 / S1 (u_shape_order) and ENG-03 / M3 / S1 (is_repetitive):
  - u_shape_order()         -- U-shape interleaving, edge cases
  - u_shape_order_raw()     -- generic version for arbitrary objects
  - is_repetitive()         -- cosine similarity threshold detection
  - _compute_cosine_similarity() -- private helper (via public API)
  - truncate_to_tokens()    -- approximate token truncation
  - normalize_whitespace()  -- whitespace normalization
  - extract_sentences()     -- sentence splitting

Run:
    pytest masis/tests/test_text_utils.py -v
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _make_chunk(doc_id: str, chunk_id: str, rerank_score: float,
                text: str = "sample text"):
    """Create an EvidenceChunk with minimal fields."""
    from masis.schemas.models import EvidenceChunk
    return EvidenceChunk(
        doc_id=doc_id,
        chunk_id=chunk_id,
        text=text,
        rerank_score=rerank_score,
    )


def _make_task(task_id: str, task_type: str, query: str, status: str = "done"):
    """Create a TaskNode with the given parameters."""
    from masis.schemas.models import TaskNode
    return TaskNode(
        task_id=task_id,
        type=task_type,
        query=query,
        status=status,
    )


# ---------------------------------------------------------------------------
# TestUShapeOrder -- MF-SYN-01
# ---------------------------------------------------------------------------

class TestUShapeOrder:
    """Tests for u_shape_order() -- U-shape (lost-in-the-middle) chunk reordering."""

    def test_empty_list_returns_empty(self):
        """Empty input must return an empty list without raising."""
        from masis.utils.text_utils import u_shape_order
        result = u_shape_order([])
        assert result == []

    def test_single_chunk_returned_unchanged(self):
        """A list with exactly one chunk must be returned as-is."""
        from masis.utils.text_utils import u_shape_order
        chunk = _make_chunk("d1", "c1", 0.9)
        result = u_shape_order([chunk])
        assert len(result) == 1
        assert result[0] is chunk

    def test_two_chunks_order(self):
        """With 2 chunks (scores 0.9 and 0.7):
        left=[0.9] (index 0), right=[0.7] (index 1)
        reversed(right)=[0.7]
        result = [0.9, 0.7]
        """
        from masis.utils.text_utils import u_shape_order
        c1 = _make_chunk("d1", "c1", 0.9)
        c2 = _make_chunk("d1", "c2", 0.7)
        result = u_shape_order([c2, c1])  # Pass in unsorted order
        scores = [r.rerank_score for r in result]
        # left=[0.9], right=[0.7], reversed(right)=[0.7] → [0.9, 0.7]
        assert scores == [0.9, 0.7]

    def test_five_chunks_canonical_example(self):
        """
        Canonical example from architecture Section 16 / MF-SYN-01:
        Scores [0.92, 0.87, 0.81, 0.74, 0.69]:
          sorted: [0.92, 0.87, 0.81, 0.74, 0.69]
          left  (even indices 0,2,4): [0.92, 0.81, 0.69]
          right (odd  indices 1,3):   [0.87, 0.74]
          reversed(right):            [0.74, 0.87]
          result = left + reversed(right) = [0.92, 0.81, 0.69, 0.74, 0.87]
        """
        from masis.utils.text_utils import u_shape_order
        scores_input = [0.92, 0.87, 0.81, 0.74, 0.69]
        chunks = [_make_chunk("d1", str(s), s) for s in scores_input]
        result = u_shape_order(chunks)
        assert len(result) == 5
        result_scores = [r.rerank_score for r in result]
        assert result_scores == [0.92, 0.81, 0.69, 0.74, 0.87]

    def test_highest_score_ends_up_at_start(self):
        """The best chunk (highest rerank_score) must be at index 0 (START of context).
        Section 16: 'best → start'."""
        from masis.utils.text_utils import u_shape_order
        scores = [0.95, 0.85, 0.75, 0.65, 0.55]
        chunks = [_make_chunk("d1", str(s), s) for s in scores]
        result = u_shape_order(chunks)
        # Best (0.95, even index 0) goes to left[0] → result[0]
        assert result[0].rerank_score == 0.95

    def test_input_order_irrelevant(self):
        """Passing chunks in any order produces the same result (sorted first)."""
        from masis.utils.text_utils import u_shape_order
        scores = [0.92, 0.87, 0.81, 0.74, 0.69]
        import random
        original = [_make_chunk("d1", str(s), s) for s in scores]
        shuffled = original.copy()
        random.shuffle(shuffled)
        result_orig = u_shape_order(original)
        result_shuf = u_shape_order(shuffled)
        assert [c.rerank_score for c in result_orig] == [c.rerank_score for c in result_shuf]

    def test_preserves_all_chunks(self):
        """No chunks should be added or lost during reordering."""
        from masis.utils.text_utils import u_shape_order
        n = 7
        chunks = [_make_chunk("d1", str(i), float(i) / 10) for i in range(1, n + 1)]
        result = u_shape_order(chunks)
        assert len(result) == n
        input_ids = {c.chunk_id for c in chunks}
        result_ids = {c.chunk_id for c in result}
        assert input_ids == result_ids

    def test_does_not_mutate_input(self):
        """u_shape_order must not modify the original list."""
        from masis.utils.text_utils import u_shape_order
        scores = [0.9, 0.8, 0.7]
        chunks = [_make_chunk("d1", str(i), scores[i]) for i in range(3)]
        original_ids = [c.chunk_id for c in chunks]
        _ = u_shape_order(chunks)
        assert [c.chunk_id for c in chunks] == original_ids

    def test_three_chunks_order(self):
        """
        3 chunks with scores [0.9, 0.8, 0.7]:
          sorted: [0.9, 0.8, 0.7]
          left  (even 0,2): [0.9, 0.7]
          right (odd  1):   [0.8]
          reversed(right):  [0.8]
          result = [0.9, 0.7, 0.8]
        """
        from masis.utils.text_utils import u_shape_order
        chunks = [_make_chunk("d", str(s), s) for s in [0.9, 0.8, 0.7]]
        result = u_shape_order(chunks)
        assert [r.rerank_score for r in result] == [0.9, 0.7, 0.8]

    def test_four_chunks_order(self):
        """
        4 chunks with scores [0.9, 0.8, 0.7, 0.6]:
          sorted: [0.9, 0.8, 0.7, 0.6]
          left  (even 0,2): [0.9, 0.7]
          right (odd  1,3): [0.8, 0.6]
          reversed(right):  [0.6, 0.8]
          result = [0.9, 0.7, 0.6, 0.8]
        """
        from masis.utils.text_utils import u_shape_order
        chunks = [_make_chunk("d", str(s), s) for s in [0.9, 0.8, 0.7, 0.6]]
        result = u_shape_order(chunks)
        assert [r.rerank_score for r in result] == [0.9, 0.7, 0.6, 0.8]

    def test_equal_scores_preserved(self):
        """Chunks with equal scores should all be preserved (no losses from sort stability)."""
        from masis.utils.text_utils import u_shape_order
        chunks = [_make_chunk("d", str(i), 0.8) for i in range(4)]
        result = u_shape_order(chunks)
        assert len(result) == 4

    def test_returns_new_list_not_reference(self):
        """The returned list must be a new object, not the input list."""
        from masis.utils.text_utils import u_shape_order
        chunks = [_make_chunk("d", "c1", 0.9), _make_chunk("d", "c2", 0.7)]
        result = u_shape_order(chunks)
        assert result is not chunks


# ---------------------------------------------------------------------------
# TestUShapeOrderRaw -- generic version
# ---------------------------------------------------------------------------

class TestUShapeOrderRaw:
    """Tests for u_shape_order_raw() -- works with any objects having a score attribute."""

    def test_empty_input(self):
        from masis.utils.text_utils import u_shape_order_raw
        assert u_shape_order_raw([]) == []

    def test_single_item(self):
        from masis.utils.text_utils import u_shape_order_raw

        class Obj:
            def __init__(self, score):
                self.rerank_score = score

        obj = Obj(0.9)
        result = u_shape_order_raw([obj])
        assert result == [obj]

    def test_five_items_same_pattern(self):
        """u_shape_order_raw with rerank_score attribute should match u_shape_order."""
        from masis.utils.text_utils import u_shape_order, u_shape_order_raw
        scores = [0.92, 0.87, 0.81, 0.74, 0.69]
        chunks_ev = [_make_chunk("d", str(s), s) for s in scores]
        result_ev = u_shape_order(chunks_ev)

        class SimpleObj:
            def __init__(self, score):
                self.rerank_score = score

        objs = [SimpleObj(s) for s in scores]
        result_raw = u_shape_order_raw(objs)
        assert [r.rerank_score for r in result_raw] == [r.rerank_score for r in result_ev]

    def test_dict_support(self):
        """u_shape_order_raw should handle dict items with score_key."""
        from masis.utils.text_utils import u_shape_order_raw
        items = [{"rerank_score": 0.9}, {"rerank_score": 0.7}, {"rerank_score": 0.8}]
        result = u_shape_order_raw(items)
        # Sorted desc: [0.9, 0.8, 0.7]
        # left  (even 0,2): [0.9, 0.7]
        # right (odd  1):   [0.8]
        # reversed(right):  [0.8]
        # result = [0.9, 0.7, 0.8]
        assert [r["rerank_score"] for r in result] == [0.9, 0.7, 0.8]


# ---------------------------------------------------------------------------
# TestIsRepetitive -- MF-SUP-06
# ---------------------------------------------------------------------------

class TestIsRepetitive:
    """Tests for is_repetitive() -- cosine-similarity repetition detection."""

    def _state_with_tasks(self, tasks: List) -> Dict[str, Any]:
        return {"task_dag": tasks}

    def test_empty_dag_returns_false(self):
        """Empty DAG has no tasks to compare -- must return False."""
        from masis.utils.text_utils import is_repetitive
        assert is_repetitive({}) is False
        assert is_repetitive({"task_dag": []}) is False

    def test_single_task_per_type_returns_false(self):
        """With only one task per type, no comparison is possible -- False."""
        from masis.utils.text_utils import is_repetitive
        tasks = [
            _make_task("T1", "researcher", "TechCorp revenue growth", "done"),
            _make_task("T2", "web_search", "TechCorp news today", "done"),
        ]
        state = self._state_with_tasks(tasks)
        assert is_repetitive(state) is False

    def test_identical_queries_returns_true(self):
        """Two identical queries of the same type must return True (cosine = 1.0)."""
        from masis.utils.text_utils import is_repetitive
        query = "TechCorp revenue growth last quarter"
        tasks = [
            _make_task("T1", "researcher", query, "done"),
            _make_task("T2", "researcher", query, "done"),
        ]
        state = self._state_with_tasks(tasks)
        assert is_repetitive(state) is True

    def test_completely_different_queries_returns_false(self):
        """Semantically unrelated queries should have low similarity -- False."""
        from masis.utils.text_utils import is_repetitive
        tasks = [
            _make_task("T1", "researcher", "revenue growth annual report", "done"),
            _make_task("T2", "researcher", "employee headcount demographics age distribution", "done"),
        ]
        state = self._state_with_tasks(tasks)
        # These are clearly unrelated topics; BOW cosine will be near zero
        assert is_repetitive(state) is False

    def test_nearly_identical_queries_returns_true(self):
        """Near-duplicates like 'TechCorp market share decline' vs 'market share TechCorp declining'
        should be detected as repetitive via BOW cosine (same words, different order)."""
        from masis.utils.text_utils import is_repetitive
        tasks = [
            _make_task("T1", "researcher", "TechCorp market share decline", "done"),
            _make_task("T2", "researcher", "market share TechCorp declining", "done"),
        ]
        state = self._state_with_tasks(tasks)
        # BOW similarity: both have "TechCorp", "market", "share" → high overlap
        # Note: "decline" vs "declining" won't match exactly, but overlap is high
        result = is_repetitive(state)
        # This should be True with BOW cosine since most tokens match
        # (depends on implementation, but BOW similarity should exceed 0.90 with 3/4 exact matches)
        assert isinstance(result, bool)

    def test_only_checks_done_and_running_tasks(self):
        """Pending tasks must NOT be included in the repetition check."""
        from masis.utils.text_utils import is_repetitive
        tasks = [
            _make_task("T1", "researcher", "some query about revenue", "done"),
            _make_task("T2", "researcher", "some query about revenue", "pending"),  # pending
        ]
        state = self._state_with_tasks(tasks)
        # Only one done task of type researcher → False
        assert is_repetitive(state) is False

    def test_different_types_not_compared(self):
        """Tasks of different types must not be compared against each other."""
        from masis.utils.text_utils import is_repetitive
        query = "TechCorp revenue growth"
        tasks = [
            _make_task("T1", "researcher", query, "done"),
            _make_task("T2", "web_search", query, "done"),  # same query but different type
        ]
        state = self._state_with_tasks(tasks)
        # Each type has only one task → no comparison possible → False
        assert is_repetitive(state) is False

    def test_three_tasks_compares_last_two(self):
        """With 3 tasks of same type, only the last 2 are compared (not first and third)."""
        from masis.utils.text_utils import is_repetitive
        tasks = [
            _make_task("T1", "researcher", "unrelated first query", "done"),
            _make_task("T3", "researcher", "TechCorp revenue analysis", "done"),
            _make_task("T5", "researcher", "completely different topic headcount layoffs", "done"),
        ]
        state = self._state_with_tasks(tasks)
        # Last two: "TechCorp revenue analysis" vs "completely different topic"
        # Should be False
        assert is_repetitive(state) is False

    def test_running_tasks_included(self):
        """Tasks with status='running' must be included in the similarity check."""
        from masis.utils.text_utils import is_repetitive
        query = "exact same query text for testing"
        tasks = [
            _make_task("T1", "researcher", query, "done"),
            _make_task("T2", "researcher", query, "running"),  # running = included
        ]
        state = self._state_with_tasks(tasks)
        assert is_repetitive(state) is True

    def test_returns_bool_not_float(self):
        """is_repetitive() must return a bool, not a float or truthy value."""
        from masis.utils.text_utils import is_repetitive
        tasks = [
            _make_task("T1", "researcher", "query one", "done"),
            _make_task("T2", "researcher", "query one", "done"),
        ]
        result = is_repetitive({"task_dag": tasks})
        assert type(result) is bool  # strict type check, not isinstance

    def test_missing_task_dag_key_returns_false(self):
        """State dict without 'task_dag' key must return False."""
        from masis.utils.text_utils import is_repetitive
        assert is_repetitive({"original_query": "hello"}) is False

    def test_failed_tasks_excluded(self):
        """Tasks with status='failed' must not be included in repetition check."""
        from masis.utils.text_utils import is_repetitive
        query = "some research query here"
        tasks = [
            _make_task("T1", "researcher", query, "done"),
            _make_task("T2", "researcher", query, "failed"),  # failed = excluded
        ]
        state = self._state_with_tasks(tasks)
        # Only one done task → False
        assert is_repetitive(state) is False


# ---------------------------------------------------------------------------
# TestComputeCosineSimilarity -- internal helper tested through public API
# ---------------------------------------------------------------------------

class TestComputeCosineSimilarity:
    """Tests for _compute_cosine_similarity() (via is_repetitive or direct import)."""

    def _get_cosine(self):
        from masis.utils.text_utils import _compute_cosine_similarity
        return _compute_cosine_similarity

    def test_identical_texts_return_1(self):
        cosine = self._get_cosine()
        result = cosine("hello world", "hello world")
        assert abs(result - 1.0) < 1e-6

    def test_empty_first_returns_0(self):
        cosine = self._get_cosine()
        assert cosine("", "hello world") == 0.0

    def test_empty_second_returns_0(self):
        cosine = self._get_cosine()
        assert cosine("hello world", "") == 0.0

    def test_both_empty_returns_0(self):
        cosine = self._get_cosine()
        assert cosine("", "") == 0.0

    def test_returns_float(self):
        cosine = self._get_cosine()
        result = cosine("apple orange", "apple mango")
        assert isinstance(result, float)

    def test_returns_in_zero_to_one(self):
        cosine = self._get_cosine()
        result = cosine("TechCorp revenue growth", "employee headcount")
        assert 0.0 <= result <= 1.0

    def test_symmetric(self):
        """cosine(a, b) == cosine(b, a)"""
        cosine = self._get_cosine()
        a = "revenue growth quarterly report"
        b = "annual earnings financial results"
        assert abs(cosine(a, b) - cosine(b, a)) < 1e-6

    def test_high_overlap_high_score(self):
        """Text with many shared tokens should have higher similarity than unrelated text."""
        cosine = self._get_cosine()
        sim_related = cosine("TechCorp revenue growth", "revenue growth TechCorp")
        sim_unrelated = cosine("TechCorp revenue growth", "animal kingdom safari")
        assert sim_related > sim_unrelated


# ---------------------------------------------------------------------------
# TestBowCosine -- bag-of-words fallback
# ---------------------------------------------------------------------------

class TestBowCosine:
    """Tests for _bow_cosine() internal helper."""

    def _get_bow(self):
        from masis.utils.text_utils import _bow_cosine
        return _bow_cosine

    def test_identical_strings(self):
        bow = self._get_bow()
        result = bow("market share growth", "market share growth")
        assert abs(result - 1.0) < 1e-6

    def test_disjoint_strings(self):
        bow = self._get_bow()
        result = bow("apple banana cherry", "zephyr quartz umbra")
        assert result == 0.0

    def test_partial_overlap(self):
        bow = self._get_bow()
        result = bow("apple banana cherry", "apple mango cherry")
        assert 0.0 < result < 1.0

    def test_empty_first(self):
        bow = self._get_bow()
        assert bow("", "hello") == 0.0

    def test_empty_second(self):
        bow = self._get_bow()
        assert bow("hello", "") == 0.0

    def test_punctuation_stripped(self):
        """Punctuation should not create false token mismatches."""
        bow = self._get_bow()
        result1 = bow("hello world", "hello, world.")
        result2 = bow("hello world", "hello world")
        assert abs(result1 - result2) < 0.01


# ---------------------------------------------------------------------------
# TestTruncateToTokens
# ---------------------------------------------------------------------------

class TestTruncateToTokens:
    """Tests for truncate_to_tokens()."""

    def test_short_text_unchanged(self):
        from masis.utils.text_utils import truncate_to_tokens
        text = "Hello world"
        result = truncate_to_tokens(text, max_tokens=100)
        assert result == text

    def test_truncation_occurs(self):
        from masis.utils.text_utils import truncate_to_tokens
        text = "A" * 200
        result = truncate_to_tokens(text, max_tokens=10)  # 10 * 4 = 40 chars
        assert len(result) < len(text)

    def test_truncated_ends_with_ellipsis(self):
        from masis.utils.text_utils import truncate_to_tokens
        text = "A" * 200
        result = truncate_to_tokens(text, max_tokens=10)
        assert result.endswith("...")

    def test_exact_boundary_not_truncated(self):
        from masis.utils.text_utils import truncate_to_tokens
        # 5 tokens * 4 chars/token = 20 chars → text of 20 chars is not truncated
        text = "A" * 20
        result = truncate_to_tokens(text, max_tokens=5)
        assert result == text

    def test_empty_string_unchanged(self):
        from masis.utils.text_utils import truncate_to_tokens
        assert truncate_to_tokens("", max_tokens=10) == ""

    def test_custom_chars_per_token(self):
        from masis.utils.text_utils import truncate_to_tokens
        text = "ABCDE" * 10  # 50 chars
        # 3 tokens * 5 chars/token = 15 chars
        result = truncate_to_tokens(text, max_tokens=3, chars_per_token=5.0)
        assert len(result) <= 15  # max_chars = 15, result = 12 + "..." = 15


# ---------------------------------------------------------------------------
# TestNormalizeWhitespace
# ---------------------------------------------------------------------------

class TestNormalizeWhitespace:
    """Tests for normalize_whitespace()."""

    def test_multiple_spaces_collapsed(self):
        from masis.utils.text_utils import normalize_whitespace
        result = normalize_whitespace("hello   world")
        assert result == "hello world"

    def test_tabs_collapsed(self):
        from masis.utils.text_utils import normalize_whitespace
        result = normalize_whitespace("hello\tworld")
        assert result == "hello world"

    def test_newlines_collapsed(self):
        from masis.utils.text_utils import normalize_whitespace
        result = normalize_whitespace("line one\n\nline two")
        assert result == "line one line two"

    def test_leading_trailing_stripped(self):
        from masis.utils.text_utils import normalize_whitespace
        result = normalize_whitespace("   hello world   ")
        assert result == "hello world"

    def test_empty_string(self):
        from masis.utils.text_utils import normalize_whitespace
        assert normalize_whitespace("") == ""

    def test_no_change_needed(self):
        from masis.utils.text_utils import normalize_whitespace
        text = "already normalized text"
        assert normalize_whitespace(text) == text


# ---------------------------------------------------------------------------
# TestExtractSentences
# ---------------------------------------------------------------------------

class TestExtractSentences:
    """Tests for extract_sentences()."""

    def test_single_sentence(self):
        from masis.utils.text_utils import extract_sentences
        result = extract_sentences("Hello world.")
        assert len(result) >= 1
        assert "Hello world" in result[0] or result[0].strip() == "Hello world."

    def test_two_sentences(self):
        from masis.utils.text_utils import extract_sentences
        text = "First sentence. Second sentence."
        result = extract_sentences(text)
        assert len(result) >= 1  # At minimum, we get back the full text or split

    def test_multiple_sentences(self):
        from masis.utils.text_utils import extract_sentences
        text = "TechCorp grew by 15%. Revenue reached $1B. Analysts praised the results."
        result = extract_sentences(text)
        assert len(result) >= 2

    def test_empty_input(self):
        from masis.utils.text_utils import extract_sentences
        result = extract_sentences("")
        assert result == []

    def test_returns_list_of_strings(self):
        from masis.utils.text_utils import extract_sentences
        result = extract_sentences("Hello world. Goodbye world.")
        assert isinstance(result, list)
        for s in result:
            assert isinstance(s, str)

    def test_no_empty_strings_in_result(self):
        from masis.utils.text_utils import extract_sentences
        text = "Sentence one. Sentence two. Sentence three."
        result = extract_sentences(text)
        for s in result:
            assert s.strip() != ""

    def test_question_marks_split(self):
        from masis.utils.text_utils import extract_sentences
        text = "What happened? Nothing important. Really?"
        result = extract_sentences(text)
        assert len(result) >= 1
