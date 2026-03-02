"""
masis.agents.web_search
=======================
Web search agent — Tavily integration with content sanitisation (ENG-10, MF-EXE-03, MF-SAFE-04).

Responsibilities
----------------
- Call the Tavily Search API with the task query.
- Parse results into EvidenceChunk format for the evidence board.
- Apply the content sanitiser to strip prompt-injection patterns (MF-SAFE-04).
- Handle timeout (15s) and no-results edge cases gracefully.
- Return a normalised AgentOutput usable by the Executor (MF-EXE-06).

Public API
----------
run_web_search(task, state) → AgentOutput
sanitize_content(text)      → str  (also used by safety infrastructure)
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase 0 schema imports with stubs
# ---------------------------------------------------------------------------

try:
    from masis.schemas.models import AgentOutput, EvidenceChunk
    from masis.schemas.thresholds import INJECTION_PATTERNS, MAX_WEB_RESULT_CHARS
except ImportError:
    logger.warning("masis.schemas not found — using stub types for web_search.py")

    class EvidenceChunk:  # type: ignore[no-redef]
        def __init__(self, **kw: Any) -> None:
            for k, v in kw.items():
                setattr(self, k, v)
        chunk_id: str = ""
        doc_id: str = ""
        text: str = ""
        retrieval_score: float = 0.0
        rerank_score: float = 0.0
        metadata: dict = {}
        source_label: str = ""

    class AgentOutput:  # type: ignore[no-redef]
        def __init__(self, **kw: Any) -> None:
            for k, v in kw.items():
                setattr(self, k, v)

    INJECTION_PATTERNS: list = [  # type: ignore[misc]
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
    MAX_WEB_RESULT_CHARS: int = 5_000

# ---------------------------------------------------------------------------
# Optional Tavily import
# ---------------------------------------------------------------------------

try:
    from tavily import TavilyClient  # type: ignore[import]
    _TAVILY_AVAILABLE = True
except ImportError:
    _TAVILY_AVAILABLE = False
    logger.warning(
        "tavily-python not installed — web search will return stub results. "
        "Install with: pip install tavily-python"
    )
    TavilyClient = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Compiled injection pattern regexes (built once at module load)
# ---------------------------------------------------------------------------

_COMPILED_INJECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in INJECTION_PATTERNS
]

# ---------------------------------------------------------------------------
# Module-level Tavily client cache
# ---------------------------------------------------------------------------

_tavily_client: Any = None

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT_S: float = 15.0
DEFAULT_MAX_RESULTS: int = 5


async def run_web_search(task: Any, state: Dict[str, Any]) -> AgentOutput:
    """Execute a Tavily web search for the task query and return AgentOutput.

    Steps:
        1. Extract query from task.query.
        2. Call Tavily API (with 15s effective timeout enforced by the Executor).
        3. Sanitise each result's content (MF-SAFE-04).
        4. Parse into EvidenceChunk objects.
        5. Return normalised AgentOutput.

    Args:
        task: TaskNode with .task_id and .query.
        state: Filtered state (only task is used; provided for interface consistency).

    Returns:
        AgentOutput. Never raises — timeout/error cases return failed AgentOutput.
    """
    task_id = getattr(task, "task_id", "unknown")
    query: str = getattr(task, "query", "")

    logger.info("WebSearch started: task_id=%s query='%.80s'", task_id, query)
    start_ts = time.monotonic()

    if not query.strip():
        logger.warning("WebSearch: empty query for task %s", task_id)
        return _error_output(task_id, "empty_query", "Web search query is empty.")

    # ── Tavily search ────────────────────────────────────────────────────────
    try:
        raw_results = await _call_tavily(query)
    except TimeoutError:
        elapsed = time.monotonic() - start_ts
        logger.error("WebSearch timeout after %.1fs for task %s", elapsed, task_id)
        return _error_output(task_id, "timeout", f"Tavily search timed out after {DEFAULT_TIMEOUT_S}s")
    except Exception as exc:
        elapsed = time.monotonic() - start_ts
        logger.error("WebSearch failed after %.1fs: %s", elapsed, exc, exc_info=True)
        return _error_output(task_id, str(type(exc).__name__), str(exc))

    # ── No results edge case ─────────────────────────────────────────────────
    if not raw_results:
        logger.info("WebSearch: no results for query '%.60s'", query)
        return AgentOutput(  # type: ignore[call-arg]
            task_id=task_id,
            agent_type="web_search",
            status="success",
            summary="Web search returned no results.",
            evidence=[],
            criteria_result={"relevant_results": 0, "timeout": False},
            tokens_used=0,
            cost_usd=0.0,
        )

    # ── Sanitise and convert to EvidenceChunks ──────────────────────────────
    chunks = _parse_results_to_chunks(task_id, raw_results)
    relevant_count = len(chunks)

    elapsed = time.monotonic() - start_ts
    logger.info(
        "WebSearch done: task_id=%s, results=%d, elapsed=%.2fs",
        task_id, relevant_count, elapsed,
    )

    summary_parts = [f"Web search found {relevant_count} result(s) for: '{query[:100]}'"]
    if chunks:
        summary_parts.append(f"Top result: {chunks[0].source_label} — {chunks[0].text[:150]}...")
    summary = " ".join(summary_parts)

    return AgentOutput(  # type: ignore[call-arg]
        task_id=task_id,
        agent_type="web_search",
        status="success",
        summary=summary,
        evidence=chunks,
        criteria_result={
            "relevant_results": relevant_count,
            "timeout": False,
        },
        tokens_used=0,   # Tavily does not consume LLM tokens
        cost_usd=0.0,    # Tavily costs handled separately via API pricing
    )


def sanitize_content(text: str) -> str:
    """Strip prompt-injection patterns from web content (MF-SAFE-04).

    Removes or replaces known injection trigger phrases that could hijack the LLM
    when web search results are included in the context.

    Patterns covered:
        - "ignore previous instructions"
        - "you are now a"
        - "system:"
        - "disregard all prior"
        - "forget everything"
        - "new instructions:"
        - "override your instructions"
        - "act as if you"
        - "pretend you are"
        - "jailbreak"
        - "do anything now"

    Args:
        text: Raw web content string.

    Returns:
        Sanitised text with injection patterns replaced by "[FILTERED]".
        Truncated to MAX_WEB_RESULT_CHARS characters.
    """
    if not text:
        return text

    sanitised = text
    for pattern in _COMPILED_INJECTION_PATTERNS:
        sanitised = pattern.sub("[FILTERED]", sanitised)

    # Hard truncation to prevent context overflow
    if len(sanitised) > MAX_WEB_RESULT_CHARS:
        sanitised = sanitised[:MAX_WEB_RESULT_CHARS]
        logger.debug("Web content truncated to %d chars", MAX_WEB_RESULT_CHARS)

    return sanitised


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

async def _call_tavily(query: str) -> List[Dict[str, Any]]:
    """Call the Tavily Search API and return raw result dicts.

    Uses asyncio.to_thread() for the synchronous Tavily client call.

    Args:
        query: The search query string.

    Returns:
        List of result dicts from Tavily, each containing url, title, content, score.

    Raises:
        TimeoutError: If the call exceeds DEFAULT_TIMEOUT_S.
        Exception: If the API call fails for any other reason.
    """
    import asyncio  # noqa: PLC0415

    if not _TAVILY_AVAILABLE or TavilyClient is None:
        logger.warning("Tavily unavailable — returning stub web search results")
        return _stub_results(query)

    client = _get_tavily_client()
    if client is None:
        logger.error("Tavily client could not be initialised — returning stub results")
        return _stub_results(query)

    def _sync_search() -> List[Dict[str, Any]]:
        """Blocking Tavily call, run in a thread pool."""
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=DEFAULT_MAX_RESULTS,
            include_raw_content=False,
        )
        return response.get("results", [])

    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(_sync_search),
            timeout=DEFAULT_TIMEOUT_S,
        )
        return results
    except asyncio.TimeoutError as te:
        raise TimeoutError(
            f"Tavily search timed out after {DEFAULT_TIMEOUT_S}s"
        ) from te


def _parse_results_to_chunks(
    task_id: str, results: List[Dict[str, Any]]
) -> List[EvidenceChunk]:
    """Convert Tavily result dicts to EvidenceChunk objects after sanitisation.

    Each result becomes one EvidenceChunk:
        - text   = sanitized content (or snippet if content is empty)
        - doc_id = URL of the source page
        - chunk_id = "<task_id>_web_<index>"
        - retrieval_score = Tavily's relevance score (0-1)
        - source_label = "source_title (url)"

    Args:
        task_id: The TaskNode's task_id (used to form chunk_ids).
        results: Raw Tavily result dicts.

    Returns:
        List of EvidenceChunk objects (sorted by retrieval_score descending).
    """
    chunks: List[EvidenceChunk] = []

    for i, result in enumerate(results):
        url: str = result.get("url", f"unknown_url_{i}")
        title: str = result.get("title", "")
        content: str = result.get("content", result.get("snippet", ""))
        score: float = float(result.get("score", 0.5))

        # Sanitise content before adding to evidence board (MF-SAFE-04)
        clean_content = sanitize_content(content)

        if not clean_content.strip():
            logger.debug("Skipping empty/fully-sanitised web result: %s", url)
            continue

        chunk = EvidenceChunk(  # type: ignore[call-arg]
            chunk_id=f"{task_id}_web_{i}",
            doc_id=url,
            parent_chunk_id=None,
            text=clean_content,
            retrieval_score=score,
            rerank_score=score,  # Tavily score serves as both retrieval and rerank proxy
            metadata={
                "url": url,
                "title": title,
                "source_type": "web_search",
                "tavily_score": score,
            },
            source_label=f"{title} ({url})" if title else url,
        )
        chunks.append(chunk)

    # Sort by relevance score (best first)
    chunks.sort(key=lambda c: c.retrieval_score, reverse=True)
    return chunks


def _error_output(task_id: str, error_type: str, detail: str) -> AgentOutput:
    """Build a failed AgentOutput for graceful error handling.

    Args:
        task_id: The TaskNode's task_id.
        error_type: Short error classification string.
        detail: Human-readable error description.

    Returns:
        AgentOutput with status="failed".
    """
    return AgentOutput(  # type: ignore[call-arg]
        task_id=task_id,
        agent_type="web_search",
        status="timeout" if error_type == "timeout" else "failed",
        summary=f"Web search failed: {error_type}",
        evidence=[],
        criteria_result={"relevant_results": 0, "timeout": error_type == "timeout"},
        tokens_used=0,
        cost_usd=0.0,
        error_detail=detail,
    )


def _get_tavily_client() -> Any:
    """Return a cached Tavily client, initialised with TAVILY_API_KEY from env."""
    global _tavily_client
    if _tavily_client is not None:
        return _tavily_client

    import os  # noqa: PLC0415
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        logger.error(
            "TAVILY_API_KEY not set in environment. "
            "Web search will return stub results."
        )
        return None

    try:
        _tavily_client = TavilyClient(api_key=api_key)
        logger.info("Tavily client initialised")
        return _tavily_client
    except Exception as exc:
        logger.error("Failed to initialise Tavily client: %s", exc)
        return None


def _stub_results(query: str) -> List[Dict[str, Any]]:
    """Return deterministic stub results when Tavily is unavailable.

    Used in development/testing to allow the pipeline to run without a live
    Tavily subscription. Each stub result clearly identifies itself as synthetic.

    Args:
        query: The original query (used in stub text for traceability).

    Returns:
        List of two stub result dicts.
    """
    return [
        {
            "url": "https://stub.masis.test/result/1",
            "title": f"Stub Result 1 for: {query[:60]}",
            "content": (
                f"[STUB] This is a synthetic web search result for the query: '{query}'. "
                "In production, this would be a real article from the web. "
                "Tavily API is not available in the current environment. "
                "Install tavily-python and set TAVILY_API_KEY to enable live search."
            ),
            "score": 0.75,
        },
        {
            "url": "https://stub.masis.test/result/2",
            "title": f"Stub Result 2 for: {query[:60]}",
            "content": (
                f"[STUB] Additional context for the query: '{query}'. "
                "This placeholder demonstrates that the web search agent correctly "
                "handles multiple results and parses them into EvidenceChunk format."
            ),
            "score": 0.60,
        },
    ]
