"""
masis.agents
============
Agent implementations for the MAISS system (Phase 1, ENG-07 through ENG-10).

These are pure Python functions called by the Executor node — they are NOT
separate LangGraph graph nodes. The Executor dispatches them based on task.type.

Agents
------
researcher  — HyDE rewrite, hybrid retrieval (vector+BM25/RRF), cross-encoder reranking,
              parent-chunk expansion, CRAG grading, Self-RAG hallucination check.
skeptic     — Claim extraction, BART-MNLI NLI pre-filter, o3-mini LLM judge, reconciliation.
synthesizer — U-shape evidence ordering, critique integration, Pydantic citation enforcement,
              post-hoc NLI verification, partial/no-evidence edge cases.
web_search  — Tavily integration, content sanitisation.
"""

from masis.agents.researcher import run_researcher, hyde_rewrite
from masis.agents.skeptic import run_skeptic, extract_claims
from masis.agents.synthesizer import run_synthesizer
from masis.agents.web_search import run_web_search

__all__ = [
    "run_researcher",
    "hyde_rewrite",
    "run_skeptic",
    "extract_claims",
    "run_synthesizer",
    "run_web_search",
]
