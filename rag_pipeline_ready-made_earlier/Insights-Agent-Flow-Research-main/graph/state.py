"""
Graph State - shared state definition for the LangGraph workflow.

All nodes read from and write to this TypedDict. The state flows through:
Router -> Rewrite -> Extract Metadata -> Retrieve -> Rerank -> Grade ->
Generate -> Hallucination Check -> Enrich (Charts + Reports)
"""

from typing import TypedDict, List, Dict, Any, Optional


class GraphState(TypedDict):
    """State passed between all nodes in the agent graph."""

    # Question fields
    question: str                       # Current question (may be rewritten)
    original_question: str              # Original question as typed by user
    rewritten_question: str             # After the rewrite step

    # Routing
    route: str                          # "retrieve" | "clarify"

    # Metadata
    query_metadata: Dict[str, Any]      # Extracted from query: year, quarter, dept, topic
    metadata_filters: List[str]         # Active filter keys (relaxed progressively)
    metadata_retries: int               # Fallback counter

    # Documents
    documents: List[Dict[str, Any]]     # Retrieved docs after hybrid search
    reranked_documents: List[Dict[str, Any]]  # After cross-encoder reranking

    # Generation
    generation: str                     # Final answer text

    # Enrichment
    chart_data: Dict[str, Any]          # Chart/graph data for visualization

    # CRAG loop
    doc_grading_retries: int            # Number of CRAG rewrites done
    irrelevancy_reason: str             # Why docs were filtered out

    # Self-RAG loop
    hallucination_retries: int          # Number of regeneration attempts
    answer_contains_hallucinations: bool  # Hallucination flag
