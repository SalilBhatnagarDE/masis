"""
Decide-to-Generate Edge — CRAG loop conditional edge.

After document grading, decides:
- "generate" -> Enough relevant docs, proceed to answer generation
- "rewrite" -> Not enough relevant docs, rewrite query and retry
- "fallback" -> Exhausted retries, give up gracefully
"""

import config


def decide_to_generate_edge(state: dict) -> str:
    """
    Conditional edge: determine if we have enough relevant docs to generate.
    
    Returns:
        "generate" | "rewrite" | "fallback"
    """
    documents = state.get("reranked_documents", [])
    doc_grading_retries = state.get("doc_grading_retries", 0)
    
    # Check if we have enough relevant documents
    if len(documents) >= config.DOCS_RELEVANCE_THRESHOLD:
        return "generate"
    
    # Not enough docs — can we retry?
    if doc_grading_retries < config.MAX_DOC_GRADING_RETRIES:
        return "rewrite"  # Loop back: rewrite query -> retrieve -> rerank -> grade again
    
    # Exhausted retries
    return "fallback"
