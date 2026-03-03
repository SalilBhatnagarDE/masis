"""
Hallucination Check Edge — Self-RAG loop conditional edge.

After hallucination checking, decides:
- "end" -> No hallucination, answer is grounded -> return to user
- "retry" -> Hallucination detected, retry generation
- "fallback" -> Exhausted retries, give up gracefully
"""

import config


def check_hallucination_edge(state: dict) -> str:
    """
    Conditional edge: determine if the answer is grounded or needs retry.
    
    Returns:
        "end" | "retry" | "fallback"
    """
    has_hallucination = state.get("answer_contains_hallucinations", False)
    hallucination_retries = state.get("hallucination_retries", 0)
    
    if not has_hallucination:
        return "end"  # Answer is grounded — done!
    
    # Hallucination detected — can we retry?
    if hallucination_retries < config.MAX_HALLUCINATION_RETRIES:
        return "retry"  # Loop back to regenerate
    
    # Exhausted retries — fall back gracefully
    return "fallback"
