"""
Fallback Node — Graceful failure when retrieval/generation loops are exhausted.
"""

import logging

logger = logging.getLogger(__name__)

import datetime as datetime
todays_date = datetime.datetime.now().strftime("%B %Y")

def fallback_response(state: dict) -> dict:
    """
    Return a graceful "insufficient information" response.
    
    Triggered when:
    - CRAG loop exceeds MAX_DOC_GRADING_RETRIES
    - Self-RAG loop exceeds MAX_HALLUCINATION_RETRIES
    """
    question = state.get("original_question", state.get("question", ""))
    doc_retries = state.get("doc_grading_retries", 0)
    hall_retries = state.get("hallucination_retries", 0)
    
    if hall_retries > 0:
        reason = (
            "I was unable to generate a sufficiently reliable answer. "
            "The answer I generated could not be fully verified against the source documents "
            "after multiple attempts."
        )
    elif doc_retries > 0:
        reason = (
            "I was unable to find enough relevant information in the available documents "
            "to answer this question reliably, even after reformulating the search query."
        )
    else:
        reason = "I encountered an issue processing this question."
    
    logger.warning(f"Fallback triggered for: '{question}' "
                  f"(doc_retries={doc_retries}, hall_retries={hall_retries})")
    
    return {
        "generation": (
            f"I'm sorry, but I couldn't confidently answer your question: "
            f'"{question}"\n\n'
            f"**Reason:** {reason}\n\n"
            f"**Suggestions:**\n"
            f"- Try rephrasing your question with more specific details\n"
            f"- Specify a time period (e.g., 'Q3 2024') or department\n"
            f"- Check that the relevant documents have been uploaded"
        )
    }
