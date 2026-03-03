"""
Retrieve Node - Runs hybrid search (Vector + BM25 + RRF) with metadata filtering.

Handles progressive metadata fallback: if the current filter set returns 0 results,
filters are dropped one at a time until results are found or no filters remain.
"""

import logging
from copy import copy

import config
from ingestion.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

# Lazy initialization - retriever loads ChromaDB + BM25 index
_retriever = None

import datetime as datetime
todays_date = datetime.datetime.now().strftime("%B %Y")

def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def retrieve_documents(state: dict) -> dict:
    """
    Retrieve documents using hybrid search with metadata filtering.

    Progressive fallback: if 0 results with filters, drops one filter at a time
    and retries until results are found or all filters are removed.
    """
    question = state.get("question", "")
    query_metadata = state.get("query_metadata", {})
    metadata_filters = state.get("metadata_filters", [])

    retriever = _get_retriever()

    # Progressive metadata fallback: try with full filters, then drop one at a time
    current_filters = copy(metadata_filters)

    if not config.METADATA_PROGRESSIVE_FALLBACK:
        results = retriever.search(
            query=question,
            query_metadata=query_metadata,
            metadata_filters=current_filters if current_filters else None,
            top_k=config.TOP_K_RETRIEVAL,
        )
    else:
        while True:
            results = retriever.search(
                query=question,
                query_metadata=query_metadata,
                metadata_filters=current_filters if current_filters else None,
                top_k=config.TOP_K_RETRIEVAL,
            )

            if results or not current_filters:
                # Either we got results or we have no more filters to drop
                break

            # Drop the last filter and retry
            removed = current_filters.pop()
            logger.info(f"Metadata fallback: dropped '{removed}' filter, retrying ({len(current_filters)} filters left)")

    logger.info(f"Retrieved {len(results)} documents (filters: {current_filters})")

    return {
        "documents": results,
        "metadata_filters": current_filters,
    }
