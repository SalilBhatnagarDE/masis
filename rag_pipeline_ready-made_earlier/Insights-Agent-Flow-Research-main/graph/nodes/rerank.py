"""
Rerank Node — Cross-encoder reranking using sentence-transformers.

Always runs after retrieval. Uses a cross-encoder model that sees query+document
together (unlike bi-encoder embeddings which encode independently), catching
relevance signals that vector similarity misses.
"""

import logging
from sentence_transformers import CrossEncoder

import config

logger = logging.getLogger(__name__)

# Lazy initialization of the cross-encoder model
_reranker = None

import datetime as datetime
todays_date = datetime.datetime.now().strftime("%B %Y")

def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(config.RERANKER_MODEL)
        logger.info(f"Loaded cross-encoder model: {config.RERANKER_MODEL}")
    return _reranker


def rerank_documents(state: dict) -> dict:
    """
    Rerank documents using cross-encoder model.
    
    Keeps top RERANK_TOP_N documents after reranking.
    Also resolves parent context (child->parent) for richer LLM context.
    """
    documents = state.get("documents", [])
    question = state.get("original_question", state.get("question", ""))
    
    if not documents:
        logger.warning("No documents to rerank")
        return {"reranked_documents": []}
    
    reranker = _get_reranker()
    
    # Extract text from documents for scoring
    doc_texts = [doc.get("text", "") for doc in documents]
    
    # Create query-document pairs for cross-encoder
    pairs = [(question, text) for text in doc_texts]
    
    # Score all pairs
    scores = reranker.predict(pairs)
    
    # Sort documents by cross-encoder score (highest first)
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Keep top N after reranking
    top_docs = [doc for doc, score in scored_docs[:config.RERANK_TOP_N]]
    
    logger.info(f"Reranked {len(documents)} -> {len(top_docs)} documents")
    
    # Resolve parent context (child->parent) for richer context
    from graph.nodes.retrieve import _get_retriever
    retriever = _get_retriever()
    top_docs = retriever.resolve_parent_context(top_docs)
    
    return {"reranked_documents": top_docs}
