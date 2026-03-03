"""
Hybrid Retriever — Combines Vector Search + BM25 + Metadata Filtering + RRF Fusion.

This is the retrieval backbone of the system. It:
1. Accepts metadata filters (from query metadata extraction)
2. Runs ChromaDB vector search within filtered subset
3. Runs BM25 keyword search on the same corpus
4. Fuses results using Reciprocal Rank Fusion (RRF)
5. Supports progressive metadata fallback (relax filters if 0 results)
6. Resolves parent context (child->parent) for richer LLM context
"""

import json
import os
import logging
from typing import Optional
from copy import copy

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from rank_bm25 import BM25Okapi

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever combining vector search, BM25, and metadata filtering
    with RRF fusion and parent-context resolution.
    """
    
    def __init__(self):
        """Initialize the hybrid retriever with ChromaDB and BM25."""
        # ChromaDB client
        self.client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        
        self.embedding_fn = OpenAIEmbeddingFunction(
            api_key=config.OPENAI_API_KEY,
            model_name=config.EMBEDDING_MODEL,
        )
        
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            embedding_function=self.embedding_fn,
        )
        
        # Load parent store
        self.parent_store = {}
        if os.path.exists(config.PARENT_STORE_PATH):
            with open(config.PARENT_STORE_PATH, "r", encoding="utf-8") as f:
                self.parent_store = json.load(f)
        
        # Build BM25 index
        self._build_bm25_index()
    
    def _build_bm25_index(self):
        """Build BM25 index from all documents in the collection."""
        # Get all documents from ChromaDB
        results = self.collection.get(include=["documents", "metadatas"])
        
        self.bm25_ids = results["ids"]
        self.bm25_documents = results["documents"] or []
        self.bm25_metadatas = results["metadatas"] or []
        
        if self.bm25_documents:
            # Tokenize for BM25
            tokenized = [doc.lower().split() for doc in self.bm25_documents]
            self.bm25_index = BM25Okapi(tokenized)
        else:
            self.bm25_index = None
        
        logger.info(f"Built BM25 index with {len(self.bm25_documents)} documents")
    
    def _vector_search(
        self, query: str, top_k: int, metadata_filter: Optional[dict] = None
    ) -> list[dict]:
        """
        Run vector similarity search on ChromaDB.
        
        Args:
            query: Search query text.
            top_k: Number of results.
            metadata_filter: ChromaDB where clause dict.
            
        Returns:
            List of result dicts with id, text, metadata, score.
        """
        kwargs = {
            "query_texts": [query],
            "n_results": min(top_k, self.collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        
        if metadata_filter:
            kwargs["where"] = metadata_filter
        
        try:
            results = self.collection.query(**kwargs)
        except Exception as e:
            logger.warning(f"Vector search failed with filter {metadata_filter}: {e}")
            # Fallback: try without filter
            kwargs.pop("where", None)
            try:
                results = self.collection.query(**kwargs)
            except Exception as e2:
                logger.error(f"Vector search failed entirely: {e2}")
                return []
        
        if not results["ids"] or not results["ids"][0]:
            return []
        
        output = []
        for i, doc_id in enumerate(results["ids"][0]):
            output.append({
                "id": doc_id,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1.0 - results["distances"][0][i],  # Convert distance to similarity
            })
        
        return output
    
    def _bm25_search(
        self, query: str, top_k: int, metadata_filter: Optional[dict] = None
    ) -> list[dict]:
        """
        Run BM25 keyword search.
        
        Args:
            query: Search query text.
            top_k: Number of results.
            metadata_filter: Dict of metadata key->value to filter by.
            
        Returns:
            List of result dicts with id, text, metadata, score.
        """
        if self.bm25_index is None or not self.bm25_documents:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Apply metadata filter
        filtered_indices = range(len(self.bm25_documents))
        if metadata_filter and self.bm25_metadatas:
            filtered_indices = []
            for i, meta in enumerate(self.bm25_metadatas):
                if self._matches_filter(meta, metadata_filter):
                    filtered_indices.append(i)
        
        # Sort by BM25 score within filtered set
        scored = [(i, scores[i]) for i in filtered_indices if scores[i] > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        output = []
        for idx, score in scored[:top_k]:
            output.append({
                "id": self.bm25_ids[idx],
                "text": self.bm25_documents[idx],
                "metadata": self.bm25_metadatas[idx] if self.bm25_metadatas else {},
                "score": score,
            })
        
        return output
    
    def _matches_filter(self, metadata: dict, filter_dict: dict) -> bool:
        """Check if a metadata dict matches a ChromaDB-style where clause.
        
        Supports:
        - Simple: {"year": "2024"}
        - Composite: {"$and": [{"year": "2024"}, {"quarter": "Q3"}]}
        """
        # Handle $and composite filter
        if "$and" in filter_dict:
            return all(
                self._matches_filter(metadata, condition)
                for condition in filter_dict["$and"]
            )
        
        # Handle $or composite filter
        if "$or" in filter_dict:
            return any(
                self._matches_filter(metadata, condition)
                for condition in filter_dict["$or"]
            )
        
        # Simple equality filter: {"year": "2024"}
        for key, value in filter_dict.items():
            if key.startswith("$"):
                continue  # Skip unknown operator keys
            if metadata.get(key) != value:
                return False
        return True
    
    def _rrf_fusion(
        self, vector_results: list[dict], bm25_results: list[dict], k: int = 60
    ) -> list[dict]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF_score(doc) = Σ 1 / (k + rank_in_method)
        
        Documents ranking highly in BOTH methods get the best combined scores.
        
        Args:
            vector_results: Results from vector search.
            bm25_results: Results from BM25 search.
            k: RRF constant (default 60, standard value).
            
        Returns:
            Fused and re-ranked list of result dicts.
        """
        doc_scores = {}  # id -> {"score": float, "data": dict}
        
        # Score from vector search
        for rank, result in enumerate(vector_results):
            doc_id = result["id"]
            rrf_score = 1.0 / (k + rank + 1)
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"score": 0, "data": result}
            doc_scores[doc_id]["score"] += rrf_score
        
        # Score from BM25 search
        for rank, result in enumerate(bm25_results):
            doc_id = result["id"]
            rrf_score = 1.0 / (k + rank + 1)
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"score": 0, "data": result}
            doc_scores[doc_id]["score"] += rrf_score
        
        # Sort by combined RRF score
        sorted_results = sorted(
            doc_scores.values(),
            key=lambda x: x["score"],
            reverse=True,
        )
        
        return [item["data"] for item in sorted_results]
    
    def _build_metadata_filter(self, query_metadata: dict, active_filters: list[str]) -> Optional[dict]:
        """
        Build a ChromaDB where clause from query metadata and active filter list.
        
        Args:
            query_metadata: Extracted metadata from query (year, quarter, etc.)
            active_filters: List of filter keys currently active.
            
        Returns:
            ChromaDB where clause dict, or None if no filters.
        """
        conditions = []
        
        for key in active_filters:
            value = query_metadata.get(key)
            if value and value != "null" and value != "None":
                conditions.append({key: str(value)})
        
        if not conditions:
            return None
        
        if len(conditions) == 1:
            return conditions[0]
        
        return {"$and": conditions}
    
    def resolve_parent_context(self, results: list[dict]) -> list[dict]:
        """
        Replace child chunk text with parent chunk text for richer LLM context.
        For table summaries, return the raw table instead.
        
        Args:
            results: List of result dicts from search.
            
        Returns:
            Same results but with text replaced by parent context.
        """
        for result in results:
            parent_id = result.get("metadata", {}).get("parent_id")
            if parent_id and parent_id in self.parent_store:
                # Keep original text as 'child_text' for reference
                result["child_text"] = result["text"]
                result["text"] = self.parent_store[parent_id]
        
        return results
    
    def search(
        self,
        query: str,
        query_metadata: Optional[dict] = None,
        metadata_filters: Optional[list[str]] = None,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Run hybrid search with metadata filtering and RRF fusion.
        
        Args:
            query: The search query (potentially rewritten).
            query_metadata: Extracted metadata from the query.
            metadata_filters: Active filter keys (for progressive fallback).
            top_k: Number of results to return.
            
        Returns:
            List of result dicts, sorted by RRF score.
        """
        top_k = top_k or config.TOP_K_RETRIEVAL
        
        # Build metadata filter
        chroma_filter = None
        if query_metadata and metadata_filters:
            chroma_filter = self._build_metadata_filter(query_metadata, metadata_filters)
            if chroma_filter:
                logger.info(f"Applying metadata filter: {chroma_filter}")
        
        # Run both search methods
        vector_results = self._vector_search(query, top_k, chroma_filter)
        bm25_results = self._bm25_search(query, top_k, chroma_filter)
        
        logger.info(f"Vector search: {len(vector_results)} results, "
                    f"BM25 search: {len(bm25_results)} results")
        
        # Fuse results with RRF
        fused_results = self._rrf_fusion(vector_results, bm25_results, k=config.RRF_K)
        
        # Return top-K fused results
        return fused_results[:top_k]
