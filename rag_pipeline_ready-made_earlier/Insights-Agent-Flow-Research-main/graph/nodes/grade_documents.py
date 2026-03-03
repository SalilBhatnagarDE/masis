"""
Document Grader Node — CRAG (Corrective RAG) implementation.

Grades each reranked document for relevance using LLM.
Filters out irrelevant documents. If not enough relevant docs remain,
the workflow loops back to rewrite the query.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

import config
from prompts import DOCUMENT_GRADER_PROMPT

logger = logging.getLogger(__name__)

import datetime as datetime
todays_date = datetime.datetime.now().strftime("%B %Y")

class DocumentGrade(BaseModel):
    """Structured output for document grading."""
    score: str = Field(description="Relevance: 'yes' or 'no'")
    reason: str = Field(description="Brief reason for the grade")


def _grade_single_document(question: str, doc: dict) -> tuple[dict, bool, str]:
    """Grade a single document for relevance."""
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.OPENAI_API_KEY,
        temperature=0,
    )
    
    structured_llm = llm.with_structured_output(DocumentGrade)
    
    prompt = ChatPromptTemplate.from_template(DOCUMENT_GRADER_PROMPT)
    
    chain = prompt | structured_llm
    
    doc_text = doc.get("text", "")[:3000]  # Limit to avoid token overflow
    
    result = chain.invoke({"question": question, "document": doc_text, "todays_date": todays_date})
    is_relevant = result.score.lower().strip() == "yes"
    
    return doc, is_relevant, result.reason


def grade_documents(state: dict) -> dict:
    """
    Grade each reranked document for relevance using LLM.
    
    Uses ThreadPoolExecutor for parallel grading (same pattern as FinSight).
    Filters out irrelevant documents and increments retry counter.
    """
    question = state.get("original_question", state.get("question", ""))
    documents = state.get("reranked_documents", [])
    doc_grading_retries = state.get("doc_grading_retries", 0)
    
    if not documents:
        return {
            "reranked_documents": [],
            "doc_grading_retries": doc_grading_retries + 1,
            "irrelevancy_reason": "No documents to grade",
        }
    
    logger.info(f"Grading {len(documents)} documents for relevance...")
    
    # Parallel grading with ThreadPoolExecutor
    relevant_docs = []
    irrelevant_reasons = []
    
    with ThreadPoolExecutor(max_workers=min(len(documents), 4)) as executor:
        futures = [
            executor.submit(_grade_single_document, question, doc)
            for doc in documents
        ]
        
        for future in futures:
            try:
                doc, is_relevant, reason = future.result()
                if is_relevant:
                    relevant_docs.append(doc)
                else:
                    irrelevant_reasons.append(reason)
            except Exception as e:
                logger.warning(f"Error grading document: {e}")
    
    logger.info(f"Grading complete: {len(relevant_docs)}/{len(documents)} relevant")
    
    return {
        "reranked_documents": relevant_docs,
        "doc_grading_retries": doc_grading_retries + 1,
        "irrelevancy_reason": "; ".join(irrelevant_reasons) if irrelevant_reasons else "",
    }
