"""
Hallucination Checker Node — Self-RAG implementation.

Compares the generated answer against the source documents to detect
any claims not supported by the context. If hallucinations are detected,
the workflow loops back to regenerate.
"""

import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

import config
from prompts import HALLUCINATION_GRADER_PROMPT

logger = logging.getLogger(__name__)

import datetime as datetime
todays_date = datetime.datetime.now().strftime("%B %Y")

class HallucinationGrade(BaseModel):
    """Structured output for hallucination assessment."""
    score: str = Field(description="'yes' if fully supported, 'no' if hallucination detected")
    reason: str = Field(description="Explanation identifying any unsupported claims")


def check_hallucination(state: dict) -> dict:
    """
    Check if the generated answer is grounded in the source documents.
    
    Uses LLM to compare answer against documents.
    Sets 'answer_contains_hallucinations' flag and increments retry counter.
    """
    answer = state.get("generation", "")
    documents = state.get("reranked_documents", [])
    hallucination_retries = state.get("hallucination_retries", 0)
    
    if not answer or not documents:
        return {
            "answer_contains_hallucinations": False,
            "hallucination_retries": hallucination_retries,
        }
    
    # Build documents text for comparison
    doc_texts = []
    for doc in documents:
        doc_texts.append(doc.get("text", ""))
    supporting_documents = "\n\n---\n\n".join(doc_texts)
    
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.OPENAI_API_KEY,
        temperature=0,
    )
    
    structured_llm = llm.with_structured_output(HallucinationGrade)
    
    prompt = ChatPromptTemplate.from_template(HALLUCINATION_GRADER_PROMPT)
    
    chain = prompt | structured_llm
    
    result = chain.invoke({
        "documents": supporting_documents,
        "answer": answer, "todays_date": todays_date
    })
    
    has_hallucination = result.score.lower().strip() != "yes"
    
    if has_hallucination:
        logger.warning(f"Hallucination detected: {result.reason}")
    else:
        logger.info("No hallucination detected — answer is grounded")
    
    return {
        "answer_contains_hallucinations": has_hallucination,
        "hallucination_retries": hallucination_retries + 1,
    }
