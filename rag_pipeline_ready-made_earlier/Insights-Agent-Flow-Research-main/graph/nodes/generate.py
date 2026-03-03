"""
Generate Node — Produces the final answer using graded documents as context.

System prompt enforces: answer based ONLY on provided context.
If information is not available, explicitly states so.
"""

import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import config
from prompts import GENERATION_PROMPT

logger = logging.getLogger(__name__)

import datetime as datetime
todays_date = datetime.datetime.now().strftime("%B %Y")

def generate_answer(state: dict) -> dict:
    """
    Generate an answer from the graded documents.
    
    Uses the GENERATION_PROMPT which enforces grounded, factual responses.
    """
    question = state.get("original_question", state.get("question", ""))
    documents = state.get("reranked_documents", [])
    
    # Build context string from documents
    context_parts = []
    for i, doc in enumerate(documents):
        source = doc.get("metadata", {}).get("source_file", "Unknown")
        page = doc.get("metadata", {}).get("page_number", "?")
        text = doc.get("text", "")
        context_parts.append(f"[Document {i+1} | Source: {source}, Page: {page}]\n{text}")
    
    context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant documents found."
    
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.OPENAI_API_KEY,
        temperature=0.1,  # Low temp for factual answers
    )
    
    prompt = ChatPromptTemplate.from_template(GENERATION_PROMPT)
    chain = prompt | llm
    
    result = chain.invoke({"context": context, "question": question, "todays_date": todays_date})
    
    logger.info(f"Generated answer ({len(result.content)} chars)")
    
    return {"generation": result.content.strip()}
