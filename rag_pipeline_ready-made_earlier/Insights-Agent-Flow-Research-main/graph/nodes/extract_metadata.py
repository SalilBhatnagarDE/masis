"""
Metadata Extractor Node — Extracts structured metadata from the user's query.

Enables metadata-filtered retrieval: instead of searching the entire corpus,
we pre-filter by year, quarter, department, etc.

Example: "What was Q3 2024 revenue?" -> {year: "2024", quarter: "Q3", topic: "revenue"}
-> Pre-filter ChromaDB WHERE year="2024" AND quarter="Q3" -> then vector search
"""

import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Optional

import config
from prompts import METADATA_EXTRACTOR_PROMPT

import datetime as datetime
todays_date = datetime.datetime.now().strftime("%B %Y")

class QueryMetadata(BaseModel):
    """Structured output for query metadata extraction."""
    document_type: Optional[str] = Field(default=None, description="Type of document")
    year: Optional[str] = Field(default=None, description="Year being asked about")
    quarter: Optional[str] = Field(default=None, description="Quarter being asked about")
    department: Optional[str] = Field(default=None, description="Department being asked about")
    topic: Optional[str] = Field(default=None, description="Main topic of the question")


def extract_metadata(state: dict) -> dict:
    """
    Extract structured metadata from the user's question.
    
    Uses LLM with structured output to identify year, quarter, department,
    topic, and document type from the question.
    
    Args:
        state: Current graph state with 'question' and 'original_question'.
        
    Returns:
        State update with 'query_metadata' and initialized 'metadata_filters'.
    """
    # Use original question for metadata extraction (not the rewritten one)
    question = state.get("original_question", state.get("question", ""))
    
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.OPENAI_API_KEY,
        temperature=0,
    )
    
    structured_llm = llm.with_structured_output(QueryMetadata)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", METADATA_EXTRACTOR_PROMPT),
        ("human", "{question}"),
    ])
    
    chain = prompt | structured_llm
    
    result = chain.invoke({"question": question, "todays_date": todays_date})
    
    # Convert to dict, filtering out None values
    metadata = {}
    if result.document_type:
        metadata["document_type"] = result.document_type
    if result.year:
        metadata["year"] = result.year
    if result.quarter:
        metadata["quarter"] = result.quarter
    if result.department:
        metadata["department"] = result.department
    if result.topic:
        metadata["topic"] = result.topic
    
    # Initialize metadata filters based on what was extracted
    active_filters = [k for k in config.METADATA_FILTER_KEYS if k in metadata]
    
    return {
        "query_metadata": metadata,
        "metadata_filters": active_filters,
        "metadata_retries": 0,
    }
