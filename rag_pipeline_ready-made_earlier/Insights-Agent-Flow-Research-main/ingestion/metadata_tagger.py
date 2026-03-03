"""
Metadata Tagger — Extracts structured metadata from documents at ingestion time.

Two levels of metadata extraction:
1. Document-level: LLM reads first ~3 pages -> document_type, year, quarter
2. Chunk-level: Auto-tagged per chunk -> content_type, page_number, source_file, section
"""

import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import config
from prompts import DOC_METADATA_PROMPT


def get_metadata_extractor():
    """Create a document metadata extraction chain."""
    llm = ChatOpenAI(
        model=config.INGESTION_LLM_MODEL,
        api_key=config.OPENAI_API_KEY,
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_template(DOC_METADATA_PROMPT)
    chain = prompt | llm
    return chain


def extract_document_metadata(text_first_pages: str) -> dict:
    """
    Extract document-level metadata from the first few pages of a document.
    
    Args:
        text_first_pages: Text content from the first ~3 pages.
        
    Returns:
        Dict with keys: document_type, year, quarter, departments_mentioned
    """
    chain = get_metadata_extractor()
    response = chain.invoke({"text": text_first_pages})
    
    # Parse JSON from LLM response
    content = response.content.strip()
    
    # Try to extract JSON from the response (handle markdown code blocks)
    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
    if json_match:
        content = json_match.group(1)
    
    try:
        metadata = json.loads(content)
    except json.JSONDecodeError:
        # Fallback: return defaults if LLM output isn't valid JSON
        metadata = {
            "document_type": "other",
            "year": None,
            "quarter": None,
            "departments_mentioned": [],
        }
    
    return metadata


def detect_section_from_text(text: str) -> str:
    """
    Detect the section name from chunk text using header patterns.
    
    Args:
        text: The chunk text content.
        
    Returns:
        Detected section name or "general".
    """
    section_keywords = {
        "Financial Performance": ["revenue", "income", "profit", "loss", "earnings", "ebitda", "margin"],
        "Risk Factors": ["risk", "uncertainty", "challenge", "threat", "concern"],
        "Operations": ["operations", "operational", "production", "efficiency", "process"],
        "Strategy": ["strategy", "strategic", "initiative", "roadmap", "vision", "mission"],
        "Personnel": ["employee", "headcount", "hiring", "attrition", "talent", "workforce"],
        "Technology": ["technology", "digital", "platform", "system", "infrastructure", "AI"],
        "Compliance": ["compliance", "regulatory", "regulation", "audit", "governance"],
        "Market": ["market", "competition", "competitor", "industry", "benchmark"],
    }
    
    text_lower = text.lower()
    scores = {}
    
    for section, keywords in section_keywords.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[section] = score
    
    if scores:
        return max(scores, key=scores.get)
    return "General"


def detect_department_from_text(text: str) -> str:
    """
    Detect department from chunk text using keyword matching.
    
    Args:
        text: The chunk text content.
        
    Returns:
        Detected department name or "Company-wide".
    """
    dept_keywords = {
        "Sales": ["sales", "selling", "revenue generation", "deals", "pipeline", "bookings"],
        "Engineering": ["engineering", "development", "r&d", "technical", "software", "product development"],
        "Marketing": ["marketing", "brand", "campaign", "advertising", "demand generation"],
        "Finance": ["finance", "accounting", "budget", "treasury", "financial planning"],
        "HR": ["human resources", "hr", "talent", "recruitment", "employee", "people operations"],
        "Operations": ["operations", "supply chain", "logistics", "manufacturing", "fulfillment"],
        "Legal": ["legal", "compliance", "regulatory", "contract", "intellectual property"],
        "IT": ["information technology", "it infrastructure", "cybersecurity", "data center"],
    }
    
    text_lower = text.lower()
    scores = {}
    
    for dept, keywords in dept_keywords.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[dept] = score
    
    if scores:
        return max(scores, key=scores.get)
    return "Company-wide"
