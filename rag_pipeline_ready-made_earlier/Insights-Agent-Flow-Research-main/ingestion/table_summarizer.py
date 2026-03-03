"""
Table Summarizer — Generates natural language summaries of tables for embedding.

Pattern: Embed the SUMMARY for retrieval, return the RAW TABLE as context.
This ensures tables are found by semantic search (e.g., "underperforming departments"
matches "Sales missed target by 17.3%") while the LLM gets exact numbers.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import config
from prompts import TABLE_SUMMARY_PROMPT

# Module-level singleton — avoids re-creating the LLM client on every call
_table_chain = None


def _get_table_chain():
    """Return a cached table summary chain (singleton)."""
    global _table_chain
    if _table_chain is None:
        llm = ChatOpenAI(
            model=config.INGESTION_LLM_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0,
        )
        prompt = ChatPromptTemplate.from_template(TABLE_SUMMARY_PROMPT)
        _table_chain = prompt | llm
    return _table_chain


def summarize_table(table_markdown: str) -> str:
    """
    Generate a natural language summary of a table for embedding.
    
    Args:
        table_markdown: The table in Markdown format.
        
    Returns:
        A 2-3 sentence natural language summary of the table contents.
    """
    chain = _get_table_chain()
    response = chain.invoke({"table": table_markdown})
    return response.content.strip()

