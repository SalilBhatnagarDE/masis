"""
Enrich Answer Node

Runs after the hallucination check passes. Does three things:
1. Chart extraction - finds numerical data and builds structured JSON for visualization
2. Report formatting - turns open-ended answers into concise executive briefings
3. Document references - appends source citations at the end of the answer
"""

import logging
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import LLM_MODEL
from prompts import CHART_EXTRACTION_PROMPT, REPORT_FORMATTER_PROMPT

logger = logging.getLogger(__name__)

import datetime as datetime
todays_date = datetime.datetime.now().strftime("%B %Y")

class ChartAnalysis(BaseModel):
    """Structured output schema for chart data extraction."""
    has_chart_data: bool = Field(description="Whether the answer contains chartable data")
    chart_type: str = Field(description="Chart type: bar, pie, line, or none")
    title: str = Field(default="", description="Chart title")
    labels: List[str] = Field(default_factory=list, description="Category labels")
    values: List[float] = Field(default_factory=list, description="Numerical values")
    unit: str = Field(default="", description="Unit of measurement")


def enrich_answer(state: dict) -> dict:
    """
    Enrich the generated answer with charts, report formatting, and source references.

    Takes the raw answer from the generate node, runs chart extraction and report
    formatting via LLM calls, then appends document source citations.
    """
    answer = state.get("generation", "")
    question = state.get("original_question", state.get("question", ""))
    documents = state.get("reranked_documents", [])

    if not answer:
        logger.warning("No answer to enrich, skipping")
        return {"chart_data": {}}

    chart_data = {}
    enriched_answer = answer

    # Step 1: Try to extract chart data from the answer
    try:
        chart_data = _extract_chart_data(answer)
        if chart_data:
            logger.info(
                "Chart extracted: %s with %d data points",
                chart_data.get("chart_type"), len(chart_data.get("labels", []))
            )
        else:
            logger.info("No chart-worthy data found in answer")
    except Exception as e:
        logger.warning("Chart extraction failed: %s", e)
        chart_data = {}

    # Step 2: Reformat the answer as a concise briefing
    try:
        enriched_answer = _format_as_report(question, answer)
        logger.info("Answer enriched: %d -> %d chars", len(answer), len(enriched_answer))
    except Exception as e:
        logger.warning("Report formatting failed: %s", e)
        enriched_answer = answer

    # Step 3: Append document references at the bottom
    if documents:
        refs = _build_references(documents)
        if refs:
            enriched_answer += "\n\n---\n" + refs

    return {
        "generation": enriched_answer,
        "chart_data": chart_data,
    }


def _extract_chart_data(answer: str) -> Dict[str, Any]:
    """Use the LLM to detect numerical data and extract it for charting."""
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(ChartAnalysis)

    prompt = ChatPromptTemplate.from_template(CHART_EXTRACTION_PROMPT)
    chain = prompt | structured_llm

    result = chain.invoke({"answer": answer})

    if result.has_chart_data and result.labels and result.values:
        if len(result.labels) != len(result.values):
            logger.warning("Chart data mismatch: labels and values have different lengths")
            return {}

        return {
            "chart_type": result.chart_type,
            "title": result.title,
            "labels": result.labels,
            "values": result.values,
            "unit": result.unit,
        }

    return {}


def _format_as_report(question: str, answer: str) -> str:
    """Use the LLM to reformat the answer as a concise executive briefing."""
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)

    prompt = ChatPromptTemplate.from_template(REPORT_FORMATTER_PROMPT)
    chain = prompt | llm

    result = chain.invoke({"question": question, "answer": answer, "todays_date": todays_date})

    formatted = result.content.strip()

    if len(formatted) < 30:
        logger.warning("Report formatting returned too-short result, keeping original")
        return answer

    return formatted


def _build_references(documents: list) -> str:
    """Build a markdown section listing the source documents used."""
    seen = set()
    refs = []

    for i, doc in enumerate(documents):
        meta = doc.get("metadata", {})
        source = meta.get("source_file", "Unknown")
        page = meta.get("page_number", "?")
        section = meta.get("section", "")

        # Skip duplicates (same file + page)
        key = f"{source}:{page}"
        if key in seen:
            continue
        seen.add(key)

        ref_line = f"**[Doc {i+1}]** {source}"
        if page != "?":
            ref_line += f", p.{page}"
        if section:
            ref_line += f" -- {section}"
        refs.append(ref_line)

    if not refs:
        return ""

    return "**Sources:**\n" + "\n".join(f"- {r}" for r in refs)
