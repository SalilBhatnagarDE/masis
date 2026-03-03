"""
masis.eval.standard_queries
==========================
Canonical Infosys demo/eval query presets used by CLI runners and Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class StandardQuery:
    idx: int
    name: str
    query: str


def load_standard_infosys_queries() -> List[StandardQuery]:
    """Return the shared six-query Infosys preset list."""
    return [
        StandardQuery(
            idx=1,
            name="Q1",
            query="What is our current revenue trend?",
        ),
        StandardQuery(
            idx=2,
            name="Q2",
            query="Which departments are underperforming in this year?",
        ),
        StandardQuery(
            idx=3,
            name="Q3",
            query="What are the risks highlighted in this year?",
        ),
        StandardQuery(
            idx=4,
            name="Q4",
            query="What was AI's impact on Infosys's operating margins?",
        ),
        StandardQuery(
            idx=5,
            name="Q5",
            query="How is the technology division performing?",
        ),
        StandardQuery(
            idx=6,
            name="Q6",
            query="Compare Infosys's cloud revenue to Adobe, and GCP",
        ),
    ]


def load_strategic_deep_research_queries() -> List[StandardQuery]:
    """Strategic deep-research queries designed to exercise multi-agent DAG planning,
    parallel evidence gathering, skeptic validation, and cross-dimensional synthesis
    on Infosys financial data.

    Query design principles:
    - Scope is bounded to data available in the knowledge base (Q1 2024 – Q3 2025)
    - Each query forces ≥2 parallel researcher tasks (entity × dimension decomposition)
    - Skeptic is challenged with cross-claim consistency validation
    - Synthesizer must reconcile multi-dimensional evidence with citations
    """
    return [
        StandardQuery(
            idx=7,
            name="SQ1",
            query=(
                "Analyze Infosys's revenue trend from Q1 2024 through Q3 2025: "
                "what sequential and year-on-year growth rates reveal about momentum, "
                "how the $4.8B large deal TCV in Q3 2025 provides forward revenue visibility, "
                "and what the primary business headwinds driving the deceleration are."
            ),
        ),
        StandardQuery(
            idx=8,
            name="SQ2",
            query=(
                "What is Infosys's strategic positioning in AI and Generative AI services? "
                "Identify the specific AI platforms launched, analyst recognition received, "
                "and how AI-driven efficiency is expected to impact operating margins and "
                "the traditional IT services portfolio."
            ),
        ),
        StandardQuery(
            idx=9,
            name="SQ3",
            query=(
                "What are the top operational and regulatory risks facing Infosys based on "
                "recent disclosures, and what mitigation strategies has management outlined? "
                "Cover talent and wage inflation, cybersecurity incidents, geopolitical "
                "and regulatory headwinds including immigration policy changes."
            ),
        ),
        StandardQuery(
            idx=10,
            name="SQ4",
            query=(
                "Assess Infosys's financial health and capital allocation strategy: "
                "how does the $965M free cash flow translate into shareholder returns, "
                "what does the operating margin trajectory signal about cost discipline, "
                "and how do large deal wins support the revenue growth outlook?"
            ),
        ),
        StandardQuery(
            idx=11,
            name="DEMO",
            query=(
                "Analyze Infosys's revenue momentum and strategic positioning for growth: "
                "what do the quarterly revenue growth rates from Q1 2024 to Q3 2025 reveal about "
                "business trajectory, how does AI and GenAI strategy through platforms like Infosys Topaz "
                "position the company competitively, and does the $4.8B large deal pipeline in Q3 2025 "
                "provide sufficient forward visibility to support management's growth guidance?"
            ),
        ),
    ]

