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

