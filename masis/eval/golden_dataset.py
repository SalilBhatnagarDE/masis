"""
masis.eval.golden_dataset
=========================
Golden dataset loader and structure for MASIS evaluation (MF-EVAL-01).

Loads curated test queries with expected outputs from JSON, used by
regression.py and scenario_tests.py.

Source data: golden_dataset.json from engineer 6's evaluation folder,
containing 3 real Infosys queries with ground truth answers.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path to the golden dataset JSON
# ---------------------------------------------------------------------------
GOLDEN_DATASET_PATH = str(
    Path(__file__).resolve().parents[2]
    / "rag_pipeline_ready-made_by_engineer6"
    / "Insights-Agent-Flow-Research-main"
    / "evaluation"
    / "golden_dataset.json"
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class GoldenEntry(BaseModel):
    """A single golden dataset entry with query, ground truth, and assertions."""

    question: str = Field(..., description="The test query to run through MASIS.")
    ground_truth: str = Field(..., description="Expected answer content for evaluation.")
    reference_files: List[str] = Field(
        default_factory=list,
        description="Expected source document filenames.",
    )
    query_type: str = Field(
        default="factual",
        description="Query category: factual, comparative, thematic, contradictory, ambiguous.",
    )
    expected_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords expected in the answer.",
    )
    expected_citations_min: int = Field(
        default=1,
        description="Minimum number of distinct source citations expected.",
    )
    expected_cost_max_usd: float = Field(
        default=0.15,
        description="Maximum acceptable cost for this query in USD.",
    )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_golden_dataset(path: Optional[str] = None) -> List[GoldenEntry]:
    """
    Load the golden dataset from a JSON file.

    Each entry in the JSON must have at minimum: question, ground_truth.
    Additional fields (reference_files, query_type, etc.) are optional
    and will use defaults if not present.

    Args:
        path: Path to the JSON file. Defaults to GOLDEN_DATASET_PATH.

    Returns:
        List of GoldenEntry objects.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
    """
    path = path or GOLDEN_DATASET_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"Golden dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_data: List[Dict[str, Any]] = json.load(f)

    entries: List[GoldenEntry] = []
    for i, item in enumerate(raw_data):
        # Enrich with defaults based on content analysis
        if "query_type" not in item:
            item["query_type"] = _classify_query_type(item.get("question", ""))

        if "expected_keywords" not in item:
            item["expected_keywords"] = _extract_keywords(item.get("ground_truth", ""))

        if "expected_citations_min" not in item:
            refs = item.get("reference_files", [])
            item["expected_citations_min"] = max(1, len(refs))

        try:
            entries.append(GoldenEntry(**item))
        except Exception as exc:
            logger.warning("Skipping golden entry %d: %s", i, exc)

    logger.info("Loaded %d golden dataset entries from %s", len(entries), path)
    return entries


def _classify_query_type(question: str) -> str:
    """Auto-classify query type from question text."""
    q = question.lower()
    if any(kw in q for kw in ["risk", "challenge", "threat"]):
        return "thematic"
    if any(kw in q for kw in ["underperform", "compare", "versus", "vs"]):
        return "comparative"
    if any(kw in q for kw in ["trend", "revenue", "growth"]):
        return "factual"
    if any(kw in q for kw in ["contradict", "disagree", "conflict"]):
        return "contradictory"
    if any(kw in q for kw in ["ambiguous", "unclear", "vague"]):
        return "ambiguous"
    return "factual"


def _extract_keywords(ground_truth: str) -> List[str]:
    """Extract key terms from ground truth for answer validation."""
    import re
    # Extract capitalized proper nouns and important terms
    words = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', ground_truth)
    # Also extract numbers and percentages
    numbers = re.findall(r'\$?[\d,.]+[BMK]?%?', ground_truth)
    keywords = list(set(words[:10] + numbers[:10]))
    return keywords[:15]  # Cap at 15


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    entries = load_golden_dataset()
    print(f"\nLoaded {len(entries)} golden entries:\n")
    for i, entry in enumerate(entries, 1):
        print(f"  Q{i} [{entry.query_type}]: {entry.question[:80]}...")
        print(f"      Keywords: {entry.expected_keywords[:5]}")
        print(f"      Refs: {entry.reference_files}")
        print(f"      Max cost: ${entry.expected_cost_max_usd}")
        print()
