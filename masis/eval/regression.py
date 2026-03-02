"""
masis.eval.regression
=====================
Regression runner for MASIS evaluation (MF-EVAL-02).

Runs all golden dataset queries through the MASIS graph, collects metrics
(latency, cost, iteration count, quality scores, decision log), and
compares against baseline thresholds.

Usage
-----
    python -m masis.eval.regression
    python -m masis.eval.regression --output eval/results/regression_report.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Baseline thresholds (from engineering_tasks.md)
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "max_cost_simple_usd": 0.05,
    "max_cost_complex_usd": 0.15,
    "min_answer_length": 50,
    "max_iterations": 15,
    "max_latency_simple_s": 60,
    "max_latency_complex_s": 180,
}


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

async def run_single_query(
    question: str,
    thread_id: str,
) -> Dict[str, Any]:
    """
    Run a single query through the MASIS graph and collect all metrics.

    Uses ainvoke_graph() from the runner module, which handles initial state
    construction, graph compilation, and invocation.

    Returns a dict with: question, answer, latency_s, cost_usd,
    iteration_count, decision_log, quality_scores, state_snapshot, error.
    """
    from masis.graph.runner import ainvoke_graph, make_config

    result: Dict[str, Any] = {
        "question": question,
        "thread_id": thread_id,
        "answer": "",
        "latency_s": 0.0,
        "cost_usd": 0.0,
        "iteration_count": 0,
        "decision_log": [],
        "quality_scores": {},
        "error": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    start = time.monotonic()
    config = make_config(thread_id=thread_id)

    try:
        final_state = await ainvoke_graph(question, config=config)

        elapsed = time.monotonic() - start
        result["latency_s"] = round(elapsed, 2)

        # Extract results from final state
        if isinstance(final_state, dict):
            # synthesis_output may be text or a dict with .text
            answer = final_state.get("synthesis_output") or ""
            if isinstance(answer, dict):
                answer = answer.get("text", str(answer))
            elif hasattr(answer, "answer"):
                answer = getattr(answer, "answer", "")
            else:
                answer = str(answer)
            result["answer"] = str(answer)

            result["iteration_count"] = final_state.get("iteration_count", 0)
            result["decision_log"] = final_state.get("decision_log", [])
            result["quality_scores"] = final_state.get("quality_scores", {})

            token_budget = final_state.get("token_budget", {})
            result["cost_usd"] = token_budget.get("total_cost_usd", 0.0)

            # Capture full state for tracing
            result["state_snapshot"] = _safe_serialize_state(final_state)

    except Exception as exc:
        elapsed = time.monotonic() - start
        result["latency_s"] = round(elapsed, 2)
        result["error"] = f"{type(exc).__name__}: {exc}"
        logger.error("Query failed: %s -- %s", question[:60], exc, exc_info=True)

    return result



def _safe_serialize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Safely serialize graph state, converting non-serializable objects to strings."""
    serialized: Dict[str, Any] = {}
    for key, value in state.items():
        try:
            json.dumps(value)
            serialized[key] = value
        except (TypeError, ValueError):
            serialized[key] = str(value)
    return serialized


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_results(
    results: List[Dict[str, Any]],
    thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate regression results against thresholds.

    Returns a report with per-query pass/fail and overall status.
    """
    thresholds = thresholds or THRESHOLDS
    evaluations: List[Dict[str, Any]] = []
    all_passed = True

    for r in results:
        checks: Dict[str, Any] = {}

        # Answer quality
        answer = r.get("answer", "")
        checks["has_answer"] = bool(answer and len(answer) > thresholds.get("min_answer_length", 50))

        # Cost
        cost = r.get("cost_usd", 0.0)
        max_cost = thresholds.get("max_cost_complex_usd", 0.15)
        checks["cost_within_budget"] = cost <= max_cost
        checks["cost_usd"] = cost

        # Latency
        latency = r.get("latency_s", 0.0)
        max_latency = thresholds.get("max_latency_complex_s", 180)
        checks["latency_acceptable"] = latency <= max_latency

        # Iterations
        iterations = r.get("iteration_count", 0)
        max_iter = thresholds.get("max_iterations", 15)
        checks["iterations_acceptable"] = iterations <= max_iter

        # No error
        checks["no_error"] = r.get("error") is None

        # Overall pass
        passed = all(v for k, v in checks.items() if isinstance(v, bool))
        checks["passed"] = passed
        if not passed:
            all_passed = False

        evaluations.append({
            "question": r["question"][:80],
            "thread_id": r.get("thread_id", ""),
            "checks": checks,
        })

    return {
        "overall_passed": all_passed,
        "total_queries": len(results),
        "passed_count": sum(1 for e in evaluations if e["checks"]["passed"]),
        "failed_count": sum(1 for e in evaluations if not e["checks"]["passed"]),
        "per_query": evaluations,
    }


# ---------------------------------------------------------------------------
# Full regression run
# ---------------------------------------------------------------------------

async def run_regression(
    output_path: Optional[str] = None,
    doc_folder: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the full regression suite: ingest docs, run queries, evaluate.

    Args:
        output_path: Where to save the JSON report.
        doc_folder: Document folder for ingestion.

    Returns:
        Complete regression report.
    """
    from masis.eval.golden_dataset import load_golden_dataset
    from masis.eval.ingest_docs import setup_retrieval
    from masis.graph.runner import generate_thread_id

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    )

    logger.info("=" * 70)
    logger.info("MASIS Regression Runner")
    logger.info("=" * 70)

    # Step 1: Ingest documents
    logger.info("Step 1: Setting up retrieval pipeline...")
    ingest_stats = setup_retrieval(doc_folder=doc_folder)
    logger.info("  Ingestion: %s", ingest_stats)

    # Step 2: Load golden dataset
    logger.info("Step 2: Loading golden dataset...")
    entries = load_golden_dataset()
    logger.info("  Loaded %d queries", len(entries))

    # Step 3: Run all queries (graph is compiled lazily by ainvoke_graph)
    logger.info("Step 3: Running queries...")
    results: List[Dict[str, Any]] = []

    for i, entry in enumerate(entries, 1):
        thread_id = generate_thread_id()
        logger.info(
            "\n[%d/%d] Query: %s\n  Thread: %s",
            i, len(entries), entry.question[:80], thread_id,
        )

        result = await run_single_query(entry.question, thread_id)
        result["ground_truth"] = entry.ground_truth
        result["query_type"] = entry.query_type
        result["reference_files"] = entry.reference_files
        results.append(result)

        logger.info(
            "  -> Answer: %d chars | Cost: $%.4f | Latency: %.1fs | Iterations: %d | Error: %s",
            len(result.get("answer", "")),
            result.get("cost_usd", 0),
            result.get("latency_s", 0),
            result.get("iteration_count", 0),
            result.get("error"),
        )

    # Step 4: Evaluate
    logger.info("\nStep 4: Evaluating results...")
    evaluation = evaluate_results(results)

    # Step 5: Build report
    report = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_queries": len(entries),
            "ingest_stats": ingest_stats,
        },
        "evaluation": evaluation,
        "results": results,
    }

    # Save report
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Report saved to %s", output_path)

    # Print summary
    print("\n" + "=" * 70)
    print("  REGRESSION RESULTS")
    print("=" * 70)
    print(f"  Queries: {evaluation['total_queries']}")
    print(f"  Passed:  {evaluation['passed_count']}")
    print(f"  Failed:  {evaluation['failed_count']}")
    print(f"  Status:  {'ALL PASSED' if evaluation['overall_passed'] else 'FAILURES'}")

    for pq in evaluation["per_query"]:
        status = "PASS" if pq["checks"]["passed"] else "FAIL"
        print(f"\n  {status} {pq['question']}")
        for k, v in pq["checks"].items():
            if k != "passed":
                icon = "Y" if (isinstance(v, bool) and v) else str(v)
                print(f"      {k}: {icon}")

    print("=" * 70)

    return report



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MASIS regression runner")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "results" / "regression_report.json"),
    )
    parser.add_argument("--doc-folder", default=None)
    args = parser.parse_args()

    report = asyncio.run(run_regression(
        output_path=args.output,
        doc_folder=args.doc_folder,
    ))
