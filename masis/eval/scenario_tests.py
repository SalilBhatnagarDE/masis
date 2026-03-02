"""
masis.eval.scenario_tests
==========================
Per-scenario end-to-end tests for MASIS evaluation (MF-EVAL-03).

Runs the 3 real Infosys queries from the golden dataset through the full
MASIS pipeline with complete LangGraph logging and tracing.

Each test:
1. Sets up document ingestion (ChromaDB + BM25)
2. Invokes the MASIS graph via ainvoke_graph()
3. Captures the full state, decision log, and execution trace
4. Asserts on answer quality, cost, and iteration count
5. Saves detailed logs to eval/results/

Usage
-----
    # Run all 3 scenarios
    python -m masis.eval.scenario_tests

    # Run with pytest (verbose)
    python -m pytest masis/eval/scenario_tests.py -v -s
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

# Ensure project root on path
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)

# Results directory
RESULTS_DIR = str(Path(__file__).parent / "results")


# ===========================================================================
# Setup: Ingestion + Graph Compilation
# ===========================================================================

_SETUP_DONE = False
_GRAPH = None


def _ensure_setup():
    """One-time setup: ingest docs and compile graph."""
    global _SETUP_DONE, _GRAPH

    if _SETUP_DONE:
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Configure comprehensive logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(RESULTS_DIR, "scenario_tests.log"),
                mode="w",
                encoding="utf-8",
            ),
        ],
        force=True,
    )

    logger.info("=" * 70)
    logger.info("MASIS Scenario Tests -- Setup")
    logger.info("=" * 70)

    # Step 1: Ingest documents
    from masis.eval.ingest_docs import setup_retrieval
    logger.info("Setting up retrieval pipeline...")
    stats = setup_retrieval()
    logger.info("Ingestion complete: %s", stats)

    # Step 2: Compile graph
    from masis.graph.runner import get_graph
    logger.info("Compiling MASIS graph...")
    _GRAPH = get_graph()
    logger.info("Graph compiled successfully")

    _SETUP_DONE = True


# ===========================================================================
# Query execution with full tracing
# ===========================================================================

async def _run_scenario_query(
    query: str,
    scenario_name: str,
) -> Dict[str, Any]:
    """
    Run a single query through the MASIS graph with full tracing.

    Returns a comprehensive result dict with the full state, decision log,
    execution metrics, and captured logs.
    """
    from masis.graph.runner import ainvoke_graph, generate_thread_id, make_config

    thread_id = generate_thread_id()
    config = make_config(thread_id=thread_id)

    logger.info("\n" + "=" * 70)
    logger.info("SCENARIO: %s", scenario_name)
    logger.info("  Query:     %s", query)
    logger.info("  Thread ID: %s", thread_id)
    logger.info("=" * 70)

    # Capture execution
    result: Dict[str, Any] = {
        "scenario": scenario_name,
        "query": query,
        "thread_id": thread_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    start = time.monotonic()

    try:
        # Use the graph tracing if available
        try:
            from masis.infra.tracing import trace_graph_invocation
            with trace_graph_invocation(query, thread_id):
                final_state = await ainvoke_graph(query, config=config)
        except ImportError:
            logger.warning("Tracing module not available  --  running without trace context")
            final_state = await ainvoke_graph(query, config=config)

        elapsed = time.monotonic() - start
        result["latency_s"] = round(elapsed, 2)
        result["success"] = True
        result["error"] = None

        # Extract comprehensive results from final state
        if isinstance(final_state, dict):
            result["final_answer"] = final_state.get("synthesis_output") or ""

            # If synthesis_output is a dict/object, extract answer text
            if isinstance(result["final_answer"], dict):
                result["final_answer"] = result["final_answer"].get("text", str(result["final_answer"]))
            elif hasattr(result["final_answer"], "answer"):
                result["final_answer"] = getattr(result["final_answer"], "answer", "")
            else:
                result["final_answer"] = str(result["final_answer"])

            result["iteration_count"] = final_state.get("iteration_count", 0)
            result["supervisor_decision"] = final_state.get("supervisor_decision", "")
            result["validation_round"] = final_state.get("validation_round", 0)

            # Decision log -- full trace of supervisor->executor->validator cycles
            decision_log = final_state.get("decision_log", [])
            result["decision_log"] = decision_log
            result["decision_log_length"] = len(decision_log)

            # Quality scores
            result["quality_scores"] = final_state.get("quality_scores", {})

            # Evidence board
            evidence = final_state.get("evidence_board", [])
            result["evidence_count"] = len(evidence) if isinstance(evidence, list) else 0

            # Cost tracking
            token_budget = final_state.get("token_budget", {})
            if hasattr(token_budget, "total_tokens_used"):
                result["total_tokens_used"] = token_budget.total_tokens_used
                result["total_cost_usd"] = token_budget.total_cost_usd
            elif isinstance(token_budget, dict):
                result["total_tokens_used"] = token_budget.get("total_tokens_used", 0)
                result["total_cost_usd"] = token_budget.get("total_cost_usd", 0.0)
            else:
                result["total_tokens_used"] = 0
                result["total_cost_usd"] = 0.0

            # Task DAG
            task_dag = final_state.get("task_dag", [])
            result["task_count"] = len(task_dag) if isinstance(task_dag, list) else 0

            # Full state snapshot (for detailed analysis)
            result["full_state"] = _safe_serialize(final_state)

        else:
            result["final_answer"] = str(final_state) if final_state else ""

    except Exception as exc:
        elapsed = time.monotonic() - start
        result["latency_s"] = round(elapsed, 2)
        result["success"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["final_answer"] = ""
        logger.error("Scenario %s FAILED: %s", scenario_name, exc, exc_info=True)

    # Log results summary
    logger.info("\n" + "-" * 50)
    logger.info("SCENARIO RESULT: %s", scenario_name)
    logger.info("  Success:     %s", result.get("success"))
    logger.info("  Answer len:  %d chars", len(result.get("final_answer", "")))
    logger.info("  Iterations:  %s", result.get("iteration_count", "N/A"))
    logger.info("  Cost:        $%.4f", result.get("total_cost_usd", 0))
    logger.info("  Latency:     %.1fs", result.get("latency_s", 0))
    logger.info("  Decisions:   %d entries", result.get("decision_log_length", 0))
    logger.info("  Evidence:    %d chunks", result.get("evidence_count", 0))
    logger.info("  Error:       %s", result.get("error"))
    if result.get("final_answer"):
        logger.info("  Answer preview: %s", result["final_answer"][:200])
    logger.info("-" * 50)

    return result


def _safe_serialize(obj: Any) -> Any:
    """Recursively convert non-serializable objects to strings."""
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_safe_serialize(item) for item in obj]
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)


# ===========================================================================
# Scenario 1: Revenue Trend (Simple Factual Query)
# ===========================================================================

def test_s1_revenue_trend():
    """
    S1  --  Simple factual query: Revenue trend analysis.

    Query: "What is our current revenue trend and how does it compare to last year?"
    Expected: Answer mentions Infosys revenue figures, growth percentages,
    references financial documents.
    """
    _ensure_setup()

    query = "What is our current revenue trend and how does it compare to last year?"
    result = asyncio.run(_run_scenario_query(query, "S1_Revenue_Trend"))

    # Save detailed trace
    _save_scenario_result("s1_revenue_trend", result)

    # Assertions
    assert result["success"], f"S1 failed with error: {result.get('error')}"
    assert len(result.get("final_answer", "")) > 50, \
        f"S1 answer too short: {len(result.get('final_answer', ''))} chars"

    logger.info("[PASS] S1 Revenue Trend -- PASSED")
    return result


# ===========================================================================
# Scenario 2: Underperforming Departments (Multi-dimensional Query)
# ===========================================================================

def test_s2_underperforming_depts():
    """
    S2  --  Multi-dimensional comparative query: Department performance.

    Query: "Which departments are underperforming and what are the root causes?"
    Expected: Answer identifies specific departments, cites evidence from
    multiple documents, provides comparative analysis.
    """
    _ensure_setup()

    query = "Which departments are underperforming and what are the root causes?"
    result = asyncio.run(_run_scenario_query(query, "S2_Underperforming_Depts"))

    _save_scenario_result("s2_underperforming_depts", result)

    assert result["success"], f"S2 failed with error: {result.get('error')}"
    assert len(result.get("final_answer", "")) > 50, \
        f"S2 answer too short: {len(result.get('final_answer', ''))} chars"

    logger.info("[PASS] S2 Underperforming Departments -- PASSED")
    return result


# ===========================================================================
# Scenario 3: Key Risks (Thematic Query)
# ===========================================================================

def test_s3_key_risks():
    """
    S3  --  Thematic risk analysis query.

    Query: "What were the key risks highlighted in the recent reports and
            how are they being mitigated?"
    Expected: Answer identifies specific risks, mentions mitigation strategies,
    references risk-related documents.
    """
    _ensure_setup()

    query = "What were the key risks highlighted in the recent reports and how are they being mitigated?"
    result = asyncio.run(_run_scenario_query(query, "S3_Key_Risks"))

    _save_scenario_result("s3_key_risks", result)

    assert result["success"], f"S3 failed with error: {result.get('error')}"
    assert len(result.get("final_answer", "")) > 50, \
        f"S3 answer too short: {len(result.get('final_answer', ''))} chars"

    logger.info("[PASS] S3 Key Risks -- PASSED")
    return result


# ===========================================================================
# Result persistence
# ===========================================================================

def _save_scenario_result(name: str, result: Dict[str, Any]):
    """Save scenario result as JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(RESULTS_DIR, f"{name}.json")

    # Save full result
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Saved scenario result: %s", filepath)

    # Also save just the decision log separately for easy reading
    log_path = os.path.join(RESULTS_DIR, f"{name}_decision_log.json")
    decision_data = {
        "scenario": name,
        "query": result.get("query"),
        "decision_log": result.get("decision_log", []),
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(decision_data, f, indent=2, default=str)


# ===========================================================================
# Summary report
# ===========================================================================

def _print_summary(results: List[Dict[str, Any]]):
    """Print a formatted summary table of all scenario results."""
    print("\n" + "=" * 80)
    print("  MASIS SCENARIO TEST RESULTS")
    print("=" * 80)
    print(f"  {'Scenario':<35} {'Status':<10} {'Answer':<10} {'Cost':<10} {'Time':<10} {'Iters':<6}")
    print("-" * 80)

    all_passed = True
    for r in results:
        name = r.get("scenario", "?")
        ok = "PASS" if r.get("success") else "FAIL"
        if not r.get("success"):
            all_passed = False
        ans_len = f"{len(r.get('final_answer', ''))} ch"
        cost = f"${r.get('total_cost_usd', 0):.4f}"
        latency = f"{r.get('latency_s', 0):.1f}s"
        iters = str(r.get("iteration_count", "?"))
        print(f"  {name:<35} {ok:<10} {ans_len:<10} {cost:<10} {latency:<10} {iters:<6}")

    print("-" * 80)
    overall = "ALL PASSED" if all_passed else "FAILURES DETECTED"
    print(f"  Overall: {overall}")
    print("=" * 80)

    # Print detailed answers
    for r in results:
        print(f"\n--- {r.get('scenario', '?')} ---")
        print(f"Query: {r.get('query', '?')}")
        answer = r.get("final_answer", "")
        if answer:
            print(f"Answer ({len(answer)} chars):")
            print(answer[:1000])
            if len(answer) > 1000:
                print(f"  ... [{len(answer) - 1000} more chars]")
        else:
            print(f"Error: {r.get('error', 'No answer')}")
        print()


# ===========================================================================
# Golden dataset loader test
# ===========================================================================

def test_golden_dataset_loads():
    """Verify golden dataset loads correctly."""
    from masis.eval.golden_dataset import load_golden_dataset

    entries = load_golden_dataset()
    assert len(entries) >= 3, f"Expected at least 3 entries, got {len(entries)}"

    for entry in entries:
        assert entry.question, "Question must not be empty"
        assert entry.ground_truth, "Ground truth must not be empty"
        assert entry.query_type in ["factual", "comparative", "thematic", "contradictory", "ambiguous"]

    logger.info("[PASS] Golden dataset loads correctly: %d entries", len(entries))


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    print("\n>>> MASIS Scenario Tests -- Running 3 E2E queries\n")

    # Run all three scenarios
    results: List[Dict[str, Any]] = []

    for test_fn in [test_s1_revenue_trend, test_s2_underperforming_depts, test_s3_key_risks]:
        try:
            result = test_fn()
            results.append(result)
        except Exception as exc:
            err_msg = str(exc).encode('ascii', errors='replace').decode('ascii')
            print(f"  [FAIL] {test_fn.__name__} FAILED: {err_msg}")
            results.append({
                "scenario": test_fn.__name__,
                "success": False,
                "error": str(exc),
                "final_answer": "",
                "total_cost_usd": 0,
                "latency_s": 0,
                "iteration_count": 0,
            })

    # Print summary
    _print_summary(results)

    # Save combined report
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "scenario_tests_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nFull report saved: {report_path}")
