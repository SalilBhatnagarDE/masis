"""
run_q1_only.py
==============
Runs ONLY scenario S1 (Q1: revenue trend) from the MASIS evaluation suite.

Outputs:
  - eval/results/q1_run.log   -- full DEBUG log of the graph execution
  - eval/results/q1_result.json -- structured JSON result

Usage:
  python -m masis.eval.run_q1_only
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# ── 1. Set up dual logging: console + file BEFORE any masis imports ────────
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = RESULTS_DIR / "q1_run.log"

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

fmt = logging.Formatter(
    "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# File handler — full DEBUG log
fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(fmt)
root_logger.addHandler(fh)

# Console handler — INFO only (less noise)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(fmt)
root_logger.addHandler(ch)

# Quiet noisy HTTP libs in console (still logged to file)
for noisy in ("httpcore", "httpx", "openai._base_client", "urllib3", "chromadb.telemetry"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ── 2. Load env + masis imports ───────────────────────────────────────────
try:
    from masis.config.settings import get_settings as _gs; _gs()  # triggers .env load
except Exception:
    pass

Q1_QUERY = "What is our current revenue trend and how does it compare to last year?"
SCENARIO_NAME = "S1_Revenue_Trend"


def _safe_serialize(obj):
    """Recursively convert non-JSON-serialisable objects to strings."""
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(i) for i in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


async def run_q1():
    from masis.eval.ingest_docs import DEFAULT_DOC_FOLDER, setup_retrieval
    from masis.graph.runner import ainvoke_graph

    # ── Ingestion (reuses existing ChromaDB if available) ─────────────────
    logger.info("=" * 70)
    logger.info("Q1 Runner: setting up retrieval pipeline")
    logger.info("=" * 70)

    doc_folder = DEFAULT_DOC_FOLDER
    setup_retrieval(doc_folder)

    # ── Run Q1 ────────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("RUNNING: %s", SCENARIO_NAME)
    logger.info("QUERY  : %s", Q1_QUERY)
    logger.info("=" * 70)

    start = time.monotonic()
    result = {
        "scenario": SCENARIO_NAME,
        "query": Q1_QUERY,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "success": False,
        "error": None,
        "final_answer": "",
        "latency_s": 0,
        "iteration_count": 0,
        "supervisor_decision": "",
        "evidence_count": 0,
        "total_tokens_used": 0,
        "total_cost_usd": 0.0,
        "task_count": 0,
        "decision_log": [],
        "full_state": {},
    }

    try:
        final_state = await ainvoke_graph(Q1_QUERY, thread_id=SCENARIO_NAME)
        elapsed = time.monotonic() - start

        result["latency_s"] = round(elapsed, 2)

        if isinstance(final_state, dict):
            result["success"] = True

            # Final answer — pull from synthesis_output or last_task_result
            synth = final_state.get("synthesis_output")
            if synth and hasattr(synth, "answer"):
                result["final_answer"] = synth.answer
            elif isinstance(synth, str):
                result["final_answer"] = synth
            else:
                last = final_state.get("last_task_result")
                if last:
                    result["final_answer"] = getattr(last, "summary", str(last))

            result["iteration_count"] = final_state.get("iteration_count", 0)
            result["supervisor_decision"] = final_state.get("supervisor_decision", "")
            result["decision_log"] = _safe_serialize(final_state.get("decision_log", []))

            evidence = final_state.get("evidence_board", [])
            result["evidence_count"] = len(evidence) if isinstance(evidence, list) else 0

            token_budget = final_state.get("token_budget", {})
            if hasattr(token_budget, "total_tokens_used"):
                result["total_tokens_used"] = token_budget.total_tokens_used
                result["total_cost_usd"] = token_budget.total_cost_usd
            elif isinstance(token_budget, dict):
                result["total_tokens_used"] = token_budget.get("total_tokens_used", 0)
                result["total_cost_usd"] = token_budget.get("total_cost_usd", 0.0)

            task_dag = final_state.get("task_dag", [])
            result["task_count"] = len(task_dag) if isinstance(task_dag, list) else 0

            result["full_state"] = _safe_serialize(final_state)
        else:
            result["error"] = f"Unexpected return type: {type(final_state)}"

    except Exception as exc:
        elapsed = time.monotonic() - start
        result["latency_s"] = round(elapsed, 2)
        result["success"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        logger.error("Q1 FAILED: %s", exc, exc_info=True)

    # ── Print summary ─────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("Q1 RESULT SUMMARY")
    logger.info("=" * 70)
    logger.info("  Success      : %s", result["success"])
    logger.info("  Latency      : %.1fs", result["latency_s"])
    logger.info("  Iterations   : %d", result["iteration_count"])
    logger.info("  Decision     : %s", result["supervisor_decision"])
    logger.info("  Evidence     : %d chunks", result["evidence_count"])
    logger.info("  Tasks        : %d", result["task_count"])
    logger.info("  Tokens used  : %d", result["total_tokens_used"])
    logger.info("  Cost         : $%.4f", result["total_cost_usd"])
    logger.info("  Error        : %s", result["error"])
    logger.info("")
    answer_preview = (result["final_answer"] or "")[:300]
    logger.info("ANSWER PREVIEW:\n%s", answer_preview)
    logger.info("=" * 70)

    if result["error"]:
        logger.warning("NOTE: Run completed with error — check q1_result.json for details")

    # ── Save JSON ─────────────────────────────────────────────────────────
    json_path = RESULTS_DIR / "q1_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Log file : %s", LOG_FILE)
    logger.info("JSON file: %s", json_path)

    return result


if __name__ == "__main__":
    asyncio.run(run_q1())
