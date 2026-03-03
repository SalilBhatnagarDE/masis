"""
run_strategic_queries.py
========================
Run strategic deep-research queries against the MASIS system and produce
structured results + analysis markdown.

Usage
-----
python -m masis.eval.run_strategic_queries
python -m masis.eval.run_strategic_queries --query-index 7
python -m masis.eval.run_strategic_queries --query-index 7,8,9,10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from masis.eval.standard_queries import load_strategic_deep_research_queries

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
QUERY_TIMEOUT_SECONDS = int(os.getenv("QUERY_TIMEOUT_SECONDS", "240"))


@dataclass
class QueryCase:
    idx: int
    name: str
    query: str


def _configure_logging(log_file: Path) -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)
    for noisy in ("httpcore", "httpx", "openai._base_client", "urllib3", "chromadb.telemetry"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _safe_serialize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def _extract_answer(final_state: Dict[str, Any]) -> str:
    synthesis = final_state.get("synthesis_output")
    if synthesis is not None and hasattr(synthesis, "answer"):
        return getattr(synthesis, "answer", "")
    if isinstance(synthesis, dict):
        return str(synthesis.get("answer") or synthesis.get("text") or "")
    if isinstance(synthesis, str):
        return synthesis
    last = final_state.get("last_task_result")
    if last is not None and hasattr(last, "summary"):
        return getattr(last, "summary", "")
    return str(last) if last is not None else ""


def _quality_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    right: List[str] = []
    wrong: List[str] = []

    if result.get("timed_out"):
        wrong.append(f"Hit watchdog timeout at {result.get('timeout_s', QUERY_TIMEOUT_SECONDS)}s")
        if result.get("last_progress"):
            right.append("Timeout interrupted safely with partial state snapshot")
        return {"right": right, "wrong": wrong, "forced_pass": False}

    if result.get("success"):
        right.append("Pipeline completed without runtime exception")
    else:
        wrong.append("Runtime error in graph execution")

    answer_len = len(result.get("final_answer", ""))
    if answer_len >= 200:
        right.append(f"Rich answer ({answer_len} chars) — suitable for deep-research query")
    elif answer_len >= 120:
        right.append(f"Adequate answer length ({answer_len} chars)")
    else:
        wrong.append(f"Answer too short for strategic query ({answer_len} chars)")

    quality = result.get("quality_scores", {}) or {}
    forced_pass = bool(quality.get("forced_pass", False))
    if forced_pass:
        wrong.append("Validator ended with forced_pass safety cap")
    else:
        right.append("Validator passed without forced safety cap")

    evidence_count = result.get("evidence_count", 0)
    if evidence_count >= 6:
        right.append(f"Strong evidence retrieval ({evidence_count} chunks) — multi-dimensional coverage")
    elif evidence_count >= 3:
        right.append(f"Adequate evidence ({evidence_count} chunks)")
    else:
        wrong.append(f"Sparse evidence ({evidence_count} chunks) — strategic query needs more coverage")

    task_count = result.get("task_count", 0)
    if task_count >= 4:
        right.append(f"Complex DAG with {task_count} tasks — demonstrates parallel agent orchestration")
    elif task_count >= 3:
        right.append(f"Standard 3-task DAG (researcher→skeptic→synthesizer)")
    else:
        wrong.append(f"Minimal DAG ({task_count} tasks) — may indicate planning failure")

    if result.get("validation_round", 0) > 1:
        wrong.append(f"Multiple validation rounds ({result.get('validation_round', 0)})")
    else:
        right.append("Passed validation in first round")

    if result.get("iteration_count", 0) >= 15:
        wrong.append("Hit supervisor iteration cap")
    else:
        right.append(f"Supervisor used {result.get('iteration_count', 0)} turns (within limit)")

    return {"right": right, "wrong": wrong, "forced_pass": forced_pass}


async def _run_single_case(case: QueryCase) -> Dict[str, Any]:
    from masis.graph.runner import generate_thread_id, stream_graph

    logger = logging.getLogger(__name__)
    thread_id = generate_thread_id()
    start = time.monotonic()

    logger.info("=" * 72)
    logger.info("STRATEGIC QUERY: %s", case.name)
    logger.info("Query: %s", case.query)
    logger.info("Thread: %s", thread_id)
    logger.info("=" * 72)

    result: Dict[str, Any] = {
        "query_index": case.idx,
        "scenario": case.name,
        "query": case.query,
        "thread_id": thread_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "success": False,
        "error": None,
        "final_answer": "",
        "latency_s": 0.0,
        "iteration_count": 0,
        "supervisor_decision": "",
        "validation_round": 0,
        "quality_scores": {},
        "decision_log": [],
        "decision_log_length": 0,
        "evidence_count": 0,
        "task_count": 0,
        "total_tokens_used": 0,
        "total_cost_usd": 0.0,
        "full_state": {},
        "timed_out": False,
        "timeout_s": QUERY_TIMEOUT_SECONDS,
    }

    last_state: Dict[str, Any] = {}

    try:
        async def _consume() -> Dict[str, Any]:
            nonlocal last_state
            async for event in stream_graph(case.query, thread_id=thread_id, stream_mode="values"):
                if isinstance(event, dict):
                    last_state = event
            return last_state

        final_state = await asyncio.wait_for(_consume(), timeout=QUERY_TIMEOUT_SECONDS)
        result["success"] = True
        result["final_answer"] = _extract_answer(final_state)
        result["iteration_count"] = final_state.get("iteration_count", 0)
        result["supervisor_decision"] = final_state.get("supervisor_decision", "")
        result["validation_round"] = final_state.get("validation_round", 0)
        result["quality_scores"] = _safe_serialize(final_state.get("quality_scores", {}) or {})
        result["decision_log"] = _safe_serialize(final_state.get("decision_log", []))
        result["decision_log_length"] = len(result["decision_log"])
        result["evidence_count"] = len(final_state.get("evidence_board", []) or [])
        result["task_count"] = len(final_state.get("task_dag", []) or [])

        budget = final_state.get("token_budget")
        if hasattr(budget, "total_tokens_used"):
            result["total_tokens_used"] = getattr(budget, "total_tokens_used", 0)
            result["total_cost_usd"] = getattr(budget, "total_cost_usd", 0.0)
        elif isinstance(budget, dict):
            result["total_tokens_used"] = budget.get("total_tokens_used", 0)
            result["total_cost_usd"] = budget.get("total_cost_usd", 0.0)

        result["full_state"] = _safe_serialize(final_state)

    except asyncio.TimeoutError:
        result["timed_out"] = True
        result["error"] = f"TimeoutError: exceeded {QUERY_TIMEOUT_SECONDS}s watchdog"
        result["full_state"] = _safe_serialize(last_state)
        result["iteration_count"] = last_state.get("iteration_count", 0)
        logger.warning("Case %s timed out at %ss", case.name, QUERY_TIMEOUT_SECONDS)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        logger.error("Case %s failed: %s", case.name, exc, exc_info=True)

    result["latency_s"] = round(time.monotonic() - start, 2)
    result["evaluation"] = _quality_summary(result)
    return result


def _save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str, ensure_ascii=False)


def _save_analysis_markdown(path: Path, results: Sequence[Dict[str, Any]]) -> None:
    lines: List[str] = [
        "# Strategic Deep-Research Query Analysis",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Overview",
        "",
        "These queries are designed to exercise MASIS's multi-agent DAG planning,",
        "parallel evidence gathering, skeptic validation, and cross-dimensional synthesis.",
        "",
    ]
    for r in results:
        lines += [
            f"## {r['scenario']}: {r['query'][:80]}{'...' if len(r['query']) > 80 else ''}",
            "",
            f"- **Success:** {r['success']}",
            f"- **Latency:** {r['latency_s']}s",
            f"- **Iterations:** {r['iteration_count']}",
            f"- **Tasks in DAG:** {r['task_count']}",
            f"- **Evidence chunks:** {r['evidence_count']}",
            f"- **Validation round:** {r['validation_round']}",
            f"- **Cost:** ${r['total_cost_usd']:.4f}",
            f"- **Tokens:** {r['total_tokens_used']}",
            f"- **Timed out:** {r.get('timed_out', False)}",
            "",
            "**Answer preview:**",
            "",
            f"> {r['final_answer'][:600]}{'...' if len(r['final_answer']) > 600 else ''}",
            "",
            "**What went right:**",
        ]
        for item in r.get("evaluation", {}).get("right", []):
            lines.append(f"- {item}")
        lines += ["", "**What went wrong:**"]
        wrong = r.get("evaluation", {}).get("wrong", [])
        for item in wrong:
            lines.append(f"- {item}")
        if not wrong:
            lines.append("- None detected")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


async def run(indices: Optional[List[int]] = None) -> Dict[str, Any]:
    from masis.eval.ingest_docs import setup_retrieval

    all_cases = [
        QueryCase(idx=q.idx, name=q.name, query=q.query)
        for q in load_strategic_deep_research_queries()
    ]
    selected = [c for c in all_cases if indices is None or c.idx in indices]
    if not selected:
        raise ValueError(f"No matching indices. Available: {[c.idx for c in all_cases]}")

    log_bootstrap = RESULTS_DIR / "strategic_setup.log"
    _configure_logging(log_bootstrap)
    logger = logging.getLogger(__name__)
    logger.info("Setting up retrieval for strategic query run...")
    setup_retrieval()
    logger.info("Retrieval ready.")

    results: List[Dict[str, Any]] = []
    for case in selected:
        log_path = RESULTS_DIR / f"strategic_{case.name.lower()}.log"
        _configure_logging(log_path)
        result = await _run_single_case(case)
        result_path = RESULTS_DIR / f"strategic_{case.name.lower()}_result.json"
        _save_json(result_path, result)
        logging.getLogger(__name__).info("Saved: %s", result_path)
        results.append(result)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "all_success": all(r.get("success", False) for r in results),
        "timeout_seconds": QUERY_TIMEOUT_SECONDS,
    }
    _save_json(RESULTS_DIR / "strategic_queries_report.json", summary)

    analysis_path = RESULTS_DIR / "strategic_queries_analysis.md"
    _save_analysis_markdown(analysis_path, results)

    print("\n--- Strategic Query Run Complete ---")
    for r in results:
        status = "OK" if r["success"] else ("TIMEOUT" if r.get("timed_out") else "FAIL")
        print(
            f"  {r['scenario']} [{status}] latency={r['latency_s']}s "
            f"tasks={r['task_count']} evidence={r['evidence_count']} "
            f"cost=${r['total_cost_usd']:.4f}"
        )
    print(f"\nAnalysis: {analysis_path}")
    return summary


def _parse_indices(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run strategic deep-research queries.")
    parser.add_argument("--query-index", default=None, help="Comma-separated indices (7..10)")
    args = parser.parse_args()
    asyncio.run(run(indices=_parse_indices(args.query_index)))
