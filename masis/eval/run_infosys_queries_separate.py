"""
run_infosys_queries_separate.py
===============================
Run Infosys golden queries one by one with separate log and JSON artifacts.

Outputs
-------
masis/eval/results/infosys_q1.log
masis/eval/results/infosys_q1_result.json
masis/eval/results/infosys_q2.log
masis/eval/results/infosys_q2_result.json
masis/eval/results/infosys_q3.log
masis/eval/results/infosys_q3_result.json
masis/eval/results/infosys_queries_report.json
masis/eval/results/infosys_queries_analysis.md

Usage
-----
python -m masis.eval.run_infosys_queries_separate
python -m masis.eval.run_infosys_queries_separate --query-index 1
python -m masis.eval.run_infosys_queries_separate --query-index 2,3
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

from masis.eval.standard_queries import load_standard_infosys_queries


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
QUERY_TIMEOUT_SECONDS = int(os.getenv("QUERY_TIMEOUT_SECONDS", "180"))


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

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    for noisy in (
        "httpcore",
        "httpx",
        "openai._base_client",
        "urllib3",
        "chromadb.telemetry",
    ):
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


def _get_infosys_queries() -> List[QueryCase]:
    return [
        QueryCase(idx=q.idx, name=q.name, query=q.query)
        for q in load_standard_infosys_queries()
    ]


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
        wrong.append(
            f"Run hit watchdog timeout at {result.get('timeout_s', QUERY_TIMEOUT_SECONDS)}s before clean completion"
        )
        if result.get("last_progress"):
            right.append("Timeout watchdog interrupted safely with partial state snapshot")
        return {
            "right": right,
            "wrong": wrong,
            "forced_pass": False,
        }

    if result.get("success"):
        right.append("Pipeline completed without runtime exception")
    else:
        wrong.append("Runtime error in graph execution")

    answer_len = len(result.get("final_answer", ""))
    if answer_len >= 120:
        right.append(f"Answer length is substantial ({answer_len} chars)")
    else:
        wrong.append(f"Answer is too short ({answer_len} chars)")

    quality = result.get("quality_scores", {}) or {}
    forced_pass = bool(quality.get("forced_pass", False))
    if forced_pass:
        wrong.append("Validator ended with forced_pass safety cap")
    else:
        right.append("Validator passed without forced safety cap")

    if result.get("validation_round", 0) > 1:
        wrong.append(
            f"Multiple validation rounds needed ({result.get('validation_round', 0)})"
        )
    else:
        right.append("Validation passed in first round")

    if result.get("iteration_count", 0) >= 15:
        wrong.append("Supervisor iteration count hit/approached safety cap")
    else:
        right.append("Supervisor stayed below iteration cap")

    return {
        "right": right,
        "wrong": wrong,
        "forced_pass": forced_pass,
    }


def _capture_last_progress(state: Dict[str, Any]) -> Dict[str, Any]:
    if not state:
        return {}
    last = state.get("last_task_result")
    return {
        "iteration_count": state.get("iteration_count", 0),
        "supervisor_decision": state.get("supervisor_decision", ""),
        "validation_round": state.get("validation_round", 0),
        "last_task_id": getattr(last, "task_id", "") if last is not None else "",
        "last_task_status": getattr(last, "status", "") if last is not None else "",
        "decision_log_length": len(state.get("decision_log", []) or []),
    }


def _infer_debug_hints(last_state: Dict[str, Any]) -> List[str]:
    hints: List[str] = []
    if not last_state:
        return ["No state snapshot captured before timeout."]

    decision_log = last_state.get("decision_log", []) or []
    recent_actions = [str(item.get("action", "")) for item in decision_log[-6:] if isinstance(item, dict)]
    if recent_actions.count("criteria_fail_slow_path") >= 2:
        hints.append("Repeated criteria failures detected; likely retry loop on same task.")
    if recent_actions.count("retry") >= 2:
        hints.append("Slow Path chose retry repeatedly; consider query rewrite strategy or modify_dag.")
    if "force_synthesize" in recent_actions:
        hints.append("Run approached safety caps; force_synthesize path is being triggered.")

    task_dag = last_state.get("task_dag", []) or []
    dag_str = " ".join(str(t) for t in task_dag)
    if "status='running'" in dag_str and "status='pending'" in dag_str:
        hints.append("Mixed running/pending tasks observed; check for blocked dependency transitions.")

    evidence = last_state.get("evidence_board", []) or []
    if evidence and any("[STUB]" in str(chunk) for chunk in evidence):
        hints.append("Stub web evidence detected; set TAVILY_API_KEY for live web results.")
    if not evidence:
        hints.append("No evidence collected before timeout; check retrieval setup and query specificity.")

    if not hints:
        hints.append("Timeout occurred without a clear single cause; inspect task-level logs and decision_log.")
    return hints


async def _run_single_case(case: QueryCase) -> Dict[str, Any]:
    from masis.graph.runner import generate_thread_id, stream_graph

    logger = logging.getLogger(__name__)
    thread_id = generate_thread_id()
    start = time.monotonic()

    logger.info("=" * 72)
    logger.info("RUNNING %s", case.name)
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
        "timeout_reason": "",
        "needs_debug": False,
        "last_progress": {},
        "debug_hints": [],
    }

    last_state: Dict[str, Any] = {}

    try:
        async def _consume_stream() -> Dict[str, Any]:
            nonlocal last_state
            async for event in stream_graph(case.query, thread_id=thread_id, stream_mode="values"):
                if isinstance(event, dict):
                    last_state = event
            return last_state

        final_state = await asyncio.wait_for(_consume_stream(), timeout=QUERY_TIMEOUT_SECONDS)
        result["success"] = True
        result["final_answer"] = _extract_answer(final_state)
        result["iteration_count"] = final_state.get("iteration_count", 0)
        result["supervisor_decision"] = final_state.get("supervisor_decision", "")
        result["validation_round"] = final_state.get("validation_round", 0)
        result["quality_scores"] = _safe_serialize(
            final_state.get("quality_scores", {}) or {}
        )
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
        result["needs_debug"] = True
        result["timeout_reason"] = (
            f"Query exceeded watchdog timeout ({QUERY_TIMEOUT_SECONDS}s) and was interrupted."
        )
        result["error"] = "TimeoutError: query watchdog interrupted run"
        result["last_progress"] = _capture_last_progress(last_state)
        result["iteration_count"] = result["last_progress"].get("iteration_count", 0)
        result["supervisor_decision"] = result["last_progress"].get("supervisor_decision", "")
        result["validation_round"] = result["last_progress"].get("validation_round", 0)
        result["debug_hints"] = _infer_debug_hints(last_state)
        result["full_state"] = _safe_serialize(last_state)
        logging.getLogger(__name__).warning(
            "Case %s timed out at %ss; last_progress=%s",
            case.name,
            QUERY_TIMEOUT_SECONDS,
            result["last_progress"],
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        logging.getLogger(__name__).error("Case %s failed: %s", case.name, exc, exc_info=True)

    result["latency_s"] = round(time.monotonic() - start, 2)
    result["evaluation"] = _quality_summary(result)
    return result


def _save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str, ensure_ascii=False)


def _save_analysis_markdown(path: Path, results: Sequence[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# Infosys Query Run Analysis")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    for r in results:
        lines.append(f"## {r['scenario']}")
        lines.append(f"- Query: {r['query']}")
        lines.append(f"- Success: {r['success']}")
        lines.append(f"- Latency: {r['latency_s']}s")
        lines.append(f"- Iterations: {r['iteration_count']}")
        lines.append(f"- Validation round: {r['validation_round']}")
        lines.append(f"- Timed out: {r.get('timed_out', False)}")
        lines.append(f"- Cost: ${r['total_cost_usd']:.4f}")
        lines.append(f"- Tokens: {r['total_tokens_used']}")
        lines.append(f"- Evidence chunks: {r['evidence_count']}")
        lines.append("")
        lines.append("What went right:")
        for item in r.get("evaluation", {}).get("right", []):
            lines.append(f"- {item}")
        lines.append("")
        lines.append("What went wrong:")
        wrong = r.get("evaluation", {}).get("wrong", [])
        if wrong:
            for item in wrong:
                lines.append(f"- {item}")
        else:
            lines.append("- No major issue detected")
        if r.get("timed_out"):
            lines.append("")
            lines.append("Timeout debug hints:")
            for hint in r.get("debug_hints", []):
                lines.append(f"- {hint}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


async def run(indices: Optional[List[int]] = None) -> Dict[str, Any]:
    from masis.eval.ingest_docs import setup_retrieval

    all_cases = _get_infosys_queries()
    selected_cases = [c for c in all_cases if indices is None or c.idx in indices]
    if not selected_cases:
        raise ValueError("No matching query indices were selected.")

    # Setup retrieval once for the full run.
    log_bootstrap = RESULTS_DIR / "infosys_setup.log"
    _configure_logging(log_bootstrap)
    logger = logging.getLogger(__name__)
    logger.info("Preparing retrieval for Infosys documents...")
    setup_retrieval()
    logger.info("Retrieval setup completed.")

    results: List[Dict[str, Any]] = []
    for case in selected_cases:
        log_path = RESULTS_DIR / f"infosys_q{case.idx}.log"
        _configure_logging(log_path)
        result = await _run_single_case(case)
        result_path = RESULTS_DIR / f"infosys_q{case.idx}_result.json"
        _save_json(result_path, result)
        logging.getLogger(__name__).info("Saved JSON: %s", result_path)
        results.append(result)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "all_success": all(r.get("success", False) for r in results),
        "forced_pass_count": sum(
            1 for r in results if r.get("evaluation", {}).get("forced_pass", False)
        ),
        "timed_out_count": sum(1 for r in results if r.get("timed_out", False)),
        "query_timeout_seconds": QUERY_TIMEOUT_SECONDS,
    }
    summary_path = RESULTS_DIR / "infosys_queries_report.json"
    _save_json(summary_path, summary)

    analysis_path = RESULTS_DIR / "infosys_queries_analysis.md"
    _save_analysis_markdown(analysis_path, results)

    print("\nRun complete:")
    print(f"- Report:   {summary_path}")
    print(f"- Analysis: {analysis_path}")
    for r in results:
        print(
            f"- {r['scenario']}: success={r['success']} latency={r['latency_s']}s "
            f"cost=${r['total_cost_usd']:.4f} forced_pass={r['evaluation']['forced_pass']} "
            f"timed_out={r.get('timed_out', False)}"
        )

    return summary


def _parse_index_arg(index_arg: Optional[str]) -> Optional[List[int]]:
    if not index_arg:
        return None
    indices: List[int] = []
    for part in index_arg.split(","):
        part = part.strip()
        if not part:
            continue
        indices.append(int(part))
    return indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Infosys queries with separate logs.")
    parser.add_argument(
        "--query-index",
        default=None,
        help="Comma-separated query indices to run (1..6). Example: --query-index 1 or 1,3,6",
    )
    args = parser.parse_args()
    asyncio.run(run(indices=_parse_index_arg(args.query_index)))
