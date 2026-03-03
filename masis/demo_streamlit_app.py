"""
Streamlit demo app for MASIS.

Run:
streamlit run masis/demo_streamlit_app.py
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import streamlit as st

from masis.config import model_routing as model_routing_cfg
from masis.eval.ingest_docs import setup_retrieval
from masis.eval.standard_queries import load_standard_infosys_queries
from masis.graph.runner import stream_graph


RESULTS_DIR = Path(__file__).parent / "eval" / "results"
QUERY_TIMEOUT_SECONDS = int(os.getenv("QUERY_TIMEOUT_SECONDS", "300"))


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def _pick_answer(state: Dict[str, Any]) -> str:
    synth = state.get("synthesis_output")
    if synth is not None and hasattr(synth, "answer"):
        return getattr(synth, "answer", "")
    if isinstance(synth, dict):
        return str(synth.get("answer") or synth.get("text") or "")
    if isinstance(synth, str):
        return synth
    last = state.get("last_task_result")
    if last is not None and hasattr(last, "summary"):
        return getattr(last, "summary", "")
    return ""


def _compact_text(value: Any, max_len: int = 220) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return f"{text[:max_len].rstrip()}..."


def _obj_field(obj: Any, key: str, default: Any = "") -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_last_result(state: Dict[str, Any]) -> Dict[str, Any]:
    last = state.get("last_task_result")
    if last is None:
        return {}

    criteria = getattr(last, "criteria_result", {})
    if not isinstance(criteria, dict):
        criteria = {"raw": _compact_text(criteria, 120)}

    return {
        "task_id": getattr(last, "task_id", ""),
        "agent_type": getattr(last, "agent_type", ""),
        "status": getattr(last, "status", ""),
        "summary": _compact_text(getattr(last, "summary", ""), 240),
        "criteria_result": criteria,
    }


def _task_dag_rows(task_dag: List[Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for t in task_dag or []:
        deps = _obj_field(t, "dependencies", []) or []
        rows.append(
            {
                "task_id": _obj_field(t, "task_id", ""),
                "type": _obj_field(t, "type", ""),
                "parallel_group": _obj_field(t, "parallel_group", ""),
                "status": _obj_field(t, "status", ""),
                "dependencies": ", ".join(deps),
                "query": _compact_text(_obj_field(t, "query", ""), 140),
                "acceptance_criteria": _compact_text(
                    _obj_field(t, "acceptance_criteria", ""), 120
                ),
                "result_summary": _compact_text(_obj_field(t, "result_summary", ""), 140),
            }
        )
    return rows


def _evidence_rows(evidence_board: List[Any], limit: int = 30) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, c in enumerate(evidence_board or []):
        if idx >= limit:
            break
        rows.append(
            {
                "chunk_id": _obj_field(c, "chunk_id", ""),
                "doc_id": _obj_field(c, "doc_id", ""),
                "source_label": _obj_field(c, "source_label", ""),
                "retrieval_score": _obj_field(c, "retrieval_score", 0.0),
                "rerank_score": _obj_field(c, "rerank_score", 0.0),
                "text_preview": _compact_text(_obj_field(c, "text", ""), 180),
            }
        )
    return rows


def _task_status_snapshot(task_dag: List[Any]) -> str:
    counts = {"pending": 0, "running": 0, "done": 0, "failed": 0}
    for task in task_dag or []:
        status = getattr(task, "status", "")
        if status in counts:
            counts[status] += 1
    return (
        f"pending={counts['pending']} | running={counts['running']} | "
        f"done={counts['done']} | failed={counts['failed']}"
    )


def _apply_model_profile(profile: str) -> None:
    """Switch runtime model routing for demo without editing .env."""
    if profile == "High Quality (demo)":
        model_routing_cfg.MODEL_ROUTING["supervisor_plan"] = "gpt-5.2"
        model_routing_cfg.MODEL_ROUTING["supervisor_slow"] = "gpt-5.2"
        model_routing_cfg.MODEL_ROUTING["researcher"] = "gpt-4.1"
        model_routing_cfg.MODEL_ROUTING["synthesizer"] = "gpt-4.1"
        model_routing_cfg.MODEL_ROUTING["skeptic_llm"] = "o3-mini"
        model_routing_cfg.MODEL_ROUTING["ambiguity_detector"] = "gpt-4.1"
    else:
        model_routing_cfg.MODEL_ROUTING["supervisor_plan"] = "gpt-4.1"
        model_routing_cfg.MODEL_ROUTING["supervisor_slow"] = "gpt-4.1"
        model_routing_cfg.MODEL_ROUTING["researcher"] = "gpt-4.1-mini"
        model_routing_cfg.MODEL_ROUTING["synthesizer"] = "gpt-4.1"
        model_routing_cfg.MODEL_ROUTING["skeptic_llm"] = "o3-mini"
        model_routing_cfg.MODEL_ROUTING["ambiguity_detector"] = "gpt-4.1-mini"


def _validator_event_summary(quality_scores: Dict[str, Any]) -> str:
    _ = quality_scores
    return "Validator completed quality check."


def _has_stub_web_evidence(state: Dict[str, Any]) -> bool:
    for chunk in state.get("evidence_board", []) or []:
        text = str(getattr(chunk, "text", ""))
        doc_id = str(getattr(chunk, "doc_id", ""))
        if "[STUB]" in text or "stub.masis.test" in doc_id:
            return True
    return False


def _capture_last_progress(state: Dict[str, Any]) -> Dict[str, Any]:
    if not state:
        return {}
    last = state.get("last_task_result")
    return {
        "iteration_count": state.get("iteration_count", 0),
        "supervisor_decision": state.get("supervisor_decision", ""),
        "validation_round": state.get("validation_round", 0),
        "last_task_id": _obj_field(last, "task_id", ""),
        "last_task_status": _obj_field(last, "status", ""),
        "decision_log_length": len(state.get("decision_log", []) or []),
    }


def _infer_debug_hints(state: Dict[str, Any]) -> List[str]:
    if not state:
        return ["No state snapshot captured before timeout."]
    hints: List[str] = []
    decision_log = state.get("decision_log", []) or []
    recent_actions = [str(x.get("action", "")) for x in decision_log[-8:] if isinstance(x, dict)]
    if recent_actions.count("criteria_fail_slow_path") >= 2:
        hints.append("Repeated criteria failures detected on recent turns.")
    if recent_actions.count("retry") >= 2:
        hints.append("Slow Path selected retry multiple times; consider modify_dag.")
    if "force_synthesize" in recent_actions:
        hints.append("Safety cap path observed (force_synthesize).")

    task_dag = state.get("task_dag", []) or []
    dag_text = " ".join(str(t) for t in task_dag)
    if "status='running'" in dag_text and "status='pending'" in dag_text:
        hints.append("Potential blocked dependency/running-task deadlock pattern.")

    evidence = state.get("evidence_board", []) or []
    if evidence and any("[STUB]" in str(c) for c in evidence):
        hints.append("Stub web evidence detected; set TAVILY_API_KEY for live web data.")
    if not evidence:
        hints.append("No evidence collected yet; check retrieval setup and query scope.")

    if not hints:
        hints.append("Timeout reached without a single dominant cause; inspect decision_log.")
    return hints


async def _run_query_with_trace(
    query: str,
    on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
    enable_ambiguity_hitl: bool = False,
) -> Dict[str, Any]:
    trace_rows: List[Dict[str, Any]] = []
    timeline_events: List[Dict[str, Any]] = []
    final_state: Dict[str, Any] = {}
    thread_id = f"streamlit-{uuid.uuid4()}"
    start = time.monotonic()
    seen_agent_events = set()
    prev_decision = ""
    prev_validation_round = 0

    timed_out = False
    timeout_reason = ""

    async def _consume_stream() -> None:
        nonlocal final_state, prev_decision, prev_validation_round
        async for state in stream_graph(
            query,
            thread_id=thread_id,
            stream_mode="values",
            initial_state_overrides={"enable_ambiguity_hitl": enable_ambiguity_hitl},
        ):
            if not isinstance(state, dict):
                continue
            final_state = state
            elapsed_s = round(time.monotonic() - start, 2)
            last = _extract_last_result(state)
            decision = state.get("supervisor_decision", "")
            validation_round = int(state.get("validation_round", 0) or 0)
            latest_event = None

            row = {
                "step": len(trace_rows) + 1,
                "t+sec": elapsed_s,
                "supervisor_iteration": state.get("iteration_count", 0),
                "supervisor_decision": decision,
                "validation_round": validation_round,
                "last_task_id": last.get("task_id", ""),
                "last_task_status": last.get("status", ""),
                "task_dag_status": _task_status_snapshot(state.get("task_dag", [])),
                "evidence_chunks": len(state.get("evidence_board", []) or []),
            }
            trace_rows.append(row)

            if decision and decision != prev_decision:
                prev_decision = decision
                latest_event = {
                    "t+sec": elapsed_s,
                    "actor": "supervisor",
                    "task_id": "-",
                    "status": "decision",
                    "summary": f"Routing decision -> {decision}",
                }
                timeline_events.append(latest_event)

            if last.get("task_id"):
                key = (last.get("task_id"), last.get("status"), row["supervisor_iteration"])
                if key not in seen_agent_events:
                    seen_agent_events.add(key)
                    latest_event = {
                        "t+sec": elapsed_s,
                        "actor": last.get("agent_type", ""),
                        "task_id": last.get("task_id", ""),
                        "status": last.get("status", ""),
                        "summary": last.get("summary", ""),
                    }
                    timeline_events.append(latest_event)

            if validation_round > prev_validation_round:
                prev_validation_round = validation_round
                latest_event = {
                    "t+sec": elapsed_s,
                    "actor": "validator",
                    "task_id": "-",
                    "status": "scored",
                    "summary": _validator_event_summary(state.get("quality_scores", {}) or {}),
                }
                timeline_events.append(latest_event)

            if on_step is not None:
                on_step(
                    {
                        "row": row,
                        "trace_rows": trace_rows,
                        "timeline_events": timeline_events,
                        "latest_event": latest_event,
                    }
                )

    try:
        await asyncio.wait_for(_consume_stream(), timeout=QUERY_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        timed_out = True
        timeout_reason = (
            f"Run exceeded watchdog timeout ({QUERY_TIMEOUT_SECONDS}s) and was interrupted."
        )

    return {
        "thread_id": thread_id,
        "latency_s": round(time.monotonic() - start, 2),
        "trace_rows": trace_rows,
        "timeline_events": timeline_events,
        "final_state": final_state,
        "timed_out": timed_out,
        "timeout_s": QUERY_TIMEOUT_SECONDS,
        "timeout_reason": timeout_reason,
        "last_progress": _capture_last_progress(final_state),
        "debug_hints": _infer_debug_hints(final_state) if timed_out else [],
    }


def _load_latest_artifacts() -> List[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("infosys_q*_result.json"), key=lambda p: p.name)


def main() -> None:
    st.set_page_config(page_title="MASIS Demo", layout="wide")
    st.title("MASIS Multi-Agent Demo")
    st.caption("Run Infosys queries with live orchestration trace and partial agent outputs.")

    presets = load_standard_infosys_queries()
    preset_options = [f"{q.name}: {q.query}" for q in presets]
    default_option = preset_options[0]

    if "preset_option" not in st.session_state:
        st.session_state["preset_option"] = default_option
    if "query_text" not in st.session_state:
        st.session_state["query_text"] = presets[0].query
    if "last_preset_option" not in st.session_state:
        st.session_state["last_preset_option"] = st.session_state["preset_option"]

    def _on_preset_change() -> None:
        selected = st.session_state.get("preset_option", default_option)
        for item in presets:
            label = f"{item.name}: {item.query}"
            if label == selected:
                st.session_state["query_text"] = item.query
                st.session_state["last_preset_option"] = selected
                break

    with st.sidebar:
        st.subheader("Query")
        st.selectbox(
            "Preset",
            options=preset_options,
            index=preset_options.index(st.session_state["preset_option"])
            if st.session_state["preset_option"] in preset_options
            else 0,
            key="preset_option",
            on_change=_on_preset_change,
        )
        model_profile = st.selectbox(
            "Model profile",
            options=["Balanced (default)", "High Quality (demo)"],
            index=0,
            help="High Quality uses stronger models for researcher/ambiguity, with higher cost/latency.",
        )
        enable_ambiguity_hitl = st.checkbox(
            "Enable ambiguity HITL",
            value=False,
            help="If enabled, ambiguous queries (for example Q5) can pause for clarification.",
        )
        query = st.text_area("Query text", key="query_text", height=120)
        run_btn = st.button("Run Query", type="primary")
        setup_btn = st.button("Setup Retrieval (Infosys index)")

    st.info(
        "Retrieval setup loads/reuses the Infosys retrieval stack: "
        "Chroma vector index + BM25 keyword index. "
        "Run it once after app start, then reuse for all queries."
    )
    st.caption(
        f"Watchdog enabled: each run auto-interrupts after {QUERY_TIMEOUT_SECONDS}s "
        "if not complete."
    )
    if setup_btn:
        try:
            with st.spinner(
                "Preparing retrieval: loading Infosys docs, reusing/building Chroma + BM25 indices..."
            ):
                setup_retrieval()
            st.success("Retrieval setup complete. Retrieval layer is ready for query runs.")
        except Exception as exc:
            st.error(
                "Retrieval setup failed. Ensure OPENAI_API_KEY is configured "
                "(CHROMA_OPENAI_API_KEY is optional and auto-falls back)."
            )
            st.code(str(exc))

    if run_btn:
        _apply_model_profile(model_profile)
        st.caption(f"Active model profile: {model_profile}")

        live_progress = st.empty()
        live_trace = st.empty()
        live_timeline = st.empty()

        def _on_step(payload: Dict[str, Any]) -> None:
            row = payload["row"]
            live_progress.info(
                "Live status: step={step}, t+{t}s, supervisor_iteration={it}, "
                "decision={dec}, validation_round={vr}, last_task={task}:{status}".format(
                    step=row["step"],
                    t=row["t+sec"],
                    it=row["supervisor_iteration"],
                    dec=row["supervisor_decision"] or "-",
                    vr=row["validation_round"],
                    task=row["last_task_id"] or "-",
                    status=row["last_task_status"] or "-",
                )
            )

            trace_df = pd.DataFrame(payload["trace_rows"][-10:])
            live_trace.dataframe(trace_df, use_container_width=True, hide_index=True)

            timeline_df = pd.DataFrame(payload["timeline_events"][-8:])
            if not timeline_df.empty:
                live_timeline.dataframe(timeline_df, use_container_width=True, hide_index=True)

        with st.spinner(
            "Running MASIS workflow (Supervisor -> Executor -> Validator) and streaming step trace..."
        ):
            run_result = _run_async(
                _run_query_with_trace(
                    query,
                    on_step=_on_step,
                    enable_ambiguity_hitl=enable_ambiguity_hitl,
                )
            )

        final_state = run_result["final_state"]
        answer = _pick_answer(final_state)
        decision_log = final_state.get("decision_log", []) or []
        timeline_events = run_result.get("timeline_events", [])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latency (s)", run_result["latency_s"])
        c2.metric("Supervisor Iterations", final_state.get("iteration_count", 0))
        c3.metric("Validation Rounds", final_state.get("validation_round", 0))
        c4.metric("Evidence Chunks", len(final_state.get("evidence_board", []) or []))

        st.info(
            "`Supervisor Iterations` are control-loop turns (plan/check/route). "
            "`Validation Rounds` are only validator retry rounds."
        )
        if run_result.get("timed_out"):
            st.error(run_result.get("timeout_reason", "Run timed out."))
            st.caption(f"Last progress: {run_result.get('last_progress', {})}")
            hints = run_result.get("debug_hints", [])
            if hints:
                st.warning("Timeout debug hints:\n- " + "\n- ".join(hints))

        if final_state.get("validation_round", 0) == 1:
            st.success("Validation passed in one round.")
        if _has_stub_web_evidence(final_state):
            st.warning(
                "Web-search evidence includes STUB results (likely missing TAVILY_API_KEY). "
                "Quality may be limited for web-dependent queries."
            )

        st.subheader("Agent / Control Timeline")
        if timeline_events:
            st.dataframe(pd.DataFrame(timeline_events), use_container_width=True, hide_index=True)
        else:
            st.info("No timeline events captured.")

        st.subheader("Supervisor Plan (DAG)")
        st.caption(f"Stop condition: {_compact_text(final_state.get('stop_condition', ''), 220)}")
        dag_rows = _task_dag_rows(final_state.get("task_dag", []) or [])
        if dag_rows:
            st.dataframe(pd.DataFrame(dag_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No task_dag available.")

        st.subheader("Evidence Board Snapshot")
        evidence = final_state.get("evidence_board", []) or []
        st.caption(f"Showing up to 30 chunks out of {len(evidence)} total.")
        ev_rows = _evidence_rows(evidence, limit=30)
        if ev_rows:
            st.dataframe(pd.DataFrame(ev_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No evidence chunks collected.")

        st.subheader("Step Trace")
        st.dataframe(pd.DataFrame(run_result["trace_rows"]), use_container_width=True, hide_index=True)

        st.subheader("Final Answer")
        st.write(answer or "No answer produced.")

        st.subheader("Decision Log")
        st.json(decision_log)

        with st.expander("Raw Final State"):
            st.code(str(final_state)[:20000])

    st.divider()
    st.subheader("Latest Saved Artifacts")
    artifact_files = _load_latest_artifacts()
    if artifact_files:
        for path in artifact_files:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                st.markdown(f"**{path.name}**")
                st.write(
                    f"Success: {payload.get('success')} | "
                    f"Latency: {payload.get('latency_s')}s | "
                    f"Supervisor Iterations: {payload.get('iteration_count')} | "
                    f"Validation Rounds: {payload.get('validation_round', 0)} | "
                    f"Timed out: {payload.get('timed_out', False)}"
                )
            except Exception:
                st.markdown(f"**{path.name}**")
    else:
        st.info("No saved `infosys_q*_result.json` artifacts found yet.")


if __name__ == "__main__":
    main()
