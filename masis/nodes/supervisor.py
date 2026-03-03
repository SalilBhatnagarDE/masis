"""
masis.nodes.supervisor
======================
Supervisor node  --  the "brain" of the MAISS system (ENG-04, MF-SUP-01 through MF-SUP-17).

The Supervisor operates in three modes:
    MODE 1  --  PLAN   (iteration_count == 0): gpt-4.1 decomposes query into DAG via
                     with_structured_output(TaskPlan). Always Slow Path.
    MODE 2  --  FAST   (rule-based, $0, <10ms): Checks budgets, iteration limits,
                     repetition, and per-task acceptance criteria.
    MODE 3  --  SLOW   (gpt-4.1, ~$0.015): Only when Fast Path criteria FAIL. Decides
                     retry / modify_dag / escalate / force_synthesize / stop.

Every decision is logged to state["decision_log"] (MF-SUP-17).
Context sent to the LLM is filtered to summaries only  --  never full evidence (MF-SUP-14).

Public API
----------
supervisor_node(state)        --  LangGraph node entry point
plan_dag(state)               --  first-turn DAG planning (MODE 1)
monitor_and_route(state)      --  Fast Path monitor (MODE 2)
supervisor_slow_path(state)   --  Slow Path LLM decision (MODE 3)
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase 0 imports with graceful stubs
# ---------------------------------------------------------------------------

try:
    from masis.schemas.models import (
        AgentOutput,
        BudgetTracker,
        EvidenceChunk,
        ModifyDagSpec,
        SupervisorDecision,
        TaskNode,
        TaskPlan,
    )
    from masis.schemas.thresholds import SAFETY_LIMITS, TOOL_LIMITS
except ImportError:
    # Stub types used when Phase 0 schemas are not yet installed.
    # This allows the file to be parsed and tested independently.
    logger.warning("masis.schemas not found  --  using stub types for supervisor.py")

    class AgentOutput:  # type: ignore[no-redef]
        task_id: str = ""
        agent_type: str = ""
        status: str = "success"
        summary: str = ""
        evidence: list = []
        criteria_result: dict = {}
        tokens_used: int = 0
        cost_usd: float = 0.0

    class BudgetTracker:  # type: ignore[no-redef]
        remaining: int = 100_000
        total_tokens_used: int = 0
        total_cost_usd: float = 0.0
        api_calls: dict = {}
        start_time: float = 0.0

        def is_exhausted(self) -> bool:
            return self.remaining <= 0

        def wall_clock_seconds(self) -> float:
            return 0.0

        def add(self, tokens: int, cost: float, agent_type: str = "") -> "BudgetTracker":
            return self

    class TaskNode:  # type: ignore[no-redef]
        task_id: str = ""
        type: str = ""
        query: str = ""
        dependencies: list = []
        parallel_group: int = 1
        acceptance_criteria: str = ""
        status: str = "pending"
        result_summary: str = ""
        retry_count: int = 0

    class TaskPlan:  # type: ignore[no-redef]
        tasks: list = []
        stop_condition: str = ""

    class SupervisorDecision:  # type: ignore[no-redef]
        action: str = "stop"
        reason: str = ""
        retry_spec: Any = None
        modify_dag_spec: Any = None
        escalate_spec: Any = None

    class ModifyDagSpec:  # type: ignore[no-redef]
        add: list = []
        remove: list = []

    SAFETY_LIMITS: dict = {  # type: ignore[misc]
        "MAX_SUPERVISOR_TURNS": 15,
        "MAX_WALL_CLOCK_SECONDS": 300,
        "REPETITION_COSINE_THRESHOLD": 0.90,
        "MAX_VALIDATION_ROUNDS": 2,
    }
    TOOL_LIMITS: dict = {}  # type: ignore[misc]

# ---------------------------------------------------------------------------
# Optional LangGraph imports
# ---------------------------------------------------------------------------

try:
    from langgraph.types import interrupt
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False
    logger.warning("langgraph not installed  --  interrupt() calls will raise RuntimeError")

    def interrupt(payload: Any) -> None:  # type: ignore[misc]
        raise RuntimeError(
            "LangGraph is not installed. Cannot call interrupt(). "
            "Install with: pip install langgraph"
        )

# ---------------------------------------------------------------------------
# Optional LangChain / OpenAI imports
# ---------------------------------------------------------------------------

try:
    from langchain_openai import ChatOpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    logger.warning("langchain_openai not installed  --  LLM calls will raise RuntimeError")
    ChatOpenAI = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Internal utilities (may fall back to lightweight implementations)
# ---------------------------------------------------------------------------

try:
    from masis.utils import (
        all_tasks_done,
        check_agent_criteria,
        find_task,
        get_next_ready_tasks,
        is_repetitive,
        log_decision as _append_decision_log,
    )

    def log_decision(state: Dict[str, Any], entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Append a supervisor decision entry to state['decision_log'].

        The safety_utils helper uses (decision_log, entry) signature with canonical
        keys "decision"/"cost". Supervisor entries internally use "action"/"cost_usd".
        Normalize both forms here to keep existing supervisor call sites unchanged.
        """
        normalized = dict(entry)
        if "decision" not in normalized and "action" in normalized:
            normalized["decision"] = normalized.get("action")
        if "cost" not in normalized and "cost_usd" in normalized:
            normalized["cost"] = normalized.get("cost_usd", 0.0)
        return _append_decision_log(state.get("decision_log", []), normalized)
except ImportError:
    logger.warning("masis.utils.dag_utils not found  --  using inline fallbacks")

    def get_next_ready_tasks(dag: List[Any]) -> List[Any]:  # type: ignore[misc]
        """Return tasks whose dependencies are all 'done' (fallback)."""
        done_ids = {t.task_id for t in dag if t.status == "done"}
        ready = [
            t for t in dag
            if t.status == "pending"
            and all(dep in done_ids for dep in (t.dependencies or []))
        ]
        if not ready:
            return []
        min_group = min(t.parallel_group for t in ready)
        return [t for t in ready if t.parallel_group == min_group]

    def all_tasks_done(dag: List[Any]) -> bool:  # type: ignore[misc]
        return all(t.status in ("done", "failed") for t in dag)

    def find_task(dag: List[Any], task_id: str) -> Optional[Any]:  # type: ignore[misc]
        return next((t for t in dag if t.task_id == task_id), None)

    def is_repetitive(state: Dict[str, Any]) -> bool:  # type: ignore[misc]
        return False  # conservative fallback

    def check_agent_criteria(task: Any, result: Any) -> str:  # type: ignore[misc]
        return "PASS"  # conservative fallback

    def log_decision(state: Dict[str, Any], entry: Dict[str, Any]) -> List[Dict[str, Any]]:  # type: ignore[misc]
        existing = list(state.get("decision_log", []))
        existing.append(entry)
        return existing

# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------

try:
    from masis.config.model_routing import get_model
except ImportError:
    import os

    def get_model(role: str, override: Optional[str] = None) -> str:  # type: ignore[misc]
        if override:
            return override
        defaults = {
            "supervisor_plan": "gpt-4.1",
            "supervisor_slow": "gpt-4.1",
        }
        return os.getenv("MODEL_SUPERVISOR", defaults.get(role, "gpt-4.1"))

try:
    from masis.infra.hitl import ambiguity_detector
    _HITL_AVAILABLE = True
except Exception:
    _HITL_AVAILABLE = False
    ambiguity_detector = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_SUPERVISOR_TURNS: int = SAFETY_LIMITS["MAX_SUPERVISOR_TURNS"]
MAX_WALL_CLOCK_SECONDS: int = SAFETY_LIMITS["MAX_WALL_CLOCK_SECONDS"]
MAX_TASK_RETRIES_PER_NODE: int = int(SAFETY_LIMITS.get("MAX_TASK_RETRIES_PER_NODE", 3))

# ---------------------------------------------------------------------------
# Planning prompt (MF-SUP-03)
# ---------------------------------------------------------------------------

SUPERVISOR_PLAN_PROMPT = """You are the MASIS Chief Research Orchestrator.

Goal:
Build a high-quality TaskPlan DAG that is clear, traceable, and complete.
Return a TaskPlan only. Do not answer the user question directly.

Available task types:
- researcher
- web_search
- skeptic
- synthesizer

Planning contract:
1. Intent detection:
   Classify the query into one or more intents:
   factual_lookup, comparative_analysis, multi_dimension_analysis, risk_or_swot_analysis, trend_analysis.
2. Scope extraction:
   Extract entities, dimensions, timeframe, geography/business-unit constraints, and comparison targets.
3. Coverage matrix:
   Build required coverage as entity x dimension cells.
   Every required cell must map to at least one researcher task.
4. Granular decomposition:
   Researcher tasks must be atomic and measurable.
   Do not merge unrelated dimensions in one researcher task.
   For comparative queries, do not merge multiple entities into one researcher task.
5. Research query rewriting:
   Each researcher task query should be retrieval-oriented and explicit:
   include entity + dimension + timeframe + business synonyms.
   Keep each query concise and specific.
6. Flow topology:
   - parallel_group=1 for initial research tasks
   - parallel_group=2 for skeptic (depends on all required research)
   - parallel_group=3 for synthesizer (depends on skeptic)
   Always end with skeptic then synthesizer.
7. Stop condition:
   Must encode completion criteria for full requested coverage.
   If coverage is incomplete, final synthesis must explicitly disclose missing cells.
8. Strategic planning discipline:
   For broad strategy queries (compare, SWOT, risks, performance drivers), decompose by:
   entity -> metric/dimension -> timeframe.
   Use one atomic researcher per cell where possible.
9. Anchor discipline:
   Keep task queries aligned to target entities in the original query.
   Never leave placeholders like [Organization Name] in final task queries.
   Every task query must be executable as written.

Internal-first policy:
- Prefer internal researcher tasks first.
- Add web_search in initial plan only when query explicitly requires external/competitor/latest data.
- For external entities, prefer internal probes first and let Slow Path add targeted web_search if needed.

Acceptance criteria templates:
- researcher: ">=2 chunks, pass_rate>=0.30, self_rag=grounded"
- web_search: ">=1 relevant result, no timeout"
- skeptic: "claims_unsupported<=2, claims_contradicted==0, logical_gaps_count<=3, overall_confidence>=0.65"
- synthesizer: "citations_count>=claims_count, all_citations_in_evidence_board==true"

Few-shot planning patterns (compact):
- factual_lookup -> 1 researcher -> skeptic -> synthesizer
- multi_dimension_analysis -> 1 researcher per dimension (parallel) -> skeptic -> synthesizer
- comparative_analysis -> 1 researcher per entity x dimension cell (parallel) -> skeptic -> synthesizer
- strategic_swot_analysis -> one researcher per entity x SWOT axis, then skeptic, then synthesizer

Quality checklist before returning TaskPlan:
- All user-requested entities are covered.
- All user-requested dimensions are covered.
- Dependencies are valid and acyclic.
- Acceptance criteria are measurable strings.
- stop_condition is explicit and testable.

Now produce TaskPlan for:
Query: "{user_query}"
"""

# ---------------------------------------------------------------------------
# Slow-path LLM decision prompt (MF-SUP-09 through MF-SUP-13)
# ---------------------------------------------------------------------------

SUPERVISOR_SLOW_PROMPT = """You are the Supervisor of a multi-agent research system.

A task failed criteria or routing is blocked.
Choose the best next control action for quality, coverage, and cost.

CONTEXT:
- Original query: {original_query}
- Iteration: {iteration_count} / {max_turns}
- Budget remaining: {budget_remaining} tokens, ${cost_remaining:.3f}
- Last task: {last_task_summary}
- DAG overview: {dag_overview}

DECISION OPTIONS:
1. retry          --  Re-run the failing task with an improved query
2. modify_dag     --  Add/remove tasks or update dependencies
3. escalate       --  Escalate to human (use when contradictions are severe or data is unavailable)
4. force_synthesize  --  Skip remaining tasks and synthesize with current evidence
5. stop           --  Abort (use only when evidence is completely absent and all options exhausted)

GUIDELINES:
- Maintain alignment to the original query intent and requested coverage.
- Prefer retry or modify_dag before escalate/force_synthesize/stop.
- Use retry when one task likely failed due weak query formulation.
- Use modify_dag when structural gaps exist (missing entity/dimension, bad dependency, deadlock).
- For external-entity gaps after internal probes, add targeted web_search tasks only for missing cells.
- If an external comparative query has >=2 failed researcher cells with no evidence, prefer modify_dag (targeted web_search per missing entity) over repeated retry.
- Use escalate only for high-risk ambiguity/contradiction that cannot be resolved automatically.
- Use force_synthesize only when safety limits are near or evidence is sufficient for partial answer.
- Use stop only when there is no meaningful evidence path left.
- If skeptic contradictions are reconciled with clear logic, continue to synthesis.

Few-shot decision patterns (compact):
- first low-recall researcher failure -> retry with rewritten atomic query
- repeated low-recall on same missing cell -> modify_dag with targeted web_search
- DAG incomplete and no runnable tasks -> modify_dag to unblock dependencies
- budget/time critical with partial coverage -> force_synthesize with explicit disclaimer

Return your decision as structured JSON matching SupervisorDecision schema.
"""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def supervisor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node: the main Supervisor entry point (MF-SUP-01 through MF-SUP-17).

    Dispatches to plan_dag() on the first turn and monitor_and_route() thereafter.

    Args:
        state: Current MASISState.

    Returns:
        Partial state update dict to merge into MASISState.
    """
    iteration = state.get("iteration_count", 0)

    # ── Early exit for terminal decisions (set by executor force_synthesize) ──
    # The hard edge executor->supervisor means we always run after executor.
    # When the executor has already produced a terminal result (e.g. after
    # force_synthesize completed), we must NOT overwrite the decision.
    current_decision = state.get("supervisor_decision", "")
    if current_decision in ("done", "failed"):
        logger.info(
            "Supervisor passthrough: decision already terminal (%s), routing to END",
            current_decision,
        )
        return {"supervisor_decision": current_decision}

    logger.info("Supervisor called (iteration=%d)", iteration)

    start_ts = time.monotonic()

    if iteration == 0:
        result = await plan_dag(state)
    else:
        result = await monitor_and_route(state)

    elapsed_ms = (time.monotonic() - start_ts) * 1000
    logger.debug("Supervisor completed in %.1f ms (decision=%s)", elapsed_ms, result.get("supervisor_decision"))

    return result


async def plan_dag(state: Dict[str, Any]) -> Dict[str, Any]:
    """First-turn DAG planning via gpt-4.1 with structured output (MF-SUP-01 through MF-SUP-03).

    Calls the LLM with a structured planning prompt to decompose the user query into a
    TaskPlan with per-task acceptance criteria. Returns a state update that
    sets task_dag, next_tasks, and supervisor_decision="continue".

    Args:
        state: MASISState at iteration_count == 0.

    Returns:
        Partial state update with task_dag, next_tasks, supervisor_decision, iteration_count.

    Raises:
        RuntimeError: If the LLM client is unavailable and no mock is provided.
    """
    original_query = state.get("original_query", "")
    logger.info("Supervisor MODE 1  --  PLAN for query: %.80s", original_query)

    # Optional pre-plan ambiguity HITL gate.
    enable_ambiguity_hitl = (
        str(os.getenv("ENABLE_AMBIGUITY_HITL", "0")).strip().lower() in {"1", "true", "yes"}
        or bool(state.get("enable_ambiguity_hitl", False))
    )
    if enable_ambiguity_hitl and _HITL_AVAILABLE and ambiguity_detector is not None:
        try:
            classification = await ambiguity_detector(original_query)
            label = getattr(getattr(classification, "label", None), "value", None) or str(
                getattr(classification, "label", "")
            )
            if str(label).upper() == "OUT_OF_SCOPE":
                logger.warning("Ambiguity gate marked query OUT_OF_SCOPE")
                return {
                    "supervisor_decision": "failed",
                    "iteration_count": 1,
                    "decision_log": log_decision(
                        state,
                        {
                            "turn": 1,
                            "mode": "slow",
                            "action": "out_of_scope",
                            "cost_usd": 0.0,
                            "latency_ms": 0.0,
                            "reason": "Ambiguity detector classified query as out_of_scope",
                        },
                    ),
                }
            suggestion = str(getattr(classification, "suggestion", "") or "").strip()
            if suggestion and suggestion.lower() != original_query.lower():
                logger.info("Ambiguity gate clarified query to: %.100s", suggestion)
                original_query = suggestion
        except Exception as exc:
            logger.warning("Ambiguity gate failed, continuing to planning: %s", exc)

    start_ts = time.monotonic()

    if not _OPENAI_AVAILABLE or ChatOpenAI is None:
        raise RuntimeError(
            "langchain_openai is not installed. "
            "Install with: pip install langchain-openai"
        )

    model_name = get_model("supervisor_plan")
    llm = ChatOpenAI(model=model_name, temperature=0.2)

    try:
        structured_llm = llm.with_structured_output(TaskPlan)
        prompt_text = SUPERVISOR_PLAN_PROMPT.format(user_query=original_query)
        plan: TaskPlan = await structured_llm.ainvoke(prompt_text)
    except Exception as exc:
        logger.error("plan_dag LLM call failed: %s", exc, exc_info=True)
        raise RuntimeError(f"Supervisor plan_dag failed: {exc}") from exc

    # Internal-first guardrail: remove web_search tasks unless query explicitly
    # requires external/competitor/latest context.
    plan.tasks = _normalize_plan_for_internal_first(original_query, plan.tasks)
    # Deterministic query cleanup to keep task prompts executable and anchored.
    plan.tasks = _normalize_task_queries(original_query, plan.tasks)
    # Deterministic complexity trim to keep first-wave strategic DAGs demo-fast.
    plan.tasks = _sanitize_initial_plan_complexity(original_query, plan.tasks)

    elapsed_ms = (time.monotonic() - start_ts) * 1000
    tokens_used = _estimate_tokens(SUPERVISOR_PLAN_PROMPT, str(plan))

    next_ready = get_next_ready_tasks(plan.tasks)
    decision_entry = {
        "turn": 1,
        "mode": "slow",
        "action": "plan",
        "task_ids": [t.task_id for t in next_ready],
        "cost_usd": _tokens_to_cost(tokens_used, model_name),
        "latency_ms": round(elapsed_ms, 1),
        "reason": "First-turn DAG planning",
    }

    updated_log = log_decision(state, decision_entry)
    new_budget = _update_budget(state, tokens_used, decision_entry["cost_usd"], "supervisor_plan")

    logger.info(
        "plan_dag completed: %d tasks planned, first batch=%s",
        len(plan.tasks),
        [t.task_id for t in next_ready],
    )

    return {
        "original_query": original_query,
        "task_dag": plan.tasks,
        "stop_condition": plan.stop_condition,
        "supervisor_decision": "continue",
        "next_tasks": next_ready,
        "iteration_count": 1,
        "token_budget": new_budget,
        "decision_log": updated_log,
    }


async def monitor_and_route(state: Dict[str, Any]) -> Dict[str, Any]:
    """Fast Path monitor  --  rule-based, no LLM, <10ms in the common case (MF-SUP-04 to MF-SUP-08).

    Checks (in order):
        1. Budget exhausted           ->  force_synthesize (MF-SUP-04)
        2. Iteration limit (15)       ->  force_synthesize (MF-SUP-05)
        3. Wall-clock > 300s          ->  force_synthesize (MF-SUP-16)
        4. Repetitive search          ->  force_synthesize (MF-SUP-06)
        5. Per-task criteria PASS     ->  dispatch next ready tasks (MF-SUP-07/08)
        6. Per-task criteria FAIL     ->  slow_path (MF-SUP-09 to MF-SUP-13)

    Args:
        state: Current MASISState (iteration_count >= 1).

    Returns:
        Partial state update dict.
    """
    iteration = state.get("iteration_count", 1)
    budget: BudgetTracker = state.get("token_budget", BudgetTracker())
    dag: List[TaskNode] = state.get("task_dag", [])
    last_result: Optional[AgentOutput] = state.get("last_task_result")
    batch_results: List[AgentOutput] = state.get("batch_task_results", []) or []
    parallel_batch_mode = bool(state.get("parallel_batch_mode", False))
    api_counts: Dict[str, int] = dict(state.get("api_call_counts", {}))
    has_synthesis = state.get("synthesis_output") is not None

    logger.info("Supervisor MODE 2  --  FAST PATH (iteration=%d)", iteration)
    start_ts = time.monotonic()

    # Parallel fan-out branches return per-task outputs without directly mutating
    # shared token_budget/api_call_counts. Aggregate once here.
    if parallel_batch_mode and batch_results:
        tokens_delta = sum(max(0, int(getattr(r, "tokens_used", 0) or 0)) for r in batch_results)
        cost_delta = sum(max(0.0, float(getattr(r, "cost_usd", 0.0) or 0.0)) for r in batch_results)
        merged_budget = budget
        if tokens_delta > 0 or cost_delta > 0:
            merged_budget = budget.add(tokens_delta, cost_delta, "")
        for result in batch_results:
            agent_type = getattr(result, "agent_type", "")
            if agent_type:
                api_counts[agent_type] = api_counts.get(agent_type, 0) + 1
        budget = merged_budget
        state = dict(state)
        state["token_budget"] = budget
        state["api_call_counts"] = api_counts
        state["__budget_override"] = budget
        state["__api_counts_override"] = api_counts

    # ── Check 1: Budget exhausted (MF-SUP-04) ───────────────────────────────
    if budget.is_exhausted():
        if has_synthesis:
            logger.warning(
                "Budget exhausted (tokens=%d, cost=$%.4f) but synthesis exists  --  ready_for_validation",
                budget.remaining,
                budget.total_cost_usd,
            )
            return _fast_decision(
                state,
                "ready_for_validation",
                "budget_exhausted_with_existing_synthesis",
                iteration,
                start_ts,
                task_dag=dag,
            )
        logger.warning(
            "Budget exhausted (tokens=%d, cost=$%.4f)  --  force_synthesize",
            budget.remaining,
            budget.total_cost_usd,
        )
        return _fast_decision(
            state,
            "force_synthesize",
            "budget_exhausted",
            iteration,
            start_ts,
            task_dag=dag,
        )

    # ── Check 2: Iteration limit (MF-SUP-05) ────────────────────────────────
    if iteration >= MAX_SUPERVISOR_TURNS:
        if has_synthesis:
            logger.warning(
                "Iteration limit %d reached and synthesis exists  --  ready_for_validation",
                MAX_SUPERVISOR_TURNS,
            )
            return _fast_decision(
                state,
                "ready_for_validation",
                "max_iterations_with_existing_synthesis",
                iteration,
                start_ts,
                task_dag=dag,
            )
        logger.warning("Iteration limit %d reached  --  force_synthesize", MAX_SUPERVISOR_TURNS)
        return _fast_decision(
            state,
            "force_synthesize",
            "max_iterations_reached",
            iteration,
            start_ts,
            task_dag=dag,
        )

    # ── Check 3: Wall-clock timeout (MF-SUP-16) ─────────────────────────────
    elapsed_wall = budget.wall_clock_seconds()
    if elapsed_wall > MAX_WALL_CLOCK_SECONDS:
        if has_synthesis:
            logger.warning(
                "Wall clock %.1fs > %ds and synthesis exists  --  ready_for_validation",
                elapsed_wall,
                MAX_WALL_CLOCK_SECONDS,
            )
            return _fast_decision(
                state,
                "ready_for_validation",
                "wall_clock_exceeded_with_existing_synthesis",
                iteration,
                start_ts,
                task_dag=dag,
            )
        logger.warning("Wall clock %.1fs > %ds  --  force_synthesize", elapsed_wall, MAX_WALL_CLOCK_SECONDS)
        return _fast_decision(
            state,
            "force_synthesize",
            "wall_clock_exceeded",
            iteration,
            start_ts,
            task_dag=dag,
        )

    # ── Check 4: Repetition detection (MF-SUP-06) ───────────────────────────
    if is_repetitive(state):
        if has_synthesis:
            logger.warning(
                "Repetitive search detected and synthesis exists  --  ready_for_validation"
            )
            return _fast_decision(
                state,
                "ready_for_validation",
                "repetitive_loop_with_existing_synthesis",
                iteration,
                start_ts,
                task_dag=dag,
            )
        logger.warning("Repetitive search detected  --  force_synthesize")
        return _fast_decision(
            state,
            "force_synthesize",
            "repetitive_loop_detected",
            iteration,
            start_ts,
            task_dag=dag,
        )

    # ── Check 5: Per-task acceptance criteria (MF-SUP-07) ───────────────────
    candidate_results: List[AgentOutput] = []
    if batch_results:
        candidate_results = batch_results
    elif last_result is not None:
        candidate_results = [last_result]

    failed_result: Optional[AgentOutput] = None
    for result in candidate_results:
        task = find_task(dag, result.task_id)
        criteria_verdict = "PASS"

        if task is not None and result.status == "success":
            criteria_verdict = check_agent_criteria(task, result)
            if criteria_verdict == "FAIL":
                task.status = "failed"
        elif result.status in ("failed", "timeout", "rate_limited"):
            criteria_verdict = "FAIL"
            if task is not None:
                task.status = "failed"

        if task is not None:
            if criteria_verdict == "PASS" and result.status == "success":
                task.status = "done"
            task.result_summary = result.summary[:500]

        if criteria_verdict == "FAIL":
            # If synthesis already exists (typically via force_synthesize), do not
            # keep looping on previously failed tasks. Let validator decide final quality.
            if has_synthesis:
                logger.info(
                    "Skipping criteria failure for task %s because synthesis already exists",
                    result.task_id,
                )
                if task is not None and task.status == "running":
                    task.status = "failed"
                continue
            failed_result = result
            break

    if failed_result is not None:
        # Deterministic recovery for external comparative/strategic queries:
        # when the first researcher wave returns no usable internal evidence,
        # inject targeted web_search tasks instead of spending extra retry turns.
        auto_recovery_update = _maybe_auto_external_web_recovery(
            state=state,
            dag=dag,
            candidate_results=candidate_results,
            iteration=iteration,
            start_ts=start_ts,
        )
        if auto_recovery_update is not None:
            return auto_recovery_update

        logger.info(
            "Task %s failed criteria  --  routing to SLOW PATH", failed_result.task_id
        )
        elapsed_ms = (time.monotonic() - start_ts) * 1000
        fast_entry = {
            "turn": iteration,
            "mode": "fast",
            "action": "criteria_fail_slow_path",
            "task_id": failed_result.task_id,
            "cost_usd": 0.0,
            "latency_ms": round(elapsed_ms, 1),
            "reason": "Criteria FAIL  --  escalating to Slow Path",
        }
        updated_log = log_decision(state, fast_entry)
        slow_input = dict(state)
        slow_input["task_dag"] = dag
        slow_input["last_task_result"] = failed_result
        slow_input["decision_log"] = updated_log
        slow_result = await supervisor_slow_path(slow_input)
        slow_result.setdefault("decision_log", updated_log)
        slow_result["iteration_count"] = iteration + 1
        slow_result["batch_task_results"] = []
        slow_result["parallel_batch_mode"] = False
        return slow_result

    # ── Check 6: All tasks done? (MF-SUP-08) ────────────────────────────────
    if all_tasks_done(dag):
        has_synthesis_task_done = any(
            getattr(t, "type", "") == "synthesizer" and getattr(t, "status", "") == "done"
            for t in dag
        )
        if state.get("synthesis_output") is None and not has_synthesis_task_done:
            logger.warning("All tasks terminal but synthesis_output missing  --  force_synthesize")
            return _fast_decision(
                state,
                "force_synthesize",
                "all_tasks_terminal_without_synthesis",
                iteration,
                start_ts,
                task_dag=dag,
            )
        logger.info("All DAG tasks complete  --  ready_for_validation")
        return _fast_decision(
            state,
            "ready_for_validation",
            "all_tasks_done",
            iteration,
            start_ts,
            task_dag=dag,
        )

    # ── Get next ready tasks ─────────────────────────────────────────────────
    next_ready = get_next_ready_tasks(dag)

    if not next_ready:
        # Recovery: stale "running" tasks can block scheduling if a prior write
        # dropped status updates. Re-queue and retry scheduler once.
        running_tasks = [t for t in dag if t.status == "running"]
        if running_tasks:
            logger.warning(
                "No ready tasks with %d running task(s); resetting stale running -> pending",
                len(running_tasks),
            )
            for task in running_tasks:
                task.status = "pending"
            next_ready = get_next_ready_tasks(dag)
            if next_ready:
                for task in next_ready:
                    task.status = "running"
                return _fast_decision(
                    state,
                    "continue",
                    "recovered_stale_running_tasks",
                    iteration,
                    start_ts,
                    next_tasks=next_ready,
                    task_dag=dag,
                )

        # DAG is incomplete and nothing is runnable: escalate to Slow Path for
        # retry/re-plan instead of terminating immediately.
        logger.warning("No ready tasks found and DAG incomplete  --  routing to Slow Path")
        elapsed_ms = (time.monotonic() - start_ts) * 1000
        fast_entry = {
            "turn": iteration,
            "mode": "fast",
            "action": "no_ready_tasks_slow_path",
            "cost_usd": 0.0,
            "latency_ms": round(elapsed_ms, 1),
            "reason": "DAG deadlock detected  --  escalating to Slow Path",
        }
        updated_log = log_decision(state, fast_entry)
        slow_input = dict(state)
        slow_input["task_dag"] = dag
        try:
            slow_result = await supervisor_slow_path(slow_input)
        except Exception as exc:
            logger.error(
                "Slow Path unavailable during no_ready_tasks recovery: %s  --  force_synthesize",
                exc,
            )
            return _fast_decision(
                state,
                "force_synthesize",
                "no_ready_tasks_slow_path_unavailable",
                iteration,
                start_ts,
                task_dag=dag,
            )
        slow_result.setdefault("decision_log", updated_log)
        slow_result["iteration_count"] = iteration + 1
        slow_result.setdefault("task_dag", dag)
        slow_result["parallel_batch_mode"] = False
        return slow_result

    # Mark next tasks as running (MF-SUP-15)
    for task in next_ready:
        task.status = "running"

    return _fast_decision(
        state,
        "continue",
        "next_tasks_dispatched",
        iteration,
        start_ts,
        next_tasks=next_ready,
        task_dag=dag,
    )


async def supervisor_slow_path(state: Dict[str, Any]) -> Dict[str, Any]:
    """Slow Path: LLM-based decision for failed tasks or complex situations (MF-SUP-09 to MF-SUP-13).

    Builds a compact context (summaries only, no full evidence) and calls gpt-4.1
    with structured output to decide: retry / modify_dag / escalate / force_synthesize / stop.

    Args:
        state: Current MASISState, with last_task_result holding the failed output.

    Returns:
        Partial state update dict.

    Raises:
        RuntimeError: If the LLM client is unavailable.
    """
    iteration = state.get("iteration_count", 1)
    logger.info("Supervisor MODE 3  --  SLOW PATH (iteration=%d)", iteration)
    start_ts = time.monotonic()

    if not _OPENAI_AVAILABLE or ChatOpenAI is None:
        raise RuntimeError("langchain_openai required for Supervisor Slow Path")

    # ── Build compact context (MF-SUP-14) ───────────────────────────────────
    context = _build_supervisor_context(state)

    model_name = get_model("supervisor_slow")
    llm = ChatOpenAI(model=model_name, temperature=0.2)

    try:
        structured_llm = llm.with_structured_output(SupervisorDecision)
        prompt_text = SUPERVISOR_SLOW_PROMPT.format(**context)
        decision: SupervisorDecision = await structured_llm.ainvoke(prompt_text)
    except Exception as exc:
        logger.error("supervisor_slow_path LLM call failed: %s", exc, exc_info=True)
        # Conservative fallback: force_synthesize rather than crash
        return _force_synthesize_update(state, iteration, f"slow_path_error: {exc}")

    elapsed_ms = (time.monotonic() - start_ts) * 1000
    tokens_used = _estimate_tokens(SUPERVISOR_SLOW_PROMPT, str(decision))
    cost = _tokens_to_cost(tokens_used, model_name)
    new_budget = _update_budget(state, tokens_used, cost, "supervisor_slow")

    logger.info("Slow Path decision: action=%s, reason=%.80s", decision.action, decision.reason)

    slow_entry = {
        "turn": iteration,
        "mode": "slow",
        "action": decision.action,
        "reason": decision.reason[:200],
        "cost_usd": cost,
        "latency_ms": round(elapsed_ms, 1),
    }
    updated_log = log_decision(state, slow_entry)

    base = {
        "token_budget": new_budget,
        "decision_log": updated_log,
        "batch_task_results": [],
        "parallel_batch_mode": False,
    }

    # ── Dispatch action ──────────────────────────────────────────────────────
    if decision.action == "retry":
        return _handle_retry(state, decision, base, iteration)

    if decision.action == "modify_dag":
        return _handle_modify_dag(state, decision, base, iteration)

    if decision.action == "escalate":
        return _handle_escalate(state, decision, base, iteration)

    if decision.action == "force_synthesize":
        return {**base, "supervisor_decision": "force_synthesize", "iteration_count": iteration + 1}

    if decision.action == "stop":
        logger.warning("Slow Path decided STOP: %s", decision.reason)
        return {**base, "supervisor_decision": "failed", "iteration_count": iteration + 1}

    # Fallback  --  should never reach here with strict Literal typing
    logger.error("Unknown Slow Path action: %s", decision.action)
    return {**base, "supervisor_decision": "force_synthesize", "iteration_count": iteration + 1}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


_EXTERNAL_QUERY_HINTS = {
    "compare",
    "comparison",
    "competitor",
    "competitors",
    "peer",
    "peers",
    "industry",
    "market",
    "sector",
    "benchmark",
    "benchmarks",
    "external",
    "outside",
    "latest",
    "today",
    "current events",
    "news",
    "macro",
    "regulatory trend",
    "aws",
    "azure",
    "gcp",
    "google",
    "alphabet",
    "adobe",
    "microsoft",
    "tcs",
    "wipro",
    "accenture",
    "compare with",
    "swot",
}

_COMPANY_ALIASES: Dict[str, List[str]] = {
    "infosys": ["infosys", "our company", "our organization", "our org"],
    "adobe": ["adobe"],
    "google": ["google", "alphabet", "gcp", "google cloud"],
    "microsoft": ["microsoft", "azure"],
    "amazon": ["amazon", "aws"],
}

_PLACEHOLDER_PATTERNS = [
    re.compile(r"\[\s*organization name\s*\]", re.IGNORECASE),
    re.compile(r"\[\s*company name\s*\]", re.IGNORECASE),
    re.compile(r"\{\s*organization\s*\}", re.IGNORECASE),
    re.compile(r"\{\s*company\s*\}", re.IGNORECASE),
]


def _query_requires_external_data(query: str) -> bool:
    """Return True when the user query explicitly asks for external context."""
    q = (query or "").lower()
    return any(hint in q for hint in _EXTERNAL_QUERY_HINTS)


def _normalize_plan_for_internal_first(
    original_query: str, tasks: List[TaskNode]
) -> List[TaskNode]:
    """Drop unnecessary web_search tasks and rewire dependencies.

    The supervisor prompt is probabilistic; this deterministic normalizer enforces
    internal-first behavior for queries answerable from internal documents.
    """
    if _query_requires_external_data(original_query):
        return tasks

    web_task_ids = {t.task_id for t in tasks if t.type == "web_search"}
    if not web_task_ids:
        return tasks

    filtered = [t for t in tasks if t.task_id not in web_task_ids]
    for task in filtered:
        if task.dependencies:
            task.dependencies = [d for d in task.dependencies if d not in web_task_ids]

    logger.info(
        "Internal-first normalization: removed %d web_search task(s) for query: %.80s",
        len(web_task_ids),
        original_query,
    )
    return filtered


def _extract_company_anchor(original_query: str) -> str:
    q = (original_query or "").lower()
    for canonical, aliases in _COMPANY_ALIASES.items():
        if any(alias in q for alias in aliases):
            return canonical.title()
    if "our " in q:
        return "Infosys"
    return "Infosys"


def _contains_known_entity(text: str) -> bool:
    q = (text or "").lower()
    return any(alias in q for aliases in _COMPANY_ALIASES.values() for alias in aliases)


def _clean_company_placeholders(text: str, company_anchor: str) -> str:
    cleaned = (text or "").strip()
    for pattern in _PLACEHOLDER_PATTERNS:
        cleaned = pattern.sub(company_anchor, cleaned)
    return cleaned


def _normalize_task_query_with_anchor(
    query: str,
    original_query: str,
    task_type: str,
) -> str:
    company_anchor = _extract_company_anchor(original_query)
    cleaned = _clean_company_placeholders(query, company_anchor)

    # For internal-first cases, keep company anchor explicit to reduce drift.
    if task_type != "web_search" and not _query_requires_external_data(original_query):
        if not _contains_known_entity(cleaned):
            cleaned = f"{cleaned} for {company_anchor}".strip()
    elif task_type == "web_search":
        if not _contains_known_entity(cleaned) and not _query_requires_external_data(original_query):
            cleaned = f"{cleaned} {company_anchor}".strip()

    return cleaned


def _normalize_task_queries(original_query: str, tasks: List[TaskNode]) -> List[TaskNode]:
    for task in tasks:
        task.query = _normalize_task_query_with_anchor(task.query, original_query, task.type)
    return tasks


def _is_strategic_query(query: str) -> bool:
    q = (query or "").lower()
    strategic_hints = (
        "swot",
        "compare",
        "versus",
        " vs ",
        "outlook",
        "scenario",
        "bull",
        "bear",
        "strategic",
    )
    return any(h in q for h in strategic_hints)


def _sanitize_initial_plan_complexity(
    original_query: str,
    tasks: List[TaskNode],
) -> List[TaskNode]:
    """Trim oversized first-turn DAGs to reduce timeout/cost blow-ups.

    Keeps broad coverage while capping parallel breadth. Dependencies are rewired
    to kept tasks only.
    """
    if len(tasks) <= 8:
        return tasks

    strategic = _is_strategic_query(original_query)
    external_needed = _query_requires_external_data(original_query)

    max_research = 6 if strategic else 4
    max_web = 2 if external_needed else 0

    kept: List[TaskNode] = []
    researcher_count = 0
    web_count = 0
    skeptic_kept = False
    synth_kept = False

    for task in tasks:
        ttype = getattr(task, "type", "")
        if ttype == "researcher":
            if researcher_count >= max_research:
                continue
            researcher_count += 1
            kept.append(task)
            continue

        if ttype in {"web_search", "deep_web_search"}:
            if web_count >= max_web:
                continue
            web_count += 1
            kept.append(task)
            continue

        if ttype == "skeptic":
            if skeptic_kept:
                continue
            skeptic_kept = True
            kept.append(task)
            continue

        if ttype == "synthesizer":
            if synth_kept:
                continue
            synth_kept = True
            kept.append(task)
            continue

        kept.append(task)

    if not kept:
        return tasks

    kept_ids = {t.task_id for t in kept}
    for task in kept:
        deps = list(getattr(task, "dependencies", []) or [])
        task.dependencies = [d for d in deps if d in kept_ids]

    if len(kept) < len(tasks):
        logger.info(
            "Initial plan complexity trim: kept %d/%d task(s) (research cap=%d, web cap=%d)",
            len(kept),
            len(tasks),
            max_research,
            max_web,
        )
    return kept


def _normalized_query_key(text: str) -> str:
    compact = re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())
    return re.sub(r"\s+", " ", compact).strip()


def _sanitize_modify_dag_additions(
    original_query: str,
    existing_dag: List[TaskNode],
    additions: List[TaskNode],
) -> List[TaskNode]:
    if not additions:
        return additions

    existing_web_keys = {
        _normalized_query_key(t.query)
        for t in existing_dag
        if getattr(t, "type", "") == "web_search"
    }
    sanitized: List[TaskNode] = []

    existing_web_count = sum(1 for t in existing_dag if getattr(t, "type", "") == "web_search")
    hard_web_cap = int(TOOL_LIMITS.get("web_search", {}).get("max_total", 4))
    strategic_cap = 2 if not _query_requires_external_data(original_query) else 3
    allowed_new_web = max(0, min(hard_web_cap - existing_web_count, strategic_cap))
    added_web = 0

    for task in additions:
        task.query = _normalize_task_query_with_anchor(task.query, original_query, task.type)

        if task.type != "web_search":
            sanitized.append(task)
            continue

        if added_web >= allowed_new_web:
            logger.info(
                "Skipping web_search task %s: web addition cap reached (%d)",
                task.task_id,
                allowed_new_web,
            )
            continue

        key = _normalized_query_key(task.query)
        if not key:
            logger.info("Skipping web_search task %s: empty query after normalization", task.task_id)
            continue
        if key in existing_web_keys:
            logger.info("Skipping duplicate web_search task %s: %s", task.task_id, task.query[:100])
            continue

        existing_web_keys.add(key)
        added_web += 1
        sanitized.append(task)

    return sanitized


def _build_supervisor_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build a compact context dict for the Slow Path prompt (MF-SUP-14).

    The Supervisor must NEVER receive full evidence chunks  --  only summaries,
    DAG statuses, and budget figures.

    Args:
        state: Full MASISState.

    Returns:
        Dict with only summary-level information for the LLM prompt.
    """
    budget: BudgetTracker = state.get("token_budget", BudgetTracker())
    dag: List[TaskNode] = state.get("task_dag", [])
    last_result: Optional[AgentOutput] = state.get("last_task_result")
    iteration = state.get("iteration_count", 1)

    dag_overview = ", ".join(
        f"{t.task_id}({t.type})={t.status}" for t in dag
    )

    last_task_summary = "None"
    if last_result:
        last_task_summary = (
            f"task_id={last_result.task_id}, "
            f"status={last_result.status}, "
            f"summary={last_result.summary[:300]}, "
            f"criteria={last_result.criteria_result}"
        )

    return {
        "original_query": state.get("original_query", ""),
        "iteration_count": iteration,
        "max_turns": MAX_SUPERVISOR_TURNS,
        "budget_remaining": budget.remaining,
        "cost_remaining": max(0.0, 0.50 - budget.total_cost_usd),
        "last_task_summary": last_task_summary,
        "dag_overview": dag_overview[:600],
    }


def _next_task_id(dag: List[TaskNode]) -> str:
    max_num = 0
    for task in dag:
        task_id = str(getattr(task, "task_id", "") or "")
        m = re.match(r"^T(\d+)$", task_id)
        if m:
            max_num = max(max_num, int(m.group(1)))
    return f"T{max_num + 1}"


def _build_external_web_query(research_query: str) -> str:
    q = (research_query or "").strip()
    if not q:
        return "Find latest reliable public financial data and disclosures relevant to the missing comparison."
    return f"Find latest reliable public sources for: {q}"


def _should_auto_external_web_recovery(
    state: Dict[str, Any],
    candidate_results: List[AgentOutput],
) -> bool:
    original_query = str(state.get("original_query", ""))
    if not _query_requires_external_data(original_query):
        return False

    if not candidate_results:
        return False

    # Trigger only on early waves to avoid aggressive mid-run rewrites.
    if int(state.get("iteration_count", 0) or 0) > 3:
        return False

    # We only auto-recover when a researcher-heavy wave clearly failed to retrieve.
    researcher_results = [r for r in candidate_results if getattr(r, "agent_type", "") == "researcher"]
    if len(researcher_results) < 2:
        return False

    failed_researchers = 0
    for r in researcher_results:
        criteria = getattr(r, "criteria_result", {}) or {}
        chunks = int(criteria.get("chunks_after_grading", 0) or 0)
        status = str(getattr(r, "status", "") or "")
        if status in {"failed", "timeout", "rate_limited"} or chunks == 0:
            failed_researchers += 1

    # Requires broad failure across the wave and no evidence yet.
    if failed_researchers < len(researcher_results):
        return False
    if (state.get("evidence_board", []) or []):
        return False
    return True


def _maybe_auto_external_web_recovery(
    state: Dict[str, Any],
    dag: List[TaskNode],
    candidate_results: List[AgentOutput],
    iteration: int,
    start_ts: float,
) -> Optional[Dict[str, Any]]:
    if not _should_auto_external_web_recovery(state, candidate_results):
        return None

    logger.info("Auto external recovery triggered: researcher wave had no internal evidence")

    # Create targeted web tasks from failed researcher queries.
    additions: List[TaskNode] = []
    for result in candidate_results:
        if getattr(result, "agent_type", "") != "researcher":
            continue
        task = find_task(dag, result.task_id)
        if task is None:
            continue
        additions.append(
            TaskNode(
                task_id=_next_task_id(dag + additions),
                type="web_search",
                query=_build_external_web_query(task.query),
                dependencies=[],
                parallel_group=getattr(task, "parallel_group", 1) or 1,
                acceptance_criteria=">=1 relevant result, no timeout",
                status="pending",
            )
        )

    if not additions:
        return None

    sanitized = _sanitize_modify_dag_additions(
        original_query=str(state.get("original_query", "")),
        existing_dag=dag,
        additions=additions,
    )
    if not sanitized:
        return None

    existing_ids = {t.task_id for t in dag}
    for new_task in sanitized:
        if new_task.task_id not in existing_ids:
            dag.append(new_task)
            existing_ids.add(new_task.task_id)
            logger.info("Auto-recovery added task %s (%s)", new_task.task_id, new_task.type)

    next_ready = get_next_ready_tasks(dag)
    for task in next_ready:
        task.status = "running"

    elapsed_ms = (time.monotonic() - start_ts) * 1000
    entry = {
        "turn": iteration,
        "mode": "fast",
        "action": "auto_modify_dag_external_recovery",
        "cost_usd": 0.0,
        "latency_ms": round(elapsed_ms, 1),
        "reason": "External comparative fallback: internal researcher wave had no evidence",
        "added_task_ids": [t.task_id for t in sanitized],
    }
    updated_log = log_decision(state, entry)

    return {
        "supervisor_decision": "continue",
        "iteration_count": iteration + 1,
        "decision_log": updated_log,
        "task_dag": dag,
        "next_tasks": next_ready,
        "batch_task_results": [],
        "parallel_batch_mode": False,
    }


def _fast_decision(
    state: Dict[str, Any],
    decision: str,
    reason: str,
    iteration: int,
    start_ts: float,
    next_tasks: Optional[List[TaskNode]] = None,
    task_dag: Optional[List[TaskNode]] = None,
) -> Dict[str, Any]:
    """Build the state update dict for a Fast Path decision.

    Args:
        state: Current MASISState (for budget and log access).
        decision: The supervisor_decision string.
        reason: Short reason for the decision log.
        iteration: Current iteration count.
        start_ts: monotonic start time for latency measurement.
        next_tasks: Tasks to dispatch (None for non-continue decisions).

    Returns:
        Partial state update dict.
    """
    elapsed_ms = (time.monotonic() - start_ts) * 1000
    entry = {
        "turn": iteration,
        "mode": "fast",
        "action": decision,
        "cost_usd": 0.0,
        "latency_ms": round(elapsed_ms, 1),
        "reason": reason,
    }
    updated_log = log_decision(state, entry)
    result: Dict[str, Any] = {
        "supervisor_decision": decision,
        "iteration_count": iteration + 1,
        "decision_log": updated_log,
        "batch_task_results": [],
        "parallel_batch_mode": False,
    }
    if "__budget_override" in state:
        result["token_budget"] = state.get("__budget_override")
    if "__api_counts_override" in state:
        result["api_call_counts"] = state.get("__api_counts_override")
    if next_tasks is not None:
        result["next_tasks"] = next_tasks
    if task_dag is not None:
        result["task_dag"] = task_dag
    return result


def _force_synthesize_update(
    state: Dict[str, Any], iteration: int, reason: str
) -> Dict[str, Any]:
    """Build a force_synthesize state update without an LLM call."""
    entry = {
        "turn": iteration,
        "mode": "slow",
        "action": "force_synthesize",
        "cost_usd": 0.0,
        "latency_ms": 0.0,
        "reason": reason,
    }
    updated_log = log_decision(state, entry)
    return {
        "supervisor_decision": "force_synthesize",
        "iteration_count": iteration + 1,
        "decision_log": updated_log,
        "batch_task_results": [],
        "parallel_batch_mode": False,
    }


def _handle_retry(
    state: Dict[str, Any],
    decision: SupervisorDecision,
    base: Dict[str, Any],
    iteration: int,
) -> Dict[str, Any]:
    """Retry a failed task, optionally rewriting its query (MF-SUP-09)."""
    dag: List[TaskNode] = list(state.get("task_dag", []))
    last_result: Optional[AgentOutput] = state.get("last_task_result")

    # Prefer retrying an actually failed task. Slow-path may request "retry"
    # when the last task succeeded but siblings failed and blocked the DAG.
    task: Optional[TaskNode] = None
    if last_result is not None:
        candidate = find_task(dag, last_result.task_id)
        # Slow-path "retry" should be able to revisit a criteria-failed task
        # even if executor already marked it done.
        if candidate is not None:
            task = candidate
    if task is None:
        failed_tasks = [t for t in dag if t.status == "failed"]
        if failed_tasks:
            task = min(failed_tasks, key=lambda t: t.retry_count)

    if task is not None:
        if task.retry_count >= MAX_TASK_RETRIES_PER_NODE:
            logger.warning(
                "Task %s exceeded retry cap (%d)  --  keeping failed and continuing with available tasks",
                task.task_id,
                MAX_TASK_RETRIES_PER_NODE,
            )
        else:
            task.status = "pending"
            task.retry_count += 1
            if decision.retry_spec and decision.retry_spec.new_query:
                task.query = _normalize_task_query_with_anchor(
                    decision.retry_spec.new_query,
                    state.get("original_query", ""),
                    task.type,
                )
            logger.info("Retrying task %s with query: %.80s", task.task_id, task.query)

    next_ready = get_next_ready_tasks(dag)
    for t in next_ready:
        t.status = "running"

    if not next_ready:
        return {
            **base,
            "task_dag": dag,
            "supervisor_decision": "force_synthesize",
            "iteration_count": iteration + 1,
        }

    return {
        **base,
        "task_dag": dag,
        "next_tasks": next_ready,
        "supervisor_decision": "continue",
        "iteration_count": iteration + 1,
    }


def _handle_modify_dag(
    state: Dict[str, Any],
    decision: SupervisorDecision,
    base: Dict[str, Any],
    iteration: int,
) -> Dict[str, Any]:
    """Add/remove tasks in the DAG (MF-SUP-10)."""
    dag: List[TaskNode] = list(state.get("task_dag", []))
    spec: Optional[ModifyDagSpec] = decision.modify_dag_spec
    original_query = state.get("original_query", "")

    if spec is not None:
        # Remove tasks
        remove_ids = set(spec.remove)
        dag = [t for t in dag if t.task_id not in remove_ids]

        # Add new tasks
        candidate_additions = _sanitize_modify_dag_additions(
            original_query=original_query,
            existing_dag=dag,
            additions=list(spec.add),
        )
        existing_ids = {t.task_id for t in dag}
        for new_task in candidate_additions:
            if new_task.task_id not in existing_ids:
                dag.append(new_task)
                logger.info("Added task %s (%s) to DAG", new_task.task_id, new_task.type)

    next_ready = get_next_ready_tasks(dag)
    for t in next_ready:
        t.status = "running"

    return {
        **base,
        "task_dag": dag,
        "next_tasks": next_ready,
        "supervisor_decision": "continue",
        "iteration_count": iteration + 1,
    }


def _handle_escalate(
    state: Dict[str, Any],
    decision: SupervisorDecision,
    base: Dict[str, Any],
    iteration: int,
) -> Dict[str, Any]:
    """Escalate to human via interrupt() (MF-SUP-11)."""
    spec = decision.escalate_spec

    if not _LANGGRAPH_AVAILABLE:
        logger.error("LangGraph interrupt() not available  --  falling back to force_synthesize")
        return {**base, "supervisor_decision": "force_synthesize", "iteration_count": iteration + 1}

    payload = {
        "type": "supervisor_escalation",
        "reason": decision.reason,
        "hitl_options": spec.hitl_options if spec else ["retry", "accept_partial", "cancel"],
    }
    logger.info("Escalating to HITL: %s", payload)
    interrupt(payload)  # LangGraph pauses the graph here

    return {
        **base,
        "supervisor_decision": "hitl_pause",
        "iteration_count": iteration + 1,
    }


def _estimate_tokens(prompt: str, response: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    return (len(prompt) + len(response)) // 4


def _tokens_to_cost(tokens: int, model: str) -> float:
    """Rough cost estimate. Real cost should come from LLM response usage metadata."""
    costs = {
        "gpt-4.1": 0.000015,        # $0.015 per 1K tokens (output)
        "gpt-4.1-mini": 0.0000003,  # $0.0003 per 1K tokens
        "o3-mini": 0.0000040,      # $0.004 per 1K tokens
    }
    rate = costs.get(model, 0.000015)
    return tokens * rate


def _update_budget(
    state: Dict[str, Any], tokens: int, cost: float, agent_type: str
) -> "BudgetTracker":
    """Return an updated BudgetTracker with consumed tokens and cost."""
    budget: BudgetTracker = state.get("token_budget", BudgetTracker())
    return budget.add(tokens, cost, agent_type)
