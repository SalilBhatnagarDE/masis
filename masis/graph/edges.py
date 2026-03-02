"""
masis.graph.edges
=================
Conditional edge routing functions for the MASIS LangGraph workflow (ENG-14).

These functions are passed to ``workflow.add_conditional_edges()`` and are
called by LangGraph after each node completes to determine the next node.

Public API
----------
route_supervisor(state)  -- routes after supervisor_node; reads supervisor_decision
route_validator(state)   -- routes after validator_node; reads validation_pass

Routing Table
-------------
Supervisor:
    "continue"            ->  "executor"   (more tasks to run this iteration)
    "ready_for_validation"  ->  "validator" (all tasks done, synthesizer ran)
    "force_synthesize"    ->  "executor"   (budget/loop cap  --  synthesize immediately)
    "hitl_pause"          ->  END          (graph interrupted for human input)
    "failed"              ->  END          (unrecoverable error)
    <unknown>             ->  END          (safe fallback)

Validator:
    validation_pass=True   ->  END         (answer meets all quality thresholds)
    validation_pass=False  ->  "supervisor" (at least one threshold missed  ->  revise)

MF-IDs covered
--------------
MF-SUP-11  route_supervisor: continue  ->  executor
MF-SUP-12  route_supervisor: ready_for_validation  ->  validator
MF-SUP-13  route_supervisor: force_synthesize  ->  executor
MF-VAL-05  route_validator: threshold enforcement gates
MF-VAL-07  route_validator: max-revision loop cap (state carries validation_round)
"""

from __future__ import annotations

import logging
from typing import Literal

try:
    from langgraph.graph import END
except ImportError:  # pragma: no cover  --  fallback for environments without langgraph
    END = "__end__"  # type: ignore[assignment]

try:
    from masis.schemas.models import MASISState
except ImportError:  # pragma: no cover  --  fallback for isolated environments
    from typing import Any
    MASISState = Any  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Routing constants  --  kept here so tests can assert on literal strings
# ---------------------------------------------------------------------------
NODE_EXECUTOR = "executor"
NODE_SUPERVISOR = "supervisor"
NODE_VALIDATOR = "validator"

# Valid supervisor_decision values
DECISION_CONTINUE = "continue"
DECISION_READY_FOR_VALIDATION = "ready_for_validation"
DECISION_FORCE_SYNTHESIZE = "force_synthesize"
DECISION_HITL_PAUSE = "hitl_pause"
DECISION_FAILED = "failed"
DECISION_DONE = "done"


def route_supervisor(state: "MASISState") -> str:
    """
    Conditional edge function called after ``supervisor_node`` completes.

    Reads ``state["supervisor_decision"]`` and returns the name of the next
    node (or END sentinel) for LangGraph to dispatch to.

    Parameters
    ----------
    state : MASISState
        Current graph state. Must contain ``supervisor_decision``.

    Returns
    -------
    str
        One of: "executor", "validator", or the END sentinel string.

    Routing logic
    -------------
    "continue"              ->  executor  (standard research loop iteration)
    "ready_for_validation"  ->  validator (synthesizer has run; validate the answer)
    "force_synthesize"      ->  executor  (budget/loop-cap; synthesizer runs immediately
                                        via the force_synthesize flag in state)
    "hitl_pause"            ->  END       (graph paused via interrupt(); resume via API)
    "failed"                ->  END       (unrecoverable; return best available partial)
    <anything else>         ->  END       (safe unknown-decision fallback, logged as WARNING)
    """
    decision: str = state.get("supervisor_decision", "failed")  # type: ignore[assignment]

    logger.debug("route_supervisor: supervisor_decision=%r", decision)

    if decision == DECISION_CONTINUE:
        logger.debug("route_supervisor  ->  executor (continue)")
        return NODE_EXECUTOR

    if decision == DECISION_READY_FOR_VALIDATION:
        logger.debug("route_supervisor  ->  validator (ready_for_validation)")
        return NODE_VALIDATOR

    if decision == DECISION_FORCE_SYNTHESIZE:
        # Executor will check state["force_synthesize"] and skip to synthesizer
        logger.debug("route_supervisor  ->  executor (force_synthesize)")
        return NODE_EXECUTOR

    if decision == DECISION_HITL_PAUSE:
        # Graph was interrupted via interrupt(); LangGraph persists state to
        # checkpoint. Resumes when POST /masis/resume is called.
        logger.info("route_supervisor  ->  END (hitl_pause  --  awaiting human input)")
        return END

    if decision == DECISION_FAILED:
        logger.warning("route_supervisor  ->  END (failed  --  unrecoverable error)")
        return END

    if decision == DECISION_DONE:
        logger.info("route_supervisor  ->  END (done)")
        return END

    # Unknown decision value  --  defensive fallback
    logger.warning(
        "route_supervisor: unexpected supervisor_decision=%r; routing to END",
        decision,
    )
    return END


def route_validator(state: "MASISState") -> str:
    """
    Conditional edge function called after ``validator_node`` completes.

    Reads ``state["validation_pass"]`` and returns the next node.

    Parameters
    ----------
    state : MASISState
        Current graph state. Must contain ``validation_pass`` (bool) and
        optionally ``validation_round`` (int, default 0).

    Returns
    -------
    str
        "supervisor" when at least one quality threshold was missed and
        max revision rounds have not been reached; END otherwise.

    MF-VAL-07: Max validation rounds
        ``validation_round`` is incremented by validator_node each call.
        When it reaches MAX_VALIDATION_ROUNDS the validator sets
        ``validation_pass=True`` regardless, so this function always sees
        True at that point and routes to END.  The cap is enforced by
        validator_node, not here  --  keeping routing logic pure.
    """
    validation_pass: bool = state.get("validation_pass", False)  # type: ignore[assignment]
    validation_round: int = state.get("validation_round", 0)  # type: ignore[assignment]
    quality_scores = state.get("quality_scores", {}) or {}
    forced_pass = bool(getattr(quality_scores, "get", lambda *_: False)("forced_pass", False))

    logger.debug(
        "route_validator: validation_pass=%s, validation_round=%d",
        validation_pass,
        validation_round,
    )

    if validation_pass:
        logger.info(
            "route_validator  ->  END (validation passed on round %d)", validation_round
        )
        return END

    # Safety guard: if validator explicitly marked forced_pass in quality_scores,
    # end the graph even if validation_pass was lost/missing during merge.
    if forced_pass:
        logger.info(
            "route_validator  ->  END (forced pass on round %d)", validation_round
        )
        return END

    # At least one threshold missed  --  send back to supervisor for revision.
    # supervisor_node will inspect quality_scores and decide whether to
    # retry specific tasks or force-synthesize with the current evidence.
    logger.info(
        "route_validator  ->  supervisor (validation failed on round %d, retrying)",
        validation_round,
    )
    return NODE_SUPERVISOR
