"""
masis.graph.workflow
====================
LangGraph StateGraph construction and compilation for the MASIS pipeline (ENG-14).

This module builds the three-node cyclic graph described in the architecture:

    START --> supervisor --[conditional]--> executor | validator | END
                  ^                            |
                  |----------------------------+  (hard edge)
              validator --[conditional]--> supervisor | END

Nodes
-----
supervisor  -- ``supervisor_node``  from masis.nodes.supervisor
executor    -- ``execute_dag_tasks`` from masis.nodes.executor
validator   -- ``validator_node``   from masis.nodes.validator

Conditional Edges
-----------------
route_supervisor  -- reads state["supervisor_decision"] to pick next node
route_validator   -- reads state["validation_pass"] to pass or revise

Compilation
-----------
The graph is compiled with a ``BaseCheckpointSaver`` obtained from
``masis.infra.persistence.get_checkpointer()`` so that every super-step is
durably persisted (MF-MEM-08) and HITL resume works (MF-HITL-05).

MF-IDs covered
--------------
All routing MF-IDs are covered by edges.py; this module covers the wiring:
  MF-MEM-08  checkpoint persistence via compiled graph
  ENG-14 M1 S1  StateGraph(MASISState) with 3 nodes
  ENG-14 M1 S2  conditional edges for supervisor
  ENG-14 M1 S3  conditional edges for validator
  ENG-14 M2 S1  compile with get_checkpointer()

Exports
-------
build_workflow()   -- create an uncompiled StateGraph (useful for testing)
compile_workflow() -- build + compile with checkpointer; returns CompiledGraph
graph              -- module-level compiled graph singleton (lazy)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from masis.schemas.models import MASISState
from masis.nodes.supervisor import supervisor_node
from masis.nodes.executor import execute_dag_tasks
from masis.nodes.validator import validator_node
from masis.infra.persistence import get_checkpointer
from masis.graph.edges import (
    route_supervisor,
    route_validator,
    NODE_EXECUTOR,
    NODE_SUPERVISOR,
    NODE_VALIDATOR,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Routing maps — maps return values of edge functions to node names / END
# ---------------------------------------------------------------------------

_SUPERVISOR_ROUTE_MAP: dict[str, str] = {
    NODE_EXECUTOR: NODE_EXECUTOR,
    NODE_VALIDATOR: NODE_VALIDATOR,
    END: END,
}

_VALIDATOR_ROUTE_MAP: dict[str, str] = {
    NODE_SUPERVISOR: NODE_SUPERVISOR,
    END: END,
}


# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------

def build_workflow() -> StateGraph:
    """Construct the uncompiled MASIS StateGraph.

    Returns a ``StateGraph`` with all three nodes wired together.  This is
    useful for unit tests that want to inspect topology before compilation,
    or for compiling with a custom checkpointer.

    Returns
    -------
    StateGraph
        An uncompiled LangGraph state graph ready for ``.compile()``.

    Raises
    ------
    RuntimeError
        If any of the required node functions fail to import (should never
        happen at runtime since the imports are at module level).
    """
    logger.info("build_workflow: constructing StateGraph(MASISState)")

    # S1a: Create the graph with the MASIS state schema
    workflow = StateGraph(MASISState)

    # S1b: Add the three core nodes
    workflow.add_node(NODE_SUPERVISOR, supervisor_node)
    workflow.add_node(NODE_EXECUTOR, execute_dag_tasks)
    workflow.add_node(NODE_VALIDATOR, validator_node)

    logger.debug(
        "build_workflow: added nodes: %s, %s, %s",
        NODE_SUPERVISOR, NODE_EXECUTOR, NODE_VALIDATOR,
    )

    # S1c: Entry point — every query starts at the supervisor for DAG planning
    workflow.add_edge(START, NODE_SUPERVISOR)

    # S1d: Hard edge — executor ALWAYS returns to supervisor for evaluation
    # The supervisor then decides whether to continue, validate, or stop.
    workflow.add_edge(NODE_EXECUTOR, NODE_SUPERVISOR)

    logger.debug("build_workflow: added hard edges: START->supervisor, executor->supervisor")

    # S2: Conditional edges from supervisor
    # route_supervisor reads state["supervisor_decision"] and returns one of:
    #   NODE_EXECUTOR  ("continue" or "force_synthesize")
    #   NODE_VALIDATOR ("ready_for_validation")
    #   END            ("hitl_pause", "failed", or unknown)
    workflow.add_conditional_edges(
        source=NODE_SUPERVISOR,
        path=route_supervisor,
        path_map=_SUPERVISOR_ROUTE_MAP,
    )

    logger.debug("build_workflow: added conditional edges from supervisor")

    # S3: Conditional edges from validator
    # route_validator reads state["validation_pass"] and returns:
    #   END             (validation passed — answer meets all quality thresholds)
    #   NODE_SUPERVISOR (validation failed — supervisor revises the plan)
    workflow.add_conditional_edges(
        source=NODE_VALIDATOR,
        path=route_validator,
        path_map=_VALIDATOR_ROUTE_MAP,
    )

    logger.debug("build_workflow: added conditional edges from validator")

    logger.info("build_workflow: StateGraph construction complete")
    return workflow


def compile_workflow(
    checkpointer: Optional[Any] = None,
    interrupt_before: Optional[list[str]] = None,
    interrupt_after: Optional[list[str]] = None,
):
    """Build and compile the MASIS graph with a checkpointer.

    Parameters
    ----------
    checkpointer : BaseCheckpointSaver, optional
        A LangGraph checkpoint saver.  If ``None``, calls
        ``get_checkpointer()`` which uses Postgres when available and
        falls back to InMemorySaver with a warning.
    interrupt_before : list[str], optional
        Node names to interrupt *before* execution (for HITL).
    interrupt_after : list[str], optional
        Node names to interrupt *after* execution (for HITL).

    Returns
    -------
    CompiledGraph
        A compiled LangGraph graph ready for ``.invoke()`` / ``.ainvoke()``.
    """
    if checkpointer is None:
        logger.info("compile_workflow: obtaining checkpointer via get_checkpointer()")
        checkpointer = get_checkpointer()

    workflow = build_workflow()

    compile_kwargs: dict[str, Any] = {"checkpointer": checkpointer}
    if interrupt_before:
        compile_kwargs["interrupt_before"] = interrupt_before
    if interrupt_after:
        compile_kwargs["interrupt_after"] = interrupt_after

    compiled = workflow.compile(**compile_kwargs)

    logger.info(
        "compile_workflow: graph compiled successfully "
        "(checkpointer=%s, interrupt_before=%s, interrupt_after=%s)",
        type(checkpointer).__name__,
        interrupt_before,
        interrupt_after,
    )
    return compiled
