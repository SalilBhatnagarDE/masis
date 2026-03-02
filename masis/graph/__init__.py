"""
masis.graph
===========
LangGraph graph assembly package for the MASIS multi-agent pipeline (ENG-14).

This package wires the three core nodes (supervisor, executor, validator) into
a cyclic LangGraph ``StateGraph`` with conditional routing edges, compiles it
with a durable checkpointer, and provides high-level invocation helpers.

Package Modules
---------------
edges.py      -- ``route_supervisor``, ``route_validator``, and node-name constants
workflow.py   -- ``build_workflow()`` (uncompiled graph), ``compile_workflow()``
runner.py     -- ``compile_graph()``, ``get_graph()``, ``ainvoke_graph()``,
                 ``stream_graph()``, ``generate_thread_id()``, ``make_config()``

Quick Start
-----------
::

    from masis.graph import ainvoke_graph, stream_graph, get_graph

    # Async invocation (e.g. from a FastAPI endpoint)
    result = await ainvoke_graph("What was TechCorp's Q3 FY26 revenue?")

    # Streaming invocation (SSE / MF-API-05)
    async for event in stream_graph("Compare Q3 revenue to competitors"):
        print(event)

    # Direct graph access (for advanced usage or testing)
    graph = get_graph()

Architecture Reference
----------------------
See ``final_architecture_and_flow.md`` Section 2 (Graph Construction) and
``engineering_tasks.md`` ENG-14 for the full specification.

Graph Topology
--------------
::

    START
      |
      v
    [supervisor] --conditional--> [executor]  | [validator] | END
         ^                            |
         |----------------------------+  (hard edge: executor -> supervisor)
    [validator] --conditional--> [supervisor] | END

Nodes:
    supervisor  -- DAG planning and monitoring (Fast Path + Slow Path)
    executor    -- dispatches agent tasks (researcher, skeptic, synthesizer, web_search)
    validator   -- quality scoring and threshold enforcement

Conditional Edges:
    route_supervisor -- reads supervisor_decision to pick executor/validator/END
    route_validator  -- reads validation_pass to pass (END) or revise (supervisor)

MF-IDs (this package)
---------------------
MF-MEM-08   checkpoint persistence via compiled graph
ENG-14 M1   graph construction (nodes, edges, conditional routing)
ENG-14 M2   compilation with checkpointer
ENG-14 M3   integration smoke test support
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Re-exports: public API of the masis.graph package
# ---------------------------------------------------------------------------

from masis.graph.runner import (
    ainvoke_graph,
    compile_graph,
    generate_thread_id,
    get_graph,
    make_config,
    reset_graph,
    stream_graph,
)

from masis.graph.edges import (
    NODE_EXECUTOR,
    NODE_SUPERVISOR,
    NODE_VALIDATOR,
    route_supervisor,
    route_validator,
)

from masis.graph.workflow import (
    build_workflow,
    compile_workflow,
)

__all__ = [
    # Runner functions
    "ainvoke_graph",
    "compile_graph",
    "generate_thread_id",
    "get_graph",
    "make_config",
    "reset_graph",
    "stream_graph",
    # Edge routing
    "route_supervisor",
    "route_validator",
    "NODE_EXECUTOR",
    "NODE_SUPERVISOR",
    "NODE_VALIDATOR",
    # Workflow building
    "build_workflow",
    "compile_workflow",
]
