"""
masis.infra.tracing
====================
OpenTelemetry / Langfuse / LangSmith tracing integration for MASIS (ENG-16 M1).

Provides a single ``get_tracing_callback()`` factory that returns the best
available tracing callback handler based on environment configuration:

    1. Langfuse  -- if LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY are set
    2. LangSmith -- if LANGSMITH_API_KEY is set
    3. No-op     -- a lightweight pass-through when no tracing backend is configured

The returned callback is intended to be passed into every LangChain/LangGraph
LLM call via ``callbacks=[handler]`` so that inputs, outputs, latency, cost,
and token counts are automatically captured.

Additionally, ``trace_graph_invocation`` provides a context-manager that creates
an OpenTelemetry-style span around a full graph invocation, recording the
thread_id, query, duration, and outcome.

MF-IDs
------
MF-API-06  Langfuse/LangSmith tracing -- every LLM call traced with
           inputs/outputs, latency, cost

Architecture reference
----------------------
final_architecture_and_flow.md Section 23.10
engineering_tasks.md ENG-16 M1
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable keys
# ---------------------------------------------------------------------------
_LANGFUSE_SECRET_KEY = "LANGFUSE_SECRET_KEY"
_LANGFUSE_PUBLIC_KEY = "LANGFUSE_PUBLIC_KEY"
_LANGFUSE_HOST = "LANGFUSE_HOST"
_LANGFUSE_DEFAULT_HOST = "https://cloud.langfuse.com"

_LANGSMITH_API_KEY = "LANGSMITH_API_KEY"
_LANGSMITH_PROJECT = "LANGSMITH_PROJECT"
_LANGSMITH_DEFAULT_PROJECT = "masis"


# ---------------------------------------------------------------------------
# No-op callback (used when no tracing backend is configured)
# ---------------------------------------------------------------------------

class NoOpTracingCallback:
    """A lightweight no-op callback handler that satisfies the LangChain
    callback interface without performing any I/O.

    This is returned by ``get_tracing_callback()`` when neither Langfuse nor
    LangSmith credentials are present in the environment.  It allows calling
    code to unconditionally pass ``callbacks=[handler]`` without branching.
    """

    def __init__(self) -> None:
        self._active = False

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str],
                     **kwargs: Any) -> None:
        """Called when an LLM call begins. No-op."""
        return

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Called when an LLM call completes. No-op."""
        return

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Called when an LLM call errors. No-op."""
        return

    def on_chain_start(self, serialized: Dict[str, Any],
                       inputs: Dict[str, Any], **kwargs: Any) -> None:
        """Called when a chain begins. No-op."""
        return

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Called when a chain completes. No-op."""
        return

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        """Called when a chain errors. No-op."""
        return

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str,
                      **kwargs: Any) -> None:
        """Called when a tool begins. No-op."""
        return

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Called when a tool completes. No-op."""
        return

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        """Called when a tool errors. No-op."""
        return

    def __repr__(self) -> str:
        return "NoOpTracingCallback()"


# ---------------------------------------------------------------------------
# Factory: get_tracing_callback
# ---------------------------------------------------------------------------

def get_tracing_callback(
    session_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    """Return the best available tracing callback handler.

    Priority order:
        1. Langfuse (if LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY are set)
        2. LangSmith (if LANGSMITH_API_KEY is set)
        3. NoOpTracingCallback (always available)

    Parameters
    ----------
    session_name : str, optional
        A session or trace name for grouping related LLM calls (e.g. the
        thread_id of a MASIS query).
    metadata : dict, optional
        Additional metadata to attach to the trace root.

    Returns
    -------
    callback handler
        An object compatible with LangChain's ``callbacks`` parameter.
    """
    # --- Try Langfuse first ---
    langfuse_secret = os.environ.get(_LANGFUSE_SECRET_KEY)
    langfuse_public = os.environ.get(_LANGFUSE_PUBLIC_KEY)

    if langfuse_secret and langfuse_public:
        try:
            from langfuse.callback import CallbackHandler as LangfuseHandler

            handler = LangfuseHandler(
                secret_key=langfuse_secret,
                public_key=langfuse_public,
                host=os.environ.get(_LANGFUSE_HOST, _LANGFUSE_DEFAULT_HOST),
                session_id=session_name,
                metadata=metadata or {},
            )
            logger.info(
                "Tracing: Langfuse callback initialised (host=%s, session=%s)",
                os.environ.get(_LANGFUSE_HOST, _LANGFUSE_DEFAULT_HOST),
                session_name,
            )
            return handler
        except ImportError:
            logger.warning(
                "Tracing: LANGFUSE_SECRET_KEY is set but 'langfuse' package "
                "is not installed. pip install langfuse to enable Langfuse tracing."
            )
        except Exception as exc:
            logger.warning(
                "Tracing: Langfuse handler initialisation failed: %s. "
                "Falling through to next backend.", exc,
            )

    # --- Try LangSmith ---
    langsmith_key = os.environ.get(_LANGSMITH_API_KEY)

    if langsmith_key:
        try:
            from langchain_core.tracers import LangChainTracer

            project = os.environ.get(_LANGSMITH_PROJECT, _LANGSMITH_DEFAULT_PROJECT)
            handler = LangChainTracer(project_name=project)
            logger.info(
                "Tracing: LangSmith callback initialised (project=%s)",
                project,
            )
            return handler
        except ImportError:
            logger.warning(
                "Tracing: LANGSMITH_API_KEY is set but 'langchain-core' tracers "
                "are not available. Ensure langchain-core is installed."
            )
        except Exception as exc:
            logger.warning(
                "Tracing: LangSmith handler initialisation failed: %s. "
                "Falling through to no-op.", exc,
            )

    # --- Fallback: no-op ---
    logger.info(
        "Tracing: No tracing backend configured. Using NoOpTracingCallback. "
        "Set LANGFUSE_SECRET_KEY/LANGFUSE_PUBLIC_KEY or LANGSMITH_API_KEY "
        "to enable production tracing."
    )
    return NoOpTracingCallback()


# ---------------------------------------------------------------------------
# Graph invocation span (OpenTelemetry-style context manager)
# ---------------------------------------------------------------------------

class GraphInvocationSpan:
    """Lightweight span that records timing and metadata for a graph invocation.

    This is not a full OpenTelemetry span -- it works without any OTel
    dependency installed.  When the ``opentelemetry`` SDK *is* available,
    it additionally creates a real OTel span.

    Usage::

        with trace_graph_invocation(thread_id, query) as span:
            result = await graph.ainvoke(state, config)
            span.set_attribute("status", "completed")
    """

    def __init__(
        self,
        thread_id: str,
        query: str,
        operation: str = "graph.ainvoke",
    ) -> None:
        self.thread_id = thread_id
        self.query = query[:200]  # Cap for safety
        self.operation = operation
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.attributes: Dict[str, Any] = {}
        self._otel_span: Any = None

    def set_attribute(self, key: str, value: Any) -> None:
        """Attach an attribute to this span."""
        self.attributes[key] = value
        if self._otel_span is not None:
            try:
                self._otel_span.set_attribute(key, str(value))
            except Exception:
                pass  # OTel attribute type mismatch -- ignore

    @property
    def duration_seconds(self) -> float:
        """Wall-clock duration in seconds (0.0 if span is still open)."""
        if self.end_time > 0:
            return self.end_time - self.start_time
        return time.time() - self.start_time if self.start_time > 0 else 0.0

    def _try_start_otel_span(self) -> None:
        """Attempt to create a real OpenTelemetry span if the SDK is installed."""
        try:
            from opentelemetry import trace as otel_trace

            tracer = otel_trace.get_tracer("masis.graph")
            self._otel_span = tracer.start_span(
                name=self.operation,
                attributes={
                    "masis.thread_id": self.thread_id,
                    "masis.query": self.query,
                },
            )
        except ImportError:
            self._otel_span = None
        except Exception as exc:
            logger.debug("OTel span creation failed: %s", exc)
            self._otel_span = None

    def _end_otel_span(self, error: Optional[BaseException] = None) -> None:
        """End the OTel span if one was created."""
        if self._otel_span is None:
            return
        try:
            if error is not None:
                self._otel_span.set_attribute("error", True)
                self._otel_span.set_attribute("error.message", str(error))
            for key, val in self.attributes.items():
                self._otel_span.set_attribute(key, str(val))
            self._otel_span.end()
        except Exception:
            pass  # Best-effort


@contextmanager
def trace_graph_invocation(
    thread_id: str,
    query: str,
    operation: str = "graph.ainvoke",
):
    """Context manager that wraps a graph invocation with tracing metadata.

    Creates a ``GraphInvocationSpan`` that records start/end time, thread_id,
    the original query, and any caller-supplied attributes.  When OpenTelemetry
    is installed, an OTel span is also created.

    Parameters
    ----------
    thread_id : str
        The MASIS thread identifier.
    query : str
        The original user query (truncated to 200 chars in the span).
    operation : str
        The span name / operation identifier.

    Yields
    ------
    GraphInvocationSpan
        The active span object. Callers may call ``span.set_attribute(...)``
        to attach additional metadata.

    Example
    -------
    ::

        with trace_graph_invocation(thread_id, query) as span:
            result = await ainvoke_graph(query, config=config)
            span.set_attribute("iterations", result.get("iteration_count", 0))
    """
    span = GraphInvocationSpan(thread_id=thread_id, query=query, operation=operation)
    span.start_time = time.time()
    span._try_start_otel_span()

    logger.info(
        "trace_graph_invocation: START operation=%s thread_id=%s query=%r",
        operation, thread_id, query[:80],
    )

    error: Optional[BaseException] = None
    try:
        yield span
    except BaseException as exc:
        error = exc
        span.set_attribute("error", True)
        span.set_attribute("error.type", type(exc).__name__)
        span.set_attribute("error.message", str(exc)[:500])
        raise
    finally:
        span.end_time = time.time()
        span._end_otel_span(error=error)
        level = logging.ERROR if error else logging.INFO
        logger.log(
            level,
            "trace_graph_invocation: END operation=%s thread_id=%s "
            "duration=%.3fs error=%s attributes=%s",
            operation,
            thread_id,
            span.duration_seconds,
            type(error).__name__ if error else None,
            span.attributes,
        )
