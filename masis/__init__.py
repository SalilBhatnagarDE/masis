"""
MASIS — Multi-Agent Supervised Intelligence System
====================================================
Phase 0: Foundation Package

This package provides the complete foundation for the MASIS system:
  - schemas/     : Pydantic data models, TypedDict state, and threshold constants
  - config/      : Model routing, fallback chains, rate limits, and settings
  - utils/       : DAG walking, text utilities, safety sanitizers, and decision logging

Architecture Overview
---------------------
MASIS implements a 3-node LangGraph execution graph:
  Supervisor → Executor → Validator

The Supervisor creates a dynamic Task DAG (stored as data in state) and dispatches
agents (Researcher, Skeptic, Synthesizer, WebSearch) through the Executor.
A two-tier decision mechanism (Fast Path: $0/rule-based, Slow Path: gpt-4.1) minimises
LLM costs while maintaining full quality control.

Usage
-----
    from masis.schemas.models import MASISState, TaskNode, TaskPlan, EvidenceChunk
    from masis.schemas.thresholds import RESEARCHER_THRESHOLDS, SAFETY_LIMITS
    from masis.config.model_routing import get_model, get_fallback, TOOL_LIMITS
    from masis.config.settings import get_settings
    from masis.utils.dag_utils import get_next_ready_tasks, check_agent_criteria
    from masis.utils.text_utils import u_shape_order, is_repetitive
    from masis.utils.safety_utils import content_sanitizer, log_decision
"""

from importlib.metadata import version, PackageNotFoundError

__title__ = "masis"
__description__ = "Multi-Agent Supervised Intelligence System — Foundation Layer"
__version__ = "0.1.0"
__author__ = "MASIS Engineering Team"

try:
    __version__ = version("masis")
except PackageNotFoundError:
    # Package not installed via pip — development mode
    __version__ = "0.1.0-dev"

__all__ = [
    "__title__",
    "__description__",
    "__version__",
    "__author__",
]
