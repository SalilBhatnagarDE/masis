"""
Workflow Assembly - Wires all nodes and edges into a LangGraph StateGraph.

This is the main graph that orchestrates the entire RAG pipeline:

START -> Router -> [Route Question Edge]
  - "clarify" -> Clarify -> END
  - "retrieve" -> Rewrite -> Metadata -> Retrieve -> Rerank -> Grade
       -> [Decide to Generate Edge]
          - "generate" -> Generate -> Hallucination Check -> [Hallucination Edge]
               - "end" -> Enrich Answer -> END
               - "retry" -> Generate (loop)
               - "fallback" -> Fallback -> END
          - "rewrite" -> Rewrite (loop back)
          - "fallback" -> Fallback -> END
"""

import logging
from langgraph.graph import StateGraph, END

from graph.state import GraphState

# Import all nodes
from graph.nodes.router import route_question
from graph.nodes.rewrite_query import rewrite_query
from graph.nodes.extract_metadata import extract_metadata
from graph.nodes.retrieve import retrieve_documents
from graph.nodes.rerank import rerank_documents
from graph.nodes.grade_documents import grade_documents
from graph.nodes.generate import generate_answer
from graph.nodes.check_hallucination import check_hallucination
from graph.nodes.clarify import ask_clarification
from graph.nodes.fallback import fallback_response
from graph.nodes.enrich_answer import enrich_answer

# Import all edges
from graph.edges.route_question import route_question_edge
from graph.edges.decide_to_generate import decide_to_generate_edge
from graph.edges.check_hallucination_edge import check_hallucination_edge

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """
    Build and compile the complete LangGraph workflow.
    Returns a compiled StateGraph ready for invoke() or stream().
    """
    workflow = StateGraph(GraphState)

    # Add all nodes
    workflow.add_node("router", route_question)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("extract_metadata", extract_metadata)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("rerank", rerank_documents)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("check_hallucination", check_hallucination)
    workflow.add_node("clarify", ask_clarification)
    workflow.add_node("fallback", fallback_response)
    workflow.add_node("enrich_answer", enrich_answer)

    # Entry point
    workflow.set_entry_point("router")

    # Edge 1: Router -> conditional routing
    workflow.add_conditional_edges(
        "router",
        route_question_edge,
        {
            "rewrite_query": "rewrite_query",  # Retrieve path
            "clarify": "clarify",              # Clarification path
        },
    )

    # Retrieve pipeline: Rewrite -> Metadata -> Retrieve -> Rerank -> Grade
    workflow.add_edge("rewrite_query", "extract_metadata")
    workflow.add_edge("extract_metadata", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "grade_documents")

    # Edge 2: Grade Documents -> decide whether to generate or rewrite
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate_edge,
        {
            "generate": "generate",       # Enough relevant docs -> generate
            "rewrite": "rewrite_query",   # CRAG loop: rewrite and retrieve again
            "fallback": "fallback",       # Exhausted retries
        },
    )

    # Generate -> Hallucination Check
    workflow.add_edge("generate", "check_hallucination")

    # Edge 3: Hallucination Check -> decide if answer is grounded
    workflow.add_conditional_edges(
        "check_hallucination",
        check_hallucination_edge,
        {
            "end": "enrich_answer",  # Grounded answer -> enrich with charts/reports
            "retry": "generate",     # Self-RAG loop: regenerate
            "fallback": "fallback",  # Exhausted retries
        },
    )

    # Terminal edges
    workflow.add_edge("clarify", END)
    workflow.add_edge("fallback", END)
    workflow.add_edge("enrich_answer", END)

    # Compile
    graph = workflow.compile()

    logger.info("LangGraph workflow compiled successfully")
    return graph


def run_agent(question: str) -> dict:
    """
    Run the full agent pipeline on a single question.
    Returns the final state dict with 'generation' containing the answer.
    """
    graph = build_graph()

    initial_state = {
        "question": question,
        "original_question": question,
        "rewritten_question": "",
        "route": "",
        "query_metadata": {},
        "metadata_filters": [],
        "metadata_retries": 0,
        "documents": [],
        "reranked_documents": [],
        "generation": "",
        "chart_data": {},
        "doc_grading_retries": 0,
        "hallucination_retries": 0,
        "answer_contains_hallucinations": False,
        "irrelevancy_reason": "",
    }

    result = graph.invoke(initial_state)
    return result


def stream_agent(question: str):
    """
    Stream the agent pipeline, yielding each node's output for real-time display.
    Yields tuples of (node_name, state_update) for each step.
    """
    graph = build_graph()

    initial_state = {
        "question": question,
        "original_question": question,
        "rewritten_question": "",
        "route": "",
        "query_metadata": {},
        "metadata_filters": [],
        "metadata_retries": 0,
        "documents": [],
        "reranked_documents": [],
        "generation": "",
        "chart_data": {},
        "doc_grading_retries": 0,
        "hallucination_retries": 0,
        "answer_contains_hallucinations": False,
        "irrelevancy_reason": "",
    }

    for output in graph.stream(initial_state):
        for node_name, state_update in output.items():
            yield node_name, state_update
