"""
Route Question Edge - Routes based on the Router node's classification.

After the Router node sets state["route"], this edge determines next node:
- "retrieve" -> rewrite_query
- "clarify" -> clarify
"""


def route_question_edge(state: dict) -> str:
    """
    Conditional edge: route based on intent classification.

    Returns:
        Node name to route to.
    """
    route = state.get("route", "retrieve")

    if route == "clarify":
        return "clarify"
    else:
        return "rewrite_query"
