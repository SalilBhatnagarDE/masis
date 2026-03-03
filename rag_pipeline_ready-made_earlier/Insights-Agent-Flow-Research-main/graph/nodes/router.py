"""
Router Node — The "Traffic Cop" that classifies user intent.

Routes to one of two paths:
- "retrieve" -> clear question, search internal docs
- "clarify" -> vague/ambiguous question, ask for specifics
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

import config
from prompts import ROUTER_PROMPT

import datetime as datetime
todays_date = datetime.datetime.now().strftime("%B %Y")

class RouteDecision(BaseModel):
    """Structured output for routing decision."""
    route: str = Field(
        description="The routing decision: 'retrieve' or 'clarify'"
    )


def route_question(state: dict) -> dict:
    """
    Classify user intent and decide the routing path.
    
    Args:
        state: Current graph state with 'question'.
        
    Returns:
        State update with 'route' and 'original_question'.
    """
    question = state["question"]
    
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.OPENAI_API_KEY,
        temperature=0,
    )
    
    structured_llm = llm.with_structured_output(RouteDecision)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", ROUTER_PROMPT),
        ("human", "{question}"),
    ])
    
    chain = prompt | structured_llm
    
    result = chain.invoke({"question": question, "todays_date": todays_date})
    route = result.route.lower().strip()
    
    # Ensure valid route
    if route not in ("retrieve", "clarify"):
        route = "retrieve"  # Default to retrieve if uncertain
    
    return {
        "route": route,
        "original_question": question,
    }
