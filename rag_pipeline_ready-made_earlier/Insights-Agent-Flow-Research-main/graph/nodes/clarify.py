"""
Clarify Node — Generates a helpful clarifying question for vague user queries.
"""

import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import config
from prompts import CLARIFICATION_PROMPT

logger = logging.getLogger(__name__)

import datetime as datetime
todays_date = datetime.datetime.now().strftime("%B %Y")

def ask_clarification(state: dict) -> dict:
    """
    Generate a clarifying question when the user's query is too vague.
    """
    question = state.get("original_question", state.get("question", ""))
    
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.OPENAI_API_KEY,
        temperature=0.5,
    )
    
    prompt = ChatPromptTemplate.from_template(CLARIFICATION_PROMPT)
    chain = prompt | llm
    
    result = chain.invoke({"question": question, "todays_date": todays_date})
    
    logger.info("Generated clarifying question for vague query")
    
    return {"generation": result.content.strip()}
