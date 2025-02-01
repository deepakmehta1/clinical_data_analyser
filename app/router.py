# router.py

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI


def router(state: list[BaseMessage]) -> str:
    """
    Simple router function to determine what to do next in the LangGraph workflow.

    Args:
        state (list[BaseMessage]): The current conversation state (history).

    Returns:
        str: The next action or tool to call.
    """
    # If we have conditions, we route to the next tool (HCC relevance check)
    if len(state) > 0 and "conditions" in state[-1].content:
        return "check_hcc_relevance"
    else:
        return "__end__"  # End the conversation if no conditions are found
