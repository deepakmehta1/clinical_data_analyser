# router.py
from langchain_core.messages import BaseMessage
from typing import Literal


def router(state: list[BaseMessage]) -> Literal["get_product_details", "__end__"]:
    """Initiates product details retrieval if the user asks for a product."""
    tool_calls = state[-1].tool_calls
    if len(tool_calls):
        return "get_product_details"
    else:
        return "__end__"
