# langgraph_app.py
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, MessageGraph
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode
from config import PROJECT_ID, LOCATION
from tools import get_product_details
from router import router


class SimpleLangGraphApp:
    def __init__(self, project: str, location: str) -> None:
        self.project_id = project
        self.location = location

    def set_up(self) -> None:
        model = ChatVertexAI(model="gemini-1.5-pro")

        builder = MessageGraph()

        model_with_tools = model.bind_tools([get_product_details])
        builder.add_node("tools", model_with_tools)

        tool_node = ToolNode([get_product_details])
        builder.add_node("get_product_details", tool_node)

        builder.add_edge("get_product_details", END)

        builder.set_entry_point("tools")
        builder.add_conditional_edges("tools", router)

        self.runnable = builder.compile()

    def query(self, message: str):
        """Query the application."""
        chat_history = self.runnable.invoke(HumanMessage(message))
        return chat_history[-1].content
