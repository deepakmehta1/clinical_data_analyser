# services/langgraph_service.py

import os
from glob import glob
from config import settings
from langgraph.graph import MessageGraph, END
from langgraph.prebuilt import ToolNode
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage
from router import router
from tools import check_hcc_relevance_tool
from models.vertex_ai_model import VertexAIModel


class LangGraphService:
    def __init__(self, project: str, location: str):
        """
        Initialize LangGraphService.

        Args:
            project (str): The project ID for Google Cloud.
            location (str): The location for Google Cloud services.
        """
        self.project_id = project
        self.location = location
        self.vertex_ai_model = (
            VertexAIModel()
        )  # Instantiate the Vertex AI model for extraction

    def set_up(self):
        """Setup LangGraph workflow."""
        model = ChatVertexAI(model="gemini-1.5-pro")
        builder = MessageGraph()

        # Bind tools to the model
        model_with_tools = model.bind_tools([check_hcc_relevance_tool])
        builder.add_node("tools", model_with_tools)

        # Adding nodes for condition extraction and HCC relevance check
        tool_node = ToolNode([check_hcc_relevance_tool])
        builder.add_node("check_hcc_relevance", tool_node)

        builder.add_edge("check_hcc_relevance", END)
        builder.set_entry_point("tools")
        builder.add_conditional_edges("tools", router)

        self.runnable = builder.compile()

    def run(self):
        """Run the LangGraph workflow."""
        # Get all the progress notes from the directory
        progress_note_files = glob(os.path.join(settings.PROGRESS_NOTE_PATH, "*.txt"))

        all_relevant_conditions = []

        for progress_note_path in progress_note_files:
            with open(progress_note_path, "r") as file:
                progress_note_text = file.read()

            # Step 1: Extract conditions from the progress note using Vertex AI
            conditions = self.vertex_ai_model.extract_conditions(progress_note_text)

            # Step 2: Use LangGraph to check HCC relevance
            relevant_conditions = self.runnable.invoke({"conditions": conditions})

            # Append the relevant conditions from this note to the final list
            all_relevant_conditions.extend(relevant_conditions)

        # Return the relevant conditions for all notes
        return all_relevant_conditions

    def query(self, message: str):
        """Query the application."""
        chat_history = self.runnable.invoke(HumanMessage(message))
        return chat_history[-1].content
