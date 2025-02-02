# services/langgraph_service.py

import os
import logging
import asyncio
from glob import glob
from config import settings
from langgraph.graph import MessageGraph, END
from langgraph.prebuilt import ToolNode
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage
from router import router
from tools import check_hcc_relevance_tool
from models.open_ai_model import OpenAIModel
from models.vertex_ai_model import VertexAIModel

# Set up logging to print to console only
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


class LangGraphService:
    def __init__(self, project: str, location: str, model: OpenAIModel):
        """
        Initialize LangGraphService with the option to select the model type (OpenAI).

        Args:
            project (str): The project ID for Google Cloud.
            location (str): The location for Google Cloud services.
            model_type (str): The model type to use (currently only "openai").
        """
        logging.info("Initializing LangGraphService...")
        self.project_id = project
        self.location = location
        self.model = model

    def set_up(self):
        """Setup LangGraph workflow."""
        logging.info("Setting up LangGraph workflow...")

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
        logging.info("LangGraph workflow setup complete.")

    async def run(self):
        """Run the LangGraph workflow."""
        logging.info("Running LangGraph workflow...")

        # Get all the progress notes from the directory
        progress_note_files = glob(os.path.join(settings.PROGRESS_NOTE_PATH, "*"))

        if not progress_note_files:
            logging.warning("No progress note files found in the specified directory.")

        all_relevant_conditions = []

        # Loop through each progress note file
        for progress_note_path in progress_note_files[:1]:
            logging.info(f"Processing progress note: {progress_note_path}")

            with open(progress_note_path, "r") as file:
                progress_note_text = file.read()

            # Step 1: Extract conditions from the progress note using the selected model
            try:
                conditions = await self.model.extract_conditions(progress_note_text)
                logging.info(f"Extracted {len(conditions)} conditions from the note.")
            except Exception as e:
                logging.error(
                    f"Error extracting conditions from {progress_note_path}: {e}"
                )
                continue

            # Step 2: Use LangGraph to check HCC relevance
            try:
                relevant_conditions = await self.runnable.invoke(
                    {"conditions": conditions}
                )
                logging.info(
                    f"Found {len(relevant_conditions)} relevant conditions for the note."
                )
            except Exception as e:
                logging.error(
                    f"Error processing HCC relevance for {progress_note_path}: {e}"
                )
                continue

            # Append the relevant conditions from this note to the final list
            all_relevant_conditions.extend(relevant_conditions)

        logging.info(f"Total relevant conditions found: {len(all_relevant_conditions)}")

        # Return the relevant conditions for all notes
        return all_relevant_conditions

    async def query(self, message: str):
        """Query the application."""
        logging.info(f"Querying with message: {message}")
        chat_history = await self.runnable.invoke(HumanMessage(message))
        return chat_history[-1].content
