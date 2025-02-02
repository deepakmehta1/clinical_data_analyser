# models/openai_model.py

import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from config import settings
from typing import List, Optional


# Define the Condition schema to store condition names and their codes
class Condition(BaseModel):
    condition: str
    code: Optional[str] = None  # Optional code for the condition, may be empty


# Define a list of conditions to store multiple Condition objects
class ConditionList(BaseModel):
    conditions: List[Condition]


class OpenAIModel:
    def __init__(self, model_name: str = "gpt-3.5-turbo-0125"):
        """
        Initialize the OpenAI model wrapper using LangChain.

        Args:
            model_name (str): The name of the OpenAI model to use (default is "gpt-3.5-turbo-0125").
        """
        self.model_name = model_name
        self.chat = ChatOpenAI(model=self.model_name, api_key=settings.OPENAI_API_KEY)
        logging.info(f"Initialized OpenAIModel with model: {self.model_name}")

    async def _generate_response(self, prompt: str):
        """
        Helper function to generate a response from OpenAI based on the provided prompt.

        Args:
            prompt (str): The input text or prompt to send to the model.

        Returns:
            str: The generated response from OpenAI.
        """
        try:
            # Create the messages in LangChain format for structured input
            messages = [
                SystemMessage(
                    content=self.system_prompt()
                ),  # Add the system message (clinical expert role)
                HumanMessage(
                    content=prompt
                ),  # Add the clinical progress note as a user message
            ]

            # Bind schema to the model for structured output (list of conditions)
            model_with_structure = self.chat.with_structured_output(ConditionList)

            # Invoke the model and get structured output
            structured_output = await model_with_structure.ainvoke(messages)
            logging.info("Model response successfully received.")
            return structured_output

        except Exception as e:
            logging.error(f"Error generating response from OpenAI: {e}")
            raise RuntimeError(f"Error generating response from OpenAI: {e}")

    def system_prompt(self):
        """
        Generate the system prompt to instruct the model to act as a clinical expert.

        This instructs the model to extract medical conditions and their associated codes.
        The system prompt includes examples for better guidance.

        Returns:
            str: The system prompt string.
        """
        return """
        You are a clinical expert. Your job is to extract the medical condition(s) and their associated codes from the given clinical progress note.

        Your task:
        - For each medical condition, provide a condition name and an optional code.
        - You are to extract the conditions in the following format:
            [
                {"condition": "<condition_name>", "code": "<optional_code>"}
            ]

        Example 1:
        Clinical Progress Note:
        "The patient presents with a history of gastroesophageal reflux disease. Stable, continue antacids, follow-up in 3 months. Code: K21.9"
        
        Extracted conditions:
        [
            {"condition": "Gastroesophageal reflux disease", "code": "K21.9"}
        ]

        Example 2:
        Clinical Progress Note:
        "Patient is experiencing hypertension and diabetes. Hypertension controlled with medication. Diabetes managed with insulin therapy."
        
        Extracted conditions:
        [
            {"condition": "Hypertension", "code": ""},
            {"condition": "Diabetes", "code": ""}
        ]
        """

    async def extract_conditions(self, text: str):
        """
        Extract medical conditions from a clinical progress note using OpenAI.

        This method sends the progress note to the model, instructing it to return the medical conditions and their codes.

        Args:
            text (str): The clinical progress note text to be processed.

        Returns:
            List[Condition]: A list of extracted conditions in a structured format (list of Condition objects).
        """
        # Format the clinical progress note into a prompt
        prompt = f"""
        Clinical Progress Note:
        {text}
        """
        logging.info(
            f"Extracting conditions from the progress note: {text[:100]}..."
        )  # Log first 100 characters of the note for visibility
        return await self._generate_response(prompt)
