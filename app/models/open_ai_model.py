# models/openai_model.py

import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from config import settings
from typing import List, Optional


# Define the Condition schema for structured output
class Condition(BaseModel):
    condition: str
    description: Optional[str] = None


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

    async def _generate_response(self, prompt: str):
        """
        Helper function to generate a response from OpenAI based on the provided prompt.

        Args:
            prompt (str): The input text or prompt to send to the model.

        Returns:
            str: The generated response from OpenAI.
        """
        try:
            # Create the messages in LangChain format
            messages = [
                SystemMessage(
                    content=self.system_prompt()
                ),  # Add the system message (clinical expert)
                HumanMessage(
                    content=prompt
                ),  # Add the user prompt (clinical progress note)
            ]

            # Bind schema to the model for structured output (list of conditions)
            model_with_structure = self.chat.with_structured_output(ConditionList)

            # Invoke the model
            structured_output = await model_with_structure.ainvoke(messages)
            return structured_output
        except Exception as e:
            raise RuntimeError(f"Error generating response from OpenAI: {e}")

    def system_prompt(self):
        """
        Generate the system prompt to instruct the model to act as a clinical expert.

        Returns:
            str: The system prompt string.
        """
        return """
        You are a clinical expert. Your job is to extract the medical condition(s) and their description from the given clinical progress note.

        Your task:
        - For each medical condition, provide a condition name and an optional description.
        - You are to extract the conditions in the following format:
            [
                {"condition": "<condition_name>", "description": "<optional_description>"}
            ]

        Example 1:
        Clinical Progress Note:
        "The patient presents with a history of hypertension and diabetes. Hypertension has been controlled with medication. Diabetes management includes insulin therapy."
        
        Extracted conditions:
        [
            {"condition": "Hypertension", "description": "Controlled with medication."},
            {"condition": "Diabetes", "description": "Managed with insulin therapy."}
        ]

        Example 2:
        Clinical Progress Note:
        "Patient is experiencing chest pain, and has been diagnosed with chronic asthma."
        
        Extracted conditions:
        [
            {"condition": "Chest Pain", "description": "Patient is experiencing chest pain."},
            {"condition": "Asthma", "description": "Chronic condition affecting the airways."}
        ]
        """

    async def extract_conditions(self, text: str):
        """
        Extract medical conditions from a clinical progress note using OpenAI.

        Args:
            text (str): The clinical progress note text.

        Returns:
            List[Condition]: List of extracted conditions in a structured format (list of Condition objects).
        """
        prompt = f"""
        Clinical Progress Note:
        {text}
        """

        return await self._generate_response(prompt)
