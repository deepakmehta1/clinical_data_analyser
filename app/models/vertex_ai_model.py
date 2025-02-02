# models/vertex_ai_model.py

from langchain_google_vertexai import ChatVertexAI
import logging


class VertexAIModel:
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        """
        Initialize the Vertex AI model wrapper.

        Args:
            model_name (str): The name of the model to use (default is "gemini-1.5-pro").
        """
        self.model_name = model_name
        self.model = ChatVertexAI(model=model_name)

    async def _generate_response(self, prompt: str, schema: dict):
        """
        Helper function to generate a response from Vertex AI based on the provided prompt and schema.

        Args:
            prompt (str): The input text or prompt to send to the model.
            schema (dict): The schema to guide the structured output.

        Returns:
            dict: The generated structured response from the model.
        """
        try:
            # Bind schema to the model for structured output
            model_with_structure = self.model.with_structured_output(schema)

            # Generate response using the model with the provided schema
            response = await model_with_structure.ainvoke(prompt)
            return response
        except Exception as e:
            raise RuntimeError(f"Error generating response from Vertex AI: {e}")

    async def extract_conditions(self, text: str):
        """
        Extract medical conditions from a clinical progress note using Vertex AI.

        Args:
            text (str): The clinical progress note text.

        Returns:
            list: List of extracted conditions in a structured format.
        """
        schema = {
            "condition": "string",  # The name of the medical condition
            "description": "string",  # Optional description for the condition
        }

        prompt = f"""
        Please extract the medical conditions mentioned in the following clinical progress note
        Clinical Progress Note:
        {text}
        """

        return await self._generate_response(prompt, schema)
