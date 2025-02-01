# models/vertex_ai_model.py
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import PromptTemplate


class VertexAIModel:
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        """
        Initialize the Vertex AI model wrapper.

        Args:
            model_name (str): The name of the model to use (default is "gemini-1.5-pro").
        """
        self.model_name = model_name
        self.model = ChatVertexAI(model=model_name)

    def _generate_response(self, prompt: str):
        """
        Helper function to generate a response from Vertex AI based on the provided prompt.

        Args:
            prompt (str): The input text or prompt to send to the model.

        Returns:
            str: The generated response from the model.
        """
        try:
            response = self.model.predict(prompt)
            return response
        except Exception as e:
            raise RuntimeError(f"Error generating response from Vertex AI: {e}")

    def extract_conditions(self, text: str):
        """
        Extract medical conditions from a clinical progress note using Vertex AI.

        Args:
            text (str): The clinical progress note text.

        Returns:
            list: List of extracted conditions in a structured format.
        """
        prompt = f"""
        Please extract the medical conditions mentioned in the following clinical progress note. Provide the output in the following structure: 
        [
            {"condition": "<condition_name>", "description": "<optional_description>"}
        ]
        
        Clinical Progress Note:
        {text}
        """

        return self._generate_response(prompt)
