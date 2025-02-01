# core/condition_extraction.py

from models.vertex_ai_model import VertexAIModel


def extract_conditions_from_progress_note(file_path: str):
    """
    Extract conditions from a clinical progress note using ChatVertexAI.

    Args:
        file_path (str): Path to the clinical progress note file.

    Returns:
        list: A list of extracted conditions in a structured format.
    """
    with open(file_path, "r") as file:
        text = file.read()

    # Initialize VertexAIModel instance
    vertex_ai_model = VertexAIModel()

    # Extract conditions from the progress note
    conditions = vertex_ai_model.extract_conditions(text)

    # Return extracted conditions (already structured as a list)
    return conditions
