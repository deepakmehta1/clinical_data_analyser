# tools.py
from langchain_core.tools import tool
from core.hcc_relevance_check import check_hcc_relevance
from config import settings


@tool
def check_hcc_relevance_tool(conditions: list) -> list:
    """
    A tool to check HCC relevance for extracted conditions.

    Args:
        conditions (list): List of extracted conditions from progress notes.

    Returns:
        list: A list of conditions with associated HCC codes.
    """
    # Use the HCC relevance check function from core
    hcc_codes_path = settings.HCC_CODES_PATH
    relevant_conditions = check_hcc_relevance(conditions, hcc_codes_path)

    return relevant_conditions
