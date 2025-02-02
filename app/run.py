# run.py

import sys
import asyncio
from services.langgraph_service import LangGraphService
from config import settings
from models.open_ai_model import OpenAIModel


async def main():
    try:
        # Initialize LangGraph service with project and location
        langgraph_service = LangGraphService(
            settings.PROJECT_ID, settings.LOCATION, OpenAIModel(model_name="gpt-4o")
        )

        # Set up the LangGraph workflow
        langgraph_service.set_up()

        # Run the LangGraph workflow and get the relevant conditions
        relevant_conditions = await langgraph_service.run()

        # Output the relevant conditions
        print("Relevant HCC Conditions for All Progress Notes:")
        for condition in relevant_conditions:
            print(
                f"Condition: {condition['condition']}, HCC Codes: {', '.join(condition['hcc_codes'])}"
            )

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())  # Run the async main function
