# main.py
import sys
from langgraph_app import SimpleLangGraphApp
from config import PROJECT_ID, LOCATION


def main():
    agent = SimpleLangGraphApp(project=PROJECT_ID, location=LOCATION)
    agent.set_up()

    # Example queries
    print(agent.query("Get product details for shoes"))
    print(agent.query("Get product details for coffee"))
    print(agent.query("Get product details for smartphone"))
    print(agent.query("Tell me about the weather"))


if __name__ == "__main__":
    main()
