from langchain_classic.chains import LLMChain, SequentialChain
from langchain_core.prompts import PromptTemplate
from langchain_classic.agents import AgentExecutor  # ancora valido nello stack moderno
from langgraph.graph import StateGraph       # equivale al vecchio "LangGraph"

from orchestrator import Orchestrator

def main():
    # Initialize the orchestrator
    orchestrator = Orchestrator()

    # Start the workflow
    orchestrator.run()

if __name__ == "__main__":
    main()