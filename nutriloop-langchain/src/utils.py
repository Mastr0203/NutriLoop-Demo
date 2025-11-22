from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate

from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import Tool

def create_prompt_template(template: str) -> PromptTemplate:
    """Create a prompt template for LangChain."""
    return PromptTemplate(template=template)

def initialize_langchain_agent(llm_chain: LLMChain, tools: list) -> None:
    """Initialize a LangChain agent with the specified LLM chain and tools."""
    agent = initialize_agent(
        tools=tools,
        llm=llm_chain,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent

def extract_tool_names(tools: list) -> list:
    """Extract names of tools from the provided list."""
    return [tool.name for tool in tools]

def log_action(action: str) -> None:
    """Log an action taken by the application."""
    print(f"Action logged: {action}")