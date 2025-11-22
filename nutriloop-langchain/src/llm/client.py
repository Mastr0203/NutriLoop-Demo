from langchain import OpenAI
from langchain.llms import LLM

class LLMClient:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.llm = OpenAI(model_name=model_name)

    def generate_response(self, prompt: str) -> str:
        response = self.llm(prompt)
        return response.strip()