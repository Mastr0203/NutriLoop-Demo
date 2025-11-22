from langchain_openai import ChatOpenAI


class LLMClient:
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature
        )

    def generate_response(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content.strip()