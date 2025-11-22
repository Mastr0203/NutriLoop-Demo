import os
from typing import Optional

from langchain_openai import ChatOpenAI


class LLMClient:
    """Thin wrapper around ChatOpenAI with a stub fallback when no API key is present."""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0):
        api_key = os.getenv("OPENAI_API_KEY")
        self.is_stub = api_key is None or api_key.strip() == ""

        if self.is_stub:
            # Keep ``llm`` None to avoid raising inside ChatOpenAI when no key is provided.
            self.llm: Optional[ChatOpenAI] = None
        else:
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=api_key,
            )

    def generate_response(self, prompt: str) -> str:
        if not self.llm:
            return "LLM unavailable: provide OPENAI_API_KEY to enable model responses."
        response = self.llm.invoke(prompt)
        return response.content.strip()
