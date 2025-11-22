from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor
from langchain_core.tools import Tool
from langchain_classic.memory import ConversationBufferMemory

class NutritionAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.memory = ConversationBufferMemory()
        self.prompt_template = PromptTemplate(
            input_variables=["preferences", "allergies", "budget"],
            template=(
                "You are a nutritionist tasked with creating a weekly meal plan. "
                "Consider the following preferences: {preferences}. "
                "Avoid these allergens: {allergies}. "
                "The budget is: {budget}. "
                "Generate a meal plan for 7 days with breakfast, lunch, and dinner."
            )
        )
        self.chain = LLMChain(llm=self.llm_client, prompt=self.prompt_template)

    def generate_meal_plan(self, preferences, allergies, budget):
        self.memory.add_user_message(f"Preferences: {preferences}, Allergies: {allergies}, Budget: {budget}")
        meal_plan = self.chain.run(preferences=preferences, allergies=allergies, budget=budget)
        self.memory.add_assistant_message(meal_plan)
        return meal_plan

    def get_memory(self):
        return self.memory.get_messages()