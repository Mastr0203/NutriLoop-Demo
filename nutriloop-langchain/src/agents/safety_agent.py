from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate

class SafetyAgent:
    def __init__(self, llm_chain: LLMChain):
        self.llm_chain = llm_chain

    def assess_goal_safety(self, patient_profile: dict, goal: dict) -> str:
        prompt = PromptTemplate(
            input_variables=["patient_profile", "goal"],
            template=(
                "You are a medical safety assistant. Given the following patient profile and goal, "
                "assess whether the goal is medically safe. Respond with either 'safe' or 'unsafe' "
                "and provide a brief explanation.\n\n"
                "Patient Profile: {patient_profile}\n"
                "Goal: {goal}\n"
                "Is this goal safe?"
            )
        )
        
        messages = prompt.format(patient_profile=patient_profile, goal=goal)
        response = self.llm_chain.run(messages)
        return response.strip()