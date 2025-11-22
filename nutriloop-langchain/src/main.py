from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from agents.nutrition_agent import NutritionAgent
from agents.safety_agent import SafetyAgent
from llm.client import LLMClient
from orchestrator import Orchestrator


def main():
    llm_client = LLMClient()
    llm = llm_client.llm

    safety_prompt = PromptTemplate(
        input_variables=["patient_profile", "goal"],
        template=(
            "You are a medical safety assistant. Given the following patient profile and goal, "
            "assess whether the goal is medically safe. Respond with either 'safe' or 'unsafe' "
            "and provide a brief explanation.\n\n"
            "Patient Profile: {patient_profile}\n"
            "Goal: {goal}\n"
            "Is this goal safe?"
        ),
    )
    safety_chain = LLMChain(llm=llm, prompt=safety_prompt) if llm else None

    nutrition_agent = NutritionAgent(llm)
    safety_agent = SafetyAgent(safety_chain)

    orchestrator = Orchestrator(llm_client, nutrition_agent, safety_agent)

    patient_info = {"name": "Jane Doe", "allergies": ["peanuts"]}
    goal = {"type": "weight_loss", "target": "lose 5 kg in 8 weeks"}

    result_state = orchestrator.run_workflow(patient_info, goal)
    print("Workflow finished. Final state:")
    print(result_state)


if __name__ == "__main__":
    main()
