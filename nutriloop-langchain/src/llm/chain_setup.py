from langchain_classic.chains import LLMChain, SequentialChain
from langchain_core.prompts import PromptTemplate

from .client import LLMClient

def setup_chains():
    # Define prompt templates for different tasks
    safety_prompt = PromptTemplate(
        input_variables=["patient_profile", "goal"],
        template="Given the patient profile: {patient_profile} and the goal: {goal}, is this goal safe? Respond with 'safe' or 'unsafe' and explain why."
    )

    meal_plan_prompt = PromptTemplate(
        input_variables=["preferences", "allergies", "budget"],
        template="Create a weekly meal plan considering the following preferences: {preferences}, avoiding these allergies: {allergies}, and respecting a budget of {budget}."
    )

    # Create LLM chains for safety assessment and meal planning
    safety_chain = LLMChain(
        llm=LLMClient(),  # Assuming LLMClient is defined in client.py
        prompt=safety_prompt
    )

    meal_plan_chain = LLMChain(
        llm=LLMClient(),
        prompt=meal_plan_prompt
    )

    # Set up a sequential chain if needed for workflow
    overall_chain = SequentialChain(
        chains=[safety_chain, meal_plan_chain],
        input_variables=["patient_profile", "goal", "preferences", "allergies", "budget"],
        output_variables=["safety_result", "meal_plan"]
    )

    return overall_chain