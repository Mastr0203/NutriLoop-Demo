from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain.agents import AgentExecutor
from langchain_core.tools import Tool
from langchain_classic.memory import ConversationBufferMemory

def validate_meal_plan(plan: str, allergies: list, budget: float) -> tuple:
    reasons = []
    lower_plan = plan.lower()
    
    for allergen in allergies:
        if allergen.lower() in lower_plan:
            reasons.append(f"contains allergen '{allergen}'")
    
    if budget is not None:
        num_days = len([line for line in plan.splitlines() if line.strip()])
        estimated_cost = num_days * 3 * 5
        if estimated_cost > budget:
            reasons.append(f"estimated cost {estimated_cost} exceeds budget {budget}")
    
    return (len(reasons) == 0, "; ".join(reasons))

def derive_grocery_list(plan: str) -> dict:
    grocery_counts = {}
    
    for line in plan.splitlines():
        if ':' in line:
            _, meals_str = line.split(':', 1)
        else:
            meals_str = line
        
        for segment in meals_str.replace('–', ':').replace(';', ',').split(','):
            item = segment.strip()
            if not item:
                continue
            
            for prefix in ["Breakfast", "Lunch", "Dinner"]:
                if item.lower().startswith(prefix.lower()):
                    item = item[len(prefix):].strip(" –-: ")
            
            key = item.lower()
            if not key:
                continue
            grocery_counts[key] = grocery_counts.get(key, 0) + 1
    
    return grocery_counts