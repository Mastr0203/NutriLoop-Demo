from src.orchestrator import Orchestrator
import pytest

@pytest.fixture
def orchestrator():
    return Orchestrator()

def test_initialization(orchestrator):
    assert orchestrator is not None

def test_workflow_execution(orchestrator):
    result = orchestrator.run_workflow()
    assert result is not None
    assert isinstance(result, dict)  # Assuming the result is a dictionary

def test_goal_safety_assessment(orchestrator):
    safe_goal = {"type": "weight_loss", "target": "lose 1 kg per week"}
    unsafe_goal = {"type": "weight_loss", "target": "lose 10 kg in 2 weeks"}
    
    assert orchestrator.assess_goal_safety(safe_goal) == "safe"
    assert orchestrator.assess_goal_safety(unsafe_goal) == "unsafe"

def test_meal_plan_generation(orchestrator):
    preferences = ["chicken", "vegetables"]
    allergies = ["peanuts"]
    budget = 100.0
    
    meal_plan = orchestrator.generate_meal_plan(preferences, allergies, budget)
    assert meal_plan is not None
    assert isinstance(meal_plan, str)  # Assuming the meal plan is a string
    assert all(allergen not in meal_plan for allergen in allergies)

def test_grocery_ordering(orchestrator):
    grocery_list = {"chicken": 2, "vegetables": 5}
    order_result = orchestrator.place_grocery_order(grocery_list)
    
    assert order_result is True  # Assuming the order returns a boolean for success