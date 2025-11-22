from langchain.agents import Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def create_grocery_order_tool() -> Tool:
    """Creates a tool for placing grocery orders."""
    return Tool(
        name="GroceryOrder",
        description="Tool to place grocery orders based on a meal plan.",
        func=place_grocery_order,
    )

def place_grocery_order(order: dict) -> str:
    """Simulate placing a grocery order.

    Args:
        order (dict): A dictionary mapping ingredient names to quantities.

    Returns:
        str: Confirmation message of the grocery order.
    """
    # Here you would integrate with a grocery delivery service.
    # For demonstration, we will just return a confirmation message.
    order_summary = ", ".join([f"{quantity}x {ingredient}" for ingredient, quantity in order.items()])
    return f"Grocery order placed for: {order_summary}"

def derive_grocery_list(meal_plan: str) -> dict:
    """Derive a grocery list from the meal plan.

    Args:
        meal_plan (str): The textual meal plan to parse.

    Returns:
        dict: A dictionary mapping ingredient names to quantities.
    """
    grocery_counts = {}
    for line in meal_plan.splitlines():
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
            if key:
                grocery_counts[key] = grocery_counts.get(key, 0) + 1
    return grocery_counts