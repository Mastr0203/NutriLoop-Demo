from src.validators import validate_meal_plan, derive_grocery_list

def test_validate_meal_plan():
    # Test case 1: Valid meal plan
    plan = "Day 1: Breakfast – oatmeal; Lunch – salad; Dinner – grilled chicken"
    allergies = ["peanuts"]
    budget = 100
    valid, reason = validate_meal_plan(plan, allergies, budget)
    assert valid == True
    assert reason == ""

    # Test case 2: Meal plan contains allergen
    plan = "Day 2: Breakfast – yogurt; Lunch – peanut butter sandwich; Dinner – steamed vegetables"
    allergies = ["peanuts"]
    valid, reason = validate_meal_plan(plan, allergies, budget)
    assert valid == False
    assert "contains allergen 'peanuts'" in reason

    # Test case 3: Meal plan exceeds budget
    plan = "Day 3: Breakfast – oatmeal; Lunch – salad; Dinner – grilled chicken"
    allergies = []
    budget = 15  # Only enough for one day
    valid, reason = validate_meal_plan(plan, allergies, budget)
    assert valid == False
    assert "estimated cost 45 exceeds budget 15" in reason

def test_derive_grocery_list():
    # Test case 1: Simple meal plan
    plan = "Day 1: Breakfast – oatmeal; Lunch – salad; Dinner – grilled chicken"
    grocery_list = derive_grocery_list(plan)
    expected_list = {"oatmeal": 1, "salad": 1, "grilled chicken": 1}
    assert grocery_list == expected_list

    # Test case 2: Meal plan with repeated items
    plan = "Day 1: Breakfast – oatmeal; Lunch – salad; Dinner – grilled chicken\nDay 2: Breakfast – oatmeal; Lunch – salad; Dinner – grilled chicken"
    grocery_list = derive_grocery_list(plan)
    expected_list = {"oatmeal": 2, "salad": 2, "grilled chicken": 2}
    assert grocery_list == expected_list

    # Test case 3: Meal plan with no meals
    plan = ""
    grocery_list = derive_grocery_list(plan)
    expected_list = {}
    assert grocery_list == expected_list