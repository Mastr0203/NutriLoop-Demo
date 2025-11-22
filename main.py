"""
Demo script for NutriLoop – a simplified agentic AI workflow for personalised nutrition.

This script demonstrates how a high‑level “orchestrator” can use a language model to
make decisions, call out to various tools (e.g. send emails, schedule calendar
entries, place grocery orders), and loop over tasks until a satisfactory result
is achieved.  The example revolves around building a weekly meal plan for a
patient based on a clinical report and a doctor’s target goal.  Along the way
the orchestrator checks that the goal is safe, gathers additional inputs from
the patient, generates and validates a meal plan, seeks doctor approval, and
finally schedules the next appointment and orders groceries.

Key agentic behaviours illustrated in this demo:

* **Non‑linear decision making** – the orchestrator evaluates whether the
  initial goal is safe; if not, it consults the doctor for a revised goal.
* **Tool calls** – sending emails to the doctor and patient, scheduling a
  calendar entry, and placing a grocery order are implemented as simple
  Python functions that print and log their actions.
* **Loops** – the meal plan is generated and then validated; if it fails
  validation (e.g. contains allergens or exceeds budget) the orchestrator
  loops back to the nutrition model with stricter instructions.
* **Integration with an LLM** – calls to OpenAI’s Chat API are used for
  reasoning about goal safety and for generating meal plans.  A fallback is
  provided so the script will still run even without an API key or the
  `openai` package installed.

To run this script you need Python 3 and optionally the `openai` package
installed.  If you wish to invoke the real model you must set the
`OPENAI_API_KEY` environment variable.  Without it the demo will fall back to
simple stubbed responses that allow the workflow to complete end‑to‑end.
"""

from __future__ import annotations

import os
import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# Try to import the OpenAI client.  If it's not available the demo will
# gracefully fall back to stubbed responses.  This allows the code to run
# without external dependencies for demonstration purposes.
try:
    import openai  # type: ignore
    _openai_available = True
except ImportError:
    openai = None  # type: ignore
    _openai_available = False


@dataclass
class NutriLoopState:
    """Container for all state information used throughout the NutriLoop run.

    The orchestrator updates this structure as it proceeds through the
    workflow: starting with patient and goal information, adding user
    preferences and allergies, storing the generated meal plan, recording
    scheduled visits, and logging messages to doctor/patient or actions
    performed.
    """

    patient: Dict[str, any]
    goal: Dict[str, any]
    # User‑provided preferences and constraints
    preferences: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    budget: Optional[float] = None
    # Resulting meal plan text
    meal_plan: str = ""
    # Next scheduled visit date in ISO format
    next_visit_date: Optional[str] = None
    # Log of tool invocations for demonstration/inspection purposes
    logs: List[Tuple[str, any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tool implementations
#
# These functions simulate side‑effects such as sending emails, scheduling
# calendar events or placing grocery orders.  In a real system these would
# integrate with email servers, calendar APIs or e‑commerce platforms.  Here
# they simply print to stdout and append to the state's log so you can see
# what would have happened.

def tool_mail_doctor(message: str, state: NutriLoopState) -> None:
    """Simulate sending an email to the doctor.

    This function prints the message and records it in the state's log.
    """
    print(f"[Mail to Doctor]: {message}")
    state.logs.append(("doctor_mail", message))


def tool_mail_patient(message: str, state: NutriLoopState) -> None:
    """Simulate sending an email or message to the patient.
    """
    print(f"[Mail to Patient]: {message}")
    state.logs.append(("patient_mail", message))


def tool_calendar_schedule_next_visit(date: str, state: NutriLoopState) -> None:
    """Simulate adding a visit to the calendar.

    The next visit date is stored on the state and printed.  In practice this
    would integrate with a calendar API.
    """
    state.next_visit_date = date
    print(f"[Calendar]: Next visit scheduled on {date}")
    state.logs.append(("calendar", date))


def tool_grocery_order(order: Dict[str, int], state: NutriLoopState) -> None:
    """Simulate placing a grocery order.

    The order is a mapping from ingredient names to quantities.  The order is
    printed and recorded in the state's log.  In a real implementation this
    would interface with a grocery delivery service.
    """
    print(f"[Grocery Order]: {order}")
    state.logs.append(("grocery_order", order))


# ---------------------------------------------------------------------------
# LLM helpers
#
# These functions wrap calls to OpenAI's ChatCompletion API.  They include a
# simple fallback to provide deterministic responses if the API key is not set
# or the `openai` package is unavailable.  This design keeps the code self‑
# contained and ensures the script will run on systems without external
# dependencies.

def _fallback_llm_response(messages: List[Dict[str, str]]) -> str:
    """Produce a simple deterministic response when the OpenAI API isn't available.

    This helper examines the user prompt and returns canned text for two
    specific tasks: checking goal safety and generating a weekly meal plan.
    For any other prompt it returns a generic acknowledgement.
    """
    # Find the last user message to decide what to stub
    last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "").lower()
    # Safety check: respond with 'safe' or 'unsafe'
    if "safe" in last_user_msg and "goal" in last_user_msg:
        # If the goal mentions aggressive weight loss or known eating disorder risk, flag as unsafe
        if any(term in last_user_msg for term in ["rapid", "aggressive", "anorexia", "dangerous"]):
            return "unsafe: The goal is too aggressive and may pose health risks."
        return "safe: The goal appears reasonable."
    # Meal plan generation stub
    if "meal plan" in last_user_msg or "weekly meal plan" in last_user_msg:
        # Very simple plan tailored minimally to avoid listed allergens
        # Extract allergies from prompt if possible
        plan_lines = []
        # Create a list of simple foods that avoid common allergens
        safe_foods = ["oatmeal", "yogurt", "salad", "grilled chicken", "steamed vegetables", "brown rice", "fruit", "quinoa", "lentil soup"]
        # Build seven days of three meals each
        for day in range(1, 8):
            breakfast = safe_foods[(day * 3) % len(safe_foods)]
            lunch = safe_foods[(day * 3 + 1) % len(safe_foods)]
            dinner = safe_foods[(day * 3 + 2) % len(safe_foods)]
            plan_lines.append(
                f"Day {day}: Breakfast – {breakfast}; Lunch – {lunch}; Dinner – {dinner}"
            )
        return "\n".join(plan_lines)
    # Generic fallback
    return "ok"


def call_llm(messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo", temperature: float = 0.5) -> str:
    """Call the OpenAI ChatCompletion API or fall back to a stubbed response.

    Parameters
    ----------
    messages : List[Dict[str, str]]
        The chat history formatted as a list of dictionaries with 'role' and
        'content' keys as required by OpenAI's API.
    model : str, optional
        The model identifier.  Defaults to ``gpt-3.5-turbo``.  You may change
        this to ``gpt-4`` if your API subscription supports it.
    temperature : float, optional
        Sampling temperature to use for OpenAI requests.

    Returns
    -------
    str
        The assistant's message content.
    """
    # Use the real API if available and an API key has been provided
    if _openai_available and os.getenv("OPENAI_API_KEY"):
        try:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return response["choices"][0]["message"]["content"]
        except Exception as exc:
            # Print an error and fall back to stubbed behaviour
            print(f"Warning: OpenAI API call failed ({exc}). Falling back to stubbed response.")
            return _fallback_llm_response(messages)
    else:
        # No API key or openai package; use stub
        return _fallback_llm_response(messages)


# ---------------------------------------------------------------------------
# Domain‑specific validation functions
#
# These functions encapsulate simple business rules that do not require an LLM.
# They demonstrate how an agent might validate LLM output using pure Python
# logic before accepting it.  In a real system these checks would be more
# sophisticated and grounded in nutritional science.

def validate_meal_plan(plan: str, allergies: List[str], budget: Optional[float]) -> Tuple[bool, str]:
    """Check that the meal plan does not violate allergies and stays within budget.

    For demo purposes the budget check is extremely rough: it assumes each
    meal costs around 5 monetary units.  If a budget is provided and the
    estimated cost exceeds it, the plan is marked invalid.  The allergy
    check flags any direct mention of a forbidden food in the plan text.

    Parameters
    ----------
    plan : str
        The textual meal plan to validate.
    allergies : List[str]
        List of foods that must not appear in the plan.
    budget : float or None
        The weekly budget.  If ``None`` the budget check is skipped.

    Returns
    -------
    (bool, str)
        A tuple of a boolean indicating validity and a string with reasons
        explaining why the plan failed validation.  An empty reason string
        means the plan passed all checks.
    """
    reasons = []
    # Allergy check: look for forbidden foods as substrings (case‑insensitive)
    lower_plan = plan.lower()
    for allergen in allergies:
        if allergen.lower() in lower_plan:
            reasons.append(f"contains allergen '{allergen}'")
    # Budget check: assume each line corresponds to a day with three meals,
    # costing roughly 5 units per meal (15 per day).  This is deliberately
    # simplistic for demonstration.
    if budget is not None:
        num_days = len([line for line in plan.splitlines() if line.strip()])
        estimated_cost = num_days * 3 * 5  # 5 units per meal
        if estimated_cost > budget:
            reasons.append(f"estimated cost {estimated_cost} exceeds budget {budget}")
    return (len(reasons) == 0, "; ".join(reasons))


def derive_grocery_list(plan: str) -> Dict[str, int]:
    """Derive a simple grocery list from the meal plan.

    This function parses each line of the meal plan looking for food items
    separated by punctuation.  It counts occurrences of each item across
    the entire week.  The result is a dictionary mapping ingredient names to
    approximate quantity counts.  Parsing is approximate and intended only
    for demonstration purposes.
    """
    grocery_counts: Dict[str, int] = {}
    for line in plan.splitlines():
        # Split on colon and semicolon to separate meals
        if ':' in line:
            _, meals_str = line.split(':', 1)
        else:
            meals_str = line
        # Replace separators with commas and split
        for segment in meals_str.replace('–', ':').replace(';', ',').split(','):
            item = segment.strip()
            # Skip empty segments
            if not item:
                continue
            # Remove descriptors like 'Breakfast', 'Lunch', 'Dinner'
            for prefix in ["Breakfast", "Lunch", "Dinner"]:
                if item.lower().startswith(prefix.lower()):
                    item = item[len(prefix):].strip(" –-: ")
            # Lowercase for key consistency
            key = item.lower()
            if not key:
                continue
            grocery_counts[key] = grocery_counts.get(key, 0) + 1
    return grocery_counts


# ---------------------------------------------------------------------------
# Main orchestrator logic
#
# The run_demo function ties everything together.  It progresses through the
# workflow step by step, interleaving calls to the LLM with deterministic
# validations and tool invocations.  Comments throughout highlight where
# agentic decision making, loops, and tool usage occur.

def run_demo() -> None:
    """Execute a single NutriLoop workflow using a hardcoded patient and goal.

    The steps executed are:

    1. Initialise state with a sample patient profile and goal.
    2. Use the LLM to assess if the goal is safe.  If unsafe, contact the
       doctor via `tool_mail_doctor` and simulate a revised goal.
    3. Collect user preferences, allergies and budget interactively via
       `input()`.  These are stored on the state.
    4. Request a weekly meal plan from the nutrition LLM.  Validate the
       plan with pure Python logic.  If it fails (e.g. due to allergies or
       budget), loop back to generate a stricter plan.  This illustrates
       iterative improvement.
    5. Send the plan to the doctor for approval.  A simple simulation is
       used to either approve the plan or request a minor change.
    6. Upon approval schedule the next visit and notify the patient.
    7. Derive a grocery list from the plan and place a grocery order.

    Throughout the process printed output and state logs show the effect of
    each tool call and decision.  This function can be invoked directly from
    the command line.
    """
    # Step 1: Define a sample patient profile and goal
    patient = {
        "name": "Jane Doe",
        "age": 30,
        "sex": "female",
        "medical_conditions": ["hypertension"],
        "allergies": ["peanuts"]
    }
    # The initial goal intentionally contains unrealistic expectations to
    # demonstrate the safety check.  Feel free to modify this for testing.
    goal = {
        "type": "weight_loss",
        "target": "lose 10 kg in 2 weeks",
        "notes": "Patient has a history of disordered eating"
    }
    # Initialise the orchestrator state
    state = NutriLoopState(patient=patient, goal=goal)

    # Step 2: Use the LLM to assess whether the goal is safe
    # The orchestrator formulates a prompt describing the patient and the goal and
    # asks for a simple 'safe' or 'unsafe' verdict.  The assistant may also
    # provide a brief explanation.  This is an example of delegating a
    # reasoning step to the LLM.
    safety_messages = [
        {
            "role": "system",
            "content": (
                "You are a medical safety assistant.  Given a patient's profile and a goal,"
                " decide whether the goal is medically safe.  Respond with either"
                " 'safe' or 'unsafe' and briefly explain why."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Patient profile: {state.patient}. Goal: {state.goal}.\n"
                "Is this goal safe?"
            ),
        },
    ]
    safety_result = call_llm(safety_messages)
    print(f"Safety assessment result: {safety_result}")
    # If the goal is unsafe, contact the doctor and substitute a revised goal
    if "unsafe" in safety_result.lower():
        # Agentic decision: the goal is unsafe, so we need to involve the doctor
        tool_mail_doctor(
            "The initial goal appears unsafe. Please provide a safer alternative.",
            state,
        )
        # Simulate doctor providing a revised goal (in a real system this would
        # arrive asynchronously via email or another channel).  Here we just
        # hard‑code the response for demonstration.
        revised_goal = {
            "type": state.goal.get("type", "unknown"),
            "target": "lose 5 kg in 8 weeks",
            "notes": "Adjusted to a more gradual pace per safety guidelines"
        }
        print(f"Doctor's revised goal received: {revised_goal}")
        state.goal = revised_goal
    else:
        print("The goal has been deemed safe.")

    # Step 3: Collect user preferences, allergies and budget
    # These inputs personalise the meal plan.  They are treated as part of the
    # agent's evolving state.
    try:
        prefs_input = input(
            "Enter your preferred foods or cuisines (comma‑separated), or leave blank: "
        )
    except EOFError:
        # If the script is run in an environment without a stdin (e.g. notebook),
        # fall back to default preferences
        prefs_input = ""
    state.preferences = [p.strip() for p in prefs_input.split(",") if p.strip()]
    try:
        allergies_input = input(
            "Enter any additional food allergies (comma‑separated), or leave blank: "
        )
    except EOFError:
        allergies_input = ""
    additional_allergies = [a.strip() for a in allergies_input.split(",") if a.strip()]
    # Combine patient allergies with those provided interactively
    state.allergies = list({*(state.patient.get("allergies", [])), *additional_allergies})
    try:
        budget_input = input(
            "Enter your weekly food budget (numeric), or leave blank for no limit: "
        )
    except EOFError:
        budget_input = ""
    if budget_input.strip():
        try:
            state.budget = float(budget_input.strip())
        except ValueError:
            print("Could not parse budget; ignoring budget constraint.")
            state.budget = None
    else:
        state.budget = None

    # Step 4: Generate a weekly meal plan and validate it.  If the plan
    # violates constraints (allergies, budget), request a stricter plan.  This
    # demonstrates an agentic loop: the orchestrator does not accept the first
    # answer blindly but iterates until a satisfactory solution is found.
    attempt = 0
    while True:
        attempt += 1
        # Compose a prompt for the nutrition LLM.  We instruct it to
        # produce a week‑long plan with three meals per day, taking into
        # account preferences, allergies and budget.  We encourage concise
        # formatting for easier parsing.
        nutrition_messages = [
            {
                "role": "system",
                "content": (
                    "You are a nutritionist tasked with creating simple weekly meal plans."
                    " The plan should cover 7 days with breakfast, lunch and dinner."
                    " Avoid any foods the patient is allergic to and aim to respect"
                    " the stated budget when possible.  List each day's meals on a"
                    " separate line in the format: 'Day X: Breakfast – ...; Lunch – ...; Dinner – ...'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Patient preferences: {state.preferences}.\n"
                    f"Allergies: {state.allergies}.\n"
                    f"Budget: {state.budget if state.budget is not None else 'no budget specified'}.\n"
                    f"Please generate a weekly meal plan."
                ),
            },
        ]
        plan = call_llm(nutrition_messages, temperature=0.7)
        state.meal_plan = plan.strip()
        print(f"Generated meal plan (attempt {attempt}):\n{state.meal_plan}\n")
        # Validate the plan using deterministic logic
        valid, reason = validate_meal_plan(state.meal_plan, state.allergies, state.budget)
        if valid:
            print("Meal plan validation passed.")
            break
        else:
            # If validation fails we ask the nutrition LLM to try again with
            # stricter instructions.  The loop continues until a valid plan is produced.
            print(f"Meal plan validation failed: {reason}.\nRegenerating with stricter constraints...\n")
            # Modify the user prompt to emphasise the issue for the next attempt
            nutrition_messages[-1]["content"] += f"\nPlease avoid the following foods: {', '.join(state.allergies)}."
            nutrition_messages[-1]["content"] += (
                " Also ensure the total estimated cost does not exceed the patient's budget."
                if state.budget is not None
                else ""
            )
            # Continue the while loop to generate a new plan
            continue

    # Step 5: Send the plan to the doctor for approval.  In this demo the
    # doctor's reply is simulated: if the plan contains fried foods the doctor
    # asks to replace them with grilled versions; otherwise the plan is approved.
    tool_mail_doctor(f"Proposed meal plan:\n{state.meal_plan}", state)
    # Simulate doctor's response
    doctor_feedback: Optional[str] = None
    if "fried" in state.meal_plan.lower():
        doctor_feedback = "Please replace all fried items with grilled alternatives."
    # Apply doctor's feedback if any
    if doctor_feedback:
        print(f"Doctor responded with feedback: {doctor_feedback}")
        # Simple replacement logic to adjust the plan
        state.meal_plan = state.meal_plan.lower().replace("fried", "grilled")
        print("Adjusted meal plan after doctor's feedback:\n" + state.meal_plan)
    else:
        print("Doctor approved the meal plan.")

    # Step 6: Schedule the next visit and notify the patient
    # For this demo we schedule the visit two weeks from today.  The date is
    # formatted in ISO YYYY‑MM‑DD.  We assume the user's locale/timezone is
    # Europe/Rome but for simplicity we use the system's current date.
    next_visit = datetime.date.today() + datetime.timedelta(days=14)
    tool_calendar_schedule_next_visit(next_visit.isoformat(), state)
    # Compose a message to the patient summarising the plan and next visit
    patient_message = (
        f"Hello {state.patient['name']},\n\nHere is your personalised weekly meal plan:\n\n"
        f"{state.meal_plan}\n\n"
        f"Your next appointment is scheduled on {state.next_visit_date}.\n"
        "Please let us know if you have any questions or concerns."
    )
    tool_mail_patient(patient_message, state)

    # Step 7: Derive a grocery list from the meal plan and place the order
    grocery_list = derive_grocery_list(state.meal_plan)
    tool_grocery_order(grocery_list, state)

    # Final state overview (optional)
    print("\nRun completed. State summary:")
    print(state)


if __name__ == "__main__":
    run_demo()