from __future__ import annotations

from typing import Dict, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph

from agents.nutrition_agent import NutritionAgent
from agents.safety_agent import SafetyAgent
from validators import derive_grocery_list, validate_meal_plan


class WorkflowState(TypedDict, total=False):
    """Typed state passed between LangGraph nodes."""

    patient_info: Dict[str, object]
    goal: Dict[str, object]
    safety_status: Literal["safe", "unsafe"]
    safety_message: str
    preferences: Dict[str, object]
    meal_plan: str
    validation_passed: bool
    validation_reason: str
    grocery_list: Dict[str, int]


class Orchestrator:
    def __init__(
        self,
        llm_client,
        nutrition_agent: NutritionAgent,
        safety_agent: SafetyAgent,
        flow=None,
    ):
        self.llm_client = llm_client
        self.nutrition_agent = nutrition_agent
        self.safety_agent = safety_agent
        self.flow = flow or self._build_flow()

    # ------------------------------------------------------------------
    # LangGraph setup
    # ------------------------------------------------------------------
    def _build_flow(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("assess_goal", self._node_assess_goal)
        graph.add_node("collect_preferences", self._node_collect_preferences)
        graph.add_node("generate_plan", self._node_generate_plan)
        graph.add_node("validate_plan", self._node_validate_plan)
        graph.add_node("finalize", self._node_finalize)

        graph.set_entry_point("assess_goal")
        graph.add_edge("assess_goal", "collect_preferences")
        graph.add_edge("collect_preferences", "generate_plan")
        graph.add_edge("generate_plan", "validate_plan")
        graph.add_conditional_edges(
            "validate_plan",
            self._validation_branch,
            {
                "retry": "generate_plan",
                "approved": "finalize",
            },
        )
        graph.add_edge("finalize", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run_workflow(self, patient_info: Dict[str, object], goal: Dict[str, object]):
        """Execute the compiled LangGraph workflow."""

        initial_state: WorkflowState = {
            "patient_info": patient_info,
            "goal": goal,
        }
        return self.flow.invoke(initial_state)

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------
    def _node_assess_goal(self, state: WorkflowState) -> WorkflowState:
        assessment = self.safety_agent.assess_goal_safety(state["patient_info"], state["goal"])
        status: Literal["safe", "unsafe"] = "unsafe" if "unsafe" in assessment.lower() else "safe"
        return {
            **state,
            "safety_status": status,
            "safety_message": assessment,
        }

    def _node_collect_preferences(self, state: WorkflowState) -> WorkflowState:
        preferences = self.gather_user_preferences(state.get("patient_info"))
        return {**state, "preferences": preferences}

    def _node_generate_plan(self, state: WorkflowState) -> WorkflowState:
        preferences: Dict[str, object] = state.get("preferences", {})
        allergies = preferences.get("allergies") or state.get("patient_info", {}).get("allergies", [])
        budget: Optional[float] = preferences.get("budget")  # type: ignore[arg-type]
        meal_plan = self.nutrition_agent.generate_meal_plan(
            preferences.get("preferences", []),
            allergies,
            budget,
        )
        return {**state, "meal_plan": meal_plan}

    def _node_validate_plan(self, state: WorkflowState) -> WorkflowState:
        allergies = state.get("preferences", {}).get("allergies") or state.get("patient_info", {}).get("allergies", [])
        budget: Optional[float] = state.get("preferences", {}).get("budget")  # type: ignore[arg-type]
        valid, reason = validate_meal_plan(state.get("meal_plan", ""), allergies, budget)
        return {
            **state,
            "validation_passed": valid,
            "validation_reason": reason,
        }

    def _node_finalize(self, state: WorkflowState) -> WorkflowState:
        grocery_list = derive_grocery_list(state.get("meal_plan", ""))
        return {**state, "grocery_list": grocery_list}

    # ------------------------------------------------------------------
    # Conditional routing helpers
    # ------------------------------------------------------------------
    def _validation_branch(self, state: WorkflowState) -> Literal["retry", "approved"]:
        return "approved" if state.get("validation_passed") else "retry"

    # ------------------------------------------------------------------
    # External hooks (still stubs for now)
    # ------------------------------------------------------------------
    def gather_user_preferences(self, patient_info: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        """Placeholder for collecting user preferences via UI or CLI."""

        allergies = patient_info.get("allergies", []) if patient_info else []
        return {
            "preferences": [],
            "allergies": allergies,
            "budget": None,
        }

    def notify_doctor(self, message):
        pass

    def validate_meal_plan(self, meal_plan, allergies):
        pass

    def notify_patient(self, meal_plan):
        pass

    def schedule_next_visit(self):
        pass
