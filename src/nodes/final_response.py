import json
from src.state import AgentState


def final_response_node(state: AgentState) -> AgentState:
    """
    Node 4: Packages everything into the final structured output.
    """
    trace = state.get("trace", [])
    trace.append(">> [final_response_node] Packaging final response...")

    final = {
        "intent":       state.get("intent"),
        "route":        state.get("route"),
        "review":       state.get("review_status"),
        "final_answer": state.get("raw_answer"),
    }

    trace.append("   [final_response_node] Done.")

    return {
        **state,
        "final_answer": json.dumps(final, ensure_ascii=False, indent=2),
        "trace": trace,
    }