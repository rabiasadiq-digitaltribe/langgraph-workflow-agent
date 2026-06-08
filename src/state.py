from typing import TypedDict, Optional, List


class AgentState(TypedDict):
    """
    Shared state passed between all nodes in the LangGraph workflow.
    """
    user_query: str                        # Original user input
    intent: Optional[str]                  # Classified intent
    route: Optional[str]                   # Which handler node to use
    raw_answer: Optional[str]              # Answer from handler node
    review_status: Optional[str]           # "passed" or "needs_revision"
    review_feedback: Optional[str]         # Reviewer's feedback if weak
    final_answer: Optional[str]            # Polished final answer
    retry_count: int                       # How many revision attempts so far
    trace: List[str]                       # Step-by-step workflow trace log