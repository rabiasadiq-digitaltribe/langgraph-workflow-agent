import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import XAI_API_KEY, MODEL_NAME, TEMPERATURE
from src.state import AgentState

llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)

CLASSIFIER_SYSTEM_PROMPT = """
You are an intent classifier. Given a user query, classify it into exactly one of these intents:

- summary_request     → user wants something summarized or explained briefly
- question_answer     → user is asking a factual or knowledge-based question
- creative_request    → user wants creative writing, ideas, or brainstorming
- general_chat        → casual conversation, greetings, or anything else

Respond with ONLY a valid JSON object. No explanation. No markdown. Example:
{"intent": "question_answer", "route": "qa_node"}

Intent to route mapping:
- summary_request  → summary_node
- question_answer  → qa_node
- creative_request → creative_node
- general_chat     → general_node
"""


def classifier_node(state: AgentState) -> AgentState:
    """
    Node 1: Classifies user query intent and sets the route.
    """
    trace = state.get("trace", [])
    trace.append(">> [classifier_node] Classifying intent...")

    messages = [
        SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
        HumanMessage(content=state["user_query"]),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    try:
        parsed = json.loads(raw)
        intent = parsed.get("intent", "general_chat")
        route = parsed.get("route", "general_node")
    except json.JSONDecodeError:
        intent = "general_chat"
        route = "general_node"
        trace.append("   [classifier_node] WARNING: Could not parse JSON, defaulting to general_chat")

    trace.append(f"   [classifier_node] Intent: {intent} | Route: {route}")

    return {
        **state,
        "intent": intent,
        "route": route,
        "trace": trace,
    }