import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import GEMINI_API_KEY, MODEL_NAME, TEMPERATURE
from src.state import AgentState

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    google_api_key=GEMINI_API_KEY,
)

# ── System prompts per intent ──────────────────────────────────────────────────

SUMMARY_PROMPT = """
You are a summarization expert. 
Give a clear, concise summary or explanation of what the user asks.
Be brief but complete. Use bullet points if helpful.
"""

QA_PROMPT = """
You are a knowledgeable assistant.
Answer the user's question accurately and directly.
If you are unsure, say so honestly. Do not make up facts.
"""

CREATIVE_PROMPT = """
You are a creative writing and brainstorming assistant.
Be imaginative, original, and engaging.
Provide ideas or creative content that directly addresses the user's request.
"""

GENERAL_PROMPT = """
You are a friendly conversational assistant.
Respond naturally and helpfully to the user's message.
Keep the tone warm and concise.
"""

PROMPT_MAP = {
    "summary_node":  SUMMARY_PROMPT,
    "qa_node":       QA_PROMPT,
    "creative_node": CREATIVE_PROMPT,
    "general_node":  GENERAL_PROMPT,
}



def _handle(state: AgentState, node_name: str) -> AgentState:
    trace = state.get("trace", [])
    retry = state.get("retry_count", 0)
    feedback = state.get("review_feedback", "")

    trace.append(f">> [{node_name}] Generating answer (attempt {retry + 1})...")

    system_prompt = PROMPT_MAP.get(node_name, GENERAL_PROMPT)

    user_content = state["user_query"]
    if retry > 0 and feedback:
        user_content = (
            f"{state['user_query']}\n\n"
            f"[Previous answer was marked weak. Reviewer feedback: {feedback}. "
            f"Please provide a better answer.]"
        )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]

    # Retry on server errors like 503
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            response = llm.invoke(messages)
            raw_answer = response.content.strip()
            break
        except Exception as e:
            if attempt < max_attempts - 1:
                wait = 10 * (attempt + 1)
                trace.append(f"   [{node_name}] Server error, retrying in {wait}s... ({e})")
                time.sleep(wait)
            else:
                raw_answer = "Sorry, the AI service is currently unavailable. Please try again."
                trace.append(f"   [{node_name}] All retries failed: {e}")

    trace.append(f"   [{node_name}] Answer generated ({len(raw_answer)} chars)")

    return {
        **state,
        "raw_answer": raw_answer,
        "trace": trace,
    }
# ── Four named handler nodes ───────────────────────────────────────────────────

def summary_node(state: AgentState) -> AgentState:
    return _handle(state, "summary_node")


def qa_node(state: AgentState) -> AgentState:
    return _handle(state, "qa_node")


def creative_node(state: AgentState) -> AgentState:
    return _handle(state, "creative_node")


def general_node(state: AgentState) -> AgentState:
    return _handle(state, "general_node")