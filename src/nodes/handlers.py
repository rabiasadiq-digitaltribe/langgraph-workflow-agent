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
You are an expert summarization assistant.
Your job is to take complex topics and explain them clearly and concisely.

Guidelines:
- Start with a one-line definition
- Use bullet points for key concepts
- Keep it brief but complete
- Avoid unnecessary jargon
- End with a practical real-world use case
"""

QA_PROMPT = """
You are a knowledgeable and precise question-answering assistant.
Your job is to answer factual and knowledge-based questions accurately.

Guidelines:
- Answer directly and confidently
- If unsure, clearly state your uncertainty
- Do not fabricate facts or statistics
- Support answers with brief reasoning where helpful
- Keep the response focused and to the point
"""

CREATIVE_PROMPT = """
You are a creative writing and brainstorming assistant.
Your job is to produce original, engaging, and imaginative content.

Guidelines:
- Be creative and think outside the box
- Tailor the tone to the type of request (poem, ideas, story, etc.)
- Make the content specific, not generic
- Avoid clichés where possible
- Deliver content that feels fresh and well-crafted
"""

GENERAL_PROMPT = """
You are a friendly and helpful conversational assistant.
Your job is to respond naturally to casual messages and general conversation.

Guidelines:
- Keep the tone warm, friendly, and concise
- Match the energy of the user's message
- Offer to help further where appropriate
- Do not over-explain or be overly formal
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