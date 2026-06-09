import json
import pytest
from unittest.mock import patch, MagicMock
from src.state import AgentState
from src.nodes.classifier import classifier_node
from src.nodes.handlers import summary_node, qa_node, creative_node, general_node
from src.nodes.reviewer import reviewer_node
from src.nodes.final_response import final_response_node


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_state(overrides: dict = {}) -> AgentState:
    base: AgentState = {
        "user_query":      "test query",
        "intent":          None,
        "route":           None,
        "raw_answer":      None,
        "review_status":   None,
        "review_feedback": None,
        "final_answer":    None,
        "retry_count":     0,
        "trace":           [],
    }
    return {**base, **overrides}


def mock_llm_response(content: str) -> MagicMock:
    mock = MagicMock()
    mock.content = content
    return mock


# ── Classifier tests ───────────────────────────────────────────────────────────

@patch("src.nodes.classifier.llm")
def test_classifier_summary_intent(mock_llm):
    mock_llm.invoke.return_value = mock_llm_response(
        '{"intent": "summary_request", "route": "summary_node"}'
    )
    state = make_state({"user_query": "Summarize machine learning"})
    result = classifier_node(state)

    assert result["intent"] == "summary_request"
    assert result["route"] == "summary_node"
    assert len(result["trace"]) > 0


@patch("src.nodes.classifier.llm")
def test_classifier_qa_intent(mock_llm):
    mock_llm.invoke.return_value = mock_llm_response(
        '{"intent": "question_answer", "route": "qa_node"}'
    )
    state = make_state({"user_query": "What is the capital of France?"})
    result = classifier_node(state)

    assert result["intent"] == "question_answer"
    assert result["route"] == "qa_node"


@patch("src.nodes.classifier.llm")
def test_classifier_creative_intent(mock_llm):
    mock_llm.invoke.return_value = mock_llm_response(
        '{"intent": "creative_request", "route": "creative_node"}'
    )
    state = make_state({"user_query": "Write a poem about rain"})
    result = classifier_node(state)

    assert result["intent"] == "creative_request"
    assert result["route"] == "creative_node"


@patch("src.nodes.classifier.llm")
def test_classifier_invalid_json_defaults_to_general(mock_llm):
    mock_llm.invoke.return_value = mock_llm_response("not valid json at all")
    state = make_state({"user_query": "hello"})
    result = classifier_node(state)

    assert result["intent"] == "general_chat"
    assert result["route"] == "general_node"


# ── Handler tests ──────────────────────────────────────────────────────────────

@patch("src.nodes.handlers.llm")
def test_summary_node_returns_answer(mock_llm):
    mock_llm.invoke.return_value = mock_llm_response("Machine learning is a subset of AI.")
    state = make_state({"user_query": "Summarize ML", "route": "summary_node"})
    result = summary_node(state)

    assert result["raw_answer"] == "Machine learning is a subset of AI."
    assert "summary_node" in result["trace"][-1]


@patch("src.nodes.handlers.llm")
def test_qa_node_returns_answer(mock_llm):
    mock_llm.invoke.return_value = mock_llm_response("Paris is the capital of France.")
    state = make_state({"user_query": "Capital of France?", "route": "qa_node"})
    result = qa_node(state)

    assert result["raw_answer"] == "Paris is the capital of France."


@patch("src.nodes.handlers.llm")
def test_handler_retry_includes_feedback(mock_llm):
    mock_llm.invoke.return_value = mock_llm_response("Improved answer here.")
    state = make_state({
        "user_query":      "Explain AI",
        "route":           "summary_node",
        "retry_count":     1,
        "review_feedback": "Answer was too vague.",
    })
    result = summary_node(state)

    # Confirm the LLM was called with feedback in the message
    call_args = mock_llm.invoke.call_args[0][0]
    human_message_content = call_args[1].content
    assert "too vague" in human_message_content
    assert result["raw_answer"] == "Improved answer here."


# ── Reviewer tests ─────────────────────────────────────────────────────────────

@patch("src.nodes.reviewer.llm")
def test_reviewer_passes_good_answer(mock_llm):
    mock_llm.invoke.return_value = mock_llm_response(
        '{"status": "passed", "feedback": "Answer is clear and complete."}'
    )
    state = make_state({
        "user_query": "What is AI?",
        "raw_answer": "AI is the simulation of human intelligence by machines.",
    })
    result = reviewer_node(state)

    assert result["review_status"] == "passed"
    assert "clear" in result["review_feedback"]


@patch("src.nodes.reviewer.llm")
def test_reviewer_flags_weak_answer(mock_llm):
    mock_llm.invoke.return_value = mock_llm_response(
        '{"status": "needs_revision", "feedback": "Answer is too short and vague."}'
    )
    state = make_state({
        "user_query": "Explain quantum computing",
        "raw_answer": "It is fast.",
    })
    result = reviewer_node(state)

    assert result["review_status"] == "needs_revision"
    assert result["review_feedback"] != ""


@patch("src.nodes.reviewer.llm")
def test_reviewer_invalid_json_defaults_to_passed(mock_llm):
    mock_llm.invoke.return_value = mock_llm_response("looks good to me")
    state = make_state({
        "user_query": "Hello",
        "raw_answer": "Hi there!",
    })
    result = reviewer_node(state)

    assert result["review_status"] == "passed"


# ── Final response tests ───────────────────────────────────────────────────────

def test_final_response_structure():
    state = make_state({
        "user_query":    "What is AI?",
        "intent":        "question_answer",
        "route":         "qa_node",
        "raw_answer":    "AI is the simulation of human intelligence.",
        "review_status": "passed",
    })
    result = final_response_node(state)

    output = json.loads(result["final_answer"])
    assert output["intent"] == "question_answer"
    assert output["route"] == "qa_node"
    assert output["review"] == "passed"
    assert "AI" in output["final_answer"]


def test_final_response_trace_updated():
    state = make_state({
        "intent":        "general_chat",
        "route":         "general_node",
        "raw_answer":    "Hello!",
        "review_status": "passed",
    })
    result = final_response_node(state)

    assert any("final_response_node" in t for t in result["trace"])


# ── State tests ────────────────────────────────────────────────────────────────

def test_agent_state_has_required_fields():
    state = make_state()
    required = [
        "user_query", "intent", "route", "raw_answer",
        "review_status", "review_feedback", "final_answer",
        "retry_count", "trace"
    ]
    for field in required:
        assert field in state, f"Missing field: {field}"


def test_retry_count_starts_at_zero():
    state = make_state()
    assert state["retry_count"] == 0


def test_trace_is_list():
    state = make_state()
    assert isinstance(state["trace"], list)