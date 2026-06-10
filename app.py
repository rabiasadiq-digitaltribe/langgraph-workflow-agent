import json
import streamlit as st
from src.graph import workflow
from src.state import AgentState

st.set_page_config(
    page_title="LangGraph Workflow Agent",
    page_icon="",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background-color: #0f1117;
        color: #e2e8f0;
    }

    .navbar {
        background: #161b27;
        border-bottom: 1px solid #1e2536;
        padding: 18px 40px;
        margin: -4rem -4rem 2rem -4rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .navbar-dot {
        width: 10px; height: 10px;
        border-radius: 50%;
        background: #4f46e5;
        display: inline-block;
    }
    .navbar-title {
        color: #e2e8f0;
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    .hero {
        text-align: center;
        padding: 48px 0 36px 0;
    }
    .hero h1 {
        font-size: 2.6rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 10px;
        letter-spacing: -0.5px;
    }
    .hero p {
        color: #64748b;
        font-size: 1rem;
        font-weight: 400;
    }

    .pipeline {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 0;
        margin: 28px 0 40px 0;
    }
    .pipeline-step {
        background: #161b27;
        border: 1px solid #1e2536;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 0.8rem;
        font-weight: 500;
        color: #94a3b8;
        white-space: nowrap;
    }
    .pipeline-step.active {
        background: #1e1b4b;
        border-color: #4f46e5;
        color: #a5b4fc;
    }
    .pipeline-arrow {
        color: #334155;
        font-size: 1rem;
        padding: 0 6px;
    }

    .stTextInput > div > div > input {
        background: #0f1117 !important;
        border: 1px solid #1e2536 !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
        font-size: 0.95rem !important;
        padding: 14px 16px !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #4f46e5 !important;
        box-shadow: 0 0 0 2px rgba(79,70,229,0.15) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #334155 !important;
    }

    .stButton > button {
        background: #4f46e5 !important;
        border: none !important;
        border-radius: 8px !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 14px 28px !important;
        letter-spacing: 0.3px !important;
        transition: background 0.2s !important;
    }
    .stButton > button:hover {
        background: #4338ca !important;
    }

    .example-label {
        color: #475569;
        font-size: 0.78rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    div[data-testid="column"] .stButton > button {
        background: #161b27 !important;
        border: 1px solid #1e2536 !important;
        color: #94a3b8 !important;
        border-radius: 6px !important;
        font-size: 0.8rem !important;
        font-weight: 400 !important;
        padding: 8px 12px !important;
    }
    div[data-testid="column"] .stButton > button:hover {
        border-color: #4f46e5 !important;
        color: #a5b4fc !important;
        background: #1e1b4b !important;
    }

    .section-header {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #475569;
        margin-bottom: 14px;
        padding-bottom: 8px;
        border-bottom: 1px solid #1e2536;
    }

    .trace-node {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 14px;
        background: #161b27;
        border-radius: 8px;
        margin-bottom: 6px;
        border-left: 3px solid #4f46e5;
    }
    .trace-node-name {
        font-size: 0.82rem;
        font-weight: 600;
        color: #a5b4fc;
        word-wrap: break-word;
        white-space: normal;
        width: 100%;
    }
    .trace-detail {
        font-size: 0.78rem;
        color: #475569;
        padding: 4px 14px 4px 27px;
        margin-bottom: 3px;
        word-wrap: break-word;
        white-space: normal;
        line-height: 1.5;
    }

    .metric-row {
        display: flex;
        gap: 12px;
        margin-bottom: 20px;
    }
    .metric-card {
        flex: 1;
        background: #161b27;
        border: 1px solid #1e2536;
        border-radius: 10px;
        padding: 16px 20px;
    }
    .metric-card-label {
        font-size: 0.7rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #475569;
        margin-bottom: 6px;
    }
    .metric-card-value {
        font-size: 1rem;
        font-weight: 600;
        color: #e2e8f0;
    }
    .metric-card-value.passed {
        color: #34d399;
    }
    .metric-card-value.failed {
        color: #f87171;
    }

    /* Answer markdown styling */
    .answer-wrapper {
        background: #161b27;
        border: 1px solid #1e2536;
        border-radius: 10px;
        padding: 22px 24px;
    }
    .answer-wrapper p {
        color: #cbd5e1 !important;
        font-size: 0.92rem !important;
        line-height: 1.9 !important;
        margin-bottom: 10px !important;
    }
    .answer-wrapper li {
        color: #cbd5e1 !important;
        font-size: 0.92rem !important;
        line-height: 1.8 !important;
        margin-bottom: 6px !important;
    }
    .answer-wrapper h1,
    .answer-wrapper h2,
    .answer-wrapper h3,
    .answer-wrapper h4 {
        color: #e2e8f0 !important;
        margin-top: 16px !important;
        margin-bottom: 8px !important;
    }
    .answer-wrapper strong {
        color: #e2e8f0 !important;
    }
    .answer-wrapper code {
        background: #0f1117 !important;
        color: #a5b4fc !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        font-size: 0.85rem !important;
    }
    .answer-wrapper ol {
        padding-left: 20px !important;
    }
    .answer-wrapper ul {
        padding-left: 20px !important;
    }

    hr { border-color: #1e2536 !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
    header { background: transparent !important; }

    /* Global markdown fix */
    .stMarkdown p {
        color: #cbd5e1;
        font-size: 0.92rem;
        line-height: 1.8;
    }
    .stMarkdown li {
        color: #cbd5e1;
        font-size: 0.92rem;
        line-height: 1.8;
    }
    .stMarkdown strong {
        color: #e2e8f0;
    }
    .stMarkdown h1, .stMarkdown h2,
    .stMarkdown h3, .stMarkdown h4 {
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ── Navbar ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <span class="navbar-dot"></span>
    <span class="navbar-title">LangGraph Workflow Agent</span>
</div>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Workflow Agent</h1>
    <p>Multi-node AI pipeline with intent classification, routing, generation, and review</p>
</div>
""", unsafe_allow_html=True)

# ── Pipeline steps ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="pipeline">
    <div class="pipeline-step active">Classifier</div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-step">Router</div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-step">Handler</div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-step">Reviewer</div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-step">Response</div>
</div>
""", unsafe_allow_html=True)

# ── Input ──────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([5, 1])
with col1:
    user_query = st.text_input(
        label="query",
        placeholder="Enter your query — summarize, question, creative writing, or general chat",
        label_visibility="collapsed",
    )
with col2:
    submit = st.button("Run", use_container_width=True)

# ── Example prompts ────────────────────────────────────────────────────────────
st.markdown('<div class="example-label">Quick examples</div>', unsafe_allow_html=True)
ex1, ex2, ex3, ex4 = st.columns(4)
with ex1:
    if st.button("Summarize machine learning", use_container_width=True):
        user_query = "Summarize what machine learning is"
        submit = True
with ex2:
    if st.button("What is artificial intelligence?", use_container_width=True):
        user_query = "What is artificial intelligence?"
        submit = True
with ex3:
    if st.button("Write a poem about rain", use_container_width=True):
        user_query = "Write a short poem about rain"
        submit = True
with ex4:
    if st.button("5 startup ideas for students", use_container_width=True):
        user_query = "Give me 5 startup ideas for students"
        submit = True

st.markdown("---")

# ── Run ────────────────────────────────────────────────────────────────────────
if submit and user_query.strip():

    with st.spinner("Running pipeline..."):
        initial_state: AgentState = {
            "user_query":      user_query.strip(),
            "intent":          None,
            "route":           None,
            "raw_answer":      None,
            "review_status":   None,
            "review_feedback": None,
            "final_answer":    None,
            "retry_count":     0,
            "trace":           [],
        }
        try:
            result = workflow.invoke(initial_state)
            error = None
        except Exception as e:
            result = None
            error = str(e)

    if error:
        st.error(f"Pipeline error: {error}")

    else:
        left, right = st.columns([1, 2], gap="large")

        # ── Trace ──────────────────────────────────────────────────────────────
        with left:
            st.markdown('<div class="section-header">Workflow Trace</div>', unsafe_allow_html=True)
            trace = result.get("trace", [])
            for step in trace:
                if step.startswith(">>"):
                    name = step.replace(">>", "").strip()
                    st.markdown(
                        f'<div class="trace-node"><div class="trace-node-name">{name}</div></div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="trace-detail">{step.strip()}</div>',
                        unsafe_allow_html=True
                    )

            retry_count = result.get("retry_count", 0)
            if retry_count > 0:
                st.warning(f"Answer revised {retry_count} time(s)")

        # ── Result ─────────────────────────────────────────────────────────────
        with right:
            st.markdown('<div class="section-header">Result</div>', unsafe_allow_html=True)

            intent  = result.get("intent", "-")
            route   = result.get("route", "-")
            review  = result.get("review_status", "-")
            r_class = "passed" if review == "passed" else "failed"
            r_label = "Passed" if review == "passed" else "Needs Revision"

            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-card">
                    <div class="metric-card-label">Intent</div>
                    <div class="metric-card-value">{intent}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-card-label">Route</div>
                    <div class="metric-card-value">{route}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-card-label">Review</div>
                    <div class="metric-card-value {r_class}">{r_label}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Answer
            st.markdown('<div class="section-header" style="margin-top:4px;">Answer</div>', unsafe_allow_html=True)

            final_raw = result.get("final_answer", "{}")
            try:
                parsed = json.loads(final_raw)
                answer_text = parsed.get("final_answer", final_raw)
            except (json.JSONDecodeError, AttributeError):
                answer_text = final_raw

            st.markdown('<div class="answer-wrapper">', unsafe_allow_html=True)
            st.markdown(answer_text)
            st.markdown('</div>', unsafe_allow_html=True)

            # Reviewer feedback
            feedback = result.get("review_feedback", "")
            if feedback:
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("Reviewer Feedback"):
                    st.markdown(
                        f'<span style="color:#94a3b8; font-size:0.88rem;">{feedback}</span>',
                        unsafe_allow_html=True
                    )

            # Raw JSON
            with st.expander("Raw JSON"):
                try:
                    st.json(json.loads(final_raw))
                except (json.JSONDecodeError, AttributeError):
                    st.code(final_raw)

elif submit and not user_query.strip():
    st.warning("Please enter a query.")