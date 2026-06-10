# LangGraph Workflow Agent

A controlled multi-node AI workflow agent built with LangGraph and Google Gemini. Each user query passes through a structured pipeline of specialized nodes: intent classification, conditional routing, answer generation, quality review, and final response packaging.

---

## How It Works

```
User Query
    |
[1] Classifier Node     ->  detects intent
    |
    |  (conditional routing)
    |
[2] Handler Node        ->  summary / qa / creative / general
    |
[3] Reviewer Node       ->  scores answer quality
    |
    |  needs_revision + retries left?  ->  back to Handler
    |  passed or retries exhausted?    ->  continue
    |
[4] Final Response Node ->  packages structured JSON output
    |
  END
```

If the reviewer marks an answer weak, the agent retries up to 2 times, passing the reviewer feedback to the handler so Gemini can improve the answer.

---

## Intents Supported

| Intent | Example Query | Handler Node |
|---|---|---|
| `summary_request` | Summarize what machine learning is | summary_node |
| `question_answer` | What is the capital of France? | qa_node |
| `creative_request` | Write a short poem about rain | creative_node |
| `general_chat` | Hello, how are you? | general_node |

---

## Project Structure

```
langgraph-workflow-agent/
├── src/
│   ├── config.py              # API keys and model settings
│   ├── state.py               # Shared AgentState TypedDict
│   ├── graph.py               # LangGraph wiring and routing logic
│   ├── main.py                # CLI interface
│   └── nodes/
│       ├── classifier.py      # Intent classification node
│       ├── handlers.py        # Four handler nodes
│       ├── reviewer.py        # Answer quality review node
│       └── final_response.py  # Output packaging node
├── tests/
│   └── test_basic.py          # 15 unit tests with mocks
├── data/
│   └── sample_inputs/
│       └── test_prompts.json  # 10 test prompts covering all intents
├── screenshots/               # App screenshots
├── app.py                     # Streamlit web interface
├── requirements.txt
├── .env.example
└── .github/
    └── workflows/
        └── ci.yml             # GitHub Actions CI
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/langgraph-workflow-agent
cd langgraph-workflow-agent
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```
GEMINI_API_KEY=your_gemini_api_key_here
MODEL_NAME=gemini-1.5-flash-8b
TEMPERATURE=0.3
MAX_RETRIES=2
```

Get a free API key from: https://aistudio.google.com/apikey

---

## Run

### Streamlit Web Interface

```bash
streamlit run app.py
```

Opens at: http://localhost:8501

### CLI Interface

```bash
python -m src.main
```

---

## Test

```bash
pytest tests/ -v
```

Expected output: **15 passed**

---

## Sample Input / Output

**Input:**
```
What is artificial intelligence?
```

**Workflow Trace:**
```
>> [classifier_node] Intent: question_answer | Route: qa_node
>> [qa_node] Answer generated (915 chars)
>> [reviewer_node] Status: passed | Feedback: Clear and complete answer
>> [final_response_node] Done
```

**Output:**
```json
{
  "intent": "question_answer",
  "route": "qa_node",
  "review": "passed",
  "final_answer": "Artificial Intelligence is a field of computer science..."
}
```

---

## Test Prompts

| # | Prompt | Intent |
|---|---|---|
| 1 | Summarize what machine learning is in simple words | summary_request |
| 2 | Explain the difference between RAM and ROM briefly | summary_request |
| 3 | What is the capital of France? | question_answer |
| 4 | Who invented the telephone and in what year? | question_answer |
| 5 | How does photosynthesis work? | question_answer |
| 6 | Write a short poem about the rain | creative_request |
| 7 | Give me 5 startup ideas for university students | creative_request |
| 8 | Brainstorm unique names for a coffee shop | creative_request |
| 9 | Hello! How are you doing today? | general_chat |
| 10 | What kinds of things can you help me with? | general_chat |

---

## CI/CD

GitHub Actions runs on every push:
- Installs dependencies
- Validates imports
- Runs all 15 tests

<img width="1920" height="1080" alt="Screenshot (494)" src="https://github.com/user-attachments/assets/6257e81d-f253-4029-beaa-d19ffcdaab71" />

<img width="1920" height="1080" alt="Screenshot (495)" src="https://github.com/user-attachments/assets/648baa52-1d71-4938-a48f-00d68f7f37dd" />


## What is Complete

- LangGraph graph with 4+ nodes: classifier, handler x4, reviewer, final response
- Conditional routing based on intent
- Retry logic when reviewer marks answer weak — up to 2 retries
- Workflow trace in CLI logs and Streamlit UI
- Streamlit web interface with dark theme, metrics, and trace viewer
- CLI interface
- 15 unit tests — all passing
- 10 test prompts covering all 4 intents
- GitHub Actions CI pipeline
- Clean modular code structure

<img width="1920" height="1080" alt="Screenshot (484)" src="https://github.com/user-attachments/assets/8d833618-be16-4182-a7b9-33a12872cc81" />

## What Can Be Improved

- Add more intent types such as translation, code generation, and math
- Add conversation memory for multi-turn chat
- Deploy to Streamlit Cloud for public access
- Add document upload and RAG-based answering
- Add confidence scoring per node
