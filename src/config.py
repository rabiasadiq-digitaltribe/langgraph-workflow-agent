import os
from dotenv import load_dotenv

load_dotenv()

XAI_API_KEY = os.getenv("XAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "grok-3-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))

if not XAI_API_KEY:
    raise ValueError("XAI_API_KEY is not set in .env file")