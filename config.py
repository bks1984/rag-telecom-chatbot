"""Shared settings for ingestion, retrieval and generation."""
from pathlib import Path
import os

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
FAQ_CSV = DATA_DIR / "faq.csv"
TICKETS_DB = DATA_DIR / "tickets.db"
GUIDE_PDF = DATA_DIR / "telecom_guide.pdf"

CHROMA_DIR = ROOT / "chroma_store"
LOG_DIR = ROOT / "logs"
INTERACTION_LOG = LOG_DIR / "interactions.jsonl"

# Collections (NFR-06: register new ones here + in retriever.py)
COLLECTIONS = ("faq", "tickets", "guides")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # FR-09
CHUNK_SIZE = 600                                             # FR-16
CHUNK_OVERLAP = 100
TOP_K = 3                                                    # FR-07

# FR-13 names qwen/qwen3.6-27b; Groq no longer serves it, so default to its successor.
LLM_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
LLM_TEMPERATURE = 0                                          # FR-12
LLM_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "none")

# Guardrail: if the best relevance score across all collections is below this,
# the bot declines and escalates instead of calling the LLM.
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.30"))

# Wording taken from FAQ #25 - keep in sync with the FAQ.
ESCALATION_CONTACT = "call 611 (free from your mobile) or use live chat in the MyTelecom app (8am-10pm)"

# ---------- Agentic refund capability (agent_PRD.md) ----------
# FR-15/16: stands in for the user ID an authenticated session would supply.
DEMO_USER_ID = os.getenv("DEMO_USER_ID", "user_123")
REFUND_WINDOW_DAYS = 7          # FR-25
REFUND_APPROVAL_LIMIT = 499     # FR-36/37: above this, refunds go to the Support Team
AGENT_MAX_STEPS = 4             # cap on LLM tool-calling rounds per refund turn
HISTORY_MESSAGES = 6            # FR-50: recent messages (3 exchanges) passed as context
AGENT_LOG = LOG_DIR / "agent.log"


def setup_logging() -> None:
    """Workflow-step log (agent_PRD §12) to the console and logs/agent.log. Safe to call twice."""
    import logging
    logger = logging.getLogger("novacell")
    if logger.handlers:
        return
    LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(AGENT_LOG, encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
