"""Central configuration for LectureLens.

Every tunable lives here so experiments only change config (or .env), never code.
Modules read `config.X` at call time, so the eval runner can override values in-process.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _env(name, default, cast=str):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    if cast is bool:
        return value.strip().lower() in ("1", "true", "yes", "on")
    return cast(value)


# ---------------------------------------------------------------- paths
DATA_DIR = ROOT / "data"
LECTURES_DIR = DATA_DIR / "lectures"     # one folder per processed lecture
CHROMA_DIR = DATA_DIR / "chroma"
CACHE_DIR = ROOT / "cache"
LLM_CACHE_DIR = CACHE_DIR / "llm"        # every LLM response, keyed by input hash
PROMPTS_DIR = ROOT / "prompts"
EVAL_DIR = ROOT / "eval"
EVAL_DATA_DIR = EVAL_DIR / "data"
RESULTS_DIR = EVAL_DIR / "results"

# ---------------------------------------------------------------- API keys
GROQ_API_KEY = _env("GROQ_API_KEY", "")
GEMINI_API_KEY = _env("GEMINI_API_KEY", "") or _env("GOOGLE_API_KEY", "")

# ---------------------------------------------------------------- models
# Model names change often on free tiers: check them on Day 1 and override in .env.
LLM_PROVIDER = _env("LLM_PROVIDER", "groq")            # groq | gemini
GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_SMALL_MODEL = _env("GEMINI_SMALL_MODEL", "gemini-3.1-flash-lite")   # judge when Gemini is the only key
GROQ_MODEL = _env("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_SMALL_MODEL = _env("GROQ_SMALL_MODEL", "openai/gpt-oss-20b")
# Gemini Flash models "think" by default, which burns free-tier tokens. -1 = don't send the setting.
GEMINI_THINKING_BUDGET = _env("GEMINI_THINKING_BUDGET", 0, int)

GROQ_MAX_OUTPUT_TOKENS = 3000
# Groq counts max_tokens against tokens-per-minute up front, so small outputs get small caps.
GROQ_TASK_MAX_TOKENS = {
    "claims_judge": 1024, "notes_tot_judge": 1200, "quiz_verify": 512, "rag_judge": 256,
    "chunk_titles": 512, "interview": 512, "mindmap": 1024, "diagram_fix": 1024,
}
GROQ_REASONING_EFFORT = _env("GROQ_REASONING_EFFORT", "low")   # for gpt-oss models

# The judge must differ from the generator. "auto" = the other provider.
JUDGE_PROVIDER = _env("JUDGE_PROVIDER", "auto")
JUDGE_MODEL = _env("JUDGE_MODEL", "")

WHISPER_MODEL = _env("WHISPER_MODEL", "whisper-large-v3")
LOCAL_WHISPER_MODEL = _env("LOCAL_WHISPER_MODEL", "small")
LOCAL_WHISPER_COMPUTE = _env("LOCAL_WHISPER_COMPUTE", "int8")

EMBED_MODEL = _env("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_MODEL_MULTILINGUAL = _env(
    "EMBED_MODEL_MULTILINGUAL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# ---------------------------------------------------------------- audio
SAMPLE_RATE = 16000
PIECE_SECONDS = 600                       # split long audio into ~10-minute pieces
API_MAX_BYTES = 24 * 1024 * 1024          # Groq free tier caps uploads at 25 MB

# ---------------------------------------------------------------- chunking
WINDOW_SENTENCES = 5
SIM_THRESHOLD = _env("SIM_THRESHOLD", 0.6, float)   # tune between 0.5 and 0.7
MIN_CHUNK_TOKENS = 300
MAX_CHUNK_TOKENS = 1500
FIXED_CHUNK_TOKENS = 1000                 # baseline for Experiment 4

# ---------------------------------------------------------------- generation
DEFAULT_STRATEGY = _env("DEFAULT_STRATEGY", "tot")   # zero_shot | few_shot | cot | tot
NOTES_TEMPERATURE = 0.25                  # factual output needs little randomness
TOP_P = 0.95
TOT_TEMPERATURE = 0.8                     # diversity for candidate generation
TOT_BREADTH = _env("TOT_BREADTH", 3, int)
TOT_KEEP = 2
TOT_DEPTH = _env("TOT_DEPTH", 2, int)
JUDGE_TEMPERATURE = 0.0
QUIZ_TEMPERATURE = 0.4
CREATIVE_TEMPERATURE = 0.6                # audio recap script
MAX_OUTPUT_TOKENS = 4096
BULLET_DEDUP_SIMILARITY = 0.9

# ---------------------------------------------------------------- quiz / flashcards
QUIZ_DEFAULT_COUNT = 8
QUIZ_BATCH_TOKENS = 3000                  # transcript tokens per quiz-generation call
FLASHCARD_MAX = 30

# ---------------------------------------------------------------- RAG
RAG_TOP_K = 4

# ---------------------------------------------------------------- diagrams
DIAGRAM_MAX = 4
DIAGRAM_MAX_RETRIES = 2
MERMAID_VALIDATE_REMOTE = _env("MERMAID_VALIDATE_REMOTE", True, bool)
MERMAID_INK_URL = "https://mermaid.ink"

# ---------------------------------------------------------------- audio recap (edge-tts)
TTS_VOICES = {
    "en": ("en-IN-NeerjaNeural", "en-IN-PrabhatNeural"),   # (teacher, student)
    "hi": ("hi-IN-SwaraNeural", "hi-IN-MadhurNeural"),
}

# ---------------------------------------------------------------- LLM reliability
LLM_CACHE_ENABLED = _env("LLM_CACHE_ENABLED", True, bool)
CACHE_SALT = ""                          # set per repeat to draw fresh samples instead of cached ones
LLM_MAX_RETRIES = 6
RATE_LIMIT_MAX_WAIT = 600              # seconds to keep waiting out per-minute limits
LLM_BACKOFF_BASE = 2.0                    # seconds; doubles on each retry
LLM_FALLBACK = True                       # switch provider when one keeps failing
MIN_SECONDS_BETWEEN_CALLS = _env("MIN_SECONDS_BETWEEN_CALLS", 0.0, float)
MAX_WORKERS = _env("MAX_WORKERS", 4, int)  # parallel LLM calls per stage
