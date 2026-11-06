"""One client for every LLM call: Gemini or Groq, with disk cache, retries,
provider fallback, usage accounting and Pydantic-validated JSON output."""
import logging
import random
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass

from pydantic import BaseModel, ValidationError

import config
from pipeline.utils import extract_json, read_json, sha, write_json

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


# ------------------------------------------------------------------ usage accounting
@dataclass
class Usage:
    calls: int = 0              # logical calls, cached or not
    api_calls: int = 0          # calls that actually hit an API
    prompt_tokens: int = 0      # includes tokens of cached calls, so cost stays comparable
    completion_tokens: int = 0
    seconds: float = 0.0

    def snapshot(self) -> dict:
        data = asdict(self)
        data["total_tokens"] = self.prompt_tokens + self.completion_tokens
        return data


USAGE = Usage()
_lock = threading.Lock()


@contextmanager
def track_usage():
    """with track_usage() as cost: ...  -> cost holds the usage delta of the block."""
    start = USAGE.snapshot()
    delta: dict = {}
    try:
        yield delta
    finally:
        end = USAGE.snapshot()
        delta.update({k: round(end[k] - start[k], 2) for k in end})


# ------------------------------------------------------------------ providers
_clients: dict = {}
_last_call: dict = {}
_no_thinking_config: set = set()   # Gemini models that reject a thinking budget


def _gemini_client():
    if "gemini" not in _clients:
        from google import genai
        from google.genai import types
        # One attempt per call: retries (and daily-quota fail-fast) are handled by _call_with_retries.
        _clients["gemini"] = genai.Client(api_key=config.GEMINI_API_KEY, http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1), timeout=180_000))
    return _clients["gemini"]


def _groq_client():
    if "groq" not in _clients:
        from groq import Groq
        _clients["groq"] = Groq(api_key=config.GROQ_API_KEY, max_retries=0)
    return _clients["groq"]


def _call_gemini(model, system, prompt, temperature, top_p, json_mode, max_tokens, task):
    from google.genai import types
    kwargs = dict(temperature=temperature, top_p=top_p, max_output_tokens=max_tokens)
    if system:
        kwargs["system_instruction"] = system
    if json_mode:
        kwargs["response_mime_type"] = "application/json"
    if config.GEMINI_THINKING_BUDGET >= 0 and model not in _no_thinking_config:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=config.GEMINI_THINKING_BUDGET)
    try:
        resp = _gemini_client().models.generate_content(
            model=model, contents=prompt, config=types.GenerateContentConfig(**kwargs))
    except Exception as exc:
        if "thinking_config" not in kwargs or "INVALID_ARGUMENT" not in str(exc):
            raise
        _no_thinking_config.add(model)
        kwargs.pop("thinking_config")
        resp = _gemini_client().models.generate_content(
            model=model, contents=prompt, config=types.GenerateContentConfig(**kwargs))
    usage = resp.usage_metadata
    return (
        resp.text or "",
        (usage.prompt_token_count or 0) if usage else 0,
        (usage.candidates_token_count or 0) if usage else 0,
    )


def _call_groq(model, system, prompt, temperature, top_p, json_mode, max_tokens, task):
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    kwargs = dict(model=model, messages=messages, temperature=temperature, top_p=top_p,
                  max_tokens=min(max_tokens, config.GROQ_TASK_MAX_TOKENS.get(task, config.GROQ_MAX_OUTPUT_TOKENS)))
    if model.startswith("openai/gpt-oss"):
        kwargs["reasoning_effort"] = config.GROQ_REASONING_EFFORT   # reasoning tokens count against TPM
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = _groq_client().chat.completions.create(**kwargs)
    usage = resp.usage
    return (
        resp.choices[0].message.content or "",
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
    )


# name -> (call function, "is a key configured?" check)
_PROVIDERS = {
    "gemini": (_call_gemini, lambda: bool(config.GEMINI_API_KEY)),
    "groq": (_call_groq, lambda: bool(config.GROQ_API_KEY)),
}


def register_provider(name, fn):
    """Add a provider (used by the tests to plug in a fake LLM)."""
    _PROVIDERS[name] = (fn, lambda: True)


def available_providers() -> list[str]:
    return [name for name, (_, has_key) in _PROVIDERS.items() if has_key()]


def default_model(provider: str, small: bool = False) -> str:
    if provider == "gemini":
        return config.GEMINI_SMALL_MODEL if small else config.GEMINI_MODEL
    if provider == "groq":
        return config.GROQ_SMALL_MODEL if small else config.GROQ_MODEL
    return provider


def resolve(role: str = "generator", provider: str | None = None, model: str | None = None):
    """Pick (provider, model) for a role. The judge uses a different model from the generator."""
    available = available_providers()
    if not available:
        raise LLMError("No LLM API key found. Add GEMINI_API_KEY and/or GROQ_API_KEY to .env")
    generator = config.LLM_PROVIDER if config.LLM_PROVIDER in available else available[0]
    if role == "judge" and provider is None:
        wanted = config.JUDGE_PROVIDER
        if wanted == "auto":
            others = [p for p in available if p != generator]
            wanted = others[0] if others else generator
        provider = wanted if wanted in available else generator
        if model is None and config.JUDGE_MODEL:
            model = config.JUDGE_MODEL
        if model is None:
            # Only one key available: judge with the smaller sibling model.
            model = default_model(provider, small=(provider == generator))
    provider = provider if provider in available else generator
    return provider, model or default_model(provider)


def _retryable(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    if any(s in text for s in ("per day", "tokens per day", "requests per day", "perday")):
        return False            # daily quota: waiting seconds won't help, fall back instead
    return any(s in text for s in (
        "429", "rate", "quota", "exhausted", "overloaded", "unavailable", "500", "502", "503",
        "504", "timeout", "timed out", "connection", "internal",
    ))


def _suggested_delay(exc: Exception) -> float | None:
    """Use the provider's own hint: Groq 'try again in 7.5s' / '1m2.5s', Gemini 'retryDelay': '33s'."""
    text = str(exc)
    match = re.search(r"try again in (?:(\d+)m)?([\d.]+)(ms|s)", text)
    if match:
        minutes, value, unit = match.groups()
        return int(minutes or 0) * 60 + float(value) / (1000 if unit == "ms" else 1)
    match = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?([\d.]+)s", text)
    return float(match.group(1)) if match else None


def _pace(provider):
    gap = config.MIN_SECONDS_BETWEEN_CALLS
    if gap <= 0:
        return
    with _lock:
        wait = _last_call.get(provider, 0) + gap - time.time()
        _last_call[provider] = max(time.time(), _last_call.get(provider, 0) + gap)
    if wait > 0:
        time.sleep(wait)


def _per_minute_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return "per minute" in text or "(tpm)" in text or "(rpm)" in text


def _call_with_retries(provider, model, *args):
    """Retry transient errors a fixed number of times. Per-minute rate limits always clear,
    so they get a time budget (RATE_LIMIT_MAX_WAIT) instead of using up the retry count."""
    fn = _PROVIDERS[provider][0]
    last_exc, attempt, started = None, 0, time.time()
    while True:
        _pace(provider)
        try:
            return fn(model, *args)
        except Exception as exc:  # SDKs raise many types; classify by message
            last_exc = exc
            if not _retryable(exc):
                break
            per_minute = _per_minute_limit(exc)
            if per_minute and time.time() - started > config.RATE_LIMIT_MAX_WAIT:
                break
            if not per_minute:
                if attempt >= config.LLM_MAX_RETRIES:
                    break
                attempt += 1
            hint = _suggested_delay(exc)
            delay = (hint + 0.5 if hint is not None else config.LLM_BACKOFF_BASE * (2 ** attempt)) + random.uniform(0, 1)
            log.warning("%s call failed (%s); retrying in %.1fs", provider, str(exc)[:160], delay)
            time.sleep(delay)
    raise LLMError(f"{provider}/{model} failed: {last_exc}") from last_exc


# ------------------------------------------------------------------ public API
def generate(
    prompt: str,
    *,
    system: str | None = None,
    role: str = "generator",
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    json_mode: bool = False,
    max_tokens: int | None = None,
    task: str = "",
    cache: bool = True,
) -> str:
    """Generate text. Identical inputs are served from the disk cache."""
    provider, model = resolve(role, provider, model)
    temperature = config.NOTES_TEMPERATURE if temperature is None else temperature
    top_p = config.TOP_P if top_p is None else top_p
    max_tokens = max_tokens or config.MAX_OUTPUT_TOKENS

    key = sha(provider, model, system or "", prompt, temperature, top_p, json_mode, max_tokens, length=32)
    path = config.LLM_CACHE_DIR / key[:2] / f"{key}.json"
    use_cache = cache and config.LLM_CACHE_ENABLED
    if use_cache:
        hit = read_json(path)
        if hit is not None:
            with _lock:
                USAGE.calls += 1
                USAGE.prompt_tokens += hit.get("prompt_tokens", 0)
                USAGE.completion_tokens += hit.get("completion_tokens", 0)
            return hit["text"]

    args = (system, prompt, temperature, top_p, json_mode, max_tokens, task)
    started = time.time()
    try:
        text, p_tok, c_tok = _call_with_retries(provider, model, *args)
    except LLMError:
        others = [p for p in available_providers() if p != provider]
        if not (config.LLM_FALLBACK and others):
            raise
        # A judge falls back to the other provider's smaller model, so it never becomes the generator.
        provider, model = others[0], default_model(others[0], small=(role == "judge"))
        log.warning("falling back to %s/%s", provider, model)
        text, p_tok, c_tok = _call_with_retries(provider, model, *args)
    elapsed = time.time() - started

    with _lock:
        USAGE.calls += 1
        USAGE.api_calls += 1
        USAGE.prompt_tokens += p_tok
        USAGE.completion_tokens += c_tok
        USAGE.seconds += elapsed
    if use_cache:
        write_json(path, {"text": text, "provider": provider, "model": model, "task": task,
                          "prompt_tokens": p_tok, "completion_tokens": c_tok})
    return text


def generate_json(prompt: str, schema: type[BaseModel], **kwargs) -> BaseModel:
    """Generate JSON and validate it against `schema`; retry once with the error message."""
    kwargs.setdefault("json_mode", True)
    text = generate(prompt, **kwargs)
    try:
        return _validate(text, schema)
    except (ValueError, ValidationError) as exc:
        error = str(exc)[:800]
    retry_prompt = (
        f"{prompt}\n\nYour previous reply was not valid for the required JSON schema.\n"
        f"Error: {error}\nPrevious reply (truncated): {text[:1500]}\n"
        "Reply again with ONLY corrected JSON."
    )
    text = generate(retry_prompt, **kwargs)
    try:
        return _validate(text, schema)
    except (ValueError, ValidationError) as exc:
        raise LLMError(f"model output failed validation for {schema.__name__}: {exc}") from exc


def _validate(text: str, schema: type[BaseModel]) -> BaseModel:
    data = extract_json(text)
    fields = list(schema.model_fields)
    if isinstance(data, list) and len(fields) >= 1:
        # The model returned the bare list instead of {"field": [...]}
        data = {fields[0]: data}
    return schema.model_validate(data)
