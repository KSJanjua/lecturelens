"""Prompt templates live in prompts/*.txt (versioned, shown in the report appendix).
Placeholders use {{name}} so literal JSON braces in templates need no escaping."""
import re
from functools import lru_cache

import config

_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")


@lru_cache(maxsize=None)
def load(name: str) -> str:
    return (config.PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def render(name: str, **values) -> str:
    template = load(name)

    def substitute(match):
        key = match.group(1)
        if key not in values:
            raise KeyError(f"prompt '{name}' needs a value for '{key}'")
        return str(values[key])

    return _PLACEHOLDER.sub(substitute, template).strip()


def system() -> str:
    return load("system_tutor").strip()


def all_templates() -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(config.PROMPTS_DIR.glob("*.txt"))}
