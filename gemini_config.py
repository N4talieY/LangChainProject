"""Load Gemini credentials from the project-root .env file."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"

def configure_gemini() -> str | None:
    """Load GOOGLE_API_KEY or GEMINI_API_KEY without overriding shell settings."""
    if ENV_FILE.is_file():
        for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            name, value = name.strip(), value.strip()
            if name not in {"GOOGLE_API_KEY", "GEMINI_API_KEY", "HUGGINGFACEHUB_API_TOKEN"}:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
                value = value[1:-1]
            if value:
                os.environ.setdefault(name, value)

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        os.environ.setdefault("GOOGLE_API_KEY", api_key)
        os.environ.setdefault("GEMINI_API_KEY", api_key)
    return api_key
