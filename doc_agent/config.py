"""Runtime settings. Everything provider-specific lives here, so switching providers is config only."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BASE_URL = "https://openrouter.ai/api"
DEFAULT_MODEL = "anthropic/claude-haiku-5.5"
KEY_VAR = "OPENROUTER_API_KEY"


@dataclass(frozen=True)
class Settings:
    base_url: str
    model: str
    api_key: str | None
    budget_file: Path
    cache_dir: Path
    timeout_s: float = 60.0
    max_attempts: int = 3
    daily_budget_usd: float = 1.0
    session_budget_usd: float = 0.10
    max_file_mb: int = 10
    max_pages: int = 20
    price_in_per_m: float = 0.10
    price_out_per_m: float = 0.50
    price_in_long_per_m: float = 0.50
    price_out_long_per_m: float = 2.50
    long_prompt_threshold: int = 100_000
    dev_mode: bool = False


def read_dotenv(path: Path) -> dict[str, str]:
    """Parse simple NAME=value lines; quotes stripped, comments and blanks ignored."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        values[name.strip()] = value.strip().strip('"').strip("'")
    return values


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def load_settings(env: Mapping[str, str], project_root: Path) -> Settings:
    """Real environment variables win; a local .env fills in whatever they leave unset."""
    dotenv = read_dotenv(project_root / ".env")

    def get(name: str) -> str | None:
        return _clean(env.get(name)) or _clean(dotenv.get(name))

    return Settings(
        base_url=get("DOC_AGENT_BASE_URL") or DEFAULT_BASE_URL,
        model=get("DOC_AGENT_MODEL") or DEFAULT_MODEL,
        api_key=get(KEY_VAR),
        budget_file=project_root / ".data" / "budget.json",
        cache_dir=project_root / "samples" / "cache",
        dev_mode=get("DEMO_DEV") == "1",
    )
