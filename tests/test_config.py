from pathlib import Path

from doc_agent.config import load_settings


def test_defaults_and_key_from_env(tmp_path: Path) -> None:
    s = load_settings({"OPENROUTER_API_KEY": "k"}, tmp_path)
    assert s.base_url == "https://openrouter.ai/api"
    assert s.model == "anthropic/claude-haiku-5.5"
    assert s.api_key == "k"
    assert s.dev_mode is False
    assert s.daily_budget_usd == 1.0
    assert s.session_budget_usd == 0.10
    assert s.budget_file == tmp_path / ".data" / "budget.json"
    assert s.cache_dir == tmp_path / "samples" / "cache"


def test_key_from_dotenv_when_env_missing(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text('# comment\nOPENROUTER_API_KEY="from-file"\n')
    assert load_settings({}, tmp_path).api_key == "from-file"


def test_env_beats_dotenv(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=from-file\n")
    assert load_settings({"OPENROUTER_API_KEY": "from-env"}, tmp_path).api_key == "from-env"


def test_missing_key_is_none(tmp_path: Path) -> None:
    assert load_settings({}, tmp_path).api_key is None


def test_blank_key_is_none(tmp_path: Path) -> None:
    assert load_settings({"OPENROUTER_API_KEY": "  "}, tmp_path).api_key is None


def test_dev_mode_flag(tmp_path: Path) -> None:
    assert load_settings({"DEMO_DEV": "1"}, tmp_path).dev_mode is True


def test_model_and_url_overrides(tmp_path: Path) -> None:
    s = load_settings(
        {"DOC_AGENT_MODEL": "claude-haiku-5-5", "DOC_AGENT_BASE_URL": "https://api.anthropic.com"},
        tmp_path,
    )
    assert s.model == "claude-haiku-5-5"
    assert s.base_url == "https://api.anthropic.com"
