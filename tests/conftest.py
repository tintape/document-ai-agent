from pathlib import Path

import pytest

from doc_agent.config import Settings, load_settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return load_settings({"OPENROUTER_API_KEY": "test"}, tmp_path)
