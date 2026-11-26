"""Real API calls through OpenRouter. Run explicitly: uv run pytest -m live (costs < $0.001)."""

import os
from pathlib import Path

import pytest

from doc_agent.budget import BudgetStore
from doc_agent.config import load_settings
from doc_agent.documents import load_document
from doc_agent.llm import LLMClient
from doc_agent.schemas import InvoiceData
from tests import factories

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.live
def test_live_invoice_extraction(tmp_path: Path) -> None:
    settings = load_settings(os.environ, PROJECT_ROOT)
    if not settings.api_key:
        pytest.skip("OPENROUTER_API_KEY not set")
    doc = load_document("inv.pdf", factories.invoice_pdf(), settings)
    client = LLMClient(settings, BudgetStore(tmp_path / "budget.json"))
    result = client.call(
        kind="extract",
        content=[doc.content_block(), {"type": "text", "text": "Extract the invoice fields."}],
        schema=InvoiceData,
        effort="low",
        max_tokens=4000,
        session_spent=0.0,
    )
    assert "332.5" in (result.parsed.total.value or "")
    assert result.call.cost_source == "provider"
    assert result.call.cost_usd < 0.01
