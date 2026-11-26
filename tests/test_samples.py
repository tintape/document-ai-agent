from pathlib import Path

import pytest

from doc_agent.cache import SAMPLES
from doc_agent.config import Settings
from doc_agent.documents import load_document

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
NO_TEXT_LAYER = {"invoice_messy_scan.pdf", "invoice_messy_scan.png"}


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_sample_loads(name: str, settings: Settings) -> None:
    doc = load_document(name, (SAMPLES_DIR / name).read_bytes(), settings)
    assert doc.has_text_layer is (name not in NO_TEXT_LAYER)


def test_multipage_and_contract_have_two_pages(settings: Settings) -> None:
    for name in ("invoice_multipage.pdf", "contract_services.pdf"):
        assert load_document(name, (SAMPLES_DIR / name).read_bytes(), settings).page_count == 2


def test_contract_termination_clause_on_page_two(settings: Settings) -> None:
    doc = load_document(
        "contract_services.pdf", (SAMPLES_DIR / "contract_services.pdf").read_bytes(), settings
    )
    assert "termination" in doc.page_texts[1].lower()
    assert "termination" not in doc.page_texts[0].lower()
