import csv
import io
import json

import pytest

from doc_agent.config import Settings
from doc_agent.documents import LoadedDocument, load_document
from doc_agent.extract import extract, to_csv, to_json, to_rows
from doc_agent.prompts import EXTRACT_PROMPT
from doc_agent.schemas import InvoiceData
from tests import factories
from tests.stubs import StubLLM
from tests.test_validate import fv, make_contract, make_invoice


@pytest.fixture
def text_doc(settings: Settings) -> LoadedDocument:
    return load_document("inv.pdf", factories.invoice_pdf(), settings)


@pytest.fixture
def mismatched() -> InvoiceData:
    return make_invoice(total=fv("400.00"))


def test_extract_runs_validation(text_doc: LoadedDocument, mismatched: InvoiceData) -> None:
    llm = StubLLM(mismatched)
    result = extract(text_doc, "invoice", llm, session_spent=0.0)
    assert result.doc_type == "invoice"
    assert result.doc_name == "inv.pdf"
    assert result.cached is False
    assert result.call is not None and result.call.kind == "extract"
    assert any(f.field == "total" for f in result.flags)


def test_extract_sends_document_and_prompt(text_doc: LoadedDocument, mismatched: InvoiceData) -> None:
    llm = StubLLM(mismatched)
    extract(text_doc, "invoice", llm, session_spent=0.03, simulate_failures=1)
    assert llm.last["schema"] is InvoiceData
    assert llm.last["effort"] == "low"
    assert llm.last["max_tokens"] == 4000
    assert llm.last["session_spent"] == 0.03
    assert llm.last["simulate_failures"] == 1
    content = llm.last["content"]
    assert content[0]["type"] == "document"
    assert content[1] == {"type": "text", "text": EXTRACT_PROMPT["invoice"]}


def test_unknown_doc_type_rejected(text_doc: LoadedDocument) -> None:
    llm = StubLLM(make_invoice())
    with pytest.raises(ValueError):
        extract(text_doc, "receipt", llm, session_spent=0.0)
    assert llm.calls == 0


def test_rows_include_issue_text(text_doc: LoadedDocument, mismatched: InvoiceData) -> None:
    rows = to_rows(extract(text_doc, "invoice", StubLLM(mismatched), session_spent=0.0))
    total = next(r for r in rows if r["field"] == "total")
    assert "do not add up" in total["issue"]
    assert total["value"] == "400.00"
    assert total["confidence"] == "high"
    assert total["page"] == "1"
    assert any(r["field"] == "line_items[2].amount" for r in rows)
    vendor = next(r for r in rows if r["field"] == "vendor")
    assert vendor["issue"] == ""


def test_rows_show_group_level_issues(text_doc: LoadedDocument) -> None:
    contract = make_contract(parties=[fv("Acme")])
    result = extract(text_doc, "contract", StubLLM(contract), session_spent=0.0)
    rows = to_rows(result)
    assert any(r["field"] == "parties" and r["issue"] for r in rows)
    assert any(r["field"] == "parties[1]" and r["value"] == "Acme" for r in rows)


def test_empty_values_render_blank(text_doc: LoadedDocument) -> None:
    invoice = make_invoice(buyer=fv(None, page=None))
    buyer = next(r for r in to_rows(extract(text_doc, "invoice", StubLLM(invoice), 0.0)) if r["field"] == "buyer")
    assert buyer["value"] == "" and buyer["page"] == ""


def test_csv_has_header_and_rows(text_doc: LoadedDocument, mismatched: InvoiceData) -> None:
    result = extract(text_doc, "invoice", StubLLM(mismatched), session_spent=0.0)
    text = to_csv(result)
    assert text.splitlines()[0] == "field,value,confidence,page,issue"
    rows = list(csv.DictReader(io.StringIO(text)))
    assert len(rows) == len(to_rows(result))


def test_json_export(text_doc: LoadedDocument, mismatched: InvoiceData) -> None:
    result = extract(text_doc, "invoice", StubLLM(mismatched), session_spent=0.0)
    data = json.loads(to_json(result))
    assert data["document"] == "inv.pdf"
    assert data["type"] == "invoice"
    assert data["fields"]["total"]["value"] == "400.00"
    assert any(issue["field"] == "total" for issue in data["issues"])
