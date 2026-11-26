import pytest

from doc_agent.ask import ask, check_quote, normalize
from doc_agent.config import Settings
from doc_agent.documents import LoadedDocument, load_document
from doc_agent.prompts import ASK_PROMPT
from doc_agent.schemas import AnswerData, Quote
from tests import factories
from tests.stubs import StubLLM


@pytest.fixture
def contract(settings: Settings) -> LoadedDocument:
    return load_document("contract.pdf", factories.contract_pdf(), settings)


@pytest.fixture
def invoice(settings: Settings) -> LoadedDocument:
    return load_document("inv.pdf", factories.invoice_pdf(), settings)


@pytest.fixture
def scan(settings: Settings) -> LoadedDocument:
    return load_document("scan.png", factories.png_bytes(), settings)


def test_normalize_unifies_quotes_dashes_case_and_space() -> None:
    assert normalize("  “Thirty” DAYS — net\n") == normalize('"thirty" days - net')


def test_quote_found_on_stated_page(contract: LoadedDocument) -> None:
    q = check_quote(Quote(document=1, page=2, text="Termination  notice: 30 days"), [contract])
    assert q.status == "verified"
    assert q.page == 2
    assert q.doc_name == "contract.pdf"
    assert q.note == ""


def test_quote_on_other_page_is_verified_with_note(contract: LoadedDocument) -> None:
    q = check_quote(Quote(document=1, page=1, text="termination notice: 30 days"), [contract])
    assert q.status == "verified"
    assert q.page == 2
    assert "page 2" in q.note


def test_invented_quote_not_found(contract: LoadedDocument) -> None:
    q = check_quote(Quote(document=1, page=1, text="penalty of $5,000"), [contract])
    assert q.status == "not_found"


@pytest.mark.parametrize("document", [2, 0, -1])
def test_out_of_range_document_not_found(contract: LoadedDocument, document: int) -> None:
    q = check_quote(Quote(document=document, page=1, text="Termination notice"), [contract])
    assert q.status == "not_found"


@pytest.mark.parametrize("page", [9, 0, -3])
def test_out_of_range_page_with_real_quote_is_corrected(
    contract: LoadedDocument, page: int
) -> None:
    q = check_quote(Quote(document=1, page=page, text="Termination notice"), [contract])
    assert q.status == "verified"
    assert q.page == 2
    assert "page 2" in q.note


@pytest.mark.parametrize("page", [9, 0])
def test_out_of_range_page_with_invented_quote_not_found(
    contract: LoadedDocument, page: int
) -> None:
    q = check_quote(Quote(document=1, page=page, text="penalty of $5,000"), [contract])
    assert q.status == "not_found"


def test_empty_quote_not_verified(contract: LoadedDocument) -> None:
    assert check_quote(Quote(document=1, page=1, text="  "), [contract]).status == "unverified"


def test_scan_is_unverified(scan: LoadedDocument) -> None:
    q = check_quote(Quote(document=1, page=1, text="Total"), [scan])
    assert q.status == "unverified"
    assert "scan" in q.note


@pytest.mark.parametrize("question", ["", "   ", "x" * 501])
def test_bad_question_rejected_before_api(contract: LoadedDocument, question: str) -> None:
    llm = StubLLM(AnswerData(answer="a", quotes=[]))
    with pytest.raises(ValueError):
        ask([contract], question, llm, session_spent=0.0)
    assert llm.calls == 0


def test_no_documents_rejected() -> None:
    llm = StubLLM(AnswerData(answer="a", quotes=[]))
    with pytest.raises(ValueError):
        ask([], "What?", llm, session_spent=0.0)
    assert llm.calls == 0


def test_ask_sends_all_docs_and_checks_quotes(
    contract: LoadedDocument, invoice: LoadedDocument
) -> None:
    answer = AnswerData(
        answer="30 days.",
        quotes=[
            Quote(document=1, page=2, text="Termination notice: 30 days in writing."),
            Quote(document=2, page=1, text="Total due: USD 999.00"),
        ],
    )
    llm = StubLLM(answer)
    result = ask([contract, invoice], "  What is the notice period? ", llm, session_spent=0.02)
    content = llm.last["content"]
    assert content[0] == {"type": "text", "text": "Document 1: contract.pdf"}
    assert content[1]["type"] == "document"
    assert content[2] == {"type": "text", "text": "Document 2: inv.pdf"}
    assert content[3]["type"] == "document"
    assert content[-1] == {"type": "text", "text": ASK_PROMPT + "What is the notice period?"}
    assert llm.last["effort"] == "medium"
    assert llm.last["max_tokens"] == 2000
    assert llm.last["schema"] is AnswerData
    assert llm.last["session_spent"] == 0.02
    assert result.question == "What is the notice period?"
    assert result.answer == "30 days."
    assert [q.status for q in result.quotes] == ["verified", "not_found"]
    assert result.cached is False
    assert result.call is not None and result.call.kind == "ask"


def test_normalize_folds_ligatures_and_soft_hyphens() -> None:
    assert normalize("ﬁnal pay­ment") == normalize("final payment")


def test_quote_from_scanned_page_of_mixed_pdf_is_unverified(settings: Settings) -> None:
    mixed = LoadedDocument(
        name="mixed.pdf",
        media_type="application/pdf",
        data=b"%PDF",
        page_texts=("Termination notice: 30 days in writing. Governing law: Oregon.", ""),
        page_count=2,
    )
    q = check_quote(Quote(document=1, page=2, text="Signed by Maya Ortiz"), [mixed])
    assert q.status == "unverified"
    assert "scan" in q.note


def test_tiny_quote_is_not_verified(contract: LoadedDocument) -> None:
    q = check_quote(Quote(document=1, page=2, text="e"), [contract])
    assert q.status != "verified"
