from pathlib import Path

import pytest

from doc_agent.ask import AnswerResult, CheckedQuote
from doc_agent.cache import save_answer, save_extraction
from doc_agent.config import Settings
from doc_agent.documents import LoadedDocument, load_document
from doc_agent.extract import ExtractionResult
from doc_agent.llm import LLMError
from doc_agent.schemas import AnswerData, Quote
from doc_agent.session import UNEXPECTED_ERROR, SessionLedger, run_extraction, run_question
from tests import factories
from tests.stubs import CALL, StubLLM
from tests.test_validate import make_invoice

SAMPLE = "invoice_clean.pdf"


@pytest.fixture
def doc(settings: Settings) -> LoadedDocument:
    return load_document(SAMPLE, factories.invoice_pdf(), settings)


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    cached = ExtractionResult(SAMPLE, "invoice", make_invoice(), (), CALL, cached=False)
    save_extraction(cached, tmp_path)
    quote = CheckedQuote(SAMPLE, 1, "Total due: USD 332.50", "verified", "")
    save_answer(
        SAMPLE, AnswerResult("What is the total?", "USD 332.50", (quote,), CALL, False), tmp_path
    )
    return tmp_path


def billed(kind: str) -> LLMError:
    return LLMError(kind, call=CALL)  # type: ignore[arg-type]


def test_ledger_is_immutable_and_sums() -> None:
    empty = SessionLedger()
    one = empty.add(CALL)
    two = one.add(CALL).add(None)
    assert empty.calls == ()
    assert len(one.calls) == 1
    assert len(two.calls) == 2
    assert two.spent_usd == pytest.approx(2 * CALL.cost_usd)


def test_live_extraction_adds_call(doc: LoadedDocument, cache_dir: Path) -> None:
    outcome = run_extraction(
        doc, "invoice", StubLLM(make_invoice()), SessionLedger(), None, cache_dir
    )
    assert outcome.result is not None and outcome.result.cached is False
    assert len(outcome.ledger.calls) == 1
    assert outcome.error is None and outcome.notice is None


def test_session_spent_is_passed_to_llm(doc: LoadedDocument, cache_dir: Path) -> None:
    llm = StubLLM(make_invoice())
    run_extraction(
        doc, "invoice", llm, SessionLedger().add(CALL), None, cache_dir, simulate_failures=1
    )
    assert llm.last["session_spent"] == pytest.approx(CALL.cost_usd)
    assert llm.last["simulate_failures"] == 1


@pytest.mark.parametrize("kind", ["budget", "no_key", "network", "overloaded", "rate_limit"])
def test_sample_falls_back_to_cache(doc: LoadedDocument, cache_dir: Path, kind: str) -> None:
    outcome = run_extraction(
        doc, "invoice", StubLLM(error=LLMError(kind)), SessionLedger(), SAMPLE, cache_dir
    )
    assert outcome.result is not None and outcome.result.cached is True
    assert outcome.error is None
    assert outcome.notice


def test_no_llm_uses_cache_for_samples(doc: LoadedDocument, cache_dir: Path) -> None:
    outcome = run_extraction(doc, "invoice", None, SessionLedger(), SAMPLE, cache_dir)
    assert outcome.result is not None and outcome.result.cached is True
    assert "No API key" in (outcome.notice or "")


def test_upload_does_not_use_cache(doc: LoadedDocument, cache_dir: Path) -> None:
    outcome = run_extraction(
        doc, "invoice", StubLLM(error=LLMError("budget")), SessionLedger(), None, cache_dir
    )
    assert outcome.result is None
    assert outcome.error == str(LLMError("budget"))


def test_auth_error_is_not_hidden_by_cache(doc: LoadedDocument, cache_dir: Path) -> None:
    outcome = run_extraction(
        doc, "invoice", StubLLM(error=LLMError("auth")), SessionLedger(), SAMPLE, cache_dir
    )
    assert outcome.result is None
    assert outcome.error


def test_cache_for_other_doc_type_not_used(doc: LoadedDocument, cache_dir: Path) -> None:
    outcome = run_extraction(doc, "contract", None, SessionLedger(), SAMPLE, cache_dir)
    assert outcome.result is None
    assert outcome.error


def test_billed_error_still_counts_in_ledger(doc: LoadedDocument, cache_dir: Path) -> None:
    outcome = run_extraction(
        doc, "invoice", StubLLM(error=billed("refusal")), SessionLedger(), None, cache_dir
    )
    assert outcome.result is None
    assert outcome.ledger.spent_usd == pytest.approx(CALL.cost_usd)


def test_question_live(doc: LoadedDocument, cache_dir: Path) -> None:
    llm = StubLLM(
        AnswerData(
            answer="USD 332.50", quotes=[Quote(document=1, page=1, text="Total due: USD 332.50")]
        )
    )
    outcome = run_question([doc], "What is the total?", llm, SessionLedger(), None, cache_dir)
    assert outcome.result is not None and outcome.result.quotes[0].status == "verified"
    assert len(outcome.ledger.calls) == 1


def test_question_falls_back_to_cache_for_sample(doc: LoadedDocument, cache_dir: Path) -> None:
    outcome = run_question([doc], "What is the total?", None, SessionLedger(), SAMPLE, cache_dir)
    assert outcome.result is not None and outcome.result.cached is True


def test_uncached_question_on_sample_reports_error(doc: LoadedDocument, cache_dir: Path) -> None:
    outcome = run_question([doc], "Something new?", None, SessionLedger(), SAMPLE, cache_dir)
    assert outcome.result is None
    assert outcome.error


def test_bad_question_is_an_error_not_a_crash(doc: LoadedDocument, cache_dir: Path) -> None:
    llm = StubLLM(AnswerData(answer="a", quotes=[]))
    outcome = run_question([doc], "   ", llm, SessionLedger(), None, cache_dir)
    assert outcome.result is None
    assert outcome.error == "Please type a question."
    assert llm.calls == 0


class ExplodingLLM:
    def call(self, **_kwargs: object) -> None:
        raise RuntimeError("secret internal detail")


def test_unexpected_errors_become_a_generic_message(doc: LoadedDocument, cache_dir: Path) -> None:
    outcome = run_extraction(doc, "invoice", ExplodingLLM(), SessionLedger(), None, cache_dir)
    assert outcome.result is None
    assert outcome.error == UNEXPECTED_ERROR
    assert "secret" not in outcome.error


def test_unexpected_errors_do_not_use_cache(doc: LoadedDocument, cache_dir: Path) -> None:
    outcome = run_question(
        [doc], "What is the total?", ExplodingLLM(), SessionLedger(), SAMPLE, cache_dir
    )
    assert outcome.result is None
    assert outcome.error == UNEXPECTED_ERROR
