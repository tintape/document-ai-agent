from pathlib import Path

from doc_agent.ask import AnswerResult, CheckedQuote
from doc_agent.cache import (
    SAMPLE_QUESTIONS,
    SAMPLES,
    load_answer,
    load_extraction,
    save_answer,
    save_extraction,
)
from doc_agent.extract import ExtractionResult
from doc_agent.validate import Flag
from tests.stubs import CALL
from tests.test_validate import fv, make_invoice


def extraction() -> ExtractionResult:
    return ExtractionResult(
        doc_name="invoice_clean.pdf",
        doc_type="invoice",
        data=make_invoice(total=fv("400.00")),
        flags=(Flag("total", "Line items do not add up"),),
        call=CALL,
        cached=False,
    )


def answer(question: str = "What is the total?") -> AnswerResult:
    quote = CheckedQuote("invoice_clean.pdf", 1, "Total due: USD 332.50", "verified", "")
    return AnswerResult(question, "USD 332.50", (quote,), CALL, cached=False)


def test_every_sample_has_a_type_and_two_questions() -> None:
    assert set(SAMPLES.values()) <= {"invoice", "contract"}
    assert set(SAMPLE_QUESTIONS) == set(SAMPLES)
    assert all(len(qs) == 2 for qs in SAMPLE_QUESTIONS.values())


def test_extraction_roundtrip(tmp_path: Path) -> None:
    original = extraction()
    save_extraction(original, tmp_path)
    loaded = load_extraction("invoice_clean.pdf", tmp_path)
    assert loaded is not None
    assert loaded.cached is True
    assert loaded.call is None
    assert loaded.data == original.data
    assert loaded.flags == original.flags
    assert (loaded.doc_name, loaded.doc_type) == ("invoice_clean.pdf", "invoice")


def test_answer_roundtrip_and_replace(tmp_path: Path) -> None:
    save_answer("invoice_clean.pdf", answer(), tmp_path)
    save_answer("invoice_clean.pdf", answer("Who is the vendor?"), tmp_path)
    save_answer("invoice_clean.pdf", answer(), tmp_path)
    loaded = load_answer("invoice_clean.pdf", " What is the total? ", tmp_path)
    assert loaded is not None
    assert loaded.cached is True
    assert loaded.call is None
    assert loaded.answer == "USD 332.50"
    assert loaded.quotes == answer().quotes
    assert load_answer("invoice_clean.pdf", "Who is the vendor?", tmp_path) is not None


def test_missing_cache_returns_none(tmp_path: Path) -> None:
    assert load_extraction("invoice_clean.pdf", tmp_path) is None
    assert load_answer("invoice_clean.pdf", "What is the total?", tmp_path) is None


def test_unknown_question_returns_none(tmp_path: Path) -> None:
    save_answer("invoice_clean.pdf", answer(), tmp_path)
    assert load_answer("invoice_clean.pdf", "Something else?", tmp_path) is None


def test_corrupted_cache_returns_none(tmp_path: Path) -> None:
    (tmp_path / "invoice_clean.pdf.extract.json").write_text("{broken")
    (tmp_path / "invoice_clean.pdf.qa.json").write_text('{"not": "a list"}')
    assert load_extraction("invoice_clean.pdf", tmp_path) is None
    assert load_answer("invoice_clean.pdf", "What is the total?", tmp_path) is None


def test_pdf_and_png_samples_do_not_share_cache(tmp_path: Path) -> None:
    pdf = ExtractionResult("invoice_messy_scan.pdf", "invoice", make_invoice(), (), CALL, False)
    png = ExtractionResult(
        "invoice_messy_scan.png", "invoice", make_invoice(vendor=fv("Other")), (), CALL, False
    )
    save_extraction(pdf, tmp_path)
    save_extraction(png, tmp_path)
    loaded = load_extraction("invoice_messy_scan.pdf", tmp_path)
    assert loaded is not None and loaded.data.vendor.value == "Blue Harbor Supplies Ltd."
