"""Questions over documents. The model cites quotes; our code checks each quote against the PDF text."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from doc_agent.documents import LoadedDocument
from doc_agent.llm import CallInfo, LLMClient
from doc_agent.prompts import ASK_PROMPT
from doc_agent.schemas import AnswerData, Quote

QuoteStatus = Literal["verified", "not_found", "unverified"]
ASK_EFFORT = "medium"
ASK_MAX_TOKENS = 2000
MAX_QUESTION_CHARS = 500
MIN_QUOTE_CHARS = 8
_TRANSLATE = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
    }
)
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class CheckedQuote:
    doc_name: str
    page: int
    text: str
    status: QuoteStatus
    note: str


@dataclass(frozen=True)
class AnswerResult:
    question: str
    answer: str
    quotes: tuple[CheckedQuote, ...]
    call: CallInfo | None
    cached: bool


def normalize(text: str) -> str:
    """Fold ligatures, soft hyphens, curly quotes, dashes, case and spacing before comparing."""
    folded = unicodedata.normalize("NFKC", text).replace("­", "")
    return _SPACES.sub(" ", folded.translate(_TRANSLATE)).strip().lower()


def _page_has_text(doc: LoadedDocument, page: int) -> bool:
    return 1 <= page <= doc.page_count and bool(doc.page_texts[page - 1].strip())


def check_quote(quote: Quote, docs: Sequence[LoadedDocument]) -> CheckedQuote:
    if not 1 <= quote.document <= len(docs):
        return CheckedQuote("?", quote.page, quote.text, "not_found", "no such document")
    doc = docs[quote.document - 1]

    def result(status: QuoteStatus, note: str, page: int = quote.page) -> CheckedQuote:
        return CheckedQuote(doc.name, page, quote.text, status, note)

    if not doc.has_text_layer:
        return result("unverified", "scan: no text to check")
    needle = normalize(quote.text)
    if len(needle) < MIN_QUOTE_CHARS:
        return result("unverified", "quote too short to check")
    if _page_has_text(doc, quote.page) and needle in normalize(doc.page_texts[quote.page - 1]):
        return result("verified", "")
    for number, page_text in enumerate(doc.page_texts, start=1):
        if needle in normalize(page_text):
            return result("verified", f"found on page {number}, not page {quote.page}", number)
    if 1 <= quote.page <= doc.page_count and not _page_has_text(doc, quote.page):
        return result("unverified", "scanned page: no text to check")
    return result("not_found", "not in the document text")


def _content(docs: Sequence[LoadedDocument], question: str) -> list[dict]:
    content: list[dict] = []
    for number, doc in enumerate(docs, start=1):
        content.append({"type": "text", "text": f"Document {number}: {doc.name}"})
        content.append(doc.content_block())
    content.append({"type": "text", "text": ASK_PROMPT + question})
    return content


def ask(
    docs: Sequence[LoadedDocument],
    question: str,
    llm: LLMClient,
    session_spent: float,
    simulate_failures: int = 0,
) -> AnswerResult:
    clean = question.strip()
    if not clean:
        raise ValueError("Please type a question.")
    if len(clean) > MAX_QUESTION_CHARS:
        raise ValueError(f"Please keep the question under {MAX_QUESTION_CHARS} characters.")
    if not docs:
        raise ValueError("Pick a sample or upload a document first.")
    result = llm.call(
        kind="ask",
        content=_content(docs, clean),
        schema=AnswerData,
        effort=ASK_EFFORT,
        max_tokens=ASK_MAX_TOKENS,
        session_spent=session_spent,
        simulate_failures=simulate_failures,
    )
    quotes = tuple(check_quote(q, docs) for q in result.parsed.quotes)
    return AnswerResult(clean, result.parsed.answer, quotes, result.call, cached=False)
