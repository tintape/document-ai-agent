"""UI-free session logic: what one visitor spent, and when a bundled sample may fall back to cached results."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from doc_agent.ask import AnswerResult, ask
from doc_agent.cache import load_answer, load_extraction
from doc_agent.documents import LoadedDocument
from doc_agent.extract import ExtractionResult, extract
from doc_agent.llm import MESSAGES, CallInfo, LLMClient, LLMError

R = TypeVar("R")
FALLBACK_KINDS = frozenset({"no_key", "budget", "network", "overloaded", "rate_limit"})


@dataclass(frozen=True)
class SessionLedger:
    calls: tuple[CallInfo, ...] = ()

    def add(self, call: CallInfo | None) -> SessionLedger:
        return self if call is None else SessionLedger(self.calls + (call,))

    @property
    def spent_usd(self) -> float:
        return sum(call.cost_usd for call in self.calls)


@dataclass(frozen=True)
class Outcome(Generic[R]):
    """result is None exactly when error is set; notice explains a cached fallback."""

    result: R | None
    ledger: SessionLedger
    error: str | None = None
    notice: str | None = None


def _run(
    live: Callable[[LLMClient, float], R],
    cached: Callable[[], R | None],
    llm: LLMClient | None,
    ledger: SessionLedger,
    is_sample: bool,
) -> Outcome[R]:
    try:
        if llm is None:
            raise LLMError("no_key")
        result = live(llm, ledger.spent_usd)
    except LLMError as err:
        ledger = ledger.add(err.call)
        fallback = cached() if is_sample and err.kind in FALLBACK_KINDS else None
        if fallback is not None:
            return Outcome(fallback, ledger, notice=f"{err} Showing a cached result.")
        return Outcome(None, ledger, error=str(err))
    except ValueError as err:
        return Outcome(None, ledger, error=str(err))
    return Outcome(result, ledger.add(getattr(result, "call", None)))


def run_extraction(
    doc: LoadedDocument,
    doc_type: str,
    llm: LLMClient | None,
    ledger: SessionLedger,
    sample: str | None,
    cache_dir: Path,
    simulate_failures: int = 0,
) -> Outcome[ExtractionResult]:
    def cached() -> ExtractionResult | None:
        result = load_extraction(sample, cache_dir) if sample else None
        return result if result is not None and result.doc_type == doc_type else None

    def live(client: LLMClient, spent: float) -> ExtractionResult:
        return extract(doc, doc_type, client, spent, simulate_failures)

    return _run(live, cached, llm, ledger, sample is not None)


def run_question(
    docs: Sequence[LoadedDocument],
    question: str,
    llm: LLMClient | None,
    ledger: SessionLedger,
    sample: str | None,
    cache_dir: Path,
    simulate_failures: int = 0,
) -> Outcome[AnswerResult]:
    def cached() -> AnswerResult | None:
        return load_answer(sample, question, cache_dir) if sample else None

    def live(client: LLMClient, spent: float) -> AnswerResult:
        return ask(docs, question, client, spent, simulate_failures)

    outcome = _run(live, cached, llm, ledger, sample is not None)
    if outcome.error == MESSAGES["no_key"] and sample:
        hint = " Try one of the suggested questions — those have cached answers."
        return Outcome(None, outcome.ledger, error=outcome.error + hint)
    return outcome
