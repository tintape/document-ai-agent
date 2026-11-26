"""A stand-in for LLMClient that returns scripted results and records what it was asked."""

from __future__ import annotations

from pydantic import BaseModel

from doc_agent.llm import CallInfo, LLMError, LLMResult

CALL = CallInfo(
    kind="extract",
    model="anthropic/claude-haiku-5.5",
    input_tokens=2000,
    output_tokens=60,
    cost_usd=0.0002,
    cost_source="provider",
    attempts=1,
    duration_s=1.2,
)


class StubLLM:
    def __init__(self, parsed: BaseModel | None = None, error: LLMError | None = None) -> None:
        self._parsed = parsed
        self._error = error
        self.calls = 0
        self.last: dict = {}

    def call(self, **kwargs: object) -> LLMResult:
        self.calls += 1
        self.last = kwargs
        if self._error is not None:
            raise self._error
        info = CallInfo(**{**CALL.__dict__, "kind": str(kwargs["kind"])})
        return LLMResult(parsed=self._parsed, call=info)
