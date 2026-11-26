"""One place that talks to the model: our own visible retries, friendly errors, cost and spend limits."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Generic, Literal, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from doc_agent.budget import BudgetExceeded, BudgetStore, check_budget
from doc_agent.config import Settings
from doc_agent.cost import cost_from_usage

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
ErrorKind = Literal[
    "auth",
    "rate_limit",
    "overloaded",
    "network",
    "refusal",
    "truncated",
    "bad_request",
    "bad_format",
    "budget",
    "no_key",
]
BACKOFF_BASE_S = 1.0
BACKOFF_JITTER_S = 0.5
MESSAGES: dict[str, str] = {
    "auth": "The AI service rejected the API key. Check OPENROUTER_API_KEY.",
    "rate_limit": "The AI service is rate-limiting requests right now. Please try again in a minute.",
    "overloaded": "The AI service is overloaded or down right now. Please try again shortly.",
    "network": "Could not reach the AI service (network problem or timeout).",
    "refusal": "The model declined to process this request.",
    "truncated": "The model's answer was cut off. Try a shorter document.",
    "bad_request": "The AI service could not process this request (the file may be unsupported).",
    "bad_format": "The model's answer did not match the expected format. Please try again.",
    "budget": "The demo spending limit has been reached. Sample results are still available.",
    "credits": "The AI account is out of credits, so live requests are off.",
    "session": "Your demo budget for this session is used up. Sample results are still available.",
    "daily": "The demo budget for today is used up. Sample results are still available.",
    "no_key": "No API key is configured, so live requests are off. Sample results are still available.",
}


class LLMError(Exception):
    """User-facing failure. `call` is set when the failed request was still billed."""

    def __init__(
        self, kind: ErrorKind, message: str | None = None, call: CallInfo | None = None
    ) -> None:
        super().__init__(message or MESSAGES[kind])
        self.kind: ErrorKind = kind
        self.call = call


class _SimulatedFailure(Exception):
    """Raised instead of a real call when the demo's 'simulate API failure' switch is on."""


@dataclass(frozen=True)
class CallInfo:
    kind: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cost_source: str
    attempts: int
    duration_s: float


@dataclass(frozen=True)
class LLMResult(Generic[T]):
    parsed: T
    call: CallInfo


def utc_today() -> date:
    return datetime.now(UTC).date()


def build_sdk(settings: Settings) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        base_url=settings.base_url,
        auth_token=settings.api_key,
        api_key=None,
        max_retries=0,
        timeout=settings.timeout_s,
    )


def output_format(schema: type[BaseModel]) -> dict:
    """Structured-output format. The schema models forbid extra fields, so the JSON schema is strict."""
    return {"type": "json_schema", "schema": schema.model_json_schema()}


def _first_text(response: object) -> str:
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


def _retry_kind(err: Exception) -> ErrorKind | None:
    """Kind to report if retries run out, or None when the error must not be retried."""
    if isinstance(err, anthropic.RateLimitError):
        return "rate_limit"
    if isinstance(err, anthropic.APIStatusError):
        return "overloaded" if err.status_code >= 500 else None
    if isinstance(err, anthropic.APIConnectionError | _SimulatedFailure):
        return "network"
    return None


def _final_error(err: Exception) -> LLMError:
    if isinstance(err, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
        return LLMError("auth")
    if isinstance(err, anthropic.APIStatusError) and err.status_code == 402:
        return LLMError("budget", MESSAGES["credits"])
    return LLMError("bad_request")


class LLMClient:
    def __init__(
        self,
        settings: Settings,
        budget: BudgetStore,
        sdk: object | None = None,
        sleep: Callable[[float], None] = time.sleep,
        today: Callable[[], date] = utc_today,
    ) -> None:
        self._settings = settings
        self._budget = budget
        self._sdk = sdk
        self._sleep = sleep
        self._today = today

    def _client(self) -> object:
        if self._sdk is None:
            if not self._settings.api_key:
                raise LLMError("no_key")
            self._sdk = build_sdk(self._settings)
        return self._sdk

    def _check_budget(self, session_spent: float) -> None:
        try:
            check_budget(self._budget, self._settings, session_spent, self._today())
        except BudgetExceeded as exc:
            raise LLMError("budget", MESSAGES[exc.scope]) from exc

    def _send(self, request: dict, simulate_failures: int) -> tuple[object, int]:
        sdk = self._client()
        for attempt in range(1, self._settings.max_attempts + 1):
            try:
                if attempt <= simulate_failures:
                    raise _SimulatedFailure()
                return sdk.messages.create(**request), attempt
            except (anthropic.APIError, _SimulatedFailure) as err:
                kind = _retry_kind(err)
                if kind is None:
                    raise _final_error(err) from err
                if attempt == self._settings.max_attempts:
                    raise LLMError(kind) from err
                log.info("Retryable error (%s), attempt %d", type(err).__name__, attempt)
                self._sleep(
                    BACKOFF_BASE_S * 2 ** (attempt - 1) + random.uniform(0, BACKOFF_JITTER_S)
                )
        raise LLMError("network")  # unreachable: the loop always returns or raises

    def call(
        self,
        *,
        kind: str,
        content: list[dict],
        schema: type[T],
        effort: str,
        max_tokens: int,
        session_spent: float,
        simulate_failures: int = 0,
    ) -> LLMResult[T]:
        self._check_budget(session_spent)
        request = {
            "model": self._settings.model,
            "max_tokens": max_tokens,
            "output_config": {"effort": effort, "format": output_format(schema)},
            "messages": [{"role": "user", "content": content}],
        }
        started = time.monotonic()
        response, attempts = self._send(request, simulate_failures)
        if getattr(response, "usage", None) is None:
            raise LLMError("bad_request")  # an error body the provider sent with HTTP 200
        info = self._record(kind, response, attempts, time.monotonic() - started)
        if response.stop_reason == "refusal":
            raise LLMError("refusal", call=info)
        if response.stop_reason == "max_tokens":
            raise LLMError("truncated", call=info)
        try:
            parsed = schema.model_validate_json(_first_text(response))
        except ValidationError as err:
            raise LLMError("bad_format", call=info) from err
        return LLMResult(parsed=parsed, call=info)

    def _record(self, kind: str, response: object, attempts: int, elapsed_s: float) -> CallInfo:
        """Every answered request is billed, so its cost counts before we look at the answer."""
        cost, source = cost_from_usage(response.usage, self._settings)
        self._budget.add(cost, self._today())
        info = CallInfo(
            kind=kind,
            model=str(response.model),
            input_tokens=int(response.usage.input_tokens),
            output_tokens=int(response.usage.output_tokens),
            cost_usd=cost,
            cost_source=source,
            attempts=attempts,
            duration_s=round(elapsed_s, 2),
        )
        log.info("LLM call %s", info)
        return info
