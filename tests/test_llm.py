from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import get_args

import anthropic
import httpx2
import pytest

from doc_agent.budget import BudgetStore
from doc_agent.config import Settings, load_settings
from doc_agent.llm import ErrorKind, LLMClient, LLMError, build_sdk
from doc_agent.schemas import AnswerData

DAY = date(2026, 11, 27)
URL = "https://openrouter.ai/api/v1/messages"


def _response(status: int) -> httpx2.Response:
    return httpx2.Response(status, request=httpx2.Request("POST", URL))


def status_error(cls: type[anthropic.APIStatusError], status: int) -> anthropic.APIStatusError:
    return cls("boom", response=_response(status), body=None)


def ok(
    parsed: object = None, cost: float | None = 0.0002, stop: str = "end_turn"
) -> SimpleNamespace:
    extra = {} if cost is None else {"cost": cost}
    answer = parsed if parsed is not None else AnswerData(answer="a", quotes=[])
    return SimpleNamespace(
        stop_reason=stop,
        content=[SimpleNamespace(type="text", text=answer.model_dump_json())],
        usage=SimpleNamespace(input_tokens=100, output_tokens=10, model_extra=extra),
        model="anthropic/claude-haiku-5.5",
    )


class FakeSDK:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0
        self.last_kwargs: dict = {}
        self.messages = SimpleNamespace(create=self._parse)

    def _parse(self, **kwargs: object) -> object:
        self.calls += 1
        self.last_kwargs = kwargs
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def store(tmp_path: Path) -> BudgetStore:
    return BudgetStore(tmp_path / "budget.json")


@pytest.fixture
def sleeps() -> list[float]:
    return []


@pytest.fixture
def make_client(settings: Settings, store: BudgetStore, sleeps: list[float]):
    def factory(outcomes: list[object]) -> tuple[LLMClient, FakeSDK]:
        fake = FakeSDK(outcomes)
        client = LLMClient(settings, store, sdk=fake, sleep=sleeps.append, today=lambda: DAY)
        return client, fake

    return factory


def call(client: LLMClient, **overrides: object):
    kwargs: dict = {
        "kind": "ask",
        "content": [{"type": "text", "text": "q"}],
        "schema": AnswerData,
        "effort": "medium",
        "max_tokens": 2000,
        "session_spent": 0.0,
    }
    kwargs.update(overrides)
    return client.call(**kwargs)


def test_success_returns_parsed_and_cost(make_client) -> None:
    client, fake = make_client([ok(cost=0.0002)])
    result = call(client)
    assert result.parsed.answer == "a"
    assert result.call.cost_usd == 0.0002
    assert result.call.cost_source == "provider"
    assert result.call.attempts == 1
    assert result.call.kind == "ask"
    assert (result.call.input_tokens, result.call.output_tokens) == (100, 10)
    assert fake.last_kwargs["model"] == "anthropic/claude-haiku-5.5"
    assert fake.last_kwargs["output_config"] == {
        "effort": "medium",
        "format": {"type": "json_schema", "schema": AnswerData.model_json_schema()},
    }
    assert fake.last_kwargs["max_tokens"] == 2000
    assert fake.last_kwargs["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "q"}]}
    ]


def test_retries_rate_limit_then_succeeds(make_client, sleeps: list[float]) -> None:
    client, fake = make_client([status_error(anthropic.RateLimitError, 429), ok()])
    assert call(client).call.attempts == 2
    assert fake.calls == 2
    assert 1.0 <= sleeps[0] <= 1.5


def test_backoff_doubles(make_client, sleeps: list[float]) -> None:
    client, _ = make_client(
        [anthropic.APIConnectionError(request=httpx2.Request("POST", URL))] * 2 + [ok()]
    )
    assert call(client).call.attempts == 3
    assert 1.0 <= sleeps[0] <= 1.5
    assert 2.0 <= sleeps[1] <= 2.5


def test_gives_up_after_three_attempts(make_client) -> None:
    client, fake = make_client([status_error(anthropic.InternalServerError, 503)] * 3)
    with pytest.raises(LLMError) as err:
        call(client)
    assert err.value.kind == "overloaded"
    assert fake.calls == 3


def test_rate_limit_exhausted_kind(make_client) -> None:
    client, _ = make_client([status_error(anthropic.RateLimitError, 429)] * 3)
    with pytest.raises(LLMError) as err:
        call(client)
    assert err.value.kind == "rate_limit"


def test_timeout_is_network(make_client) -> None:
    timeout = anthropic.APITimeoutError(request=httpx2.Request("POST", URL))
    client, _ = make_client([timeout] * 3)
    with pytest.raises(LLMError) as err:
        call(client)
    assert err.value.kind == "network"


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (status_error(anthropic.AuthenticationError, 401), "auth"),
        (status_error(anthropic.PermissionDeniedError, 403), "auth"),
        (status_error(anthropic.BadRequestError, 400), "bad_request"),
        (status_error(anthropic.APIStatusError, 402), "budget"),
    ],
)
def test_client_errors_not_retried(make_client, error: Exception, kind: str) -> None:
    client, fake = make_client([error])
    with pytest.raises(LLMError) as err:
        call(client)
    assert err.value.kind == kind
    assert fake.calls == 1
    assert "boom" not in str(err.value)


@pytest.mark.parametrize(("stop", "kind"), [("refusal", "refusal"), ("max_tokens", "truncated")])
def test_refusal_and_truncation(make_client, stop: str, kind: str) -> None:
    client, _ = make_client([ok(stop=stop)])
    with pytest.raises(LLMError) as err:
        call(client)
    assert err.value.kind == kind


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_billed_failures_still_count_against_budget(
    make_client, store: BudgetStore, stop: str
) -> None:
    client, _ = make_client([ok(stop=stop, cost=0.0003)])
    with pytest.raises(LLMError) as err:
        call(client)
    assert store.spent_on(DAY) == pytest.approx(0.0003)
    assert err.value.call is not None
    assert err.value.call.cost_usd == pytest.approx(0.0003)


def test_unbilled_errors_carry_no_call(make_client) -> None:
    client, _ = make_client([status_error(anthropic.AuthenticationError, 401)])
    with pytest.raises(LLMError) as err:
        call(client)
    assert err.value.call is None


def test_missing_text_is_bad_format(make_client) -> None:
    response = ok()
    response.content = []
    client, _ = make_client([response])
    with pytest.raises(LLMError) as err:
        call(client)
    assert err.value.kind == "bad_format"


def test_simulated_failure_is_retried_without_network(make_client) -> None:
    client, fake = make_client([ok()])
    assert call(client, simulate_failures=1).call.attempts == 2
    assert fake.calls == 1


def test_session_budget_blocks_before_calling(make_client) -> None:
    client, fake = make_client([ok()])
    with pytest.raises(LLMError) as err:
        call(client, session_spent=0.10)
    assert err.value.kind == "budget"
    assert "session" in str(err.value)
    assert fake.calls == 0


def test_daily_budget_blocks_before_calling(make_client, store: BudgetStore) -> None:
    store.add(1.0, DAY)
    client, fake = make_client([ok()])
    with pytest.raises(LLMError) as err:
        call(client)
    assert err.value.kind == "budget"
    assert "today" in str(err.value)
    assert fake.calls == 0


def test_cost_is_recorded_in_daily_budget(make_client, store: BudgetStore) -> None:
    client, _ = make_client([ok(cost=0.0002)])
    call(client)
    assert store.spent_on(DAY) == pytest.approx(0.0002)


def test_estimated_cost_when_provider_sends_none(make_client) -> None:
    client, _ = make_client([ok(cost=None)])
    result = call(client)
    assert result.call.cost_source == "estimated"
    assert result.call.cost_usd == pytest.approx((100 * 0.10 + 10 * 0.50) / 1_000_000)


def test_missing_key_gives_no_key_error(tmp_path: Path, store: BudgetStore) -> None:
    client = LLMClient(load_settings({}, tmp_path), store)
    with pytest.raises(LLMError) as err:
        call(client)
    assert err.value.kind == "no_key"


def test_build_sdk_uses_bearer_and_no_sdk_retries(settings: Settings) -> None:
    sdk = build_sdk(settings)
    assert str(sdk.base_url).startswith("https://openrouter.ai/api")
    assert sdk.auth_token == "test"
    assert sdk.api_key is None
    assert sdk.max_retries == 0


@pytest.mark.parametrize("kind", list(get_args(ErrorKind)))
def test_every_error_kind_has_a_default_message(kind: str) -> None:
    assert str(LLMError(kind))
