"""LLMClient driven through the real anthropic SDK with a mocked HTTP transport (no network)."""

import json
from datetime import date
from pathlib import Path

import anthropic
import httpx2
import pytest

from doc_agent.budget import BudgetStore
from doc_agent.config import Settings
from doc_agent.llm import LLMClient, LLMError
from doc_agent.schemas import AnswerData

DAY = date(2026, 11, 27)
GOOD_JSON = json.dumps({"answer": "USD 332.50", "quotes": []})


def message(text: str | None, stop: str, cost: float = 0.0023) -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "anthropic/claude-haiku-5.5",
        "content": [] if text is None else [{"type": "text", "text": text}],
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": 3000, "output_tokens": 400, "cost": cost},
    }


def real_sdk(body: dict, seen: list[dict] | None = None) -> anthropic.Anthropic:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if seen is not None:
            seen.append(json.loads(request.content))
        return httpx2.Response(200, json=body)

    return anthropic.Anthropic(
        base_url="http://test.invalid",
        auth_token="t",
        api_key=None,
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )


def run(settings: Settings, store: BudgetStore, body: dict, seen: list[dict] | None = None):
    client = LLMClient(
        settings, store, sdk=real_sdk(body, seen), sleep=lambda _: None, today=lambda: DAY
    )
    return client.call(
        kind="ask",
        content=[{"type": "text", "text": "q"}],
        schema=AnswerData,
        effort="medium",
        max_tokens=2000,
        session_spent=0.0,
    )


@pytest.fixture
def store(tmp_path: Path) -> BudgetStore:
    return BudgetStore(tmp_path / "budget.json")


def test_valid_answer_is_parsed_and_billed(settings: Settings, store: BudgetStore) -> None:
    seen: list[dict] = []
    result = run(settings, store, message(GOOD_JSON, "end_turn"), seen)
    assert result.parsed.answer == "USD 332.50"
    assert store.spent_on(DAY) == pytest.approx(0.0023)
    sent = seen[0]
    assert sent["output_config"]["effort"] == "medium"
    assert sent["output_config"]["format"]["type"] == "json_schema"
    assert sent["output_config"]["format"]["schema"]["additionalProperties"] is False


@pytest.mark.parametrize(
    ("text", "stop", "kind"),
    [
        ('{"answer": "USD 3', "max_tokens", "truncated"),
        (None, "refusal", "refusal"),
        ("I can't help with that", "refusal", "refusal"),
        ('{"foo": 1}', "end_turn", "bad_format"),
        ("not json at all", "end_turn", "bad_format"),
        (None, "end_turn", "bad_format"),
    ],
)
def test_billed_failures_are_recorded_and_friendly(
    settings: Settings, store: BudgetStore, text: str | None, stop: str, kind: str
) -> None:
    with pytest.raises(LLMError) as err:
        run(settings, store, message(text, stop))
    assert err.value.kind == kind
    assert err.value.call is not None
    assert store.spent_on(DAY) == pytest.approx(0.0023)
    assert "validation error" not in str(err.value)


def test_error_body_with_200_is_friendly(settings: Settings, store: BudgetStore) -> None:
    with pytest.raises(LLMError) as err:
        run(settings, store, {"error": {"message": "Provider returned error", "code": 502}})
    assert err.value.kind in {"bad_request", "overloaded"}
