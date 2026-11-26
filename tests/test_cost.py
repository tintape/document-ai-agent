from types import SimpleNamespace

import pytest

from doc_agent.config import Settings
from doc_agent.cost import cost_from_usage, estimate_cost


def test_short_prompt_price(settings: Settings) -> None:
    assert estimate_cost(10_000, 1_000, settings) == pytest.approx(0.0015)


def test_long_prompt_price(settings: Settings) -> None:
    assert estimate_cost(200_000, 1_000, settings) == pytest.approx(0.1025)


def test_threshold_is_inclusive_short(settings: Settings) -> None:
    assert estimate_cost(100_000, 0, settings) == pytest.approx(0.01)


def test_provider_cost_wins(settings: Settings) -> None:
    usage = SimpleNamespace(input_tokens=2017, output_tokens=59, model_extra={"cost": 0.0002312})
    assert cost_from_usage(usage, settings) == (0.0002312, "provider")


def test_falls_back_to_estimate_when_cost_missing(settings: Settings) -> None:
    usage = SimpleNamespace(input_tokens=10_000, output_tokens=1_000, model_extra={})
    cost, source = cost_from_usage(usage, settings)
    assert cost == pytest.approx(0.0015)
    assert source == "estimated"


def test_falls_back_when_cost_is_not_a_number(settings: Settings) -> None:
    usage = SimpleNamespace(input_tokens=10_000, output_tokens=1_000, model_extra={"cost": None})
    assert cost_from_usage(usage, settings)[1] == "estimated"


def test_usage_without_model_extra(settings: Settings) -> None:
    usage = SimpleNamespace(input_tokens=10_000, output_tokens=1_000)
    assert cost_from_usage(usage, settings)[1] == "estimated"
