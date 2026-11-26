"""Request cost: the provider's own figure when it sends one, otherwise an estimate from tokens."""

from __future__ import annotations

from typing import Literal

from doc_agent.config import Settings

CostSource = Literal["provider", "estimated"]
PER_MILLION = 1_000_000


def estimate_cost(input_tokens: int, output_tokens: int, settings: Settings) -> float:
    long_prompt = input_tokens > settings.long_prompt_threshold
    price_in = settings.price_in_long_per_m if long_prompt else settings.price_in_per_m
    price_out = settings.price_out_long_per_m if long_prompt else settings.price_out_per_m
    return (input_tokens * price_in + output_tokens * price_out) / PER_MILLION


def cost_from_usage(usage: object, settings: Settings) -> tuple[float, CostSource]:
    extra = getattr(usage, "model_extra", None) or {}
    reported = extra.get("cost")
    if isinstance(reported, int | float) and not isinstance(reported, bool):
        return float(reported), "provider"
    estimate = estimate_cost(
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
        settings,
    )
    return estimate, "estimated"
