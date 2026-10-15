from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..fallbacks import FallbackLog


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cost_usd: float | None = None  # None = price unknown (and recorded as a fallback)

    def __add__(self, other: Usage) -> Usage:
        cost = None if self.cost_usd is None or other.cost_usd is None else self.cost_usd + other.cost_usd
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_input_tokens + other.cache_read_input_tokens,
            self.cache_creation_input_tokens + other.cache_creation_input_tokens,
            cost,
        )


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: Usage
    stop_reason: str | None = None
    raw: Any = field(default=None, repr=False)


# (input $/MTok, output $/MTok, cache-read $/MTok). Pass price= to override or add models.
ANTHROPIC_PRICES: dict[str, tuple[float, float, float]] = {
    "claude-fable-5-1": (10.0, 50.0, 0.25),
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-opus-5": (5.0, 25.0, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20),
    "claude-sonnet-5": (2.0, 10.0, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}


def price_usage(usage: Usage, price: tuple[float, ...] | None, model: str,
                log: FallbackLog) -> Usage:
    if price is None:
        log.record("cost_unknown", f"no price configured for model {model!r}; cost not computed",
                   model=model)
        return usage
    p_in, p_out = price[0], price[1]
    p_cache = price[2] if len(price) > 2 else p_in
    usage.cost_usd = (
        usage.input_tokens * p_in
        + usage.output_tokens * p_out
        + usage.cache_read_input_tokens * p_cache
        + usage.cache_creation_input_tokens * p_in * 1.25
    ) / 1_000_000
    return usage


@runtime_checkable
class LLM(Protocol):
    """Minimal chat interface. Implement this to plug in any provider."""

    model: str

    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        on_text: Callable[[str], None] | None = None,
        log: FallbackLog | None = None,
    ) -> LLMResponse: ...
