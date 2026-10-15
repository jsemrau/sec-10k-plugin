from __future__ import annotations

from collections.abc import Callable

from ..fallbacks import FallbackLog
from .base import ANTHROPIC_PRICES, LLMResponse, Usage, price_usage

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLLM:
    """Claude via the official ``anthropic`` SDK (``pip install sec10k[anthropic]``).

    ``server_fallbacks=True`` lets the API re-run a request on another model if the
    requested one declines it. When that happens it is recorded as a fallback event,
    so you always know which model actually answered.
    """

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: str = "medium",
        max_tokens: int = 16000,
        server_fallbacks: bool = True,
        price: tuple[float, ...] | None = None,
        client=None,
    ) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover
                raise ImportError("AnthropicLLM needs `pip install sec10k[anthropic]`") from e
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.server_fallbacks = server_fallbacks
        self.price = price or ANTHROPIC_PRICES.get(model)

    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        on_text: Callable[[str], None] | None = None,
        log: FallbackLog | None = None,
    ) -> LLMResponse:
        log = log or FallbackLog()
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            output_config={"effort": self.effort},
        )
        if self.server_fallbacks:
            ctx = self.client.beta.messages.stream(**kwargs, betas=[FALLBACK_BETA], fallbacks="default")
        else:
            ctx = self.client.messages.stream(**kwargs)
        with ctx as stream:
            if on_text:
                for text in stream.text_stream:
                    on_text(text)
            msg = stream.get_final_message()

        for block in msg.content:
            if block.type == "fallback":
                log.record(
                    "llm_server_fallback",
                    f"{block.from_.model} declined; {block.to.model} continued",
                    requested=self.model,
                )
        if msg.model != self.model and not any(b.type == "fallback" for b in msg.content):
            log.record("llm_model_substituted", f"requested {self.model}, served by {msg.model}")
        if msg.stop_reason == "refusal":
            details = getattr(msg, "stop_details", None)
            log.record("llm_refusal", "model declined to answer",
                       category=getattr(details, "category", None))
        elif msg.stop_reason == "max_tokens":
            log.record("llm_truncated", f"answer hit max_tokens={self.max_tokens}")

        u = msg.usage
        usage = Usage(
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
        price = self.price if msg.model == self.model else ANTHROPIC_PRICES.get(msg.model)
        usage = price_usage(usage, price, msg.model, log)
        text = "".join(b.text for b in msg.content if b.type == "text")
        return LLMResponse(text=text, model=msg.model, usage=usage, stop_reason=msg.stop_reason, raw=msg)
