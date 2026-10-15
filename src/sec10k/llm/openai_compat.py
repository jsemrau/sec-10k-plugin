from __future__ import annotations

from collections.abc import Callable

from ..fallbacks import FallbackLog
from .base import LLMResponse, Usage, price_usage


class OpenAICompatibleLLM:
    """Any OpenAI-compatible chat endpoint: Ollama, vLLM, LM Studio, OpenRouter, OpenAI, ...

    Local Ollama (free): ``OpenAICompatibleLLM("qwen2.5:14b",
    base_url="http://localhost:11434/v1", api_key="ollama", price=(0, 0))``
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 4096,
        price: tuple[float, ...] | None = None,
        client=None,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as e:  # pragma: no cover
                raise ImportError("OpenAICompatibleLLM needs `pip install sec10k[openai]`") from e
            client = OpenAI(base_url=base_url, api_key=api_key)
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.price = price

    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        on_text: Callable[[str], None] | None = None,
        log: FallbackLog | None = None,
    ) -> LLMResponse:
        log = log or FallbackLog()
        msgs = [{"role": "system", "content": system}, *messages]
        stop_reason = None
        usage = None
        if on_text:
            parts: list[str] = []
            stream = self.client.chat.completions.create(
                model=self.model, messages=msgs, max_tokens=self.max_tokens, stream=True,
                stream_options={"include_usage": True},
            )
            for chunk in stream:
                if chunk.usage:
                    usage = chunk.usage
                if chunk.choices:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        parts.append(delta)
                        on_text(delta)
                    stop_reason = chunk.choices[0].finish_reason or stop_reason
            text = "".join(parts)
        else:
            resp = self.client.chat.completions.create(
                model=self.model, messages=msgs, max_tokens=self.max_tokens
            )
            text = resp.choices[0].message.content or ""
            stop_reason = resp.choices[0].finish_reason
            usage = resp.usage

        if usage is None:
            log.record("usage_unavailable", "endpoint returned no token usage", model=self.model)
            u = Usage()
        else:
            u = Usage(input_tokens=usage.prompt_tokens or 0, output_tokens=usage.completion_tokens or 0)
        if stop_reason == "length":
            log.record("llm_truncated", f"answer hit max_tokens={self.max_tokens}")
        u = price_usage(u, self.price, self.model, log)
        return LLMResponse(text=text, model=self.model, usage=u, stop_reason=stop_reason)
