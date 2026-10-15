"""Optional embedders. None are required - the store works on BM25 alone."""

from __future__ import annotations


class OpenAICompatibleEmbedder:
    """Any OpenAI-compatible /embeddings endpoint: Ollama, vLLM, LM Studio, OpenAI, ...

    Ollama example: ``OpenAICompatibleEmbedder("nomic-embed-text",
    base_url="http://localhost:11434/v1", api_key="ollama")``
    """

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 batch_size: int = 64) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError("OpenAICompatibleEmbedder needs `pip install sec10k[openai]`") from e
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.name = f"openai-compat:{model}"
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            resp = self._client.embeddings.create(model=self.model, input=texts[i:i + self.batch_size])
            out.extend(d.embedding for d in resp.data)
        return out


class FastEmbedEmbedder:
    """Local CPU embeddings via fastembed (ONNX). ``pip install sec10k[local-embed]``."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as e:  # pragma: no cover
            raise ImportError("FastEmbedEmbedder needs `pip install sec10k[local-embed]`") from e
        self._model = TextEmbedding(model)
        self.name = f"fastembed:{model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.embed(texts)]
