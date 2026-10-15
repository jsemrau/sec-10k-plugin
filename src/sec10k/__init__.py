"""sec10k - pull Form 10-K filings from SEC EDGAR, remember them locally, ask questions cheaply.

Quick start::

    from sec10k import SEC10KAnalyst, AnthropicLLM

    analyst = SEC10KAnalyst(llm=AnthropicLLM())
    analyst.ingest("APLD")
    print(analyst.ask("APLD", "Does the company have legal issues?").text)
"""

from .analyst import (
    DEFAULT_SYSTEM_PROMPT,
    Answer,
    IngestError,
    IngestResult,
    Prompt,
    SEC10KAnalyst,
    SearchResult,
    user_namespace,
)
from .chunking import Chunk, chunk_filing
from .fallbacks import FallbackLog
from .llm import LLM, AnthropicLLM, LLMResponse, OpenAICompatibleLLM, Usage
from .memory import (
    FILING, NOTE, PROFILE, QA, Embedder, FastEmbedEmbedder, MemoryRecord, MemoryStore,
    OpenAICompatibleEmbedder, SearchHit,
)
from .models import ITEMS, Filing, FilingRef, Section, normalize_item
from .parsing import html_to_text, split_items
from .sources import EdgarSource, FilingSource

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_SYSTEM_PROMPT", "FILING", "ITEMS", "LLM", "NOTE", "PROFILE", "QA",
    "AnthropicLLM", "Answer", "Chunk", "EdgarSource", "Embedder", "FallbackLog",
    "FastEmbedEmbedder", "Filing", "FilingRef", "FilingSource", "IngestError", "IngestResult",
    "LLMResponse", "MemoryRecord", "MemoryStore", "OpenAICompatibleEmbedder",
    "OpenAICompatibleLLM", "Prompt", "SEC10KAnalyst", "SearchHit", "SearchResult", "Section",
    "Usage", "chunk_filing", "html_to_text", "normalize_item", "split_items", "user_namespace",
]


def __getattr__(name: str):
    if name == "SecApiSource":
        from .sources.secapi import SecApiSource

        return SecApiSource
    raise AttributeError(name)
