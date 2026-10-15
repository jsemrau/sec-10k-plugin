from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

# kinds of memory
FILING = "filing"  # a chunk of a filing (structured: item, accession, fiscal year)
NOTE = "note"  # a free-text fact you (or an agent) chose to remember
QA = "qa"  # a past question and its answer
PROFILE = "profile"  # facts about the user: investment profile, preferences


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryRecord:
    namespace: str  # ticker for filing data, "user:<id>" for user profiles
    kind: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(default_factory=_now)
    superseded_at: str | None = None  # set when a newer filing replaces this one
    expires_at: str | None = None  # ISO timestamp; expired records are not returned

    @property
    def item(self) -> str | None:
        return self.metadata.get("item")

    @property
    def accession(self) -> str | None:
        return self.metadata.get("accession")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "namespace": self.namespace,
            "kind": self.kind,
            "text": self.text,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "superseded_at": self.superseded_at,
            "expires_at": self.expires_at,
        }


@dataclass
class SearchHit:
    record: MemoryRecord
    score: float
    lexical_rank: int | None = None
    vector_rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = self.record.to_dict()
        d.update(score=round(self.score, 6), lexical_rank=self.lexical_rank,
                 vector_rank=self.vector_rank)
        return d


@runtime_checkable
class Embedder(Protocol):
    """Bring your own embeddings: local model, Ollama, OpenAI, Voyage, ..."""

    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...
