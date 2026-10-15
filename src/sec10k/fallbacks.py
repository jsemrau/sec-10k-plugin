"""Loud fallback accounting.

Every time the library degrades (a section is missing, a search returns nothing,
an LLM provider re-routes a request, a default User-Agent is used, ...) it calls
``FallbackLog.record``. The event is logged at WARNING level, counted, and kept
so callers can inspect ``trace.counts`` before trusting a result.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("sec10k")


@dataclass
class FallbackEvent:
    kind: str
    reason: str
    context: dict[str, Any] = field(default_factory=dict)
    at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "reason": self.reason, "context": self.context, "at": self.at}


class FallbackLog:
    """Collects fallback events. Pass one through a call chain to get a per-call trace."""

    def __init__(self, parent: FallbackLog | None = None) -> None:
        self.events: list[FallbackEvent] = []
        self.counts: Counter[str] = Counter()
        self._parent = parent

    def record(self, kind: str, reason: str, **context: Any) -> None:
        event = FallbackEvent(kind=kind, reason=reason, context=context)
        self.events.append(event)
        self.counts[kind] += 1
        if self._parent is None:
            logger.warning("fallback[%s]: %s %s", kind, reason, context or "")
        else:
            # The parent logs; children only count, so each event is logged once.
            self._parent.record(kind, reason, **context)

    def child(self) -> FallbackLog:
        """A per-call log whose events also propagate into this (session) log."""
        return FallbackLog(parent=self)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def summary(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "counts": dict(self.counts),
            "events": [e.to_dict() for e in self.events],
        }
