from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..fallbacks import FallbackLog
from ..models import Filing, FilingRef


@runtime_checkable
class FilingSource(Protocol):
    """Anything that can list and fetch 10-K filings."""

    name: str

    def list_filings(
        self,
        ticker: str,
        form: str = "10-K",
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 10,
        log: FallbackLog | None = None,
    ) -> list[FilingRef]:
        """Newest first."""
        ...

    def fetch(self, ref: FilingRef, log: FallbackLog | None = None) -> Filing:
        """Download one filing and split it into Items."""
        ...
