"""Optional source backed by sec-api.io (the service the original notebook used).

Requires ``pip install sec10k[secapi]`` and an API key (SEC_API_KEY).
Each section is one ExtractorApi call, so a full filing costs ~23 calls.
"""

from __future__ import annotations

import os

from ..fallbacks import FallbackLog
from ..models import ITEMS, Filing, FilingRef, Section


class SecApiSource:
    name = "sec-api"

    def __init__(self, api_key: str | None = None, log: FallbackLog | None = None) -> None:
        try:
            from sec_api import ExtractorApi, QueryApi
        except ImportError as e:  # pragma: no cover - depends on optional extra
            raise ImportError("SecApiSource needs `pip install sec10k[secapi]`") from e
        key = api_key or os.environ.get("SEC_API_KEY")
        if not key:
            raise ValueError("SecApiSource needs api_key= or SEC_API_KEY")
        self._query = QueryApi(api_key=key)
        self._extractor = ExtractorApi(key)
        self.log = log or FallbackLog()

    def list_filings(
        self,
        ticker: str,
        form: str = "10-K",
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 10,
        log: FallbackLog | None = None,
    ) -> list[FilingRef]:
        q = f'ticker:{ticker} AND formType:"{form}"'
        if start_date or end_date:
            q += f" AND filedAt:[{start_date or '1990-01-01'} TO {end_date or '2100-01-01'}]"
        resp = self._query.get_filings(
            {"query": q, "from": "0", "size": str(limit), "sort": [{"filedAt": {"order": "desc"}}]}
        )
        return [
            FilingRef(
                ticker=ticker.upper(),
                cik=f"{int(f['cik']):010d}",
                form=f["formType"],
                accession=f["accessionNo"],
                filed_at=f["filedAt"][:10],
                period_of_report=(f.get("periodOfReport") or None),
                url=f["linkToFilingDetails"],
                company=f.get("companyName"),
            )
            for f in resp.get("filings", [])
        ]

    def fetch(self, ref: FilingRef, log: FallbackLog | None = None) -> Filing:
        log = log or self.log
        sections: dict[str, Section] = {}
        for key, title in ITEMS.items():
            try:
                text = self._extractor.get_section(ref.url, key.lower(), "text")
            except Exception as e:  # sec-api raises plain Exceptions
                log.record("section_fetch_failed", str(e), item=key, accession=ref.accession)
                continue
            if not text or not text.strip():
                log.record("section_missing", f"sec-api returned no text for Item {key}", item=key)
                continue
            sections[key] = Section(item=key, title=title, text=text.strip())
        return Filing(ref=ref, sections=sections, source=self.name)
