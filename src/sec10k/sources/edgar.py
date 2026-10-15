"""Free filing source: SEC EDGAR directly. No API key, no third-party service.

SEC fair-access rules: send a User-Agent that names you and gives a contact email,
and stay under 10 requests/second. https://www.sec.gov/os/accessing-edgar-data
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import httpx

from ..fallbacks import FallbackLog
from ..models import Filing, FilingRef
from ..parsing import html_to_text, split_items

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
DEFAULT_USER_AGENT = "sec10k-library (set SEC10K_USER_AGENT='Your Name you@example.com')"


class EdgarSource:
    name = "edgar"

    def __init__(
        self,
        user_agent: str | None = None,
        cache_dir: str | Path | None = None,
        min_interval: float = 0.15,
        timeout: float = 30.0,
        log: FallbackLog | None = None,
    ) -> None:
        self.log = log or FallbackLog()
        ua = user_agent or os.environ.get("SEC10K_USER_AGENT")
        self._ua_is_default = not ua
        ua = ua or DEFAULT_USER_AGENT
        self._client = httpx.Client(
            headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"},
            timeout=timeout,
            follow_redirects=True,
        )
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._min_interval = min_interval
        self._last_request = 0.0
        self._lock = threading.Lock()
        self._tickers: dict[str, dict] | None = None

    # --- HTTP ----------------------------------------------------------------------

    def _get(self, url: str) -> httpx.Response:
        if self._ua_is_default:  # reported once, when we actually talk to SEC
            self._ua_is_default = False
            self.log.record(
                "default_user_agent",
                "no User-Agent given; SEC may throttle or block requests. "
                "Set SEC10K_USER_AGENT='Your Name you@example.com'",
            )
        with self._lock:
            wait = self._min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
        resp = self._client.get(url)
        resp.raise_for_status()
        return resp

    def _cached_text(self, url: str, key: str) -> str:
        if self.cache_dir:
            path = self.cache_dir / key
            if path.exists():
                return path.read_text()
        text = self._get(url).text
        if self.cache_dir:
            (self.cache_dir / key).write_text(text)
        return text

    # --- lookups -------------------------------------------------------------------

    def cik_for(self, ticker: str) -> tuple[str, str]:
        """Return (10-digit CIK, company name)."""
        if self._tickers is None:
            data = json.loads(self._cached_text(TICKERS_URL, "company_tickers.json"))
            self._tickers = {row["ticker"].upper(): row for row in data.values()}
        row = self._tickers.get(ticker.upper())
        if row is None:
            raise LookupError(f"Ticker {ticker!r} not found in SEC company_tickers.json")
        return f"{int(row['cik_str']):010d}", row["title"]

    def list_filings(
        self,
        ticker: str,
        form: str = "10-K",
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 10,
        log: FallbackLog | None = None,
    ) -> list[FilingRef]:
        cik, company = self.cik_for(ticker)
        data = self._get(SUBMISSIONS_URL.format(cik=cik)).json()
        recent = data["filings"]["recent"]
        refs: list[FilingRef] = []
        for i, f in enumerate(recent["form"]):
            if f != form:
                continue
            filed = recent["filingDate"][i]
            if start_date and filed < start_date:
                continue
            if end_date and filed > end_date:
                continue
            acc = recent["accessionNumber"][i]
            doc = recent["primaryDocument"][i]
            refs.append(
                FilingRef(
                    ticker=ticker.upper(),
                    cik=cik,
                    form=f,
                    accession=acc,
                    filed_at=filed,
                    period_of_report=recent["reportDate"][i] or None,
                    url=ARCHIVE_URL.format(cik=int(cik), acc=acc.replace("-", ""), doc=doc),
                    company=company,
                )
            )
        refs.sort(key=lambda r: r.filed_at, reverse=True)
        # "recent" holds the last ~1000 filings; older ones live in paged files we don't read.
        if not refs and data["filings"].get("files"):
            (log or self.log).record(
                "older_filings_not_searched",
                f"no {form} in recent submissions for {ticker}; older paged history not read",
                ticker=ticker,
            )
        return refs[:limit]

    def fetch(self, ref: FilingRef, log: FallbackLog | None = None) -> Filing:
        log = log or self.log
        if not ref.url:
            raise ValueError(f"FilingRef {ref.accession} has no document URL")
        raw = self._cached_text(ref.url, f"{ref.accession}.htm")
        sections = split_items(html_to_text(raw), log=log)
        return Filing(ref=ref, sections=sections, source=self.name)

    def close(self) -> None:
        self._client.close()
