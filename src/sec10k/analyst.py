"""SEC10KAnalyst: fetch a 10-K once, remember it locally, answer questions cheaply.

The LLM never sees the whole filing. A question is answered from the handful of
chunks retrieved for it, plus any notes and user-profile memories. The exact prompt
is available via ``build_prompt`` before you spend anything.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .chunking import chunk_filing
from .fallbacks import FallbackLog
from .llm.base import LLM, Usage
from .memory.base import FILING, NOTE, PROFILE, QA, MemoryRecord, SearchHit
from .memory.store import MemoryStore
from .models import Filing, FilingRef, Section, normalize_item
from .routing import route_weights
from .sources.base import FilingSource

DEFAULT_SYSTEM_PROMPT = """\
You are a buy-side equity analyst reading a company's Form 10-K.
Answer the question using only the filing excerpts and memories provided.
Cite every claim with its excerpt id in brackets, e.g. [E3] (Item 1A).
If the excerpts do not contain the answer, say what is missing instead of guessing,
and name the 10-K Item that would most likely contain it.
Take the user's profile into account when it is relevant (risk tolerance, horizon, holdings).
Be concise and specific: figures, dates, counterparties, amounts."""


class IngestError(RuntimeError):
    pass


def default_home() -> Path:
    return Path(os.environ.get("SEC10K_HOME") or Path.home() / ".sec10k").expanduser()


def user_namespace(user_id: str) -> str:
    return f"user:{user_id}"


@dataclass
class IngestResult:
    ref: FilingRef
    sections: dict[str, int]  # item -> chars
    chunks: int
    superseded: int
    already_ingested: bool
    trace: FallbackLog

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ref.ticker,
            "company": self.ref.company,
            "form": self.ref.form,
            "accession": self.ref.accession,
            "filed_at": self.ref.filed_at,
            "period_of_report": self.ref.period_of_report,
            "url": self.ref.url,
            "sections": self.sections,
            "chunks": self.chunks,
            "superseded_older_chunks": self.superseded,
            "already_ingested": self.already_ingested,
            "fallbacks": self.trace.summary(),
        }


@dataclass
class Prompt:
    system: str
    messages: list[dict[str, str]]
    hits: list[SearchHit]
    memories: list[MemoryRecord]

    @property
    def chars(self) -> int:
        return len(self.system) + sum(len(m["content"]) for m in self.messages)

    @property
    def approx_tokens(self) -> int:
        """Rough estimate (chars / 4). Use your provider's token counter for exact numbers."""
        return self.chars // 4


@dataclass
class Answer:
    text: str
    model: str
    usage: Usage
    citations: list[dict[str, Any]]
    prompt: Prompt
    trace: FallbackLog
    stop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "usage": vars(self.usage),
            "citations": self.citations,
            "prompt_chars": self.prompt.chars,
            "stop_reason": self.stop_reason,
            "fallbacks": self.trace.summary(),
        }


@dataclass
class SearchResult:
    hits: list[SearchHit]
    routed_items: list[str]
    trace: FallbackLog = field(default_factory=FallbackLog)

    def to_dict(self) -> dict[str, Any]:
        return {
            "routed_items": self.routed_items,
            "hits": [h.to_dict() for h in self.hits],
            "fallbacks": self.trace.summary(),
        }


class SEC10KAnalyst:
    """
    >>> analyst = SEC10KAnalyst(llm=AnthropicLLM())         # or OpenAICompatibleLLM(...) or None
    >>> analyst.ingest("APLD")                               # free: SEC EDGAR + local SQLite
    >>> analyst.ask("APLD", "Does the company have legal issues?").text
    """

    def __init__(
        self,
        llm: LLM | None = None,
        source: FilingSource | None = None,
        store: MemoryStore | None = None,
        home: str | Path | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        chunk_chars: int = 1500,
        log: FallbackLog | None = None,
    ) -> None:
        self.home = Path(home).expanduser() if home else default_home()
        self.log = log or FallbackLog()
        self.llm = llm
        if source is None:
            from .sources.edgar import EdgarSource

            source = EdgarSource(cache_dir=self.home / "cache", log=self.log)
        self.source = source
        self.store = store or MemoryStore(self.home / "memory.db", log=self.log)
        self.system_prompt = system_prompt
        self.chunk_chars = chunk_chars
        self.total_usage = Usage(cost_usd=0.0)

    # --- filings -------------------------------------------------------------------

    def _filing_path(self, ticker: str, accession: str) -> Path:
        return self.home / "filings" / ticker.upper() / f"{accession}.json"

    def filings(self, ticker: str, form: str = "10-K", limit: int = 10,
                start_date: str | None = None, end_date: str | None = None) -> list[FilingRef]:
        return self.source.list_filings(ticker, form=form, start_date=start_date,
                                        end_date=end_date, limit=limit, log=self.log)

    def _pick(self, ticker: str, form: str, accession: str | None, fiscal_year: int | None,
              log: FallbackLog) -> FilingRef:
        refs = self.source.list_filings(ticker, form=form, limit=40, log=log)
        if not refs:
            raise IngestError(f"No {form} filings found for {ticker}")
        if accession:
            for r in refs:
                if r.accession == accession:
                    return r
            raise IngestError(f"{ticker}: accession {accession} not among recent {form} filings")
        if fiscal_year:
            for r in refs:
                if r.fiscal_year == fiscal_year:
                    return r
            raise IngestError(
                f"{ticker}: no {form} for fiscal year {fiscal_year}; available: "
                + ", ".join(str(r.fiscal_year) for r in refs)
            )
        return refs[0]

    def ingest(
        self,
        ticker: str,
        accession: str | None = None,
        fiscal_year: int | None = None,
        form: str = "10-K",
        force: bool = False,
        keep_older: bool = False,
    ) -> IngestResult:
        """Fetch (latest by default) filing, split into Items, chunk and store it.

        Older filings of the same ticker are marked superseded (hidden from search
        but kept) unless ``keep_older=True``. Re-ingesting is a no-op unless ``force``.
        """
        trace = self.log.child()
        ref = self._pick(ticker.upper(), form, accession, fiscal_year, trace)
        ns = ref.ticker
        path = self._filing_path(ns, ref.accession)
        existing = self.store.list(ns, kinds=[FILING], accession=ref.accession,
                                   include_superseded=True, limit=1)
        if existing and path.exists() and not force:
            filing = Filing.load(path)
            n = len(self.store.list(ns, kinds=[FILING], accession=ref.accession, include_superseded=True))
            return IngestResult(ref, {k: s.chars for k, s in filing.sections.items()}, n, 0, True, trace)
        filing = self.source.fetch(ref, log=trace)
        return self.ingest_filing(filing, keep_older=keep_older, trace=trace)

    def ingest_filing(self, filing: Filing, keep_older: bool = False,
                      trace: FallbackLog | None = None) -> IngestResult:
        """Store an already-fetched Filing (e.g. from your own source or a JSON file)."""
        trace = trace or self.log.child()
        ref = filing.ref
        if not filing.sections:
            raise IngestError(f"{ref.ticker} {ref.accession}: no sections could be extracted")
        ns = ref.ticker.upper()
        path = self._filing_path(ns, ref.accession)
        path.parent.mkdir(parents=True, exist_ok=True)
        filing.save(path)

        self.store.forget(ns, kind=FILING, accession=ref.accession)
        records = [
            MemoryRecord(
                namespace=ns,
                kind=FILING,
                text=c.text,
                metadata={
                    "item": c.item, "chunk": c.index, "accession": ref.accession,
                    "form": ref.form, "filed_at": ref.filed_at,
                    "fiscal_year": ref.fiscal_year, "url": ref.url,
                },
            )
            for c in chunk_filing(filing, max_chars=self.chunk_chars)
        ]
        self.store.add(records, log=trace)
        superseded = 0
        if not keep_older:
            # The latest *filed* 10-K stays active, even when an older one is ingested later.
            dates = {r.accession: r.metadata.get("filed_at") or ""
                     for r in self.store.list(ns, kinds=[FILING], include_superseded=True)}
            latest = max(dates, key=lambda acc: dates[acc])
            superseded = self.store.supersede(ns, FILING, latest)
            if latest != ref.accession:
                trace.record("ingested_filing_superseded",
                             f"{ref.accession} is older than {latest}; stored but hidden from "
                             "search (use include_superseded=True or keep_older=True)",
                             ticker=ns)
        return IngestResult(ref, {k: s.chars for k, s in filing.sections.items()},
                            len(records), superseded, False, trace)

    def load_filing(self, ticker: str, accession: str | None = None) -> Filing:
        """Load a previously ingested filing from disk (latest non-superseded by default)."""
        ns = ticker.upper()
        if accession is None:
            recs = self.store.list(ns, kinds=[FILING])
            if not recs:
                raise IngestError(f"{ns} has not been ingested; call ingest({ns!r}) first")
            accession = max(recs, key=lambda r: r.metadata.get("filed_at") or "").accession
        path = self._filing_path(ns, accession)
        if not path.exists():
            raise IngestError(f"{ns} {accession}: filing JSON missing at {path}; re-ingest")
        return Filing.load(path)

    def section(self, ticker: str, item: str, accession: str | None = None) -> Section:
        filing = self.load_filing(ticker, accession)
        key = normalize_item(item)
        sec = filing.sections.get(key)
        if sec is None:
            raise KeyError(f"{ticker}: Item {key} was not found in filing {filing.ref.accession}")
        return sec

    # --- memory --------------------------------------------------------------------

    def remember(self, namespace: str, text: str, kind: str = NOTE,
                 ttl_days: float | None = None, **metadata: Any) -> MemoryRecord:
        """Store a fact. namespace = ticker for company notes, user_namespace(id) for profile."""
        expires = None
        if ttl_days is not None:
            expires = (datetime.now(timezone.utc) + timedelta(days=ttl_days)).isoformat()
        rec = MemoryRecord(namespace=namespace if namespace.startswith("user:") else namespace.upper(),
                           kind=kind, text=text, metadata=metadata, expires_at=expires)
        self.store.add([rec])
        return rec

    def remember_profile(self, user_id: str, text: str, **metadata: Any) -> MemoryRecord:
        return self.remember(user_namespace(user_id), text, kind=PROFILE, **metadata)

    def recall(self, namespace: str, query: str | None = None, kinds: Sequence[str] | None = None,
               k: int = 10) -> list[MemoryRecord]:
        ns = namespace if namespace.startswith("user:") else namespace.upper()
        kinds = list(kinds) if kinds else [NOTE, QA, PROFILE]
        if query:
            return [h.record for h in self.store.search(ns, query, k=k, kinds=kinds)]
        return self.store.list(ns, kinds=kinds)[-k:]

    def forget(self, namespace: str, kind: str | None = None, accession: str | None = None,
               ids: Sequence[str] | None = None, superseded_only: bool = False) -> int:
        ns = namespace if namespace.startswith("user:") else namespace.upper()
        if kind in (None, FILING) and not ids:
            # Keep filing JSON in sync with the store when whole filings are dropped.
            accs = {r.accession for r in self.store.list(ns, kinds=[FILING], accession=accession,
                                                         include_superseded=True)
                    if not superseded_only or r.superseded_at}
        else:
            accs = set()
        n = self.store.forget(ns, kind=kind, accession=accession, ids=ids,
                              superseded_only=superseded_only)
        for acc in accs:
            self._filing_path(ns, acc).unlink(missing_ok=True)
        return n

    # --- retrieval -----------------------------------------------------------------

    def search(self, ticker: str, query: str, k: int = 8, items: Sequence[str] | None = None,
               include_superseded: bool = False, trace: FallbackLog | None = None) -> SearchResult:
        trace = trace or self.log.child()
        ns = ticker.upper()
        hard = [normalize_item(i) for i in items] if items else None
        weights = route_weights(query)
        routed = list(weights)
        if self.store.count(ns, kinds=[FILING]) == 0:
            trace.record("not_ingested", f"{ns} has no ingested filing; search returns nothing",
                         ticker=ns)
            return SearchResult([], routed, trace)
        hits = self.store.search(ns, query, k=k, kinds=[FILING], items=hard, boost_items=weights,
                                 include_superseded=include_superseded, log=trace)
        if not hits:
            fallback_items = hard or routed or ["1", "7"]
            trace.record(
                "no_retrieval_hits",
                f"no chunk matched {query!r}; using opening chunks of Items {fallback_items}",
                ticker=ns,
            )
            per = max(1, k // len(fallback_items))
            for it in fallback_items:
                recs = self.store.list(ns, kinds=[FILING], items=[it], limit=per)
                hits.extend(SearchHit(r, 0.0) for r in recs)
        return SearchResult(hits[:k], routed, trace)

    def build_prompt(self, ticker: str, question: str, user_id: str | None = None, k: int = 8,
                     items: Sequence[str] | None = None, trace: FallbackLog | None = None) -> Prompt:
        trace = trace or self.log.child()
        result = self.search(ticker, question, k=k, items=items, trace=trace)
        ns = ticker.upper()
        memories = [h.record for h in self.store.search(ns, question, k=5, kinds=[NOTE, QA])]
        profile = self.store.list(user_namespace(user_id), kinds=[PROFILE]) if user_id else []

        parts: list[str] = []
        if profile:
            parts.append("<user_profile>\n" + "\n".join(f"- {m.text}" for m in profile)
                         + "\n</user_profile>")
        if memories:
            parts.append("<memories>\n" + "\n".join(f"- ({m.kind}, {m.created_at[:10]}) {m.text}"
                                                    for m in memories) + "\n</memories>")
        excerpts = "\n\n".join(
            f'<excerpt id="E{i}" item="{h.record.item}">\n{h.record.text}\n</excerpt>'
            for i, h in enumerate(result.hits, 1)
        )
        parts.append(f"<filing_excerpts>\n{excerpts}\n</filing_excerpts>")
        parts.append(f"Question about {ns}: {question}")
        return Prompt(
            system=self.system_prompt,
            messages=[{"role": "user", "content": "\n\n".join(parts)}],
            hits=result.hits,
            memories=profile + memories,
        )

    def ask(
        self,
        ticker: str,
        question: str,
        user_id: str | None = None,
        k: int = 8,
        items: Sequence[str] | None = None,
        remember_answer: bool = True,
        on_text: Callable[[str], None] | None = None,
        llm: LLM | None = None,
    ) -> Answer:
        llm = llm or self.llm
        if llm is None:
            raise ValueError("No LLM configured. Pass llm= or use search()/build_prompt() "
                             "and let your own agent (e.g. Claude via MCP) do the reasoning.")
        trace = self.log.child()
        prompt = self.build_prompt(ticker, question, user_id=user_id, k=k, items=items, trace=trace)
        resp = llm.complete(prompt.system, prompt.messages, on_text=on_text, log=trace)
        self.total_usage = self.total_usage + resp.usage  # cost becomes None once any is unknown
        citations = [
            {"id": f"E{i}", "item": h.record.item, "accession": h.record.accession,
             "chunk": h.record.metadata.get("chunk"), "url": h.record.metadata.get("url")}
            for i, h in enumerate(prompt.hits, 1)
        ]
        if remember_answer and resp.text and resp.stop_reason != "refusal":
            self.remember(ticker, f"Q: {question}\nA: {resp.text[:2000]}", kind=QA,
                          model=resp.model, user_id=user_id)
        return Answer(resp.text, resp.model, resp.usage, citations, prompt, trace, resp.stop_reason)

    def fallback_report(self) -> dict[str, Any]:
        """Session-wide fallback counts. Check this before trusting results."""
        return self.log.summary()
