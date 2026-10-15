"""MCP server exposing sec10k to Claude (Claude Code plugin, Claude Desktop, any MCP client).

No LLM runs in here: the server fetches, stores and retrieves; Claude does the reasoning
with only the excerpts it asks for. That keeps cost at "the tokens Claude reads".

Run: ``sec10k-mcp`` (stdio). Env: SEC10K_USER_AGENT (required by SEC), SEC10K_HOME.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from .analyst import SEC10KAnalyst, user_namespace
from .memory.base import NOTE, PROFILE, QA
from .models import ITEMS

INSTRUCTIONS = """\
Tools for reading SEC Form 10-K filings with persistent local memory.
Workflow: ingest_10k(ticker) once -> search_10k for targeted excerpts -> get_10k_section
for full text of one Item (paged). Never try to read the whole filing; search first.
Cite answers as (TICKER 10-K FY, Item N). Store durable facts with remember(); store the
user's investment profile with remember(namespace="user:<id>", kind="profile").
Every result carries a `fallbacks` block - if total > 0, mention what degraded."""

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

server = _Server("sec10k", instructions=INSTRUCTIONS)
_analyst: SEC10KAnalyst | None = None


def analyst() -> SEC10KAnalyst:
    global _analyst
    if _analyst is None:
        _analyst = SEC10KAnalyst(llm=None)
    return _analyst


def _ns(namespace: str) -> str:
    return namespace if namespace.startswith("user:") else namespace.upper()


@server.tool()
def list_10k_filings(ticker: str, limit: int = 5, form: str = "10-K") -> dict[str, Any]:
    """List a company's most recent 10-K filings on SEC EDGAR (newest first)."""
    a = analyst()
    trace = a.log.child()
    refs = a.source.list_filings(ticker, form=form, limit=limit, log=trace)
    return {
        "filings": [
            {"accession": r.accession, "filed_at": r.filed_at, "period_of_report": r.period_of_report,
             "fiscal_year": r.fiscal_year, "company": r.company, "url": r.url}
            for r in refs
        ],
        "fallbacks": trace.summary(),
    }


@server.tool()
def ingest_10k(ticker: str, fiscal_year: int | None = None, accession: str | None = None,
               force: bool = False, keep_older: bool = False) -> dict[str, Any]:
    """Download a 10-K from SEC EDGAR (latest unless fiscal_year/accession given), split it
    into Items, and index it in local memory. Free; idempotent unless force=True. Older
    filings of the ticker become 'superseded' (hidden from search) unless keep_older=True."""
    return analyst().ingest(ticker, accession=accession, fiscal_year=fiscal_year,
                            force=force, keep_older=keep_older).to_dict()


@server.tool()
def search_10k(ticker: str, query: str, k: int = 8, items: list[str] | None = None,
               include_superseded: bool = False) -> dict[str, Any]:
    """Search an ingested 10-K and return the k most relevant excerpts with their Item.
    items: optional hard filter, e.g. ["3", "1A"]. Without it, the query wording softly
    boosts likely Items (legal -> 3/1A, revenue -> 7/8, ...)."""
    res = analyst().search(ticker, query, k=k, items=items, include_superseded=include_superseded)
    out = res.to_dict()
    for h in out["hits"]:
        h.pop("created_at", None)
        h.pop("expires_at", None)
    return out


@server.tool()
def get_10k_section(ticker: str, item: str, offset: int = 0, max_chars: int = 20000,
                    accession: str | None = None) -> dict[str, Any]:
    """Return the text of one 10-K Item (e.g. "1A", "7"), paged by offset/max_chars."""
    sec = analyst().section(ticker, item, accession=accession)
    chunk = sec.text[offset:offset + max_chars]
    nxt = offset + len(chunk)
    return {
        "item": sec.item, "title": sec.title, "total_chars": sec.chars, "offset": offset,
        "text": chunk, "next_offset": nxt if nxt < sec.chars else None,
    }


@server.tool()
def list_10k_items() -> dict[str, str]:
    """The standard Form 10-K Items and their titles."""
    return ITEMS


@server.tool()
def remember(namespace: str, text: str, kind: str = NOTE, ttl_days: float | None = None) -> dict[str, Any]:
    """Store a durable memory. namespace: a ticker (company facts/notes) or "user:<id>"
    (the user's investment profile, preferences). kind: note | profile | qa.
    ttl_days makes it expire (e.g. for time-sensitive views)."""
    if kind not in (NOTE, PROFILE, QA):
        raise ValueError(f"kind must be one of {NOTE}, {PROFILE}, {QA}")
    rec = analyst().remember(namespace, text, kind=kind, ttl_days=ttl_days)
    return rec.to_dict()


@server.tool()
def recall(namespace: str, query: str | None = None, k: int = 10) -> list[dict[str, Any]]:
    """Retrieve notes/profile/Q&A memories for a ticker or "user:<id>" (search if query given)."""
    return [r.to_dict() for r in analyst().recall(namespace, query=query, k=k)]


@server.tool()
def forget(namespace: str, kind: str | None = None, accession: str | None = None,
           ids: list[str] | None = None, superseded_only: bool = False) -> dict[str, int]:
    """Permanently delete memories. Narrow with kind (filing|note|qa|profile), accession,
    ids, or superseded_only=True (drop only replaced older filings)."""
    n = analyst().forget(namespace, kind=kind, accession=accession, ids=ids,
                         superseded_only=superseded_only)
    return {"deleted": n}


@server.tool()
def memory_overview() -> dict[str, Any]:
    """What is stored: record counts per namespace and kind, plus session fallback counts."""
    a = analyst()
    return {"namespaces": a.store.namespaces(), "home": str(a.home),
            "fallbacks": {"total": a.log.total, "counts": dict(a.log.counts)}}


def main() -> None:
    # stdout carries the MCP protocol; logs (incl. fallback warnings) go to stderr.
    logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    server.run()


if __name__ == "__main__":
    main()
