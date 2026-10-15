---
name: sec-10k-analyst
description: Analyze a public company's SEC Form 10-K (annual report) - legal proceedings, risk factors, MD&A, financials, governance, compensation - using the sec10k MCP tools with persistent memory. Use when the user asks about a US-listed company's 10-K, annual report, SEC filing, or a ticker's risks, lawsuits, revenue drivers, auditor, or disclosures.
---

# SEC 10-K analyst

You have the `sec10k` MCP tools. They fetch filings from SEC EDGAR (free), keep them in a
local SQLite memory, and return only the excerpts you ask for. A 10-K is 300-500K characters;
never try to read all of it. Search, then read the one Item you need.

## Workflow

1. **Ingest once.** `ingest_10k(ticker)` gets the latest 10-K (or `fiscal_year=` /
   `accession=`). It is idempotent, so call it whenever you are unsure. Check the returned
   `sections` sizes: a tiny Item (under ~200 chars) is usually "None" or "incorporated by
   reference to the proxy statement"; say so instead of guessing. Many Items just point
   elsewhere ("see Note 12 in Item 15"): follow the pointer with `search_10k` on that Item.
   If the result has a `low_item_coverage` fallback (common for banks), most content is in
   pseudo-Item `"X"` (an integrated annual report). Search it with `items=["X"]`.
2. **Recall context.** `recall("user:<id>")` for the user's investment profile and
   `recall(ticker, query)` for earlier notes and answers. Use the profile to frame the answer
   (risk tolerance, horizon, positions).
3. **Search narrowly.** `search_10k(ticker, query, k=6-10)`. Pass `items=[...]` when you know
   where the answer lives:

   | Question about | Items |
   |---|---|
   | Lawsuits, investigations | 3, 1A, and the "Contingencies" note (in 8, or 15 when statements are in F-pages) |
   | Risks | 1A (company), 7A (market: rates, FX, commodities) |
   | Growth, margins, liquidity, guidance | 7, then 8 for exact figures |
   | Business model, customers, competition | 1 |
   | Cybersecurity | 1C |
   | Controls, material weaknesses | 9A |
   | Auditor and fees | 14 (often in the proxy), 9 |
   | Pay, board, ownership | 11, 10, 12 (often in the proxy) |

4. **Read in full when it matters.** `get_10k_section(ticker, item, offset, max_chars)` pages
   through one Item. Use it for short Items (3, 1C, 9A) or when the excerpts are cut off.
5. **Answer with citations**, e.g. "(APLD 10-K FY2026, Item 3)". Give figures, dates,
   counterparties and amounts. If the filing doesn't say, say that.
6. **Remember what's durable.** `remember(ticker, text)` for conclusions worth keeping (for
   example "Securities class action McConnell v. APLD stayed Sep 2025 pending MTD"). Use
   `remember("user:<id>", text, kind="profile")` when the user tells you about their
   portfolio or preferences. Use `ttl_days` for views that go stale.

## Freshness and forgetting

- When a newer 10-K is ingested, older chunks are marked superseded and hidden from search.
  To compare years, call `ingest_10k(ticker, fiscal_year=YYYY, keep_older=True)` for each
  year, then `search_10k(..., include_superseded=True)`.
- `forget(namespace, ...)` deletes data permanently. Only call it when the user asks.

## Fallbacks

Every tool result has a `fallbacks` block. If `total > 0`, tell the user what degraded,
for example `section_missing` (Item not found in the filing), `no_retrieval_hits` (generic
excerpts were used), or `default_user_agent` (SEC may throttle; set `SEC10K_USER_AGENT`).
Don't present a degraded result as if it were a normal one.
