from __future__ import annotations

import pytest

from sec10k import Filing, FilingRef, SEC10KAnalyst, split_items, html_to_text
from sec10k.llm.base import LLMResponse, Usage

TOC = "".join(
    f"<tr><td>Item {k}.</td><td>{t}</td><td>{p}</td></tr>"
    for k, t, p in [("1", "Business", 3), ("1A", "Risk Factors", 9), ("3", "Legal Proceedings", 30),
                    ("7", "Management's Discussion", 35), ("9B", "Other Information", 60)]
)


def make_html(year: int = 2024, legal: str = "Smith v. Acme alleges breach of a hosting contract "
              "and seeks damages of $12 million. The case is pending in Delaware.") -> str:
    return f"""<html><head><title>acme-10k</title><style>.x{{color:red}}</style></head><body>
<div style="display:none"><ix:header><div>hidden xbrl junk Item 3. nope</div></ix:header></div>
<p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p><p>FORM 10-K</p>
<p>For the fiscal year ended May 31, {year}</p>
<table>{TOC}</table>
<p>PART I</p>
<p><b>Item 1. Business</b></p>
<p>Acme Corp designs, develops and operates next-generation digital infrastructure for
high-performance computing across North America.</p>
<p>We compete with large hyperscale data center operators.<br/>Our customers include AI labs.</p>
<p><b>Item 1A. Risk Factors</b></p>
<p>Our business depends on access to low-cost power. Rising interest rates may raise our cost
of capital. A highly evolving regulatory landscape may impact our operations.</p>
<p><b>Item 1B. Unresolved Staff Comments</b></p><p>None.</p>
<p><b>Item 3. Legal Proceedings</b></p>
<p>{legal}</p>
<p><b>Item 4. Mine Safety Disclosures</b></p><p>Not applicable.</p>
<p>PART II</p>
<p><b>ITEM 7. MANAGEMENT&#8217;S DISCUSSION AND ANALYSIS</b></p>
<p>Revenue for fiscal {year} increased 92% to $165.6 million, driven by hosting contracts.
Adjusted EBITDA was $32.1 million. As described in Item 1A, power costs matter.</p>
<p><b>Item 9B. Other Information</b></p><p>None.</p>
</body></html>"""


def make_filing(year: int = 2024, accession: str | None = None, **kw) -> Filing:
    ref = FilingRef(ticker="ACME", cik="0000000001", form="10-K",
                    accession=accession or f"0000000001-{year % 100}-000001",
                    filed_at=f"{year}-07-30", period_of_report=f"{year}-05-31",
                    url=f"https://example.test/{year}.htm", company="Acme Corp")
    return Filing(ref=ref, sections=split_items(html_to_text(make_html(year, **kw))), source="test")


class FakeSource:
    name = "fake"

    def __init__(self, filings: list[Filing]):
        self.filings = {f.ref.accession: f for f in filings}
        self.fetches = 0

    def list_filings(self, ticker, form="10-K", start_date=None, end_date=None, limit=10, log=None):
        refs = sorted((f.ref for f in self.filings.values()), key=lambda r: r.filed_at, reverse=True)
        return refs[:limit]

    def fetch(self, ref, log=None):
        self.fetches += 1
        return self.filings[ref.accession]


class FakeLLM:
    model = "fake-1"

    def __init__(self, reply: str = "Yes - Smith v. Acme [E1] (Item 3)."):
        self.reply = reply
        self.calls: list[tuple[str, list]] = []

    def complete(self, system, messages, on_text=None, log=None):
        self.calls.append((system, messages))
        if on_text:
            on_text(self.reply)
        return LLMResponse(self.reply, self.model, Usage(1000, 50, cost_usd=0.001), "end_turn")


@pytest.fixture
def analyst(tmp_path):
    src = FakeSource([make_filing(2024)])
    return SEC10KAnalyst(llm=FakeLLM(), source=src, home=tmp_path)
