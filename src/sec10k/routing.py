"""Map question wording to the 10-K Items most likely to answer it.

Used as a soft boost in search, never as a hard filter, so a miss costs little.
"""

from __future__ import annotations

import re

ROUTES: list[tuple[str, tuple[str, ...]]] = [
    (r"legal|lawsuit|litigation|sued|court|proceeding|settlement|subpoena|investigation", ("3", "1A")),
    (r"risk|threat|uncertaint|exposure|downside", ("1A", "7A")),
    (r"interest rate|currency|fx|foreign exchange|commodity|hedg|market risk", ("7A",)),
    (r"cyber|security incident|breach|ransomware|information security", ("1C", "1A")),
    (r"revenue|margin|profit|loss|earnings|ebitda|cash flow|liquidity|capital|debt|guidance|"
     r"segment|outlook|results of operations", ("7", "8")),
    (r"balance sheet|income statement|financial statement|footnote|accounting polic", ("8",)),
    (r"business|product|customer|competit|strategy|employee|human capital|what does .* do", ("1",)),
    (r"propert|facilit|data center|real estate|lease|site", ("2",)),
    (r"dividend|buyback|repurchase|share price|stock price|holders of record", ("5",)),
    (r"auditor|audit fee|accountant", ("14", "9")),
    (r"internal control|material weakness|disclosure control|sox", ("9A",)),
    (r"compensation|salary|bonus|pay\b|stock award|ceo pay", ("11",)),
    (r"director|board|governance|officer|management team|executive", ("10",)),
    (r"ownership|shareholder|beneficial owner|insider", ("12",)),
    (r"related part|related transaction|independen", ("13",)),
    (r"sec comment|staff comment", ("1B",)),
    (r"mine safety", ("4",)),
    (r"exhibit", ("15",)),
]


def route(question: str) -> list[str]:
    """Items the question is likely about, most specific first."""
    return list(route_weights(question))


def route_weights(question: str) -> dict[str, float]:
    """Item -> score multiplier. The first Item of a matching route is primary (x1.6),
    the rest secondary (x1.25)."""
    q = question.lower()
    weights: dict[str, float] = {}
    for pattern, targets in ROUTES:
        if re.search(pattern, q):
            for i, t in enumerate(targets):
                weights[t] = max(weights.get(t, 0.0), 1.6 if i == 0 else 1.25)
    return dict(sorted(weights.items(), key=lambda kv: -kv[1]))
