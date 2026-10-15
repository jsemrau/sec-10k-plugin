"""Data model for 10-K filings and their sections."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Form 10-K items as defined by the SEC (https://www.sec.gov/files/reada10k.pdf).
ITEMS: dict[str, str] = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity, Related Stockholder Matters and "
    "Issuer Purchases of Equity Securities",
    "6": "[Reserved] / Selected Financial Data",
    "7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants on Accounting and Financial Disclosure",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management and "
    "Related Stockholder Matters",
    "13": "Certain Relationships and Related Transactions, and Director Independence",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}


# Pseudo-item for filing text outside any Item (see parsing.split_items).
OTHER_ITEM = "X"


def normalize_item(item: str) -> str:
    """'item 1a' / '1a' / 'section_1a' -> '1A'. Raises KeyError for unknown items."""
    key = item.strip().upper().replace("ITEM", "").replace("SECTION_", "").replace(" ", "")
    key = key.rstrip(".")
    if key not in ITEMS and key != OTHER_ITEM:
        raise KeyError(f"Unknown 10-K item {item!r}; valid items: {', '.join(ITEMS)}")
    return key


@dataclass
class FilingRef:
    """Metadata identifying one filing on EDGAR."""

    ticker: str
    cik: str
    form: str
    accession: str  # e.g. 0001144879-24-000216
    filed_at: str  # YYYY-MM-DD
    period_of_report: str | None = None  # YYYY-MM-DD
    url: str | None = None  # primary document URL
    company: str | None = None

    @property
    def fiscal_year(self) -> int | None:
        date = self.period_of_report or self.filed_at
        return int(date[:4]) if date else None


@dataclass
class Section:
    item: str
    title: str
    text: str

    @property
    def chars(self) -> int:
        return len(self.text)


@dataclass
class Filing:
    ref: FilingRef
    sections: dict[str, Section] = field(default_factory=dict)
    source: str = "unknown"

    def section(self, item: str) -> Section | None:
        return self.sections.get(normalize_item(item))

    def missing_items(self) -> list[str]:
        return [k for k in ITEMS if k not in self.sections or not self.sections[k].text.strip()]

    # --- serialization -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": asdict(self.ref),
            "source": self.source,
            "sections": {k: asdict(s) for k, s in self.sections.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Filing:
        return cls(
            ref=FilingRef(**data["ref"]),
            source=data.get("source", "unknown"),
            sections={k: Section(**s) for k, s in data["sections"].items()},
        )

    def to_notebook_dict(self) -> dict[str, dict[str, str]]:
        """The ``{"section_1a": {"Risk Factors": "..."}}`` shape the original notebook wrote."""
        return {f"section_{k.lower()}": {s.title: s.text} for k, s in self.sections.items()}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> Filing:
        return cls.from_dict(json.loads(Path(path).read_text()))
