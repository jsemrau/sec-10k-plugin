"""HTML -> text conversion and splitting a 10-K into its Items.

Only the standard library is used, so the parser works anywhere.
"""

from __future__ import annotations

import bisect
import html
import re
from html.parser import HTMLParser

from .fallbacks import FallbackLog
from .models import ITEMS, OTHER_ITEM, Section

_BLOCK_TAGS = {
    "p", "div", "br", "tr", "li", "ul", "ol", "table", "h1", "h2", "h3", "h4", "h5", "h6",
    "section", "article", "header", "footer", "hr", "title", "blockquote", "pre",
}
_SKIP_TAGS = {"script", "style", "head", "ix:header", "noscript"}
_VOID_TAGS = {"br", "img", "hr", "meta", "input", "link", "col", "area", "base", "wbr"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if self._skip_depth:
            if tag not in _VOID_TAGS:
                self._skip_depth += 1
            return
        style = (dict(attrs).get("style") or "").replace(" ", "").lower()
        if tag in _SKIP_TAGS or "display:none" in style:
            # Scripts, <head>, and hidden inline-XBRL blocks carry no readable text.
            self._skip_depth = 1
            return
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # <br/> etc.: no matching end tag, so it must not touch the skip depth.
        if not self._skip_depth and tag.lower() in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._skip_depth:
            self._skip_depth -= 1
            return
        if tag.lower() in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def html_to_text(raw_html: str) -> str:
    parser = _TextExtractor()
    parser.feed(raw_html)
    parser.close()
    text = html.unescape("".join(parser.parts))
    text = text.replace("\xa0", " ").replace("​", "")
    lines = [re.sub(r"[ \t\r\f\v]+", " ", ln).strip(" |") for ln in text.split("\n")]
    out: list[str] = []
    for ln in lines:
        ln = ln.strip()
        if ln or (out and out[-1]):
            out.append(ln)
    return "\n".join(out).strip()


# Longer keys first so "1A" wins over "1" and "10" over "1".
_ITEM_KEYS = sorted(ITEMS, key=lambda k: (-len(k), k))
_ITEM_RE = re.compile(
    r"^\s*item\s*(" + "|".join(_ITEM_KEYS) + r")(?![0-9a-z])\s*[\.:\-–—]?\s*(.*)$",
    re.IGNORECASE | re.MULTILINE,
)

# "PART II" and "SIGNATURES" headings also end the preceding Item.
_BOUNDARY_RE = re.compile(r"^\s*(?:part\s+(?:iv|iii|ii|i)\b[^\n]{0,80}|signatures?)\s*$",
                          re.IGNORECASE | re.MULTILINE)
# Running page headers/footers inside a section: "PART II", "Item 7", "Item 1B, 1C", "34".
_NOISE_LINE_RE = re.compile(
    r"^\s*(?:part\s+(?:iv|iii|ii|i)|item\s+[0-9]{1,2}[a-c]?(?:\s*,\s*[0-9]{1,2}[a-c]?)*|\d{1,3})\s*$",
    re.IGNORECASE | re.MULTILINE,
)

OTHER_TITLE = "Text outside the Item sections (e.g. an integrated annual report)"


def _clean(body: str) -> str:
    body = _NOISE_LINE_RE.sub("", body)
    return re.sub(r"\n{3,}", "\n\n", body).strip()


def split_items(text: str, log: FallbackLog | None = None,
                min_coverage: float = 0.5) -> dict[str, Section]:
    """Split filing text into Items.

    A 10-K mentions each Item heading at least twice (table of contents, then the
    body), and many filers repeat "PART II / Item 7" as a running page header.
    We walk the headings in order: a heading for the Item already open continues
    it; a different Item closes it (at a preceding PART/SIGNATURES line if there is
    one). Per Item the *longest* body wins, so TOC entries lose.

    If the chosen Items cover less than ``min_coverage`` of the text (filers like
    banks put MD&A and financials in an integrated annual report and cross-reference
    it), the remainder is kept as pseudo-Item ``"X"`` and a fallback is recorded.
    """
    log = log or FallbackLog()
    events = sorted(
        [(m.start(), m.end(), m.group(1).upper(), m.group(2).strip()) for m in _ITEM_RE.finditer(text)]
        + [(m.start(), m.end(), None, "") for m in _BOUNDARY_RE.finditer(text)]
    )
    segments: list[tuple[str, int, int, str]] = []  # key, start, end, first-line remainder
    cur: tuple[str, int, str] | None = None  # key, body start, heading remainder
    pending_stop: int | None = None
    for start, end, key, rest in events:
        if key is None:
            if cur is not None and pending_stop is None:
                pending_stop = start
            continue
        if cur is not None and key == cur[0]:
            pending_stop = None  # running header, the section continues
            continue
        if cur is not None:
            segments.append((cur[0], cur[1], pending_stop if pending_stop is not None else start, cur[2]))
        cur, pending_stop = (key, end, rest), None
    if cur is not None:
        segments.append((cur[0], cur[1], pending_stop if pending_stop is not None else len(text), cur[2]))

    best: dict[str, tuple[str, int, int]] = {}
    for key, start, stop, rest in segments:
        following = _clean(text[start:stop])
        # Body text sometimes shares the heading's line ("Item 3. Legal Proceedings. We are ...").
        body = f"{rest}\n{following}".strip() if len(rest) > 120 else following
        if key not in best or len(body) > len(best[key][0]):
            best[key] = (body, start, stop)

    sections = {key: Section(item=key, title=title, text=best[key][0])
                for key, title in ITEMS.items() if key in best}

    if not events or not best:
        log.record("no_item_headings", "no 'Item N.' headings found; filing could not be split")
    for key in ITEMS:
        if key not in sections:
            log.record("section_missing", f"Item {key} heading not found", item=key)

    covered = sum(stop - start for _, start, stop in best.values())
    if text and covered / len(text) < min_coverage:
        spans = sorted((start, stop) for _, start, stop in best.values())
        rest_parts, pos = [], 0
        for start, stop in spans:
            if start > pos:
                rest_parts.append(text[pos:start])
            pos = max(pos, stop)
        rest_parts.append(text[pos:])
        other = _clean("\n".join(rest_parts))
        if other:
            sections[OTHER_ITEM] = Section(item=OTHER_ITEM, title=OTHER_TITLE, text=other)
        log.record(
            "low_item_coverage",
            f"Item sections cover only {covered / len(text):.0%} of the filing text "
            f"(likely cross-referenced to an annual report); the rest is kept as Item "
            f"{OTHER_ITEM!r}",
            coverage=round(covered / len(text), 3),
        )
    return sections
