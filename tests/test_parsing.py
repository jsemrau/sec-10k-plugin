from sec10k import FallbackLog, chunk_filing, html_to_text, normalize_item, split_items
from sec10k.chunking import chunk_text

from .conftest import make_filing, make_html


def test_html_to_text_drops_hidden_and_styles():
    text = html_to_text(make_html())
    assert "hidden xbrl junk" not in text
    assert "color:red" not in text
    assert "MANAGEMENT’S DISCUSSION" in text


def test_split_items_picks_body_not_toc():
    log = FallbackLog()
    secs = split_items(html_to_text(make_html()), log=log)
    assert secs["3"].title == "Legal Proceedings"
    assert "Smith v. Acme" in secs["3"].text
    assert secs["1"].text.startswith("Acme Corp designs")
    assert "Revenue for fiscal 2024" in secs["7"].text
    # "Item 1A" inside a sentence of Item 7 must not start a new section
    assert "power costs matter" in secs["7"].text
    assert secs["9B"].text == "None."
    assert secs["1B"].text == "None."


def test_missing_items_are_counted_loudly():
    log = FallbackLog()
    split_items(html_to_text(make_html()), log=log)
    missing = {e.context["item"] for e in log.events if e.kind == "section_missing"}
    assert "8" in missing and "1C" in missing and "3" not in missing
    assert log.counts["section_missing"] == len(missing)


def test_no_headings():
    log = FallbackLog()
    secs = split_items("just some text", log=log)
    assert list(secs) == ["X"] and secs["X"].text == "just some text"  # still searchable
    assert log.counts["no_item_headings"] == 1 and log.counts["low_item_coverage"] == 1


def test_normalize_item():
    assert normalize_item("item 1a") == "1A"
    assert normalize_item("section_9b") == "9B"
    assert normalize_item("7A.") == "7A"


def test_chunks_carry_context_header():
    chunks = chunk_filing(make_filing())
    legal = [c for c in chunks if c.item == "3"]
    assert legal and legal[0].text.startswith("[ACME 10-K FY2024 | Item 3. Legal Proceedings]")


def test_chunk_text_respects_size():
    text = "\n".join(f"Paragraph {i}. " + "word " * 80 for i in range(40))
    chunks = chunk_text(text, max_chars=1000, overlap=100)
    assert len(chunks) > 5
    assert all(len(c) <= 1400 for c in chunks)


def test_notebook_format_round_trip(tmp_path):
    f = make_filing()
    nb = f.to_notebook_dict()
    assert nb["section_3"] == {"Legal Proceedings": f.sections["3"].text}
    f.save(tmp_path / "x.json")
    assert type(f).load(tmp_path / "x.json").sections["7"].text == f.sections["7"].text


RUNNING_HEADERS = """Table of Contents
Item 1. Business 3
Item 1A. Risk Factors 9
Item 7. MD&A 30
PART I
ITEM 1. BUSINESS
We sell widgets.
12
PART I
Item 1
More about widgets on page two.
ITEM 1A. RISK FACTORS
Widgets may break.
PART II
Item 1A
Widgets may also rust.
PART II
ITEM 7. MANAGEMENT'S DISCUSSION
Revenue rose.
34
PART II
Item 7
Margins fell.
SIGNATURES
Pursuant to the requirements..."""


def test_running_page_headers_do_not_split_sections():
    secs = split_items(RUNNING_HEADERS)
    assert secs["1"].text == "We sell widgets.\n\nMore about widgets on page two."
    assert "Widgets may break." in secs["1A"].text and "rust" in secs["1A"].text
    assert secs["7"].text.startswith("Revenue rose.") and "Margins fell." in secs["7"].text
    assert "Pursuant" not in secs["7"].text and "PART II" not in secs["7"].text


def test_low_coverage_keeps_remainder_loudly():
    text = ("Item 7. MD&A\nRefer to pages 50-120 of the Annual Report.\nSIGNATURES\n"
            + "Annual report: net interest income grew strongly. " * 200)
    log = FallbackLog()
    secs = split_items(text, log=log)
    assert log.counts["low_item_coverage"] == 1
    assert "net interest income" in secs["X"].text
    assert normalize_item("x") == "X"
