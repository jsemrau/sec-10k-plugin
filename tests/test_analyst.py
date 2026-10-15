import pytest

from sec10k import FILING, PROFILE, QA, MemoryRecord, MemoryStore, SEC10KAnalyst

from .conftest import FakeLLM, FakeSource, make_filing


def test_ingest_is_idempotent(analyst):
    r1 = analyst.ingest("acme")
    assert not r1.already_ingested and r1.chunks > 0
    assert r1.sections["3"] > 0
    r2 = analyst.ingest("ACME")
    assert r2.already_ingested and r2.chunks == r1.chunks
    assert analyst.source.fetches == 1


def test_search_routes_legal_questions_to_item_3(analyst):
    analyst.ingest("ACME")
    res = analyst.search("ACME", "Does the company have legal issues or lawsuits?")
    assert "3" in res.routed_items
    assert res.hits[0].record.item == "3"
    assert res.trace.total == 0


def test_search_hard_item_filter(analyst):
    analyst.ingest("ACME")
    res = analyst.search("ACME", "revenue", items=["1a"])
    assert {h.record.item for h in res.hits} <= {"1A"}


def test_no_hits_fallback_is_loud(analyst):
    analyst.ingest("ACME")
    res = analyst.search("ACME", "zzyzx quux")
    assert res.trace.counts["no_retrieval_hits"] == 1
    assert res.hits  # degraded result still returned, but flagged
    assert analyst.fallback_report()["counts"]["no_retrieval_hits"] == 1


def test_search_before_ingest_is_loud(analyst):
    res = analyst.search("ACME", "legal")
    assert res.hits == [] and res.trace.counts["not_ingested"] == 1


def test_ask_sends_only_excerpts_and_profile(analyst):
    analyst.ingest("ACME")
    analyst.remember_profile("jan", "Long-term investor, low risk tolerance.")
    ans = analyst.ask("ACME", "Any legal issues?", user_id="jan", k=3)
    system, messages = analyst.llm.calls[0]
    content = messages[0]["content"]
    assert "low risk tolerance" in content
    assert "Smith v. Acme" in content
    assert content.count("<excerpt ") <= 3
    assert len(content) < 6000  # never the whole filing
    assert ans.citations[0]["item"] == "3"
    assert ans.usage.cost_usd == 0.001
    # The Q&A became a memory and shows up in later prompts
    assert analyst.recall("ACME", kinds=[QA])
    analyst.ask("ACME", "Remind me about legal issues")
    assert "Q: Any legal issues?" in analyst.llm.calls[1][1][0]["content"]


def test_ask_without_llm_raises(tmp_path):
    a = SEC10KAnalyst(source=FakeSource([make_filing()]), home=tmp_path)
    a.ingest("ACME")
    with pytest.raises(ValueError):
        a.ask("ACME", "legal?")
    assert a.build_prompt("ACME", "legal?").approx_tokens > 0


def test_newer_filing_supersedes_older(tmp_path):
    old = make_filing(2023, legal="No material legal proceedings in 2023.")
    new = make_filing(2024)
    a = SEC10KAnalyst(llm=FakeLLM(), source=FakeSource([old]), home=tmp_path)
    a.ingest("ACME")
    a.source = FakeSource([old, new])
    res = a.ingest("ACME")
    assert res.superseded > 0
    texts = " ".join(h.record.text for h in a.search("ACME", "legal proceedings").hits)
    assert "Smith v. Acme" in texts and "2023" not in texts
    with_old = a.search("ACME", "legal proceedings 2023", include_superseded=True)
    assert any("2023" in h.record.text for h in with_old.hits)
    # older filing still loadable by accession, then forgettable
    assert a.load_filing("ACME").ref.accession == new.ref.accession
    assert a.forget("ACME", superseded_only=True) > 0
    assert not (tmp_path / "filings" / "ACME" / f"{old.ref.accession}.json").exists()
    assert (tmp_path / "filings" / "ACME" / f"{new.ref.accession}.json").exists()


def test_ttl_expiry_and_forget(analyst):
    analyst.remember("ACME", "temporary view", ttl_days=-1)
    analyst.remember("ACME", "durable view")
    assert [r.text for r in analyst.recall("ACME")] == ["durable view"]
    assert analyst.forget("ACME", kind="note") == 2


def test_namespaces_do_not_mix(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.add([MemoryRecord("AAA", FILING, "lawsuit filed against AAA", {"item": "3"}),
               MemoryRecord("BBB", FILING, "lawsuit filed against BBB", {"item": "3"}),
               MemoryRecord("user:jan", PROFILE, "likes lawsuits")])
    hits = store.search("AAA", "lawsuit")
    assert [h.record.namespace for h in hits] == ["AAA"]


class BrokenEmbedder:
    name = "broken"

    def embed(self, texts):
        raise RuntimeError("model unavailable")


class ToyEmbedder:
    name = "toy"
    vocab = ["lawsuit", "revenue", "power"]

    def embed(self, texts):
        return [[float(t.lower().count(w)) + 0.01 for w in self.vocab] for t in texts]


def test_embedding_failure_is_loud_and_lexical_still_works(tmp_path):
    store = MemoryStore(tmp_path / "m.db", embedder=BrokenEmbedder())
    store.add([MemoryRecord("AAA", FILING, "a lawsuit", {"item": "3"})])
    assert store.log.counts["embedding_failed"] == 1
    assert store.search("AAA", "lawsuit")
    assert store.log.counts["query_embedding_failed"] == 1


def test_hybrid_search_uses_vectors(tmp_path):
    store = MemoryStore(tmp_path / "m.db", embedder=ToyEmbedder())
    store.add([MemoryRecord("AAA", FILING, "lawsuit lawsuit", {"item": "3"}),
               MemoryRecord("AAA", FILING, "revenue grew", {"item": "7"})])
    hits = store.search("AAA", "lawsuit")
    assert hits[0].record.item == "3" and hits[0].vector_rank == 1 and hits[0].lexical_rank == 1


def test_ingesting_older_filing_keeps_newer_active(tmp_path):
    old, new = make_filing(2023, legal="No material legal proceedings in 2023."), make_filing(2024)
    a = SEC10KAnalyst(source=FakeSource([old, new]), home=tmp_path)
    a.ingest("ACME")  # latest = 2024
    res = a.ingest("ACME", fiscal_year=2023)
    assert res.trace.counts["ingested_filing_superseded"] == 1
    texts = " ".join(h.record.text for h in a.search("ACME", "legal proceedings").hits)
    assert "Smith v. Acme" in texts and "2023" not in texts
