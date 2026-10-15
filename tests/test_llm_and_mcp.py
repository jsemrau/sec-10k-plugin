from types import SimpleNamespace

from sec10k import AnthropicLLM, FallbackLog, OpenAICompatibleLLM


class _Stream:
    def __init__(self, msg):
        self.msg = msg
        self.text_stream = iter(["hi"])

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.msg


def _msg(model, content, stop="end_turn"):
    usage = SimpleNamespace(input_tokens=1_000_000, output_tokens=100_000,
                            cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return SimpleNamespace(model=model, content=content, stop_reason=stop, usage=usage)


def _client(msg, sink):
    def stream(**kw):
        sink.append(kw)
        return _Stream(msg)
    ns = SimpleNamespace(stream=stream)
    return SimpleNamespace(messages=ns, beta=SimpleNamespace(messages=ns))


def test_anthropic_cost_and_request_shape():
    sink = []
    text = SimpleNamespace(type="text", text="answer")
    llm = AnthropicLLM(client=_client(_msg("claude-opus-5-5", [text]), sink))
    log = FallbackLog()
    r = llm.complete("sys", [{"role": "user", "content": "q"}], log=log)
    assert r.text == "answer"
    assert r.usage.cost_usd == 4.0 + 2.0  # 1M in @ $4, 100k out @ $20
    assert sink[0]["model"] == "claude-opus-5-5"
    assert sink[0]["fallbacks"] == "default"
    assert sink[0]["output_config"] == {"effort": "medium"}
    assert log.total == 0


def test_anthropic_server_fallback_is_recorded():
    sink = []
    fb = SimpleNamespace(type="fallback", from_=SimpleNamespace(model="claude-opus-5-5"),
                         to=SimpleNamespace(model="claude-opus-5"))
    text = SimpleNamespace(type="text", text="answer")
    llm = AnthropicLLM(client=_client(_msg("claude-opus-5", [fb, text]), sink))
    log = FallbackLog()
    r = llm.complete("sys", [{"role": "user", "content": "q"}], log=log)
    assert log.counts["llm_server_fallback"] == 1
    assert r.model == "claude-opus-5" and r.usage.cost_usd == 5.0 + 2.5


def test_openai_compat_unknown_price_is_loud():
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **k: resp)))
    log = FallbackLog()
    r = OpenAICompatibleLLM("qwen", client=client).complete("s", [], log=log)
    assert r.usage.cost_usd is None and log.counts["cost_unknown"] == 1
    log2 = FallbackLog()
    r2 = OpenAICompatibleLLM("qwen", client=client, price=(0, 0)).complete("s", [], log=log2)
    assert r2.usage.cost_usd == 0 and log2.total == 0


def test_mcp_tools(tmp_path, monkeypatch):
    from sec10k import SEC10KAnalyst, mcp_server

    from .conftest import FakeSource, make_filing

    monkeypatch.setattr(mcp_server, "_analyst",
                        SEC10KAnalyst(source=FakeSource([make_filing()]), home=tmp_path))
    out = mcp_server.ingest_10k("ACME")
    assert out["chunks"] > 0 and out["sections"]["3"] > 0
    hits = mcp_server.search_10k("ACME", "legal proceedings lawsuit", k=2)["hits"]
    assert hits[0]["metadata"]["item"] == "3"
    sec = mcp_server.get_10k_section("ACME", "7", max_chars=20)
    assert len(sec["text"]) == 20 and sec["next_offset"] == 20
    mcp_server.remember("user:jan", "Holds a 2% position", kind="profile")
    assert mcp_server.recall("user:jan")[0]["text"] == "Holds a 2% position"
    assert any(n["namespace"] == "ACME" for n in mcp_server.memory_overview()["namespaces"])
