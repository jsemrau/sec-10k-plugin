# sec10k

**Read SEC 10-K filings with any LLM, without pasting a 500-page document into the prompt.**

`sec10k` downloads a company's annual report (Form 10-K) from SEC EDGAR, splits it into its
standard Items (Business, Risk Factors, Legal Proceedings, MD&A, ...), and stores it in a
local SQLite memory. When you ask a question, only the few passages that answer it go to
the model.

It works three ways:

- **As a Python library.** `SEC10KAnalyst` gives you ingest, search, ask and remember.
- **As a Claude plugin.** An MCP server, a skill and a slash command let Claude read filings
  directly.
- **As a CLI.** `sec10k ingest APLD`, `sec10k ask APLD "any legal issues?"`.

```python
from sec10k import SEC10KAnalyst, AnthropicLLM

analyst = SEC10KAnalyst(llm=AnthropicLLM())
analyst.ingest("APLD")                       # free: SEC EDGAR -> Items -> local SQLite
answer = analyst.ask("APLD", "Does the company have legal issues?")
print(answer.text)                           # cites excerpts, e.g. [E2] (Item 3)
print(answer.usage.cost_usd)                 # what this question cost
```

## Why

This started as a [notebook](https://github.com/ternary-ai/cookbooks/blob/main/SEC_API_10K_Extractor.ipynb)
that extracted 10-K sections, plus an analyst agent built on Mem0. Putting the whole
filing in the prompt cost **$3.25 per question**. The agent was tied to OpenAI, its memory
kept 16 opaque facts out of the whole document, and there was no way to forget anything.
`sec10k` is the rebuild:

- **Cheap.** A question sends about 8 relevant excerpts, around 2.8K tokens instead of around
  137K for the full APLD filing. Fetching and indexing are free: SEC EDGAR has no API key,
  and search runs on SQLite full-text search (BM25), so no embeddings are needed.
- **Any model.** Claude, Ollama, vLLM, LM Studio, OpenAI, or your own class. Embeddings are
  optional too.
- **Transparent.** `build_prompt()` shows exactly what will be sent. Every answer reports
  its tokens, its cost, and which excerpts it cited.
- **Memory you control.** Every chunk keeps its Item, filing and fiscal year. Notes, past
  answers and your investor profile are separate, explicit records. A newer 10-K
  supersedes the older one, and `forget()` and `ttl_days` let you remove things.
- **Loud failures.** When something degrades (an Item is missing, nothing matched, a
  different model answered), it is logged, counted, and returned with the result.

## Install

Requires Python 3.10+ with the `sqlite3` module.

```bash
pip install "sec10k[anthropic]"   # Claude
pip install "sec10k[openai]"      # any OpenAI-compatible endpoint (Ollama, vLLM, ...)
pip install "sec10k[mcp]"         # MCP server for Claude Desktop / Claude Code
pip install "sec10k[all]"         # everything

# from a clone
uv venv && uv pip install -e ".[dev]"
```

The SEC requires every client to identify itself with a name and email:

```bash
export SEC10K_USER_AGENT="Your Name you@example.com"
export SEC10K_HOME=~/.sec10k          # optional: where memory.db, filings and cache live
export ANTHROPIC_API_KEY=...          # only if you use AnthropicLLM
```

## Usage

### Ingest and read

```python
analyst = SEC10KAnalyst()                          # no LLM needed for any of this
analyst.filings("APLD")                            # recent 10-Ks on EDGAR
analyst.ingest("APLD")                             # latest; or fiscal_year=2025 / accession=...
analyst.section("APLD", "1A").text                 # full text of one Item
analyst.search("APLD", "securities class action", k=5).hits
```

`ingest()` is idempotent and returns the size of every Item, so you can see right away
which ones are empty, "Not applicable", or incorporated by reference to the proxy statement.

### Ask

```python
answer = analyst.ask("APLD", "How fast did revenue grow and why?",
                     on_text=lambda t: print(t, end=""))     # streams
answer.citations     # [{"id": "E1", "item": "7", "accession": "...", "url": "..."}, ...]
answer.usage         # Usage(input_tokens=..., output_tokens=..., cost_usd=...)
answer.trace.counts  # empty Counter = nothing fell back
```

To see the prompt without calling a model:

```python
prompt = analyst.build_prompt("APLD", "Any legal issues?")
prompt.system, prompt.messages, prompt.approx_tokens
```

### Choose a model

```python
from sec10k import AnthropicLLM, OpenAICompatibleLLM

AnthropicLLM()                                   # claude-opus-5-5, effort="medium"
AnthropicLLM(model="claude-sonnet-5-5", effort="low")

OpenAICompatibleLLM("qwen2.5:14b",               # local Ollama, free
                    base_url="http://localhost:11434/v1", api_key="ollama", price=(0, 0))
```

Any object with `complete(system, messages, on_text=None, log=None) -> LLMResponse` works.
See `sec10k.llm.base.LLM`.

`AnthropicLLM` turns on the API's server-side fallback by default: if the requested model
declines a request, another model completes it. When that happens it is recorded in the
trace as `llm_server_fallback`. Pass `server_fallbacks=False` to turn it off.

### Memory

Memories are grouped by namespace: the **ticker** for company data, **`user:<id>`** for a
person.

```python
analyst.remember_profile("jan", "Long-term holder, low risk tolerance, 2% position in APLD")
analyst.remember("APLD", "Watch the stayed securities class action", ttl_days=120)

analyst.ask("APLD", "Should the lawsuits worry me?", user_id="jan")  # profile is included
analyst.recall("APLD", "lawsuit")      # notes and past Q&A (answers are saved by default)
```

When a newer 10-K is ingested, older filings are **superseded**: they're kept on disk but
hidden from search. To compare years:

```python
analyst.ingest("APLD", fiscal_year=2025, keep_older=True)
analyst.search("APLD", "revenue growth", include_superseded=True)
analyst.forget("APLD", superseded_only=True)        # delete replaced filings for good
```

### Better recall with embeddings (optional)

Search runs on BM25 alone by default. Add an embedder to get hybrid search, where BM25 and
cosine similarity rankings are fused:

```python
from sec10k import MemoryStore, OpenAICompatibleEmbedder, FastEmbedEmbedder

store = MemoryStore("~/.sec10k/memory.db",
                    embedder=OpenAICompatibleEmbedder("nomic-embed-text",
                                                      base_url="http://localhost:11434/v1",
                                                      api_key="ollama"))
# or FastEmbedEmbedder()  ->  pip install "sec10k[local-embed]"
analyst = SEC10KAnalyst(store=store)
```

### Other sources

```python
from sec10k import EdgarSource, SecApiSource, Filing

SEC10KAnalyst(source=EdgarSource(user_agent="Me me@example.com"))   # default
SEC10KAnalyst(source=SecApiSource(api_key="..."))     # sec-api.io, pip install "sec10k[secapi]"
analyst.ingest_filing(Filing.load("APLD-10-K.json"))  # bring your own
```

Implement `list_filings()` and `fetch()` (the `FilingSource` protocol) to plug in any other
source.

`Filing.to_notebook_dict()` and `sec10k export APLD --notebook-format` write the
`{"section_1a": {"Risk Factors": "..."}}` JSON the original notebook produced.

## Claude plugin

This repository is a Claude Code plugin and its own plugin marketplace. It gives Claude:

- **MCP tools:** `ingest_10k`, `search_10k`, `get_10k_section`, `list_10k_filings`,
  `list_10k_items`, `remember`, `recall`, `forget`, `memory_overview`
- **Skill:** `sec-10k-analyst` teaches Claude the workflow: ingest, search the right Items,
  read in full when needed, cite, and remember.
- **Command:** `/sec10k:10k <TICKER> [question]`. Without a question you get a one-screen
  briefing.

No LLM runs inside the server. Claude does the reasoning and reads only the excerpts it asks
for, so you don't need an extra API key.

**Claude Code**

```bash
export SEC10K_USER_AGENT="Your Name you@example.com"    # before starting Claude Code
```
```
/plugin marketplace add jsemrau/sec-10k-plugin     # or a local path to this repo
/plugin install sec10k@sec10k
/sec10k:10k APLD any legal issues?
```

The server is launched with [uv](https://docs.astral.sh/uv/), which must be on your PATH.
Without the plugin system:
`claude mcp add sec10k -e SEC10K_USER_AGENT="Me me@example.com" -- sec10k-mcp`

**Claude desktop app (macOS / Windows)**

The app has no `/plugin` command, but it runs local MCP servers, which is where all the
tools live. You don't need to clone anything; uv installs the server from GitHub. Do these
steps on the computer where Claude Desktop runs. Each computer keeps its own memory in
`~/.sec10k`, so ingest your tickers again there; it's free and takes about a second.

1. Install [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
   and note its full path from `which uv`. The app doesn't see your shell's PATH.
2. Open *Settings → Developer → Edit Config* and add the following to
   `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "sec10k": {
      "command": "/Users/YOU/.local/bin/uv",
      "args": ["tool", "run", "--python", "3.12",
               "--from", "sec10k[mcp] @ git+https://github.com/jsemrau/sec-10k-plugin",
               "sec10k-mcp"],
      "env": { "SEC10K_USER_AGENT": "Your Name you@example.com" }
    }
  }
}
```

3. Quit and reopen Claude. The sec10k tools appear under the tools/connectors menu in a
   chat. Then ask, for example: "Ingest APLD's latest 10-K and tell me about its legal
   proceedings."
4. Optional: teach Claude the workflow. Zip the `skills/sec-10k-analyst` folder and upload
   it under *Settings → Capabilities → Skills*.

`--python 3.12` makes uv use its own Python, which always includes SQLite. Data is stored in
`~/.sec10k`. To use the local clone instead, replace the args with
`["run", "--quiet", "--project", "/path/to/sec10k", "--extra", "mcp", "sec10k-mcp"]`.

## CLI

```bash
sec10k filings APLD
sec10k ingest APLD [--fiscal-year 2025] [--keep-older] [--force]
sec10k section APLD 1A | less
sec10k search APLD "securities class action" -k 5 [--items 3 1A]
sec10k ask APLD "Any legal issues?" --dry-run        # print the prompt, spend nothing
sec10k ask APLD "Any legal issues?" --user jan       # Claude, via ANTHROPIC_API_KEY
sec10k ask APLD "Any legal issues?" --llm openai-compatible \
       --base-url http://localhost:11434/v1 --model qwen2.5:14b --price 0,0
sec10k remember user:jan "Low risk tolerance" --kind profile
sec10k recall APLD lawsuit
sec10k forget APLD --superseded-only
sec10k export APLD --notebook-format -o APLD-10-K.json
sec10k overview
```

## How it works

1. **Fetch.** The ticker is mapped to a CIK via `company_tickers.json`. The submissions
   API lists the filings, and the primary HTML document is downloaded and cached. Requests
   are rate-limited to stay within SEC fair-access rules.
2. **Split.** The HTML is converted to text, and the text is cut at `Item N.`, `PART` and
   `SIGNATURES` headings.
   - Repeated running page headers ("PART II / Item 7") continue a section instead of
     splitting it.
   - For each Item the longest body wins, so table-of-contents entries lose.
   - If the Items cover less than half the text, the rest is kept as pseudo-Item `X`. This
     happens with filers such as banks, whose 10-K cross-references an integrated annual
     report.
3. **Chunk.** Paragraphs are packed into chunks of about 1,500 characters. Each chunk starts
   with a context header such as `[APLD 10-K FY2026 | Item 3. Legal Proceedings]`.
4. **Retrieve.** BM25 search (plus cosine similarity if an embedder is set) runs within the
   ticker's namespace. Words in the question boost the Items most likely to answer it:
   legal → 3/1A, revenue → 7/8, auditor → 14, cyber → 1C, and so on.
5. **Ask.** Your profile, relevant notes and past Q&A, the numbered excerpts and the question
   go to the model. The model is told to cite excerpts and to say when the filing doesn't
   answer the question.

## Fallbacks

Every degraded path is recorded in a `FallbackLog`: logged at WARNING, counted, and returned
with the call. Check `answer.trace.counts` per call, or `analyst.fallback_report()` per
session, before trusting a result.

| kind | meaning |
|---|---|
| `section_missing` | an Item heading wasn't found in the filing |
| `no_item_headings` | the filing couldn't be split; all text went to Item `X` |
| `low_item_coverage` | most text is outside the Items (cross-referenced annual report) |
| `no_retrieval_hits` | nothing matched, so the opening chunks of the likely Items were used |
| `not_ingested` | search on a ticker that was never ingested |
| `ingested_filing_superseded` | an older 10-K was ingested; it's hidden behind the newer one |
| `embedding_failed`, `query_embedding_failed`, `no_embeddings` | vector search unavailable; BM25 only |
| `llm_server_fallback`, `llm_model_substituted` | a different model answered than requested |
| `llm_refusal`, `llm_truncated` | the model declined, or the answer hit `max_tokens` |
| `cost_unknown`, `usage_unavailable` | no price or usage data, so cost is `None` rather than 0 |
| `default_user_agent` | `SEC10K_USER_AGENT` is unset and the SEC may throttle |
| `older_filings_not_searched` | no match in EDGAR's recent submissions; older pages not read |

## Limitations

- Only 10-K is supported for now. `form="10-K/A"` lists amendments, but splitting assumes
  10-K Items.
- Item splitting is heuristic. It was checked on APLD, AAPL, MSFT, TSLA, JPM, KO, NVDA and
  BRK-B, but unusual layouts can still mis-split. Watch `section_missing` and the section
  sizes that `ingest()` returns.
- Many Items only point elsewhere ("see Note 12 in Item 15", "incorporated by reference to
  the proxy statement"). The content they point to is searchable when it is in the 10-K;
  proxy statements are not fetched.
- Costs are computed from a built-in price table for Claude models. Pass `price=` for
  anything else, or the cost is reported as unknown.

## Project layout

```
src/sec10k/
  analyst.py        SEC10KAnalyst: ingest, search, build_prompt, ask, remember, forget
  sources/          EdgarSource (default), SecApiSource (optional), FilingSource protocol
  parsing.py        HTML -> text; split into Items
  chunking.py       paragraph-packed chunks with context headers
  memory/           SQLite + FTS5 store; optional embedders
  llm/              AnthropicLLM, OpenAICompatibleLLM, LLM protocol, pricing
  routing.py        question wording -> likely Items
  fallbacks.py      FallbackLog
  mcp_server.py     MCP tools
  cli.py            sec10k command
.claude-plugin/     plugin.json (declares the MCP server), marketplace.json
skills/             sec-10k-analyst skill
commands/           /sec10k:10k
tests/              offline tests: synthetic 10-K, fake source, fake LLM, mocked SDKs
```

## Development

```bash
uv venv && uv pip install -e ".[dev]"
pytest
```

The tests need no network or API keys. Some pyenv-built Pythons lack `sqlite3`; the Pythons
uv installs include it.

## License

MIT
