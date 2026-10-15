"""Command line: ``sec10k ingest APLD``, ``sec10k ask APLD "legal issues?" --llm anthropic``."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys

from .analyst import SEC10KAnalyst


def _llm(args):
    if args.llm == "none":
        return None
    if args.llm == "anthropic":
        from .llm.anthropic_llm import AnthropicLLM

        return AnthropicLLM(model=args.model or "claude-opus-5-5", effort=args.effort)
    from .llm.openai_compat import OpenAICompatibleLLM

    price = tuple(float(x) for x in args.price.split(",")) if args.price else None
    return OpenAICompatibleLLM(model=args.model or "qwen2.5:14b", base_url=args.base_url,
                               api_key=args.api_key or "none", price=price)


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _warn_fallbacks(trace) -> None:
    if trace.total:
        print(f"\n!! {trace.total} fallback(s): {dict(trace.counts)}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sec10k", description=__doc__)
    p.add_argument("--home", help="data directory (default $SEC10K_HOME or ~/.sec10k)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("filings", help="list recent 10-K filings")
    s.add_argument("ticker")
    s.add_argument("--limit", type=int, default=5)

    s = sub.add_parser("ingest", help="download, split and index a 10-K")
    s.add_argument("ticker")
    s.add_argument("--fiscal-year", type=int)
    s.add_argument("--accession")
    s.add_argument("--force", action="store_true")
    s.add_argument("--keep-older", action="store_true")

    s = sub.add_parser("section", help="print one Item of an ingested 10-K")
    s.add_argument("ticker")
    s.add_argument("item")

    s = sub.add_parser("export", help="write the ingested filing as JSON")
    s.add_argument("ticker")
    s.add_argument("-o", "--output")
    s.add_argument("--notebook-format", action="store_true",
                   help='original notebook shape: {"section_1a": {"Risk Factors": "..."}}')

    s = sub.add_parser("search", help="retrieve relevant excerpts (no LLM)")
    s.add_argument("ticker")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=8)
    s.add_argument("--items", nargs="*")

    s = sub.add_parser("ask", help="answer a question with an LLM over retrieved excerpts")
    s.add_argument("ticker")
    s.add_argument("question")
    s.add_argument("--user")
    s.add_argument("-k", type=int, default=8)
    s.add_argument("--items", nargs="*")
    s.add_argument("--llm", choices=["anthropic", "openai-compatible", "none"], default="anthropic")
    s.add_argument("--model")
    s.add_argument("--effort", default="medium")
    s.add_argument("--base-url", help="OpenAI-compatible endpoint, e.g. http://localhost:11434/v1")
    s.add_argument("--api-key")
    s.add_argument("--price", help="input,output $/MTok for openai-compatible models, e.g. 0,0")
    s.add_argument("--dry-run", action="store_true", help="print the prompt, don't call the LLM")

    s = sub.add_parser("remember", help="store a note (ticker) or profile fact (user:<id>)")
    s.add_argument("namespace")
    s.add_argument("text")
    s.add_argument("--kind", default="note", choices=["note", "profile"])
    s.add_argument("--ttl-days", type=float)

    s = sub.add_parser("recall", help="list or search notes")
    s.add_argument("namespace")
    s.add_argument("query", nargs="?")

    s = sub.add_parser("forget", help="delete memories")
    s.add_argument("namespace")
    s.add_argument("--kind")
    s.add_argument("--accession")
    s.add_argument("--superseded-only", action="store_true")

    sub.add_parser("overview", help="what is stored")

    args = p.parse_args(argv)
    if hasattr(signal, "SIGPIPE"):  # `sec10k section X 1A | head` should exit quietly
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    a = SEC10KAnalyst(llm=_llm(args) if args.cmd == "ask" and not args.dry_run else None,
                      home=args.home)

    if args.cmd == "filings":
        _print([vars(r) for r in a.filings(args.ticker, limit=args.limit)])
    elif args.cmd == "ingest":
        res = a.ingest(args.ticker, accession=args.accession, fiscal_year=args.fiscal_year,
                       force=args.force, keep_older=args.keep_older)
        _print(res.to_dict())
    elif args.cmd == "section":
        sec = a.section(args.ticker, args.item)
        print(f"Item {sec.item}. {sec.title}\n\n{sec.text}")
    elif args.cmd == "export":
        f = a.load_filing(args.ticker)
        data = f.to_notebook_dict() if args.notebook_format else f.to_dict()
        out = json.dumps(data, indent=2)
        if args.output:
            with open(args.output, "w") as fh:
                fh.write(out)
        else:
            print(out)
    elif args.cmd == "search":
        res = a.search(args.ticker, args.query, k=args.k, items=args.items)
        for i, h in enumerate(res.hits, 1):
            print(f"--- E{i} Item {h.record.item} score={h.score:.4f}\n{h.record.text}\n")
        _warn_fallbacks(res.trace)
    elif args.cmd == "ask":
        if args.dry_run:
            pr = a.build_prompt(args.ticker, args.question, user_id=args.user, k=args.k,
                                items=args.items)
            print(f"[system]\n{pr.system}\n\n[user]\n{pr.messages[0]['content']}")
            print(f"\n~{pr.approx_tokens} input tokens (chars/4 estimate)", file=sys.stderr)
            return 0
        ans = a.ask(args.ticker, args.question, user_id=args.user, k=args.k, items=args.items,
                    on_text=lambda t: print(t, end="", flush=True))
        u = ans.usage
        cost = f"${u.cost_usd:.4f}" if u.cost_usd is not None else "unknown"
        print(f"\n\n[{ans.model}] in={u.input_tokens} out={u.output_tokens} cost={cost}",
              file=sys.stderr)
        _warn_fallbacks(ans.trace)
    elif args.cmd == "remember":
        _print(a.remember(args.namespace, args.text, kind=args.kind, ttl_days=args.ttl_days).to_dict())
    elif args.cmd == "recall":
        _print([r.to_dict() for r in a.recall(args.namespace, query=args.query)])
    elif args.cmd == "forget":
        _print({"deleted": a.forget(args.namespace, kind=args.kind, accession=args.accession,
                                    superseded_only=args.superseded_only)})
    elif args.cmd == "overview":
        _print(a.store.namespaces())
    return 0


if __name__ == "__main__":
    sys.exit(main())
