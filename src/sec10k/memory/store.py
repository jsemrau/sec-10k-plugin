"""Local memory store: one SQLite file, BM25 full-text search, optional vectors.

* Costs nothing to index - no embeddings are required.
* Namespaced (ticker or ``user:<id>``) so filings of different companies never mix.
* Every chunk keeps its structure (item, accession, fiscal year) for filtering.
* Forgetting is explicit: ``supersede`` hides an older filing once a newer one is
  ingested, ``forget`` deletes, ``expires_at`` lets notes age out.
"""

from __future__ import annotations

import json
import math
import re
import struct
import threading
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from ..fallbacks import FallbackLog
from .base import Embedder, MemoryRecord, SearchHit

_STOPWORDS = set(
    "a an and are as at be by can could did do does for from had has have how i if in into is "
    "it its me my of on or our please should so tell than that the their them then there these "
    "they this to was we were what when where which who why will with would you your".split()
)
_RRF_K = 60


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pack(vec: Sequence[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def query_terms(query: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9]+", query.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


class MemoryStore:
    def __init__(
        self,
        path: str | Path = ":memory:",
        embedder: Embedder | None = None,
        log: FallbackLog | None = None,
    ) -> None:
        try:
            import sqlite3
        except ImportError as e:  # some pyenv builds lack _sqlite3
            raise ImportError(
                "MemoryStore needs Python's sqlite3 module (your interpreter was built "
                "without it). Use a Python from python.org, uv, or your OS package manager."
            ) from e
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self.embedder = embedder
        self.log = log or FallbackLog()
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._fts = True
        self._init_schema()

    def _init_schema(self) -> None:
        import sqlite3

        with self._lock, self._db:
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    item TEXT,
                    accession TEXT,
                    created_at TEXT NOT NULL,
                    superseded_at TEXT,
                    expires_at TEXT,
                    embedding BLOB,
                    embedder TEXT
                )"""
            )
            self._db.execute("CREATE INDEX IF NOT EXISTS ix_ns ON memories(namespace, kind)")
            self._db.execute("CREATE INDEX IF NOT EXISTS ix_acc ON memories(accession)")
            try:
                self._db.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5("
                    "id UNINDEXED, text, tokenize='porter unicode61')"
                )
            except sqlite3.OperationalError as e:
                self._fts = False
                self.log.record(
                    "fts5_unavailable",
                    f"SQLite has no FTS5 ({e}); search falls back to slow term counting",
                )

    # --- writing -------------------------------------------------------------------

    def add(self, records: Iterable[MemoryRecord], log: FallbackLog | None = None) -> int:
        records = list(records)
        if not records:
            return 0
        log = log or self.log
        vectors: list[list[float]] | list[None] = [None] * len(records)
        if self.embedder is not None:
            try:
                vectors = self.embedder.embed([r.text for r in records])
            except Exception as e:
                log.record(
                    "embedding_failed",
                    f"{self.embedder.name}: {e}; records stored for lexical search only",
                    n=len(records),
                )
        with self._lock, self._db:
            for rec, vec in zip(records, vectors):
                self._db.execute(
                    "INSERT OR REPLACE INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        rec.id, rec.namespace, rec.kind, rec.text, json.dumps(rec.metadata),
                        rec.item, rec.accession, rec.created_at, rec.superseded_at,
                        rec.expires_at, _pack(vec) if vec else None,
                        self.embedder.name if vec and self.embedder else None,
                    ),
                )
                if self._fts:
                    self._db.execute("DELETE FROM memories_fts WHERE id = ?", (rec.id,))
                    self._db.execute(
                        "INSERT INTO memories_fts(id, text) VALUES (?, ?)", (rec.id, rec.text)
                    )
        return len(records)

    def supersede(self, namespace: str, kind: str, keep_accession: str) -> int:
        """Mark ``namespace``'s ``kind`` records from other filings as superseded, and
        un-supersede ``keep_accession``."""
        with self._lock, self._db:
            self._db.execute(
                "UPDATE memories SET superseded_at = NULL WHERE namespace = ? AND kind = ? "
                "AND accession = ?", (namespace, kind, keep_accession),
            )
            cur = self._db.execute(
                "UPDATE memories SET superseded_at = ? WHERE namespace = ? AND kind = ? "
                "AND accession IS NOT ? AND superseded_at IS NULL",
                (_now(), namespace, kind, keep_accession),
            )
            return cur.rowcount

    def forget(
        self,
        namespace: str,
        kind: str | None = None,
        accession: str | None = None,
        ids: Sequence[str] | None = None,
        superseded_only: bool = False,
    ) -> int:
        """Permanently delete matching records. Returns the number deleted."""
        where, args = ["namespace = ?"], [namespace]
        if kind:
            where.append("kind = ?")
            args.append(kind)
        if accession:
            where.append("accession = ?")
            args.append(accession)
        if ids:
            where.append(f"id IN ({','.join('?' * len(ids))})")
            args.extend(ids)
        if superseded_only:
            where.append("superseded_at IS NOT NULL")
        clause = " AND ".join(where)
        with self._lock, self._db:
            if self._fts:
                self._db.execute(
                    f"DELETE FROM memories_fts WHERE id IN (SELECT id FROM memories WHERE {clause})",
                    args,
                )
            return self._db.execute(f"DELETE FROM memories WHERE {clause}", args).rowcount

    # --- reading -------------------------------------------------------------------

    @staticmethod
    def _row_to_record(row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"], namespace=row["namespace"], kind=row["kind"], text=row["text"],
            metadata=json.loads(row["metadata"]), created_at=row["created_at"],
            superseded_at=row["superseded_at"], expires_at=row["expires_at"],
        )

    def _filters(
        self,
        namespace: str,
        kinds: Sequence[str] | None,
        items: Sequence[str] | None,
        accession: str | None,
        include_superseded: bool,
        alias: str = "m",
    ) -> tuple[str, list]:
        where = [f"{alias}.namespace = ?", f"({alias}.expires_at IS NULL OR {alias}.expires_at > ?)"]
        args: list = [namespace, _now()]
        if kinds:
            where.append(f"{alias}.kind IN ({','.join('?' * len(kinds))})")
            args.extend(kinds)
        if items:
            where.append(f"{alias}.item IN ({','.join('?' * len(items))})")
            args.extend(items)
        if accession:
            where.append(f"{alias}.accession = ?")
            args.append(accession)
        if not include_superseded:
            where.append(f"{alias}.superseded_at IS NULL")
        return " AND ".join(where), args

    def list(
        self,
        namespace: str,
        kinds: Sequence[str] | None = None,
        items: Sequence[str] | None = None,
        accession: str | None = None,
        include_superseded: bool = False,
        limit: int | None = None,
    ) -> list[MemoryRecord]:
        clause, args = self._filters(namespace, kinds, items, accession, include_superseded)
        sql = f"SELECT * FROM memories m WHERE {clause} ORDER BY m.created_at, m.rowid"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._lock:
            return [self._row_to_record(r) for r in self._db.execute(sql, args)]

    def count(self, namespace: str, kinds: Sequence[str] | None = None,
              include_superseded: bool = False) -> int:
        clause, args = self._filters(namespace, kinds, None, None, include_superseded)
        with self._lock:
            return self._db.execute(f"SELECT COUNT(*) FROM memories m WHERE {clause}", args).fetchone()[0]

    def namespaces(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT namespace, kind, COUNT(*) n, SUM(superseded_at IS NOT NULL) superseded "
                "FROM memories GROUP BY namespace, kind ORDER BY namespace, kind"
            )
            return [dict(r) for r in rows]

    def _lexical(self, terms: list[str], clause: str, args: list, limit: int) -> list[tuple[str, float]]:
        if not terms:
            return []
        if self._fts:
            match = " OR ".join(f'"{t}"' for t in terms)
            sql = (
                "SELECT m.id, bm25(memories_fts) AS s FROM memories_fts "
                "JOIN memories m ON m.id = memories_fts.id "
                f"WHERE memories_fts MATCH ? AND {clause} ORDER BY s LIMIT ?"
            )
            with self._lock:
                return [(r["id"], -r["s"]) for r in self._db.execute(sql, [match, *args, limit])]
        # No FTS5: naive term frequency (already logged at startup).
        with self._lock:
            rows = self._db.execute(f"SELECT m.id, m.text FROM memories m WHERE {clause}", args)
            scored = []
            for r in rows:
                low = r["text"].lower()
                s = sum(low.count(t) for t in terms)
                if s:
                    scored.append((r["id"], float(s)))
        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    def _vector(self, query: str, clause: str, args: list, limit: int,
                log: FallbackLog) -> list[tuple[str, float]]:
        if self.embedder is None:
            return []
        try:
            qvec = self.embedder.embed([query])[0]
        except Exception as e:
            log.record("query_embedding_failed", f"{self.embedder.name}: {e}; lexical search only")
            return []
        with self._lock:
            rows = list(self._db.execute(
                f"SELECT m.id, m.embedding FROM memories m WHERE {clause} AND m.embedding IS NOT NULL",
                args,
            ))
        if not rows:
            log.record("no_embeddings", "embedder set but no stored records have vectors")
            return []
        scored = [(r["id"], _cosine(qvec, _unpack(r["embedding"]))) for r in rows]
        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    def search(
        self,
        namespace: str,
        query: str,
        k: int = 8,
        kinds: Sequence[str] | None = None,
        items: Sequence[str] | None = None,
        boost_items: Mapping[str, float] | Sequence[str] | None = None,
        accession: str | None = None,
        include_superseded: bool = False,
        log: FallbackLog | None = None,
    ) -> list[SearchHit]:
        """Hybrid search: BM25 (+ cosine if an embedder is set), fused with reciprocal rank.

        ``items`` is a hard filter; ``boost_items`` only nudges those items upward
        (a sequence boosts x1.5, a mapping gives a per-item multiplier).
        """
        log = log or self.log
        clause, args = self._filters(namespace, kinds, items, accession, include_superseded)
        pool = max(k * 6, 50)
        lexical = self._lexical(query_terms(query), clause, args, pool)
        vector = self._vector(query, clause, args, pool, log)

        fused: dict[str, dict] = {}
        for rank, (rid, _) in enumerate(lexical, 1):
            fused.setdefault(rid, {})["lex"] = rank
        for rank, (rid, _) in enumerate(vector, 1):
            fused.setdefault(rid, {})["vec"] = rank
        if not fused:
            return []

        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM memories WHERE id IN ({','.join('?' * len(fused))})", list(fused)
            )
            records = {r["id"]: self._row_to_record(r) for r in rows}
        if isinstance(boost_items, Mapping):
            boost = dict(boost_items)
        else:
            boost = {i: 1.5 for i in boost_items or ()}
        hits = []
        for rid, ranks in fused.items():
            score = sum(1.0 / (_RRF_K + r) for r in ranks.values())
            rec = records[rid]
            score *= boost.get(rec.item, 1.0)
            hits.append(SearchHit(rec, score, ranks.get("lex"), ranks.get("vec")))
        hits.sort(key=lambda h: -h.score)
        return hits[:k]

    def close(self) -> None:
        with self._lock:
            self._db.close()
