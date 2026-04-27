from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterable, Iterator

from .models import Parser, PipelineOutcome


# Per-customer file locks. File storage is a demo choice; real system uses a DB.
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    key = str(path)
    with _locks_guard:
        lk = _locks.get(key)
        if lk is None:
            lk = threading.Lock()
            _locks[key] = lk
        return lk


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


class JsonlStore:
    """Append/rewrite JSONL store with per-path locking."""

    def __init__(self, path: Path) -> None:
        self.path = path
        _ensure_parent(self.path)

    def append(self, obj: dict) -> None:
        line = json.dumps(obj, ensure_ascii=False)
        with _lock_for(self.path):
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        with _lock_for(self.path):
            with self.path.open("r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]

    def rewrite(self, items: Iterable[dict]) -> None:
        with _lock_for(self.path):
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for obj in items:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            tmp.replace(self.path)


class ParserRepository:
    """Per-customer parser persistence. One JSONL per customer.

    Re-reads the file on every operation — fine for the demo's scale. In prod this
    would be a DB with proper indexes and optimistic concurrency.
    """

    def __init__(self, storage_root: Path) -> None:
        self.base = storage_root / "parsers"
        self.base.mkdir(parents=True, exist_ok=True)

    def _path(self, customer_id: str) -> Path:
        return self.base / f"{customer_id}.jsonl"

    def list(self, customer_id: str) -> list[Parser]:
        store = JsonlStore(self._path(customer_id))
        return [Parser.model_validate(row) for row in store.read_all()]

    def save_all(self, customer_id: str, parsers: list[Parser]) -> None:
        store = JsonlStore(self._path(customer_id))
        store.rewrite(p.model_dump() for p in parsers)

    def upsert(self, parser: Parser) -> None:
        items = self.list(parser.customer_id)
        replaced = False
        for i, existing in enumerate(items):
            if existing.id == parser.id:
                items[i] = parser
                replaced = True
                break
        if not replaced:
            items.append(parser)
        self.save_all(parser.customer_id, items)

    def find_by_cluster(self, customer_id: str, cluster_signature: str) -> list[Parser]:
        return [p for p in self.list(customer_id) if p.cluster_signature == cluster_signature]


class ClusterIndex:
    """Per-customer cluster → parser_id binding, plus observation counts.

    Lets us short-circuit the router on repeat clusters even before a parser is active.
    """

    def __init__(self, storage_root: Path) -> None:
        self.base = storage_root / "clusters"
        self.base.mkdir(parents=True, exist_ok=True)

    def _path(self, customer_id: str) -> Path:
        return self.base / f"{customer_id}.jsonl"

    def load(self, customer_id: str) -> dict[str, dict]:
        store = JsonlStore(self._path(customer_id))
        rows = store.read_all()
        # Last write wins per signature.
        index: dict[str, dict] = {}
        for row in rows:
            index[row["signature"]] = row
        return index

    def save(self, customer_id: str, index: dict[str, dict]) -> None:
        store = JsonlStore(self._path(customer_id))
        store.rewrite(index.values())

    def bump(self, customer_id: str, signature: str, parser_id: str | None) -> dict:
        index = self.load(customer_id)
        row = index.get(signature) or {"signature": signature, "count": 0, "parser_id": None}
        row["count"] = row.get("count", 0) + 1
        if parser_id is not None:
            row["parser_id"] = parser_id
        index[signature] = row
        self.save(customer_id, index)
        return row


class OutcomeWriter:
    """Writes extracted events and quarantined lines."""

    def __init__(self, storage_root: Path) -> None:
        self.events_base = storage_root / "events"
        self.quarantine_base = storage_root / "quarantine"
        self.events_base.mkdir(parents=True, exist_ok=True)
        self.quarantine_base.mkdir(parents=True, exist_ok=True)

    def _events_path(self, customer_id: str) -> Path:
        return self.events_base / f"{customer_id}.jsonl"

    def _quarantine_path(self, customer_id: str) -> Path:
        return self.quarantine_base / f"{customer_id}.jsonl"

    def write(self, outcome: PipelineOutcome) -> None:
        target = (
            self._quarantine_path(outcome.customer_id)
            if outcome.source == "quarantine"
            else self._events_path(outcome.customer_id)
        )
        JsonlStore(target).append(outcome.model_dump())

    def _append_many(self, path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        _ensure_parent(path)
        with _lock_for(path):
            with path.open("a", encoding="utf-8") as f:
                for obj in rows:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def append_many_events(self, customer_id: str, rows: list[dict]) -> None:
        self._append_many(self._events_path(customer_id), rows)

    def append_many_quarantine(self, customer_id: str, rows: list[dict]) -> None:
        self._append_many(self._quarantine_path(customer_id), rows)

    def read_events(self, customer_id: str) -> list[dict]:
        return JsonlStore(self._events_path(customer_id)).read_all()

    def read_quarantine(self, customer_id: str) -> list[dict]:
        return JsonlStore(self._quarantine_path(customer_id)).read_all()


# gpt-4.1-mini pricing (per 1M tokens)
_COST_INPUT_PER_M = 0.40
_COST_OUTPUT_PER_M = 1.60
# Fallback estimates when the SDK doesn't expose token counts
_EST_INPUT_PER_CALL = 400
_EST_OUTPUT_PER_CALL = 100


def estimate_cost(input_tokens: int, output_tokens: int, calls: int) -> float:
    if input_tokens or output_tokens:
        return (input_tokens / 1_000_000 * _COST_INPUT_PER_M) + (output_tokens / 1_000_000 * _COST_OUTPUT_PER_M)
    return calls * (
        (_EST_INPUT_PER_CALL / 1_000_000 * _COST_INPUT_PER_M)
        + (_EST_OUTPUT_PER_CALL / 1_000_000 * _COST_OUTPUT_PER_M)
    )


class LLMUsageStore:
    """Persists per-customer cumulative LLM call and token counts across batches."""

    def __init__(self, storage_root: Path) -> None:
        self.base = storage_root / "llm_usage"
        self.base.mkdir(parents=True, exist_ok=True)

    def _path(self, customer_id: str) -> Path:
        return self.base / f"{customer_id}.jsonl"

    def append(self, customer_id: str, record: dict) -> None:
        JsonlStore(self._path(customer_id)).append(record)

    def totals(self, customer_id: str) -> dict:
        records = JsonlStore(self._path(customer_id)).read_all()
        calls = sum(r.get("calls", 0) for r in records)
        input_tokens = sum(r.get("input_tokens", 0) for r in records)
        output_tokens = sum(r.get("output_tokens", 0) for r in records)
        cost = estimate_cost(input_tokens, output_tokens, calls)
        tokens_known = bool(input_tokens or output_tokens)
        return {
            "total_llm_calls": calls,
            "total_input_tokens": input_tokens,
            "total_output_tokens": output_tokens,
            "total_cost_usd": round(cost, 6),
            "tokens_known": tokens_known,
            "model": "gpt-4.1-mini",
        }

    def delete(self, customer_id: str) -> None:
        path = self._path(customer_id)
        if path.exists():
            path.unlink()
