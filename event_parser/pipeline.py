from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .agents import BaseExtractor, build_extractor
from .clusterer import cluster_signature
from .config import AppConfig
from .models import ExtractedEvent, PipelineOutcome, Parser
from .normalizer import normalize
from .router import Router
from .storage import ClusterIndex, OutcomeWriter, ParserRepository, estimate_cost

logger = logging.getLogger(__name__)


_MAX_LOG_ENTRIES = 300


class _ChunkCachedExtractor:
    """Per-batch wrapper. The pipeline primes results for a chunk of lines via
    `extract_many` (concurrent in the OpenAI path), so the router's per-line
    `extract()` is a memo hit rather than a fresh network round-trip.

    Cache entries are consumed on read — duplicate lines within a chunk fall
    through to the base extractor on second access, which is fine because the
    chunk pre-pass dedupes before priming.
    """

    def __init__(self, base: BaseExtractor) -> None:
        self._base = base
        self._cache: dict[str, ExtractedEvent] = {}

    def prime(self, lines: list[str], results: list[ExtractedEvent]) -> None:
        for line, result in zip(lines, results):
            self._cache[line] = result

    def clear(self) -> None:
        self._cache.clear()

    def extract(self, line: str, customer_id: str, config: AppConfig) -> ExtractedEvent:
        # Don't pop — duplicate lines within the chunk should all hit the cache.
        # The cache is cleared by the pipeline at batch boundaries.
        cached = self._cache.get(line)
        if cached is not None:
            return cached
        return self._base.extract(line, customer_id, config)

    def extract_many(
        self, lines: list[str], customer_id: str, config: AppConfig
    ) -> list[ExtractedEvent]:
        return self._base.extract_many(lines, customer_id, config)

    def get_usage(self) -> dict:
        if hasattr(self._base, "get_usage"):
            return self._base.get_usage()
        return {"calls": 0, "input_tokens": 0, "output_tokens": 0}


def _has_serving_parser(parsers_by_id: dict[str, Parser], sig: str) -> bool:
    """A line will short-circuit the LLM iff there's an active/canary parser
    for its cluster. Shadow parsers never short-circuit (they only observe)."""
    return any(
        p.cluster_signature == sig and p.status in ("active", "canary")
        for p in parsers_by_id.values()
    )


@dataclass
class BatchStats:
    customer_id: str
    total_lines: int = 0
    events_extracted: int = 0
    from_parser: int = 0
    from_llm: int = 0
    quarantined: int = 0
    quarantined_from_parser: int = 0
    quarantined_from_llm: int = 0
    clusters_seen: int = 0
    parsers_minted: int = 0
    parsers_promoted: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    outcomes: list[PipelineOutcome] = field(default_factory=list)
    processing_log: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        cost = estimate_cost(self.llm_input_tokens, self.llm_output_tokens, self.from_llm)
        return {
            "customer_id": self.customer_id,
            "total_lines": self.total_lines,
            "events_extracted": self.events_extracted,
            "from_parser": self.from_parser,
            "from_llm": self.from_llm,
            "quarantined": self.quarantined,
            "quarantined_from_parser": self.quarantined_from_parser,
            "quarantined_from_llm": self.quarantined_from_llm,
            "clusters_seen": self.clusters_seen,
            "parsers_minted": self.parsers_minted,
            "parsers_promoted": self.parsers_promoted,
            "llm_call_rate": (self.from_llm / self.total_lines) if self.total_lines else 0.0,
            "parser_hit_rate": (self.from_parser / self.total_lines) if self.total_lines else 0.0,
            "quarantine_rate": (self.quarantined / self.total_lines) if self.total_lines else 0.0,
            "quarantine_parser_rate": (self.quarantined_from_parser / self.total_lines) if self.total_lines else 0.0,
            "quarantine_llm_rate": (self.quarantined_from_llm / self.total_lines) if self.total_lines else 0.0,
            "quarantine_parser_share": (self.quarantined_from_parser / self.quarantined) if self.quarantined else 0.0,
            "quarantine_llm_share": (self.quarantined_from_llm / self.quarantined) if self.quarantined else 0.0,
            "llm_input_tokens": self.llm_input_tokens,
            "llm_output_tokens": self.llm_output_tokens,
            "llm_cost_usd": round(cost, 6),
            "llm_tokens_known": bool(self.llm_input_tokens or self.llm_output_tokens),
            "processing_log": self.processing_log,
        }


class BatchPipeline:
    """Reads a file (or iterable of lines), runs the full pipeline, persists outcomes."""

    def __init__(
        self,
        config: AppConfig,
        extractor: BaseExtractor | None = None,
    ) -> None:
        self.config = config
        self.repo = ParserRepository(config.storage_root)
        self.clusters = ClusterIndex(config.storage_root)
        self.writer = OutcomeWriter(config.storage_root)
        self.extractor = extractor or build_extractor()
        # Wrap once: the router calls .extract() per-line, and the pipeline
        # pre-populates the wrapper with concurrent results before each chunk.
        self._cached_extractor = _ChunkCachedExtractor(self.extractor)
        self.router = Router(config, self.repo, self.clusters, self._cached_extractor)

    def process_file(self, path: str | Path, customer_id: str) -> BatchStats:
        p = Path(path)
        with p.open("r", encoding="utf-8", errors="replace") as f:
            return self.process_lines(f, customer_id)

    def stream_lines(self, lines: Iterable[str], customer_id: str):
        """Generator that yields one dict per processed line, then a final BatchStats.

        Per-line dicts have keys: n, raw, source, event_type, fields, confidence,
        reason, pattern_type, pattern_body, parser_status, parser_id.
        The final item is the BatchStats object (not a dict).
        """
        stats = BatchStats(customer_id=customer_id)
        usage_before = self.extractor.get_usage() if hasattr(self.extractor, "get_usage") else None
        # Don't carry cached LLM results across batches — config or customer state may differ.
        self._cached_extractor.clear()

        parsers_by_id: dict[str, object] = {p.id: p for p in self.repo.list(customer_id)}
        parsers_before = set(parsers_by_id.keys())
        active_before = {pid for pid, p in parsers_by_id.items() if p.status == "active"}
        cluster_index = self.clusters.load(customer_id)

        outcome_rows: list[dict] = []
        quarantine_rows: list[dict] = []
        cluster_counts: dict[str, int] = {}

        max_bytes = self.config.lifecycle.max_line_bytes
        chunk_size = max(1, self.config.lifecycle.chunk_size)
        buf: list[tuple[str, str, str]] = []  # (raw, normalized, sig)

        def _route_one(raw: str, normalized: str, sig: str):
            cluster_counts[sig] = cluster_counts.get(sig, 0) + 1
            stats.total_lines += 1

            decision = self.router.route(
                raw_line=raw,
                normalized_line=normalized,
                cluster_signature=sig,
                customer_id=customer_id,
                parsers_by_id=parsers_by_id,
            )

            row = cluster_index.get(sig) or {"signature": sig, "count": 0, "parser_id": None}
            row["count"] = row.get("count", 0) + 1
            if decision.parser is not None:
                row["parser_id"] = decision.parser.id
            cluster_index[sig] = row

            outcome = self.router.to_outcome(
                decision=decision,
                raw_line=raw,
                normalized_line=normalized,
                cluster_signature=sig,
                customer_id=customer_id,
            )
            (quarantine_rows if decision.source == "quarantine" else outcome_rows).append(outcome.model_dump())
            stats.outcomes.append(outcome)

            if decision.source == "parser":
                stats.from_parser += 1
                stats.events_extracted += 1
            elif decision.source == "llm":
                stats.from_llm += 1
                stats.events_extracted += 1
            else:
                stats.quarantined += 1
                if decision.parser is not None:
                    stats.quarantined_from_parser += 1
                else:
                    stats.quarantined_from_llm += 1

            return {
                "n": stats.total_lines,
                "raw": raw[:200],
                "source": decision.source,
                "event_type": decision.event_type,
                "fields": decision.fields,
                "confidence": round(decision.confidence, 3),
                "reason": decision.reason,
                "pattern_type": decision.pattern_type or None,
                "pattern_body": decision.pattern_body[:800] if decision.pattern_body else None,
                "parser_status": decision.parser.status if decision.parser else None,
                "parser_id": decision.parser.id if decision.parser else None,
                "llm_event_type": decision.llm_event_type or None,
            }

        def _flush_chunk():
            if not buf:
                return
            # 1. Pick the lines that will need the LLM (no active/canary parser
            #    for their cluster) and dedupe — a chunk often repeats the same
            #    log shape many times, and one extraction covers all of them.
            seen: set[str] = set()
            to_extract: list[str] = []
            for _raw, normalized, sig in buf:
                if normalized in seen:
                    continue
                if not _has_serving_parser(parsers_by_id, sig):
                    seen.add(normalized)
                    to_extract.append(normalized)
            # 2. Fan out concurrently (OpenAI path) or fall back to a loop (mock).
            if to_extract:
                results = self._cached_extractor.extract_many(
                    to_extract, customer_id, self.config
                )
                self._cached_extractor.prime(to_extract, results)
            # 3. Serial route — same logic as before, just with cached LLM hits.
            for raw, normalized, sig in buf:
                yield _route_one(raw, normalized, sig)
            buf.clear()

        for raw in lines:
            if raw is None:
                continue
            normalized = normalize(raw)
            if not normalized:
                continue

            if len(normalized.encode("utf-8")) > max_bytes:
                # Preserve global ordering of yields by flushing first.
                yield from _flush_chunk()
                stats.total_lines += 1
                outcome = PipelineOutcome(
                    customer_id=customer_id,
                    raw_line=raw,
                    normalized_line=normalized[:max_bytes],
                    cluster_signature="",
                    event_type="unknown",
                    source="quarantine",
                    reason="exceeds max_line_bytes",
                )
                quarantine_rows.append(outcome.model_dump())
                stats.outcomes.append(outcome)
                stats.quarantined += 1
                stats.quarantined_from_llm += 1
                yield {
                    "n": stats.total_lines,
                    "raw": raw[:200],
                    "source": "quarantine",
                    "event_type": "unknown",
                    "fields": {},
                    "confidence": 0.0,
                    "reason": "exceeds max_line_bytes",
                    "pattern_type": None,
                    "pattern_body": None,
                    "parser_status": None,
                    "parser_id": None,
                }
                continue

            sig = cluster_signature(normalized)
            buf.append((raw, normalized, sig))
            if len(buf) >= chunk_size:
                yield from _flush_chunk()

        yield from _flush_chunk()

        stats.clusters_seen = len(cluster_counts)

        if usage_before is not None and hasattr(self.extractor, "get_usage"):
            usage_after = self.extractor.get_usage()
            stats.llm_input_tokens = usage_after["input_tokens"] - usage_before["input_tokens"]
            stats.llm_output_tokens = usage_after["output_tokens"] - usage_before["output_tokens"]

        self.repo.save_all(customer_id, list(parsers_by_id.values()))
        self.clusters.save(customer_id, cluster_index)
        self.writer.append_many_events(customer_id, outcome_rows)
        self.writer.append_many_quarantine(customer_id, quarantine_rows)

        active_after = {pid for pid, p in parsers_by_id.items() if p.status == "active"}
        stats.parsers_minted = len(set(parsers_by_id.keys()) - parsers_before)
        stats.parsers_promoted = len(active_after - active_before)

        logger.info(
            "batch_done",
            extra={
                "customer_id": customer_id,
                "total_lines": stats.total_lines,
                "from_parser": stats.from_parser,
                "from_llm": stats.from_llm,
                "quarantined": stats.quarantined,
                "clusters_seen": stats.clusters_seen,
            },
        )
        yield stats  # sentinel: consumer knows the batch is finished

    def process_lines(self, lines: Iterable[str], customer_id: str) -> BatchStats:
        for item in self.stream_lines(lines, customer_id):
            if isinstance(item, BatchStats):
                return item
        raise RuntimeError("stream_lines ended without yielding BatchStats")
