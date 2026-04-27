from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .agents import BaseExtractor, build_extractor
from .clusterer import cluster_signature
from .config import AppConfig
from .models import PipelineOutcome
from .normalizer import normalize
from .router import Router
from .storage import ClusterIndex, OutcomeWriter, ParserRepository, estimate_cost

logger = logging.getLogger(__name__)


_MAX_LOG_ENTRIES = 300


@dataclass
class BatchStats:
    customer_id: str
    total_lines: int = 0
    events_extracted: int = 0
    from_parser: int = 0
    from_llm: int = 0
    quarantined: int = 0
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
            "clusters_seen": self.clusters_seen,
            "parsers_minted": self.parsers_minted,
            "parsers_promoted": self.parsers_promoted,
            "llm_call_rate": (self.from_llm / self.total_lines) if self.total_lines else 0.0,
            "parser_hit_rate": (self.from_parser / self.total_lines) if self.total_lines else 0.0,
            "quarantine_rate": (self.quarantined / self.total_lines) if self.total_lines else 0.0,
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
        self.router = Router(config, self.repo, self.clusters, self.extractor)

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

        parsers_by_id: dict[str, object] = {p.id: p for p in self.repo.list(customer_id)}
        parsers_before = set(parsers_by_id.keys())
        active_before = {pid for pid, p in parsers_by_id.items() if p.status == "active"}
        cluster_index = self.clusters.load(customer_id)

        outcome_rows: list[dict] = []
        quarantine_rows: list[dict] = []
        cluster_counts: dict[str, int] = {}

        for raw in lines:
            if raw is None:
                continue
            normalized = normalize(raw)
            if not normalized:
                continue
            stats.total_lines += 1

            if len(normalized.encode("utf-8")) > self.config.lifecycle.max_line_bytes:
                outcome = PipelineOutcome(
                    customer_id=customer_id,
                    raw_line=raw,
                    normalized_line=normalized[: self.config.lifecycle.max_line_bytes],
                    cluster_signature="",
                    event_type="unknown",
                    source="quarantine",
                    reason="exceeds max_line_bytes",
                )
                quarantine_rows.append(outcome.model_dump())
                stats.outcomes.append(outcome)
                stats.quarantined += 1
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
            cluster_counts[sig] = cluster_counts.get(sig, 0) + 1

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

            yield {
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
            }

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
