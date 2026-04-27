from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .agents import BaseExtractor
from .config import AppConfig
from .models import ExtractedEvent, Parser, PipelineOutcome
from .parsers import ParserExecutionError, execute_parser, validate_parser
from .storage import ClusterIndex, OutcomeWriter, ParserRepository

logger = logging.getLogger(__name__)


@dataclass
class RouteDecision:
    source: str  # "parser" | "llm" | "quarantine"
    event_type: str
    fields: dict[str, str]
    parser: Parser | None = None
    llm_result: ExtractedEvent | None = None
    reason: str = ""
    confidence: float = 0.0
    llm_prompt: str = ""      # prompt built/sent for LLM calls
    pattern_type: str = ""    # pattern type used or proposed
    pattern_body: str = ""    # the actual pattern body


class Router:
    """Parser library first, LLM fallback, quarantine last.

    Keeps a small per-call in-memory view of the customer's parsers to avoid
    re-reading the JSONL on every line. The view is refreshed by the pipeline
    after each batch flush.
    """

    def __init__(
        self,
        config: AppConfig,
        repo: ParserRepository,
        clusters: ClusterIndex,
        extractor: BaseExtractor,
    ) -> None:
        self.config = config
        self.repo = repo
        self.clusters = clusters
        self.extractor = extractor

    # -- prompt builder (for logging / UI) --------------------------------

    def _build_prompt(self, line: str) -> str:
        catalog = "\n".join(
            f"  {name}: required={spec.required_fields}, optional={spec.optional_fields}"
            for name, spec in self.config.events.items()
        )
        return (
            "Extract a structured product-analytics event from the log line.\n"
            "Return JSON: {event_type, fields, proposed_parser, confidence, notes}\n"
            "If unsure, set event_type='unknown' and confidence<0.5.\n\n"
            f"Canonical events:\n{catalog}\n\n"
            f"Log line:\n{line}"
        )

    # -- parser library lookup -------------------------------------------

    def _applicable_parsers(self, parsers: list[Parser], cluster_signature: str) -> list[Parser]:
        # Prefer parsers tied to this cluster; fall back to any active parser as a safety net.
        same_cluster = [p for p in parsers if p.cluster_signature == cluster_signature and p.status != "retired"]
        if same_cluster:
            # Active first, then canary, then shadow.
            order = {"active": 0, "canary": 1, "shadow": 2}
            same_cluster.sort(key=lambda p: order.get(p.status, 99))
            return same_cluster
        return []

    def route(
        self,
        raw_line: str,
        normalized_line: str,
        cluster_signature: str,
        customer_id: str,
        parsers_by_id: dict[str, Parser],
    ) -> RouteDecision:
        """`parsers_by_id` is a live, mutable per-batch view. Mutations are
        persisted by the pipeline at batch flush — not per-line, to avoid
        O(N_parsers) disk ops per log line."""
        parsers = self._applicable_parsers(list(parsers_by_id.values()), cluster_signature)

        # 1. Active/canary parsers answer first.
        for p in parsers:
            if p.status in ("active", "canary"):
                try:
                    fields = execute_parser(p, normalized_line)
                except ParserExecutionError as e:
                    logger.warning(
                        "parser_exec_error",
                        extra={"customer_id": customer_id, "parser_id": p.id, "err": str(e)},
                    )
                    continue
                if fields is None:
                    continue
                spec = self.config.events.get(p.event_type)
                if spec is None or any(f not in fields for f in spec.required_fields):
                    continue
                return RouteDecision(
                    source="parser",
                    event_type=p.event_type,
                    fields={k: v for k, v in fields.items() if k in spec.all_fields},
                    parser=p,
                    reason="matched active/canary parser",
                    confidence=p.confidence_score or 0.9,
                    pattern_type=p.pattern_type,
                    pattern_body=p.pattern_body,
                )

        # 2. No active parser → call the LLM.
        prompt = self._build_prompt(normalized_line)
        llm = self.extractor.extract(normalized_line, customer_id, self.config)

        # Output guardrail: event_type must be canonical and required fields present.
        if llm.event_type == "unknown" or llm.event_type not in self.config.events:
            return RouteDecision(
                source="quarantine",
                event_type="unknown",
                fields={},
                llm_result=llm,
                reason=llm.notes or "llm returned unknown",
                confidence=llm.confidence,
                llm_prompt=prompt,
            )
        spec = self.config.events[llm.event_type]
        missing = [f for f in spec.required_fields if f not in llm.fields]
        if missing:
            return RouteDecision(
                source="quarantine",
                event_type="unknown",
                fields={},
                llm_result=llm,
                reason=f"missing required fields: {missing}",
                confidence=llm.confidence,
                llm_prompt=prompt,
            )

        # 3. Shadow parsers: consult them and compare against the LLM. Promote on threshold.
        shadow_match_used: Optional[Parser] = None
        for p in [p for p in parsers if p.status == "shadow"]:
            try:
                s_fields = execute_parser(p, normalized_line)
            except ParserExecutionError:
                continue
            if s_fields is None:
                p.record_match(agreed_with_llm=False)
                shadow_match_used = p
                continue
            # Agreement: parser extracted same values for required fields.
            agrees = all(s_fields.get(f) == llm.fields.get(f) for f in spec.required_fields)
            p.record_match(agreed_with_llm=agrees)
            shadow_match_used = p
            # Promote on threshold.
            if (
                p.observed_count >= self.config.lifecycle.shadow_to_active_matches
                and p.agreement_ratio >= self.config.lifecycle.shadow_to_active_agreement
            ):
                p.status = "active"
                p.confidence_score = max(p.confidence_score, llm.confidence)
                logger.info(
                    "parser_promoted",
                    extra={
                        "customer_id": customer_id,
                        "parser_id": p.id,
                        "event_type": p.event_type,
                    },
                )
            parsers_by_id[p.id] = p

        if shadow_match_used is None and llm.proposed_parser is not None:
            # Mint a new shadow parser from the LLM's proposal — after pre-shadow validation.
            candidate = Parser(
                customer_id=customer_id,
                event_type=llm.event_type,
                pattern_type=llm.proposed_parser.pattern_type,
                pattern_body=llm.proposed_parser.pattern_body,
                status="shadow",
                created_by="llm",
                cluster_signature=cluster_signature,
                confidence_score=llm.confidence,
            )
            # Pre-shadow gate: the parser must re-extract the LLM's fields from this line.
            try:
                if validate_parser(candidate, [normalized_line], llm.fields):
                    candidate.record_match(agreed_with_llm=True)
                    parsers_by_id[candidate.id] = candidate
                    logger.info(
                        "parser_minted",
                        extra={
                            "customer_id": customer_id,
                            "parser_id": candidate.id,
                            "event_type": candidate.event_type,
                            "pattern_type": candidate.pattern_type,
                        },
                    )
            except ParserExecutionError as e:
                logger.warning(
                    "parser_proposal_invalid",
                    extra={"customer_id": customer_id, "err": str(e)},
                )

        proposed_type = llm.proposed_parser.pattern_type if llm.proposed_parser else ""
        proposed_body = llm.proposed_parser.pattern_body if llm.proposed_parser else ""
        return RouteDecision(
            source="llm",
            event_type=llm.event_type,
            fields={k: v for k, v in llm.fields.items() if k in spec.all_fields},
            llm_result=llm,
            reason="llm extraction",
            confidence=llm.confidence,
            llm_prompt=prompt,
            pattern_type=proposed_type,
            pattern_body=proposed_body,
        )

    # -- outcome assembly -------------------------------------------------

    def to_outcome(
        self,
        decision: RouteDecision,
        raw_line: str,
        normalized_line: str,
        cluster_signature: str,
        customer_id: str,
    ) -> PipelineOutcome:
        return PipelineOutcome(
            customer_id=customer_id,
            raw_line=raw_line,
            normalized_line=normalized_line,
            cluster_signature=cluster_signature,
            event_type=decision.event_type,
            fields=decision.fields,
            source=decision.source,  # type: ignore[arg-type]
            parser_id=decision.parser.id if decision.parser else None,
            parser_status_at_match=decision.parser.status if decision.parser else None,
            confidence=decision.confidence,
            reason=decision.reason,
            llm_prompt=decision.llm_prompt,
            pattern_type=decision.pattern_type,
            pattern_body=decision.pattern_body,
        )
