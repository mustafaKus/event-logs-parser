from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


EventType = str  # validated against AppConfig.event_names at runtime

ParserStatus = Literal["shadow", "canary", "active", "retired"]
PatternType = Literal["grok", "jsonpath", "kv", "regex"]
CreatedBy = Literal["llm", "human"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProposedParser(BaseModel):
    pattern_type: PatternType
    pattern_body: str
    rationale: str = ""

    # The LLM often emits pattern_body as a JSON object/array (matching the
    # shape we describe in the prompt) instead of the JSON-encoded string the
    # downstream parser executors expect. Coerce here so strict TypeAdapter
    # validation in the Agents SDK doesn't reject otherwise-valid proposals.
    @field_validator("pattern_body", mode="before")
    @classmethod
    def _stringify_pattern_body(cls, v):
        if isinstance(v, (dict, list)):
            return json.dumps(v)
        return v


class ExtractedEvent(BaseModel):
    """Strict LLM output contract. Mirrors the JSON contract in the system prompt."""

    event_type: str = Field(description="One of the canonical event types, or 'unknown'.")
    fields: dict[str, str] = Field(default_factory=dict)
    proposed_parser: Optional[ProposedParser] = None
    confidence: float = 0.0
    notes: str = ""

    # The LLM frequently emits numeric scalars for quantity/total even when the
    # prompt asks for strings. Coerce here so strict TypeAdapter validation in
    # the Agents SDK doesn't reject otherwise-valid extractions.
    @field_validator("fields", mode="before")
    @classmethod
    def _stringify_field_values(cls, v):
        if isinstance(v, dict):
            return {k: str(val) if val is not None else "" for k, val in v.items()}
        return v


class Parser(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    customer_id: str
    event_type: str
    pattern_type: PatternType
    pattern_body: str
    status: ParserStatus = "shadow"
    created_by: CreatedBy = "llm"
    observed_count: int = 0
    llm_agreement_count: int = 0
    llm_disagreement_count: int = 0
    last_matched_at: Optional[str] = None
    confidence_score: float = 0.0
    version: int = 1
    created_at: str = Field(default_factory=_now)
    cluster_signature: Optional[str] = None  # ties the parser to the cluster that birthed it

    def record_match(self, agreed_with_llm: Optional[bool] = None) -> None:
        self.observed_count += 1
        self.last_matched_at = _now()
        if agreed_with_llm is True:
            self.llm_agreement_count += 1
        elif agreed_with_llm is False:
            self.llm_disagreement_count += 1

    @property
    def agreement_ratio(self) -> float:
        total = self.llm_agreement_count + self.llm_disagreement_count
        return self.llm_agreement_count / total if total > 0 else 1.0


class PipelineOutcome(BaseModel):
    """Per-line pipeline result. Written to events.jsonl or quarantine.jsonl."""

    customer_id: str
    raw_line: str
    normalized_line: str
    cluster_signature: str
    event_type: str
    fields: dict[str, str] = Field(default_factory=dict)
    source: Literal["parser", "llm", "quarantine"] = "quarantine"
    parser_id: Optional[str] = None
    parser_status_at_match: Optional[ParserStatus] = None
    confidence: float = 0.0
    processed_at: str = Field(default_factory=_now)
    reason: str = ""  # why it landed where it did (esp. useful for quarantine)
    # Debug fields — populated for the processing-log panel in the UI.
    llm_prompt: str = ""          # the prompt that was (or would be) sent to the LLM
    pattern_type: str = ""        # pattern type used/proposed (kv/grok/regex/jsonpath)
    pattern_body: str = ""        # the actual pattern string
    # When the line was quarantined but the LLM still identified an event type,
    # we keep that here so the UI can route the failed attempt under the right
    # event card's expression history.
    llm_event_type: str = ""
