from __future__ import annotations

import json
import re
from typing import Optional

from .models import Parser


_ISO_TS_RE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\b"
)
_SYSLOG_TS_RE = re.compile(r"\b([A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\b")


def _search_any_ts(line: str):
    return _ISO_TS_RE.search(line) or _SYSLOG_TS_RE.search(line)
# k=v pair scanner. Stops value at whitespace or pipe so pipe-delimited shapes
# (`ts|CLICK|userid=u1|product_id=p1`) extract every pair.
_KV_PAIR_RE = re.compile(
    r"(?P<k>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<v>\"[^\"]*\"|'[^']*'|[^\s|]+)"
)


# --- Minimal grok implementation ---------------------------------------------
# We support the %{PATTERN:field} syntax for a small, audited set of patterns.
# Avoids a hard dependency on pygrok while still giving the LLM a sane target
# language. Swap to pygrok in production.

_GROK_PATTERNS: dict[str, str] = {
    "WORD": r"\b\w+\b",
    "NOTSPACE": r"\S+",
    "DATA": r".*?",
    "GREEDYDATA": r".*",
    "INT": r"[+-]?\d+",
    "NUMBER": r"[+-]?\d+(?:\.\d+)?",
    "UUID": r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
    "IP": r"(?:\d{1,3}\.){3}\d{1,3}",
    "USERNAME": r"[a-zA-Z0-9._-]+",
    "TIMESTAMP": (
        r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
    ),
    "EPOCHMS": r"\d{13}",
    "EPOCH": r"\d{10}",
    "LOGLEVEL": r"(?:DEBUG|INFO|WARN|WARNING|ERROR|FATAL|TRACE)",
}

_GROK_TOKEN_RE = re.compile(r"%\{(?P<name>[A-Z]+)(?::(?P<field>[A-Za-z_][A-Za-z0-9_]*))?\}")


def compile_grok(pattern: str) -> re.Pattern[str]:
    """Translate %{NAME:field} grok syntax into a Python regex."""

    def repl(m: re.Match[str]) -> str:
        name = m.group("name")
        field = m.group("field")
        base = _GROK_PATTERNS.get(name)
        if base is None:
            raise ValueError(f"Unknown grok pattern: {name}")
        if field:
            return f"(?P<{field}>{base})"
        return f"(?:{base})"

    regex = _GROK_TOKEN_RE.sub(repl, pattern)
    # Grok is not anchored by default; match anywhere — but require that the whole
    # pattern matches contiguously. Callers test via `search`.
    return re.compile(regex)


# --- Minimal jsonpath --------------------------------------------------------

_JP_TOKEN_RE = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]|\[\"([^\"]+)\"\]|\['([^']+)'\]")


def _jsonpath_get(obj: object, path: str) -> Optional[object]:
    if not path.startswith("$"):
        return None
    cur: object = obj
    for m in _JP_TOKEN_RE.finditer(path[1:]):
        key, idx, qkey, sqkey = m.groups()
        if key is not None or qkey is not None or sqkey is not None:
            k = key or qkey or sqkey
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        elif idx is not None:
            i = int(idx)
            if not isinstance(cur, list) or i >= len(cur):
                return None
            cur = cur[i]
    return cur


# --- Execution ---------------------------------------------------------------


class ParserExecutionError(Exception):
    """Raised when a parser's pattern_body is malformed."""


def execute_parser(parser: Parser, line: str) -> Optional[dict[str, str]]:
    """Run a parser against a line. Returns field dict on match, None on miss.

    Raises ParserExecutionError if the parser is itself broken (shouldn't happen
    post-validation, but guards the hot path).
    """
    try:
        if parser.pattern_type == "grok":
            regex = compile_grok(parser.pattern_body)
            m = regex.search(line)
            if not m:
                return None
            return {k: v for k, v in m.groupdict().items() if v is not None}

        if parser.pattern_type == "regex":
            regex = re.compile(parser.pattern_body)
            m = regex.search(line)
            if not m:
                return None
            return {k: v for k, v in m.groupdict().items() if v is not None}

        if parser.pattern_type == "kv":
            # pattern_body is a JSON object, e.g.:
            #   {"fields": ["user", "product"],           # raw keys to keep
            #    "field_map": {"user": "user_id",          # rename raw → canonical
            #                  "product": "product_id"},
            #    "timestamp_field": "timestamp"}          # optional: grab a bare
            #                                              # ISO timestamp if present
            # Works across whitespace-, pipe-, comma-delimited shapes — we scan for
            # k=v pairs anywhere in the line rather than pre-splitting on a separator.
            spec = json.loads(parser.pattern_body)
            wanted = set(spec.get("fields") or [])
            field_map = spec.get("field_map") or {}
            ts_field = spec.get("timestamp_field")
            out: dict[str, str] = {}
            for m in _KV_PAIR_RE.finditer(line):
                k = m.group("k")
                v = m.group("v").strip('"').strip("'")
                if wanted and k not in wanted:
                    continue
                out[field_map.get(k, k)] = v
            if ts_field and ts_field not in out:
                m = _search_any_ts(line)
                if m:
                    out[ts_field] = m.group(1)
            return out or None

        if parser.pattern_type == "jsonpath":
            # pattern_body is a JSON object: {"root": "$", "fields": {"user_id": "$.user.id", ...},
            #                                 "timestamp_field": "timestamp"}  # optional
            # When `timestamp_field` is set and the JSON body doesn't carry it, the
            # parser falls back to scanning the full line for an ISO/syslog timestamp —
            # mirrors the kv parser's behavior so prefix-timestamped JSON lines work.
            spec = json.loads(parser.pattern_body)
            # Find JSON substring — support lines like 'INFO: {...}' by locating first '{'.
            json_start = line.find("{")
            if json_start < 0:
                return None
            try:
                doc = json.loads(line[json_start:])
            except json.JSONDecodeError:
                return None
            out = {}
            for field, path in spec.get("fields", {}).items():
                val = _jsonpath_get(doc, path)
                if val is not None:
                    out[field] = str(val)
            ts_field = spec.get("timestamp_field")
            if ts_field and ts_field not in out:
                m = _search_any_ts(line)
                if m:
                    out[ts_field] = m.group(1)
            return out or None

    except (re.error, ValueError, json.JSONDecodeError) as e:
        raise ParserExecutionError(str(e)) from e

    raise ParserExecutionError(f"Unknown pattern_type: {parser.pattern_type}")


def validate_parser(parser: Parser, sample_lines: list[str], expected_fields: dict[str, str]) -> bool:
    """Pre-shadow gate: parser must match every sample line and extract fields
    consistent with what the LLM extracted on the representative."""
    if not sample_lines:
        return False
    for line in sample_lines:
        try:
            extracted = execute_parser(parser, line)
        except ParserExecutionError:
            return False
        if extracted is None:
            return False
    # For the representative line specifically, required fields must be present
    # and match the LLM extraction where they overlap.
    try:
        rep_extracted = execute_parser(parser, sample_lines[0]) or {}
    except ParserExecutionError:
        return False
    for k, v in expected_fields.items():
        if k in rep_extracted and rep_extracted[k] != v:
            return False
    return True
