from __future__ import annotations

import re

# Demo-grade token-mask clusterer (Drain3-spirit, without the dependency).
# Rationale: batch demo, no shared state, fits the "basic txt files" storage ask.
# For production, swap in drain3 and persist its state per customer.

_UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_ISO_TS_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b")
_EPOCH_MS_RE = re.compile(r"\b\d{13}\b")
_EPOCH_S_RE = re.compile(r"\b\d{10}\b")
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_HEX_RE = re.compile(r"\b[0-9a-fA-F]{12,}\b")
_NUM_RE = re.compile(r"(?<!\w)-?\d+(?:\.\d+)?(?!\w)")
_QUOTED_RE = re.compile(r'"[^"]*"|\'[^\']*\'')
# Mask prefixed IDs with separator first: prod_456, user-789 → prod_<ID>, user-<ID>
_PREFIXED_ID_RE = re.compile(r"\b([A-Za-z]+[_-])[A-Za-z0-9]+\b")
# Mask bare alphanumeric IDs: u123, sku001, abc42 → <ALNUM>
_ALNUM_ID_RE = re.compile(r"\b[A-Za-z]+\d[A-Za-z0-9]*\b")


def _mask(s: str) -> str:
    s = _QUOTED_RE.sub("<STR>", s)
    s = _ISO_TS_RE.sub("<TS>", s)
    s = _UUID_RE.sub("<UUID>", s)
    s = _EPOCH_MS_RE.sub("<TS>", s)
    s = _EPOCH_S_RE.sub("<TS>", s)
    s = _IP_RE.sub("<IP>", s)
    s = _HEX_RE.sub("<HEX>", s)
    # Prefixed IDs first (prod_456), then bare alphanumeric (u123), then raw numbers.
    s = _PREFIXED_ID_RE.sub(r"\1<ID>", s)
    s = _ALNUM_ID_RE.sub("<ALNUM>", s)
    s = _NUM_RE.sub("<NUM>", s)
    # Collapse whitespace.
    s = re.sub(r"\s+", " ", s).strip()
    return s


def cluster_signature(normalized_line: str) -> str:
    """Stable signature for near-duplicate collapsing.

    Same template → same signature regardless of specific IDs, timestamps, numbers.
    """
    return _mask(normalized_line)
