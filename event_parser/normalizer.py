from __future__ import annotations

import re

_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def normalize(line: str) -> str:
    """Strip ANSI escapes, trailing newline noise, collapse internal whitespace lightly.

    We intentionally do not lower-case or strip quotes — those carry signal for the parser
    and the LLM. Normalization is only about removing ambient junk so identical content in
    different terminals clusters together.
    """
    if not isinstance(line, str):
        return ""
    s = _ANSI_RE.sub("", line)
    s = s.replace("\r", "").strip("\n")
    # Preserve leading/trailing spaces that might be significant for regex anchors? No —
    # trim. Grok patterns built from these lines shouldn't depend on incidental padding.
    s = s.strip()
    return s
