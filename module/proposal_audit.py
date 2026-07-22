"""Small, dependency-free helpers for auditable Alpamayo proposals."""

from __future__ import annotations

import hashlib


def coc_audit_fields(coc_text: str | None) -> dict[str, object]:
    """Return the full CoC and stable integrity metadata for JSONL telemetry.

    Display code may shorten the text, but the proposal event must retain the
    extracted model output byte-for-byte so rejected plans remain inspectable.
    """

    text = "" if coc_text is None else str(coc_text)
    encoded = text.encode("utf-8")
    return {
        "coc_text_full": text,
        "coc_sha256": hashlib.sha256(encoded).hexdigest(),
        "coc_character_count": len(text),
        "coc_utf8_byte_count": len(encoded),
        "coc_truncated": False,
    }
