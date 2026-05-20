from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
from pathlib import Path
import re
import unicodedata
from typing import TypeAlias


DateInput: TypeAlias = str | date | datetime | None

OUTPUT_EXTENSION = ".docx"
OUTPUT_PREFIX = "redacted-transcript"

_DATE_PATTERN = re.compile(r"(?<!\d)((?:19|20)\d{2})[-_](\d{1,2})[-_](\d{1,2})(?!\d)")
_DOCUMENT_IDENTIFIER_PATTERN = re.compile(r"(?<![A-Za-z0-9])([Pp]\d+[A-Za-z0-9-]*)(?![A-Za-z0-9])")
_UNSAFE_PATH_CHARS = re.compile(r"[<>:\"/\\|?*\x00-\x1f]")
_UNSAFE_COMPONENT_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_SEPARATOR_RUN = re.compile(r"[-\s]+")
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


@dataclass(frozen=True)
class OutputFilenameCandidate:
    """Inputs needed to produce one deterministic output filename."""

    chat_date: DateInput
    user_identifier: str
    discriminator: str | None = None


def normalize_chat_date(
    value: DateInput,
    *,
    processing_date: date | None = None,
) -> str:
    """Normalize a date-like value to YYYY-MM-DD."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()

    if value is not None:
        text = str(value).strip()
        match = _DATE_PATTERN.search(text)
        if match:
            year, month, day = (int(part) for part in match.groups())
            try:
                return date(year, month, day).isoformat()
            except ValueError:
                pass

        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            pass

    return (processing_date or date.today()).isoformat()


def infer_chat_date_from_stem(
    stem: str,
    *,
    processing_date: date | None = None,
) -> str:
    """Infer a chat date from a document stem, falling back to processing date."""
    return normalize_chat_date(stem, processing_date=processing_date)


def sanitize_filename_component(
    value: object,
    *,
    fallback: str = "unknown",
    max_length: int = 120,
) -> str:
    """Return a filesystem-safe filename component."""
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = _UNSAFE_PATH_CHARS.sub(" ", text)
    text = _UNSAFE_COMPONENT_CHARS.sub("-", text)
    text = _SEPARATOR_RUN.sub("-", text)
    safe = text.strip(" .-_")

    if not safe:
        safe = fallback if fallback != str(value) else "unknown"
        safe = sanitize_filename_component(safe, fallback="unknown", max_length=max_length)

    if safe.upper() in _WINDOWS_RESERVED_NAMES:
        safe = f"{safe}-file"

    if len(safe) > max_length:
        safe = safe[:max_length].rstrip(" .-_") or "unknown"
    return safe


def infer_user_identifier_from_stem(stem: str) -> str:
    """Infer a useful document identifier, falling back to the filename stem."""
    source_stem = Path(stem).stem
    match = _DOCUMENT_IDENTIFIER_PATTERN.search(source_stem)
    if match:
        return sanitize_filename_component(match.group(1))
    return sanitize_filename_component(source_stem)


def stable_suffix(value: object, *, length: int = 8) -> str:
    """Build a short deterministic suffix from a session id or source path."""
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return digest[:length]


def build_output_filename(
    chat_date: DateInput,
    user_identifier: str,
    *,
    duplicate_key: str | None = None,
    processing_date: date | None = None,
) -> str:
    """Build the unified redacted transcript output filename."""
    stem = _build_output_stem(
        chat_date,
        user_identifier,
        processing_date=processing_date,
    )
    if duplicate_key:
        stem = f"{stem}-{stable_suffix(duplicate_key)}"
    return f"{stem}{OUTPUT_EXTENSION}"


def build_unique_output_filenames(
    candidates: list[OutputFilenameCandidate] | tuple[OutputFilenameCandidate, ...],
    *,
    processing_date: date | None = None,
) -> tuple[str, ...]:
    """Build filenames, adding deterministic suffixes only for duplicates."""
    base_stems = [
        _build_output_stem(
            candidate.chat_date,
            candidate.user_identifier,
            processing_date=processing_date,
        )
        for candidate in candidates
    ]
    counts = Counter(base_stems)
    used_stems: set[str] = set()
    filenames: list[str] = []

    for index, (candidate, base_stem) in enumerate(zip(candidates, base_stems)):
        stem = base_stem
        if counts[base_stem] > 1:
            discriminator = candidate.discriminator or f"{base_stem}:{index}"
            stem = f"{base_stem}-{stable_suffix(discriminator)}"
            while stem in used_stems:
                discriminator = f"{discriminator}:{index}"
                stem = f"{base_stem}-{stable_suffix(discriminator)}"
        used_stems.add(stem)
        filenames.append(f"{stem}{OUTPUT_EXTENSION}")

    return tuple(filenames)


def document_output_filename(
    source_name: str,
    *,
    duplicate_key: str | None = None,
    processing_date: date | None = None,
) -> str:
    """Build an output filename from document filename metadata."""
    stem = Path(source_name).stem
    return build_output_filename(
        infer_chat_date_from_stem(stem, processing_date=processing_date),
        infer_user_identifier_from_stem(stem),
        duplicate_key=duplicate_key,
        processing_date=processing_date,
    )


def _build_output_stem(
    chat_date: DateInput,
    user_identifier: str,
    *,
    processing_date: date | None = None,
) -> str:
    normalized_date = normalize_chat_date(chat_date, processing_date=processing_date)
    safe_identifier = sanitize_filename_component(user_identifier)
    return f"{OUTPUT_PREFIX}-{normalized_date}-{safe_identifier}"
