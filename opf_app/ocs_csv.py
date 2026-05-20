from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import TextIO

from .filenames import OutputFilenameCandidate, build_unique_output_filenames
from .models import ParsedItem, ParseStatus


REQUIRED_COLUMNS = frozenset(
    {
        "Message Date",
        "Message Type",
        "Message Content",
        "Session ID",
        "Participant Public ID",
    }
)
ROLE_LABELS = {
    "human": "User",
    "ai": "Chatbot",
}


class OcsCsvValidationError(ValueError):
    """Raised when an OCS CSV cannot be safely parsed."""


@dataclass(frozen=True)
class _OcsRow:
    row_number: int
    message_date: str
    parsed_date: datetime
    message_type: str
    message_content: str = field(repr=False)
    session_id: str
    participant_public_id: str


@dataclass(frozen=True)
class _SessionDraft:
    item_name: str
    session_id: str
    chat_date: str
    user_identifier: str
    body_text: str = field(repr=False)
    message_count: int
    character_count: int
    parse_status: ParseStatus
    warnings: tuple[str, ...]


def parse_ocs_csv(path: str | Path) -> tuple[ParsedItem, ...]:
    """Parse an OCS CSV export file into one item per chat session."""
    source_path = Path(path)
    with source_path.open(newline="", encoding="utf-8-sig") as stream:
        return parse_ocs_csv_stream(
            stream,
            source_name=source_path.name,
            source_path=str(source_path),
        )


def parse_ocs_csv_text(
    csv_text: str,
    *,
    source_name: str = "uploaded.csv",
) -> tuple[ParsedItem, ...]:
    """Parse OCS CSV content that is already loaded in memory."""
    return parse_ocs_csv_stream(StringIO(csv_text), source_name=source_name)


def parse_ocs_csv_stream(
    stream: TextIO,
    *,
    source_name: str,
    source_path: str | None = None,
) -> tuple[ParsedItem, ...]:
    """Parse an OCS CSV stream into one item per chat session."""
    reader = csv.DictReader(stream)
    _validate_required_columns(reader.fieldnames)

    rows_by_session: dict[str, list[_OcsRow]] = {}
    for row_number, row in enumerate(reader, start=2):
        parsed_row = _parse_row(row, row_number)
        rows_by_session.setdefault(parsed_row.session_id, []).append(parsed_row)

    drafts = [
        _build_session_draft(position, session_id, rows)
        for position, (session_id, rows) in enumerate(rows_by_session.items(), start=1)
    ]
    filenames = build_unique_output_filenames(
        [
            OutputFilenameCandidate(
                draft.chat_date,
                draft.user_identifier,
                draft.session_id,
            )
            for draft in drafts
        ]
    )

    return tuple(
        ParsedItem(
            item_name=draft.item_name,
            source_name=source_name,
            source_type="ocs_csv",
            chat_date=draft.chat_date,
            user_identifier=draft.user_identifier,
            body_text=draft.body_text,
            output_filename=filename,
            parse_status=draft.parse_status,
            session_id=draft.session_id,
            source_path=source_path,
            message_count=draft.message_count,
            character_count=draft.character_count,
            warnings=draft.warnings,
        )
        for draft, filename in zip(drafts, filenames)
    )


def _validate_required_columns(fieldnames: list[str] | None) -> None:
    if fieldnames is None:
        raise OcsCsvValidationError("CSV is missing a header row.")

    missing = sorted(REQUIRED_COLUMNS.difference(fieldnames))
    if missing:
        raise OcsCsvValidationError(
            "OCS CSV is missing required column(s): " + ", ".join(missing)
        )


def _parse_row(row: dict[str, str], row_number: int) -> _OcsRow:
    session_id = (row.get("Session ID") or "").strip()
    if not session_id:
        raise OcsCsvValidationError(f"Row {row_number} is missing Session ID.")

    raw_date = (row.get("Message Date") or "").strip()
    if not raw_date:
        raise OcsCsvValidationError(f"Row {row_number} is missing Message Date.")

    try:
        parsed_date = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
    except ValueError as exc:
        raise OcsCsvValidationError(
            f"Row {row_number} has an invalid Message Date."
        ) from exc

    return _OcsRow(
        row_number=row_number,
        message_date=raw_date,
        parsed_date=parsed_date,
        message_type=(row.get("Message Type") or "").strip().lower(),
        message_content=row.get("Message Content") or "",
        session_id=session_id,
        participant_public_id=(row.get("Participant Public ID") or "").strip(),
    )


def _build_session_draft(
    position: int,
    session_id: str,
    rows: list[_OcsRow],
) -> _SessionDraft:
    sorted_rows = sorted(rows, key=lambda row: (row.parsed_date, row.row_number))
    included_rows = [
        row for row in sorted_rows if row.message_type in ROLE_LABELS
    ]
    skipped_count = len(sorted_rows) - len(included_rows)
    transcript_lines = [
        f"[{row.message_date}] {ROLE_LABELS[row.message_type]}: {row.message_content}"
        for row in included_rows
    ]
    body_text = "\n".join(transcript_lines)
    user_identifier = next(
        (
            row.participant_public_id
            for row in sorted_rows
            if row.participant_public_id
        ),
        session_id,
    )
    warnings: tuple[str, ...] = ()
    parse_status: ParseStatus = "parsed"
    if skipped_count:
        warnings = (f"Skipped {skipped_count} unsupported message type row(s).",)
        parse_status = "warning"

    return _SessionDraft(
        item_name=f"Session {position}",
        session_id=session_id,
        chat_date=sorted_rows[0].parsed_date.date().isoformat(),
        user_identifier=user_identifier,
        body_text=body_text,
        message_count=len(included_rows),
        character_count=len(body_text),
        parse_status=parse_status,
        warnings=warnings,
    )
