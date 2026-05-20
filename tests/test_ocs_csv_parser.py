from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures import (
    build_ocs_csv,
    sample_ocs_csv_text,
    write_sample_ocs_csv,
)
from opf_app.ocs_csv import (
    OcsCsvValidationError,
    parse_ocs_csv,
    parse_ocs_csv_text,
)


def test_sample_csv_parses_into_sessions(tmp_path: Path) -> None:
    csv_path = write_sample_ocs_csv(tmp_path / "sample.csv")

    items = parse_ocs_csv(csv_path)

    assert len(items) == 6
    assert items[0].chat_date == "2026-05-11"
    assert items[0].user_identifier == "eb8649a3-aa7a-4e92-9875-6b006fc2d2fb"
    assert items[0].source_type == "ocs_csv"
    assert items[0].output_filename == (
        "redacted-transcript-2026-05-11-"
        "eb8649a3-aa7a-4e92-9875-6b006fc2d2fb.docx"
    )
    assert items[0].body_text.startswith("[2026-05-11 ")
    assert "] User: " in items[0].body_text
    assert "] Chatbot: " in items[0].body_text


def test_groups_multiple_sessions() -> None:
    csv_text = build_ocs_csv(
        [
            _row("2026-05-11T10:00:00+00:00", "human", "hello", "s1", "u1"),
            _row("2026-05-12T10:00:00+00:00", "ai", "bonjour", "s2", "u2"),
        ]
    )

    items = parse_ocs_csv_text(csv_text)

    assert [item.session_id for item in items] == ["s1", "s2"]
    assert [item.user_identifier for item in items] == ["u1", "u2"]


def test_sorts_messages_chronologically_within_session() -> None:
    csv_text = build_ocs_csv(
        [
            _row("2026-05-11T10:02:00+00:00", "ai", "second", "s1", "u1"),
            _row("2026-05-11T10:01:00+00:00", "human", "first", "s1", "u1"),
        ]
    )

    item = parse_ocs_csv_text(csv_text)[0]

    assert item.body_text.splitlines() == [
        "[2026-05-11T10:01:00+00:00] User: first",
        "[2026-05-11T10:02:00+00:00] Chatbot: second",
    ]


def test_missing_required_columns_raise_validation_error() -> None:
    csv_text = build_ocs_csv(
        [_row("2026-05-11T10:00:00+00:00", "human", "hello", "s1", "u1")],
        fieldnames=["Message Date", "Message Type", "Message Content", "Session ID"],
    )

    with pytest.raises(OcsCsvValidationError, match="Participant Public ID"):
        parse_ocs_csv_text(csv_text)


def test_unsupported_message_types_are_counted_and_excluded() -> None:
    csv_text = build_ocs_csv(
        [
            _row("2026-05-11T10:00:00+00:00", "system", "exclude me", "s1", "u1"),
            _row("2026-05-11T10:01:00+00:00", "human", "include me", "s1", "u1"),
        ]
    )

    item = parse_ocs_csv_text(csv_text)[0]

    assert item.parse_status == "warning"
    assert item.message_count == 1
    assert item.warnings == ("Skipped 1 unsupported message type row(s).",)
    assert "include me" in item.body_text
    assert "exclude me" not in item.body_text


def test_multiline_message_content_is_preserved() -> None:
    csv_text = build_ocs_csv(
        [
            _row(
                "2026-05-11T10:00:00+00:00",
                "ai",
                "line one\nline two",
                "s1",
                "u1",
            )
        ]
    )

    item = parse_ocs_csv_text(csv_text)[0]

    assert item.body_text == "[2026-05-11T10:00:00+00:00] Chatbot: line one\nline two"


def test_csv_text_helper_exercises_parser_without_local_docs() -> None:
    items = parse_ocs_csv_text(sample_ocs_csv_text())

    assert len(items) == 6
    assert items[0].message_count == 2


def _row(
    message_date: str,
    message_type: str,
    message_content: str,
    session_id: str,
    participant_public_id: str,
) -> dict[str, str]:
    return {
        "Message Date": message_date,
        "Message Type": message_type,
        "Message Content": message_content,
        "Session ID": session_id,
        "Participant Public ID": participant_public_id,
    }
