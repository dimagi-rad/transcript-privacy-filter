from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

from docx import Document


OCS_FIELDNAMES = [
    "Message Date",
    "Message Type",
    "Message Content",
    "Session ID",
    "Participant Public ID",
]


def sample_ocs_csv_text() -> str:
    rows = []
    for index in range(6):
        rows.extend(
            [
                _ocs_row(
                    f"2026-05-11 09:{index:02d}:00+00:00",
                    "human",
                    f"hello {index}",
                    f"session-{index}",
                    _participant_id(index),
                ),
                _ocs_row(
                    f"2026-05-11 09:{index:02d}:30+00:00",
                    "ai",
                    f"reply {index}",
                    f"session-{index}",
                    _participant_id(index),
                ),
            ]
        )
    return build_ocs_csv(rows)


def write_sample_ocs_csv(path: Path) -> Path:
    path.write_text(sample_ocs_csv_text(), encoding="utf-8")
    return path


def write_sample_docx(path: Path) -> Path:
    document = Document()
    table = document.add_table(rows=4, cols=3)
    table.rows[0].cells[0].text = "Transcription details"
    table.rows[0].cells[1].text = "metadata"
    table.rows[1].cells[0].text = "Input sound file"
    table.rows[1].cells[1].text = "recording.wav"
    table.rows[2].cells[0].text = "S1:"
    table.rows[2].cells[1].text = "00:00"
    table.rows[2].cells[2].text = "Hello."
    table.rows[3].cells[0].text = "S2:"
    table.rows[3].cells[1].text = "00:13"
    table.rows[3].cells[2].text = "Thanks."
    document.save(path)
    return path


def build_ocs_csv(
    rows: list[dict[str, str]],
    *,
    fieldnames: list[str] | None = None,
) -> str:
    fieldnames = OCS_FIELDNAMES if fieldnames is None else fieldnames
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _ocs_row(
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


def _participant_id(index: int) -> str:
    if index == 0:
        return "eb8649a3-aa7a-4e92-9875-6b006fc2d2fb"
    return f"participant-{index}"
