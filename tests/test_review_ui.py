from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfWriter

from tests.fixtures import sample_ocs_csv_text, write_sample_docx, write_sample_ocs_csv
from opf_app.ocs_csv import parse_ocs_csv
from opf_app.ui import (
    parse_csv_upload_bytes,
    parse_document_folder_path,
    parse_uploaded_document_files,
    prepare_parse_error_table,
    prepare_review_table,
)


def test_review_table_excludes_transcript_body_for_sample_csv(tmp_path: Path) -> None:
    items = parse_ocs_csv(write_sample_ocs_csv(tmp_path / "sample.csv"))

    rows = prepare_review_table(items)

    assert len(rows) == 6
    assert rows[0]["Chat date"] == "2026-05-11"
    assert rows[0]["User identifier"] == "eb8649a3-aa7a-4e92-9875-6b006fc2d2fb"
    assert "body_text" not in rows[0]
    assert "Message Content" not in rows[0]


def test_csv_upload_bytes_parses_sample_csv() -> None:
    data = sample_ocs_csv_text().encode("utf-8")

    outcome = parse_csv_upload_bytes(data, source_name="example-input.csv")

    assert outcome.errors == ()
    assert len(outcome.items) == 6


def test_document_folder_path_parses_sample_docx(tmp_path: Path) -> None:
    write_sample_docx(tmp_path / "P695 Interview 2026_01_07.docx")

    outcome = parse_document_folder_path(str(tmp_path))

    assert outcome.errors == ()
    assert len(outcome.items) == 1
    assert outcome.items[0].chat_date == "2026-01-07"
    assert outcome.items[0].user_identifier == "P695"
    rows = prepare_review_table(outcome.items)
    assert rows[0]["Output filename preview"] == "redacted-transcript-2026-01-07-P695.docx"


def test_document_folder_warnings_and_item_errors(tmp_path: Path) -> None:
    (tmp_path / "ignore.zip").write_bytes(b"not supported")
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with (tmp_path / "empty.pdf").open("wb") as stream:
        writer.write(stream)

    outcome = parse_document_folder_path(str(tmp_path))

    assert outcome.warnings == ("Ignored 1 unsupported file(s).",)
    assert len(outcome.items) == 1
    assert outcome.items[0].parse_status == "error"
    error_rows = prepare_parse_error_table(outcome.items)
    assert len(error_rows) == 1
    assert "Error" in error_rows[0]


def test_missing_document_folder_returns_parse_error() -> None:
    outcome = parse_document_folder_path("/path/that/does/not/exist")

    assert outcome.items == ()
    assert outcome.errors


def test_uploaded_document_files_parse_sample_docx(tmp_path: Path) -> None:
    path = write_sample_docx(tmp_path / "P695 Interview 2026_01_07.docx")
    data = path.read_bytes()

    outcome = parse_uploaded_document_files(
        (
            FakeUploadedFile(
                name="folder/P695 Interview 2026_01_07.docx",
                data=data,
            ),
        )
    )

    assert outcome.errors == ()
    assert outcome.warnings == ()
    assert len(outcome.items) == 1
    assert outcome.items[0].chat_date == "2026-01-07"
    assert outcome.items[0].user_identifier == "P695"


def test_uploaded_document_files_ignore_unsupported_files() -> None:
    outcome = parse_uploaded_document_files(
        (FakeUploadedFile(name="../ignored.zip", data=b"not supported"),)
    )

    assert outcome.items == ()
    assert outcome.warnings == (
        "Ignored 1 unsupported uploaded file(s).",
        "No supported documents found in the upload.",
    )


@dataclass(frozen=True)
class FakeUploadedFile:
    name: str
    data: bytes

    def getvalue(self) -> bytes:
        return self.data
