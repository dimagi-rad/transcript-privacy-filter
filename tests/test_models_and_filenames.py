from __future__ import annotations

from datetime import date

from opf_app.filenames import (
    OutputFilenameCandidate,
    build_output_filename,
    build_unique_output_filenames,
    document_output_filename,
    infer_chat_date_from_stem,
    infer_user_identifier_from_stem,
    sanitize_filename_component,
    stable_suffix,
)
from opf_app.models import ParsedItem, RedactionJob, RedactionResult


SAMPLE_DOCUMENT_STEM = "P695 Interview 2026_01_07"


def test_sample_document_stem_infers_date_and_identifier() -> None:
    assert infer_chat_date_from_stem(SAMPLE_DOCUMENT_STEM) == "2026-01-07"
    assert infer_user_identifier_from_stem(SAMPLE_DOCUMENT_STEM) == "P695"


def test_date_inference_supports_hyphen_underscore_and_fallback() -> None:
    fallback = date(2026, 5, 19)

    assert infer_chat_date_from_stem("Interview 2026_01_07") == "2026-01-07"
    assert infer_chat_date_from_stem("Interview 2026-01-07") == "2026-01-07"
    assert infer_chat_date_from_stem("Interview", processing_date=fallback) == "2026-05-19"


def test_identifier_inference_falls_back_to_sanitized_stem() -> None:
    assert infer_user_identifier_from_stem("P695 Interview") == "P695"
    assert infer_user_identifier_from_stem("Session Notes Draft") == "Session-Notes-Draft"


def test_csv_style_metadata_produces_expected_output_filename() -> None:
    participant_public_id = "eb8649a3-aa7a-4e92-9875-6b006fc2d2fb"

    filename = build_output_filename(
        "2026-05-11 09:33:29.622201+00:00",
        participant_public_id,
    )

    assert (
        filename
        == "redacted-transcript-2026-05-11-eb8649a3-aa7a-4e92-9875-6b006fc2d2fb.docx"
    )


def test_duplicate_filename_suffixing_is_deterministic() -> None:
    candidates = [
        OutputFilenameCandidate("2026-05-11", "same-user", "session-a"),
        OutputFilenameCandidate("2026-05-11", "same-user", "session-b"),
        OutputFilenameCandidate("2026-05-12", "same-user", "session-c"),
    ]

    filenames = build_unique_output_filenames(candidates)

    assert filenames == (
        f"redacted-transcript-2026-05-11-same-user-{stable_suffix('session-a')}.docx",
        f"redacted-transcript-2026-05-11-same-user-{stable_suffix('session-b')}.docx",
        "redacted-transcript-2026-05-12-same-user.docx",
    )


def test_unsafe_characters_whitespace_and_docx_extension_are_normalized() -> None:
    assert sanitize_filename_component("  Jane / Doe: * test?  ") == "Jane-Doe-test"
    assert (
        document_output_filename(
            "Folder Report 2026-01-07.txt",
            processing_date=date(2026, 5, 19),
        )
        == "redacted-transcript-2026-01-07-Folder-Report-2026-01-07.docx"
    )


def test_models_hide_sensitive_text_from_repr_and_expose_review_row() -> None:
    item = ParsedItem(
        item_name="Session 1",
        source_name="example-input.csv",
        source_type="ocs_csv",
        chat_date="2026-05-11",
        user_identifier="public-id",
        body_text="sensitive transcript body",
        output_filename="redacted-transcript-2026-05-11-public-id.docx",
        session_id="session-id",
        message_count=2,
        warnings=["skipped one row"],
    )
    job = RedactionJob(item=item, selected_categories=["private_person"])
    result = RedactionResult(
        item=item,
        output_filename=item.output_filename,
        redacted_text="redacted sensitive body",
        detected_categories=["private_person"],
        selected_categories=job.selected_categories,
    )

    assert "sensitive transcript body" not in repr(item)
    assert "redacted sensitive body" not in repr(result)
    assert item.review_row()["count"] == 2
    assert job.selected_categories == ("private_person",)
    assert result.detected_categories == ("private_person",)
