from __future__ import annotations

from dataclasses import dataclass
import zipfile
from io import BytesIO

from opf_app.models import ParsedItem
from opf_app.redaction import SpanForReplacement
from opf_app.ui import (
    category_state_keys,
    run_redaction_for_ui,
    selected_categories_from_state,
    set_category_selection_state,
)


@dataclass(frozen=True)
class FakeOpfResult:
    text: str
    detected_spans: tuple[SpanForReplacement, ...]
    warning: str | None = None


class FakeRedactor:
    def redact(self, text: str) -> FakeOpfResult:
        if text == "fail":
            raise RuntimeError("planned failure")
        spans = (
            (SpanForReplacement("private_person", 0, 5, "<PRIVATE_PERSON>"),)
            if text.startswith("Alice")
            else ()
        )
        return FakeOpfResult(text=text, detected_spans=spans)


def test_category_control_state_helpers() -> None:
    labels = ("private_person", "private_email")
    state: dict[str, object] = {}

    assert selected_categories_from_state(labels, state) == labels

    set_category_selection_state(labels, state, selected=False)
    assert selected_categories_from_state(labels, state) == ()

    state[category_state_keys(labels)["private_person"]] = True
    assert selected_categories_from_state(labels, state) == ("private_person",)


def test_run_redaction_for_ui_with_fake_redactor_produces_zip_and_rows() -> None:
    outcome = run_redaction_for_ui(
        (
            _item("one", "Alice"),
            _item("two", "fail"),
        ),
        selected_labels=("private_person",),
        concurrency=2,
        redactor=FakeRedactor(),
    )

    assert outcome.total_count == 2
    assert outcome.complete_count == 1
    assert outcome.failed_count == 1
    assert outcome.failed_rows
    assert any(
        message.startswith("Now redacting document")
        for message in outcome.progress_messages
    )
    with zipfile.ZipFile(BytesIO(outcome.zip_bytes)) as archive:
        assert archive.namelist() == [
            "redacted-transcript-2026-05-11-one.docx"
        ]


def _item(identifier: str, body_text: str) -> ParsedItem:
    return ParsedItem(
        item_name=identifier,
        source_name=f"{identifier}.txt",
        source_type="document",
        chat_date="2026-05-11",
        user_identifier=identifier,
        body_text=body_text,
        output_filename=f"redacted-transcript-2026-05-11-{identifier}.docx",
    )
