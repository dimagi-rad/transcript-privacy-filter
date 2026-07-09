from __future__ import annotations

from dataclasses import dataclass
import zipfile
from io import BytesIO

from docx import Document

from opf_app.models import ParsedItem, RedactionResult
from opf_app.redaction import SpanForReplacement, parse_preserved_values
from opf_app.ui import (
    category_state_keys,
    run_redaction_for_ui,
    run_v2_redaction_for_ui,
    selected_categories_from_state,
    set_category_selection_state,
)
from opf_app.v2_redaction import (
    V2ChunkMetadata,
    V2RedactionErrorCategory,
    V2RedactionMetadata,
    V2RedactionServiceResult,
    V2UsageTotals,
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
        spans: list[SpanForReplacement] = []
        if "Alice" in text:
            start = text.index("Alice")
            spans.append(
                SpanForReplacement(
                    "private_person",
                    start,
                    start + len("Alice"),
                    "<PRIVATE_PERSON>",
                )
            )
        if "Suzy" in text:
            start = text.index("Suzy")
            spans.append(
                SpanForReplacement(
                    "private_person",
                    start,
                    start + len("Suzy"),
                    "<PRIVATE_PERSON>",
                )
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


def test_preserved_values_parser_for_ui_text_input() -> None:
    assert parse_preserved_values("Suzy, https://example.test, suzy") == (
        "Suzy",
        "https://example.test",
    )


def test_run_redaction_for_ui_with_fake_redactor_produces_zip_and_rows() -> None:
    outcome = run_redaction_for_ui(
        (
            _item("one", "Alice"),
            _item("two", "fail"),
        ),
        selected_labels=("private_person",),
        preserved_values=(),
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


def test_run_redaction_for_ui_preserves_values_in_generated_output() -> None:
    outcome = run_redaction_for_ui(
        (_item("chatbot", "Alice asked Suzy."),),
        selected_labels=("private_person",),
        preserved_values=("suzy",),
        concurrency=1,
        redactor=FakeRedactor(),
    )

    assert outcome.result_rows[0]["Selected redactions"] == 1
    with zipfile.ZipFile(BytesIO(outcome.zip_bytes)) as archive:
        docx_bytes = archive.read("redacted-transcript-2026-05-11-chatbot.docx")
    document = Document(BytesIO(docx_bytes))
    paragraphs = [paragraph.text for paragraph in document.paragraphs]
    assert "<PRIVATE_PERSON> asked Suzy." in paragraphs


def test_run_v2_redaction_for_ui_uses_config_and_privacy_safe_summary() -> None:
    service = FakeV2Service()

    outcome = run_v2_redaction_for_ui(
        (_item("chatbot", "Alice asked Suzy."),),
        api_key="sk-test-secret",
        model_id="gpt-test-redactor",
        sentence_chunk_size=2,
        api_concurrency=1,
        preserved_values=("Suzy",),
        redaction_service=service,
    )

    assert service.calls == [
        {
            "item_name": "chatbot",
            "model_id": "gpt-test-redactor",
            "sentence_chunk_size": 2,
            "max_attempts": 3,
            "preserved_values": ("Suzy",),
        }
    ]
    assert outcome.summary["model_id"] == "gpt-test-redactor"
    assert outcome.summary["sentence_chunk_size"] == 2
    assert outcome.summary["api_concurrency"] == 1
    assert outcome.summary["total_sentence_count"] == 2
    assert outcome.summary["retry_count"] == 1
    assert outcome.summary["total_tokens"] == 15
    assert outcome.result_rows == (
        {
            "Item name": "chatbot",
            "Status": "complete",
            "Sentences": 2,
            "Chunks": 1,
            "Retries": 1,
            "Output filename": "redacted-transcript-2026-05-11-chatbot.docx",
        },
    )

    summary_blob = (
        f"{outcome.summary} {outcome.result_rows} {outcome.failed_rows} "
        f"{outcome.progress_messages}"
    )
    assert "Alice" not in summary_blob
    assert "Suzy" not in summary_blob
    assert "sk-test-secret" not in summary_blob
    assert "<PRIVATE_PERSON> asked Suzy." not in summary_blob

    with zipfile.ZipFile(BytesIO(outcome.zip_bytes)) as archive:
        assert archive.namelist() == [
            "redacted-transcript-2026-05-11-chatbot.docx"
        ]


def test_run_v2_redaction_for_ui_reports_api_failures_without_source_text() -> None:
    service = FakeV2Service(success=False, error_categories=("authentication",))

    outcome = run_v2_redaction_for_ui(
        (_item("chatbot", "Alice asked Suzy."),),
        api_key="sk-test-secret",
        model_id="gpt-test-redactor",
        sentence_chunk_size=3,
        api_concurrency=1,
        preserved_values=("Suzy",),
        redaction_service=service,
    )

    assert outcome.complete_count == 0
    assert outcome.failed_count == 1
    assert outcome.failed_rows == (
        {
            "Item name": "chatbot",
            "Output filename": "redacted-transcript-2026-05-11-chatbot.docx",
            "Error category": "authentication",
            "Error": "v2_redaction_failed:authentication",
        },
    )
    failure_blob = f"{outcome.summary} {outcome.failed_rows} {outcome.progress_messages}"
    assert "Alice" not in failure_blob
    assert "Suzy" not in failure_blob
    assert "sk-test-secret" not in failure_blob

    with zipfile.ZipFile(BytesIO(outcome.zip_bytes)) as archive:
        assert archive.namelist() == []


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


@dataclass
class FakeV2Service:
    success: bool = True
    error_categories: tuple[V2RedactionErrorCategory, ...] = ()

    def __post_init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def redact_item(
        self,
        item: ParsedItem,
        *,
        model_id: str,
        sentence_chunk_size: int,
        max_attempts: int,
        preserved_values: object,
        backoff_seconds: float = 0.0,
    ) -> V2RedactionServiceResult:
        self.calls.append(
            {
                "item_name": item.item_name,
                "model_id": model_id,
                "sentence_chunk_size": sentence_chunk_size,
                "max_attempts": max_attempts,
                "preserved_values": tuple(preserved_values),
            }
        )
        categories = self.error_categories
        return V2RedactionServiceResult(
            redaction_result=RedactionResult(
                item=item,
                output_filename=item.output_filename,
                redacted_text=(
                    "<PRIVATE_PERSON> asked Suzy." if self.success else item.body_text
                ),
                success=self.success,
                errors=()
                if self.success
                else tuple(f"v2_redaction_failed:{category}" for category in categories),
            ),
            metadata=V2RedactionMetadata(
                model_id=model_id,
                sentence_chunk_size=sentence_chunk_size,
                max_attempts=max_attempts,
                total_sentence_count=2,
                chunk_count=1,
                successful_chunk_count=1 if self.success else 0,
                failed_chunk_count=0 if self.success else 1,
                retry_count=1,
                error_categories=categories,
                usage=V2UsageTotals(
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                ),
                chunks=(
                    V2ChunkMetadata(
                        chunk_index=0,
                        sentence_count=2,
                        attempts=2,
                        success=self.success,
                        error_categories=categories,
                        usage=V2UsageTotals(
                            input_tokens=10,
                            output_tokens=5,
                            total_tokens=15,
                        ),
                    ),
                ),
            ),
        )
