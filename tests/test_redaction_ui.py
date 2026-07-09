from __future__ import annotations

from dataclasses import dataclass
import zipfile
from io import BytesIO

from opf_app.models import ParsedItem, RedactionResult
from opf_app.ui import run_v2_redaction_for_ui
from opf_app.v2_redaction import (
    V2ChunkMetadata,
    V2RedactionErrorCategory,
    V2RedactionMetadata,
    V2RedactionServiceResult,
    V2UsageTotals,
)


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
