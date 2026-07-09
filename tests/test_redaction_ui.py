from __future__ import annotations

from dataclasses import dataclass
import inspect
import threading
import zipfile
from io import BytesIO

from opf_app.models import ParsedItem, RedactionResult
from opf_app.ui import (
    MAX_VISIBLE_PROGRESS_MESSAGES,
    _render_redaction_results,
    _render_v2_live_progress,
    prepare_progress_history_display,
    run_v2_redaction_for_ui,
)
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


def test_live_ui_progress_is_rendered_on_caller_thread_before_service_starts() -> None:
    caller_thread_id = threading.get_ident()
    plan_rendered = threading.Event()
    progress_bar = FakeProgressBar()
    status = FakeStatus()
    service = PlanAwareFakeV2Service(plan_rendered=plan_rendered)

    outcome = run_v2_redaction_for_ui(
        (_item("chatbot", "Alice asked Suzy."),),
        model_id="gpt-test-redactor",
        sentence_chunk_size=1,
        api_concurrency=1,
        redaction_service=service,
        progress_callback=lambda event: _render_and_ack_plan(
            progress_bar,
            status,
            event,
            plan_rendered,
        ),
    )

    assert outcome.complete_count == 1
    assert progress_bar.updates[0][0] == 0
    assert progress_bar.updates[-1][0] == 100
    assert "Estimating remaining time" in status.messages[0][0]
    assert "about 0s" in status.messages[-1][0]
    assert {
        thread_id for _value, _text, thread_id in progress_bar.updates
    } == {caller_thread_id}
    assert {thread_id for _message, thread_id in status.messages} == {
        caller_thread_id
    }


def test_large_progress_history_is_bounded_to_recent_messages() -> None:
    messages = tuple(f"privacy-safe progress {index}" for index in range(9_493))

    visible_messages, omitted_count = prepare_progress_history_display(messages)

    assert len(visible_messages) == MAX_VISIBLE_PROGRESS_MESSAGES
    assert visible_messages == messages[-MAX_VISIBLE_PROGRESS_MESSAGES:]
    assert omitted_count == 9_293
    assert len(messages) == 9_493


def test_download_control_renders_before_bounded_progress_history() -> None:
    source = inspect.getsource(_render_redaction_results)

    assert source.index("st.download_button(") < source.index(
        'with st.expander("Progress messages")'
    )
    assert "for message in progress_messages" not in source
    assert 'st.code("\\n".join(visible_messages)' in source


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


@dataclass
class PlanAwareFakeV2Service(FakeV2Service):
    plan_rendered: threading.Event | None = None

    def redact_item(self, item: ParsedItem, **kwargs: object) -> V2RedactionServiceResult:
        assert self.plan_rendered is not None
        assert self.plan_rendered.is_set()
        kwargs.pop("progress_callback", None)
        return super().redact_item(item, **kwargs)


@dataclass
class FakeProgressBar:
    def __post_init__(self) -> None:
        self.updates: list[tuple[int, str, int]] = []

    def progress(self, value: int, *, text: str) -> None:
        self.updates.append((value, text, threading.get_ident()))


@dataclass
class FakeStatus:
    def __post_init__(self) -> None:
        self.messages: list[tuple[str, int]] = []

    def info(self, message: str) -> None:
        self.messages.append((message, threading.get_ident()))


def _render_and_ack_plan(
    progress_bar: FakeProgressBar,
    status: FakeStatus,
    event: object,
    plan_rendered: threading.Event,
) -> None:
    _render_v2_live_progress(progress_bar, status, event)
    if event.event_type == "plan":
        assert event.snapshot is not None
        assert event.snapshot.percentage == 0
        assert event.snapshot.total_chunk_count == 1
        plan_rendered.set()
