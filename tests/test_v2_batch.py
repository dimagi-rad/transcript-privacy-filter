from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
import zipfile

from opf_app.models import ParsedItem, RedactionResult
from opf_app.v2_batch import (
    clamp_v2_api_concurrency,
    clamp_v2_retry_limit,
    clamp_v2_sentence_chunk_size,
    run_v2_redaction_batch,
)
from opf_app.v2_redaction import (
    V2ChunkMetadata,
    V2RedactionErrorCategory,
    V2RedactionMetadata,
    V2RedactionServiceResult,
    V2UsageTotals,
)


def test_v2_api_concurrency_never_exceeds_selected_limit(tmp_path: Path) -> None:
    service = TrackingV2Service(delay=0.05)

    result = run_v2_redaction_batch(
        _items(6),
        model_id="gpt-test-redactor",
        redaction_service=service,
        output_dir=tmp_path,
        api_concurrency=3,
    )

    assert service.max_active <= 3
    assert service.max_active > 1
    assert result.complete_count == 6
    assert result.summary.api_concurrency == 3


def test_v2_concurrency_one_processes_items_sequentially(tmp_path: Path) -> None:
    service = TrackingV2Service()
    items = _items(3)

    run_v2_redaction_batch(
        items,
        model_id="gpt-test-redactor",
        redaction_service=service,
        output_dir=tmp_path,
        api_concurrency=1,
    )

    assert service.calls == [item.item_name for item in items]


def test_v2_results_and_outputs_keep_input_order(tmp_path: Path) -> None:
    service = TrackingV2Service(
        delays_by_item={
            "item-0": 0.06,
            "item-1": 0.01,
            "item-2": 0.03,
        }
    )
    items = _items(3)

    result = run_v2_redaction_batch(
        items,
        model_id="gpt-test-redactor",
        redaction_service=service,
        output_dir=tmp_path,
        api_concurrency=3,
    )

    assert [item.item.item_name for item in result.items] == [
        "item-0",
        "item-1",
        "item-2",
    ]
    assert [output.filename for output in result.output_package.outputs] == [
        item.output_filename for item in items
    ]


def test_v2_failure_isolation_keeps_successful_outputs(tmp_path: Path) -> None:
    service = TrackingV2Service(fail_items={"item-1"})

    result = run_v2_redaction_batch(
        _items(3),
        model_id="gpt-test-redactor",
        redaction_service=service,
        output_dir=tmp_path,
        api_concurrency=2,
    )

    assert result.complete_count == 2
    assert result.failed_count == 1
    assert [item.item.item_name for item in result.failed_items] == ["item-1"]
    assert result.zip_path.is_file()
    with zipfile.ZipFile(result.zip_path) as archive:
        assert len(archive.namelist()) == 2


def test_v2_retry_and_usage_are_aggregated(tmp_path: Path) -> None:
    service = TrackingV2Service(
        metadata_by_item={
            "item-0": _metadata(
                total_sentence_count=2,
                successful_chunk_count=2,
                retry_count=1,
                usage=V2UsageTotals(
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                    cached_input_tokens=2,
                    reasoning_output_tokens=1,
                ),
                chunks=(
                    V2ChunkMetadata(
                        chunk_index=0,
                        sentence_count=1,
                        attempts=2,
                        success=True,
                        usage=V2UsageTotals(
                            input_tokens=10,
                            output_tokens=5,
                            total_tokens=15,
                            cached_input_tokens=2,
                            reasoning_output_tokens=1,
                        ),
                    ),
                    V2ChunkMetadata(
                        chunk_index=1,
                        sentence_count=1,
                        attempts=1,
                        success=True,
                    ),
                ),
            ),
            "item-1": _metadata(
                total_sentence_count=1,
                successful_chunk_count=1,
                retry_count=0,
                usage=V2UsageTotals(input_tokens=4, output_tokens=3, total_tokens=7),
            ),
        }
    )

    result = run_v2_redaction_batch(
        _items(2),
        model_id="gpt-test-redactor",
        redaction_service=service,
        output_dir=tmp_path,
        api_concurrency=2,
    )

    assert result.summary.total_sentence_count == 3
    assert result.summary.successful_chunk_count == 3
    assert result.summary.retry_count == 1
    assert result.summary.usage.summary() == {
        "input_tokens": 14,
        "output_tokens": 8,
        "total_tokens": 22,
        "cached_input_tokens": 2,
        "reasoning_output_tokens": 1,
    }


def test_v2_summary_audit_fields_are_privacy_safe(tmp_path: Path) -> None:
    service = TrackingV2Service(
        metadata_by_item={
            "item-0": _metadata(
                total_sentence_count=2,
                successful_chunk_count=1,
                retry_count=1,
                usage=V2UsageTotals(input_tokens=10, output_tokens=5, total_tokens=15),
            )
        },
        redacted_text="<PRIVATE_PERSON> shared <SECRET>.",
    )

    result = run_v2_redaction_batch(
        (
            _item(
                0,
                body_text="[00:00] S1: Alice shared a private value.",
            ),
        ),
        model_id="gpt-test-redactor",
        redaction_service=service,
        output_dir=tmp_path,
        sentence_chunk_size=2,
        api_concurrency=1,
    )

    summary = result.summary.privacy_safe_dict()

    assert summary["model_id"] == "gpt-test-redactor"
    assert summary["sentence_chunk_size"] == 2
    assert summary["api_concurrency"] == 1
    assert summary["total_sentence_count"] == 2
    assert summary["successful_chunk_count"] == 1
    assert summary["retry_count"] == 1
    assert summary["usage"] == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "cached_input_tokens": 0,
        "reasoning_output_tokens": 0,
    }
    summary_blob = str(summary)
    assert "Alice" not in summary_blob
    assert "private value" not in summary_blob
    assert "<PRIVATE_PERSON>" not in summary_blob
    assert "<SECRET>" not in summary_blob


def test_v2_progress_events_are_privacy_safe(tmp_path: Path) -> None:
    events = []
    sensitive_body = "[00:00] S1: Alice shared a private value."
    service = TrackingV2Service(redacted_text="<PRIVATE_PERSON> shared <SECRET>.")

    result = run_v2_redaction_batch(
        (_item(0, body_text=sensitive_body),),
        model_id="gpt-test-redactor",
        redaction_service=service,
        output_dir=tmp_path,
        progress_callback=events.append,
    )

    assert events == list(result.progress_events)
    assert {event.event_type for event in events} == {"plan", "item", "chunk"}
    progress_text = " ".join(event.message for event in events)
    assert "Alice" not in progress_text
    assert "private value" not in progress_text
    assert "<PRIVATE_PERSON>" not in progress_text
    assert "<SECRET>" not in progress_text
    assert any(
        event.display_filename == "redacted-transcript-2026-05-11-user-0.docx"
        for event in events
    )


def test_v2_progress_percentages_are_monotonic_with_out_of_order_completions(
    tmp_path: Path,
) -> None:
    service = TrackingV2Service(
        delays_by_item={
            "item-0": 0.06,
            "item-1": 0.01,
            "item-2": 0.03,
        }
    )
    events = []

    result = run_v2_redaction_batch(
        _items(3),
        model_id="gpt-test-redactor",
        redaction_service=service,
        output_dir=tmp_path,
        api_concurrency=3,
        progress_callback=events.append,
    )

    percentages = [
        event.snapshot.percentage
        for event in events
        if event.snapshot is not None
    ]
    assert percentages == sorted(percentages)
    assert percentages[0] == 0
    assert percentages[-1] == 100
    assert result.final_progress.is_terminal is True


def test_v2_batch_clamps_user_tunable_limits(tmp_path: Path) -> None:
    service = TrackingV2Service()

    result = run_v2_redaction_batch(
        _items(1),
        model_id="gpt-test-redactor",
        redaction_service=service,
        output_dir=tmp_path,
        sentence_chunk_size=99,
        api_concurrency=99,
        retry_limit=0,
    )

    assert result.summary.sentence_chunk_size == 5
    assert result.summary.api_concurrency == 8
    assert result.summary.retry_limit == 1
    assert service.received_chunk_sizes == [5]
    assert service.received_retry_limits == [1]


def test_v2_clamp_helpers_use_config_ranges() -> None:
    assert clamp_v2_api_concurrency(None) == 4
    assert clamp_v2_api_concurrency(0) == 1
    assert clamp_v2_api_concurrency(99) == 8
    assert clamp_v2_sentence_chunk_size(None) == 3
    assert clamp_v2_sentence_chunk_size(0) == 1
    assert clamp_v2_sentence_chunk_size(99) == 5
    assert clamp_v2_retry_limit(None) == 3
    assert clamp_v2_retry_limit(0) == 1
    assert clamp_v2_retry_limit(4) == 4


def test_v2_unexpected_service_exception_becomes_failed_item(tmp_path: Path) -> None:
    service = TrackingV2Service(raise_items={"item-0"})

    result = run_v2_redaction_batch(
        _items(2),
        model_id="gpt-test-redactor",
        redaction_service=service,
        output_dir=tmp_path,
        api_concurrency=2,
    )

    assert result.complete_count == 1
    assert result.failed_count == 1
    assert result.summary.error_categories == ("api_error",)
    assert "RuntimeError" in result.failed_items[0].redaction_result.errors[0]


@dataclass
class TrackingV2Service:
    delay: float = 0.0
    delays_by_item: dict[str, float] | None = None
    fail_items: set[str] | None = None
    raise_items: set[str] | None = None
    metadata_by_item: dict[str, V2RedactionMetadata] | None = None
    redacted_text: str | None = None

    def __post_init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.calls: list[str] = []
        self.received_chunk_sizes: list[int] = []
        self.received_retry_limits: list[int] = []
        self.lock = threading.Lock()

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
        with self.lock:
            self.calls.append(item.item_name)
            self.received_chunk_sizes.append(sentence_chunk_size)
            self.received_retry_limits.append(max_attempts)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            delay = self.delay
            if self.delays_by_item is not None:
                delay = self.delays_by_item.get(item.item_name, delay)
            if delay:
                time.sleep(delay)
            if self.raise_items and item.item_name in self.raise_items:
                raise RuntimeError("planned v2 service failure")
            if self.fail_items and item.item_name in self.fail_items:
                return _service_result(
                    item,
                    success=False,
                    model_id=model_id,
                    sentence_chunk_size=sentence_chunk_size,
                    retry_limit=max_attempts,
                    metadata=_metadata(
                        failed_chunk_count=1,
                        retry_count=1,
                        error_categories=("rate_limit",),
                        chunks=(
                            V2ChunkMetadata(
                                chunk_index=0,
                                sentence_count=1,
                                attempts=2,
                                success=False,
                                error_categories=("rate_limit",),
                            ),
                        ),
                    ),
                )
            return _service_result(
                item,
                success=True,
                model_id=model_id,
                sentence_chunk_size=sentence_chunk_size,
                retry_limit=max_attempts,
                metadata=(
                    self.metadata_by_item or {}
                ).get(
                    item.item_name,
                    _metadata(),
                ),
                redacted_text=self.redacted_text,
            )
        finally:
            with self.lock:
                self.active -= 1


def _service_result(
    item: ParsedItem,
    *,
    success: bool,
    model_id: str,
    sentence_chunk_size: int,
    retry_limit: int,
    metadata: V2RedactionMetadata | None = None,
    redacted_text: str | None = None,
) -> V2RedactionServiceResult:
    return V2RedactionServiceResult(
        redaction_result=RedactionResult(
            item=item,
            output_filename=item.output_filename,
            redacted_text=redacted_text or f"redacted {item.item_name}",
            success=success,
            errors=() if success else ("v2_redaction_failed:rate_limit",),
        ),
        metadata=metadata
        or _metadata(
            model_id=model_id,
            sentence_chunk_size=sentence_chunk_size,
            retry_limit=retry_limit,
            failed_chunk_count=0 if success else 1,
            error_categories=() if success else ("rate_limit",),
        ),
    )


def _metadata(
    *,
    model_id: str = "gpt-test-redactor",
    sentence_chunk_size: int = 3,
    retry_limit: int = 3,
    total_sentence_count: int = 1,
    successful_chunk_count: int = 1,
    failed_chunk_count: int = 0,
    retry_count: int = 0,
    usage: V2UsageTotals | None = None,
    error_categories: tuple[V2RedactionErrorCategory, ...] = (),
    chunks: tuple[V2ChunkMetadata, ...] | None = None,
) -> V2RedactionMetadata:
    if chunks is None:
        chunks = (
            V2ChunkMetadata(
                chunk_index=0,
                sentence_count=1,
                attempts=retry_count + 1,
                success=failed_chunk_count == 0,
                error_categories=error_categories,
                usage=usage or V2UsageTotals(),
            ),
        )
    return V2RedactionMetadata(
        model_id=model_id,
        sentence_chunk_size=sentence_chunk_size,
        max_attempts=retry_limit,
        total_sentence_count=total_sentence_count,
        chunk_count=len(chunks),
        successful_chunk_count=successful_chunk_count,
        failed_chunk_count=failed_chunk_count,
        retry_count=retry_count,
        error_categories=error_categories,
        usage=usage or V2UsageTotals(),
        chunks=chunks,
    )


def _items(count: int) -> tuple[ParsedItem, ...]:
    return tuple(_item(index) for index in range(count))


def _item(index: int, *, body_text: str | None = None) -> ParsedItem:
    return ParsedItem(
        item_name=f"item-{index}",
        source_name=f"source-{index}.txt",
        source_type="document",
        chat_date="2026-05-11",
        user_identifier=f"user-{index}",
        body_text=body_text or f"body-{index}",
        output_filename=f"redacted-transcript-2026-05-11-user-{index}.docx",
    )
