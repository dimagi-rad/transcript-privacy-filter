from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import Literal, Protocol

from .config import (
    API_CONCURRENCY_RANGE,
    RETRY_ATTEMPTS_DEFAULT,
    SENTENCE_CHUNK_RANGE,
)
from .models import ParsedItem, RedactionResult
from .outputs import GeneratedOutput, OutputPackage, package_successful_outputs
from .v2_redaction import (
    V2RedactionErrorCategory,
    V2RedactionMetadata,
    V2RedactionService,
    V2RedactionServiceResult,
    V2UsageTotals,
)


V2BatchItemStatus = Literal["pending", "running", "complete", "failed"]
V2ProgressEventType = Literal["item", "chunk"]
V2BatchProgressCallback = Callable[["V2BatchProgressEvent"], None]


class V2RedactionServiceLike(Protocol):
    def redact_item(
        self,
        item: ParsedItem,
        *,
        model_id: str,
        sentence_chunk_size: int,
        max_attempts: int,
        preserved_values: Iterable[str],
        backoff_seconds: float = 0.0,
    ) -> V2RedactionServiceResult:
        """Redact one parsed item through the v2 service contract."""


@dataclass(frozen=True)
class V2BatchProgressEvent:
    """Privacy-safe v2 batch progress event."""

    event_type: V2ProgressEventType
    status: V2BatchItemStatus
    current_index: int
    total_count: int
    display_filename: str
    message: str
    chunk_index: int | None = None
    chunk_count: int | None = None
    error_categories: tuple[V2RedactionErrorCategory, ...] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "error_categories", tuple(self.error_categories)
        )


@dataclass(frozen=True)
class V2BatchItemResult:
    item: ParsedItem
    status: V2BatchItemStatus
    redaction_result: RedactionResult
    metadata: V2RedactionMetadata
    output: GeneratedOutput


@dataclass(frozen=True)
class V2BatchSummary:
    model_id: str
    sentence_chunk_size: int
    api_concurrency: int
    retry_limit: int
    total_item_count: int
    complete_item_count: int
    failed_item_count: int
    total_sentence_count: int
    successful_chunk_count: int
    failed_chunk_count: int
    retry_count: int
    usage: V2UsageTotals = field(default_factory=V2UsageTotals)
    error_categories: tuple[V2RedactionErrorCategory, ...] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "error_categories", tuple(self.error_categories)
        )

    def privacy_safe_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "sentence_chunk_size": self.sentence_chunk_size,
            "api_concurrency": self.api_concurrency,
            "retry_limit": self.retry_limit,
            "total_item_count": self.total_item_count,
            "complete_item_count": self.complete_item_count,
            "failed_item_count": self.failed_item_count,
            "total_sentence_count": self.total_sentence_count,
            "successful_chunk_count": self.successful_chunk_count,
            "failed_chunk_count": self.failed_chunk_count,
            "retry_count": self.retry_count,
            "usage": self.usage.summary(),
            "error_categories": self.error_categories,
        }


@dataclass(frozen=True)
class V2BatchResult:
    items: tuple[V2BatchItemResult, ...]
    output_package: OutputPackage
    progress_events: tuple[V2BatchProgressEvent, ...]
    summary: V2BatchSummary

    @property
    def total_count(self) -> int:
        return len(self.items)

    @property
    def complete_count(self) -> int:
        return sum(1 for item in self.items if item.status == "complete")

    @property
    def failed_count(self) -> int:
        return sum(1 for item in self.items if item.status == "failed")

    @property
    def successful_items(self) -> tuple[V2BatchItemResult, ...]:
        return tuple(item for item in self.items if item.status == "complete")

    @property
    def failed_items(self) -> tuple[V2BatchItemResult, ...]:
        return tuple(item for item in self.items if item.status == "failed")

    @property
    def zip_path(self) -> Path:
        return self.output_package.zip_path


def clamp_v2_api_concurrency(value: int | None) -> int:
    return API_CONCURRENCY_RANGE.clamp(value)


def clamp_v2_sentence_chunk_size(value: int | None) -> int:
    return SENTENCE_CHUNK_RANGE.clamp(value)


def clamp_v2_retry_limit(value: int | None) -> int:
    if value is None:
        return RETRY_ATTEMPTS_DEFAULT
    return max(1, int(value))


def run_v2_redaction_batch(
    items: Iterable[ParsedItem],
    *,
    model_id: str,
    redaction_service: V2RedactionService | V2RedactionServiceLike,
    output_dir: str | Path,
    sentence_chunk_size: int | None = None,
    api_concurrency: int | None = None,
    retry_limit: int | None = None,
    preserved_values: Iterable[str] = (),
    backoff_seconds: float = 0.0,
    progress_callback: V2BatchProgressCallback | None = None,
) -> V2BatchResult:
    """Run v2 redaction with bounded API concurrency and ordered outputs."""
    parsed_items = tuple(items)
    total_count = len(parsed_items)
    chunk_size = clamp_v2_sentence_chunk_size(sentence_chunk_size)
    worker_count = clamp_v2_api_concurrency(api_concurrency)
    max_attempts = clamp_v2_retry_limit(retry_limit)
    values_to_preserve = tuple(preserved_values)
    progress_events: list[V2BatchProgressEvent] = []
    progress_lock = threading.Lock()

    def emit(event: V2BatchProgressEvent) -> None:
        with progress_lock:
            progress_events.append(event)
        if progress_callback is not None:
            progress_callback(event)

    def process_item(
        index: int,
        item: ParsedItem,
    ) -> tuple[int, V2RedactionServiceResult]:
        current_index = index + 1
        emit(
            V2BatchProgressEvent(
                event_type="item",
                status="running",
                current_index=current_index,
                total_count=total_count,
                display_filename=item.output_filename,
                message=(
                    f"Started v2 redaction {current_index}/{total_count}: "
                    f"{item.output_filename}"
                ),
            )
        )

        try:
            service_result = redaction_service.redact_item(
                item,
                model_id=model_id,
                sentence_chunk_size=chunk_size,
                max_attempts=max_attempts,
                preserved_values=values_to_preserve,
                backoff_seconds=backoff_seconds,
            )
        except Exception as exc:  # noqa: BLE001 - isolate one parsed item
            service_result = _unexpected_failure_result(
                item,
                model_id=model_id,
                sentence_chunk_size=chunk_size,
                retry_limit=max_attempts,
                error_type=type(exc).__name__,
            )

        _emit_chunk_events(
            emit,
            service_result,
            current_index=current_index,
            total_count=total_count,
            display_filename=item.output_filename,
        )

        status: V2BatchItemStatus = (
            "complete" if service_result.success else "failed"
        )
        emit(
            V2BatchProgressEvent(
                event_type="item",
                status=status,
                current_index=current_index,
                total_count=total_count,
                display_filename=item.output_filename,
                message=(
                    f"{'Completed' if service_result.success else 'Failed'} "
                    f"v2 redaction {current_index}/{total_count}: "
                    f"{item.output_filename}"
                ),
                error_categories=service_result.metadata.error_categories,
            )
        )
        return index, service_result

    service_results: list[V2RedactionServiceResult | None] = [None] * total_count
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(process_item, index, item)
            for index, item in enumerate(parsed_items)
        ]
        for future in as_completed(futures):
            index, service_result = future.result()
            service_results[index] = service_result

    ordered_results = tuple(
        result for result in service_results if result is not None
    )
    ordered_redaction_results = tuple(
        result.redaction_result for result in ordered_results
    )
    output_package = package_successful_outputs(ordered_redaction_results, output_dir)
    batch_items = tuple(
        V2BatchItemResult(
            item=service_result.redaction_result.item,
            status="complete" if service_result.success else "failed",
            redaction_result=service_result.redaction_result,
            metadata=service_result.metadata,
            output=output,
        )
        for service_result, output in zip(ordered_results, output_package.outputs)
    )

    return V2BatchResult(
        items=batch_items,
        output_package=output_package,
        progress_events=tuple(progress_events),
        summary=_build_summary(
            model_id=model_id,
            sentence_chunk_size=chunk_size,
            api_concurrency=worker_count,
            retry_limit=max_attempts,
            items=batch_items,
        ),
    )


def _emit_chunk_events(
    emit: Callable[[V2BatchProgressEvent], None],
    service_result: V2RedactionServiceResult,
    *,
    current_index: int,
    total_count: int,
    display_filename: str,
) -> None:
    chunk_count = service_result.metadata.chunk_count
    for chunk in service_result.metadata.chunks:
        status: V2BatchItemStatus = "complete" if chunk.success else "failed"
        emit(
            V2BatchProgressEvent(
                event_type="chunk",
                status=status,
                current_index=current_index,
                total_count=total_count,
                display_filename=display_filename,
                chunk_index=chunk.chunk_index + 1,
                chunk_count=chunk_count,
                message=(
                    f"{'Completed' if chunk.success else 'Failed'} chunk "
                    f"{chunk.chunk_index + 1}/{chunk_count} for v2 redaction "
                    f"{current_index}/{total_count}: {display_filename}"
                ),
                error_categories=chunk.error_categories,
            )
        )


def _unexpected_failure_result(
    item: ParsedItem,
    *,
    model_id: str,
    sentence_chunk_size: int,
    retry_limit: int,
    error_type: str,
) -> V2RedactionServiceResult:
    return V2RedactionServiceResult(
        redaction_result=RedactionResult(
            item=item,
            output_filename=item.output_filename,
            redacted_text=item.body_text,
            success=False,
            warnings=item.warnings,
            errors=(f"{error_type}: v2 redaction failed for item.",),
        ),
        metadata=V2RedactionMetadata(
            model_id=model_id,
            sentence_chunk_size=sentence_chunk_size,
            max_attempts=retry_limit,
            total_sentence_count=0,
            chunk_count=0,
            successful_chunk_count=0,
            failed_chunk_count=1,
            retry_count=0,
            error_categories=("api_error",),
        ),
    )


def _build_summary(
    *,
    model_id: str,
    sentence_chunk_size: int,
    api_concurrency: int,
    retry_limit: int,
    items: tuple[V2BatchItemResult, ...],
) -> V2BatchSummary:
    usage = V2UsageTotals(
        input_tokens=sum(item.metadata.usage.input_tokens for item in items),
        output_tokens=sum(item.metadata.usage.output_tokens for item in items),
        total_tokens=sum(item.metadata.usage.total_tokens for item in items),
        cached_input_tokens=sum(
            item.metadata.usage.cached_input_tokens for item in items
        ),
        reasoning_output_tokens=sum(
            item.metadata.usage.reasoning_output_tokens for item in items
        ),
    )
    categories: list[V2RedactionErrorCategory] = []
    for item in items:
        categories.extend(item.metadata.error_categories)

    return V2BatchSummary(
        model_id=model_id,
        sentence_chunk_size=sentence_chunk_size,
        api_concurrency=api_concurrency,
        retry_limit=retry_limit,
        total_item_count=len(items),
        complete_item_count=sum(1 for item in items if item.status == "complete"),
        failed_item_count=sum(1 for item in items if item.status == "failed"),
        total_sentence_count=sum(item.metadata.total_sentence_count for item in items),
        successful_chunk_count=sum(
            item.metadata.successful_chunk_count for item in items
        ),
        failed_chunk_count=sum(item.metadata.failed_chunk_count for item in items),
        retry_count=sum(item.metadata.retry_count for item in items),
        usage=usage,
        error_categories=tuple(dict.fromkeys(categories)),
    )
