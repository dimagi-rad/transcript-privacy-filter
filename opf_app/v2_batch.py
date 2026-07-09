from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import inspect
import math
from pathlib import Path
import threading
import time
from typing import Literal, Protocol

from .config import (
    API_CONCURRENCY_RANGE,
    RETRY_ATTEMPTS_DEFAULT,
    SENTENCE_CHUNK_RANGE,
)
from .models import ParsedItem, RedactionResult
from .outputs import GeneratedOutput, OutputPackage, package_v2_successful_outputs
from .sentences import segment_parsed_item
from .v2_redaction import (
    V2ChunkLifecycleEvent,
    V2ChunkProgressCallback,
    V2RedactionErrorCategory,
    V2RedactionMetadata,
    V2RedactionService,
    V2RedactionServiceResult,
    V2UsageTotals,
)


V2BatchItemStatus = Literal[
    "pending",
    "running",
    "retrying",
    "complete",
    "failed",
    "skipped",
]
V2ProgressEventType = Literal["plan", "item", "chunk"]
V2BatchProgressCallback = Callable[["V2BatchProgressEvent"], None]
V2MonotonicClock = Callable[[], float]


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
        progress_callback: V2ChunkProgressCallback | None = None,
    ) -> V2RedactionServiceResult:
        """Redact one parsed item through the v2 service contract."""


@dataclass(frozen=True)
class V2WorkPlanItem:
    """Privacy-safe local work counts for one parsed item."""

    item_index: int
    sentence_count: int
    planned_chunk_count: int


@dataclass(frozen=True)
class V2WorkPlan:
    """Stable local denominator created before any Responses API call."""

    sentence_chunk_size: int
    total_item_count: int
    total_sentence_count: int
    total_chunk_count: int
    items: tuple[V2WorkPlanItem, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))

    def item_at(self, item_index: int) -> V2WorkPlanItem:
        return self.items[item_index]


@dataclass(frozen=True)
class V2ProgressSnapshot:
    """Thread-safe aggregate counts for one instant in a v2 run."""

    total_item_count: int
    completed_item_count: int
    failed_item_count: int
    total_sentence_count: int
    total_chunk_count: int
    completed_chunk_count: int
    failed_chunk_count: int
    skipped_chunk_count: int
    retry_count: int
    unresolved_chunk_count: int
    percentage: int
    eta_seconds: float | None
    elapsed_seconds: float
    is_terminal: bool

    @property
    def resolved_chunk_count(self) -> int:
        return (
            self.completed_chunk_count
            + self.failed_chunk_count
            + self.skipped_chunk_count
        )


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
    attempt_number: int | None = None
    error_categories: tuple[V2RedactionErrorCategory, ...] = field(
        default_factory=tuple
    )
    snapshot: V2ProgressSnapshot | None = None

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
    planned_chunk_count: int
    successful_chunk_count: int
    failed_chunk_count: int
    skipped_chunk_count: int
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
            "planned_chunk_count": self.planned_chunk_count,
            "successful_chunk_count": self.successful_chunk_count,
            "failed_chunk_count": self.failed_chunk_count,
            "skipped_chunk_count": self.skipped_chunk_count,
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
    work_plan: V2WorkPlan
    final_progress: V2ProgressSnapshot

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


class V2ProgressTracker:
    """Aggregate concurrent chunk events against a stable work plan."""

    def __init__(
        self,
        work_plan: V2WorkPlan,
        *,
        monotonic: V2MonotonicClock | None = None,
    ) -> None:
        self.work_plan = work_plan
        self._monotonic = monotonic or time.monotonic
        self._started_at = self._monotonic()
        self._lock = threading.RLock()
        self._completed_items: set[int] = set()
        self._failed_items: set[int] = set()
        self._completed_chunks: set[tuple[int, int]] = set()
        self._failed_chunks: set[tuple[int, int]] = set()
        self._skipped_chunks: set[tuple[int, int]] = set()
        self._retry_counts: dict[tuple[int, int], int] = {}

    def snapshot(self) -> V2ProgressSnapshot:
        with self._lock:
            return self._snapshot_unlocked()

    def record_retry(
        self,
        item_index: int,
        chunk_index: int,
        retry_count: int,
    ) -> tuple[bool, V2ProgressSnapshot]:
        with self._lock:
            key = (item_index, chunk_index)
            previous = self._retry_counts.get(key, 0)
            self._retry_counts[key] = max(previous, retry_count)
            return self._retry_counts[key] != previous, self._snapshot_unlocked()

    def resolve_chunk(
        self,
        item_index: int,
        chunk_index: int,
        status: Literal["complete", "failed", "skipped"],
    ) -> tuple[bool, V2ProgressSnapshot]:
        with self._lock:
            key = (item_index, chunk_index)
            if self._is_chunk_resolved_unlocked(key):
                return False, self._snapshot_unlocked()
            if status == "complete":
                self._completed_chunks.add(key)
            elif status == "failed":
                self._failed_chunks.add(key)
            else:
                self._skipped_chunks.add(key)
            return True, self._snapshot_unlocked()

    def record_item_terminal(
        self,
        item_index: int,
        *,
        success: bool,
    ) -> V2ProgressSnapshot:
        with self._lock:
            if success:
                self._completed_items.add(item_index)
            else:
                self._failed_items.add(item_index)
            return self._snapshot_unlocked()

    def is_chunk_resolved(self, item_index: int, chunk_index: int) -> bool:
        with self._lock:
            return self._is_chunk_resolved_unlocked((item_index, chunk_index))

    def _is_chunk_resolved_unlocked(self, key: tuple[int, int]) -> bool:
        return (
            key in self._completed_chunks
            or key in self._failed_chunks
            or key in self._skipped_chunks
        )

    def _snapshot_unlocked(self) -> V2ProgressSnapshot:
        completed_item_count = len(self._completed_items)
        failed_item_count = len(self._failed_items)
        completed_chunk_count = len(self._completed_chunks)
        failed_chunk_count = len(self._failed_chunks)
        skipped_chunk_count = len(self._skipped_chunks)
        resolved_chunk_count = (
            completed_chunk_count + failed_chunk_count + skipped_chunk_count
        )
        unresolved_chunk_count = max(
            0,
            self.work_plan.total_chunk_count - resolved_chunk_count,
        )
        elapsed_seconds = max(0.0, self._monotonic() - self._started_at)
        is_terminal = (
            self.work_plan.total_item_count > 0
            and completed_item_count + failed_item_count
            >= self.work_plan.total_item_count
            and unresolved_chunk_count == 0
        )
        return V2ProgressSnapshot(
            total_item_count=self.work_plan.total_item_count,
            completed_item_count=completed_item_count,
            failed_item_count=failed_item_count,
            total_sentence_count=self.work_plan.total_sentence_count,
            total_chunk_count=self.work_plan.total_chunk_count,
            completed_chunk_count=completed_chunk_count,
            failed_chunk_count=failed_chunk_count,
            skipped_chunk_count=skipped_chunk_count,
            retry_count=sum(self._retry_counts.values()),
            unresolved_chunk_count=unresolved_chunk_count,
            percentage=calculate_v2_progress_percentage(
                resolved_chunk_count,
                self.work_plan.total_chunk_count,
                is_terminal=is_terminal,
            ),
            eta_seconds=calculate_v2_eta_seconds(
                elapsed_seconds=elapsed_seconds,
                resolved_chunk_count=resolved_chunk_count,
                unresolved_chunk_count=unresolved_chunk_count,
            ),
            elapsed_seconds=elapsed_seconds,
            is_terminal=is_terminal,
        )


def clamp_v2_api_concurrency(value: int | None) -> int:
    return API_CONCURRENCY_RANGE.clamp(value)


def clamp_v2_sentence_chunk_size(value: int | None) -> int:
    return SENTENCE_CHUNK_RANGE.clamp(value)


def clamp_v2_retry_limit(value: int | None) -> int:
    if value is None:
        return RETRY_ATTEMPTS_DEFAULT
    return max(1, int(value))


def build_v2_work_plan(
    items: Iterable[ParsedItem],
    *,
    sentence_chunk_size: int | None = None,
) -> V2WorkPlan:
    """Count local sentences and API chunks without making an API request."""
    parsed_items = tuple(items)
    chunk_size = clamp_v2_sentence_chunk_size(sentence_chunk_size)
    planned_items: list[V2WorkPlanItem] = []
    for item_index, item in enumerate(parsed_items):
        sentence_count = 0
        if item.parse_status != "error" and not item.errors:
            segmented_item, _next_sentence_number = segment_parsed_item(
                item,
                item_index=item_index,
            )
            sentence_count = len(segmented_item.sentences)
        planned_items.append(
            V2WorkPlanItem(
                item_index=item_index,
                sentence_count=sentence_count,
                planned_chunk_count=math.ceil(sentence_count / chunk_size),
            )
        )
    return V2WorkPlan(
        sentence_chunk_size=chunk_size,
        total_item_count=len(parsed_items),
        total_sentence_count=sum(item.sentence_count for item in planned_items),
        total_chunk_count=sum(
            item.planned_chunk_count for item in planned_items
        ),
        items=tuple(planned_items),
    )


def calculate_v2_progress_percentage(
    resolved_chunk_count: int,
    total_chunk_count: int,
    *,
    is_terminal: bool = False,
) -> int:
    """Return a clamped integer percentage using the stable chunk denominator."""
    total = max(0, int(total_chunk_count))
    resolved = min(total, max(0, int(resolved_chunk_count)))
    if total == 0:
        return 100 if is_terminal else 0
    return min(100, max(0, int((resolved * 100) / total)))


def calculate_v2_eta_seconds(
    *,
    elapsed_seconds: float,
    resolved_chunk_count: int,
    unresolved_chunk_count: int,
) -> float | None:
    """Estimate remaining seconds from observed effective chunk throughput."""
    resolved = max(0, int(resolved_chunk_count))
    unresolved = max(0, int(unresolved_chunk_count))
    if resolved == 0:
        return None
    if unresolved == 0:
        return 0.0
    elapsed = max(0.0, float(elapsed_seconds))
    return max(0.0, (elapsed / resolved) * unresolved)


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
    monotonic: V2MonotonicClock | None = None,
) -> V2BatchResult:
    """Run v2 redaction with live chunk progress and ordered outputs."""
    parsed_items = tuple(items)
    total_count = len(parsed_items)
    chunk_size = clamp_v2_sentence_chunk_size(sentence_chunk_size)
    worker_count = clamp_v2_api_concurrency(api_concurrency)
    max_attempts = clamp_v2_retry_limit(retry_limit)
    values_to_preserve = tuple(preserved_values)
    work_plan = build_v2_work_plan(
        parsed_items,
        sentence_chunk_size=chunk_size,
    )
    tracker = V2ProgressTracker(work_plan, monotonic=monotonic)
    progress_events: list[V2BatchProgressEvent] = []
    progress_lock = threading.RLock()

    def emit_locked(event: V2BatchProgressEvent) -> None:
        progress_events.append(event)
        if progress_callback is not None:
            progress_callback(event)

    with progress_lock:
        initial_snapshot = tracker.snapshot()
        emit_locked(
            V2BatchProgressEvent(
                event_type="plan",
                status="pending",
                current_index=0,
                total_count=total_count,
                display_filename="",
                message=(
                    f"Planned {work_plan.total_item_count} item(s), "
                    f"{work_plan.total_sentence_count} sentence(s), and "
                    f"{work_plan.total_chunk_count} API chunk(s)."
                ),
                snapshot=initial_snapshot,
            )
        )

    def emit_chunk_status(
        item_index: int,
        item: ParsedItem,
        *,
        status: Literal["running", "retrying", "complete", "failed", "skipped"],
        chunk_index: int,
        attempt_number: int | None = None,
        error_categories: tuple[V2RedactionErrorCategory, ...] = (),
    ) -> None:
        planned_chunk_count = work_plan.item_at(item_index).planned_chunk_count
        if chunk_index < 0 or chunk_index >= planned_chunk_count:
            return
        with progress_lock:
            if status == "retrying":
                changed, snapshot = tracker.record_retry(
                    item_index,
                    chunk_index,
                    max(1, (attempt_number or 2) - 1),
                )
                if not changed:
                    return
            elif status in {"complete", "failed", "skipped"}:
                changed, snapshot = tracker.resolve_chunk(
                    item_index,
                    chunk_index,
                    status,
                )
                if not changed:
                    return
            else:
                snapshot = tracker.snapshot()

            current_index = item_index + 1
            action = {
                "running": "Started",
                "retrying": "Retrying",
                "complete": "Completed",
                "failed": "Failed",
                "skipped": "Skipped",
            }[status]
            attempt_text = (
                f" (attempt {attempt_number}/{max_attempts})"
                if status == "retrying" and attempt_number is not None
                else ""
            )
            emit_locked(
                V2BatchProgressEvent(
                    event_type="chunk",
                    status=status,
                    current_index=current_index,
                    total_count=total_count,
                    display_filename=item.output_filename,
                    chunk_index=chunk_index + 1,
                    chunk_count=planned_chunk_count,
                    attempt_number=attempt_number,
                    message=(
                        f"{action} chunk {chunk_index + 1}/{planned_chunk_count}"
                        f"{attempt_text} for v2 redaction "
                        f"{current_index}/{total_count}: {item.output_filename}"
                    ),
                    error_categories=error_categories,
                    snapshot=snapshot,
                )
            )

    def process_item(
        index: int,
        item: ParsedItem,
    ) -> tuple[int, V2RedactionServiceResult]:
        current_index = index + 1
        with progress_lock:
            emit_locked(
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
                    snapshot=tracker.snapshot(),
                )
            )

        def on_chunk_event(event: V2ChunkLifecycleEvent) -> None:
            emit_chunk_status(
                index,
                item,
                status=event.status,
                chunk_index=event.chunk_index,
                attempt_number=event.attempt_number,
                error_categories=event.error_categories,
            )

        service_kwargs: dict[str, object] = {
            "model_id": model_id,
            "sentence_chunk_size": chunk_size,
            "max_attempts": max_attempts,
            "preserved_values": values_to_preserve,
            "backoff_seconds": backoff_seconds,
        }
        if _service_accepts_progress_callback(redaction_service):
            service_kwargs["progress_callback"] = on_chunk_event

        try:
            service_result = redaction_service.redact_item(
                item,
                **service_kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - isolate one parsed item
            service_result = _unexpected_failure_result(
                item,
                model_id=model_id,
                sentence_chunk_size=chunk_size,
                retry_limit=max_attempts,
                error_type=type(exc).__name__,
            )

        for chunk in service_result.metadata.chunks:
            if chunk.retry_count:
                emit_chunk_status(
                    index,
                    item,
                    status="retrying",
                    chunk_index=chunk.chunk_index,
                    attempt_number=chunk.retry_count + 1,
                    error_categories=chunk.error_categories,
                )
            emit_chunk_status(
                index,
                item,
                status="complete" if chunk.success else "failed",
                chunk_index=chunk.chunk_index,
                attempt_number=chunk.attempts,
                error_categories=chunk.error_categories,
            )

        planned_chunk_count = work_plan.item_at(index).planned_chunk_count
        for chunk_index in range(planned_chunk_count):
            if not tracker.is_chunk_resolved(index, chunk_index):
                emit_chunk_status(
                    index,
                    item,
                    status="skipped",
                    chunk_index=chunk_index,
                    error_categories=service_result.metadata.error_categories,
                )

        status: Literal["complete", "failed"] = (
            "complete" if service_result.success else "failed"
        )
        with progress_lock:
            snapshot = tracker.record_item_terminal(
                index,
                success=service_result.success,
            )
            emit_locked(
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
                    snapshot=snapshot,
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
    output_package = package_v2_successful_outputs(
        ordered_redaction_results,
        output_dir,
    )
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
    final_progress = tracker.snapshot()

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
            work_plan=work_plan,
            final_progress=final_progress,
        ),
        work_plan=work_plan,
        final_progress=final_progress,
    )


def _service_accepts_progress_callback(
    redaction_service: V2RedactionService | V2RedactionServiceLike,
) -> bool:
    try:
        parameters = inspect.signature(redaction_service.redact_item).parameters
    except (TypeError, ValueError):
        return False
    return "progress_callback" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
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
    work_plan: V2WorkPlan,
    final_progress: V2ProgressSnapshot,
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
        planned_chunk_count=work_plan.total_chunk_count,
        successful_chunk_count=sum(
            item.metadata.successful_chunk_count for item in items
        ),
        failed_chunk_count=sum(item.metadata.failed_chunk_count for item in items),
        skipped_chunk_count=final_progress.skipped_chunk_count,
        retry_count=sum(item.metadata.retry_count for item in items),
        usage=usage,
        error_categories=tuple(dict.fromkeys(categories)),
    )
