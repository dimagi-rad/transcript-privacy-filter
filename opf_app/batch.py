from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Literal

from .models import ParsedItem, RedactionResult
from .outputs import GeneratedOutput, OutputPackage, package_successful_outputs
from .redaction import RedactionService


BatchItemStatus = Literal["pending", "running", "complete", "failed"]
ProgressCallback = Callable[["BatchProgressEvent"], None]

MIN_CONCURRENCY = 1
DEFAULT_CONCURRENCY = 2
MAX_CONCURRENCY = 8


@dataclass(frozen=True)
class BatchProgressEvent:
    status: BatchItemStatus
    current_index: int
    total_count: int
    display_filename: str
    message: str


@dataclass(frozen=True)
class BatchItemResult:
    item: ParsedItem
    status: BatchItemStatus
    redaction_result: RedactionResult
    output: GeneratedOutput


@dataclass(frozen=True)
class BatchResult:
    items: tuple[BatchItemResult, ...]
    output_package: OutputPackage
    progress_events: tuple[BatchProgressEvent, ...]
    concurrency: int

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
    def zip_path(self) -> Path:
        return self.output_package.zip_path


def clamp_concurrency(value: int | None) -> int:
    """Clamp user-selected redaction concurrency to the safe local range."""
    if value is None:
        return DEFAULT_CONCURRENCY
    return max(MIN_CONCURRENCY, min(MAX_CONCURRENCY, int(value)))


def run_redaction_batch(
    items: Iterable[ParsedItem],
    *,
    selected_labels: Iterable[str],
    redaction_service: RedactionService,
    output_dir: str | Path,
    concurrency: int | None = DEFAULT_CONCURRENCY,
    lock_redactor: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> BatchResult:
    """Redact parsed items with bounded concurrency and generate outputs."""
    parsed_items = tuple(items)
    total_count = len(parsed_items)
    worker_count = clamp_concurrency(concurrency)
    labels = tuple(selected_labels)
    progress_events: list[BatchProgressEvent] = []
    progress_lock = threading.Lock()
    redactor_lock = threading.Lock()

    def emit(event: BatchProgressEvent) -> None:
        with progress_lock:
            progress_events.append(event)
        if progress_callback is not None:
            progress_callback(event)

    def process_item(index: int, item: ParsedItem) -> tuple[int, RedactionResult]:
        current_index = index + 1
        emit(
            BatchProgressEvent(
                status="running",
                current_index=current_index,
                total_count=total_count,
                display_filename=item.output_filename,
                message=(
                    f"Now redacting document {current_index}/{total_count}: "
                    f"{item.output_filename}"
                ),
            )
        )

        if item.parse_status == "error" or item.errors:
            result = RedactionResult(
                item=item,
                output_filename=item.output_filename,
                redacted_text=item.body_text,
                success=False,
                selected_categories=labels,
                warnings=item.warnings,
                errors=item.errors or ("Parsed item is not redaction-ready.",),
            )
        else:
            try:
                if lock_redactor:
                    with redactor_lock:
                        result = redaction_service.redact_item(
                            item,
                            selected_labels=labels,
                        )
                else:
                    result = redaction_service.redact_item(
                        item,
                        selected_labels=labels,
                    )
            except Exception as exc:  # noqa: BLE001 - keep batch item isolated
                result = RedactionResult(
                    item=item,
                    output_filename=item.output_filename,
                    redacted_text=item.body_text,
                    success=False,
                    selected_categories=labels,
                    warnings=item.warnings,
                    errors=(f"{type(exc).__name__}: redaction failed for item.",),
                )

        status: BatchItemStatus = "complete" if result.success else "failed"
        emit(
            BatchProgressEvent(
                status=status,
                current_index=current_index,
                total_count=total_count,
                display_filename=item.output_filename,
                message=(
                    f"{'Completed' if result.success else 'Failed'} document "
                    f"{current_index}/{total_count}: {item.output_filename}"
                ),
            )
        )
        return index, result

    redaction_results: list[RedactionResult | None] = [None] * total_count
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(process_item, index, item)
            for index, item in enumerate(parsed_items)
        ]
        for future in as_completed(futures):
            index, result = future.result()
            redaction_results[index] = result

    ordered_results = tuple(
        result for result in redaction_results if result is not None
    )
    output_package = package_successful_outputs(ordered_results, output_dir)
    batch_items = tuple(
        BatchItemResult(
            item=result.item,
            status="complete" if result.success else "failed",
            redaction_result=result,
            output=output,
        )
        for result, output in zip(ordered_results, output_package.outputs)
    )

    return BatchResult(
        items=batch_items,
        output_package=output_package,
        progress_events=tuple(progress_events),
        concurrency=worker_count,
    )
