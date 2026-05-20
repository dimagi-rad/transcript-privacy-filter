from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from pathlib import Path
import zipfile

from opf_app.batch import clamp_concurrency, run_redaction_batch
from opf_app.models import ParsedItem
from opf_app.redaction import RedactionService, SpanForReplacement


@dataclass(frozen=True)
class FakeOpfResult:
    text: str
    detected_spans: tuple[SpanForReplacement, ...] = ()
    warning: str | None = None


class TrackingRedactor:
    def __init__(self, *, fail_on: str | None = None, delay: float = 0.0) -> None:
        self.fail_on = fail_on
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.calls: list[str] = []
        self.lock = threading.Lock()

    def redact(self, text: str) -> FakeOpfResult:
        with self.lock:
            self.calls.append(text)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            if text == self.fail_on:
                raise RuntimeError("planned failure")
            return FakeOpfResult(text=text.upper())
        finally:
            with self.lock:
                self.active -= 1


def test_concurrency_never_exceeds_selected_limit(tmp_path: Path) -> None:
    redactor = TrackingRedactor(delay=0.05)
    result = run_redaction_batch(
        _items(6),
        selected_labels=[],
        redaction_service=RedactionService(redactor),
        output_dir=tmp_path,
        concurrency=3,
    )

    assert redactor.max_active <= 3
    assert redactor.max_active > 1
    assert result.complete_count == 6


def test_concurrency_one_processes_sequentially(tmp_path: Path) -> None:
    redactor = TrackingRedactor()
    items = _items(3)

    result = run_redaction_batch(
        items,
        selected_labels=[],
        redaction_service=RedactionService(redactor),
        output_dir=tmp_path,
        concurrency=1,
    )

    assert redactor.calls == [item.body_text for item in items]
    assert result.concurrency == 1


def test_redactor_lock_serializes_opf_calls_when_requested(tmp_path: Path) -> None:
    redactor = TrackingRedactor(delay=0.05)

    run_redaction_batch(
        _items(4),
        selected_labels=[],
        redaction_service=RedactionService(redactor),
        output_dir=tmp_path,
        concurrency=4,
        lock_redactor=True,
    )

    assert redactor.max_active == 1


def test_error_isolation_keeps_successful_outputs(tmp_path: Path) -> None:
    redactor = TrackingRedactor(fail_on="body-2")

    result = run_redaction_batch(
        _items(3),
        selected_labels=[],
        redaction_service=RedactionService(redactor),
        output_dir=tmp_path,
        concurrency=2,
    )

    assert result.complete_count == 2
    assert result.failed_count == 1
    assert result.zip_path.is_file()
    with zipfile.ZipFile(result.zip_path) as archive:
        assert len(archive.namelist()) == 2


def test_outputs_are_generated_for_successful_items(tmp_path: Path) -> None:
    redactor = TrackingRedactor()

    result = run_redaction_batch(
        _items(2),
        selected_labels=[],
        redaction_service=RedactionService(redactor),
        output_dir=tmp_path,
        concurrency=2,
    )

    assert all(item.output.success for item in result.items)
    assert all(item.output.path and item.output.path.is_file() for item in result.items)


def test_concurrency_clamping() -> None:
    assert clamp_concurrency(None) == 2
    assert clamp_concurrency(0) == 1
    assert clamp_concurrency(99) == 8
    assert clamp_concurrency(3) == 3


def test_progress_events_are_ui_friendly(tmp_path: Path) -> None:
    events = []

    result = run_redaction_batch(
        _items(1),
        selected_labels=[],
        redaction_service=RedactionService(TrackingRedactor()),
        output_dir=tmp_path,
        concurrency=1,
        progress_callback=events.append,
    )

    assert events == list(result.progress_events)
    assert events[0].message.startswith("Now redacting document 1/1:")
    assert events[0].display_filename == "redacted-transcript-2026-05-11-user-0.docx"
    assert events[-1].status == "complete"


def _items(count: int) -> tuple[ParsedItem, ...]:
    return tuple(
        ParsedItem(
            item_name=f"item-{index}",
            source_name=f"source-{index}.txt",
            source_type="document",
            chat_date="2026-05-11",
            user_identifier=f"user-{index}",
            body_text=f"body-{index}",
            output_filename=f"redacted-transcript-2026-05-11-user-{index}.docx",
        )
        for index in range(count)
    )
