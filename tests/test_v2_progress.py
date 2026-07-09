from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from opf_app.models import ParsedItem
from opf_app.responses_client import (
    RedactedSentence,
    RedactionApiResult,
    ResponsesClientError,
    ResponsesUsage,
)
from opf_app.v2_batch import (
    V2ProgressTracker,
    V2WorkPlan,
    V2WorkPlanItem,
    build_v2_work_plan,
    calculate_v2_eta_seconds,
    calculate_v2_progress_percentage,
    run_v2_redaction_batch,
)
from opf_app.v2_redaction import V2ChunkLifecycleEvent, V2RedactionService


def test_work_plan_counts_items_sentences_chunks_and_zero_chunk_items() -> None:
    items = (
        _item("first", "Alice one. Bob two. Carol three. Dan four. Eve five."),
        _item("empty", "   "),
        _item("invalid", "Private text.", errors=("parse failed",)),
    )

    plan = build_v2_work_plan(items, sentence_chunk_size=2)

    assert plan.total_item_count == 3
    assert plan.total_sentence_count == 5
    assert plan.total_chunk_count == 3
    assert [item.planned_chunk_count for item in plan.items] == [3, 0, 0]
    assert "Alice" not in repr(plan)


def test_percentage_and_eta_are_deterministic_with_fake_monotonic_clock() -> None:
    clock_values = iter((10.0, 14.0))
    plan = V2WorkPlan(
        sentence_chunk_size=1,
        total_item_count=1,
        total_sentence_count=4,
        total_chunk_count=4,
        items=(V2WorkPlanItem(0, 4, 4),),
    )
    tracker = V2ProgressTracker(plan, monotonic=lambda: next(clock_values))

    changed, snapshot = tracker.resolve_chunk(0, 0, "complete")

    assert changed is True
    assert snapshot.percentage == 25
    assert snapshot.elapsed_seconds == 4.0
    assert snapshot.eta_seconds == 12.0
    assert calculate_v2_progress_percentage(9, 4) == 100
    assert calculate_v2_progress_percentage(0, 0) == 0
    assert calculate_v2_progress_percentage(0, 0, is_terminal=True) == 100
    assert calculate_v2_eta_seconds(
        elapsed_seconds=5.0,
        resolved_chunk_count=0,
        unresolved_chunk_count=4,
    ) is None


def test_service_emits_terminal_chunk_before_starting_the_next_api_call() -> None:
    events: list[V2ChunkLifecycleEvent] = []
    client = SequencedClient(events=events)

    result = V2RedactionService(client).redact_item(
        _item("multi", "Alice one. Bob two."),
        model_id="gpt-test-redactor",
        sentence_chunk_size=1,
        progress_callback=events.append,
    )

    assert result.success is True
    assert [event.status for event in events] == [
        "running",
        "complete",
        "running",
        "complete",
    ]
    assert [event.chunk_index for event in events] == [0, 0, 1, 1]


def test_retry_does_not_inflate_denominator_or_percentage(tmp_path: Path) -> None:
    events = []
    result = run_v2_redaction_batch(
        (_item("retry", "Alice arrived."),),
        model_id="gpt-test-redactor",
        redaction_service=V2RedactionService(RetryThenSuccessClient()),
        output_dir=tmp_path,
        sentence_chunk_size=1,
        api_concurrency=1,
        retry_limit=2,
        progress_callback=events.append,
    )

    chunk_events = [event for event in events if event.event_type == "chunk"]
    retry_event = next(event for event in chunk_events if event.status == "retrying")
    complete_event = next(event for event in chunk_events if event.status == "complete")
    assert result.work_plan.total_chunk_count == 1
    assert retry_event.snapshot is not None
    assert retry_event.snapshot.percentage == 0
    assert complete_event.snapshot is not None
    assert complete_event.snapshot.percentage == 100
    assert result.final_progress.retry_count == 1
    assert result.summary.retry_count == 1


def test_terminal_failure_resolves_later_planned_chunks_as_skipped(
    tmp_path: Path,
) -> None:
    events = []
    result = run_v2_redaction_batch(
        (_item("failure", "Alice one. Bob two. Carol three."),),
        model_id="gpt-test-redactor",
        redaction_service=V2RedactionService(AuthenticationFailureClient()),
        output_dir=tmp_path,
        sentence_chunk_size=1,
        api_concurrency=1,
        progress_callback=events.append,
    )

    terminal_chunk_statuses = [
        event.status
        for event in events
        if event.event_type == "chunk"
        and event.status in {"complete", "failed", "skipped"}
    ]
    assert terminal_chunk_statuses == ["failed", "skipped", "skipped"]
    assert result.failed_count == 1
    assert result.final_progress.failed_chunk_count == 1
    assert result.final_progress.skipped_chunk_count == 2
    assert result.final_progress.unresolved_chunk_count == 0
    assert result.final_progress.percentage == 100
    assert result.summary.skipped_chunk_count == 2


@dataclass
class SequencedClient:
    events: list[V2ChunkLifecycleEvent]

    def __post_init__(self) -> None:
        self.call_count = 0

    def redact_sentence_batch(self, *, model_id: str, masked_batch: object) -> RedactionApiResult:
        self.call_count += 1
        if self.call_count == 2:
            assert any(
                event.status == "complete" and event.chunk_index == 0
                for event in self.events
            )
        return _success_result(masked_batch)


@dataclass
class RetryThenSuccessClient:
    call_count: int = 0

    def redact_sentence_batch(self, *, model_id: str, masked_batch: object) -> RedactionApiResult:
        self.call_count += 1
        if self.call_count == 1:
            return RedactionApiResult(sentences=(), usage=ResponsesUsage())
        return _success_result(masked_batch)


class AuthenticationFailureClient:
    def redact_sentence_batch(self, *, model_id: str, masked_batch: object) -> RedactionApiResult:
        raise ResponsesClientError("authentication")


def _success_result(masked_batch: object) -> RedactionApiResult:
    return RedactionApiResult(
        sentences=tuple(
            RedactedSentence(
                id=sentence.sentence_id,
                redacted_text="<PRIVATE_PERSON>.",
            )
            for sentence in masked_batch.sentences
        ),
        usage=ResponsesUsage(input_tokens=1, output_tokens=1, total_tokens=2),
    )


def _item(
    identifier: str,
    body_text: str,
    *,
    errors: tuple[str, ...] = (),
) -> ParsedItem:
    return ParsedItem(
        item_name=identifier,
        source_name=f"{identifier}.txt",
        source_type="document",
        chat_date="2026-05-11",
        user_identifier=identifier,
        body_text=body_text,
        output_filename=f"redacted-transcript-2026-05-11-{identifier}.docx",
        parse_status="error" if errors else "ready",
        errors=errors,
    )
