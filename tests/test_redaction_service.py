from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from opf_app.models import ParsedItem
from opf_app.redaction import (
    DEFAULT_RUNTIME_LABELS,
    RedactionService,
    SpanForReplacement,
    apply_selected_replacements,
    create_typed_opf_redactor,
    list_runtime_labels,
    redact_item,
    resolve_default_opf_device,
)


@dataclass(frozen=True)
class FakeOpfResult:
    text: str
    detected_spans: tuple[SpanForReplacement, ...]
    warning: str | None = None


class FakeRedactor:
    def __init__(
        self,
        *,
        text: str,
        spans: tuple[SpanForReplacement, ...],
        warning: str | None = None,
    ) -> None:
        self._result = FakeOpfResult(text=text, detected_spans=spans, warning=warning)

    def redact(self, _text: str) -> FakeOpfResult:
        return self._result


class FailingRedactor:
    def redact(self, _text: str) -> object:
        raise RuntimeError("simulated failure with no source text")


def test_selecting_all_labels_redacts_all_detected_spans() -> None:
    item = _item("Alice emailed alice@example.test.")
    spans = (
        SpanForReplacement("private_person", 0, 5, "<PRIVATE_PERSON>"),
        SpanForReplacement("private_email", 14, 32, "<PRIVATE_EMAIL>"),
    )

    result = redact_item(
        item,
        FakeRedactor(text=item.body_text, spans=spans),
        selected_labels=["private_person", "private_email"],
    )

    assert result.redacted_text == "<PRIVATE_PERSON> emailed <PRIVATE_EMAIL>."
    assert result.detected_span_count == 2
    assert result.selected_span_count == 2
    assert result.detected_counts_by_label == {
        "private_email": 1,
        "private_person": 1,
    }


def test_selecting_person_only_leaves_unselected_spans_unchanged() -> None:
    item = _item("Alice emailed alice@example.test.")
    spans = (
        SpanForReplacement("private_person", 0, 5, "<PRIVATE_PERSON>"),
        SpanForReplacement("private_email", 14, 32, "<PRIVATE_EMAIL>"),
    )

    result = redact_item(
        item,
        FakeRedactor(text=item.body_text, spans=spans),
        selected_labels=["private_person"],
    )

    assert result.redacted_text == "<PRIVATE_PERSON> emailed alice@example.test."
    assert result.selected_counts_by_label == {"private_person": 1}


def test_selecting_no_labels_leaves_text_unchanged() -> None:
    item = _item("Alice emailed alice@example.test.")
    spans = (
        SpanForReplacement("private_person", 0, 5, "<PRIVATE_PERSON>"),
        SpanForReplacement("private_email", 14, 32, "<PRIVATE_EMAIL>"),
    )

    result = redact_item(
        item,
        FakeRedactor(text=item.body_text, spans=spans),
        selected_labels=[],
    )

    assert result.redacted_text == item.body_text
    assert result.detected_span_count == 2
    assert result.selected_span_count == 0


def test_adjacent_and_non_overlapping_spans_use_typed_placeholders() -> None:
    text = "AliceBob visited Rome"
    spans = (
        SpanForReplacement("private_person", 0, 5, "<PRIVATE_PERSON>"),
        SpanForReplacement("private_person", 5, 8, "<PRIVATE_PERSON>"),
        SpanForReplacement("private_address", 17, 21, "<PRIVATE_ADDRESS>"),
    )

    redacted = apply_selected_replacements(text, spans)

    assert redacted == "<PRIVATE_PERSON><PRIVATE_PERSON> visited <PRIVATE_ADDRESS>"


def test_warning_and_error_propagation() -> None:
    item = _item("Alice")
    warning_result = redact_item(
        item,
        FakeRedactor(
            text=item.body_text,
            spans=(SpanForReplacement("private_person", 0, 5, "<PRIVATE_PERSON>"),),
            warning="Tokenizer round-trip warning",
        ),
        selected_labels=["private_person"],
    )

    assert warning_result.warnings == ("Tokenizer round-trip warning",)

    error_result = redact_item(
        item,
        FailingRedactor(),
        selected_labels=["private_person"],
    )

    assert error_result.success is False
    assert error_result.redacted_text == item.body_text
    assert error_result.errors == ("RuntimeError: redaction failed for item.",)


def test_runtime_label_listing_uses_default_and_custom_labels() -> None:
    assert list_runtime_labels() == DEFAULT_RUNTIME_LABELS

    custom_redactor = SimpleNamespace(runtime_labels=("O", "custom_one", "custom_two"))
    assert list_runtime_labels(custom_redactor) == ("custom_one", "custom_two")

    runtime_redactor = SimpleNamespace(
        get_runtime=lambda: SimpleNamespace(
            label_info=SimpleNamespace(
                span_class_names=("O", "runtime_one", "runtime_two")
            )
        )
    )
    assert list_runtime_labels(runtime_redactor) == ("runtime_one", "runtime_two")


def test_service_reuses_injected_redactor() -> None:
    item = _item("Alice")
    service = RedactionService(
        FakeRedactor(
            text=item.body_text,
            spans=(SpanForReplacement("private_person", 0, 5, "<PRIVATE_PERSON>"),),
        )
    )

    result = service.redact_item(item, selected_labels=["private_person"])

    assert result.redacted_text == "<PRIVATE_PERSON>"


def test_resolve_default_opf_device_uses_cuda_when_available() -> None:
    assert resolve_default_opf_device(cuda_available=lambda: True) == "cuda"
    assert resolve_default_opf_device(cuda_available=lambda: False) == "cpu"


def test_create_typed_opf_redactor_uses_resolved_default_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, object] = {}

    class CapturingOpf:
        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setattr("opf.OPF", CapturingOpf)
    monkeypatch.setattr("opf_app.redaction.resolve_default_opf_device", lambda: "cuda")

    redactor = create_typed_opf_redactor()

    assert isinstance(redactor, CapturingOpf)
    assert captured_kwargs["device"] == "cuda"
    assert captured_kwargs["output_mode"] == "typed"
    assert captured_kwargs["output_text_only"] is False


def test_create_typed_opf_redactor_preserves_device_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, object] = {}

    class CapturingOpf:
        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setattr("opf.OPF", CapturingOpf)

    create_typed_opf_redactor(device="cuda")

    assert captured_kwargs["device"] == "cuda"


def _item(body_text: str) -> ParsedItem:
    return ParsedItem(
        item_name="item",
        source_name="source.txt",
        source_type="document",
        chat_date="2026-05-19",
        user_identifier="source",
        body_text=body_text,
        output_filename="redacted-transcript-2026-05-19-source.docx",
    )
