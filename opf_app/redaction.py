from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from opf._common.label_space import BACKGROUND_CLASS_LABEL, SPAN_CLASS_NAMES_BY_CATEGORY_VERSION

from .models import ParsedItem, RedactionResult


DEFAULT_RUNTIME_LABELS = tuple(
    label
    for label in SPAN_CLASS_NAMES_BY_CATEGORY_VERSION["v2"]
    if label != BACKGROUND_CLASS_LABEL
)


class RedactorLike(Protocol):
    def redact(self, text: str) -> object:
        """Return an OPF structured redaction result."""


@dataclass(frozen=True)
class SpanForReplacement:
    label: str
    start: int
    end: int
    placeholder: str


class RedactionService:
    """Category-selective wrapper around a reusable OPF-like redactor."""

    def __init__(self, redactor: RedactorLike | None = None) -> None:
        self._redactor = redactor

    @property
    def redactor(self) -> RedactorLike:
        if self._redactor is None:
            self._redactor = create_typed_opf_redactor()
        return self._redactor

    def runtime_labels(self) -> tuple[str, ...]:
        return list_runtime_labels(self._redactor)

    def redact_item(
        self,
        item: ParsedItem,
        *,
        selected_labels: Iterable[str],
    ) -> RedactionResult:
        return redact_item(item, self.redactor, selected_labels=selected_labels)


def create_typed_opf_redactor(**kwargs: object) -> RedactorLike:
    """Create a reusable OPF instance configured for typed span output."""
    from opf import OPF

    kwargs.setdefault("device", resolve_default_opf_device())
    return OPF(output_mode="typed", output_text_only=False, **kwargs)


def resolve_default_opf_device(
    *,
    cuda_available: Callable[[], bool] | None = None,
) -> str:
    """Use CUDA when available, otherwise keep local app startup CPU-safe."""
    if cuda_available is None:
        try:
            import torch
        except Exception:  # noqa: BLE001 - CPU fallback is safest for local app use
            return "cpu"
        cuda_available = torch.cuda.is_available
    return "cuda" if cuda_available() else "cpu"


def list_runtime_labels(redactor: object | None = None) -> tuple[str, ...]:
    """Return available span labels from a redactor or the default OPF taxonomy."""
    labels = _labels_from_redactor(redactor)
    return labels if labels else DEFAULT_RUNTIME_LABELS


def redact_item(
    item: ParsedItem,
    redactor: RedactorLike,
    *,
    selected_labels: Iterable[str],
) -> RedactionResult:
    """Run OPF and apply placeholders only for selected categories."""
    selected_label_set = set(selected_labels)
    try:
        opf_result = redactor.redact(item.body_text)
    except Exception as exc:  # noqa: BLE001 - convert per-item failures to result state
        return RedactionResult(
            item=item,
            output_filename=item.output_filename,
            redacted_text=item.body_text,
            success=False,
            selected_categories=tuple(sorted(selected_label_set)),
            warnings=item.warnings,
            errors=(f"{type(exc).__name__}: redaction failed for item.",),
        )

    spans = _coerce_spans(getattr(opf_result, "detected_spans", ()))
    text = str(getattr(opf_result, "text", item.body_text))
    selected_spans = tuple(
        span for span in spans if span.label in selected_label_set
    )
    redacted_text = apply_selected_replacements(text, selected_spans)
    detected_counts = _count_by_label(spans)
    selected_counts = _count_by_label(selected_spans)

    warnings = list(item.warnings)
    warning = getattr(opf_result, "warning", None)
    if warning:
        warnings.append(str(warning))

    return RedactionResult(
        item=item,
        output_filename=item.output_filename,
        redacted_text=redacted_text,
        success=True,
        detected_span_count=len(spans),
        selected_span_count=len(selected_spans),
        detected_counts_by_label=detected_counts,
        selected_counts_by_label=selected_counts,
        detected_categories=tuple(sorted(detected_counts)),
        selected_categories=tuple(sorted(selected_label_set)),
        warnings=tuple(warnings),
        errors=item.errors,
    )


def apply_selected_replacements(
    text: str,
    spans: Sequence[SpanForReplacement],
) -> str:
    """Apply selected non-overlapping span replacements with stable offsets."""
    if not spans:
        return text

    pieces: list[str] = []
    cursor = 0
    for span in sorted(spans, key=lambda item: (item.start, item.end)):
        if span.start < cursor or span.end <= span.start:
            continue
        if span.start > len(text):
            continue
        end = min(span.end, len(text))
        pieces.append(text[cursor : span.start])
        pieces.append(span.placeholder)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _coerce_spans(spans: Iterable[object]) -> tuple[SpanForReplacement, ...]:
    coerced: list[SpanForReplacement] = []
    for span in spans:
        coerced.append(
            SpanForReplacement(
                label=str(getattr(span, "label")),
                start=int(getattr(span, "start")),
                end=int(getattr(span, "end")),
                placeholder=str(getattr(span, "placeholder")),
            )
        )
    return tuple(coerced)


def _count_by_label(spans: Iterable[SpanForReplacement]) -> dict[str, int]:
    return dict(sorted(Counter(span.label for span in spans).items()))


def _labels_from_redactor(redactor: object | None) -> tuple[str, ...]:
    if redactor is None:
        return ()

    direct_labels = getattr(redactor, "runtime_labels", None)
    if direct_labels is not None:
        return _without_background(direct_labels)

    get_runtime = getattr(redactor, "get_runtime", None)
    if callable(get_runtime):
        runtime = get_runtime()
        label_info = getattr(runtime, "label_info", None)
        labels = getattr(label_info, "span_class_names", None)
        if labels is not None:
            return _without_background(labels)

    return ()


def _without_background(labels: Any) -> tuple[str, ...]:
    return tuple(
        str(label)
        for label in labels
        if str(label) != BACKGROUND_CLASS_LABEL
    )
