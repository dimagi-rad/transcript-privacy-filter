from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


SourceType = Literal["ocs_csv", "document"]
ParseStatus = Literal["parsed", "warning", "error"]


@dataclass(frozen=True)
class ParsedItem:
    """One transcript-like item parsed from a CSV session or source document."""

    item_name: str
    source_name: str
    source_type: SourceType
    chat_date: str
    user_identifier: str
    body_text: str = field(repr=False)
    output_filename: str
    parse_status: ParseStatus = "parsed"
    session_id: str | None = None
    source_path: str | None = None
    message_count: int | None = None
    line_count: int | None = None
    character_count: int | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "errors", tuple(self.errors))
        if self.character_count is None:
            object.__setattr__(self, "character_count", len(self.body_text))

    def review_row(self) -> dict[str, object]:
        """Return non-sensitive fields for Streamlit review tables."""
        count = self.message_count if self.message_count is not None else self.line_count
        if count is None:
            count = self.character_count
        return {
            "item_name": self.item_name,
            "source_type": self.source_type,
            "chat_date": self.chat_date,
            "user_identifier": self.user_identifier,
            "count": count,
            "parse_status": self.parse_status,
            "output_filename": self.output_filename,
        }


@dataclass(frozen=True)
class RedactionJob:
    """A parsed item plus the runtime category choices for redaction."""

    item: ParsedItem
    selected_categories: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "selected_categories", tuple(self.selected_categories)
        )


@dataclass(frozen=True)
class RedactionResult:
    """Result metadata for one redaction job."""

    item: ParsedItem
    output_filename: str
    redacted_text: str = field(repr=False)
    success: bool = True
    detected_span_count: int = 0
    selected_span_count: int = 0
    detected_counts_by_label: dict[str, int] = field(default_factory=dict)
    selected_counts_by_label: dict[str, int] = field(default_factory=dict)
    detected_categories: tuple[str, ...] = field(default_factory=tuple)
    selected_categories: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "detected_counts_by_label", dict(self.detected_counts_by_label)
        )
        object.__setattr__(
            self, "selected_counts_by_label", dict(self.selected_counts_by_label)
        )
        object.__setattr__(
            self, "detected_categories", tuple(self.detected_categories)
        )
        object.__setattr__(
            self, "selected_categories", tuple(self.selected_categories)
        )
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "errors", tuple(self.errors))
