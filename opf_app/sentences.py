from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import re

from .models import ParsedItem


SENTENCE_ID_PREFIX = "s_"
_SENTENCE_ID_WIDTH = 6
_BOUNDARY_PUNCTUATION = ".?!\u3002\uff01\uff1f"
_CLOSING_PUNCTUATION = "\"')]}>\u201d\u2019"
_TRANSCRIPT_PREFIX_RE = re.compile(
    r"^(?P<prefix>\[[^\]\n]+\]\s+"
    r"(?P<role>[A-Za-z][A-Za-z0-9_-]*)\:\s*)"
)
_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
_NON_BOUNDARY_ABBREVIATION_RE = re.compile(
    r"(?i)(?:\b(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|no)|\b(?:e\.g|i\.e))\.$"
)


@dataclass(frozen=True)
class TranscriptLine:
    """One parsed body line with local structure separated from utterance text."""

    item_index: int
    line_index: int
    role: str | None = None
    prefix: str = field(default="", repr=False)
    utterance_text: str = field(default="", repr=False)
    line_ending: str = field(default="", repr=False)

    @property
    def has_redactable_text(self) -> bool:
        return bool(self.utterance_text.strip())

    def render(self, utterance_text: str | None = None) -> str:
        text = self.utterance_text if utterance_text is None else utterance_text
        return f"{self.prefix}{text}{self.line_ending}"


@dataclass(frozen=True)
class SentenceUnit:
    """One API-bound sentence plus local reconstruction coordinates."""

    sentence_id: str
    item_index: int
    line_index: int
    sentence_index: int
    start_char: int
    end_char: int
    role: str | None = None
    original_text: str = field(default="", repr=False)

    @property
    def api_text(self) -> str:
        return self.original_text


@dataclass(frozen=True)
class SegmentedParsedItem:
    """Sentence segmentation result for one parsed item."""

    item: ParsedItem
    item_index: int
    lines: tuple[TranscriptLine, ...] = field(default_factory=tuple)
    sentences: tuple[SentenceUnit, ...] = field(default_factory=tuple)


def segment_parsed_items(items: Iterable[ParsedItem]) -> tuple[SegmentedParsedItem, ...]:
    """Split parsed items into sentence units with run-stable unique IDs."""
    segmented_items: list[SegmentedParsedItem] = []
    next_sentence_number = 1
    for item_index, item in enumerate(items):
        segmented_item, next_sentence_number = segment_parsed_item(
            item,
            item_index=item_index,
            start_sentence_number=next_sentence_number,
        )
        segmented_items.append(segmented_item)
    return tuple(segmented_items)


def segment_parsed_item(
    item: ParsedItem,
    *,
    item_index: int = 0,
    start_sentence_number: int = 1,
) -> tuple[SegmentedParsedItem, int]:
    """Split one parsed body into transcript lines and API-bound sentences."""
    lines: list[TranscriptLine] = []
    sentences: list[SentenceUnit] = []
    next_sentence_number = start_sentence_number
    inherited_role: str | None = None

    for line_index, raw_line in enumerate(item.body_text.splitlines(keepends=True)):
        line_text, line_ending = _split_line_ending(raw_line)
        line = parse_transcript_line(
            line_text,
            item_index=item_index,
            line_index=line_index,
            line_ending=line_ending,
            inherited_role=inherited_role,
        )
        lines.append(line)

        if line.role is not None:
            inherited_role = line.role
        if not line.has_redactable_text:
            inherited_role = None
            continue

        sentence_index = 0
        for start_char, end_char in split_sentence_spans(line.utterance_text):
            sentence_id = format_sentence_id(next_sentence_number)
            sentences.append(
                SentenceUnit(
                    sentence_id=sentence_id,
                    item_index=item_index,
                    line_index=line_index,
                    sentence_index=sentence_index,
                    start_char=start_char,
                    end_char=end_char,
                    role=line.role,
                    original_text=line.utterance_text[start_char:end_char],
                )
            )
            next_sentence_number += 1
            sentence_index += 1

    return (
        SegmentedParsedItem(
            item=item,
            item_index=item_index,
            lines=tuple(lines),
            sentences=tuple(sentences),
        ),
        next_sentence_number,
    )


def parse_transcript_line(
    line_text: str,
    *,
    item_index: int = 0,
    line_index: int = 0,
    line_ending: str = "",
    inherited_role: str | None = None,
) -> TranscriptLine:
    """Separate a structural transcript prefix from redactable line text."""
    match = _TRANSCRIPT_PREFIX_RE.match(line_text)
    if match:
        prefix = match.group("prefix")
        return TranscriptLine(
            item_index=item_index,
            line_index=line_index,
            role=match.group("role"),
            prefix=prefix,
            utterance_text=line_text[len(prefix) :],
            line_ending=line_ending,
        )

    return TranscriptLine(
        item_index=item_index,
        line_index=line_index,
        role=inherited_role,
        utterance_text=line_text,
        line_ending=line_ending,
    )


def split_sentences(text: str) -> tuple[str, ...]:
    """Return API-bound sentence text without structural prefixes."""
    return tuple(text[start:end] for start, end in split_sentence_spans(text))


def split_sentence_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Return start/end spans for sentence-like chunks in utterance text."""
    if not text.strip():
        return ()

    protected_ranges = _protected_punctuation_ranges(text)
    spans: list[tuple[int, int]] = []
    sentence_start = _next_non_whitespace(text, 0)
    index = sentence_start if sentence_start is not None else len(text)

    while index < len(text):
        character = text[index]
        if character in _BOUNDARY_PUNCTUATION and _is_sentence_boundary(
            text,
            index,
            protected_ranges,
        ):
            sentence_end = _include_closing_punctuation(text, index + 1)
            spans.append((sentence_start, sentence_end))
            sentence_start = _next_non_whitespace(text, sentence_end)
            if sentence_start is None:
                break
            index = sentence_start
            continue
        index += 1

    if sentence_start is not None:
        sentence_end = _previous_non_whitespace_end(text, len(text))
        if sentence_end > sentence_start:
            spans.append((sentence_start, sentence_end))

    return tuple(spans)


def reconstruct_body(
    segmented_item: SegmentedParsedItem,
    sentence_text_by_id: Mapping[str, str] | None = None,
) -> str:
    """Rebuild a parsed item body from unchanged or redacted sentence text."""
    replacements = sentence_text_by_id or {}
    sentences_by_line: dict[int, list[SentenceUnit]] = {}
    for sentence in segmented_item.sentences:
        sentences_by_line.setdefault(sentence.line_index, []).append(sentence)

    rendered_lines: list[str] = []
    for line in segmented_item.lines:
        cursor = 0
        utterance_parts: list[str] = []
        line_sentences = sorted(
            sentences_by_line.get(line.line_index, ()),
            key=lambda sentence: sentence.start_char,
        )
        for sentence in line_sentences:
            utterance_parts.append(line.utterance_text[cursor : sentence.start_char])
            utterance_parts.append(
                replacements.get(sentence.sentence_id, sentence.original_text)
            )
            cursor = sentence.end_char
        utterance_parts.append(line.utterance_text[cursor:])
        rendered_lines.append(line.render("".join(utterance_parts)))

    return "".join(rendered_lines)


def format_sentence_id(sentence_number: int) -> str:
    if sentence_number < 1:
        raise ValueError("Sentence numbers must start at 1.")
    return f"{SENTENCE_ID_PREFIX}{sentence_number:0{_SENTENCE_ID_WIDTH}d}"


def _split_line_ending(raw_line: str) -> tuple[str, str]:
    if raw_line.endswith("\r\n"):
        return raw_line[:-2], "\r\n"
    if raw_line.endswith("\n") or raw_line.endswith("\r"):
        return raw_line[:-1], raw_line[-1]
    return raw_line, ""


def _protected_punctuation_ranges(text: str) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    for pattern in (_URL_RE, _EMAIL_RE):
        for match in pattern.finditer(text):
            start, end = match.span()
            while end > start and text[end - 1] in ".,;:!?":
                end -= 1
            ranges.append((start, end))
    return tuple(ranges)


def _is_sentence_boundary(
    text: str,
    index: int,
    protected_ranges: tuple[tuple[int, int], ...],
) -> bool:
    if _is_index_in_ranges(index, protected_ranges):
        return False
    if text[index] == ".":
        if _is_decimal_point(text, index):
            return False
        if _is_initial_period(text, index):
            return False
        if _is_non_boundary_abbreviation(text, index):
            return False

    next_index = _include_closing_punctuation(text, index + 1)
    if next_index >= len(text):
        return True
    return text[next_index].isspace()


def _is_index_in_ranges(index: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= index < end for start, end in ranges)


def _is_decimal_point(text: str, index: int) -> bool:
    return (
        index > 0
        and index + 1 < len(text)
        and text[index - 1].isdigit()
        and text[index + 1].isdigit()
    )


def _is_initial_period(text: str, index: int) -> bool:
    if index == 0 or not text[index - 1].isalpha():
        return False
    token_start = index - 1
    while token_start > 0 and text[token_start - 1].isalpha():
        token_start -= 1
    return index - token_start == 1 and text[token_start:index].isupper()


def _is_non_boundary_abbreviation(text: str, index: int) -> bool:
    fragment = text[max(0, index - 12) : index + 1]
    return bool(_NON_BOUNDARY_ABBREVIATION_RE.search(fragment))


def _include_closing_punctuation(text: str, index: int) -> int:
    while index < len(text) and text[index] in _CLOSING_PUNCTUATION:
        index += 1
    return index


def _next_non_whitespace(text: str, start: int) -> int | None:
    for index in range(start, len(text)):
        if not text[index].isspace():
            return index
    return None


def _previous_non_whitespace_end(text: str, end: int) -> int:
    index = end
    while index > 0 and text[index - 1].isspace():
        index -= 1
    return index
