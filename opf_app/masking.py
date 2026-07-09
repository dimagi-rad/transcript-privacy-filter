from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import re
from typing import Literal

from .redaction import parse_preserved_values
from .sentences import SentenceUnit


MASK_TOKEN_PREFIX = "__KEEP_"
MASK_TOKEN_WIDTH = 6
MASK_TOKEN_RE = re.compile(r"__KEEP_\d{6}__")
_KEEP_LIKE_RE = re.compile(r"(?i)_*KEEP[_-]?\d{1,12}_*")

MaskValidationIssueKind = Literal["missing", "duplicated", "modified", "unexpected"]


@dataclass(frozen=True)
class PreserveMask:
    """Local-only map from a non-sensitive token to its original value."""

    token: str
    sentence_id: str
    original_text: str = field(repr=False)


@dataclass(frozen=True)
class MaskedSentenceUnit:
    """Sentence unit prepared for API submission with preserve masks applied."""

    sentence: SentenceUnit
    masked_text: str = field(repr=False)
    mask_tokens: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mask_tokens", tuple(self.mask_tokens))

    @property
    def sentence_id(self) -> str:
        return self.sentence.sentence_id

    @property
    def api_text(self) -> str:
        return self.masked_text


@dataclass(frozen=True)
class MaskedSentenceBatch:
    """Masked sentence set plus local restoration data."""

    sentences: tuple[MaskedSentenceUnit, ...] = field(default_factory=tuple)
    masks_by_token: Mapping[str, PreserveMask] = field(
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "sentences", tuple(self.sentences))
        object.__setattr__(self, "masks_by_token", dict(self.masks_by_token))

    def masked_text_by_id(self) -> dict[str, str]:
        return {
            masked_sentence.sentence_id: masked_sentence.masked_text
            for masked_sentence in self.sentences
        }


@dataclass(frozen=True)
class MaskValidationIssue:
    """A non-sensitive preserve-token validation issue."""

    sentence_id: str
    token: str
    issue: MaskValidationIssueKind
    count: int = 0


class MaskValidationError(ValueError):
    """Raised when preserve mask tokens are missing or damaged."""

    def __init__(self, issues: Iterable[MaskValidationIssue]) -> None:
        self.issues = tuple(issues)
        super().__init__(_summarize_validation_issues(self.issues))


@dataclass(frozen=True)
class _PreserveMatch:
    start: int
    end: int
    original_text: str = field(repr=False)


def format_mask_token(mask_number: int) -> str:
    if mask_number < 1:
        raise ValueError("Mask numbers must start at 1.")
    return f"{MASK_TOKEN_PREFIX}{mask_number:0{MASK_TOKEN_WIDTH}d}__"


def mask_sentence_units(
    sentences: Iterable[SentenceUnit],
    preserved_values: Iterable[str] | str,
    *,
    start_mask_number: int = 1,
) -> MaskedSentenceBatch:
    """Apply local preserved-value masks to sentence units."""
    if start_mask_number < 1:
        raise ValueError("Mask numbers must start at 1.")

    values = normalize_preserved_values(preserved_values)
    next_mask_number = start_mask_number
    masked_sentences: list[MaskedSentenceUnit] = []
    masks_by_token: dict[str, PreserveMask] = {}

    for sentence in sentences:
        masked_text, sentence_masks, next_mask_number = _mask_text(
            sentence.original_text,
            sentence_id=sentence.sentence_id,
            preserved_values=values,
            next_mask_number=next_mask_number,
        )
        for mask in sentence_masks:
            masks_by_token[mask.token] = mask
        masked_sentences.append(
            MaskedSentenceUnit(
                sentence=sentence,
                masked_text=masked_text,
                mask_tokens=tuple(mask.token for mask in sentence_masks),
            )
        )

    return MaskedSentenceBatch(
        sentences=tuple(masked_sentences),
        masks_by_token=masks_by_token,
    )


def normalize_preserved_values(
    preserved_values: Iterable[str] | str,
) -> tuple[str, ...]:
    """Normalize v2 preserve values with v1 comma-separated behavior."""
    if isinstance(preserved_values, str):
        return parse_preserved_values(preserved_values)

    values: list[str] = []
    seen: set[str] = set()
    for raw_value in preserved_values:
        value = str(raw_value).strip()
        normalized = value.casefold()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        values.append(value)
    return tuple(values)


def validate_mask_tokens(
    redacted_text_by_id: Mapping[str, str],
    masked_batch: MaskedSentenceBatch,
) -> tuple[MaskValidationIssue, ...]:
    """Return preserve-token issues without exposing source or redacted text."""
    expected_by_sentence = {
        masked_sentence.sentence_id: set(masked_sentence.mask_tokens)
        for masked_sentence in masked_batch.sentences
    }
    issues: list[MaskValidationIssue] = []

    for masked_sentence in masked_batch.sentences:
        sentence_id = masked_sentence.sentence_id
        expected_tokens = expected_by_sentence[sentence_id]
        text = str(redacted_text_by_id.get(sentence_id, ""))
        token_counts = Counter(MASK_TOKEN_RE.findall(text))

        for token in masked_sentence.mask_tokens:
            count = token_counts.get(token, 0)
            if count == 0:
                issues.append(
                    MaskValidationIssue(
                        sentence_id=sentence_id,
                        token=token,
                        issue="missing",
                    )
                )
            elif count > 1:
                issues.append(
                    MaskValidationIssue(
                        sentence_id=sentence_id,
                        token=token,
                        issue="duplicated",
                        count=count,
                    )
                )

        issues.extend(
            _unexpected_or_modified_token_issues(
                sentence_id=sentence_id,
                text=text,
                expected_tokens=expected_tokens,
            )
        )

    expected_sentence_ids = set(expected_by_sentence)
    for sentence_id, text in redacted_text_by_id.items():
        if sentence_id in expected_sentence_ids:
            continue
        issues.extend(
            _unexpected_or_modified_token_issues(
                sentence_id=sentence_id,
                text=str(text),
                expected_tokens=set(),
            )
        )

    return tuple(_dedupe_issues(issues))


def require_valid_mask_tokens(
    redacted_text_by_id: Mapping[str, str],
    masked_batch: MaskedSentenceBatch,
) -> None:
    issues = validate_mask_tokens(redacted_text_by_id, masked_batch)
    if issues:
        raise MaskValidationError(issues)


def restore_masked_text(
    redacted_text: str,
    masks_by_token: Mapping[str, PreserveMask],
) -> str:
    """Restore original preserved values from a validated model output string."""
    restored_text = redacted_text
    for token, mask in sorted(masks_by_token.items(), key=lambda item: item[0]):
        restored_text = restored_text.replace(token, mask.original_text)
    return restored_text


def restore_sentence_texts(
    redacted_text_by_id: Mapping[str, str],
    masked_batch: MaskedSentenceBatch,
    *,
    validate: bool = True,
) -> dict[str, str]:
    """Validate and restore masked sentence outputs by sentence ID."""
    if validate:
        require_valid_mask_tokens(redacted_text_by_id, masked_batch)
    return {
        sentence_id: restore_masked_text(str(text), masked_batch.masks_by_token)
        for sentence_id, text in redacted_text_by_id.items()
    }


def _mask_text(
    text: str,
    *,
    sentence_id: str,
    preserved_values: tuple[str, ...],
    next_mask_number: int,
) -> tuple[str, tuple[PreserveMask, ...], int]:
    if not preserved_values:
        return text, (), next_mask_number

    matches = _find_preserve_matches(text, preserved_values)
    if not matches:
        return text, (), next_mask_number

    pieces: list[str] = []
    masks: list[PreserveMask] = []
    cursor = 0
    for match in matches:
        token = format_mask_token(next_mask_number)
        next_mask_number += 1
        pieces.append(text[cursor : match.start])
        pieces.append(token)
        cursor = match.end
        masks.append(
            PreserveMask(
                token=token,
                sentence_id=sentence_id,
                original_text=match.original_text,
            )
        )

    pieces.append(text[cursor:])
    return "".join(pieces), tuple(masks), next_mask_number


def _find_preserve_matches(
    text: str,
    preserved_values: tuple[str, ...],
) -> tuple[_PreserveMatch, ...]:
    matches: list[_PreserveMatch] = []
    occupied_ranges: list[tuple[int, int]] = []
    values_by_priority = sorted(
        enumerate(preserved_values),
        key=lambda item: (-len(item[1]), item[0]),
    )

    for _value_index, value in values_by_priority:
        pattern = re.compile(re.escape(value), re.IGNORECASE)
        for match in pattern.finditer(text):
            start, end = match.span()
            if not _has_literal_boundaries(text, start, end, value):
                continue
            if _overlaps_any(start, end, occupied_ranges):
                continue
            occupied_ranges.append((start, end))
            matches.append(
                _PreserveMatch(
                    start=start,
                    end=end,
                    original_text=text[start:end],
                )
            )

    return tuple(sorted(matches, key=lambda match: (match.start, match.end)))


def _has_literal_boundaries(text: str, start: int, end: int, value: str) -> bool:
    if not value:
        return False
    if (
        _is_word_character(value[0])
        and start > 0
        and _is_word_character(text[start - 1])
    ):
        return False
    if (
        _is_word_character(value[-1])
        and end < len(text)
        and _is_word_character(text[end])
    ):
        return False
    return True


def _is_word_character(character: str) -> bool:
    return character.isalnum() or character == "_"


def _overlaps_any(
    start: int,
    end: int,
    occupied_ranges: Iterable[tuple[int, int]],
) -> bool:
    return any(
        start < occupied_end and occupied_start < end
        for occupied_start, occupied_end in occupied_ranges
    )


def _unexpected_or_modified_token_issues(
    *,
    sentence_id: str,
    text: str,
    expected_tokens: set[str],
) -> tuple[MaskValidationIssue, ...]:
    issues: list[MaskValidationIssue] = []
    token_counts = Counter(MASK_TOKEN_RE.findall(text))
    for token, count in token_counts.items():
        if token not in expected_tokens:
            issues.append(
                MaskValidationIssue(
                    sentence_id=sentence_id,
                    token=token,
                    issue="unexpected",
                    count=count,
                )
            )

    for match in _KEEP_LIKE_RE.finditer(text):
        candidate = match.group(0)
        if MASK_TOKEN_RE.fullmatch(candidate):
            continue
        issues.append(
            MaskValidationIssue(
                sentence_id=sentence_id,
                token=candidate,
                issue="modified",
            )
        )

    return tuple(issues)


def _dedupe_issues(
    issues: Iterable[MaskValidationIssue],
) -> tuple[MaskValidationIssue, ...]:
    deduped: list[MaskValidationIssue] = []
    seen: set[tuple[str, str, str, int]] = set()
    for issue in issues:
        key = (issue.sentence_id, issue.token, issue.issue, issue.count)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(issue)
    return tuple(deduped)


def _summarize_validation_issues(issues: tuple[MaskValidationIssue, ...]) -> str:
    if not issues:
        return "Preserve mask validation failed."
    counts = Counter(issue.issue for issue in issues)
    summary = ", ".join(
        f"{issue}: {count}" for issue, count in sorted(counts.items())
    )
    return f"Preserve mask validation failed ({summary})."
