from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
import time
from typing import Literal, Protocol

from .masking import (
    MaskValidationError,
    MaskedSentenceBatch,
    mask_sentence_units,
    restore_sentence_texts,
)
from .models import ParsedItem, RedactionResult
from .responses_client import (
    RedactionApiResult,
    ResponsesClientError,
    ResponsesUsage,
)
from .sentences import (
    reconstruct_body,
    segment_parsed_item,
)


DEFAULT_V2_MAX_ATTEMPTS = 3
DEFAULT_V2_SENTENCE_CHUNK_SIZE = 3

V2RedactionErrorCategory = Literal[
    "api_error",
    "authentication",
    "permission",
    "rate_limit",
    "transient",
    "incomplete",
    "refusal",
    "malformed_response",
    "wrong_sentence_count",
    "missing_sentence_id",
    "duplicate_sentence_id",
    "extra_sentence_id",
    "preserve_mask_damage",
    "parsed_item_not_ready",
]
V2ChunkLifecycleStatus = Literal["running", "retrying", "complete", "failed"]
V2ChunkProgressCallback = Callable[["V2ChunkLifecycleEvent"], None]

_RETRYABLE_CATEGORIES: frozenset[str] = frozenset(
    {
        "api_error",
        "rate_limit",
        "transient",
        "incomplete",
        "refusal",
        "malformed_response",
        "wrong_sentence_count",
        "missing_sentence_id",
        "duplicate_sentence_id",
        "extra_sentence_id",
        "preserve_mask_damage",
    }
)


class V2RedactionClient(Protocol):
    def redact_sentence_batch(
        self,
        *,
        model_id: str,
        masked_batch: MaskedSentenceBatch,
    ) -> RedactionApiResult:
        """Redact one masked sentence chunk with the Responses API."""


@dataclass(frozen=True)
class V2ChunkValidationIssue:
    """Privacy-safe structured-output validation issue."""

    category: V2RedactionErrorCategory
    sentence_id: str | None = None
    count: int | None = None


class V2ChunkValidationError(ValueError):
    """Raised when a structured chunk result fails local validation."""

    def __init__(self, issues: Iterable[V2ChunkValidationIssue]) -> None:
        self.issues = tuple(issues)
        self.categories = tuple(
            dict.fromkeys(issue.category for issue in self.issues)
        )
        super().__init__(f"Invalid v2 redaction chunk: {','.join(self.categories)}.")


@dataclass(frozen=True)
class V2UsageTotals:
    """Aggregated token usage without request or response payloads."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_output_tokens: int = 0

    def add(self, usage: ResponsesUsage) -> V2UsageTotals:
        return V2UsageTotals(
            input_tokens=self.input_tokens + (usage.input_tokens or 0),
            output_tokens=self.output_tokens + (usage.output_tokens or 0),
            total_tokens=self.total_tokens + (usage.total_tokens or 0),
            cached_input_tokens=self.cached_input_tokens
            + (usage.cached_input_tokens or 0),
            reasoning_output_tokens=self.reasoning_output_tokens
            + (usage.reasoning_output_tokens or 0),
        )

    def summary(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "reasoning_output_tokens": self.reasoning_output_tokens,
        }


@dataclass(frozen=True)
class V2ChunkMetadata:
    """Privacy-safe metadata for one sentence chunk attempt sequence."""

    chunk_index: int
    sentence_count: int
    attempts: int
    success: bool
    error_categories: tuple[V2RedactionErrorCategory, ...] = field(
        default_factory=tuple
    )
    usage: V2UsageTotals = field(default_factory=V2UsageTotals)

    @property
    def retry_count(self) -> int:
        return max(0, self.attempts - 1)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "error_categories", tuple(self.error_categories)
        )


@dataclass(frozen=True)
class V2ChunkLifecycleEvent:
    """Privacy-safe lifecycle update emitted while one chunk is processed."""

    status: V2ChunkLifecycleStatus
    chunk_index: int
    chunk_count: int
    attempt_number: int
    max_attempts: int
    error_categories: tuple[V2RedactionErrorCategory, ...] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "error_categories", tuple(self.error_categories)
        )


@dataclass(frozen=True)
class V2RedactionMetadata:
    """Privacy-safe service summary for one parsed item."""

    model_id: str
    sentence_chunk_size: int
    max_attempts: int
    total_sentence_count: int
    chunk_count: int
    successful_chunk_count: int
    failed_chunk_count: int
    retry_count: int
    error_categories: tuple[V2RedactionErrorCategory, ...] = field(
        default_factory=tuple
    )
    usage: V2UsageTotals = field(default_factory=V2UsageTotals)
    chunks: tuple[V2ChunkMetadata, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "error_categories", tuple(self.error_categories)
        )
        object.__setattr__(self, "chunks", tuple(self.chunks))


@dataclass(frozen=True)
class V2RedactionServiceResult:
    """V2 service result plus metadata consumable by later batch stories."""

    redaction_result: RedactionResult
    metadata: V2RedactionMetadata

    @property
    def success(self) -> bool:
        return self.redaction_result.success


class V2RedactionService:
    """Validate, retry, restore, and reconstruct v2 sentence redactions."""

    def __init__(
        self,
        client: V2RedactionClient,
        *,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._client = client
        self._sleep = sleep or time.sleep

    def redact_item(
        self,
        item: ParsedItem,
        *,
        model_id: str,
        sentence_chunk_size: int = DEFAULT_V2_SENTENCE_CHUNK_SIZE,
        max_attempts: int = DEFAULT_V2_MAX_ATTEMPTS,
        preserved_values: Iterable[str] = (),
        backoff_seconds: float = 0.0,
        backoff_multiplier: float = 2.0,
        progress_callback: V2ChunkProgressCallback | None = None,
    ) -> V2RedactionServiceResult:
        """Redact one parsed item with sentence-chunk validation and retries."""
        chunk_size = _require_positive_int(
            sentence_chunk_size,
            name="sentence_chunk_size",
        )
        attempt_limit = _require_positive_int(max_attempts, name="max_attempts")

        if item.parse_status == "error" or item.errors:
            return self._failed_result(
                item,
                model_id=model_id,
                sentence_chunk_size=chunk_size,
                max_attempts=attempt_limit,
                total_sentence_count=0,
                chunks=(),
                error_categories=("parsed_item_not_ready",),
                errors=item.errors or ("Parsed item is not redaction-ready.",),
            )

        segmented_item, _next_sentence_number = segment_parsed_item(item)
        masked_batch = mask_sentence_units(
            segmented_item.sentences,
            preserved_values,
        )
        chunks = chunk_masked_sentence_batch(masked_batch, chunk_size)

        if not chunks:
            metadata = _build_metadata(
                model_id=model_id,
                sentence_chunk_size=chunk_size,
                max_attempts=attempt_limit,
                total_sentence_count=0,
                chunks=(),
            )
            return V2RedactionServiceResult(
                redaction_result=RedactionResult(
                    item=item,
                    output_filename=item.output_filename,
                    redacted_text=item.body_text,
                    success=True,
                    warnings=item.warnings,
                    errors=item.errors,
                ),
                metadata=metadata,
            )

        restored_text_by_id: dict[str, str] = {}
        chunk_metadata: list[V2ChunkMetadata] = []
        failed_categories: tuple[V2RedactionErrorCategory, ...] = ()

        for chunk_index, chunk in enumerate(chunks):
            chunk_result = self._redact_chunk_with_retries(
                model_id=model_id,
                masked_batch=chunk,
                chunk_index=chunk_index,
                max_attempts=attempt_limit,
                backoff_seconds=backoff_seconds,
                backoff_multiplier=backoff_multiplier,
                chunk_count=len(chunks),
                progress_callback=progress_callback,
            )
            chunk_metadata.append(chunk_result.metadata)
            if chunk_result.success:
                restored_text_by_id.update(chunk_result.restored_text_by_id)
                continue
            failed_categories = chunk_result.metadata.error_categories
            break

        if failed_categories:
            return self._failed_result(
                item,
                model_id=model_id,
                sentence_chunk_size=chunk_size,
                max_attempts=attempt_limit,
                total_sentence_count=len(segmented_item.sentences),
                chunks=tuple(chunk_metadata),
                error_categories=failed_categories,
                errors=tuple(
                    f"v2_redaction_failed:{category}"
                    for category in failed_categories
                ),
            )

        redacted_text = reconstruct_body(segmented_item, restored_text_by_id)
        return V2RedactionServiceResult(
            redaction_result=RedactionResult(
                item=item,
                output_filename=item.output_filename,
                redacted_text=redacted_text,
                success=True,
                warnings=item.warnings,
                errors=item.errors,
            ),
            metadata=_build_metadata(
                model_id=model_id,
                sentence_chunk_size=chunk_size,
                max_attempts=attempt_limit,
                total_sentence_count=len(segmented_item.sentences),
                chunks=tuple(chunk_metadata),
            ),
        )

    def _redact_chunk_with_retries(
        self,
        *,
        model_id: str,
        masked_batch: MaskedSentenceBatch,
        chunk_index: int,
        max_attempts: int,
        backoff_seconds: float,
        backoff_multiplier: float,
        chunk_count: int,
        progress_callback: V2ChunkProgressCallback | None,
    ) -> _V2ChunkRunResult:
        last_categories: tuple[V2RedactionErrorCategory, ...] = ()
        usage = V2UsageTotals()

        for attempt_number in range(1, max_attempts + 1):
            if attempt_number == 1:
                _emit_chunk_lifecycle_event(
                    progress_callback,
                    status="running",
                    chunk_index=chunk_index,
                    chunk_count=chunk_count,
                    attempt_number=attempt_number,
                    max_attempts=max_attempts,
                )
            try:
                api_result = self._client.redact_sentence_batch(
                    model_id=model_id,
                    masked_batch=masked_batch,
                )
                restored_text_by_id = validate_and_restore_chunk_result(
                    api_result,
                    masked_batch,
                )
                usage = usage.add(api_result.usage)
                _emit_chunk_lifecycle_event(
                    progress_callback,
                    status="complete",
                    chunk_index=chunk_index,
                    chunk_count=chunk_count,
                    attempt_number=attempt_number,
                    max_attempts=max_attempts,
                )
                return _V2ChunkRunResult(
                    success=True,
                    restored_text_by_id=restored_text_by_id,
                    metadata=V2ChunkMetadata(
                        chunk_index=chunk_index,
                        sentence_count=len(masked_batch.sentences),
                        attempts=attempt_number,
                        success=True,
                        usage=usage,
                    ),
                )
            except ResponsesClientError as exc:
                last_categories = (exc.category,)
                retryable = _is_retryable(exc.category)
            except V2ChunkValidationError as exc:
                last_categories = exc.categories
                retryable = any(_is_retryable(category) for category in exc.categories)
            except MaskValidationError:
                last_categories = ("preserve_mask_damage",)
                retryable = True

            if not retryable or attempt_number >= max_attempts:
                _emit_chunk_lifecycle_event(
                    progress_callback,
                    status="failed",
                    chunk_index=chunk_index,
                    chunk_count=chunk_count,
                    attempt_number=attempt_number,
                    max_attempts=max_attempts,
                    error_categories=last_categories,
                )
                return _V2ChunkRunResult(
                    success=False,
                    restored_text_by_id={},
                    metadata=V2ChunkMetadata(
                        chunk_index=chunk_index,
                        sentence_count=len(masked_batch.sentences),
                        attempts=attempt_number,
                        success=False,
                        error_categories=last_categories,
                        usage=usage,
                    ),
                )

            _emit_chunk_lifecycle_event(
                progress_callback,
                status="retrying",
                chunk_index=chunk_index,
                chunk_count=chunk_count,
                attempt_number=attempt_number + 1,
                max_attempts=max_attempts,
                error_categories=last_categories,
            )
            _sleep_before_retry(
                self._sleep,
                attempt_number=attempt_number,
                backoff_seconds=backoff_seconds,
                backoff_multiplier=backoff_multiplier,
            )

        return _V2ChunkRunResult(
            success=False,
            restored_text_by_id={},
            metadata=V2ChunkMetadata(
                chunk_index=chunk_index,
                sentence_count=len(masked_batch.sentences),
                attempts=max_attempts,
                success=False,
                error_categories=last_categories,
                usage=usage,
            ),
        )

    def _failed_result(
        self,
        item: ParsedItem,
        *,
        model_id: str,
        sentence_chunk_size: int,
        max_attempts: int,
        total_sentence_count: int,
        chunks: tuple[V2ChunkMetadata, ...],
        error_categories: tuple[V2RedactionErrorCategory, ...],
        errors: tuple[str, ...],
    ) -> V2RedactionServiceResult:
        return V2RedactionServiceResult(
            redaction_result=RedactionResult(
                item=item,
                output_filename=item.output_filename,
                redacted_text=item.body_text,
                success=False,
                warnings=item.warnings,
                errors=errors,
            ),
            metadata=_build_metadata(
                model_id=model_id,
                sentence_chunk_size=sentence_chunk_size,
                max_attempts=max_attempts,
                total_sentence_count=total_sentence_count,
                chunks=chunks,
                error_categories=error_categories,
            ),
        )


def _emit_chunk_lifecycle_event(
    progress_callback: V2ChunkProgressCallback | None,
    *,
    status: V2ChunkLifecycleStatus,
    chunk_index: int,
    chunk_count: int,
    attempt_number: int,
    max_attempts: int,
    error_categories: tuple[V2RedactionErrorCategory, ...] = (),
) -> None:
    if progress_callback is None:
        return
    progress_callback(
        V2ChunkLifecycleEvent(
            status=status,
            chunk_index=chunk_index,
            chunk_count=chunk_count,
            attempt_number=attempt_number,
            max_attempts=max_attempts,
            error_categories=error_categories,
        )
    )


@dataclass(frozen=True)
class _V2ChunkRunResult:
    success: bool
    restored_text_by_id: Mapping[str, str] = field(default_factory=dict)
    metadata: V2ChunkMetadata | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "restored_text_by_id", dict(self.restored_text_by_id))
        if self.metadata is None:
            raise ValueError("metadata is required.")


def validate_and_restore_chunk_result(
    api_result: RedactionApiResult,
    masked_batch: MaskedSentenceBatch,
) -> dict[str, str]:
    """Validate exact sentence coverage and restore preserve masks."""
    issues = validate_chunk_result(api_result, masked_batch)
    if issues:
        raise V2ChunkValidationError(issues)
    return restore_sentence_texts(api_result.text_by_id(), masked_batch)


def validate_chunk_result(
    api_result: RedactionApiResult,
    masked_batch: MaskedSentenceBatch,
) -> tuple[V2ChunkValidationIssue, ...]:
    """Return privacy-safe validation issues for a structured API chunk."""
    expected_ids = [sentence.sentence_id for sentence in masked_batch.sentences]
    expected_id_set = set(expected_ids)
    result_ids = [sentence.id for sentence in api_result.sentences]
    result_id_counts = Counter(result_ids)
    result_id_set = set(result_ids)
    issues: list[V2ChunkValidationIssue] = []

    if len(result_ids) != len(expected_ids):
        issues.append(
            V2ChunkValidationIssue(
                category="wrong_sentence_count",
                count=len(result_ids),
            )
        )

    for sentence_id, count in sorted(result_id_counts.items()):
        if count > 1:
            issues.append(
                V2ChunkValidationIssue(
                    category="duplicate_sentence_id",
                    sentence_id=sentence_id,
                    count=count,
                )
            )

    for sentence_id in sorted(expected_id_set - result_id_set):
        issues.append(
            V2ChunkValidationIssue(
                category="missing_sentence_id",
                sentence_id=sentence_id,
            )
        )

    for sentence_id in sorted(result_id_set - expected_id_set):
        issues.append(
            V2ChunkValidationIssue(
                category="extra_sentence_id",
                sentence_id=sentence_id,
            )
        )

    if not issues:
        try:
            restore_sentence_texts(api_result.text_by_id(), masked_batch)
        except MaskValidationError:
            issues.append(V2ChunkValidationIssue(category="preserve_mask_damage"))

    return tuple(issues)


def chunk_masked_sentence_batch(
    masked_batch: MaskedSentenceBatch,
    sentence_chunk_size: int,
) -> tuple[MaskedSentenceBatch, ...]:
    """Split a masked sentence set into API chunks without changing masks."""
    chunk_size = _require_positive_int(
        sentence_chunk_size,
        name="sentence_chunk_size",
    )
    chunks: list[MaskedSentenceBatch] = []
    sentences = tuple(masked_batch.sentences)
    for chunk_start in range(0, len(sentences), chunk_size):
        chunk_sentences = sentences[chunk_start : chunk_start + chunk_size]
        chunk_tokens = {
            token
            for masked_sentence in chunk_sentences
            for token in masked_sentence.mask_tokens
        }
        chunks.append(
            MaskedSentenceBatch(
                sentences=chunk_sentences,
                masks_by_token={
                    token: mask
                    for token, mask in masked_batch.masks_by_token.items()
                    if token in chunk_tokens
                },
            )
        )
    return tuple(chunks)


def _build_metadata(
    *,
    model_id: str,
    sentence_chunk_size: int,
    max_attempts: int,
    total_sentence_count: int,
    chunks: tuple[V2ChunkMetadata, ...],
    error_categories: tuple[V2RedactionErrorCategory, ...] = (),
) -> V2RedactionMetadata:
    chunk_usage = V2UsageTotals()
    categories: list[V2RedactionErrorCategory] = list(error_categories)
    for chunk in chunks:
        chunk_usage = V2UsageTotals(
            input_tokens=chunk_usage.input_tokens + chunk.usage.input_tokens,
            output_tokens=chunk_usage.output_tokens + chunk.usage.output_tokens,
            total_tokens=chunk_usage.total_tokens + chunk.usage.total_tokens,
            cached_input_tokens=chunk_usage.cached_input_tokens
            + chunk.usage.cached_input_tokens,
            reasoning_output_tokens=chunk_usage.reasoning_output_tokens
            + chunk.usage.reasoning_output_tokens,
        )
        categories.extend(chunk.error_categories)

    return V2RedactionMetadata(
        model_id=model_id,
        sentence_chunk_size=sentence_chunk_size,
        max_attempts=max_attempts,
        total_sentence_count=total_sentence_count,
        chunk_count=len(chunks),
        successful_chunk_count=sum(1 for chunk in chunks if chunk.success),
        failed_chunk_count=sum(1 for chunk in chunks if not chunk.success),
        retry_count=sum(chunk.retry_count for chunk in chunks),
        error_categories=tuple(dict.fromkeys(categories)),
        usage=chunk_usage,
        chunks=chunks,
    )


def _is_retryable(category: str) -> bool:
    return category in _RETRYABLE_CATEGORIES


def _sleep_before_retry(
    sleep: Callable[[float], None],
    *,
    attempt_number: int,
    backoff_seconds: float,
    backoff_multiplier: float,
) -> None:
    if backoff_seconds <= 0:
        return
    multiplier = max(1.0, backoff_multiplier)
    sleep(backoff_seconds * (multiplier ** (attempt_number - 1)))


def _require_positive_int(value: int, *, name: str) -> int:
    coerced = int(value)
    if coerced < 1:
        raise ValueError(f"{name} must be at least 1.")
    return coerced
