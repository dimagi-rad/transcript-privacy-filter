from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pytest

from opf_app.masking import mask_sentence_units
from opf_app.models import ParsedItem
from opf_app.responses_client import (
    RedactedSentence,
    RedactionApiResult,
    ResponsesClientError,
    ResponsesUsage,
)
from opf_app.sentences import segment_parsed_item
from opf_app.v2_redaction import (
    V2ChunkValidationError,
    V2RedactionService,
    chunk_masked_sentence_batch,
    validate_and_restore_chunk_result,
)


def test_v2_service_validates_restores_and_reconstructs_item() -> None:
    item = _item("[00:00] S1: Alice met Suzy. Bob waited.")
    client = FakeV2Client(
        [
            lambda batch: _api_result(
                {
                    "s_000001": "<PRIVATE_PERSON> met __KEEP_000001__.",
                    "s_000002": "<PRIVATE_PERSON> waited.",
                },
                usage=ResponsesUsage(input_tokens=8, output_tokens=6, total_tokens=14),
            )
        ]
    )

    service_result = V2RedactionService(client).redact_item(
        item,
        model_id="gpt-test-redactor",
        sentence_chunk_size=2,
        preserved_values=("Suzy",),
    )

    assert service_result.success is True
    assert service_result.redaction_result.redacted_text == (
        "[00:00] S1: <PRIVATE_PERSON> met Suzy. <PRIVATE_PERSON> waited."
    )
    assert service_result.metadata.total_sentence_count == 2
    assert service_result.metadata.chunk_count == 1
    assert service_result.metadata.successful_chunk_count == 1
    assert service_result.metadata.retry_count == 0
    assert service_result.metadata.usage.summary()["total_tokens"] == 14
    assert "Suzy" not in repr(service_result.metadata)


def test_wrong_sentence_count_is_rejected_without_source_text() -> None:
    masked_batch = _masked_batch("[00:00] S1: Alice arrived.")

    with pytest.raises(V2ChunkValidationError) as exc_info:
        validate_and_restore_chunk_result(_api_result({}), masked_batch)

    assert "wrong_sentence_count" in exc_info.value.categories
    assert "missing_sentence_id" in exc_info.value.categories
    assert "Alice" not in str(exc_info.value)


def test_missing_sentence_id_is_rejected() -> None:
    masked_batch = _masked_batch("[00:00] S1: Alice arrived. Bob waited.")

    with pytest.raises(V2ChunkValidationError) as exc_info:
        validate_and_restore_chunk_result(
            _api_result({"s_000001": "<PRIVATE_PERSON> arrived."}),
            masked_batch,
        )

    assert "missing_sentence_id" in exc_info.value.categories


def test_duplicate_sentence_id_is_rejected() -> None:
    masked_batch = _masked_batch("[00:00] S1: Alice arrived.")
    api_result = RedactionApiResult(
        sentences=(
            RedactedSentence(id="s_000001", redacted_text="<PRIVATE_PERSON>."),
            RedactedSentence(id="s_000001", redacted_text="<PRIVATE_PERSON>."),
        ),
        usage=ResponsesUsage(),
    )

    with pytest.raises(V2ChunkValidationError) as exc_info:
        validate_and_restore_chunk_result(api_result, masked_batch)

    assert "duplicate_sentence_id" in exc_info.value.categories


def test_extra_sentence_id_is_rejected() -> None:
    masked_batch = _masked_batch("[00:00] S1: Alice arrived.")

    with pytest.raises(V2ChunkValidationError) as exc_info:
        validate_and_restore_chunk_result(
            _api_result(
                {
                    "s_000001": "<PRIVATE_PERSON> arrived.",
                    "s_999999": "Extra.",
                }
            ),
            masked_batch,
        )

    assert "extra_sentence_id" in exc_info.value.categories


def test_preserve_mask_damage_is_rejected() -> None:
    masked_batch = _masked_batch("[00:00] S1: Alice met Suzy.", ("Suzy",))

    with pytest.raises(V2ChunkValidationError) as exc_info:
        validate_and_restore_chunk_result(
            _api_result({"s_000001": "<PRIVATE_PERSON> met someone."}),
            masked_batch,
        )

    assert exc_info.value.categories == ("preserve_mask_damage",)
    assert "Suzy" not in str(exc_info.value)


def test_invalid_chunk_is_retried_and_can_succeed() -> None:
    item = _item("[00:00] S1: Alice arrived.")
    client = FakeV2Client(
        [
            _api_result({}),
            _api_result({"s_000001": "<PRIVATE_PERSON> arrived."}),
        ]
    )

    service_result = V2RedactionService(client).redact_item(
        item,
        model_id="gpt-test-redactor",
        max_attempts=2,
    )

    assert service_result.success is True
    assert client.call_count == 2
    assert service_result.metadata.retry_count == 1


def test_retry_exhaustion_fails_only_the_current_item() -> None:
    item = _item("[00:00] S1: Alice arrived.")
    client = FakeV2Client([_api_result({}), _api_result({})])

    service_result = V2RedactionService(client).redact_item(
        item,
        model_id="gpt-test-redactor",
        max_attempts=2,
    )

    assert service_result.success is False
    assert service_result.redaction_result.redacted_text == item.body_text
    assert service_result.metadata.failed_chunk_count == 1
    assert service_result.metadata.retry_count == 1
    assert "wrong_sentence_count" in service_result.metadata.error_categories
    assert all("Alice" not in error for error in service_result.redaction_result.errors)


def test_transient_api_error_retries_with_deterministic_backoff() -> None:
    item = _item("[00:00] S1: Alice arrived.")
    sleeps: list[float] = []
    client = FakeV2Client(
        [
            ResponsesClientError("rate_limit"),
            ResponsesClientError("transient"),
            _api_result({"s_000001": "<PRIVATE_PERSON> arrived."}),
        ]
    )

    service_result = V2RedactionService(client, sleep=sleeps.append).redact_item(
        item,
        model_id="gpt-test-redactor",
        max_attempts=3,
        backoff_seconds=0.5,
        backoff_multiplier=2.0,
    )

    assert service_result.success is True
    assert sleeps == [0.5, 1.0]
    assert service_result.metadata.retry_count == 2


def test_authentication_error_does_not_retry() -> None:
    item = _item("[00:00] S1: Alice arrived.")
    client = FakeV2Client([ResponsesClientError("authentication")])

    service_result = V2RedactionService(client).redact_item(
        item,
        model_id="gpt-test-redactor",
        max_attempts=3,
    )

    assert service_result.success is False
    assert client.call_count == 1
    assert service_result.metadata.error_categories == ("authentication",)


def test_empty_item_succeeds_without_api_call() -> None:
    item = _item("[00:00] S1:   ")
    client = FakeV2Client([])

    service_result = V2RedactionService(client).redact_item(
        item,
        model_id="gpt-test-redactor",
    )

    assert service_result.success is True
    assert service_result.redaction_result.redacted_text == item.body_text
    assert service_result.metadata.chunk_count == 0
    assert client.call_count == 0


def test_chunking_preserves_only_chunk_local_masks() -> None:
    masked_batch = _masked_batch(
        "[00:00] S1: Alice met Suzy. Bob met Ana.",
        ("Suzy", "Ana"),
    )

    chunks = chunk_masked_sentence_batch(masked_batch, 1)

    assert len(chunks) == 2
    assert tuple(chunks[0].masks_by_token) == ("__KEEP_000001__",)
    assert tuple(chunks[1].masks_by_token) == ("__KEEP_000002__",)


@dataclass
class FakeV2Client:
    responses: list[RedactionApiResult | ResponsesClientError | Callable[[object], RedactionApiResult]]

    def __post_init__(self) -> None:
        self.calls: list[object] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def redact_sentence_batch(
        self,
        *,
        model_id: str,
        masked_batch: object,
    ) -> RedactionApiResult:
        self.calls.append((model_id, masked_batch))
        response = self.responses.pop(0)
        if isinstance(response, ResponsesClientError):
            raise response
        if callable(response):
            return response(masked_batch)
        return response


def _api_result(
    redacted_text_by_id: dict[str, str],
    *,
    usage: ResponsesUsage | None = None,
) -> RedactionApiResult:
    return RedactionApiResult(
        sentences=tuple(
            RedactedSentence(id=sentence_id, redacted_text=redacted_text)
            for sentence_id, redacted_text in redacted_text_by_id.items()
        ),
        usage=usage or ResponsesUsage(),
        model="gpt-test-redactor",
        response_id="resp_test",
    )


def _masked_batch(
    body_text: str,
    preserved_values: tuple[str, ...] = (),
) -> object:
    segmented, _next_sentence_number = segment_parsed_item(_item(body_text))
    return mask_sentence_units(segmented.sentences, preserved_values)


def _item(body_text: str) -> ParsedItem:
    return ParsedItem(
        item_name="item",
        source_name="source.txt",
        source_type="document",
        chat_date="2026-05-11",
        user_identifier="participant",
        body_text=body_text,
        output_filename="redacted-transcript-2026-05-11-participant.docx",
    )
