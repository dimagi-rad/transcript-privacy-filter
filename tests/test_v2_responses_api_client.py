from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from opf_app.masking import mask_sentence_units
from opf_app.models import ParsedItem
from opf_app.prompts import REDACTION_PROMPT_VERSION, build_redaction_prompt
from opf_app.responses_client import (
    REDACTION_RESPONSE_FORMAT_NAME,
    REDACTION_RESPONSE_SCHEMA,
    ResponsesClientError,
    ResponsesRedactionClient,
    build_redaction_request,
    build_text_format,
)
from opf_app.sentences import segment_parsed_item


def test_structured_output_schema_matches_sentence_contract() -> None:
    text_format = build_text_format()

    assert text_format == {
        "type": "json_schema",
        "name": REDACTION_RESPONSE_FORMAT_NAME,
        "strict": True,
        "schema": REDACTION_RESPONSE_SCHEMA,
    }
    assert REDACTION_RESPONSE_SCHEMA["additionalProperties"] is False
    sentence_schema = REDACTION_RESPONSE_SCHEMA["properties"]["sentences"]["items"]
    assert sentence_schema["required"] == ["id", "redacted_text"]
    assert sentence_schema["additionalProperties"] is False


def test_prompt_asset_contains_reviewed_privacy_instructions() -> None:
    prompt = build_redaction_prompt()

    assert "Return only the structured output requested by the schema." in prompt
    assert "Do not add explanations." in prompt
    assert "Do not change protected mask tokens" in prompt
    assert "__KEEP_000001__" in prompt


def test_successful_response_request_is_stateless_and_privacy_constrained() -> None:
    masked_batch = _masked_batch()
    response = _response(
        output_text=json.dumps(
            {
                "sentences": [
                    {
                        "id": "s_000001",
                        "redacted_text": "<PRIVATE_PERSON> met __KEEP_000001__.",
                    }
                ]
            }
        ),
        usage=SimpleNamespace(
            input_tokens=42,
            output_tokens=12,
            total_tokens=54,
            input_tokens_details=SimpleNamespace(cached_tokens=3),
            output_tokens_details=SimpleNamespace(reasoning_tokens=4),
        ),
    )
    fake_client = FakeOpenAIClient(response=response)
    client = ResponsesRedactionClient(api_key="sk-test-secret", client=fake_client)

    result = client.redact_sentence_batch(
        model_id="gpt-test-redactor",
        masked_batch=masked_batch,
    )

    request = fake_client.responses.calls[0]
    assert request["model"] == "gpt-test-redactor"
    assert request["store"] is False
    assert request["tools"] == []
    assert "previous_response_id" not in request
    assert "conversation" not in request
    assert request["text"]["format"]["type"] == "json_schema"
    assert request["text"]["format"]["strict"] is True
    request_text = request["input"][0]["content"][0]["text"]
    assert "__KEEP_000001__" in request_text
    assert REDACTION_PROMPT_VERSION in request_text
    assert "Suzy" not in str(request)
    assert "sk-test-secret" not in str(request)

    assert result.text_by_id() == {
        "s_000001": "<PRIVATE_PERSON> met __KEEP_000001__."
    }
    assert result.model == "gpt-test-redactor"
    assert result.response_id == "resp_test"
    assert result.usage.summary() == {
        "input_tokens": 42,
        "output_tokens": 12,
        "total_tokens": 54,
        "cached_input_tokens": 3,
        "reasoning_output_tokens": 4,
    }


def test_build_redaction_request_submits_only_masked_sentence_payload() -> None:
    request = build_redaction_request(
        model_id="gpt-test-redactor",
        masked_batch=_masked_batch(),
    )

    input_text = request["input"][0]["content"][0]["text"]
    payload = json.loads(input_text)

    assert payload == {
        "prompt_version": REDACTION_PROMPT_VERSION,
        "sentences": [
            {
                "id": "s_000001",
                "text": "<PRIVATE_PERSON> met __KEEP_000001__.",
            }
        ],
    }
    assert "source" not in input_text
    assert "participant" not in input_text
    assert "output_filename" not in input_text
    assert "Suzy" not in input_text


def test_refusal_response_is_normalized_without_response_text() -> None:
    fake_client = FakeOpenAIClient(
        response=_response(
            output=[
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "refusal": "unsafe refusal with sensitive detail",
                        }
                    ],
                }
            ],
            output_text=None,
        )
    )
    client = ResponsesRedactionClient(api_key="sk-test-secret", client=fake_client)

    with pytest.raises(ResponsesClientError) as exc_info:
        client.redact_sentence_batch(
            model_id="gpt-test-redactor",
            masked_batch=_masked_batch(),
        )

    assert exc_info.value.category == "refusal"
    assert "sensitive detail" not in str(exc_info.value)


def test_incomplete_response_is_normalized() -> None:
    fake_client = FakeOpenAIClient(
        response=_response(
            status="incomplete",
            incomplete_details={"reason": "max_output_tokens"},
        )
    )
    client = ResponsesRedactionClient(api_key="sk-test-secret", client=fake_client)

    with pytest.raises(ResponsesClientError) as exc_info:
        client.redact_sentence_batch(
            model_id="gpt-test-redactor",
            masked_batch=_masked_batch(),
        )

    assert exc_info.value.category == "incomplete"


def test_api_exception_is_normalized_without_secret_or_source_text() -> None:
    fake_client = FakeOpenAIClient(
        error=RuntimeError("sk-test-secret failed while sending Suzy")
    )
    client = ResponsesRedactionClient(api_key="sk-test-secret", client=fake_client)

    with pytest.raises(ResponsesClientError) as exc_info:
        client.redact_sentence_batch(
            model_id="gpt-test-redactor",
            masked_batch=_masked_batch(),
        )

    assert exc_info.value.category == "api_error"
    assert "sk-test-secret" not in str(exc_info.value)
    assert "Suzy" not in str(exc_info.value)


def test_status_api_errors_are_bucketed_for_retry_logic() -> None:
    error = FakeStatusError(status_code=429)
    fake_client = FakeOpenAIClient(error=error)
    client = ResponsesRedactionClient(api_key="sk-test-secret", client=fake_client)

    with pytest.raises(ResponsesClientError) as exc_info:
        client.redact_sentence_batch(
            model_id="gpt-test-redactor",
            masked_batch=_masked_batch(),
        )

    assert exc_info.value.category == "rate_limit"


def test_malformed_structured_output_is_normalized() -> None:
    fake_client = FakeOpenAIClient(
        response=_response(output_text=json.dumps({"sentences": [{"id": "s_000001"}]}))
    )
    client = ResponsesRedactionClient(api_key="sk-test-secret", client=fake_client)

    with pytest.raises(ResponsesClientError) as exc_info:
        client.redact_sentence_batch(
            model_id="gpt-test-redactor",
            masked_batch=_masked_batch(),
        )

    assert exc_info.value.category == "malformed_response"


class FakeResponses:
    def __init__(self, *, response: object | None = None, error: object = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            if isinstance(self.error, BaseException):
                raise self.error
            raise RuntimeError("fake sdk wrapper")
        return self.response


class FakeOpenAIClient:
    def __init__(self, *, response: object | None = None, error: object = None) -> None:
        self.responses = FakeResponses(response=response, error=error)


class FakeStatusError(Exception):
    def __init__(self, *, status_code: int) -> None:
        self.status_code = status_code
        super().__init__("fake status error with no source text")


def _response(
    *,
    output_text: str | None = '{"sentences":[]}',
    status: str = "completed",
    incomplete_details: object | None = None,
    usage: object | None = None,
    output: object | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id="resp_test",
        model="gpt-test-redactor",
        status=status,
        incomplete_details=incomplete_details,
        output_text=output_text,
        usage=usage,
        output=output or [],
    )


def _masked_batch() -> object:
    segmented, _next_sentence_number = segment_parsed_item(
        ParsedItem(
            item_name="item",
            source_name="source.txt",
            source_type="document",
            chat_date="2026-05-11",
            user_identifier="participant",
            body_text="[00:00] S1: Alice met Suzy.",
            output_filename="redacted-transcript-2026-05-11-participant.docx",
        )
    )
    masked_batch = mask_sentence_units(segmented.sentences, ("Suzy",))
    redacted_masked_text = "<PRIVATE_PERSON> met __KEEP_000001__."
    return type(masked_batch)(
        sentences=(
            type(masked_batch.sentences[0])(
                sentence=masked_batch.sentences[0].sentence,
                masked_text=redacted_masked_text,
                mask_tokens=masked_batch.sentences[0].mask_tokens,
            ),
        ),
        masks_by_token=masked_batch.masks_by_token,
    )
