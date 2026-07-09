from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from typing import Literal, Protocol

from .masking import MaskedSentenceBatch
from .prompts import REDACTION_PROMPT_VERSION, build_redaction_prompt


REDACTION_RESPONSE_FORMAT_NAME = "privacy_filter_redacted_sentences"

REDACTION_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "sentences": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "redacted_text": {"type": "string"},
                },
                "required": ["id", "redacted_text"],
            },
        }
    },
    "required": ["sentences"],
}

ResponseErrorCategory = Literal[
    "api_error",
    "authentication",
    "permission",
    "rate_limit",
    "transient",
    "incomplete",
    "refusal",
    "malformed_response",
]


class ResponsesCreateProtocol(Protocol):
    def create(self, **kwargs: object) -> object:
        """Create a stateless Responses API result."""


class OpenAIClientProtocol(Protocol):
    responses: ResponsesCreateProtocol


@dataclass(frozen=True)
class RedactedSentence:
    id: str
    redacted_text: str = field(repr=False)


@dataclass(frozen=True)
class ResponsesUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_output_tokens: int | None = None

    def summary(self) -> dict[str, int | None]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "reasoning_output_tokens": self.reasoning_output_tokens,
        }


@dataclass(frozen=True)
class RedactionApiResult:
    sentences: tuple[RedactedSentence, ...]
    usage: ResponsesUsage
    model: str | None = None
    response_id: str | None = None

    def text_by_id(self) -> dict[str, str]:
        return {sentence.id: sentence.redacted_text for sentence in self.sentences}


class ResponsesClientError(RuntimeError):
    """Privacy-safe API client error without request or response payloads."""

    def __init__(self, category: ResponseErrorCategory) -> None:
        self.category = category
        super().__init__(f"OpenAI Responses API redaction failed: {category}.")


class ResponsesRedactionClient:
    """Small stateless wrapper around the OpenAI Responses API."""

    def __init__(
        self,
        *,
        api_key: str,
        client: OpenAIClientProtocol | None = None,
    ) -> None:
        self._api_key = api_key.strip()
        self._client = client

    @property
    def client(self) -> OpenAIClientProtocol:
        if self._client is None:
            if not self._api_key:
                raise ResponsesClientError("authentication")
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def redact_sentence_batch(
        self,
        *,
        model_id: str,
        masked_batch: MaskedSentenceBatch,
    ) -> RedactionApiResult:
        request = build_redaction_request(
            model_id=model_id,
            masked_batch=masked_batch,
        )
        try:
            response = self.client.responses.create(**request)
        except Exception as exc:  # noqa: BLE001 - normalize SDK/network details
            raise ResponsesClientError(_api_error_category(exc)) from None
        return parse_redaction_response(response)


def build_text_format() -> dict[str, object]:
    return {
        "type": "json_schema",
        "name": REDACTION_RESPONSE_FORMAT_NAME,
        "strict": True,
        "schema": REDACTION_RESPONSE_SCHEMA,
    }


def build_redaction_request(
    *,
    model_id: str,
    masked_batch: MaskedSentenceBatch,
) -> dict[str, object]:
    """Build a privacy-constrained Responses API request payload."""
    return {
        "model": model_id,
        "instructions": build_redaction_prompt(),
        "input": _build_input(masked_batch),
        "text": {"format": build_text_format()},
        "store": False,
        "tools": [],
    }


def parse_redaction_response(response: object) -> RedactionApiResult:
    """Parse a structured Responses API result without logging response text."""
    if _response_refusal(response):
        raise ResponsesClientError("refusal")
    if _response_incomplete(response):
        raise ResponsesClientError("incomplete")

    output_text = _extract_output_text(response)
    try:
        payload = json.loads(output_text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ResponsesClientError("malformed_response") from exc

    sentences_payload = payload.get("sentences") if isinstance(payload, dict) else None
    if not isinstance(sentences_payload, list):
        raise ResponsesClientError("malformed_response")

    sentences: list[RedactedSentence] = []
    for sentence in sentences_payload:
        if not isinstance(sentence, dict):
            raise ResponsesClientError("malformed_response")
        sentence_id = sentence.get("id")
        redacted_text = sentence.get("redacted_text")
        if not isinstance(sentence_id, str) or not isinstance(redacted_text, str):
            raise ResponsesClientError("malformed_response")
        sentences.append(RedactedSentence(id=sentence_id, redacted_text=redacted_text))

    return RedactionApiResult(
        sentences=tuple(sentences),
        usage=_extract_usage(_get_value(response, "usage")),
        model=_coerce_optional_string(_get_value(response, "model")),
        response_id=_coerce_optional_string(_get_value(response, "id")),
    )


def _build_input(masked_batch: MaskedSentenceBatch) -> list[dict[str, object]]:
    payload = {
        "prompt_version": REDACTION_PROMPT_VERSION,
        "sentences": [
            {
                "id": masked_sentence.sentence_id,
                "text": masked_sentence.masked_text,
            }
            for masked_sentence in masked_batch.sentences
        ],
    }
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": json.dumps(payload, separators=(",", ":")),
                }
            ],
        }
    ]


def _response_incomplete(response: object) -> bool:
    status = _get_value(response, "status")
    incomplete_details = _get_value(response, "incomplete_details")
    return status not in (None, "completed") or bool(incomplete_details)


def _response_refusal(response: object) -> bool:
    if _get_value(response, "refusal"):
        return True
    for output_item in _iter_sequence(_get_value(response, "output")):
        if _get_value(output_item, "refusal"):
            return True
        for content_item in _iter_sequence(_get_value(output_item, "content")):
            if _get_value(content_item, "refusal"):
                return True
            if _get_value(content_item, "type") == "refusal":
                return True
    return False


def _extract_output_text(response: object) -> str:
    output_text = _get_value(response, "output_text")
    if isinstance(output_text, str) and output_text:
        return output_text

    for output_item in _iter_sequence(_get_value(response, "output")):
        for content_item in _iter_sequence(_get_value(output_item, "content")):
            if _get_value(content_item, "type") == "output_text":
                text = _get_value(content_item, "text")
                if isinstance(text, str):
                    return text

    raise ResponsesClientError("malformed_response")


def _extract_usage(usage: object) -> ResponsesUsage:
    if usage is None:
        return ResponsesUsage()

    input_details = _get_value(usage, "input_tokens_details")
    output_details = _get_value(usage, "output_tokens_details")
    return ResponsesUsage(
        input_tokens=_coerce_optional_int(_get_value(usage, "input_tokens")),
        output_tokens=_coerce_optional_int(_get_value(usage, "output_tokens")),
        total_tokens=_coerce_optional_int(_get_value(usage, "total_tokens")),
        cached_input_tokens=_coerce_optional_int(
            _get_value(input_details, "cached_tokens")
        ),
        reasoning_output_tokens=_coerce_optional_int(
            _get_value(output_details, "reasoning_tokens")
        ),
    )


def _api_error_category(exc: Exception) -> ResponseErrorCategory:
    status_code = _coerce_optional_int(_get_value(exc, "status_code"))
    if status_code in (401,):
        return "authentication"
    if status_code in (403,):
        return "permission"
    if status_code == 429:
        return "rate_limit"
    if status_code is not None and status_code >= 500:
        return "transient"
    return "api_error"


def _get_value(value: object, key: str) -> object:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _iter_sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, list | tuple):
        return tuple(value)
    return ()


def _coerce_optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
