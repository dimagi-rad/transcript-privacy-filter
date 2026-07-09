from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import os
from typing import Literal


API_KEY_ENV_VAR = "OPENAI_API_KEY"
SESSION_API_KEY_STATE_KEY = "openai_api_key"

SENTENCE_CHUNK_MIN = 1
SENTENCE_CHUNK_DEFAULT = 3
SENTENCE_CHUNK_MAX = 5

API_CONCURRENCY_MIN = 1
API_CONCURRENCY_DEFAULT = 4
API_CONCURRENCY_MAX = 8

RETRY_ATTEMPTS_DEFAULT = 3

ApiKeySource = Literal["environment", "session", "missing"]


@dataclass(frozen=True)
class ModelOption:
    """Configured model choice for the v2 Responses API workflow."""

    display_label: str
    model_id: str


V2_MODEL_CATALOG: tuple[ModelOption, ...] = (
    ModelOption(display_label="GPT-5.5", model_id="gpt-5.5"),
    ModelOption(display_label="GPT-5.4", model_id="gpt-5.4"),
    ModelOption(display_label="GPT-5.4 Mini", model_id="gpt-5.4-mini"),
)
DEFAULT_MODEL_ID = V2_MODEL_CATALOG[0].model_id


@dataclass(frozen=True)
class NumericRange:
    minimum: int
    default: int
    maximum: int

    def clamp(self, value: int | None) -> int:
        if value is None:
            return self.default
        return max(self.minimum, min(self.maximum, int(value)))


SENTENCE_CHUNK_RANGE = NumericRange(
    minimum=SENTENCE_CHUNK_MIN,
    default=SENTENCE_CHUNK_DEFAULT,
    maximum=SENTENCE_CHUNK_MAX,
)
API_CONCURRENCY_RANGE = NumericRange(
    minimum=API_CONCURRENCY_MIN,
    default=API_CONCURRENCY_DEFAULT,
    maximum=API_CONCURRENCY_MAX,
)


@dataclass(frozen=True)
class ResponsesApiDefaults:
    """Privacy-safe defaults for later stateless Responses API calls."""

    store: bool = False
    previous_response_id: str | None = None
    tools: tuple[Mapping[str, object], ...] = ()

    def request_options(self) -> dict[str, object]:
        options: dict[str, object] = {"store": self.store}
        if self.previous_response_id is not None:
            options["previous_response_id"] = self.previous_response_id
        if self.tools:
            options["tools"] = list(self.tools)
        return options


RESPONSES_API_DEFAULTS = ResponsesApiDefaults()


@dataclass(frozen=True)
class ApiKeyCredential:
    """OpenAI API key plus its source, with secret-safe helper output."""

    source: ApiKeySource
    value: str = field(default="", repr=False, compare=False)

    @property
    def is_configured(self) -> bool:
        return bool(self.value)

    def require_value(self) -> str:
        if not self.value:
            raise RuntimeError("OpenAI API key is not configured.")
        return self.value

    def summary(self) -> dict[str, object]:
        return {
            "api_key_configured": self.is_configured,
            "api_key_source": self.source,
        }


def resolve_api_key(
    *,
    env: Mapping[str, str] | None = None,
    session_api_key: str | None = None,
) -> ApiKeyCredential:
    """Resolve the OpenAI API key without persisting or exposing the value."""
    environment = os.environ if env is None else env
    env_value = _clean_secret(environment.get(API_KEY_ENV_VAR))
    if env_value:
        return ApiKeyCredential(source="environment", value=env_value)

    session_value = _clean_secret(session_api_key)
    if session_value:
        return ApiKeyCredential(source="session", value=session_value)

    return ApiKeyCredential(source="missing")


def configured_model_ids(
    catalog: Sequence[ModelOption] = V2_MODEL_CATALOG,
) -> tuple[str, ...]:
    return tuple(option.model_id for option in catalog)


def default_model_option(
    catalog: Sequence[ModelOption] = V2_MODEL_CATALOG,
) -> ModelOption:
    for option in catalog:
        if option.model_id == DEFAULT_MODEL_ID:
            return option
    raise ValueError("Default v2 model ID is not present in the configured catalog.")


def normalize_custom_model_id(value: str | None) -> str | None:
    """Trim and validate an optional custom Responses API model ID."""
    if value is None:
        return None

    model_id = value.strip()
    if not model_id:
        return None
    if any(character.isspace() for character in model_id):
        raise ValueError("Custom model ID cannot contain whitespace.")
    if any(ord(character) < 33 or ord(character) == 127 for character in model_id):
        raise ValueError("Custom model ID contains an unsupported control character.")

    return model_id


def resolve_model_id(
    *,
    selected_model_id: str | None = None,
    custom_model_id: str | None = None,
    catalog: Sequence[ModelOption] = V2_MODEL_CATALOG,
) -> str:
    """Resolve either a custom model ID or a configured catalog model ID."""
    normalized_custom = normalize_custom_model_id(custom_model_id)
    if normalized_custom is not None:
        return normalized_custom

    if selected_model_id is None:
        return DEFAULT_MODEL_ID

    normalized_selected = selected_model_id.strip()
    if normalized_selected in configured_model_ids(catalog):
        return normalized_selected

    raise ValueError("Selected model ID is not in the configured v2 model catalog.")


def _clean_secret(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip()
