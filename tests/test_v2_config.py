from __future__ import annotations

import pytest

from opf_app.config import (
    API_CONCURRENCY_RANGE,
    API_KEY_ENV_VAR,
    DEFAULT_MODEL_ID,
    RESPONSES_API_DEFAULTS,
    RETRY_ATTEMPTS_DEFAULT,
    SENTENCE_CHUNK_RANGE,
    V2_MODEL_CATALOG,
    default_model_option,
    normalize_custom_model_id,
    resolve_api_key,
    resolve_model_id,
)


def test_v2_model_catalog_defaults_are_single_source_of_truth() -> None:
    assert [option.model_id for option in V2_MODEL_CATALOG] == [
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
    ]
    assert [option.display_label for option in V2_MODEL_CATALOG] == [
        "GPT-5.5",
        "GPT-5.4",
        "GPT-5.4 Mini",
    ]
    assert DEFAULT_MODEL_ID == "gpt-5.5"
    assert default_model_option().model_id == DEFAULT_MODEL_ID


def test_v2_runtime_ranges_match_prd_defaults() -> None:
    assert (SENTENCE_CHUNK_RANGE.minimum, SENTENCE_CHUNK_RANGE.default) == (1, 3)
    assert SENTENCE_CHUNK_RANGE.maximum == 5
    assert SENTENCE_CHUNK_RANGE.clamp(0) == 1
    assert SENTENCE_CHUNK_RANGE.clamp(99) == 5

    assert (API_CONCURRENCY_RANGE.minimum, API_CONCURRENCY_RANGE.default) == (1, 4)
    assert API_CONCURRENCY_RANGE.maximum == 8
    assert API_CONCURRENCY_RANGE.clamp(None) == 4
    assert API_CONCURRENCY_RANGE.clamp(99) == 8

    assert RETRY_ATTEMPTS_DEFAULT == 3


def test_responses_api_defaults_are_stateless_and_zdr_safe() -> None:
    assert RESPONSES_API_DEFAULTS.store is False
    assert RESPONSES_API_DEFAULTS.previous_response_id is None
    assert RESPONSES_API_DEFAULTS.tools == ()
    assert RESPONSES_API_DEFAULTS.request_options() == {"store": False}


def test_custom_model_id_trimming_validation_and_resolution() -> None:
    assert normalize_custom_model_id("  gpt-5.5-2026-07-09  ") == "gpt-5.5-2026-07-09"
    assert normalize_custom_model_id("") is None
    assert normalize_custom_model_id("   ") is None
    assert resolve_model_id(custom_model_id="  custom-model:preview  ") == (
        "custom-model:preview"
    )
    assert resolve_model_id(selected_model_id="gpt-5.4-mini") == "gpt-5.4-mini"
    assert resolve_model_id() == DEFAULT_MODEL_ID

    with pytest.raises(ValueError, match="whitespace"):
        normalize_custom_model_id("custom model")
    with pytest.raises(ValueError, match="configured v2 model catalog"):
        resolve_model_id(selected_model_id="unknown-model")


def test_api_key_resolution_prefers_environment_key_without_storing_output() -> None:
    secret = "sk-test-env-secret"
    session_secret = "sk-test-session-secret"

    credential = resolve_api_key(
        env={API_KEY_ENV_VAR: f" {secret} "},
        session_api_key=session_secret,
    )

    assert credential.source == "environment"
    assert credential.require_value() == secret
    assert secret not in repr(credential)
    assert session_secret not in repr(credential)
    assert secret not in str(credential.summary())
    assert session_secret not in str(credential.summary())
    assert credential.summary() == {
        "api_key_configured": True,
        "api_key_source": "environment",
    }


def test_session_api_key_is_memory_only_fallback_and_repr_safe() -> None:
    secret = "sk-test-session-secret"

    credential = resolve_api_key(env={}, session_api_key=f" {secret} ")

    assert credential.source == "session"
    assert credential.require_value() == secret
    assert secret not in repr(credential)
    assert secret not in str(credential.summary())
    assert credential.summary() == {
        "api_key_configured": True,
        "api_key_source": "session",
    }


def test_missing_api_key_errors_do_not_include_secret_values() -> None:
    credential = resolve_api_key(env={}, session_api_key=" ")

    assert credential.source == "missing"
    assert credential.summary() == {
        "api_key_configured": False,
        "api_key_source": "missing",
    }
    with pytest.raises(RuntimeError) as exc_info:
        credential.require_value()
    assert str(exc_info.value) == "OpenAI API key is not configured."
