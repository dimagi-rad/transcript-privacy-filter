from __future__ import annotations

import pytest

from opf_app.masking import (
    MaskValidationError,
    format_mask_token,
    mask_sentence_units,
    require_valid_mask_tokens,
    restore_sentence_texts,
    validate_mask_tokens,
)
from opf_app.models import ParsedItem
from opf_app.masking import parse_preserved_values
from opf_app.sentences import reconstruct_body, segment_parsed_item


def test_parse_and_mask_preserved_values_case_insensitively() -> None:
    segmented, _next_sentence_number = segment_parsed_item(
        _item("[00:00] S1: Suzy met suzy.")
    )

    masked_batch = mask_sentence_units(
        segmented.sentences,
        parse_preserved_values(" suzy, , SUZY "),
    )

    assert masked_batch.masked_text_by_id() == {
        "s_000001": "__KEEP_000001__ met __KEEP_000002__."
    }
    assert "Suzy" not in masked_batch.sentences[0].api_text
    assert "suzy" not in masked_batch.sentences[0].api_text
    assert tuple(masked_batch.masks_by_token) == (
        "__KEEP_000001__",
        "__KEEP_000002__",
    )


def test_longest_match_first_handles_overlapping_values() -> None:
    segmented, _next_sentence_number = segment_parsed_item(
        _item("[00:00] S1: New York and York are separate.")
    )

    masked_batch = mask_sentence_units(
        segmented.sentences,
        ("York", "New York"),
    )

    assert masked_batch.masked_text_by_id()["s_000001"] == (
        "__KEEP_000001__ and __KEEP_000002__ are separate."
    )


def test_punctuation_boundaries_match_without_matching_inside_words() -> None:
    segmented, _next_sentence_number = segment_parsed_item(
        _item("[00:00] S1: Suzy, Suzyville, and suzy.")
    )

    masked_batch = mask_sentence_units(segmented.sentences, ("suzy",))

    assert masked_batch.masked_text_by_id()["s_000001"] == (
        "__KEEP_000001__, Suzyville, and __KEEP_000002__."
    )


def test_no_match_keeps_sentence_text_and_empty_mask_map() -> None:
    segmented, _next_sentence_number = segment_parsed_item(
        _item("[00:00] S1: Nothing to preserve.")
    )

    masked_batch = mask_sentence_units(segmented.sentences, " , Missing ")

    assert masked_batch.masked_text_by_id() == {
        "s_000001": "Nothing to preserve."
    }
    assert masked_batch.masks_by_token == {}
    assert validate_mask_tokens(
        {"s_000001": "Nothing to preserve."},
        masked_batch,
    ) == ()


def test_restore_recovers_original_preserved_values_after_redaction() -> None:
    segmented, _next_sentence_number = segment_parsed_item(
        _item("[00:00] S1: Alice met Suzy.")
    )
    masked_batch = mask_sentence_units(segmented.sentences, ("suzy",))

    restored = restore_sentence_texts(
        {"s_000001": "<PRIVATE_PERSON> met __KEEP_000001__."},
        masked_batch,
    )

    assert restored == {"s_000001": "<PRIVATE_PERSON> met Suzy."}


def test_damaged_missing_duplicated_and_unexpected_masks_are_detected() -> None:
    segmented, _next_sentence_number = segment_parsed_item(
        _item("[00:00] S1: Alice met Suzy and Bob.")
    )
    masked_batch = mask_sentence_units(segmented.sentences, ("Suzy", "Bob"))

    issues = validate_mask_tokens(
        {
            "s_000001": (
                "<PRIVATE_PERSON> met __KEEP_000002__ and "
                "__KEEP_000002__ plus __KEEP_000001_."
            ),
            "s_extra": "Unexpected __KEEP_999999__.",
        },
        masked_batch,
    )

    assert _issue_pairs(issues) == {
        ("s_000001", "__KEEP_000001__", "missing"),
        ("s_000001", "__KEEP_000002__", "duplicated"),
        ("s_000001", "__KEEP_000001_", "modified"),
        ("s_extra", "__KEEP_999999__", "unexpected"),
    }


def test_validation_error_message_does_not_expose_preserved_values() -> None:
    segmented, _next_sentence_number = segment_parsed_item(
        _item("[00:00] S1: Alice met Suzy.")
    )
    masked_batch = mask_sentence_units(segmented.sentences, ("Suzy",))

    with pytest.raises(MaskValidationError) as exc_info:
        require_valid_mask_tokens(
            {"s_000001": "<PRIVATE_PERSON> met someone."},
            masked_batch,
        )

    assert exc_info.value.issues[0].issue == "missing"
    assert "Suzy" not in str(exc_info.value)
    assert "__KEEP_000001__" not in str(exc_info.value)


def test_masking_integrates_with_sentence_reconstruction() -> None:
    body = "[2026-05-11T10:01:00+00:00] User: Alice visited Suzy. Bob stayed."
    segmented, _next_sentence_number = segment_parsed_item(_item(body))
    masked_batch = mask_sentence_units(segmented.sentences, ("Suzy",))

    restored_redactions = restore_sentence_texts(
        {
            "s_000001": "<PRIVATE_PERSON> visited __KEEP_000001__.",
            "s_000002": "<PRIVATE_PERSON> stayed.",
        },
        masked_batch,
    )

    assert reconstruct_body(segmented, restored_redactions) == (
        "[2026-05-11T10:01:00+00:00] User: "
        "<PRIVATE_PERSON> visited Suzy. <PRIVATE_PERSON> stayed."
    )


def test_mask_batch_repr_does_not_expose_preserved_values() -> None:
    segmented, _next_sentence_number = segment_parsed_item(
        _item("[00:00] S1: Alice met Suzy.")
    )
    masked_batch = mask_sentence_units(segmented.sentences, ("Suzy",))

    assert "Suzy" not in repr(masked_batch)
    assert "Alice met" not in repr(masked_batch)


def test_invalid_mask_numbers_are_rejected() -> None:
    with pytest.raises(ValueError, match="Mask numbers must start at 1"):
        format_mask_token(0)


def _issue_pairs(issues: object) -> set[tuple[str, str, str]]:
    return {
        (issue.sentence_id, issue.token, issue.issue)
        for issue in issues
    }


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
