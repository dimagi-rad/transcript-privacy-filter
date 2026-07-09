from __future__ import annotations

from opf_app.models import ParsedItem
from opf_app.sentences import (
    format_sentence_id,
    parse_transcript_line,
    reconstruct_body,
    segment_parsed_item,
    segment_parsed_items,
    split_sentences,
)


def test_csv_style_user_and_chatbot_lines_exclude_structural_prefixes() -> None:
    item = _item(
        "csv",
        (
            "[2026-05-11T10:01:00+00:00] User: Hello there. My email is a.b@example.com.\n"
            "[2026-05-11T10:02:00+00:00] Chatbot: I can help! Send details."
        ),
    )

    segmented, next_sentence_number = segment_parsed_item(item)

    assert [sentence.sentence_id for sentence in segmented.sentences] == [
        "s_000001",
        "s_000002",
        "s_000003",
        "s_000004",
    ]
    assert [sentence.role for sentence in segmented.sentences] == [
        "User",
        "User",
        "Chatbot",
        "Chatbot",
    ]
    assert [sentence.api_text for sentence in segmented.sentences] == [
        "Hello there.",
        "My email is a.b@example.com.",
        "I can help!",
        "Send details.",
    ]
    assert all("2026-05-11" not in sentence.api_text for sentence in segmented.sentences)
    assert all("User:" not in sentence.api_text for sentence in segmented.sentences)
    assert next_sentence_number == 5


def test_document_style_timestamp_and_speaker_prefix_stay_local() -> None:
    item = _item("doc", "[00:00] S1: Alice arrived. Bob stayed.")

    segmented, _next_sentence_number = segment_parsed_item(item)

    assert segmented.lines[0].role == "S1"
    assert segmented.lines[0].render() == item.body_text
    assert [sentence.original_text for sentence in segmented.sentences] == [
        "Alice arrived.",
        "Bob stayed.",
    ]
    assert all("[00:00]" not in sentence.api_text for sentence in segmented.sentences)


def test_reconstruct_body_round_trips_unchanged_text_and_line_endings() -> None:
    body = "[00:00] S1: First sentence.  Second sentence!\r\n\nPlain text?"
    segmented, _next_sentence_number = segment_parsed_item(_item("doc", body))

    assert reconstruct_body(segmented) == body


def test_reconstruct_body_uses_redacted_sentence_text_without_moving_prefixes() -> None:
    body = "[2026-05-11T10:01:00+00:00] User: Alice arrived.  Bob left!"
    segmented, _next_sentence_number = segment_parsed_item(_item("csv", body))

    redacted = reconstruct_body(
        segmented,
        {
            "s_000001": "<PRIVATE_PERSON> arrived.",
            "s_000002": "<PRIVATE_PERSON> left!",
        },
    )

    assert redacted == (
        "[2026-05-11T10:01:00+00:00] User: "
        "<PRIVATE_PERSON> arrived.  <PRIVATE_PERSON> left!"
    )


def test_split_sentences_avoids_commas_urls_emails_decimals_and_abbreviations() -> None:
    text = (
        "Dr. A. B. Smith emailed a.b@example.com, then paid 3.50. "
        "Visit https://example.com/path. Done!"
    )

    assert split_sentences(text) == (
        "Dr. A. B. Smith emailed a.b@example.com, then paid 3.50.",
        "Visit https://example.com/path.",
        "Done!",
    )


def test_empty_lines_and_whitespace_utterances_do_not_produce_api_work() -> None:
    item = _item("empty", "[00:00] S1:   \n\n   ")

    segmented, next_sentence_number = segment_parsed_item(item)

    assert segmented.sentences == ()
    assert reconstruct_body(segmented) == item.body_text
    assert next_sentence_number == 1


def test_multiline_csv_continuation_line_inherits_previous_role() -> None:
    item = _item(
        "multiline",
        "[2026-05-11T10:00:00+00:00] Chatbot: line one.\nline two?",
    )

    segmented, _next_sentence_number = segment_parsed_item(item)

    assert [sentence.role for sentence in segmented.sentences] == [
        "Chatbot",
        "Chatbot",
    ]
    assert [sentence.original_text for sentence in segmented.sentences] == [
        "line one.",
        "line two?",
    ]
    assert reconstruct_body(segmented) == item.body_text


def test_sentence_ids_are_stable_and_unique_across_items() -> None:
    segmented_items = segment_parsed_items(
        [
            _item("one", "[00:00] S1: One. Two."),
            _item("two", "[00:01] S2: Three."),
        ]
    )

    assert [
        sentence.sentence_id
        for segmented in segmented_items
        for sentence in segmented.sentences
    ] == ["s_000001", "s_000002", "s_000003"]


def test_sentence_dataclass_reprs_do_not_expose_full_text() -> None:
    secret_text = "Sensitive patient detail."
    segmented, _next_sentence_number = segment_parsed_item(
        _item("secret", f"[00:00] S1: {secret_text}")
    )

    assert secret_text not in repr(segmented.lines[0])
    assert secret_text not in repr(segmented.sentences[0])
    assert secret_text not in repr(segmented)


def test_parse_transcript_line_handles_plain_text_without_prefix() -> None:
    line = parse_transcript_line("Plain document sentence.", line_index=2)

    assert line.role is None
    assert line.render() == "Plain document sentence."
    assert split_sentences(line.utterance_text) == ("Plain document sentence.",)


def test_invalid_sentence_numbers_are_rejected() -> None:
    try:
        format_sentence_id(0)
    except ValueError as exc:
        assert str(exc) == "Sentence numbers must start at 1."
    else:
        raise AssertionError("Expected invalid sentence number to raise ValueError.")


def _item(identifier: str, body_text: str) -> ParsedItem:
    return ParsedItem(
        item_name=f"item-{identifier}",
        source_name=f"source-{identifier}.txt",
        source_type="document",
        chat_date="2026-05-11",
        user_identifier=identifier,
        body_text=body_text,
        output_filename=f"redacted-transcript-2026-05-11-{identifier}.docx",
    )
