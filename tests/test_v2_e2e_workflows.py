from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import re
import zipfile

from docx import Document
import pytest

from opf_app.config import DEFAULT_MODEL_ID
from opf_app.documents import parse_document
from opf_app.masking import MaskedSentenceBatch
from opf_app.models import ParsedItem
from opf_app.ocs_csv import parse_ocs_csv_text
from opf_app.responses_client import (
    RedactedSentence,
    RedactionApiResult,
    ResponsesRedactionClient,
    ResponsesUsage,
)
from opf_app.v2_batch import run_v2_redaction_batch
from opf_app.v2_redaction import V2RedactionService
from tests.fixtures import build_ocs_csv, write_sample_docx


RUN_OPENAI_API_SMOKE_ENV = "RUN_OPENAI_API_SMOKE"
OPENAI_API_SMOKE_MODEL_ENV = "OPENAI_API_SMOKE_MODEL"


@dataclass
class DeterministicFakeResponsesClient:
    submitted_texts: list[str] = field(default_factory=list)
    model_ids: list[str] = field(default_factory=list)

    def redact_sentence_batch(
        self,
        *,
        model_id: str,
        masked_batch: MaskedSentenceBatch,
    ) -> RedactionApiResult:
        self.model_ids.append(model_id)
        sentences: list[RedactedSentence] = []
        for masked_sentence in masked_batch.sentences:
            self.submitted_texts.append(masked_sentence.masked_text)
            redacted_text = re.sub(
                r"\bAlice\b",
                "<PRIVATE_PERSON>",
                masked_sentence.masked_text,
            )
            redacted_text = redacted_text.replace(
                "alice@example.test",
                "<PRIVATE_EMAIL>",
            )
            redacted_text = redacted_text.replace("Hello.", "Welcome.")
            sentences.append(
                RedactedSentence(
                    id=masked_sentence.sentence_id,
                    redacted_text=redacted_text,
                )
            )
        return RedactionApiResult(
            sentences=tuple(sentences),
            usage=ResponsesUsage(
                input_tokens=len(sentences) * 4,
                output_tokens=len(sentences) * 3,
                total_tokens=len(sentences) * 7,
            ),
            model=model_id,
            response_id="fake-response",
        )


def test_fake_client_v2_csv_workflow_redacts_both_roles_and_restores_values(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    source_text = build_ocs_csv(
        [
            {
                "Message Date": "2026-05-11 09:00:00+00:00",
                "Message Type": "human",
                "Message Content": "Alice met Suzy.",
                "Session ID": "session-1",
                "Participant Public ID": "participant-1",
            },
            {
                "Message Date": "2026-05-11 09:01:00+00:00",
                "Message Type": "ai",
                "Message Content": "Email alice@example.test.",
                "Session ID": "session-1",
                "Participant Public ID": "participant-1",
            },
        ]
    )
    items = parse_ocs_csv_text(source_text, source_name="sample.csv")
    client = DeterministicFakeResponsesClient()

    result = run_v2_redaction_batch(
        items,
        model_id="gpt-test-redactor",
        redaction_service=V2RedactionService(client),
        output_dir=tmp_path,
        sentence_chunk_size=2,
        api_concurrency=1,
        preserved_values=("Suzy",),
    )

    assert result.complete_count == 1
    assert result.failed_count == 0
    assert result.summary.total_sentence_count == 2
    assert result.summary.successful_chunk_count == 1
    assert client.model_ids == ["gpt-test-redactor"]
    submitted_blob = " ".join(client.submitted_texts)
    assert "Suzy" not in submitted_blob
    assert "__KEEP_000001__" in submitted_blob
    assert "participant-1" not in submitted_blob
    assert "[2026-05-11" not in submitted_blob

    output_path = result.items[0].output.path
    assert output_path is not None
    output_text = "\n".join(_paragraphs(output_path))
    assert "] USER: <PRIVATE_PERSON> met Suzy." in output_text
    assert "] CHATBOT: Email <PRIVATE_EMAIL>." in output_text
    assert "Alice" not in output_text
    with zipfile.ZipFile(result.zip_path) as archive:
        assert archive.namelist() == [result.items[0].output.filename]

    assert "Alice" not in caplog.text
    assert "Suzy" not in caplog.text
    assert "alice@example.test" not in caplog.text


def test_fake_client_v2_document_workflow_preserves_transcript_structure(
    tmp_path: Path,
) -> None:
    source_path = write_sample_docx(tmp_path / "P695 Interview 2026_01_07.docx")
    item = parse_document(source_path)
    client = DeterministicFakeResponsesClient()

    result = run_v2_redaction_batch(
        (item,),
        model_id="gpt-test-redactor",
        redaction_service=V2RedactionService(client),
        output_dir=tmp_path / "outputs",
        sentence_chunk_size=1,
        api_concurrency=2,
    )

    assert result.complete_count == 1
    assert result.summary.total_sentence_count == 2
    assert client.submitted_texts == ["Hello.", "Thanks."]
    assert all("[00:" not in text for text in client.submitted_texts)
    output_path = result.items[0].output.path
    assert output_path is not None
    paragraphs = _paragraphs(output_path)
    assert "[00:00] S1: Welcome." in paragraphs
    assert "[00:13] S2: Thanks." in paragraphs
    assert "Transcription details" not in paragraphs


@pytest.mark.skipif(
    os.environ.get(RUN_OPENAI_API_SMOKE_ENV) != "1",
    reason=(
        "Real Responses API smoke is opt-in; set "
        f"{RUN_OPENAI_API_SMOKE_ENV}=1 to enable it."
    ),
)
def test_optional_real_openai_api_smoke(tmp_path: Path) -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        pytest.skip("OPENAI_API_KEY is required for the opt-in real API smoke test.")

    model_id = os.environ.get(OPENAI_API_SMOKE_MODEL_ENV, DEFAULT_MODEL_ID).strip()
    item = ParsedItem(
        item_name="api-smoke",
        source_name="api-smoke.txt",
        source_type="document",
        chat_date="2026-01-01",
        user_identifier="api-smoke",
        body_text="[00:00] S1: This is a synthetic API smoke sentence.",
        output_filename="redacted-transcript-2026-01-01-api-smoke.docx",
    )

    result = run_v2_redaction_batch(
        (item,),
        model_id=model_id,
        redaction_service=V2RedactionService(
            ResponsesRedactionClient(api_key=api_key)
        ),
        output_dir=tmp_path,
        sentence_chunk_size=1,
        api_concurrency=1,
        retry_limit=1,
    )

    assert result.complete_count == 1
    assert result.failed_count == 0
    assert result.summary.successful_chunk_count == 1
    assert result.zip_path.is_file()


def _paragraphs(path: Path) -> list[str]:
    return [paragraph.text for paragraph in Document(path).paragraphs]
