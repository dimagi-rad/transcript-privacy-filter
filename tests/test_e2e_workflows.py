from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import zipfile

from docx import Document
import pytest

from tests.fixtures import write_sample_docx, write_sample_ocs_csv
from opf_app.batch import run_redaction_batch
from opf_app.documents import parse_document
from opf_app.models import ParsedItem
from opf_app.ocs_csv import parse_ocs_csv
from opf_app.redaction import RedactionService, SpanForReplacement


@dataclass(frozen=True)
class FakeOpfResult:
    text: str
    detected_spans: tuple[SpanForReplacement, ...]
    warning: str | None = None


class PassthroughRedactor:
    def redact(self, text: str) -> FakeOpfResult:
        return FakeOpfResult(text=text, detected_spans=())


class CategoryRedactor:
    def redact(self, text: str) -> FakeOpfResult:
        return FakeOpfResult(
            text=text,
            detected_spans=(
                SpanForReplacement("private_person", 0, 5, "<PRIVATE_PERSON>"),
                SpanForReplacement("secret", 6, 18, "<SECRET>"),
            ),
        )


class FailingOnTextRedactor:
    def __init__(self, fail_text: str) -> None:
        self.fail_text = fail_text

    def redact(self, text: str) -> FakeOpfResult:
        if text == self.fail_text:
            raise RuntimeError("planned failure")
        return FakeOpfResult(text=text, detected_spans=())


class TrackingRedactor:
    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def redact(self, text: str) -> FakeOpfResult:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            return FakeOpfResult(text=text, detected_spans=())
        finally:
            with self.lock:
                self.active -= 1


def test_e2e_sample_csv_workflow_produces_six_docx_outputs(tmp_path: Path) -> None:
    items = parse_ocs_csv(write_sample_ocs_csv(tmp_path / "sample.csv"))

    result = run_redaction_batch(
        items,
        selected_labels=(),
        redaction_service=RedactionService(PassthroughRedactor()),
        output_dir=tmp_path,
        concurrency=2,
    )

    assert result.complete_count == 6
    assert result.failed_count == 0
    with zipfile.ZipFile(result.zip_path) as archive:
        names = archive.namelist()
    assert len(names) == 6
    assert all(name.endswith(".docx") for name in names)
    first_docx = result.items[0].output.path
    assert first_docx is not None
    paragraphs = _paragraphs(first_docx)
    assert any(paragraph.startswith("[2026-05-11 ") for paragraph in paragraphs)
    assert any("] USER: " in paragraph or "] CHATBOT: " in paragraph for paragraph in paragraphs)


def test_e2e_sample_docx_workflow_preserves_normalized_body(tmp_path: Path) -> None:
    item = parse_document(write_sample_docx(tmp_path / "P695 Interview 2026_01_07.docx"))

    result = run_redaction_batch(
        (item,),
        selected_labels=(),
        redaction_service=RedactionService(PassthroughRedactor()),
        output_dir=tmp_path,
        concurrency=1,
    )

    assert result.complete_count == 1
    with zipfile.ZipFile(result.zip_path) as archive:
        assert archive.namelist() == ["redacted-transcript-2026-01-07-P695.docx"]
    output_path = result.items[0].output.path
    assert output_path is not None
    paragraphs = _paragraphs(output_path)
    assert any(paragraph.startswith("[00:00] S1:") for paragraph in paragraphs)
    assert "Transcription details" not in paragraphs


def test_e2e_category_filtering_through_pipeline(tmp_path: Path) -> None:
    item = _item("category", "Alice token-12345")

    result = run_redaction_batch(
        (item,),
        selected_labels=("private_person",),
        redaction_service=RedactionService(CategoryRedactor()),
        output_dir=tmp_path,
        concurrency=1,
    )

    assert result.complete_count == 1
    output_path = result.items[0].output.path
    assert output_path is not None
    paragraphs = _paragraphs(output_path)
    assert "<PRIVATE_PERSON> token-12345" in paragraphs
    assert "<SECRET>" not in paragraphs


def test_e2e_one_failed_item_keeps_successful_zip_outputs(tmp_path: Path) -> None:
    items = (_item("ok", "ok"), _item("bad", "fail"))

    result = run_redaction_batch(
        items,
        selected_labels=(),
        redaction_service=RedactionService(FailingOnTextRedactor("fail")),
        output_dir=tmp_path,
        concurrency=2,
    )

    assert result.complete_count == 1
    assert result.failed_count == 1
    assert any(
        event.message.startswith("Now redacting document")
        for event in result.progress_events
    )
    with zipfile.ZipFile(result.zip_path) as archive:
        assert archive.namelist() == ["redacted-transcript-2026-05-11-ok.docx"]


def test_e2e_concurrency_through_pipeline(tmp_path: Path) -> None:
    redactor = TrackingRedactor()

    result = run_redaction_batch(
        tuple(_item(str(index), f"body-{index}") for index in range(5)),
        selected_labels=(),
        redaction_service=RedactionService(redactor),
        output_dir=tmp_path,
        concurrency=3,
    )

    assert result.complete_count == 5
    assert 1 < redactor.max_active <= 3


def test_streamlit_startup_smoke() -> None:
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "streamlit_app.py",
            "--server.headless",
            "true",
            "--server.port",
            "8521",
            "--server.runOnSave",
            "false",
        ],
        cwd=Path.cwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        output = _read_until(process, "Local URL:", timeout=20)
        assert "Local URL:" in output
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def _opf_checkpoint_available() -> bool:
    checkpoint = Path(
        os.environ.get("OPF_CHECKPOINT", str(Path.home() / ".opf" / "privacy_filter"))
    ).expanduser()
    return (
        checkpoint.is_dir()
        and (checkpoint / "config.json").is_file()
        and any(checkpoint.glob("*.safetensors"))
    )


@pytest.mark.skipif(
    not _opf_checkpoint_available(),
    reason="OPF checkpoint not available locally; skipping real model smoke.",
)
def test_optional_real_opf_smoke_when_checkpoint_available() -> None:
    from opf import OPF

    checkpoint = os.environ.get("OPF_CHECKPOINT") or str(
        Path.home() / ".opf" / "privacy_filter"
    )
    result = OPF(model=checkpoint, device="cpu", output_mode="typed").redact("Alice")

    assert hasattr(result, "detected_spans")


def _item(identifier: str, body_text: str) -> ParsedItem:
    return ParsedItem(
        item_name=identifier,
        source_name=f"{identifier}.txt",
        source_type="document",
        chat_date="2026-05-11",
        user_identifier=identifier,
        body_text=body_text,
        output_filename=f"redacted-transcript-2026-05-11-{identifier}.docx",
    )


def _paragraphs(path: Path) -> list[str]:
    return [paragraph.text for paragraph in Document(path).paragraphs]


def _read_until(
    process: subprocess.Popen[str],
    expected: str,
    *,
    timeout: float,
) -> str:
    deadline = time.monotonic() + timeout
    output_parts: list[str] = []
    assert process.stdout is not None
    while time.monotonic() < deadline:
        line = process.stdout.readline()
        if line:
            output_parts.append(line)
            if expected in line:
                break
        elif process.poll() is not None:
            break
    return "".join(output_parts)
