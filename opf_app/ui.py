from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Sequence

import streamlit as st

from . import APP_TITLE
from .batch import BatchResult, clamp_concurrency, run_redaction_batch
from .config import (
    API_CONCURRENCY_RANGE,
    RETRY_ATTEMPTS_DEFAULT,
    SENTENCE_CHUNK_RANGE,
    SESSION_API_KEY_STATE_KEY,
    V2_MODEL_CATALOG,
    ApiKeyCredential,
    resolve_api_key,
    resolve_model_id,
)
from .documents import SUPPORTED_DOCUMENT_EXTENSIONS, parse_document
from .models import ParsedItem
from .ocs_csv import OcsCsvValidationError, parse_ocs_csv_text
from .redaction import (
    DEFAULT_RUNTIME_LABELS,
    RedactionService,
    RedactorLike,
    parse_preserved_values,
)
from .responses_client import ResponsesRedactionClient
from .v2_batch import (
    V2BatchResult,
    V2RedactionServiceLike,
    clamp_v2_api_concurrency,
    clamp_v2_sentence_chunk_size,
    run_v2_redaction_batch,
)
from .v2_redaction import V2RedactionService


CSV_MODE = "OCS CSV export"
DOCUMENT_MODE = "Document folder"
PARSED_ITEMS_STATE_KEY = "parsed_items"
PARSE_ERRORS_STATE_KEY = "parse_errors"
PARSE_WARNINGS_STATE_KEY = "parse_warnings"
ZIP_BYTES_STATE_KEY = "zip_bytes"
REDACTION_SUMMARY_STATE_KEY = "redaction_summary"
REDACTION_RESULT_ROWS_STATE_KEY = "redaction_result_rows"
REDACTION_FAILED_ROWS_STATE_KEY = "redaction_failed_rows"
REDACTION_PROGRESS_STATE_KEY = "redaction_progress"
DOCUMENT_FOLDER_PATH_STATE_KEY = "document_folder_path"
PRESERVED_VALUES_STATE_KEY = "preserved_values"
V2_MODEL_STATE_KEY = "v2_model_id"
V2_CUSTOM_MODEL_STATE_KEY = "v2_custom_model_id"
V2_SENTENCE_CHUNK_SIZE_STATE_KEY = "v2_sentence_chunk_size"
V2_API_CONCURRENCY_STATE_KEY = "v2_api_concurrency"
CATEGORY_STATE_PREFIX = "category_"
SUPPORTED_UPLOAD_TYPES = tuple(
    extension.removeprefix(".") for extension in sorted(SUPPORTED_DOCUMENT_EXTENSIONS)
)
_UNSAFE_UPLOAD_NAME_RE = re.compile(r"[^A-Za-z0-9._ -]+")


@dataclass(frozen=True)
class ParseOutcome:
    items: tuple[ParsedItem, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RedactionUiOutcome:
    total_count: int
    complete_count: int
    failed_count: int
    zip_bytes: bytes
    result_rows: tuple[dict[str, object], ...]
    failed_rows: tuple[dict[str, str], ...]
    progress_messages: tuple[str, ...]


@dataclass(frozen=True)
class V2RedactionUiOutcome:
    total_count: int
    complete_count: int
    failed_count: int
    zip_bytes: bytes
    summary: dict[str, object]
    result_rows: tuple[dict[str, object], ...]
    failed_rows: tuple[dict[str, str], ...]
    progress_messages: tuple[str, ...]


def prepare_review_table(
    items: tuple[ParsedItem, ...] | list[ParsedItem],
) -> list[dict[str, object]]:
    """Build non-sensitive parsed item rows for the review table."""
    rows: list[dict[str, object]] = []
    for item in items:
        count_value = item.message_count
        count_label = "messages"
        if count_value is None:
            count_value = item.line_count
            count_label = "lines"
        if count_value is None:
            count_value = item.character_count or 0
            count_label = "characters"

        rows.append(
            {
                "Item name": item.item_name,
                "Source type": item.source_type,
                "Chat date": item.chat_date,
                "User identifier": item.user_identifier,
                "Count": count_value,
                "Count type": count_label,
                "Parse status": item.parse_status,
                "Output filename preview": item.output_filename,
            }
        )
    return rows


def prepare_parse_error_table(
    items: tuple[ParsedItem, ...] | list[ParsedItem],
) -> list[dict[str, str]]:
    """Build per-item parse error rows without transcript bodies."""
    rows: list[dict[str, str]] = []
    for item in items:
        for error in item.errors:
            rows.append(
                {
                    "Item name": item.item_name,
                    "Output filename preview": item.output_filename,
                    "Error": error,
                }
            )
    return rows


def parse_csv_upload_bytes(data: bytes, *, source_name: str) -> ParseOutcome:
    """Parse an uploaded OCS CSV file into reviewable items."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return ParseOutcome(errors=("CSV file must be UTF-8 encoded.",))

    try:
        items = parse_ocs_csv_text(text, source_name=source_name)
    except OcsCsvValidationError as exc:
        return ParseOutcome(errors=(str(exc),))
    return ParseOutcome(items=items)


def parse_document_folder_path(folder_path: str) -> ParseOutcome:
    """Scan a local document folder and parse supported non-zipped files."""
    if not folder_path.strip():
        return ParseOutcome(errors=("Enter a document folder path.",))

    folder = Path(folder_path).expanduser()
    if not folder.is_dir():
        return ParseOutcome(errors=(f"Document folder does not exist: {folder}",))

    supported_files = [
        path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS
    ]
    ignored_count = sum(
        1
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS
    )
    warnings: list[str] = []
    if ignored_count:
        warnings.append(f"Ignored {ignored_count} unsupported file(s).")
    if not supported_files:
        warnings.append("No supported documents found in that folder.")

    return ParseOutcome(
        items=tuple(parse_document(path) for path in supported_files),
        warnings=tuple(warnings),
    )


def parse_uploaded_document_files(uploaded_files: Sequence[Any]) -> ParseOutcome:
    """Parse documents uploaded through Streamlit's directory/file uploader."""
    if not uploaded_files:
        return ParseOutcome(errors=("Choose a document folder or document files.",))

    warnings: list[str] = []
    ignored_count = 0
    items: list[ParsedItem] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_root = Path(tmpdir)
        for index, uploaded_file in enumerate(uploaded_files):
            relative_path = _safe_uploaded_relative_path(
                str(getattr(uploaded_file, "name", f"uploaded-{index}"))
            )
            if relative_path.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
                ignored_count += 1
                continue
            temp_path = _unique_upload_path(temp_root / relative_path)
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_bytes(_uploaded_file_bytes(uploaded_file))
            items.append(parse_document(temp_path))

    if ignored_count:
        warnings.append(f"Ignored {ignored_count} unsupported uploaded file(s).")
    if not items:
        warnings.append("No supported documents found in the upload.")

    return ParseOutcome(items=tuple(items), warnings=tuple(warnings))


def category_state_keys(labels: tuple[str, ...] | list[str]) -> dict[str, str]:
    return {label: f"{CATEGORY_STATE_PREFIX}{label}" for label in labels}


def selected_categories_from_state(
    labels: tuple[str, ...] | list[str],
    state: dict[str, object],
) -> tuple[str, ...]:
    keys = category_state_keys(labels)
    return tuple(label for label in labels if bool(state.get(keys[label], True)))


def set_category_selection_state(
    labels: tuple[str, ...] | list[str],
    state: dict[str, object],
    *,
    selected: bool,
) -> None:
    for key in category_state_keys(labels).values():
        state[key] = selected


def run_redaction_for_ui(
    items: tuple[ParsedItem, ...] | list[ParsedItem],
    *,
    selected_labels: tuple[str, ...] | list[str],
    preserved_values: tuple[str, ...] | list[str] = (),
    concurrency: int,
    redactor: RedactorLike | None = None,
) -> RedactionUiOutcome:
    """Run local redaction and return UI-safe rows plus zip bytes."""
    progress_messages: list[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        batch_result = run_redaction_batch(
            tuple(items),
            selected_labels=tuple(selected_labels),
            preserved_values=tuple(preserved_values),
            redaction_service=RedactionService(redactor),
            output_dir=tmpdir,
            concurrency=concurrency,
            lock_redactor=True,
            progress_callback=lambda event: progress_messages.append(event.message),
        )
        zip_bytes = batch_result.zip_path.read_bytes()
        result_rows = tuple(prepare_redaction_result_rows(batch_result))
        failed_rows = tuple(prepare_failed_redaction_rows(batch_result))

    return RedactionUiOutcome(
        total_count=batch_result.total_count,
        complete_count=batch_result.complete_count,
        failed_count=batch_result.failed_count,
        zip_bytes=zip_bytes,
        result_rows=result_rows,
        failed_rows=failed_rows,
        progress_messages=tuple(progress_messages),
    )


def run_v2_redaction_for_ui(
    items: tuple[ParsedItem, ...] | list[ParsedItem],
    *,
    api_key: str = "",
    model_id: str,
    sentence_chunk_size: int,
    api_concurrency: int,
    preserved_values: tuple[str, ...] | list[str] = (),
    retry_limit: int = RETRY_ATTEMPTS_DEFAULT,
    redaction_service: V2RedactionServiceLike | None = None,
) -> V2RedactionUiOutcome:
    """Run v2 redaction and return only privacy-safe UI data."""
    service = redaction_service
    if service is None:
        service = V2RedactionService(
            ResponsesRedactionClient(api_key=api_key),
        )

    progress_messages: list[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        batch_result = run_v2_redaction_batch(
            tuple(items),
            model_id=model_id,
            redaction_service=service,
            output_dir=tmpdir,
            sentence_chunk_size=sentence_chunk_size,
            api_concurrency=api_concurrency,
            retry_limit=retry_limit,
            preserved_values=tuple(preserved_values),
            progress_callback=lambda event: progress_messages.append(event.message),
        )
        zip_bytes = batch_result.zip_path.read_bytes()
        result_rows = tuple(prepare_v2_redaction_result_rows(batch_result))
        failed_rows = tuple(prepare_v2_failed_redaction_rows(batch_result))
        summary = prepare_v2_run_summary(batch_result)

    return V2RedactionUiOutcome(
        total_count=batch_result.total_count,
        complete_count=batch_result.complete_count,
        failed_count=batch_result.failed_count,
        zip_bytes=zip_bytes,
        summary=summary,
        result_rows=result_rows,
        failed_rows=failed_rows,
        progress_messages=tuple(progress_messages),
    )


def prepare_redaction_result_rows(batch_result: BatchResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item_result in batch_result.items:
        rows.append(
            {
                "Item name": item_result.item.item_name,
                "Status": item_result.status,
                "Detected spans": item_result.redaction_result.detected_span_count,
                "Selected redactions": item_result.redaction_result.selected_span_count,
                "Output filename": item_result.output.filename,
            }
        )
    return rows


def prepare_v2_redaction_result_rows(
    batch_result: V2BatchResult,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item_result in batch_result.items:
        rows.append(
            {
                "Item name": item_result.item.item_name,
                "Status": item_result.status,
                "Sentences": item_result.metadata.total_sentence_count,
                "Chunks": item_result.metadata.chunk_count,
                "Retries": item_result.metadata.retry_count,
                "Output filename": item_result.output.filename,
            }
        )
    return rows


def prepare_failed_redaction_rows(batch_result: BatchResult) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item_result in batch_result.items:
        if item_result.status != "failed":
            continue
        for error in item_result.redaction_result.errors or item_result.output.errors:
            rows.append(
                {
                    "Item name": item_result.item.item_name,
                    "Output filename": item_result.output.filename,
                    "Error": error,
                }
            )
    return rows


def prepare_v2_failed_redaction_rows(
    batch_result: V2BatchResult,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item_result in batch_result.failed_items:
        categories = ", ".join(item_result.metadata.error_categories) or "unknown"
        errors = (
            item_result.redaction_result.errors
            or item_result.output.errors
            or (f"v2_redaction_failed:{categories}",)
        )
        for error in errors:
            rows.append(
                {
                    "Item name": item_result.item.item_name,
                    "Output filename": item_result.output.filename,
                    "Error category": categories,
                    "Error": error,
                }
            )
    return rows


def prepare_v2_run_summary(batch_result: V2BatchResult) -> dict[str, object]:
    summary = batch_result.summary
    usage = summary.usage.summary()
    return {
        "model_id": summary.model_id,
        "sentence_chunk_size": summary.sentence_chunk_size,
        "api_concurrency": summary.api_concurrency,
        "retry_limit": summary.retry_limit,
        "total_count": summary.total_item_count,
        "complete_count": summary.complete_item_count,
        "failed_count": summary.failed_item_count,
        "total_sentence_count": summary.total_sentence_count,
        "successful_chunk_count": summary.successful_chunk_count,
        "failed_chunk_count": summary.failed_chunk_count,
        "retry_count": summary.retry_count,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
        "cached_input_tokens": usage["cached_input_tokens"],
        "reasoning_output_tokens": usage["reasoning_output_tokens"],
        "error_categories": summary.error_categories,
    }


def render_app() -> None:
    """Render the local-first Streamlit parse and review workflow."""
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    st.title(APP_TITLE)
    st.info(
        "This app is designed for local transcript redaction. Source files and "
        "transcript text should stay on this machine; do not paste sensitive "
        "content into logs, issues, or support messages."
    )

    _ensure_session_state()

    input_mode = st.radio(
        "Input mode",
        (CSV_MODE, DOCUMENT_MODE),
        horizontal=True,
        index=0,
    )

    st.subheader("Input")
    if input_mode == CSV_MODE:
        uploaded_file = st.file_uploader("OCS CSV file", type=["csv"])
        parse_clicked = st.button(
            "Parse CSV",
            disabled=uploaded_file is None,
        )
        if parse_clicked and uploaded_file is not None:
            _store_parse_outcome(
                parse_csv_upload_bytes(
                    uploaded_file.getvalue(),
                    source_name=uploaded_file.name,
                )
            )
    else:
        _render_document_folder_input()

    _render_parse_messages()
    _render_review()
    _render_redaction_controls()
    _render_redaction_results()


def _ensure_session_state() -> None:
    st.session_state.setdefault(PARSED_ITEMS_STATE_KEY, ())
    st.session_state.setdefault(PARSE_ERRORS_STATE_KEY, ())
    st.session_state.setdefault(PARSE_WARNINGS_STATE_KEY, ())
    st.session_state.setdefault(ZIP_BYTES_STATE_KEY, b"")
    st.session_state.setdefault(REDACTION_SUMMARY_STATE_KEY, {})
    st.session_state.setdefault(REDACTION_RESULT_ROWS_STATE_KEY, ())
    st.session_state.setdefault(REDACTION_FAILED_ROWS_STATE_KEY, ())
    st.session_state.setdefault(REDACTION_PROGRESS_STATE_KEY, ())
    st.session_state.setdefault(DOCUMENT_FOLDER_PATH_STATE_KEY, "")
    st.session_state.setdefault(PRESERVED_VALUES_STATE_KEY, "")
    st.session_state.setdefault(SESSION_API_KEY_STATE_KEY, "")
    st.session_state.setdefault(V2_CUSTOM_MODEL_STATE_KEY, "")


def _store_parse_outcome(outcome: ParseOutcome) -> None:
    st.session_state[PARSED_ITEMS_STATE_KEY] = outcome.items
    st.session_state[PARSE_ERRORS_STATE_KEY] = outcome.errors
    st.session_state[PARSE_WARNINGS_STATE_KEY] = outcome.warnings
    st.session_state[ZIP_BYTES_STATE_KEY] = b""
    st.session_state[REDACTION_SUMMARY_STATE_KEY] = {}
    st.session_state[REDACTION_RESULT_ROWS_STATE_KEY] = ()
    st.session_state[REDACTION_FAILED_ROWS_STATE_KEY] = ()
    st.session_state[REDACTION_PROGRESS_STATE_KEY] = ()


def _render_parse_messages() -> None:
    for error in st.session_state[PARSE_ERRORS_STATE_KEY]:
        st.error(error)
    for warning in st.session_state[PARSE_WARNINGS_STATE_KEY]:
        st.warning(warning)


def _render_document_folder_input() -> None:
    uploaded_files = st.file_uploader(
        "Document folder or files",
        type=SUPPORTED_UPLOAD_TYPES,
        accept_multiple_files="directory",
        help=(
            "Click Browse to choose a folder, or drag a folder/files here. "
            "Only supported document files are uploaded into this local "
            "Streamlit session."
        ),
    )
    parse_upload_clicked = st.button(
        "Parse uploaded documents",
        disabled=not uploaded_files,
    )
    if parse_upload_clicked:
        _store_parse_outcome(parse_uploaded_document_files(uploaded_files or ()))

    with st.expander("Advanced: scan a local folder path"):
        st.text_input("Document folder path", key=DOCUMENT_FOLDER_PATH_STATE_KEY)
        scan_clicked = st.button("Scan folder path")
        if scan_clicked:
            _store_parse_outcome(
                parse_document_folder_path(
                    str(st.session_state.get(DOCUMENT_FOLDER_PATH_STATE_KEY, ""))
                )
            )


def _render_review() -> None:
    items = tuple(st.session_state[PARSED_ITEMS_STATE_KEY])
    st.subheader("Parsed Item Review")
    st.caption("Transcript bodies are not shown in this review.")

    if not items:
        st.write("No parsed items yet.")
        return

    st.dataframe(
        prepare_review_table(items),
        hide_index=True,
        width="stretch",
    )
    error_rows = prepare_parse_error_table(items)
    if error_rows:
        st.error("Some parsed items need attention before redaction.")
        st.dataframe(error_rows, hide_index=True, width="stretch")


def _render_redaction_controls() -> None:
    items = tuple(st.session_state[PARSED_ITEMS_STATE_KEY])
    if not items:
        return

    st.subheader("Redaction")
    st.caption(
        "OpenAI Responses API v2 redacts sentence batches after local parsing "
        "and masking. Review generated outputs before using them as final."
    )

    _render_api_key_controls()
    selected_model_id = _render_model_controls()

    preserved_values_text = st.text_input(
        "Values to keep unredacted",
        key=PRESERVED_VALUES_STATE_KEY,
        placeholder="Optional comma-separated values",
        help=(
            "Values are masked locally before API submission and restored after "
            "validated output. They are not shown in summaries."
        ),
    )
    preserved_values = parse_preserved_values(preserved_values_text)
    throughput_column, concurrency_column = st.columns(2)
    with throughput_column:
        sentence_chunk_size = st.number_input(
            "Sentences per API call",
            min_value=SENTENCE_CHUNK_RANGE.minimum,
            max_value=SENTENCE_CHUNK_RANGE.maximum,
            value=SENTENCE_CHUNK_RANGE.default,
            step=1,
            key=V2_SENTENCE_CHUNK_SIZE_STATE_KEY,
            help="Controls how many masked sentences are sent in each API request.",
        )
    with concurrency_column:
        api_concurrency = st.number_input(
            "Parallel API calls",
            min_value=API_CONCURRENCY_RANGE.minimum,
            max_value=API_CONCURRENCY_RANGE.maximum,
            value=API_CONCURRENCY_RANGE.default,
            step=1,
            key=V2_API_CONCURRENCY_STATE_KEY,
            help="Controls how many API-backed item redactions can run at once.",
        )

    if st.button("Run redaction", type="primary"):
        credential = resolve_api_key(
            session_api_key=str(st.session_state.get(SESSION_API_KEY_STATE_KEY, ""))
        )
        if not credential.is_configured:
            st.error(
                "OpenAI API key is not configured. Set OPENAI_API_KEY or enter a "
                "session-only key."
            )
            return

        try:
            model_id = resolve_model_id(
                selected_model_id=selected_model_id,
                custom_model_id=str(
                    st.session_state.get(V2_CUSTOM_MODEL_STATE_KEY, "")
                ),
            )
        except ValueError as exc:
            st.error(str(exc))
            return

        status = st.empty()
        status.info("Running v2 redaction...")
        with st.spinner("Processing parsed items with the Responses API"):
            outcome = run_v2_redaction_for_ui(
                items,
                api_key=credential.require_value(),
                model_id=model_id,
                sentence_chunk_size=clamp_v2_sentence_chunk_size(
                    int(sentence_chunk_size)
                ),
                api_concurrency=clamp_v2_api_concurrency(int(api_concurrency)),
                preserved_values=preserved_values,
            )
        status.success("Redaction run complete.")
        _store_v2_redaction_outcome(outcome)


def _render_api_key_controls() -> ApiKeyCredential:
    credential = resolve_api_key(
        session_api_key=str(st.session_state.get(SESSION_API_KEY_STATE_KEY, ""))
    )
    if credential.source == "environment":
        st.success("OpenAI API key configured from environment.")
        return credential

    session_api_key = st.text_input(
        "OpenAI API key",
        type="password",
        key=SESSION_API_KEY_STATE_KEY,
        help="Stored only in this Streamlit session; never written to app outputs.",
    )
    credential = resolve_api_key(session_api_key=session_api_key)
    if credential.is_configured:
        st.success("OpenAI API key configured for this session.")
    else:
        st.warning("OpenAI API key is required before running v2 redaction.")
    return credential


def _render_model_controls() -> str:
    labels_by_id = {
        option.model_id: f"{option.display_label} ({option.model_id})"
        for option in V2_MODEL_CATALOG
    }
    model_ids = tuple(labels_by_id)
    selected_model_id = st.selectbox(
        "Model",
        options=model_ids,
        index=0,
        format_func=lambda model_id: labels_by_id[model_id],
        key=V2_MODEL_STATE_KEY,
        help="Choose a configured model, or enter a custom model ID below.",
    )
    st.text_input(
        "Custom model ID",
        key=V2_CUSTOM_MODEL_STATE_KEY,
        placeholder="Optional model override",
        help="Overrides the configured model dropdown when provided.",
    )
    return str(selected_model_id)


def _ensure_category_state(labels: tuple[str, ...]) -> None:
    for key in category_state_keys(labels).values():
        st.session_state.setdefault(key, True)


def _store_redaction_outcome(outcome: RedactionUiOutcome) -> None:
    st.session_state[ZIP_BYTES_STATE_KEY] = outcome.zip_bytes
    st.session_state[REDACTION_SUMMARY_STATE_KEY] = {
        "total_count": outcome.total_count,
        "complete_count": outcome.complete_count,
        "failed_count": outcome.failed_count,
    }
    st.session_state[REDACTION_RESULT_ROWS_STATE_KEY] = outcome.result_rows
    st.session_state[REDACTION_FAILED_ROWS_STATE_KEY] = outcome.failed_rows
    st.session_state[REDACTION_PROGRESS_STATE_KEY] = outcome.progress_messages


def _store_v2_redaction_outcome(outcome: V2RedactionUiOutcome) -> None:
    st.session_state[ZIP_BYTES_STATE_KEY] = outcome.zip_bytes
    st.session_state[REDACTION_SUMMARY_STATE_KEY] = outcome.summary
    st.session_state[REDACTION_RESULT_ROWS_STATE_KEY] = outcome.result_rows
    st.session_state[REDACTION_FAILED_ROWS_STATE_KEY] = outcome.failed_rows
    st.session_state[REDACTION_PROGRESS_STATE_KEY] = outcome.progress_messages


def _render_redaction_results() -> None:
    summary = st.session_state[REDACTION_SUMMARY_STATE_KEY]
    if not summary:
        return

    st.subheader("Redaction Results")
    total_column, complete_column, failed_column = st.columns(3)
    total_column.metric("Total items", summary["total_count"])
    complete_column.metric("Complete", summary["complete_count"])
    failed_column.metric("Failed", summary["failed_count"])

    if "model_id" in summary:
        model_column, sentence_column, chunk_column, retry_column = st.columns(4)
        model_column.metric("Model", summary["model_id"])
        sentence_column.metric("Sentences", summary["total_sentence_count"])
        chunk_column.metric(
            "Chunks",
            (
                f"{summary['successful_chunk_count']} complete / "
                f"{summary['failed_chunk_count']} failed"
            ),
        )
        retry_column.metric("Retries", summary["retry_count"])

        token_column, input_column, output_column = st.columns(3)
        token_column.metric("Total tokens", summary["total_tokens"])
        input_column.metric("Input tokens", summary["input_tokens"])
        output_column.metric("Output tokens", summary["output_tokens"])

    progress_messages = st.session_state[REDACTION_PROGRESS_STATE_KEY]
    if progress_messages:
        with st.expander("Progress messages"):
            for message in progress_messages:
                st.write(message)

    result_rows = st.session_state[REDACTION_RESULT_ROWS_STATE_KEY]
    if result_rows:
        st.dataframe(result_rows, hide_index=True, width="stretch")

    failed_rows = st.session_state[REDACTION_FAILED_ROWS_STATE_KEY]
    if failed_rows:
        st.error("Some items failed. Successful outputs remain available.")
        st.dataframe(failed_rows, hide_index=True, width="stretch")

    zip_bytes = st.session_state[ZIP_BYTES_STATE_KEY]
    st.download_button(
        "Download generated zip",
        data=zip_bytes,
        file_name="redacted-transcripts.zip",
        mime="application/zip",
        disabled=not zip_bytes or summary["complete_count"] == 0,
    )


def _safe_uploaded_relative_path(uploaded_name: str) -> Path:
    normalized_name = uploaded_name.replace("\\", "/")
    parts = [
        _safe_upload_name_part(part)
        for part in PurePosixPath(normalized_name).parts
        if part not in {"", ".", "..", "/"}
    ]
    if not parts:
        return Path("uploaded-file")
    return Path(*parts)


def _safe_upload_name_part(name_part: str) -> str:
    safe_part = _UNSAFE_UPLOAD_NAME_RE.sub("_", name_part).strip(" .")
    return safe_part or "uploaded"


def _unique_upload_path(path: Path) -> Path:
    if not path.exists():
        return path

    counter = 2
    while True:
        candidate = path.with_name(f"{path.stem}-{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _uploaded_file_bytes(uploaded_file: Any) -> bytes:
    if hasattr(uploaded_file, "getvalue"):
        return bytes(uploaded_file.getvalue())
    data = uploaded_file.read()
    return data if isinstance(data, bytes) else bytes(data)
