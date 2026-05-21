from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Sequence

import streamlit as st

from . import APP_TITLE
from .batch import BatchResult, clamp_concurrency, run_redaction_batch
from .documents import SUPPORTED_DOCUMENT_EXTENSIONS, parse_document
from .models import ParsedItem
from .ocs_csv import OcsCsvValidationError, parse_ocs_csv_text
from .redaction import (
    DEFAULT_RUNTIME_LABELS,
    RedactionService,
    RedactorLike,
    parse_preserved_values,
)


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

    labels = DEFAULT_RUNTIME_LABELS
    _ensure_category_state(labels)

    st.subheader("Redaction")
    st.caption(
        "Unselected categories are left unchanged even when the privacy filter "
        "detects them. OPF is a redaction aid and may miss or over-redact spans."
    )

    select_column, clear_column = st.columns(2)
    with select_column:
        if st.button("Select all categories"):
            set_category_selection_state(labels, st.session_state, selected=True)
    with clear_column:
        if st.button("Clear all categories"):
            set_category_selection_state(labels, st.session_state, selected=False)

    checkbox_columns = st.columns(2)
    keys = category_state_keys(labels)
    for index, label in enumerate(labels):
        with checkbox_columns[index % 2]:
            st.checkbox(label, key=keys[label])

    selected_labels = selected_categories_from_state(labels, st.session_state)
    preserved_values_text = st.text_input(
        "Values to keep unredacted",
        key=PRESERVED_VALUES_STATE_KEY,
        placeholder="Optional comma-separated values",
        help=(
            "Detected spans that exactly match one of these values stay unchanged, "
            "even when their category is selected."
        ),
    )
    preserved_values = parse_preserved_values(preserved_values_text)
    concurrency = st.number_input(
        "Parallel redaction jobs",
        min_value=1,
        max_value=8,
        value=2,
        step=1,
        help=(
            "Controls how many parsed sessions/documents are submitted to the "
            "privacy filter at the same time."
        ),
    )

    if st.button("Run redaction", type="primary"):
        status = st.empty()
        status.info("Running local redaction...")
        with st.spinner("Processing parsed items locally"):
            outcome = run_redaction_for_ui(
                items,
                selected_labels=selected_labels,
                preserved_values=preserved_values,
                concurrency=clamp_concurrency(int(concurrency)),
            )
        status.success("Redaction run complete.")
        _store_redaction_outcome(outcome)


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


def _render_redaction_results() -> None:
    summary = st.session_state[REDACTION_SUMMARY_STATE_KEY]
    if not summary:
        return

    st.subheader("Redaction Results")
    total_column, complete_column, failed_column = st.columns(3)
    total_column.metric("Total items", summary["total_count"])
    complete_column.metric("Complete", summary["complete_count"])
    failed_column.metric("Failed", summary["failed_count"])

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
