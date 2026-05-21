# Transcript Privacy Filter

Transcript Privacy Filter is a local Streamlit app for redacting transcript-like files with OpenAI Privacy Filter (OPF). It keeps the original OPF package, CLI, and model runtime available, but the primary user-facing workflow in this repo is now a local web app for parsing, reviewing, redacting, and downloading generated transcript outputs.

The app is designed for local-first use. It does not add a hosted backend, account system, database, or persistent job history. OPF is a redaction aid and can miss or over-redact spans; generated files should not be treated as a legal anonymization or compliance guarantee.

Repository resources: [License](LICENSE) and [Security Policy](SECURITY.md).

## What The App Does

- Parses one OpenChatStudio CSV export into one reviewable item per chat session.
- Parses uploaded document folders/files, or an advanced local folder path, for supported documents: `.txt`, `.docx`, `.doc`, and embedded-text `.pdf`.
- Shows a review table with inferred date, identifier, parse status, and output filename preview.
- Lets users choose which OPF privacy categories should be redacted.
- Lets users keep known literal values unredacted with an optional comma-separated list.
- Lets users set `Parallel redaction jobs` from `1` to `8`.
- Generates one plain `.docx` file per successful item.
- Formats transcript speaker labels in generated `.docx` files for easier review.
- Packages successful outputs into `redacted-transcripts.zip`.

Full transcript bodies are not shown in the default review UI.

## Setup

Use Python 3.10 or newer. Python 3.12 is recommended.

```bash
python3.12 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

## Run The App

From the repository root:

```bash
streamlit run streamlit_app.py
```

Then open the local Streamlit URL shown in the terminal.

## Input Workflows

### OCS CSV Export

Use the `OCS CSV export` mode and upload a single CSV file exported from OpenChatStudio.

The parser:

- Groups rows by `Session ID`.
- Sorts each session by `Message Date`.
- Includes `human` and `ai` messages.
- Renders `human` as `User` and `ai` as `Chatbot`.
- Produces one parsed item per session.

### Document Folder Or Files

Use the `Document folder` mode to choose a folder or drag documents into the Streamlit uploader. The app uploads supported files into the local Streamlit session and ignores unsupported files with a warning.

Supported document extensions:

- `.txt`
- `.docx`
- `.doc`
- `.pdf`

An advanced expander also keeps the original local folder path scan for same-machine workflows.

For document transcripts, the parser can infer dates and identifiers from filenames and normalizes recognizable transcript rows to the `[timestamp] speaker: text` format.

`.doc` files require local conversion tooling such as LibreOffice/`soffice`. PDFs must contain embedded text; OCR is not supported in v1.

## Redaction Controls

Category selection defaults to all available OPF labels. If a category is unchecked, spans detected for that category are left unchanged in the generated output.

`Values to keep unredacted` accepts an optional comma-separated list of literal values that should stay unchanged even when OPF detects them in a selected category. Matching is case-insensitive and applies across detected categories, so users can preserve known non-sensitive names such as chatbot display names, as well as URLs, dates, IDs, or custom-category values.

`Parallel redaction jobs` controls how many parsed sessions or documents are submitted at the same time. The app clamps this value to `1` through `8` and uses `2` by default.

## Outputs

Successful items are written as plain generated `.docx` files and packaged into a downloadable zip. Failed items are reported in the UI and are not included in the zip.

Generated transcript lines keep timestamps in regular text and render speaker labels such as `USER:`, `CHATBOT:`, or `S1:` in bold uppercase.

Output filenames follow this pattern:

```text
redacted-transcript-<chat-date>-<user-identifier>.docx
```

Duplicate filename bases receive a short stable suffix.

## OPF Checkpoint And Runtime

The app uses the existing OPF Python API underneath the Streamlit workflow. By default, OPF resolves its checkpoint from `OPF_CHECKPOINT` or `~/.opf/privacy_filter`. If no default checkpoint is present, OPF may need to download one or be pointed at a local checkpoint before redaction can run.

The Streamlit app uses CUDA automatically when PyTorch reports that CUDA is available and otherwise falls back to CPU.

The OPF CLI remains available after installation:

```bash
opf --device cpu "Alice was born on 1990-01-02."
```

For CLI output modes and schemas, see [EVAL_AND_OUTPUT_MODES.md](EVAL_AND_OUTPUT_MODES.md) and [OUTPUT_SCHEMAS.md](OUTPUT_SCHEMAS.md).

## Test

Run the full deterministic regression suite:

```bash
python -m pytest
```

The end-to-end tests use fake OPF redactors and do not require a model download. A real OPF smoke test skips unless a checkpoint already exists locally.

## Repository Layout

- [streamlit_app.py](streamlit_app.py): Streamlit entrypoint.
- [opf_app/](opf_app): app models, parsers, redaction wrapper, batch orchestration, output generation, and UI helpers.
- [tests/](tests): unit and end-to-end regression suite.
- [opf/](opf): retained OPF package, runtime, public API, and CLI.
- [examples/](examples): inherited OPF example data and scripts.
- [FINETUNING.md](FINETUNING.md): retained OPF fine-tuning guidance.

## Known V1 Limitations

- No hosted deployment.
- No OCR for scanned or image-only PDFs.
- No source-layout-preserving redaction of Word or PDF files.
- `.doc` input depends on locally installed conversion tooling such as LibreOffice/`soffice`.
- Output is `.docx` only and is delivered as a zip.
- No database, accounts, persistent job history, or hosted storage.
- OPF redaction is not a legal anonymization or compliance guarantee.

## More OPF Context

This repository still includes the original OPF model/runtime code for local inference, evaluation, and fine-tuning. For deeper OPF usage, see:

- [FINETUNING.md](FINETUNING.md)
- [EVAL_AND_OUTPUT_MODES.md](EVAL_AND_OUTPUT_MODES.md)
- [OUTPUT_SCHEMAS.md](OUTPUT_SCHEMAS.md)
- [examples/data/README.md](examples/data/README.md)
- [examples/scripts/README.md](examples/scripts/README.md)
