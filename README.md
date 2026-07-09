# Transcript Privacy Filter

Transcript Privacy Filter is a locally run Streamlit app that redacts transcript-like files with the OpenAI Responses API. Parsing, preserved-value masking, response validation, reconstruction, generated files, and temporary artifacts stay on the local machine. Only the masked sentence batches selected for redaction are submitted to the API.

The app is a redaction aid, not a legal anonymization or compliance guarantee. A person must review every generated document before it is shared or treated as final.

Repository resources: [License](LICENSE) and [Security Policy](SECURITY.md).

## What The App Does

- Parses one OpenChatStudio (OCS) CSV export into one reviewable item per chat session.
- Parses uploaded document folders/files, or an advanced local folder path, for `.txt`, `.docx`, `.doc`, and embedded-text `.pdf` documents.
- Keeps timestamps, speaker prefixes, line order, filenames, and generated headers out of API sentence text.
- Lets users select a configured model or enter a custom model ID.
- Lets users send `1` to `5` sentences per API call and run `1` to `8` API calls concurrently.
- Masks optional comma-separated values locally before API submission and restores them only after response validation.
- Validates structured responses, retries invalid or transient failures, and isolates failed items.
- Generates one plain `.docx` per successful item and packages successful outputs in `redacted-transcripts.zip`.
- Shows privacy-safe status, retry, error-category, and token-usage summaries.

Full transcript bodies, redacted text, API keys, and preserved-value lists are not shown in summaries or written to application logs.

## Requirements And Setup

Use Python 3.10 or newer. Python 3.12 is recommended.

The app requires an OpenAI API key associated with the account approved for this workflow. The product design assumes a Zero Data Retention (ZDR)-enabled account; supplying a key does not itself verify ZDR status.

```bash
python3.12 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
export OPENAI_API_KEY="your-api-key"
```

If `OPENAI_API_KEY` is not set, the app offers a password field that keeps the key only in the active Streamlit session. The app never writes the key to disk or generated outputs.

## Run The App

From the repository root:

```bash
streamlit run streamlit_app.py
```

Then open the local URL shown in the terminal.

## Input Workflows

### OCS CSV Export

Choose `OCS CSV export`, upload one OCS export, and select `Parse CSV`. The parser groups rows by `Session ID`, orders them by `Message Date`, includes `human` and `ai` messages, and maps those roles to `User` and `Chatbot`.

### Document Folder Or Files

Choose `Document folder`, then upload a folder/files or use the advanced same-machine folder-path scan. Supported extensions are `.txt`, `.docx`, `.doc`, and `.pdf`.

`.doc` files require local conversion software such as LibreOffice/`soffice`. PDFs must contain embedded text; OCR and source-layout-preserving redaction are not supported.

## Redaction Controls

- `OpenAI API key`: shown only when the environment does not already provide one.
- `Model`: a configured Responses API model.
- `Custom model ID`: an optional override for testing an available model without changing code.
- `Values to keep unredacted`: comma-separated literal values, matched case-insensitively, masked locally, and restored locally.
- `Sentences per API call`: `1-5`, default `3`.
- `Parallel API calls`: `1-8`, default `4`.

Each request is stateless, uses Structured Outputs, sets `store: false`, and enables no hosted tools. The app does not use conversations or `previous_response_id`.

## Outputs And Review

Successful items are written as generated `.docx` files and packaged into a downloadable zip. Failed items are reported without exposing source or redacted text and are excluded from the zip.

Output filenames follow:

```text
redacted-transcript-<chat-date>-<user-identifier>.docx
```

Duplicate filename bases receive a short stable suffix. Generated transcript lines keep readable timestamps and bold uppercase speaker labels.

Always inspect the generated files. Model output can miss private information, redact too much, or preserve content that should have been removed.

## Privacy Boundaries

- Local: source parsing, sentence segmentation, structure separation, preserved-value masking/restoration, validation, reconstruction, outputs, zip packaging, and temporary files.
- API: only the minimum masked sentence text required for redaction.
- Never persisted by the app: API keys, source text, redacted text, preserved values, request payloads, or raw API responses.
- Never sent for redaction: source filenames, participant identifiers as metadata, output filenames, generated headers, or transcript structure prefixes.

## Tests

Run the deterministic suite:

```bash
python -m pytest
```

Fake-client end-to-end tests cover both CSV and document workflows without network calls or paid usage. The Streamlit smoke test verifies that v2 controls render.

The real Responses API smoke test is opt-in:

```bash
RUN_OPENAI_API_SMOKE=1 OPENAI_API_KEY="your-api-key" python -m pytest \
  tests/test_v2_e2e_workflows.py -k real_openai_api
```

Set `OPENAI_API_SMOKE_MODEL` to override the configured default model for that smoke test. Enabling it makes a real API request and may incur usage charges.

## Repository Layout

- [streamlit_app.py](streamlit_app.py): Streamlit entrypoint.
- [opf_app/](opf_app): active parsers, masking, Responses API client, validation, batching, output generation, and UI.
- [tests/](tests): deterministic regressions plus the opt-in real API smoke test.

## Legacy OPF Source

The historical local OPF model, CLI, evaluation, and fine-tuning source remains in `opf/` and `examples/` for repository compatibility, but the Streamlit app does not import or use it. Its large local-model dependencies are isolated from the default app install behind the optional `legacy-opf` extra.
