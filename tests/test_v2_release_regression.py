from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from opf_app.models import ParsedItem
from opf_app.ui import PARSED_ITEMS_STATE_KEY


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_app_renders_v2_controls() -> None:
    app = AppTest.from_file(str(REPO_ROOT / "streamlit_app.py"), default_timeout=10)
    app.run()
    assert not app.exception

    app.session_state[PARSED_ITEMS_STATE_KEY] = (
        ParsedItem(
            item_name="render-smoke",
            source_name="render-smoke.txt",
            source_type="document",
            chat_date="2026-01-01",
            user_identifier="render-smoke",
            body_text="[00:00] S1: Synthetic render smoke.",
            output_filename="redacted-transcript-2026-01-01-render-smoke.docx",
        ),
    )
    app.run()

    assert not app.exception
    assert "Model" in [element.label for element in app.selectbox]
    assert "Custom model ID" in [element.label for element in app.text_input]
    assert "Values to keep unredacted" in [
        element.label for element in app.text_input
    ]
    number_input_labels = [element.label for element in app.number_input]
    assert "Sentences per API call" in number_input_labels
    assert "Parallel API calls" in number_input_labels
    assert "Run redaction" in [element.label for element in app.button]
    assert "Parallel redaction jobs" not in number_input_labels
    assert not app.checkbox


def test_public_docs_describe_v2_as_active_workflow() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    user_guide = (REPO_ROOT / "docs/streamlit-app-user-guide.md").read_text(
        encoding="utf-8"
    )
    active_readme = readme.split("## Legacy OPF Source", maxsplit=1)[0]
    active_docs = f"{active_readme}\n{user_guide}"

    assert "OpenAI Responses API" in active_docs
    assert "store: false" in active_docs
    assert "Zero Data Retention" in active_docs
    assert "review every generated" in active_docs.lower()
    assert "OPF_CHECKPOINT" not in active_docs
    assert "Category selection defaults" not in active_docs
    assert "CUDA" not in active_docs
    assert "falls back to CPU" not in active_docs


def test_default_app_dependencies_exclude_legacy_local_model_runtime() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project_dependencies = pyproject.split(
        "[project.optional-dependencies]", maxsplit=1
    )[0]
    optional_dependencies = pyproject.split(
        "[project.optional-dependencies]", maxsplit=1
    )[1]

    for dependency in (
        "huggingface_hub",
        "numpy",
        "packaging",
        "safetensors",
        "tiktoken",
        "torch",
    ):
        assert f'"{dependency}"' not in project_dependencies
        assert f'"{dependency}"' in optional_dependencies


def test_active_v2_import_path_does_not_reference_legacy_redaction_modules() -> None:
    ui_source = (REPO_ROOT / "opf_app/ui.py").read_text(encoding="utf-8")
    masking_source = (REPO_ROOT / "opf_app/masking.py").read_text(encoding="utf-8")

    assert "from .redaction import" not in ui_source
    assert "from .batch import" not in ui_source
    assert "from .redaction import" not in masking_source
