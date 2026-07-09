from __future__ import annotations

import importlib
import subprocess
import sys


def test_opf_app_imports() -> None:
    app_package = importlib.import_module("opf_app")

    assert app_package.APP_TITLE == "Privacy Filter Redaction"


def test_streamlit_entrypoint_imports() -> None:
    module = importlib.import_module("streamlit_app")

    assert module.render_app is not None


def test_streamlit_ui_import_does_not_load_legacy_opf_modules() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import opf_app.ui; "
                "print(any(name == 'opf' or name.startswith('opf.') "
                "for name in sys.modules))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"
