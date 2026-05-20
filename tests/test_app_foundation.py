from __future__ import annotations

import importlib


def test_opf_app_imports() -> None:
    app_package = importlib.import_module("opf_app")

    assert app_package.APP_TITLE == "Privacy Filter Redaction"


def test_streamlit_entrypoint_imports() -> None:
    module = importlib.import_module("streamlit_app")

    assert module.render_app is not None


def test_opf_public_api_import_smoke() -> None:
    from opf._api import OPF

    assert OPF.__name__ == "OPF"
