from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    ROOT / "HyperParameterInspect.ipynb",
    ROOT / "HyperParameterInspect_EN.ipynb",
)


def test_generated_notebooks_are_current() -> None:
    subprocess.run(
        [sys.executable, "tools/build_notebooks.py", "--check"],
        cwd=ROOT,
        check=True,
    )


def test_bilingual_notebooks_share_code_and_have_clean_outputs() -> None:
    korean, english = [
        nbformat.read(path, as_version=4)
        for path in NOTEBOOKS
    ]
    for notebook in (korean, english):
        nbformat.validate(notebook)
        assert all(
            not cell.outputs and cell.execution_count is None
            for cell in notebook.cells
            if cell.cell_type == "code"
        )

    korean_code = [
        cell.source for cell in korean.cells if cell.cell_type == "code"
    ]
    english_code = [
        cell.source for cell in english.cells if cell.cell_type == "code"
    ]
    assert korean_code == english_code
    assert korean.metadata["hpo_lab"]["language"] == "ko"
    assert english.metadata["hpo_lab"]["language"] == "en"
