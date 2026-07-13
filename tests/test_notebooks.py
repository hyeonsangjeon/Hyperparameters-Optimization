from __future__ import annotations

import importlib.util
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    ROOT / "HyperParameterInspect.ipynb",
    ROOT / "HyperParameterInspect_EN.ipynb",
)
SNAPSHOT_SPEC = importlib.util.spec_from_file_location(
    "render_benchmark_snapshot",
    ROOT / "tools" / "render_benchmark_snapshot.py",
)
if SNAPSHOT_SPEC is None or SNAPSHOT_SPEC.loader is None:
    raise RuntimeError("Unable to load benchmark snapshot renderer.")
SNAPSHOT_MODULE = importlib.util.module_from_spec(SNAPSHOT_SPEC)
SNAPSHOT_SPEC.loader.exec_module(SNAPSHOT_MODULE)


def test_generated_notebooks_are_current() -> None:
    subprocess.run(
        [sys.executable, "tools/build_notebooks.py", "--check"],
        cwd=ROOT,
        check=True,
    )


def test_generated_benchmark_tables_are_current() -> None:
    subprocess.run(
        [sys.executable, "tools/render_benchmark_snapshot.py", "--check"],
        cwd=ROOT,
        check=True,
    )


def test_benchmark_snapshot_rejects_duplicate_rows_and_wrong_budgets() -> None:
    rows, config, _ = SNAPSHOT_MODULE.load_snapshot()

    duplicate = [*deepcopy(rows), deepcopy(rows[0])]
    with pytest.raises(ValueError, match="exactly once"):
        SNAPSHOT_MODULE.validate(duplicate, config)

    wrong_budget = deepcopy(rows)
    wrong_budget[1]["fit_count"] = "37.9"
    with pytest.raises(ValueError, match="must report 37 fits"):
        SNAPSHOT_MODULE.validate(wrong_budget, config)


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
