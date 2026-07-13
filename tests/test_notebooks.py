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
BUILD_SPEC = importlib.util.spec_from_file_location(
    "build_notebooks",
    ROOT / "tools" / "build_notebooks.py",
)
if BUILD_SPEC is None or BUILD_SPEC.loader is None:
    raise RuntimeError("Unable to load notebook builder.")
BUILD_MODULE = importlib.util.module_from_spec(BUILD_SPEC)
sys.modules[BUILD_SPEC.name] = BUILD_MODULE
BUILD_SPEC.loader.exec_module(BUILD_MODULE)


def test_generated_notebooks_are_current() -> None:
    subprocess.run(
        [sys.executable, "tools/build_notebooks.py", "--check"],
        cwd=ROOT,
        check=True,
    )


def test_published_notebook_outputs_are_current() -> None:
    subprocess.run(
        [sys.executable, "tools/execute_notebooks.py", "--check"],
        cwd=ROOT,
        check=True,
    )


def test_source_regeneration_preserves_current_execution() -> None:
    existing = nbformat.read(NOTEBOOKS[0], as_version=4)
    generated = BUILD_MODULE.build_notebook("ko")

    BUILD_MODULE.preserve_execution(existing, generated)

    existing_code = [
        cell for cell in existing.cells if cell.cell_type == "code"
    ]
    generated_code = [
        cell for cell in generated.cells if cell.cell_type == "code"
    ]
    assert [cell.execution_count for cell in generated_code] == [
        cell.execution_count for cell in existing_code
    ]
    assert [cell.outputs for cell in generated_code] == [
        cell.outputs for cell in existing_code
    ]
    assert (
        generated.metadata["hpo_lab"]["execution"]
        == existing.metadata["hpo_lab"]["execution"]
    )


def test_execution_fingerprint_covers_cell_metadata() -> None:
    original = BUILD_MODULE.build_notebook("ko")
    changed = deepcopy(original)
    changed.cells[2].metadata["tags"] = ["skip-execution"]

    assert (
        BUILD_MODULE.runtime_fingerprint(original)
        != BUILD_MODULE.runtime_fingerprint(changed)
    )


def test_runtime_fingerprint_normalizes_line_endings(tmp_path: Path) -> None:
    unix = tmp_path / "unix.txt"
    windows = tmp_path / "windows.txt"
    unix.write_bytes(b"first\nsecond\n")
    windows.write_bytes(b"first\r\nsecond\r\n")

    assert BUILD_MODULE.normalized_text_bytes(
        unix
    ) == BUILD_MODULE.normalized_text_bytes(windows)


def test_source_validation_rejects_markdown_attachments() -> None:
    expected = BUILD_MODULE.build_notebook("ko")
    attached = deepcopy(expected)
    attached.cells[0]["attachments"] = {
        "local.txt": {"text/plain": "unexpected attachment"}
    }

    assert not BUILD_MODULE.source_matches(attached, expected)


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


def test_bilingual_notebooks_share_code_and_published_outputs() -> None:
    korean, english = [
        nbformat.read(path, as_version=4)
        for path in NOTEBOOKS
    ]
    for notebook in (korean, english):
        nbformat.validate(notebook)
        code_cells = [
            cell for cell in notebook.cells if cell.cell_type == "code"
        ]
        assert [cell.execution_count for cell in code_cells] == list(
            range(1, len(code_cells) + 1)
        )
        assert all(cell.outputs for cell in code_cells)
        assert all(
            output.output_type != "error"
            for cell in code_cells
            for output in cell.outputs
        )

    korean_code = [
        cell.source for cell in korean.cells if cell.cell_type == "code"
    ]
    english_code = [
        cell.source for cell in english.cells if cell.cell_type == "code"
    ]
    assert korean_code == english_code
    korean_outputs = [
        cell.outputs for cell in korean.cells if cell.cell_type == "code"
    ]
    english_outputs = [
        cell.outputs for cell in english.cells if cell.cell_type == "code"
    ]
    assert korean_outputs == english_outputs
    assert korean.metadata["hpo_lab"]["language"] == "ko"
    assert english.metadata["hpo_lab"]["language"] == "en"
