#!/usr/bin/env python
"""Execute one notebook and publish identical outputs to both languages."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import tempfile

from nbclient import NotebookClient
import nbformat

from build_notebooks import (
    EXECUTION_ENV,
    EXECUTION_MODE,
    EXECUTOR_SETTINGS,
    OUTPUTS,
    ROOT,
    build_notebook,
    runtime_fingerprint,
    source_matches,
)


def execute_korean() -> nbformat.NotebookNode:
    notebook = build_notebook("ko")
    with tempfile.TemporaryDirectory(prefix="hpo-notebook-") as workdir:
        previous = {
            key: os.environ.get(key)
            for key in EXECUTION_ENV
        }
        os.environ.update(EXECUTION_ENV)
        try:
            client = NotebookClient(
                notebook,
                resources={"metadata": {"path": workdir}},
                **EXECUTOR_SETTINGS,
            )
            executed = client.execute()
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    return executed


def copy_execution(
    executed: nbformat.NotebookNode,
    target: nbformat.NotebookNode,
) -> None:
    expected = build_notebook("ko")
    executed_source = [
        (cell.cell_type, cell.get("id"), cell.source, cell.metadata)
        for cell in executed.cells
    ]
    expected_source = [
        (cell.cell_type, cell.get("id"), cell.source, cell.metadata)
        for cell in expected.cells
    ]
    if executed_source != expected_source:
        raise ValueError("Executed notebook source differs from the generated Korean source.")
    executed_code = [cell for cell in executed.cells if cell.cell_type == "code"]
    target_code = [cell for cell in target.cells if cell.cell_type == "code"]
    if [cell.source for cell in executed_code] != [cell.source for cell in target_code]:
        raise ValueError("Bilingual code cells are not synchronized.")

    for source_cell, target_cell in zip(executed_code, target_code):
        target_cell.execution_count = source_cell.execution_count
        target_cell.outputs = deepcopy(source_cell.outputs)


def stamp_execution(notebook: nbformat.NotebookNode) -> None:
    notebook.metadata["hpo_lab"]["execution"] = {
        "mode": EXECUTION_MODE,
        "runtime_sha256": runtime_fingerprint(notebook),
        "outputs_synchronized": True,
    }


def output_payload(notebook: nbformat.NotebookNode) -> list[dict]:
    return [
        json.loads(nbformat.writes(nbformat.v4.new_notebook(cells=[deepcopy(cell)])))["cells"][0][
            "outputs"
        ]
        for cell in notebook.cells
        if cell.cell_type == "code"
    ]


def validate_published_notebooks() -> list[str]:
    errors: list[str] = []
    notebooks: dict[str, nbformat.NotebookNode] = {}
    for language, path in OUTPUTS.items():
        if not path.exists():
            errors.append(f"{path.name} is missing")
            continue
        notebook = nbformat.read(path, as_version=4)
        notebooks[language] = notebook
        nbformat.validate(notebook)
        expected = build_notebook(language)
        if not source_matches(notebook, expected):
            errors.append(f"{path.name} source is stale")
            continue
        execution = notebook.metadata.get("hpo_lab", {}).get("execution", {})
        if execution.get("mode") != EXECUTION_MODE:
            errors.append(f"{path.name} was not executed in {EXECUTION_MODE} mode")
        if execution.get("runtime_sha256") != runtime_fingerprint(notebook):
            errors.append(f"{path.name} outputs are stale for the current runtime")

        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        counts = [cell.execution_count for cell in code_cells]
        if counts != list(range(1, len(code_cells) + 1)):
            errors.append(f"{path.name} does not contain a complete sequential execution")
        if any(not cell.outputs for cell in code_cells):
            errors.append(f"{path.name} contains code cells without published output")
        if any(
            output.output_type == "error"
            for cell in code_cells
            for output in cell.outputs
        ):
            errors.append(f"{path.name} contains execution errors")
        image_count = sum(
            "image/png" in output.get("data", {})
            for cell in code_cells
            for output in cell.outputs
        )
        if image_count < 5:
            errors.append(f"{path.name} contains only {image_count} embedded charts")

    if set(notebooks) == set(OUTPUTS):
        korean_code = [
            cell.source for cell in notebooks["ko"].cells if cell.cell_type == "code"
        ]
        english_code = [
            cell.source for cell in notebooks["en"].cells if cell.cell_type == "code"
        ]
        if korean_code != english_code:
            errors.append("Bilingual code cells differ")
        if output_payload(notebooks["ko"]) != output_payload(notebooks["en"]):
            errors.append("Bilingual outputs differ")
    return errors


def publish() -> None:
    executed = execute_korean()
    for language, path in OUTPUTS.items():
        notebook = build_notebook(language)
        copy_execution(executed, notebook)
        stamp_execution(notebook)
        path.write_text(
            nbformat.writes(notebook, version=4) + "\n",
            encoding="utf-8",
        )
        print(f"published {path.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if not args.check:
        publish()
    errors = validate_published_notebooks()
    if errors:
        raise SystemExit("\n".join(errors))
    if args.check:
        print("published notebook outputs are current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
