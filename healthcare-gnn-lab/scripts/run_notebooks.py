#!/usr/bin/env python3
"""Execute the four smoke-enabled lab notebooks without Jupyter packages."""
from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import os
import traceback
from pathlib import Path

RUNNABLE_NOTEBOOKS = (
    "01_patient_similarity_mi.ipynb",
    "02_ptbxl_ecg_graph.ipynb",
    "03_mimic_ehr_graph.ipynb",
    "04_brain_hypergraph_optional.ipynb",
)


def execute_source(source: str, namespace: dict[str, object]) -> tuple[str, str | None]:
    """Execute a cell and return captured streams plus its final expression repr."""
    tree = ast.parse(source, mode="exec")
    expression = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        expression = ast.Expression(tree.body.pop().value)
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        exec(compile(tree, "<notebook-cell>", "exec"), namespace)
        value = eval(compile(expression, "<notebook-cell>", "eval"), namespace) if expression else None
    return stream.getvalue(), None if expression is None else repr(value)


def run_notebook(source_path: Path, output_path: Path, lab_dir: Path) -> None:
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__main__"}
    execution_count = 0
    original_cwd = Path.cwd()
    try:
        os.chdir(lab_dir)
        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            execution_count += 1
            cell["execution_count"] = execution_count
            cell["outputs"] = []
            source = "".join(cell.get("source", []))
            try:
                stream, result = execute_source(source, namespace)
                if stream:
                    cell["outputs"].append({"name": "stdout", "output_type": "stream", "text": stream.splitlines(keepends=True)})
                if result is not None:
                    cell["outputs"].append({"data": {"text/plain": result.splitlines(keepends=True)}, "execution_count": execution_count, "metadata": {}, "output_type": "execute_result"})
            except Exception as error:
                lines = traceback.format_exc().splitlines()
                cell["outputs"].append({"ename": type(error).__name__, "evalue": str(error), "output_type": "error", "traceback": lines})
                raise RuntimeError(f"Failed cell {execution_count} in {source_path}") from error
    finally:
        os.chdir(original_cwd)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    lab_dir = args.lab_dir.resolve()
    sources = [lab_dir / "notebooks" / name for name in RUNNABLE_NOTEBOOKS]
    missing = [str(source) for source in sources if not source.is_file()]
    if missing:
        raise SystemExit("Missing smoke notebook(s): " + ", ".join(missing))
    for source in sources:
        destination = lab_dir / "reports" / "notebook-runs" / source.name
        run_notebook(source, destination, lab_dir)
        print(f"executed {source.name} -> {destination.relative_to(lab_dir)}")


if __name__ == "__main__":
    main()
