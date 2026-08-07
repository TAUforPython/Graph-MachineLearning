#!/usr/bin/env python3
"""Inventory and classify the legacy notebooks stored in the repository root."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


CLASSIFICATION = {
    "AGC - agglomerative clustering medical image.ipynb": "Clustering / medical imaging",
    "GCN genom classification.ipynb": "Graph neural networks / genomics",
    "Graph from DataFrame tSNE.ipynb": "Graph construction / dimensionality reduction",
    "Gromov-Waserstein graph clastering.ipynb": "Gromov-Wasserstein / clustering",
    "Gromov-Wasserstein distance Transport Task.ipynb": "Gromov-Wasserstein / optimal transport",
    "Hyperbolic GNN for genomic data.ipynb": "Hyperbolic learning / genomics",
    "Hyperbolic Graph Neural Network.ipynb": "Hyperbolic learning / GNN",
    "LLM Graph triplets visualisation.ipynb": "Knowledge graphs / LLM visualization",
    "ML task table PCA ICA tSNE DBSCAN AggClustering SVM.ipynb": "Classical ML / method survey",
    "Poincare ball for Graph.ipynb": "Hyperbolic learning / visualization",
    "example Gated Graph Attention Network.ipynb": "Graph neural networks / attention",
    "example_Interactive_Graph_Visualisation.ipynb": "Graph visualization / interactive",
    "utils_ERD2MMD.ipynb": "Utilities / ERD conversion",
    "utils_ERD2MMD_graph_visualisation.ipynb": "Utilities / ERD visualization",
    "utils_ERD2MMD_interactive_graph_visualisation.ipynb": "Utilities / ERD interactive visualization",
}


@dataclass(frozen=True)
class NotebookAudit:
    file: str
    category: str
    size_mb: float
    cells: int
    code_cells: int
    markdown_cells: int
    executed_code_cells: int
    stored_outputs: int
    stored_errors: list[str]
    has_colab_badge: bool
    uses_colab_features: bool
    setup_commands: int
    external_urls: int


def audit_notebook(path: Path) -> NotebookAudit:
    """Return reproducibility signals without executing notebook code."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    markdown_cells = [cell for cell in cells if cell.get("cell_type") == "markdown"]
    source = "\n".join("".join(cell.get("source", [])) for cell in cells)
    errors = [
        f"{output.get('ename', 'Error')}: {output.get('evalue', '')}".strip()
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    first_cell = "".join(cells[0].get("source", [])) if cells else ""
    setup_pattern = re.compile(r"(?m)^\s*(?:!|%)(?:pip|apt(?:-get)?|wget|git)\b")

    return NotebookAudit(
        file=path.name,
        category=CLASSIFICATION.get(path.name, "Unclassified"),
        size_mb=round(path.stat().st_size / 1_000_000, 2),
        cells=len(cells),
        code_cells=len(code_cells),
        markdown_cells=len(markdown_cells),
        executed_code_cells=sum(
            cell.get("execution_count") is not None for cell in code_cells
        ),
        stored_outputs=sum(len(cell.get("outputs", [])) for cell in code_cells),
        stored_errors=errors,
        has_colab_badge="colab.research.google.com" in first_cell,
        uses_colab_features="google.colab" in source or "/content/" in source,
        setup_commands=len(setup_pattern.findall(source)),
        external_urls=len(set(re.findall(r"https?://[^\s\)\]\"'<>]+", source))),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audits = [audit_notebook(path) for path in sorted(args.root.glob("*.ipynb"))]
    if args.json:
        print(json.dumps([asdict(audit) for audit in audits], ensure_ascii=False, indent=2))
        return

    print("file\tcategory\tcells\texecuted\toutputs\terrors\tsize_mb")
    for audit in audits:
        print(
            f"{audit.file}\t{audit.category}\t{audit.cells}\t"
            f"{audit.executed_code_cells}/{audit.code_cells}\t"
            f"{audit.stored_outputs}\t{len(audit.stored_errors)}\t{audit.size_mb:.2f}"
        )
    print(f"\nRoot notebooks: {len(audits)}")
    unclassified = [audit.file for audit in audits if audit.category == "Unclassified"]
    if unclassified:
        raise SystemExit(f"Unclassified root notebooks: {', '.join(unclassified)}")


if __name__ == "__main__":
    main()
