# Graph Machine Learning

Practical notebooks for graph visualization, graph neural networks (GNNs),
Gromov–Wasserstein methods, and hyperbolic graph learning.

> **Languages:** the project navigation is in English. The hyperbolic-graph
> notes are bilingual (English + Русский).

## Start here

For the leakage-aware healthcare experiments, shared library, tests, and four
task notebooks, see **[`healthcare-gnn-lab/`](healthcare-gnn-lab/README.md)**.
Its synthetic smoke mode checks software plumbing only and is not clinical
evidence.

The notebooks run in a browser with Google Colab; no local installation is
needed. Open an example, then select **Runtime → Run all**.

| Beginner example | What it demonstrates | Colab |
|---|---|---|
| Graph learning basics | Build, inspect, and visualize a graph | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/D2718281828nis/ML-MachineLearning-Graphs/blob/main/examples/graph_learning_basics.ipynb) |
| Hyperbolic graph basics | Poincaré-ball distances and a tree embedding | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/D2718281828nis/ML-MachineLearning-Graphs/blob/main/examples/hyperbolic_graph_basics.ipynb) |

Each example contains its own Colab badge, learning objectives, explanations,
and deterministic checks. Verification status is stated inside each notebook.

## Repository map

```text
.
├── examples/             # Small, documented, Colab-ready starting points
├── theory/hyperbolic-graphs/
│   ├── README.md         # Bilingual theory index
│   ├── 01-foundations.md # Geometry and the Poincaré ball
│   ├── 02-learning.md    # Embeddings and hyperbolic GNNs
│   ├── 03-models-and-maps.md # General-curvature Poincaré operations
│   └── paper-map.md      # Evidence state and verified source metadata
├── datasets/             # Small data files used by selected notebooks
├── *.ipynb               # Research and experimental notebooks
└── REPOSITORY_PROMPT.md  # Reusable prompt for future repository cleanup
```

## Notebook catalogue

The root contains 15 legacy research notebooks. See the
**[full notebook review and classification](docs/NOTEBOOK_REVIEW.md)** for a
file-by-file reproducibility assessment, detected blockers, maturity labels,
and recommended migration order. The short list below is navigation, not a
claim that every research notebook currently runs end to end.

### Hyperbolic learning

- `Poincare ball for Graph.ipynb` — visual intuition for graph layouts in the
  Poincaré ball.
- `Hyperbolic Graph Neural Network.ipynb` — a hyperbolic GNN experiment.
- `Hyperbolic GNN for genomic data.ipynb` — hyperbolic learning on genomic
  features.

### Graph neural networks and visualization

- `GCN genom classification.ipynb` — population classification with a GCN.
- `example Gated Graph Attention Network.ipynb` — gated graph attention.
- `example_Interactive_Graph_Visualisation.ipynb` — interactive graph display.
- `Graph from DataFrame tSNE.ipynb` — graph construction from tabular data.
- `LLM Graph triplets visualisation.ipynb` — visualization of extracted
  knowledge-graph triplets.

### Distances, clustering, and utilities

- `Gromov-Wasserstein distance Transport Task.ipynb` — optimal-transport
  distance between structured datasets.
- `Gromov-Waserstein graph clastering.ipynb` — graph clustering with GW ideas.
- `AGC - agglomerative clustering medical image.ipynb` — image clustering.
- `ML task table PCA ICA tSNE DBSCAN AggClustering SVM.ipynb` — a broad
  comparison of classical methods.
- `utils_ERD2MMD*.ipynb` — ERD-to-Mermaid and visualization utilities.

Research notebooks preserve their original exploratory form. New users should
begin in [`examples/`](examples/), which contains the curated learning material;
do not infer a successful clean run unless the notebook records one.

## Run locally

```bash
git clone https://github.com/D2718281828nis/ML-MachineLearning-Graphs.git
cd ML-MachineLearning-Graphs
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install jupyter numpy matplotlib networkx
jupyter lab
```

Package requirements differ between research notebooks. Read their first cells
before running them; Colab installation cells are notebook-specific.

## Hyperbolic graph theory / Теория гиперболических графов

Read the bilingual learning path in
[`theory/hyperbolic-graphs/`](theory/hyperbolic-graphs/README.md). It covers:

1. why trees and hierarchies fit hyperbolic space;
2. the Poincaré-ball model and its distance function;
3. hyperbolic embeddings, message passing, and practical numerical safeguards.

The current rigorous increment declares its curvature convention, maps formulas
to notebook functions, and records unresolved source checks in the
[theory editorial plan](docs/HYPERBOLIC_THEORY_PLAN.md). Research notebooks keep
their existing reproducibility warnings.

## Contributing

When adding a notebook:

1. use a descriptive `snake_case.ipynb` filename;
2. include objectives, prerequisites, and an **Open in Colab** badge;
3. install non-Colab dependencies in the first executable cell;
4. set random seeds and avoid machine-specific paths;
5. keep downloaded data small and use stable HTTPS URLs;
6. restart the runtime and verify **Run all** before committing;
7. clear accidental secrets and unnecessarily large cell outputs.

## License

This repository is distributed under the terms in [`LICENSE`](LICENSE).
