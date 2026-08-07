# Root notebook review and classification

This document reviews the **15 legacy `.ipynb` files in the repository root**.
The two curated notebooks under `examples/` are not included in this inventory.
The review is static: notebook JSON, cells, stored outputs, imports, paths, and
setup commands were inspected, but scientific correctness was not re-evaluated.

## Status vocabulary

| Status | Meaning |
|---|---|
| **Runnable candidate** | Self-contained or downloads its input, has no stored error, and has no obvious secret/manual-upload requirement. It still needs a clean Colab **Run all** test. |
| **Needs repair** | A missing input/setup step, incompatible dependency pin, malformed workflow, or stored exception prevents a reproducibility claim. |
| **Interactive** | Requires uploads, credentials, widgets, or user interaction by design; it cannot be validated as a fully unattended run. |
| **Archive / split** | Valuable exploratory material, but too broad or partially executed to serve as one maintained tutorial. |

The labels are deliberately conservative. An old successful output is evidence
of a previous run, not proof that the notebook works today.

## Classification summary

| Domain | Files | Count |
|---|---|---:|
| Hyperbolic graph learning | `Poincare ball for Graph`, `Hyperbolic Graph Neural Network`, `Hyperbolic GNN for genomic data` | 3 |
| GNNs | `GCN genom classification`, `example Gated Graph Attention Network` | 2 |
| Gromov–Wasserstein and graph construction | `Graph from DataFrame tSNE`, `Gromov-Waserstein graph clastering`, `Gromov-Wasserstein distance Transport Task` | 3 |
| Classical ML and medical clustering | `ML task table…`, `AGC - agglomerative clustering medical image` | 2 |
| Knowledge-graph visualization | `LLM Graph triplets visualisation`, `example_Interactive_Graph_Visualisation` | 2 |
| ERD utilities | `utils_ERD2MMD` and its two visualization variants | 3 |
| **Total** |  | **15** |

## File-by-file review

### Hyperbolic graph learning

| Notebook | Status | Review |
|---|---|---|
| [`Poincare ball for Graph.ipynb`](../Poincare%20ball%20for%20Graph.ipynb) | **Runnable candidate** | Small visualization prototype using NumPy, PyTorch, and Matplotlib. It has no explanatory headings beyond the badge and no install cell; add objectives, formulas, and assertions before promoting it to `examples/`. |
| [`Hyperbolic Graph Neural Network.ipynb`](../Hyperbolic%20Graph%20Neural%20Network.ipynb) | **Needs repair** | Installs `geoopt` and PyG, but reads `df_red_green_overlay.csv` from the current working directory without downloading it or using `datasets/`. It contains large stored plots and only limited prose. |
| [`Hyperbolic GNN for genomic data.ipynb`](../Hyperbolic%20GNN%20for%20genomic%20data.ipynb) | **Needs repair** | Broad protein-sequence experiment with no dependency setup cell. A stored output ends with `NameError: name 'embeddings' is not defined`, so the saved run is not clean. |

### Graph neural networks

| Notebook | Status | Review |
|---|---|---|
| [`GCN genom classification.ipynb`](../GCN%20genom%20classification.ipynb) | **Runnable candidate** | The best documented root research notebook: it explains the graph assumption, downloads genomic data, installs PyG, and includes evaluation plots. It still needs pinned/compatible dependencies and a fresh end-to-end Colab test. |
| [`example Gated Graph Attention Network.ipynb`](../example%20Gated%20Graph%20Attention%20Network.ipynb) | **Needs repair** | Uses a built-in PyG dataset, but includes mutually inconsistent PyTorch/CUDA wheel commands (`torch 2.4/cu121` and `torch 2.0/cu118`). Replace them with one runtime-aware PyG installation path and add prose. |

### Gromov–Wasserstein and graph construction

| Notebook | Status | Review |
|---|---|---|
| [`Graph from DataFrame tSNE.ipynb`](../Graph%20from%20DataFrame%20tSNE.ipynb) | **Needs repair** | Combines download, t-SNE, random forest, graph construction, and GW clustering. The saved source contains compressed statements around the download/load step and inconsistent download/output names (`red_green_overlay.csv` versus `df_red_green_overlay.csv`). Split it into focused stages. |
| [`Gromov-Waserstein graph clastering.ipynb`](../Gromov-Waserstein%20graph%20clastering.ipynb) | **Needs repair** | Downloads the repository dataset and clones an external implementation, but mixes relative and `/content/` paths. Its stored run ends in `KeyboardInterrupt`; the filename also contains “Waserstein” and “clastering” typos. |
| [`Gromov-Wasserstein distance Transport Task.ipynb`](../Gromov-Wasserstein%20distance%20Transport%20Task.ipynb) | **Needs repair** | Reads `df_red_green_overlay.csv` without a setup/download cell and has only one minimal heading. Add the data acquisition step, method explanation, and numerical checks. |

### Classical ML and medical clustering

| Notebook | Status | Review |
|---|---|---|
| [`AGC - agglomerative clustering medical image.ipynb`](../AGC%20-%20agglomerative%20clustering%20medical%20image.ipynb) | **Runnable candidate** | Downloads a version-addressed image and documents the paper-inspired workflow. Almost all of its 4.5 MB is stored output; clear or compress plots and add a dependency/setup cell. |
| [`ML task table PCA ICA tSNE DBSCAN AggClustering SVM.ipynb`](../ML%20task%20table%20PCA%20ICA%20tSNE%20DBSCAN%20AggClustering%20SVM.ipynb) | **Archive / split** | A 208-cell, 10.35 MB survey containing seven generated cases and many methods. Only 17 of 144 code cells carry execution counts while 150 outputs are stored, so cell state is inconsistent. Split this into one notebook per task and remove generated CSV round-trips. |

### Knowledge graphs and interactive visualization

| Notebook | Status | Review |
|---|---|---|
| [`LLM Graph triplets visualisation.ipynb`](../LLM%20Graph%20triplets%20visualisation.ipynb) | **Interactive** | Uses a Mistral API key from Colab `userdata`, generates hundreds of requests, and expects CSV interaction. Credential retrieval is appropriate (no literal key was found), but cost, quota, sample-size, and offline-fixture guidance are needed. |
| [`example_Interactive_Graph_Visualisation.ipynb`](../example_Interactive_Graph_Visualisation.ipynb) | **Runnable candidate** | A compact NetworkX/D3 HTML demonstration with no external input. Add dependency setup, explanations, and HTML-output expectations; then it can become a curated example. |

### ERD conversion and visualization utilities

| Notebook | Status | Review |
|---|---|---|
| [`utils_ERD2MMD.ipynb`](../utils_ERD2MMD.ipynb) | **Interactive** | Colab upload/widget workflow that converts ERD XML to Mermaid and writes output files. It is intentionally manual; document accepted XML dialects and include a tiny fixture. |
| [`utils_ERD2MMD_graph_visualisation.ipynb`](../utils_ERD2MMD_graph_visualisation.ipynb) | **Interactive** | Extends ERD parsing with NetworkX, D3, Graphviz, and `pygraphviz`. It requires Colab uploads and apt/pip setup; consolidate duplicate rendering paths and add a fixture-based smoke test. |
| [`utils_ERD2MMD_interactive_graph_visualisation.ipynb`](../utils_ERD2MMD_interactive_graph_visualisation.ipynb) | **Interactive** | A more structured interactive visualizer using widgets, typed helpers, NetworkX, and Graphviz. Clarify how it differs from the other visualization notebook and extract shared parser code into a Python module. |

## Cross-repository findings

1. **All 15 root notebooks have a Colab badge**, but a badge does not guarantee
   unattended execution.
2. **Two notebooks preserve failed outputs:** the hyperbolic genomic notebook
   has a `NameError`, and the misspelled GW clustering notebook has a
   `KeyboardInterrupt`.
3. **Input handling is the largest reproducibility gap.** Several notebooks
   read a current-directory CSV without downloading it or resolving
   `datasets/df_red_green_overlay.csv` from the repository.
4. **Stored output dominates repository size.** The largest offenders are the
   classical ML survey (~10.35 MB), medical clustering (~4.51 MB), GCN genomics
   (~3.58 MB), and hyperbolic GNN (~3.49 MB).
5. **Colab portability varies.** Six notebooks explicitly depend on
   `google.colab` or `/content/`; interactive utilities should retain that label,
   while tutorials should prefer portable paths and optional Colab adapters.
6. **No literal API credential was found.** The LLM notebook correctly requests
   `Mistral_API` from Colab secrets, but it remains a paid/external-service
   workflow rather than an unattended example.

## Recommended migration order

1. Repair input paths in the two hyperbolic/GW notebooks that read
   `df_red_green_overlay.csv`; use one documented dataset helper.
2. Remove stored exceptions and run every repaired notebook from a clean Colab
   CPU runtime before assigning a “verified” badge.
3. Standardize top sections: purpose, maturity, runtime, dependencies, data,
   expected outputs, and last verification date.
4. Promote the GCN genomics and interactive D3 notebooks after clean execution;
   keep API-key and upload-driven notebooks clearly marked **Interactive**.
5. Split the 208-cell classical ML survey and extract shared ERD parsing into a
   tested Python module.
6. Only after links are updated, move legacy files from the root into domain
   folders. Moving them now would break all existing Colab URLs.

## Reproduce the inventory

Run the dependency-free static audit from the repository root:

```bash
python scripts/audit_notebooks.py
python scripts/audit_notebooks.py --json
```

The script fails when a new root notebook has not been assigned a category.
It never executes notebook cells and therefore cannot certify runtime behavior.
