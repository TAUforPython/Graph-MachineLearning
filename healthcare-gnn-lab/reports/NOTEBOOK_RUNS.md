# Structured notebook run report / Отчёт о запуске notebooks

## Scope / Объём проверки

On 2026-08-21, the four numbered notebooks in `notebooks/` were executed from top
to bottom on the local CPU environment. The `notebooks/legacy/` directory was
intentionally excluded. Executed copies, including cell outputs and execution
counts, are stored in [`notebook-runs/`](notebook-runs/). Detailed
machine-readable results are stored in the four `*_smoke.json` files in this
folder.

21 августа 2026 года четыре нумерованных notebook из `notebooks/` были полностью
выполнены в локальном CPU-окружении. Папка `notebooks/legacy/` намеренно
исключена. Выполненные копии с outputs и execution counts находятся в
[`notebook-runs/`](notebook-runs/), а подробные машиночитаемые результаты — в
четырёх файлах `*_smoke.json` этой папки.

**Status:** all four runs completed; every report has
`status = synthetic_smoke_not_clinical`. These runs validate software plumbing,
not clinical performance, dataset fidelity, or Colab reproducibility. /
**Статус:** все четыре запуска завершены; каждый отчёт содержит
`status = synthetic_smoke_not_clinical`. Проверен программный pipeline, но не
клиническая эффективность, соответствие реальным данным или запуск в Colab.

## Environment / Окружение

The reports record Python 3.14.4, NumPy 2.5.2, scikit-learn 1.9.0, PyTorch
2.13.0+cu130, and PyTorch Geometric 2.8.0.post1. Although the PyTorch build has a
CUDA suffix, these notebook runs used the available local execution path and do
not constitute a GPU benchmark. / В отчётах записаны Python 3.14.4, NumPy 2.5.2,
scikit-learn 1.9.0, PyTorch 2.13.0+cu130 и PyTorch Geometric 2.8.0.post1. Суффикс
CUDA в сборке PyTorch не превращает этот запуск в GPU-benchmark.

## Results / Результаты

All thresholds were selected on validation data; metrics below are calculated on
the synthetic test split. AUPRC confidence intervals are wide because these are
small smoke fixtures. / Все thresholds выбраны по validation; метрики рассчитаны
на synthetic test split. Интервалы AUPRC широкие из-за малого размера smoke data.

| Notebook | Split (train/val/test) | Graph (density; components) | Logistic AUPRC / balanced accuracy | GCN AUPRC / balanced accuracy | Hyperbolic GCN AUPRC / balanced accuracy |
|---|---:|---:|---:|---:|---:|
| MI patient similarity | 108 / 36 / 36 | 0.0273; 1 | 0.8725 / 0.7630 | 0.8142 / 0.7305 | 0.8222 / 0.8214 |
| PTB-XL lead graph entry point | 57 / 19 / 20 | 0.0513; 1 | 0.4026 / 0.5000 | 0.6512 / 0.7917 | 0.6580 / 0.7500 |
| MIMIC EHR graph entry point | 90 / 30 / 30 | 0.0336; 1 | 0.8512 / 0.7273 | 0.5732 / 0.7784 | 0.6455 / 0.6591 |
| Optional brain hypergraph entry point | 72 / 24 / 24 | 0.0546; 1 | 0.6214 / 0.6875 | 0.7151 / 0.6562 | 0.6251 / 0.5938 |

### Interpretation / Интерпретация

- **MI:** logistic regression has the highest AUPRC; the hyperbolic adaptation
  has the highest balanced accuracy. / Logistic regression имеет лучший AUPRC,
  hyperbolic-модель — лучший balanced accuracy.
- **PTB-XL fixture:** both graph models exceed logistic regression on these two
  metrics, but this synthetic patient-node smoke runner is not a PTB-XL ECG
  evaluation. / Обе graph-модели лучше по этим двум метрикам, но synthetic runner
  не является оценкой на PTB-XL ECG.
- **MIMIC fixture:** logistic regression has the highest AUPRC; GCN has the
  highest balanced accuracy. / Logistic regression имеет лучший AUPRC, GCN —
  лучший balanced accuracy.
- **Brain fixture:** GCN has the highest AUPRC, while logistic regression has the
  highest balanced accuracy. This run does not use ABIDE or clinical fMRI. / GCN
  имеет лучший AUPRC, logistic regression — лучший balanced accuracy; ABIDE и
  клинические fMRI не использовались.
- Edge dropping increased fragmentation or isolation in the MI, MIMIC, and brain
  fixtures; the PTB-XL fixture changed from one to three components. This is a
  graph-integrity diagnostic, not evidence of causal graph value. / Edge dropping
  увеличил фрагментацию или число isolated nodes; это диагностика целостности
  графа, а не доказательство причинной ценности графа.

## Reproduction / Воспроизведение

From the repository root, run:

```bash
PYTHONPATH=healthcare-gnn-lab/src \
  python healthcare-gnn-lab/scripts/run_notebooks.py
```

The standard-library runner executes only `notebooks/*.ipynb`, fails on the first
cell error, writes executed copies to `reports/notebook-runs/`, and leaves source
notebooks without generated outputs. / Runner на стандартной библиотеке выполняет
только `notebooks/*.ipynb`, завершается при первой ошибке, сохраняет выполненные
копии в `reports/notebook-runs/` и не добавляет outputs в исходные notebooks.
