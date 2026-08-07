# Healthcare GNN Lab

## Architecture and assumptions

The `data` package owns patient splits and temporal guards. `graphs` fits
preprocessing only on training indices and builds label-free inductive graphs.
`models` supplies matched non-graph, Euclidean, and Poincaré/tangent-aggregation
models. `training` selects thresholds and early stopping on validation data.
`evaluation` computes classification metrics and patient-cluster bootstrap
intervals. Notebooks contain orchestration—not hidden model logic.

Assumptions:

1. the prediction time, cohort, target, and feature availability are declared
   before modeling;
2. the patient is the split unit, including repeat records;
3. protected attributes are excluded from similarity edges by default;
4. validation data select thresholds/hyperparameters and test data are final;
5. synthetic fixtures verify plumbing, never expected clinical performance;
6. a graph is useful only if it beats feature-matched baselines and perturbation
   controls under the same leakage-safe protocol.

## User guide: how it works

### What you do

1. **Choose an example.** Start with the MI notebook. It is the shortest path
   through patient splitting, graph construction, model training, and reporting.
   The PTB-XL, MIMIC, and brain notebooks demonstrate the interfaces for more
   specialized data.
2. **Open it in Colab.** Use a link in the table below and run the cells from top
   to bottom. The first executable cell clones this repository and installs the
   `healthcare_gnn` package. Core logic stays in `src/`; the notebook only selects
   a configuration, starts a run, and displays its report.
3. **Run synthetic smoke mode first.** It creates deterministic artificial rows,
   splits them by synthetic patient ID, fits preprocessing on training patients,
   builds a graph without labels, and compares logistic regression, a GCN, and a
   Poincaré/tangent-aggregation GCN.
4. **Read the JSON report.** Confirm `status` is
   `synthetic_smoke_not_clinical`. Inspect the patient split, graph diagnostics,
   validation-selected threshold, test metrics, edge-drop audit, seed, curvature,
   configuration snapshot, and package versions.
5. **Only then prepare real data.** Follow the relevant access policy and review
   the target, prediction time, feature availability, patient grouping, and
   license. The CLI intentionally refuses the example full-data configuration;
   this prevents an unreviewed file from being mistaken for a valid experiment.

### What happens inside one run

```text
configuration
    ↓
patient-level train / validation / test split
    ↓
training-only imputation and scaling
    ↓
label-free graph construction
    ↓
baseline + Euclidean GNN + geometry-aware model
    ↓
validation threshold / early stopping
    ↓
final test metrics + graph perturbation audit + JSON report
```

The logistic baseline receives the same train-fitted features as the graph
models. During GNN optimization, only train-to-train edges are used. Validation
data select the checkpoint and decision threshold. Test labels are used only for
the final report. A real study still needs nested tuning, repeated seeds,
calibration, confidence intervals, subgroup review, and an independently checked
data dictionary.

### How to interpret the comparison

- **Baseline wins:** report that the graph did not add value under this setup.
- **GNN wins:** verify the advantage survives rewired/dropped edges, shuffled
  features, repeated seeds, and paired patient-level uncertainty estimates.
- **Hyperbolic model wins:** additionally check parameter counts, curvature
  sensitivity, finite gradients, radius distribution, and Euclidean controls.
- **Smoke run wins or loses:** draw no clinical conclusion. Synthetic data only
  verify that the software path executes.

### Common user errors

| Message or symptom | Meaning | Action |
|---|---|---|
| `Strict inductive mode refuses preprocessing fit on the full cohort` | Training rows were not declared separately. | Supply training indices; do not disable strict mode for a deployment/generalization claim. |
| `MIMIC files are absent` | Credentialed data were not found locally. | Complete PhysioNet access and download manually; the project never requests credentials or downloads full MIMIC automatically. |
| Colab cannot import `healthcare_gnn` | The clone/install cell did not finish or the working directory is wrong. | Re-run setup and confirm the directory is `/content/ML-MachineLearning-Graphs/healthcare-gnn-lab`. |
| A metric is unexpectedly high | Leakage, duplicated patients, target-derived features, or graph construction may be wrong. | Stop interpretation and audit patient IDs, timestamps, feature provenance, edges, and split-specific fitting. |
| GNN output changes between runs | Hardware or an operation may be nondeterministic. | Check the recorded versions/device, use the same seed, and repeat the full experiment over several seeds. |

## Руководство пользователя

### Назначение проекта

`Healthcare GNN Lab` — учебная исследовательская среда для честного сравнения
табличной модели, евклидовой графовой нейронной сети и геометрически осознанной
модели. Проект помогает проверить, даёт ли структура графа дополнительную
информацию. Он **не является диагностической системой**, не предлагает лечение
и не подтверждает клиническую готовность модели.

Текущий режим `synthetic` использует только искусственные данные. Он проверяет
загрузку конфигурации, разделение по пациентам, отсутствие очевидной утечки,
построение графа, обучение и формирование отчёта. Его метрики нельзя переносить
на UCI MI, PTB-XL, MIMIC или ABIDE.

### Что делает пользователь

1. **Выберите пример.** Начните с notebook для осложнений инфаркта миокарда: это
   самый короткий полный маршрут от данных до отчёта. Notebook PTB-XL показывает
   интерфейсы графа из 12 отведений ЭКГ, MIMIC — временные ограничения EHR, а
   дополнительный brain-пример — прозрачную incidence-матрицу гиперграфа.
2. **Откройте notebook в Google Colab.** Используйте ссылку в таблице ниже и
   выполняйте ячейки сверху вниз. Первая исполняемая ячейка клонирует репозиторий
   и устанавливает пакет `healthcare_gnn`. Основной код находится в `src/`, а не
   скрыт в notebook.
3. **Сначала запустите synthetic smoke mode.** Генерируются детерминированные
   искусственные записи. Они разделяются по идентификатору пациента. Imputer и
   scaler обучаются только на train. Рёбра строятся из признаков без меток.
4. **Проверьте JSON-отчёт.** Поле `status` должно быть равно
   `synthetic_smoke_not_clinical`. Просмотрите размеры train/validation/test,
   свойства графа, порог, выбранный на validation, метрики test, edge-drop audit,
   seed, кривизну, снимок конфигурации и версии пакетов.
5. **Только после этого готовьте реальные данные.** Проверьте лицензию, момент
   прогноза, целевую переменную, доступность каждого признака к этому моменту и
   группировку записей одного пациента. Пример full-конфигурации специально не
   запускается автоматически: сначала нужен независимый аудит данных.

### Что происходит внутри запуска

```text
конфигурация
    ↓
разделение train / validation / test по пациентам
    ↓
imputation и scaling только по train
    ↓
построение графа без использования меток
    ↓
baseline + евклидова GNN + geometry-aware модель
    ↓
выбор checkpoint и порога по validation
    ↓
финальные test-метрики + perturbation audit + JSON-отчёт
```

Logistic regression получает те же признаки, преобразованные только по train,
что и графовые модели. При оптимизации GNN используются только рёбра между
train-вершинами. Validation выбирает checkpoint и порог решения. Test-метки
используются только для финального отчёта. Для полноценного исследования всё
равно нужны nested tuning, несколько seed, калибровка, доверительные интервалы,
анализ подгрупп и проверенный словарь данных.

### Как понимать результат

- **Baseline лучше:** прямо сообщите, что при данном протоколе граф не помог.
- **GNN лучше:** проверьте, сохраняется ли эффект после rewiring/edge dropping,
  перемешивания признаков, повторов с разными seed и парного bootstrap по
  пациентам.
- **Гиперболическая модель лучше:** дополнительно сравните число параметров,
  чувствительность к кривизне, устойчивость градиентов, распределение радиусов и
  евклидовы контрольные модели.
- **Результат smoke run:** не делайте медицинских выводов независимо от значения
  метрик. Искусственные данные проверяют только программный pipeline.

### Типичные ошибки

| Сообщение или симптом | Что это означает | Действие |
|---|---|---|
| `Strict inductive mode refuses preprocessing fit on the full cohort` | Не выделены строки train. | Передайте train-индексы; не отключайте strict mode для оценки обобщения. |
| `MIMIC files are absent` | Локальные credentialed-файлы не найдены. | Получите разрешение PhysioNet и загрузите их вручную. Проект не загружает full MIMIC автоматически. |
| Colab не импортирует `healthcare_gnn` | Не завершилась установка или выбрана неверная папка. | Повторите setup и проверьте путь `/content/ML-MachineLearning-Graphs/healthcare-gnn-lab`. |
| Метрика подозрительно высокая | Возможна утечка, дубликаты пациентов или target-derived признаки. | Остановите интерпретацию и проверьте ID, время, происхождение признаков, рёбра и fitting по split. |
| Результат GNN меняется | Возможна недетерминированная операция или другое окружение. | Сравните версии и устройство из отчёта, зафиксируйте seed и повторите эксперимент с несколькими seed. |

## Install and run

```bash
cd healthcare-gnn-lab
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,data]'
pytest
healthcare-gnn --config configs/mi_smoke.json --output reports/mi_smoke.json
```

Open the notebooks in Google Colab:

| Example | Colab | Implemented smoke path |
|---|---|---|
| MI patient similarity | [Open](https://colab.research.google.com/github/D2718281828nis/ML-MachineLearning-Graphs/blob/main/healthcare-gnn-lab/notebooks/01_patient_similarity_mi.ipynb) | Logistic vs GCN vs Poincaré adaptation on synthetic patient nodes |
| PTB-XL lead graph | [Open](https://colab.research.google.com/github/D2718281828nis/ML-MachineLearning-Graphs/blob/main/healthcare-gnn-lab/notebooks/02_ptbxl_ecg_graph.ipynb) | Shared runner plus tested 12-lead adjacency/encoder components |
| MIMIC EHR graph | [Open](https://colab.research.google.com/github/D2718281828nis/ML-MachineLearning-Graphs/blob/main/healthcare-gnn-lab/notebooks/03_mimic_ehr_graph.ipynb) | Shared runner plus tested temporal/phenotype guards |
| Optional brain hypergraph | [Open](https://colab.research.google.com/github/D2718281828nis/ML-MachineLearning-Graphs/blob/main/healthcare-gnn-lab/notebooks/04_brain_hypergraph_optional.ipynb) | Shared runner plus transparent tested incidence operator |

A clean Colab run is not yet recorded. Each notebook states that limitation.

## Repository tree

```text
healthcare-gnn-lab/
├── README.md, pyproject.toml, .gitignore
├── configs/                    # four smoke configs + reviewed full-data template
├── notebooks/                  # four thin Colab entry points
├── reports/.gitkeep            # generated JSON/CSV/plots are ignored
├── src/healthcare_gnn/
│   ├── cli.py
│   ├── data/                   # adapters, splits, EHR time guards, fixtures
│   ├── graphs/                 # graph API, diagnostics, ECG, hypergraph
│   ├── models/                 # baselines, PyG GNNs, ECG CNN, Poincaré ops
│   ├── training/               # shared smoke runner
│   ├── evaluation/             # metrics, validation threshold, bootstrap CI
│   └── explainability/         # interpretation policy
└── tests/                      # safety, graph, geometry, evaluation, smoke tests
```

## Dataset access and leakage policy

| Dataset | Access | Adapter state | Required full-data controls |
|---|---|---|---|
| [UCI MI Complications](https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications) | Public; source terms apply | Local CSV loader, checksum manifest, target validation | Inspect official dictionary; publish prevalence; choose target explicitly; admission-only columns by default; day-3 mode only after timing review |
| [PTB-XL](https://physionet.org/content/ptb-xl/) | PhysioNet terms | Lead graph and shared temporal encoder implemented; download orchestration not yet implemented | Official folds and patient grouping; declared sampling/filtering/normalization; missing-lead policy; train/validation-selected loss and thresholds |
| [MIMIC-IV Demo](https://physionet.org/content/mimic-iv-demo/) | Open demo | Target fallback, HF ICD prefixes, two-visit and pre-index guards | Demo validates pipeline only; report chosen fallback and positive counts |
| [MIMIC-IV](https://physionet.org/content/mimiciv/) | Credentialed; never auto-downloaded | Local-path refusal with access instructions | DUA, exact index/horizon, patient split, train-only vocab/embeddings, repeated-seed CI |
| [ABIDE I/II](https://fcon_1000.projects.nitrc.org/indi/abide/) | Source terms apply | Incidence operator and adapter specification only | This is an **fMRI adaptation** of an EEG idea; site-aware or leave-one-site-out evaluation and train-only harmonization |

Raw clinical data, caches, predictions, checkpoints, and generated reports are
ignored. `write_manifest` records source URLs and SHA-256 hashes for manually
reviewed public downloads. Credentialed data are never downloaded automatically.

## Shared graph API

`build_graph(X, method, metric, k, threshold, fit_indices, ...)` returns typed
edges, weights, transformed features, fit indices, and construction metadata.
Labels are not accepted by the function. In strict inductive mode:

- fitting on the whole cohort is rejected;
- imputation, ranges, and scaling use training rows only;
- validation/test nodes choose neighbors only from the training reference set;
- mutual-KNN or union symmetrization is explicit;
- self-loop policy is explicit;
- Gower-like range-normalized distance is available for mixed-scale numeric
  clinical features; categorical encoding still requires a reviewed adapter;
- Poincaré KNN refuses unvalidated coordinates rather than silently treating
  arbitrary clinical features as ball embeddings.

`graph_diagnostics` reports density, isolated nodes, connected components,
mean/max degree, edge homophily, and label mixing. Edge drop and feature shuffle
are provided for graph-value audits. `k` and thresholds must be tuned inside
nested validation in full experiments; smoke configs do not establish them.

## Models and comparison contract

| Family | Implemented | Scientific status |
|---|---|---|
| Non-graph | class-balanced logistic regression, random forest, MLP | Baselines; same train-fitted features |
| Euclidean graph | GCN, GraphSAGE, GAT, TransformerConv | Node-classification components; task-specific full runs pending |
| ECG | shared per-lead 1-D CNN; physiological/functional/combined adjacency | ECG adaptation, not a cardiac-surface reproduction |
| Hyperbolic | Poincaré exp/log, projection, Möbius addition/distance; tangent-aggregation GCN with learned positive curvature | Geometry-aware adaptation, not a fully intrinsic paper reproduction |
| Hypergraph | explicit incidence matrix and normalized propagation operator | Synthetic unit tested; ABIDE adapter pending data review |

The hyperbolic model maps features into the ball, returns to the origin tangent
space for GCN aggregation, projects after manifold operations, and maps to the
tangent space for prediction. This design is deliberately named a
**tangent-aggregation Poincaré adaptation**. It is not evidence that hyperbolic
geometry helps, and it is parameter-matched only within the smoke runner's fixed
hidden width. Full studies must compare exact trainable parameter counts.

## Quick versus full experiment matrix

| Task | Quick command | Full experiment requirements | Current claim |
|---|---|---|---|
| MI complication | `healthcare-gnn --config configs/mi_smoke.json` | Dictionary/timing audit, prevalence-selected target, modality ablations, nested `k`, GNN explainers | Synthetic pipeline only |
| PTB-XL diagnosis/age | `... configs/ptbxl_smoke.json` | Official folds, signal loader, multi-label metrics/calibration, age MAE/RMSE/R²/Spearman/decade MAE | Components only; no PTB-XL result |
| MIMIC next-visit HF/fallback | `... configs/mimic_smoke.json` | Reviewed extraction, exact horizon, demo prevalence fallback, heterogeneous view, loss ablations | Guards only; no MIMIC result |
| Brain ASD | `... configs/brain_smoke.json` | ABIDE connectivity derivatives, site-aware split, train-only threshold/harmonization | Synthetic hypergraph plumbing only |

The full-data template is `configs/full.example.json`; it intentionally cannot be
run without completing task-specific review. Missing credentialed-data work is
not disguised with fabricated labels or records.

## Evaluation and output schema

A classification report includes AUPRC (primary), AUROC, F1, sensitivity,
specificity, precision, balanced accuracy, confusion matrix, Brier score, the
validation-selected threshold, and a patient-cluster bootstrap AUPRC interval.
Full experiments require at least three seeds, paired patient-level bootstrap
differences, calibration error/plots, learning curves, and subgroup counts plus
uncertainty. Test data are touched once per final run.

Generated JSON follows:

```json
{
  "schema_version": "1.0",
  "status": "synthetic_smoke_not_clinical",
  "example": "mi",
  "seed": 17,
  "versions": {},
  "split": {"unit": "patient", "train": 108, "validation": 36, "test": 36},
  "graph": {},
  "edge_drop_audit": {},
  "models": [{"model": "logistic", "threshold_selected_on": "validation", "metrics": {}}],
  "prediction_schema": ["deidentified_record_id", "split", "label", "probability", "seed"]
}
```

No example metric table or plot is committed because no compatible environment
was available for an actual run. Inventing an “expected plot” would violate the
result-labeling policy. Once run, plots must be generated from saved deidentified
predictions and labeled **synthetic smoke**, **local reproduction**, or
**paper-reported**, never mixed.

## Paper-to-code traceability

| Source | Implemented design choice | Relationship |
|---|---|---|
| *Cardiac age prediction using graph neural networks*, DOI [10.1101/2023.04.19.23287590](https://doi.org/10.1101/2023.04.19.23287590) | Shared temporal lead encoder and regression experiment plan | **Adaptation:** ECG leads are not a moving cardiac surface; output must be “ECG-predicted age” |
| *GNNs for Heart Failure Prediction on an EHR-Based Patient Similarity Graph*, DOI [10.5753/sbcas_estendido.2025.7013](https://doi.org/10.5753/sbcas_estendido.2025.7013) | Patient KNN view, GCN/SAGE/GAT/Transformer choices, focal-loss research plan, AUPRC | **Adaptation**, not reproduction; this code has not run the paper's MIMIC-III cohort |
| *A Novel Prediction Model for Multimodal Medical Data Based on GNNs*, DOI [10.3390/make7030092](https://doi.org/10.3390/make7030092) | Patient nodes, modality-aware similarity plan, GraphSAGE comparison | **Adaptation**; UCI target/timing still requires official dictionary review |
| *Variationally Regularized Graph-based Representation Learning for EHRs*, arXiv:1912.03761 | Train-only concept-relation and optional KL/annealing research plan | **Not implemented yet**; listed under extensions rather than implied |
| 2025 ASD multilayer hypergraph manuscript | Transparent incidence operator and multilayer ABIDE plan | **fMRI adaptation**; reported 81%/57% belongs only to that manuscript/cohort |
| *EEGraph*, DOI [10.1016/j.neucom.2022.11.050](https://doi.org/10.1016/j.neucom.2022.11.050) | Electrode nodes, connectivity edges, threshold modes | **Methodological adaptation**, not package reproduction |
| Repository hyperbolic theory corpus | Curvature, exp/log, projection, distance, geometry diagnostics | Tested mathematical subset; contrastive, Geometry Score, IMD, and GW remain research extensions |

There are **no exact paper reproductions** in this increment and no paper metric
is presented as a local result.

## Explainability and research extensions

Permutation importance is implemented for tabular smoke baselines. Attention is
never described as causal. Captum/PyG node-edge masks, lead/time attribution,
nearest-neighbor clinical summaries, calibration plots, radius distributions,
distortion/stress, hyperbolic contrastive pretraining, VGNN KL annealing,
heterogeneous MIMIC graphs, Geometry Score, intrinsic-distance evaluation, and
split-safe GW cohort alignment remain research extensions. They must not operate
on the full cohort before splitting.

## What this project does not prove

- It does not diagnose, recommend treatment, or demonstrate clinical readiness.
- It does not reproduce any cited headline result.
- Synthetic smoke metrics do not predict real-dataset performance.
- A graph derived from human-designed lead relationships is not a learned
  physiological mechanism.
- Attention or attribution is not causality.
- A lower loss or attractive embedding does not establish graph or hyperbolic
  value. If a strong baseline wins, that result must be reported plainly.
- ABIDE fMRI is not equivalent to the cited EEG multilayer construction.
- “ECG-predicted age” is not biological or cardiac age without a normative
  cohort protocol and external validation.

## Final review checklist

- [x] Patient-level split primitive and overlap test.
- [x] Training-only preprocessing and strict inductive graph test.
- [x] EHR event-before-index guard and explicit demo target fallback.
- [x] Deterministic graph, symmetry, self-loop, isolation, and hypergraph tests.
- [x] Hyperbolic identity, symmetry, round-trip, projection, and finite-gradient tests.
- [x] Validation-selected classification threshold and patient bootstrap CI.
- [x] Seeds and package versions recorded in reports.
- [x] Raw/credentialed data excluded and MIMIC auto-download refused.
- [ ] Clean Google Colab run recorded.
- [ ] Real-data adapters and prediction-time dictionaries independently reviewed.
- [ ] Full calibration, paired uncertainty, learning curves, and subgroup analyses run.
- [ ] Clinical, privacy, site-shift, missingness, and fairness review completed.
