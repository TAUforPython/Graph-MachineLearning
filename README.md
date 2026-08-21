# Graph Machine Learning / Машинное обучение на графах

A collection of educational examples, exploratory notebooks, and a structured,
leakage-aware healthcare GNN laboratory. / Коллекция учебных примеров,
исследовательских notebooks и структурированной лаборатории healthcare GNN с
контролем утечек данных.

> **Research notice / Важно:** stored outputs in exploratory notebooks show prior
> experiments; they are not proof of a clean reproducible run or clinical
> validity. / Сохранённые результаты exploratory-notebooks не доказывают
> воспроизводимость или клиническую применимость.

## Start here / С чего начать

| Path / Раздел | English | Русский |
|---|---|---|
| [`examples/`](examples/) | Curated, small learning notebooks | Небольшие учебные notebooks |
| [`healthcare-gnn-lab/`](healthcare-gnn-lab/README.md) | Tested healthcare pipeline and task entry points | Тестируемый healthcare pipeline и входные notebooks |
| [`research-notebooks/`](research-notebooks/) | Preserved exploratory graph-ML work | Сохранённые исследовательские эксперименты |
| [`utilities/notebooks/`](utilities/notebooks/) | ERD conversion and visualization tools | Утилиты преобразования и визуализации ERD |
| [`docs/NOTEBOOK_ANALYSIS.md`](docs/NOTEBOOK_ANALYSIS.md) | What every notebook does and its limitations | Назначение и ограничения каждого notebook |
| [`theory/hyperbolic-graphs/`](theory/hyperbolic-graphs/README.md) | Bilingual hyperbolic-graph theory | Двуязычная теория гиперболических графов |

The repository root intentionally contains only notebooks whose filenames begin
with `example` or `demo`. / В корне намеренно оставлены только notebooks, имена
которых начинаются с `example` или `demo`.

## Repository map / Структура репозитория

```text
.
├── example*.ipynb              # root-level demos / корневые примеры
├── examples/                   # curated tutorials / учебные материалы
├── research-notebooks/         # non-medical exploratory work / исследования
├── utilities/notebooks/        # ERD tools / утилиты ERD
├── healthcare-gnn-lab/
│   ├── notebooks/              # numbered task entry points and explorations
│   ├── src/healthcare_gnn/     # reusable implementation
│   ├── configs/ and tests/
│   └── README.md               # bilingual lab guide / руководство
├── datasets/                   # small public example assets
├── theory/hyperbolic-graphs/   # EN/RU theory
└── scripts/audit_notebooks.py  # JSON/inventory/root-policy audit
```

## Notebook groups / Группы notebooks

### Root examples / Примеры в корне

- `example Gated Graph Attention Network.ipynb` implements a custom gated
  multi-head attention layer. / Реализует gated multi-head attention layer.
- `example_Interactive_Graph_Visualisation.ipynb` builds a small NetworkX graph
  and renders it interactively. / Строит небольшой граф NetworkX и создаёт
  интерактивную визуализацию.

### Healthcare / Здравоохранение

Medical and genomic notebooks now live under
[`healthcare-gnn-lab/notebooks/`](healthcare-gnn-lab/notebooks/). The numbered
notebooks `01`–`04` are thin entry points into tested code in `src/`; reviewed
exploratory studies are numbered `05`–`07` in the same folder. / Медицинские и
геномные notebooks находятся в `healthcare-gnn-lab/notebooks/`: notebooks
`01`–`04` используют тестируемый код из `src/`, а исследовательские работы
пронумерованы `05`–`07` в той же папке.

The exploratory genomic GCN, protein-sequence hyperbolic GNN, and vitiligo image
clustering notebooks are exploratory adaptations, **not** diagnostic tools. /
Exploratory-notebooks с genomic GCN, гиперболической GNN для белковых
последовательностей и кластеризацией изображений витилиго являются
исследовательскими примерами, **не** диагностическими средствами.

### Research and utilities / Исследования и утилиты

General graph construction, Gromov–Wasserstein, hyperbolic GNN, LLM triplet, and
classical-ML surveys are in [`research-notebooks/`](research-notebooks/). ERD to
Mermaid and ERD visualization notebooks are in
[`utilities/notebooks/`](utilities/notebooks/). / Общие эксперименты перенесены
в `research-notebooks/`, а ERD-конвертеры и визуализаторы — в
`utilities/notebooks/`.

See the [bilingual notebook analysis](docs/NOTEBOOK_ANALYSIS.md) for code flow,
data dependencies, and known blockers. / Подробный двуязычный разбор кода,
зависимостей и ограничений находится в
[анализе notebooks](docs/NOTEBOOK_ANALYSIS.md).

## Run / Запуск

For lightweight notebooks / Для простых notebooks:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install jupyter numpy matplotlib networkx
jupyter lab
```

Healthcare lab / Healthcare-лаборатория:

```bash
cd healthcare-gnn-lab
python -m pip install -e '.[test,data]'
pytest
healthcare-gnn --config configs/mi_smoke.json --output reports/mi_smoke.json
```

Audit notebook JSON and placement / Проверка JSON и расположения notebooks:

```bash
python scripts/audit_notebooks.py
```

Research notebooks have notebook-specific dependencies, external downloads, and
occasionally Colab-only APIs. Read the analysis before running them. /
Исследовательские notebooks имеют собственные зависимости, внешние загрузки и
иногда требуют Google Colab. Перед запуском прочитайте анализ.

## Contributing / Как внести вклад

1. Put only `demo*.ipynb` or `example*.ipynb` in the root; place reusable
   tutorials in `examples/`, healthcare work in `healthcare-gnn-lab/`, and tools
   in `utilities/`. / В корне оставляйте только `demo*` или `example*`.
2. Use descriptive `snake_case` filenames for new notebooks. / Используйте
   понятные имена в `snake_case`.
3. State objectives, prerequisites, expected runtime, data provenance, and
   verification status. / Укажите цели, зависимости, время, происхождение данных
   и статус проверки.
4. Set seeds, avoid absolute paths and secrets, and fit preprocessing on training
   data only. / Фиксируйте seed, не используйте абсолютные пути и секреты,
   обучайте preprocessing только на train.
5. Validate JSON, run relevant tests, and never claim clinical performance from
   synthetic data. / Проверяйте JSON и тесты; не делайте клинических выводов по
   synthetic-данным.

## License / Лицензия

Distributed under [`LICENSE`](LICENSE). / Распространяется на условиях
[`LICENSE`](LICENSE).
