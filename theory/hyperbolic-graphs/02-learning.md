# 2. Embeddings and hyperbolic GNNs / Эмбеддинги и гиперболические GNN

## English

### Hyperbolic embeddings

An embedding assigns each node `i` a point `z_i` inside a hyperbolic manifold.
A common objective makes connected or semantically related nodes close under
hyperbolic distance and pushes negative samples farther away. Evaluation should
match the task: link prediction, reconstruction distortion, ranking, or
downstream classification.

### From a GNN to a hyperbolic GNN

Ordinary vector addition is not globally valid on a curved manifold. A typical
layer therefore:

1. maps node states to a tangent space with a logarithmic map;
2. performs linear transformation and neighbor aggregation there;
3. maps the result back with an exponential map;
4. projects points safely inside the ball.

Some architectures instead use Möbius operations directly. These are design
choices, not interchangeable notation. Curvature may be fixed or learned.

### Practical checklist

- Use double precision when points approach the boundary.
- Clamp denominators and inverse-hyperbolic-function arguments.
- Project after optimizer updates and manifold operations.
- Use a Riemannian optimizer for manifold-valued parameters.
- Report curvature, tangent-space choice, negative sampling, and seeds.
- Test on a tiny tree before training on real data.

## Русский

### Гиперболические эмбеддинги

Эмбеддинг сопоставляет каждой вершине `i` точку `z_i` внутри гиперболического
многообразия. Типичная целевая функция сближает связанные или семантически
похожие вершины по гиперболическому расстоянию и удаляет отрицательные примеры.
Метрика оценки должна соответствовать задаче: предсказание рёбер, искажение
реконструкции, ранжирование или последующая классификация.

### От GNN к гиперболической GNN

Обычное сложение векторов не определено глобально на искривлённом
многообразии. Поэтому типичный слой:

1. переводит состояния вершин в касательное пространство логарифмическим
   отображением;
2. выполняет там линейное преобразование и агрегацию соседей;
3. возвращает результат экспоненциальным отображением;
4. безопасно проецирует точки внутрь шара.

Некоторые архитектуры вместо этого напрямую используют операции Мёбиуса. Это
разные проектные решения, а не взаимозаменяемые обозначения. Кривизну можно
зафиксировать или обучать.

### Практический чек-лист

- Используйте двойную точность, если точки приближаются к границе.
- Ограничивайте знаменатели и аргументы обратных гиперболических функций.
- Выполняйте проекцию после шагов оптимизатора и операций на многообразии.
- Для параметров на многообразии используйте риманов оптимизатор.
- Указывайте кривизну, касательное пространство, negative sampling и seed.
- До реальных данных проверяйте реализацию на маленьком дереве.

## Run it / Запуск

The [hyperbolic graph basics notebook](../../examples/hyperbolic_graph_basics.ipynb)
turns the distance formula into a small executable experiment.
