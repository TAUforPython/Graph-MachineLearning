# 1. Foundations and the Poincaré ball / Основы и шар Пуанкаре


### Why hyperbolic geometry?

A balanced tree has exponentially more nodes at each successive depth. A
Euclidean disk has only polynomially growing area, while a hyperbolic disk has
exponentially growing area. Hyperbolic space can therefore assign increasing
room to successive hierarchy levels with comparatively low distortion.

This is useful for taxonomies, social graphs, knowledge graphs, phylogenetic
trees, and other data with latent hierarchy. It is an inductive bias—not proof
that every graph should be modeled hyperbolically.

### The Poincaré ball

For curvature `-c`, where `c > 0`, the `d`-dimensional ball is

```math
\mathbb{D}_c^d = \{x \in \mathbb{R}^d : c\lVert x\rVert^2 < 1\}.
```

For the unit ball (`c = 1`), the distance between `u` and `v` is

```math
d_{\mathbb{D}}(u,v)=\mathrn{arcosh}\left(
1+2\frac{\lVert u-v\rVert^2}
{(1-\lVert u\rVert^2)(1-\lVert v\rVert^2)}\right).
```

Distances grow rapidly near the boundary. The center commonly represents a
root or a general concept; increasingly specific descendants lie closer to the
boundary. The boundary itself is not part of the model.

### Interpretation cautions

- A plotted straight segment is generally not a hyperbolic geodesic.
- Coordinates depend on curvature and on the chosen model.
- Never allow a numerical point to reach the boundary; project to radius
  `1 - ε` (or its curvature-scaled equivalent).
- Compare hyperbolic and Euclidean baselines rather than assuming an advantage.

## Русский

### Зачем нужна гиперболическая геометрия?

В сбалансированном дереве число вершин экспоненциально растёт с глубиной.
Площадь евклидова диска растёт лишь полиномиально, а площадь гиперболического —
экспоненциально. Поэтому гиперболическое пространство предоставляет всё больше
места следующим уровням иерархии при сравнительно малом искажении.

Это полезно для таксономий, социальных и онтологических графов,
филогенетических деревьев и других данных со скрытой иерархией. Это
индуктивное предположение, а не доказательство того, что любой граф следует
моделировать в гиперболическом пространстве.

### Шар Пуанкаре

При кривизне `-c`, где `c > 0`, `d`-мерный шар определяется как

```math
\mathbb{D}_c^d = \{x \in \mathbb{R}^d : c\lVert x\rVert^2 < 1\}.
```

Для единичного шара (`c = 1`) расстояние между `u` и `v` равно

```math
d_{\mathbb{D}}(u,v)=\mathrm{arcosh}\left(
1+2\frac{\lVert u-v\rVert^2}
{(1-\lVert u\rVert^2)(1-\lVert v\rVert^2)}\right).
```

Вблизи границы расстояния быстро возрастают. Центр обычно представляет корень
или общее понятие, а более конкретные потомки располагаются ближе к границе.
Сама граница модели не принадлежит.

### Важные оговорки

- Прямой отрезок на рисунке обычно не является гиперболической геодезической.
- Координаты зависят от кривизны и выбранной модели.
- Численная точка не должна достигать границы: проектируйте её на радиус
  `1 - ε` (или эквивалент с учётом кривизны).
- Сравнивайте гиперболическую модель с евклидовым baseline, а не предполагайте
  её преимущество заранее.

## Next / Далее

Continue with [embeddings and hyperbolic GNNs](02-learning.md) or run the
[Poincaré-ball example](../../examples/hyperbolic_graph_basics.ipynb).
