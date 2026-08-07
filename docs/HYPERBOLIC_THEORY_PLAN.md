# Hyperbolic theory editorial plan

**Status:** first reviewable increment, 2026-08-07. This plan separates what was
observed in the repository from what is supported by the supplied bibliography.
The paper PDFs and two local lecture files are **not present in this checkout**.
Network access was blocked during identifier verification. Exact paper equation,
page, and slide locators are therefore marked for manual verification rather
than guessed.

## 1. Severity-ranked gap analysis

| Severity | Gap | Evidence in this checkout | Action |
|---|---|---|---|
| Critical | Every Colab/clone link named the former `TAUforPython/Graph-MachineLearning` repository. | README and first cells of all root notebooks | Point repository-owned links to `D2718281828nis/ML-MachineLearning-Graphs`; do not rewrite historical outputs. |
| Critical | The notes gave formulas without paper-level provenance or an explicit convention. | Old `01-foundations.md` used `-c`, but only defined the unit-ball distance. | Declare curvature `K=-c`, `c>0`, ball radius `1/sqrt(c)`, conformal factor, scaling, domains, and source status. |
| Critical | The curated notebook covered only `c=1`, drew chords rather than geodesics, and did not test maps or limiting behavior. | `examples/hyperbolic_graph_basics.ipynb` | Implement general `c`, true geodesic interpolation, projection, origin exp/log maps, and deterministic tests. |
| High | No evidence matrix, notation concordance, or distinction between source, code, and editorial inference. | Entire theory folder | Add these controls before expanding claims. |
| High | Lorentz and Klein models, conversions, learning operations, and numerical safeguards were missing. | Theory index contained only foundations and a short learning note. | Add a models-and-maps module first; plan later modules without pretending they exist. |
| High | Research notebooks are linked next to curated examples despite unresolved data paths and saved failures. | `docs/NOTEBOOK_REVIEW.md` | Preserve the existing maturity warnings and label implementation mappings precisely. |
| Medium | No prerequisites, exercises, bilingual glossary, evaluation/GW separation, or limitations module. | Theory folder | Add in the staged outline below. |
| Medium | The supplied lecture files are absent, so slide citations cannot be checked. | Filesystem audit | Do not cite their contents until files are supplied and inspected. |

## 2. Evidence matrix

Labels: **S** = source-supported, **R** = repository implementation, **E** =
editorial inference. “Pending” means that a source locator must be checked in the
paper itself before the claim is promoted to **S**.

| Concept | Exact claim | Source and locator | Assumptions / convention | Target | Related code | Label / confidence |
|---|---|---|---|---|---|---|
| Negative curvature | Exponential metric-volume growth gives a geometric capacity compatible with exponential tree branching; this is an inductive bias, not universal superiority. | Yang et al. (2025), locator pending PDF check | Constant negative curvature; qualitative motivation | `01-foundations.md` | No volume implementation | E, high for geometry; medium for modeling interpretation |
| Poincaré domain | For `K=-c`, `c>0`, `D_c^n={x in R^n: c||x||^2<1}`. | Yang et al. (2025), exact equation pending | Radius `1/sqrt(c)` | `03-models-and-maps.md` | `inside_ball`, `project` | E pending source pinpoint; formula numerically tested |
| Poincaré metric | `g_x=(lambda_x^c)^2 I`, `lambda_x^c=2/(1-c||x||^2)`. | Yang et al. (2025), exact equation pending | Conformal convention with Euclidean-limit distance `2||u-v||` | `03-models-and-maps.md` | `poincare_distance` | E pending source pinpoint; high algebraic confidence |
| Distance | The arcosh expression and Möbius-norm expression are equivalent inside the ball. | Yang et al. (2025), exact equation pending | Same `-c` convention; finite interior points | `03-models-and-maps.md` | `poincare_distance` | E pending source pinpoint; tested numerically |
| Origin exp/log | `exp_0^c` and `log_0^c` map between `T_0 D_c^n` and `D_c^n` and are inverses away from numerical clipping. | Yang et al. (2025), exact equation pending | Tangent vectors use Euclidean coordinates; `lambda_0=2` | `03-models-and-maps.md` | `expmap0`, `logmap0` | E pending source pinpoint; round trip tested |
| Community methods | Gerald et al. use hyperbolic community representations and Riemannian clustering machinery. | Gerald et al. (2020), exact section/equation pending | Paper-specific method and experiments | Future `02-learning.md` revision | Missing | S at title/abstract scope only; details blocked |
| Hyperbolic contrastive learning | Poincaré geometry is used in a contrastive-learning method. | Yue et al. (2023), exact objective/experiment locators pending | Paper-specific scope | Future `06-contrastive-and-generative.md` | Missing | S at title/abstract scope only |
| Hyperbolic graph generation | GGBall is an advanced Poincaré-ball graph-generation case study, not a general foundation. | Bu et al. (2026), exact architecture locators pending | Paper-specific architecture | Future `06-contrastive-and-generative.md` | Missing | E framing; paper details blocked |
| Geometry Score | Geometry Score compares generated and real distributions using topological summaries. | Khrulkov & Oseledets (2018), exact definition pending | Evaluation geometry, not embedding distance | Future `07-evaluation-and-gw.md` | Missing | S at abstract scope only |
| IMD | Intrinsic multiscale comparison is distinct from distance inside one Poincaré embedding. | Tsitsulin et al. (2020), exact theorem/complexity locators pending | Distribution/graph comparison | Future `07-evaluation-and-gw.md` | Missing | E distinction, high confidence; technical claims blocked |
| GW clustering | GW learning compares relational structures through a coupling; it is not Poincaré point distance. | Chowdhury & Needham (2021), exact formulation pending | Metric/measure or relational structures | Future `07-evaluation-and-gw.md` | Root GW notebooks, unverified | S at paper scope; R notebooks need repair |
| Oversmoothing | Kuramoto dynamics may be discussed only as an adjacent dynamical view, not as hyperbolic geometry. | Supplied deck absent; slide locator unavailable | No authorship/date inferred | Future `04-hyperbolic-gnn-layers.md` | Missing | E guardrail, high confidence |

## 3. Notation and convention concordance

Only the repository convention is normative in this increment. Paper columns
remain explicitly unverified until the PDFs are available.

| Object | Repository convention | Alternative convention | Conversion / warning |
|---|---|---|---|
| Curvature | `K=-c`, `c>0` | Some works use `c<0` directly | If a source's `c_src<0`, set `c_repo=-c_src>0` before using repository formulas. |
| Ball | `D_c^n={x: c||x||^2<1}` | Unit ball with curvature absorbed into the metric | Repository radius is `1/sqrt(c)`; rescale coordinates before comparison. |
| Conformal factor | `lambda_x^c=2/(1-c||x||^2)` | Factors omitting `2` occur under different scaling | With the repository metric, `d_c(u,v) -> 2||u-v||` as `c -> 0`, not `||u-v||`. |
| Lorentz sign | Planned: `<x,y>_L=-x_0y_0+sum_i x_i y_i` | Opposite signature is common | Negate the bilinear form and adjust constraints/distance together; never mix signs. |
| Tangent norm | Euclidean coordinate norm at the origin | Riemannian norm includes `lambda_0=2` | The origin exp/log formulas below are paired under the declared coordinate convention. |
| Möbius order | `u ⊕_c v` is ordered | Möbius addition is generally not commutative | Do not swap operands in geodesic interpolation without checking the expression. |
| Paper-specific notation | **Manual verification required** | — | No paper notation is asserted from filename or memory. |

## 4. Proposed outline and old-to-new map

| New module | Scope | Old content / status |
|---|---|---|
| `00-prerequisites.md` | Graph operators, metric/manifold vocabulary | Missing; planned |
| `01-foundations.md` | Motivation, hierarchy distinctions, curvature intuition | Existing short text retained for later source-grounded rewrite |
| `02-learning.md` | Embeddings, objectives, downstream tasks | Existing short text retained; later revise around Gerald et al. |
| `03-models-and-maps.md` | Poincaré convention, operations, maps, geodesics, tests | **Added in this increment** |
| `04-hyperbolic-gnn-layers.md` | Typed layer pipeline, architecture comparison | Missing; planned |
| `05-optimization-and-numerics.md` | Gradients, optimizers, projection, test suite | Missing; planned |
| `06-contrastive-and-generative.md` | Yue et al.; GGBall case study | Missing; blocked on equation-level source review |
| `07-evaluation-and-gw.md` | Geometry Score, IMD, GW clustering | Missing; blocked on source review |
| `08-limitations.md` | Failure modes and fair baselines | Missing; planned |
| `glossary.md` | Bilingual symbols and terms | Missing; planned |
| `paper-map.md` | Bibliography, evidence state, identifiers | **Added in this increment** |

## 5. First patch scope

This increment fixes ownership links, establishes evidence and notation controls,
adds one rigorous bilingual Poincaré module, and makes the curated hyperbolic
notebook test the same convention. It intentionally does **not** rewrite the
three experimental hyperbolic notebooks or claim that they are reproducible.

## 6. Identifier verification result

Authoritative checks were attempted against DOI, arXiv, PMLR, and OpenReview on
2026-08-07. Both the browser tool and direct HTTPS requests were blocked by the
execution environment (`401 Unauthorized` and proxy `403`). The bibliography
therefore records the identifiers supplied in the task as **user-verified, not
independently re-verified in this environment**. This limitation must be removed
before calling the bibliography independently verified.

## 7. Remaining work by impact

1. **Scientific:** obtain the seven papers and two local decks; fill every exact
   section/page/equation/slide locator and resolve notation against the PDFs.
2. **Reproducibility:** run both curated notebooks in a fresh Colab CPU session
   and record runtime/package versions and a verification date.
3. **Mathematical:** add Lorentz/Klein conversions, general-point exp/log maps,
   parallel transport, and gradient checks.
4. **Graph learning:** write typed GNN, embedding, contrastive, and generative
   modules only after equation-to-source review.
5. **Evaluation:** separate embedding distance from Geometry Score, IMD, and GW
   with precise assumptions and supported complexity language.
6. **Pedagogy:** add prerequisites, bilingual glossary, tiny graph examples,
   questions, and exercises to every major concept.
