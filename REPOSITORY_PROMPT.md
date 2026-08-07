# Repository improvement prompt

Copy the prompt below into an AI coding assistant when maintaining this
repository. Replace bracketed values when needed.

```text
Act as a documentation engineer and graph-machine-learning maintainer.

Goal: improve the clarity and reproducibility of the Graph-MachineLearning
repository without changing the scientific meaning of existing experiments.

Tasks:
1. Audit the repository tree, README links, notebook metadata, data paths, and
   dependency-installation cells. Report broken or ambiguous items first.
2. Keep beginner, self-contained notebooks in examples/. Every example must:
   - use a descriptive snake_case filename;
   - begin with an Open in Colab badge pointing to its real GitHub path;
   - state learning objectives, prerequisites, expected runtime, and outputs;
   - install missing dependencies, set seeds, and avoid local absolute paths;
   - run from a fresh Google Colab CPU runtime using Runtime > Run all;
   - explain each mathematical formula and each major code block.
3. Maintain a concise README with: project purpose, Start Here table, Colab
   links, repository map, categorized notebook catalogue, local setup,
   bilingual theory link, contribution checklist, and license.
4. Maintain theory/hyperbolic-graphs/ in paired English and Russian sections.
   Cover intuition, definitions, equations, assumptions, numerical cautions,
   and links to executable examples. Keep mathematical notation identical in
   both languages.
5. Do not claim that a notebook works unless its JSON validates and every cell
   has been executed successfully in a clean compatible environment. Clearly
   label research notebooks that require external data or special hardware.
6. Do not commit generated caches, credentials, private data, or large outputs.
7. After editing, validate all notebook JSON, scan Markdown links, run available
   tests, and summarize changed files plus any limitations.

Constraints: preserve useful history, prefer small reviewable changes, use
relative repository links, and never silently rewrite scientific results.

Deliverables: updated files, validation commands and results, and a short list
of recommended follow-up work ordered by impact.
```
