# vahtian-epinet 0.4.3

`pip install --upgrade vahtian-epinet`

The headline of this release is a **license change**: `vahtian-epinet` is now
distributed under the **Apache License 2.0**. This is also the first release to
carry the accumulated model-backend work since `0.4.1`.

## License: MIT → Apache-2.0

From `0.4.3` onward the project is licensed under **Apache-2.0**, which keeps the
permissive spirit of MIT and adds an explicit **patent grant** and a trademark
clause. SPDX headers (`SPDX-License-Identifier: Apache-2.0`) are on every source
file, and a `NOTICE` file accompanies the `LICENSE`.

> **Earlier releases stay MIT.** `vahtian-epinet` `0.4.1` and `0.4.2` were
> published under the MIT License and remain available under those terms — a
> published version's license cannot be changed retroactively. Apache-2.0 applies
> from `0.4.3` forward.

## Selectable outcome-model backends

The honest-evaluation pipeline is no longer random-forest-only. You can now pick
the estimator (the evaluation, calibration, permutation-null, and contestability
machinery around it is unchanged):

- `random_forest` (default)
- `logistic_regression`
- `xgboost` — optional, `pip install "vahtian-epinet[xgboost]"`
- `mlp` — a small PyTorch MLP, optional, `pip install "vahtian-epinet[torch]"`

Available from the CLI (`--model`), the Python API, and the R interface
(`epinet(..., model = "...")`).

## Also in this release

- **Epistemic claim gate** — headline claims are checked against the evidence and
  downgraded when the supporting test wasn't run or didn't pass.
- **Publish workflow hardening** — the PyPI workflow now fires once per release
  and treats an already-published version as a no-op, so a release + tag no longer
  produces spurious "file already exists" failures.
- **Docs** — the JOSS paper (`paper/`) reframed around reproducible evaluation
  (not a new algorithm) with a runnable example and a permutation-null figure;
  README aligned to the same framing and the PyPI install.
- **CI** — GitHub Actions bumped to current Node-24 majors.

## Quality

- Full unittest suite green on **Python 3.10 / 3.11 / 3.12**.
- Separate `R-CMD-check` for the R interface (`DESCRIPTION`: `Apache License (>= 2)`).

## Scope

Unchanged: EpiNet is a **research and education demonstrator, not clinical or
public-health decision support.** Any model it produces must be validated on
independent, outcome-linked data before it carries clinical meaning.

## Install & cite

```bash
pip install --upgrade vahtian-epinet
```

Cite via [`CITATION.cff`](CITATION.cff).
