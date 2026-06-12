# Contributing to EpiNet

Thanks for your interest in EpiNet — the Epistemic Network toolkit. EpiNet is a
research and education demonstrator, and contributions that keep it honest,
inspectable, and conservatively evaluated are especially welcome.

## Reporting issues

Open a GitHub issue (check for an existing one first). Three kinds of report are
particularly valuable:

- **Bugs** — include the command or call, the input shape, and the full error.
- **Scientific errors** — if a metric, a statistical identity (e.g. the
  closed-form flip-distance or the additive reconstruction of the scaler and
  centroids), or a methodological claim looks wrong, please say so. A minimal
  reproduction and the expected versus observed value help most.
- **Governance / security / privacy concerns** — if you spot a way the governance
  gate could disclose more than intended (e.g. small-cell leakage, an unhandled
  payload shape), please report it. For sensitive disclosures, mark the issue
  accordingly or contact the maintainer directly rather than posting details
  publicly.

## Running the examples and tests

```bash
pip install -e ".[dev]"
python -m unittest discover -s tests      # or: pytest  (adds the hypothesis property tests)
ruff check .
```

The runnable demonstrations under `examples/` (federated, governance, registry,
nodule, lymphoma, baselines, external validation, …) are the quickest way to see
expected behaviour and outputs; each prints what it checked.

## Making changes

1. Fork and branch.
2. Make the change with clear, descriptive commits.
3. **Add or update tests.** New analytic behaviour should be pinned by a test —
   ideally against a known closed form or a centralized reference, in the style of
   the existing suite.
4. Run the tests and `ruff check .`; keep CI green (Python 3.10–3.12).
5. Update the relevant docs (`README.md`, `docs/*`) when behaviour or interfaces
   change.
6. Open a pull request describing the change and how you verified it.

## What counts as acceptable validation

EpiNet's value is its conservatism, so new claims must be earned:

- numerical results checked against a closed form, a centralized computation, or
  an independent implementation where possible;
- evaluation that does not leak (use the leakage-aware splitting and the
  permutation null);
- honest scope — keep the "research demonstrator, not clinical decision support"
  framing, and do not add wording that implies regulatory compliance.

## Conduct

By participating you agree to keep the project a welcoming, respectful space.

## Questions?

Open an issue or reach out to the maintainer. Thank you for contributing.
