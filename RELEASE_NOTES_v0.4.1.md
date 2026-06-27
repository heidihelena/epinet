# vahtian-epinet 0.4.1 — first PyPI release

`pip install vahtian-epinet`

EpiNet (the *Epistemic Network* toolkit) is now installable from PyPI under its
unified name. This release is mostly packaging and naming — the analysis core is
unchanged — but it's the first version you can install with one command, so the
notes below double as a short tour for new users.

## What this is — in one sentence

**One reproducible pipeline that honestly evaluates an outcome model on
graph-shaped data and emits the evidence trail to back (or retract) every
headline claim.**

Everything else in the repo — community-aware splitting, contestability scoring,
the workbench UI, the optional federation/governance layers, the R interface — is
a layer *around* that one pipeline, not a separate product.

## Statement of need

NetworkX, scikit-learn, and friends each do their part well, but a researcher
who wants a *defensible* result still has to wire together graph-feature
construction, leakage-aware evaluation, calibration, a permutation null,
provenance capture, and a write-up — by hand, differently each time. EpiNet
provides that as a single, reproducible path where **honest evaluation is the
default, not an add-on**: the label-permutation null, calibration, and (where
appropriate) community-aware splitting run alongside the headline metric, so a
good score reflects real signal rather than leakage or chance. The output is not
just a number but a publication-ready evidence bundle.

## What you get from one run

- **Model card** — metrics, calibration, and caveats in one document.
- **Provenance record** — inputs, config, and seeds, hashed for reproducibility.
- **Claims check** — each headline claim paired with the evidence (or flagged as
  unsupported).
- **HTML report** — the whole thing, publication-ready.
- **Contestability lens** — a per-call score for how contestable a prediction is
  (offered as a software feature for triage and review, not as a methodological
  claim; formal validation belongs in a separate methods paper).

## What changed in 0.4.1

- **Renamed and published.** The package is now `vahtian.epinet` (import) /
  `vahtian-epinet` (PyPI), resolving a name collision with unrelated `epinet`
  projects on PyPI and CRAN. CLI entry points (`epinet`, `epinet-workbench`,
  `epinet-llmvahti`) are unchanged.
- **Trusted Publishing.** Releases now go to PyPI via GitHub Actions OIDC (no
  API tokens).
- **R interface (`vahtian.epinet`).** A thin `reticulate` wrapper over the same
  tested Python core — single-sourced, so the algorithms cannot diverge across
  languages. Ships `fit`, `contestability`, `graph`, and `federated` surfaces
  with native R plots; its testthat suite runs green against this release.
- **Fixes.** Corrected stale package-name references left over from the rename
  (including R test guards that were silently skipping, and a README install
  line that pointed at the wrong PyPI package).

## Quality

- **197-test** unittest suite, green on **Python 3.10 / 3.11 / 3.12** in CI.
- Separate R `R-CMD-check` workflow for the R interface.
- `CITATION.cff`, `CONTRIBUTING.md`, `examples/`, `docs/`, and a JOSS `paper/`
  draft are in the repository.

## Scope

This is a **research and education demonstrator, not clinical or public-health
decision support.** Any model it produces must be validated on independent,
outcome-linked data before it carries clinical meaning. The novel ideas here
(contestability, epistemic safety) are presented as *software capabilities*;
their methodological validation is the subject of a separate Methods article, not
of this tool.

## Install & cite

```bash
pip install vahtian-epinet
```

If you use EpiNet in published work, please cite it via the repository's
[`CITATION.cff`](CITATION.cff).
