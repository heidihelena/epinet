# EpiNet — federated pipeline

Companion to the [README](../README.md) and the
[governance note](governance-and-consent.md). This describes how EpiNet's
standardization and contestability spine is reconstructed from **per-site
aggregates only**, so records never leave a site — and how the governance gate
makes that egress mandatory and disclosed.

All claims below are demonstrated by runnable scripts under `examples/` and
pinned by the test suite. This is a research demonstrator; see
[Scope and caveats](../README.md#scope-and-caveats).

## Why it federates

EpiNet's nearest-centroid / contestability spine is built from **additive
sufficient statistics**:

- the global z-score scaler from per-feature `count`, `sum`, and the per-site
  **centered** sum-of-squared-deviations `m2`;
- the class centroids from per-class `count` and `sum` (z-scoring is affine, so a
  standardized centroid is `(raw_class_mean − global_mean) / global_sd`);
- the Mahalanobis precision from per-site **centered** co-moment matrices
  `Σ (x−μ)(x−μ)ᵀ`.

Sums are additive and the centered moments combine across sites by the parallel
(Chan et al.) update, so a coordinator combines one small aggregate message per
site and recovers **exactly** what a centralized run would compute. Centering
each site's moments *before* they cross avoids the catastrophic cancellation of
`Σx² ⁄ n − μ²` — without it the reconstructed standard deviation drifts by
~1e-4 on a feature whose mean dwarfs its spread; with it the fit matches the
centralized scaler to floating-point precision. The RandomForest
outcome model is *not* mean-poolable (trees cannot be averaged) and is out of
scope — the federatable part is precisely the centroid/contestability spine,
EpiNet's differentiated lens. This is the
[DataSHIELD](https://doi.org/10.1093/ije/dyu188) "take the analysis to the data"
principle, extended to the contestability analytic.

## The two stages

**1. Federated fit** (`epinet_federated.site_aggregates` → `combine_aggregates`).
Per-site aggregates combine into the shared scaler, centroids, and Mahalanobis
precision. On the bundled synthetic cohort, split across two sites, the
reconstructed scaler and centroids match a centralized run to ~1e-14, and the
Mahalanobis flip-distance to ~8e-10.
Demo: `examples/federated_two_site_demo.py`.

**2. Federated contestability** (`site_contestability` → `combine_contestability`).
Given the pooled fit, each site computes flip-distance **locally** against the
global centroids; the per-node scores match a centralized run exactly. Only a
de-identified summary crosses (flip-distance count/sum/sumsq/min/max + a
shared-bin histogram, runner-up class counts, a per-feature value-of-information
sum, and nearest-centroid agreement). The contested-threshold is the one
approximate piece (a histogram quantile; the contested *count* is `q·N` by
definition). Demo: `examples/federated_contestability_demo.py`.

## Mandatory, disclosed egress

The cross-boundary object is **sealed**. `contribute_aggregate()` and
`contribute_contestability()` return a `SiteContribution` that is not
JSON-serializable; the only way to obtain a shippable payload is
`.disclose(policy, consent)`, which runs the governance gate
(`epinet_governance.check_egress`). The coordinator's `combine_*` functions accept
the disclosed payloads. Raw `site_aggregates`/`site_contestability` remain the
in-boundary computation layer (used for the two-site simulation and local
scoring), which never cross a trust boundary.

This closes the "forgot to call the gate" gap for **accidental** egress. It does
not stop a determined caller from reaching a private attribute — true
non-repudiation would need signing, which the audit ledger deliberately does not
claim. Full governance details, including what the gate enforces versus what
remains a policy/legal responsibility, are in
[governance-and-consent.md](governance-and-consent.md). Demo of the full governed
round: `examples/federated_governed_round_demo.py`.

## The registry on-ramp

`epinet_registry.adapt()` maps a flat clinical / registry / MDT export (one row
per case, coded columns) onto canonical nodes and edges, driven by a declarative
`RegistryProfile` (id/outcome columns, numeric features, edge strategy). It
**formats only** — no risk, stage, or treatment is computed, and no clinical
decision is made; every run emits a manifest and a source hash. Crucially, the
same profile applied at every site is the **shared feature contract** that makes
the per-site tables column-compatible — the precondition the federated fit needs.
Demo: `examples/registry_adapter_demo.py`.

## Honest limits

- Reconstruction is exact for the **empirical** covariance; matching
  production's Ledoit–Wolf shrinkage exactly would need 4th-moment aggregates
  (noted, not built).
- The federation assumes a **shared feature contract**; whether a graph feature
  is *comparable* across differently-structured sites is a modelling question,
  not a math one.
- Sealing prevents accidental egress, not a determined caller; the audit ledger
  is tamper-evident, not signed.
- Small-cell suppression is a configurable threshold, not a formal privacy
  guarantee — set it, and review egress, with a DPO (see the governance note).
