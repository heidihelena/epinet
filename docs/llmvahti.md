# LLMvahti (experimental): a blinded-second-rater audit for LLM judges

> **Status: experimental.** This module is a beta-quality research demonstrator
> of EpiNet's honest-evaluation stance applied to LLM-as-judge pipelines. It is
> not part of the v0.4.0 citation snapshot's core claims, not a benchmark, and
> not a substitute for human review. The market context that motivates it is in
> [`llmvahti-gap-analysis.md`](llmvahti-gap-analysis.md).

## The stance

Most LLM-evaluation tools treat the LLM judge as the authority and the human as
an optional calibration step. LLMvahti inverts that, the same way
[citevahti](https://github.com/heidihelena/citevahti) does for citations:

- the **human is the primary rater** and rates first,
- the **LLM judge is a second rater**, rated blind to the human and compared
  afterwards with standard inter-rater statistics, and
- every judge verdict carries a **contestability** reading — how small a move
  in rubric-criterion space would flip it, and which criterion it hinges on.

The evidence that this is needed — judge self-preference bias, overconfidence,
and agreement that collapses to 64–68% exactly in expert domains — is collected,
verified, and cited in the [gap analysis](llmvahti-gap-analysis.md).

## What `epinet.llmvahti` does

Given two CSVs:

| file | required columns | optional columns |
|---|---|---|
| `human.csv` | `item_id`, `human_label` | — |
| `judge.csv` | `item_id`, `judge_label` | `judge_confidence` ∈ [0, 1], any numeric `criterion_*` rubric scores, any categorical `group_*` strata |

from the command line:

```bash
epinet-llmvahti --human human.csv --judge judge.csv --output-dir out/
# reproducible CIs: --n-boot 1000 --random-state 0 (defaults); --n-boot 0 to skip
```

or equivalently from Python:

```python
from epinet import llmvahti as elv

results = elv.run_blinded_audit("human.csv", "judge.csv", "out/")
```

(`python -m epinet.llmvahti ...` works identically without installing the
console script.) Either path produces `judge_audit.md`, `judge_audit.json`, and
per-verdict `verdict_assignments.csv` containing:

1. **Blinded protocol enforcement.** The human ratings are sealed
   (SHA-256-hashed) before judge ratings are accepted; the `BlindedAudit`
   object refuses judge input before the seal and closes after results. The
   seal makes the ordering tamper-evident *within the run*; keeping the human
   genuinely unexposed to judge output is a process responsibility, stated in
   the report rather than claimed by the software.
2. **Inter-rater agreement** — raw agreement, Cohen's kappa, two-rater nominal
   Krippendorff's alpha, and the full confusion matrix, each reported with a
   seeded percentile-bootstrap confidence interval (`n_boot`/`random_state`
   arguments; skipped below 10 jointly-rated items, where an interval would be
   more noise than signal). The interval is the point: on the tens-of-items
   golden sets these audits run on, a bare kappa is badly under-determined — a
   κ of 0.47 on 45 items can span "fair" to "substantial", and the report shows
   that. Agreement is with the *human standard* by design; the audit says so in
   its caveats.
3. **Judge calibration** — when `judge_confidence` is present, it is scored
   against being right by the human standard: Brier score and the same Cox
   weak-calibration slope/intercept the outcome-model report uses
   (slope < 1 = overconfident judge), each with a seeded percentile-bootstrap
   confidence interval (same `n_boot`/`random_state` and small-sample handling
   as the agreement metrics).
4. **Verdict contestability** — `epinet.contest`'s exact nearest-centroid
   flip-distance, pointed at the judge's verdicts in `criterion_*` space:
   per-verdict flip-distance, the contested grey zone (lowest-decile by
   default), criterion-level value-of-information ("which rubric criterion
   drives verdict flips"), and the headline table — verdicts that are **both
   contested and human-disagreeing**, i.e. the calls to re-review first.

5. **Subgroup error funnel** — for each categorical `group_*` column, an
   exploratory differential-error screen: per-stratum judge-vs-human
   disagreement rates against funnel control limits around the pooled rate at
   each stratum's size (the quality-indicator funnel, pointed at the judge).
   Strata outside the outer 99.8% limit flag `high`/`low`; inner 95%
   excursions are reported but not flagged, since with many strata some are
   expected by chance. A flag says "unusual given size" — it is not proof of
   causal bias, and it inherits the limitations of the human standard.

## Design notes and limits

- **Reuse over reinvention.** The contestability layer is `epinet.contest`
  unchanged — the same closed-form Rocchio flip-distance, with the same
  additive sufficient statistics. That means the federated reconstruction
  documented in [`federated.md`](federated.md) applies to judge audits too:
  multi-site judge auditing without pooling item-level data is the same math.
- **Contestability is over the rubric, not the text.** Flip-distance lives in
  the space of the judge's numeric criterion scores. If the judge emits only a
  label, you still get the agreement and calibration audit, but no
  contestability lens. Embedding-space contestability (scoring prompt/response
  embeddings directly) is future work.
- **The human standard is the standard.** Where human ratings are themselves
  uncertain or single-rater, the audit inherits that uncertainty. Multi-human
  panels (alpha over >2 raters) are future work.
- **Fail-closed posture.** Protocol violations (judge before seal, duplicate
  item ids, missing confidences, NaN criteria) raise rather than degrade,
  matching the governance gate's behaviour elsewhere in the toolkit.
