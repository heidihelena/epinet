# Sample outputs (frozen)

A representative `epinet` run on the bundled synthetic cohort, committed so you
can see what EpiNet produces without running it:

```bash
epinet --nodes synthetic_nodes.csv --edges synthetic_edges.csv \
       --output-dir epinet_outputs --n-iterations 10 --permutation-test 50
```

- [`model_card.md`](model_card.md) — TRIPOD+AI-flavoured model card.
- [`model_metrics.json`](model_metrics.json) — discrimination, classification,
  calibration, bootstrap CI, permutation test, data warnings, provenance.
- [`run_summary.json`](run_summary.json) — full run summary.
- `plots/` — calibration reliability diagram, learning curve, network overview,
  and the permutation-null distribution.

**Note.** The synthetic cohort has no real signal by construction, so this run
*correctly* shows near-chance discrimination, non-significant permutation
p-values, and poor calibration — the toolkit exposing weakness rather than
hiding it. The `provenance` block records the exact commit, package versions,
and input hashes for this snapshot.
