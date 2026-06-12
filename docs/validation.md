# EpiNet — baselines and external validation

Companion to the [README](../README.md) and [methods](methods.md). Two things a
reviewer of any graph-ML method asks for, and how EpiNet answers them. As
everywhere, this is a research demonstrator — see
[Scope and caveats](../README.md#scope-and-caveats).

## Baselines: do the graph features earn their place?

It is easy to report a good score without checking whether anything simpler does
just as well. `epinet_baselines.compare_representations` runs the **same honest
harness** (calibration, repeated splits, optional permutation null) across
several node representations at one shared seed, so the only thing that varies is
the representation:

- `no_information` — a constant feature (the majority-class floor);
- `graph_features` — EpiNet's degree/clustering/component features;
- `spectral_embedding` — a learned node embedding (Laplacian eigenmaps via
  scikit-learn), the representative *node-embedding* baseline;
- `graph_features+spectral` — both concatenated.

Run it: `python examples/baseline_comparison_demo.py`.

**What it surfaces, honestly.** On a *similarity* graph (edges built from
feature similarity, as in the lymphoma example), the spectral embedding recovers
the feature-space structure and classifies well, while EpiNet's graph-topology
features — degree, clustering, component size — carry little subtype signal and
sit near the floor. That is the right lesson to draw and to publish: topology
summaries are for genuinely *relational* graphs (referral networks, evidence
graphs, contact structure); for similarity graphs, an embedding or the original
features do the work. The comparison makes this visible instead of letting a
single headline number imply the graph features were essential.

**On GNNs.** A full message-passing GNN (GCN/GraphSAGE) is the heavier
comparison. It needs a deep-learning dependency the single-file toolkit
deliberately avoids, and on the small cohorts EpiNet targets GNNs tend to overfit
and rarely beat well-chosen features. The spectral embedding is the
dependency-light node-embedding baseline; a GNN baseline behind an optional
extra is reasonable future work, not a gap in the core claim.

## External validation: does the model transport?

Internal evaluation guards against leakage and chance *within one dataset*. It
cannot tell you whether a model holds up on a genuinely independent cohort —
different site, scanner, era, or case mix. That is the bar a clinical-prediction
claim must clear ([TRIPOD+AI](https://doi.org/10.1136/bmj-2023-078378)).

`epinet_validation.external_validation` fits the model on a **development**
cohort and evaluates it, untouched, on an **independent external** cohort,
reporting discrimination (AUROC, AUPRC), classification (balanced accuracy, MCC,
F1), and — for binary outcomes — calibration (Brier, slope/intercept). It reports
the external metrics next to the honest internal metrics and the **drift**
between them.

Run it: `python examples/external_validation_demo.py`.

The demo develops on one lymphoma cohort and validates on a second, independently
generated **and systematically shifted** cohort (a marker offset standing in for
a different site/scanner). The instructive result is that AUROC can stay high
(ranking is robust) while balanced accuracy and the operating point degrade under
shift — discrimination and calibration are not the same thing, and external
validation is where the difference shows up.

**What this is and is not.** It is the measurement and the framing. It is not a
guarantee: external validity is only as strong as how genuinely independent the
external cohort is. Same-generator resamples (no distribution shift) will look
deceptively good; a real external cohort differs in ways a synthetic shift only
approximates. For any real use, validate on independent, outcome-linked data
collected separately from the development data.
