# Nordic Lung Cancer Quality Indicators — capability network

This example applies the toolkit to a **real domain**: quality indicators (QIs)
for the lung cancer diagnostic and treatment pathway across the five Nordic
countries (Denmark, Finland, Iceland, Norway, Sweden). It is a worked
demonstration, not a benchmarking result — see the boundaries at the end.

## What the network represents

The graph is a **measurement-capability network** with three node types:

| Node type | Count | Role |
|-----------|-------|------|
| Country registry | 5 | scaffold (unlabeled) |
| Data source / infrastructure | 9 | scaffold (unlabeled) |
| Quality indicator | 17 | labeled by **feasibility tier** |

Edges:

- **captures** (country → indicator): present where that national registry
  currently measures, or directly derives, the indicator. Edge weight encodes
  measurement quality (explicit operational definition = 1.0; derivable from
  linkage = 0.6).
- **requires** (indicator → data source): the registry/linkage infrastructure
  the indicator depends on (clinical registry, pathology registry, hospital
  discharge codes, mortality linkage, radiotherapy registry, systemic-therapy
  data, PROM module, palliative-service data, population cancer registry).

The per-country capture matrix is transcribed from a curated Nordic QI table
(DLCR Årsrapport 2023; Swedish *Lungcancerregistret* manual; the Norwegian
national lung cancer quality register; the Finnish Cancer Registry / FICAN; the
Icelandic Cancer Registry). Country five-year relative survival is approximate,
from NORDCAN 2.0 / the *Acta Oncologica* registry-comparison literature.

The dataset is generated reproducibly by
[`build_nordic_lung_cancer_qi.py`](build_nordic_lung_cancer_qi.py).

## Feasibility tier (the modelled outcome)

Each indicator is labeled by how many of the five registries capture it:

- **broad** (≥4 countries): the three survival/early-mortality outcome
  indicators — measurable everywhere from cancer-registry + mortality linkage.
- **partial** (2–3 countries): 11 process indicators — staging, timeliness,
  MDT, surgery/radiotherapy/systemic-therapy utilisation, smoking status, PROMs
  — captured mainly by the mature Danish and Swedish registries.
- **gap** (≤1 country): R0 resection rate, postoperative complications, and
  palliative-care referral timing — captured by at most one registry.

The country and data-source nodes are **unlabeled scaffold**: they shape every
indicator's graph features (degree, centrality, community) but are excluded from
supervised training. This is the semi-supervised setting the toolkit now
supports — only one node type carries the label of interest.

## How to reproduce

```bash
python examples/build_nordic_lung_cancer_qi.py     # regenerate the CSVs

python -m epinet.toolkit \
  --nodes examples/nordic_lung_cancer_qi_nodes.csv \
  --edges examples/nordic_lung_cancer_qi_edges.csv \
  --outcome-column Outcome --target-outcome broad --weight-column Weight \
  --path-mode strength --include-centrality --split-strategy community \
  --n-iterations 50 --permutation-test 200 \
  --source-nodes IND_StageCompleteness,IND_TimeToDiagnosis,IND_TimeToTreatment,IND_MDT_Documented,IND_SurgeryRate_EarlyNSCLC,IND_R0_ResectionRate,IND_30dPostopMortality,IND_PostopComplications,IND_RadiotherapyUtil,IND_SystemicTherapy_IV,IND_CurativeIntentRate,IND_SmokingStatusRecorded,IND_PROM_Capture,IND_PalliativeReferralTiming \
  --output-dir examples/nordic_qi_outputs
```

## Interpretation

### 1. Graph structure

31 nodes, 80 edges, a single connected component. Denmark (degree 15) and Sweden
are the dominant hubs — they capture nearly every indicator — while Finland and
Iceland connect almost exclusively to the universal outcome indicators. The
network *is* the harmonization story: capability is concentrated in two
registries.

### 2. Outcome model — and why the split strategy decides the conclusion

Predicting feasibility tier from graph position + clinical attributes, evaluated
over 50 repeated splits with a 200-permutation null:

| Split strategy | F1 (mean ± sd) | Null F1 | p-value | Conclusion |
|----------------|----------------|---------|---------|------------|
| Random | 0.72 ± 0.14 | 0.35 | **0.015** | "structure predicts tier" |
| Community-aware | 0.65 ± 0.35 | 0.49 | 0.31 | **not** distinguishable from chance |

This contrast is the most important result. Under **random** splits the model
looks significant — but connected indicators share neighbours, so a random split
leaks structure between train and test. Under **community-aware** splits (whole
data-infrastructure clusters held out) the apparent signal largely disappears:
the variance explodes (F1 ranges 0.1–1.0 across splits) and the observed score
sits inside the permutation null. With only 17 labeled nodes, the toolkit's
honest verdict is that feasibility tier is **not** robustly predictable beyond
the trivial fact that universally-captured indicators are high-degree hubs.

Feature importance confirms the mechanism: the top five features are all
structural (weighted degree, degree, PageRank, closeness, betweenness ≈ 74%
combined), with clinical type (Donabedian outcome-vs-process ≈ 10%, linkage
requirement ≈ 7%) adding little. Feasibility here is essentially connectivity.

This is exactly the failure mode the iterative-evaluation, permutation-test, and
community-split features were built to expose — and the synthetic-data example
(random labels, p ≈ 0.44) is the contrasting null case.

### 3. Shortest paths — harmonization routes

Treating the three **broad** indicators as targets and asking, for each
non-broad indicator, the strongest route to one (maximising the product of edge
weights):

- Every indicator **already captured by Denmark** sits two hops from the
  universal outcome indicators at full strength (e.g. `IND_MDT_Documented →
  DK_DLCR → IND_1yrOS`). Translation: these are not real gaps — the capability
  exists, it is a harmonization/definition problem, not a data problem.
- The genuine **gaps** route through *shared data infrastructure* instead,
  at reduced strength, because no country edge exists:
  - `IND_R0_ResectionRate → DS_ClinicalRegistry → IND_StageCompleteness → DK_DLCR → IND_1yrOS` (4 hops) — to add R0, extend the **pathology/clinical-registry** infrastructure already used for stage completeness.
  - `IND_PalliativeReferralTiming → DS_MortalityLinkage → IND_1yrOS` (2 hops) — palliative-timing measurement can be anchored on the **mortality-linkage** infrastructure every registry already runs for survival.

The strength drop (1.0 → 0.64) cleanly separates "Denmark already does it" from
the indicators that require new infrastructure — a concrete, prioritised
harmonization to-do list that mirrors the project's own
"importance-exceeds-feasibility" logic.

### 4. Feature-space clustering and centroid distances

The path lens above measures *topological* distance. The clustering lens
(`--run-clusters --distance-metric mahalanobis`) measures **feature-space**
distance instead: each node becomes a standardized vector (graph features +
attributes), k-means finds centroids, and each node gets a Mahalanobis distance
to every feasibility-tier centroid — the network analogue of scoring a lung
nodule by its distance to risk-tier centroids in a standardized feature space.

The nearest-tier-centroid classifier recovers 16 of 17 indicator tiers in
feature space (in-sample accuracy 0.94). The single **feature-space outlier** is
the actionable finding:

| Indicator | Actual tier | Nearest centroid | dist→broad | dist→partial |
|-----------|-------------|------------------|-----------|--------------|
| 30-day postoperative mortality | partial | **broad** | 2.44 | 3.18 |

30-day postoperative mortality is currently captured by only two registries
(tier *partial*), yet its feature vector sits closest to the **broad** centroid.
The reason is mechanistic: it is an outcome-type indicator built on
**mortality linkage** — the exact infrastructure every Nordic registry already
runs for survival reporting. In other words, of all the non-universal
indicators, this is the one most structurally ready to go pan-Nordic, because
the data substrate is already in place everywhere. That is precisely the kind of
"near a higher tier despite its nominal label" signal the nodule centroid work
looks for, transplanted to the registry-capability setting.

Euclidean and Mahalanobis agree on the clustering but Mahalanobis is the better
metric here (nearest-centroid accuracy 0.94 vs 0.88) because the graph features
are strongly correlated — degree, weighted degree, and PageRank move together —
and Mahalanobis discounts that shared variance.

## Boundaries

- This is a **structural / capability** model, not a clinical or outcomes model.
  It says nothing about whether any indicator improves survival.
- The feasibility tier is defined *by* the capture matrix, so the model
  partly recovers its own construction — which is precisely why the
  community-split + permutation evaluation matters, and why the honest read is
  "no robust signal beyond hub structure on 17 nodes."
- The capture matrix is a point-in-time transcription and will drift as
  registries evolve (e.g. the 2026 FICAN/THL pilot). Regenerate before reuse.
- Country survival figures are approximate and used only for context/colour,
  not in the model.
```
