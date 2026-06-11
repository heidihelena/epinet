"""Suggested workflow: lymphoma digital-pathology -> an EpiNet ML model to test.

EpiNet works on tabular node/edge data, while a digital-pathology dataset is
whole-slide images. The bridge is feature extraction: turn each case into a row
of numbers, then let EpiNet build a patient-similarity network and train +
honestly evaluate a subtype classifier on it.

Pipeline
--------
1. Extract per-case features from the slides (your step, off-line). Any numeric
   summary works: cell size, nuclear contour irregularity, Ki-67 proliferation
   index, mitotic count, IHC marker positivity (CD20, CD10, BCL2, CD5, CD23,
   cyclin D1, MYC, ...), or deep patch-embedding summaries. One row per case,
   plus a subtype/outcome label column.
2. `build_similarity_graph()` standardises the features and connects each case
   to its k nearest neighbours in feature space (a patient-similarity network).
   Graph position then becomes informative, and a community-aware split keeps
   near-duplicate cases out of train/test together.
3. Run EpiNet to generate the model AND map where the subtype call is contested:

   epinet --nodes lymphoma_nodes.csv --edges lymphoma_edges.csv \
     --outcome-column Subtype --no-run-paths \
     --n-iterations 30 --permutation-test 200 \
     --run-clusters --run-contest --distance-metric mahalanobis \
     --cluster-labeled-only --interactive-network --output-dir lymphoma_outputs

   This produces: a RandomForest subtype model with feature importances, an
   honest accuracy estimate (repeated stratified splits + a permutation null, so
   a good score means real signal not chance), centroid clustering of the
   subtypes, an interactive network.html, and — via `--run-contest` — a
   per-case contestability map: which cases sit on a subtype boundary, which
   subtype they would flip to, and which single marker most cheaply resolves the
   call (the value-of-information / "order this stain next" signal).

   Use the default *random* (stratified) split here, NOT `--split-strategy
   community`: in a feature-similarity graph the communities ARE the subtypes,
   so a community split would hold out whole subtypes. Community splitting is for
   graphs where communities are nuisance grouping (e.g. several samples per
   patient) — if your cases are grouped by patient, add a patient id and that
   changes.

Why this cohort is not trivially separable
------------------------------------------
The synthetic demo deliberately includes two clinically real *confusable pairs*
that overlap on routine markers and are separated by a single decisive one:

- **CLL vs MCL** — both CD5+ / CD10-; cyclin D1 (t(11;14)) is what defines MCL.
- **DLBCL vs Burkitt** — both CD20+ / CD10+; near-100% Ki-67 and MYC mark Burkitt
  (the unresolved middle is "high-grade B-cell lymphoma").

With `--grey-zone N` the demo also injects N genuinely ambiguous cases drawn near
a pair's midpoint. A full-marker classifier still separates the bulk easily, but
those boundary cases are where the contestability lens earns its keep: it flags
them and names the marker (e.g. cyclin D1) that would settle them.

Usage
-----
  # Real data: a CSV with an id column, a label column, and numeric features.
  python build_lymphoma_workflow.py --features cases.csv --id-col CaseID \
      --label-col Subtype --k 6 --output examples

  # Demo: synthetic lymphoma-shaped cohort with grey-zone CLL/MCL cases.
  python build_lymphoma_workflow.py --demo --output examples

NOT a diagnostic tool. This is a research/education workflow; any model it
produces must be validated on independent, outcome-linked data before it means
anything clinically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Plausible per-subtype feature means for the synthetic demo (illustrative only).
# Features echo common nodal B-cell lymphoma discriminators. Marker columns are
# 0..1 positivity fractions; the rest are morphometric counts/sizes.
#            CellSize Ki67 NuclearContour CD20  CD10  BCL2  CD5   CD23  CyclinD1 MYC   MitoticCount
_SUBTYPE_PROFILES = {
    "DLBCL": (8.5,    75,   7.0,           0.95, 0.55, 0.60, 0.08, 0.10, 0.02,    0.35, 18),
    "FL":    (5.5,    18,   5.5,           0.95, 0.85, 0.92, 0.05, 0.10, 0.02,    0.05, 4),
    "CLL":   (4.0,    6,    2.5,           0.55, 0.03, 0.75, 0.92, 0.80, 0.02,    0.03, 1),
    "MCL":   (5.5,    30,   4.5,           0.95, 0.05, 0.70, 0.88, 0.40, 0.93,    0.10, 6),
    "BL":    (6.5,    99,   6.0,           0.95, 0.90, 0.08, 0.03, 0.05, 0.02,    0.95, 35),
}
_FEATURES = ["CellSize", "Ki67", "NuclearContour", "CD20", "CD10", "BCL2",
             "CD5", "CD23", "CyclinD1", "MYC", "MitoticCount"]
_MARKERS = {"CD20", "CD10", "BCL2", "CD5", "CD23", "CyclinD1", "MYC"}

# Clinically real confusable pairs: overlap on routine markers, separated by one
# decisive discriminator. These are the cases `--run-contest` should surface.
_CONFUSABLE_PAIRS = {
    ("CLL", "MCL"): "CyclinD1",   # both CD5+ / CD10-; cyclin D1 (t(11;14)) defines MCL
    ("DLBCL", "BL"): "Ki67",      # both CD20+ / CD10+; near-100% Ki-67 + MYC mark Burkitt
}

# Fixed, data-independent per-feature scale (spread of the profile means), used
# only to label grey-zone cases by their nearest subtype in scale-free space.
_PROFILE_SCALE = np.std(np.array(list(_SUBTYPE_PROFILES.values())), axis=0)
_PROFILE_SCALE[_PROFILE_SCALE == 0] = 1.0


def _sample_case(rng: np.random.Generator, means, *, spread: float = 1.0) -> dict[str, float]:
    """One noisy case around a profile. ``spread`` widens the noise (grey zone)."""
    vals: dict[str, float] = {}
    for f, mu in zip(_FEATURES, means):
        if f in _MARKERS:  # 0..1 positivity: tighter additive noise, clipped
            vals[f] = float(np.clip(rng.normal(mu, 0.10 * spread), 0.0, 1.0))
        else:              # counts/sizes: proportional noise, non-negative
            vals[f] = float(max(0.0, rng.normal(mu, mu * 0.20 * spread)))
    return vals


def synthetic_lymphoma_cohort(
    n_per_class: int = 40,
    seed: int = 0,
    *,
    grey_zone: int = 0,
    grey_zone_pair: tuple[str, str] = ("CLL", "MCL"),
) -> pd.DataFrame:
    """Synthetic nodal B-cell lymphoma cohort across five subtypes.

    ``grey_zone`` injects that many genuinely ambiguous cases drawn near the
    midpoint of ``grey_zone_pair``, each labeled by whichever subtype it actually
    lands closest to (in scale-free profile space). These are the boundary cases
    the contestability lens is meant to flag.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for subtype, means in _SUBTYPE_PROFILES.items():
        for i in range(n_per_class):
            rows.append({"CaseID": f"{subtype}_{i:03d}", "Subtype": subtype,
                         **_sample_case(rng, means)})

    if grey_zone > 0:
        a, b = grey_zone_pair
        if a not in _SUBTYPE_PROFILES or b not in _SUBTYPE_PROFILES:
            raise ValueError(f"grey_zone_pair must be two known subtypes, got {grey_zone_pair!r}")
        mean_a = np.array(_SUBTYPE_PROFILES[a], dtype=float)
        mean_b = np.array(_SUBTYPE_PROFILES[b], dtype=float)
        midpoint = (mean_a + mean_b) / 2.0
        for i in range(grey_zone):
            vals = _sample_case(rng, midpoint, spread=1.6)
            v = np.array([vals[f] for f in _FEATURES], dtype=float)
            d_a = np.linalg.norm((v - mean_a) / _PROFILE_SCALE)
            d_b = np.linalg.norm((v - mean_b) / _PROFILE_SCALE)
            label = a if d_a <= d_b else b
            rows.append({"CaseID": f"GZ_{a}_{b}_{i:03d}", "Subtype": label, **vals})

    return pd.DataFrame(rows)


def build_similarity_graph(
    features: pd.DataFrame, *, id_col: str, label_col: str, feature_cols: list[str] | None = None,
    k: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardise features and connect each case to its k nearest neighbours."""
    if feature_cols is None:
        feature_cols = [c for c in features.columns
                        if c not in {id_col, label_col} and pd.api.types.is_numeric_dtype(features[c])]
    if not feature_cols:
        raise ValueError("No numeric feature columns found for the similarity graph")

    ids = features[id_col].astype(str).tolist()
    X = StandardScaler().fit_transform(features[feature_cols].to_numpy(dtype=float))
    k = min(k, len(ids) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, idx = nn.kneighbors(X)

    seen: set[tuple[str, str]] = set()
    edge_rows = []
    for i in range(len(ids)):
        for j, d in zip(idx[i][1:], dist[i][1:]):  # skip self (first neighbour)
            a, b = sorted((ids[i], ids[j]))
            if (a, b) in seen:
                continue
            seen.add((a, b))
            edge_rows.append({"SourceID": a, "TargetID": b, "Relationship": "similar",
                              "Weight": round(float(1.0 / (1.0 + d)), 4)})

    node_rows = []
    for _, r in features.iterrows():
        # The outcome is stored under its real name (label_col) so the run uses
        # --outcome-column <label_col> directly.
        row = {"ID": str(r[id_col]), "NodeType": "Case", label_col: r[label_col],
               "Label": f"{r[label_col]} {r[id_col]}"}
        for f in feature_cols:
            row[f] = r[f]
        node_rows.append(row)
    nodes = pd.DataFrame(node_rows, columns=["ID", "NodeType", label_col, "Label"] + feature_cols)
    edges = pd.DataFrame(edge_rows, columns=["SourceID", "TargetID", "Relationship", "Weight"])
    return nodes, edges


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", help="CSV of cases x features (+ id and label columns)")
    ap.add_argument("--id-col", default="CaseID")
    ap.add_argument("--label-col", default="Subtype")
    ap.add_argument("--k", type=int, default=6, help="nearest neighbours per case")
    ap.add_argument("--demo", action="store_true", help="use a synthetic lymphoma cohort")
    ap.add_argument("--grey-zone", type=int, default=10,
                    help="inject this many ambiguous CLL/MCL boundary cases in --demo (0 to disable)")
    ap.add_argument("--output", default="examples")
    args = ap.parse_args()

    if args.demo:
        features = synthetic_lymphoma_cohort(grey_zone=args.grey_zone)
    elif args.features:
        features = pd.read_csv(args.features)
    else:
        ap.error("provide --features <csv>, or --demo")

    nodes, edges = build_similarity_graph(
        features, id_col=args.id_col, label_col=args.label_col, k=args.k)
    out = Path(args.output)
    nodes.to_csv(out / "lymphoma_nodes.csv", index=False)
    edges.to_csv(out / "lymphoma_edges.csv", index=False)

    counts = nodes[args.label_col].value_counts().to_dict()
    print(f"Wrote {len(nodes)} cases and {len(edges)} similarity edges "
          f"to lymphoma_nodes.csv / lymphoma_edges.csv")
    print("Subtype counts:", counts)
    print("\nNow generate the model and map the contested cases:")
    print("  epinet --nodes examples/lymphoma_nodes.csv --edges examples/lymphoma_edges.csv \\")
    print(f"    --outcome-column {args.label_col} --no-run-paths \\")
    print("    --n-iterations 30 --permutation-test 200 \\")
    print("    --run-clusters --run-contest --distance-metric mahalanobis \\")
    print("    --cluster-labeled-only --interactive-network --output-dir examples/lymphoma_outputs")
    print("\nThen read examples/lymphoma_outputs/node_contestability.csv: the most-contested")
    print("cases are the subtype boundaries, and most_decision_relevant_feature names the")
    print("marker that would settle each one (cyclin D1 for the CLL/MCL grey zone).")


if __name__ == "__main__":
    main()
