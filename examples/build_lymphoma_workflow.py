"""Suggested workflow: lymphoma digital-pathology -> an EpiNet ML model to test.

EpiNet works on tabular node/edge data, while a digital-pathology dataset is
whole-slide images. The bridge is feature extraction: turn each case into a row
of numbers, then let EpiNet build a patient-similarity network and train +
honestly evaluate a subtype classifier on it.

Pipeline
--------
1. Extract per-case features from the slides (your step, off-line). Any numeric
   summary works: cell size, nuclear contour irregularity, Ki-67 proliferation
   index, mitotic count, IHC marker positivity (CD20, CD10, BCL2, CD5, ...), or
   deep patch-embedding summaries. One row per case, plus a subtype/outcome
   label column.
2. `build_similarity_graph()` standardises the features and connects each case
   to its k nearest neighbours in feature space (a patient-similarity network).
   Graph position then becomes informative, and a community-aware split keeps
   near-duplicate cases out of train/test together.
3. Run EpiNet to generate the model:

   epinet --nodes lymphoma_nodes.csv --edges lymphoma_edges.csv \
     --outcome-column Subtype --no-run-paths \
     --n-iterations 30 --permutation-test 200 \
     --run-clusters --distance-metric mahalanobis --cluster-labeled-only \
     --interactive-network --output-dir lymphoma_outputs

   This produces: a RandomForest subtype model with feature importances, an
   honest accuracy estimate (repeated stratified splits + a permutation null, so
   a good score means real signal not chance), centroid clustering of the
   subtypes, and an interactive network.html.

   Use the default *random* (stratified) split here, NOT `--split-strategy
   community`: in a feature-similarity graph the communities ARE the subtypes,
   so a community split would hold out whole subtypes. Community splitting is for
   graphs where communities are nuisance grouping (e.g. several samples per
   patient) — if your cases are grouped by patient, add a patient id and that
   changes.

Usage
-----
  # Real data: a CSV with an id column, a label column, and numeric features.
  python build_lymphoma_workflow.py --features cases.csv --id-col CaseID \
      --label-col Subtype --k 6 --output examples

  # Demo: synthetic lymphoma-shaped cohort to see the workflow run end to end.
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
# Features echo common nodal B-cell lymphoma discriminators.
_SUBTYPE_PROFILES = {
    # subtype: (CellSize, Ki67, NuclearContour, CD20, CD10, BCL2, CD5, MitoticCount)
    "DLBCL": (8.5, 70, 7.0, 0.95, 0.55, 0.6, 0.10, 18),
    "FL":    (5.5, 20, 5.5, 0.95, 0.85, 0.9, 0.05, 4),
    "CLL":   (4.0, 8, 2.5, 0.85, 0.05, 0.7, 0.90, 1),
}
_FEATURES = ["CellSize", "Ki67", "NuclearContour", "CD20", "CD10", "BCL2", "CD5", "MitoticCount"]


def synthetic_lymphoma_cohort(n_per_class: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for subtype, means in _SUBTYPE_PROFILES.items():
        for i in range(n_per_class):
            vals = {}
            for f, mu in zip(_FEATURES, means):
                # Markers (0..1) get tighter noise; counts/sizes get proportional noise.
                if f in {"CD20", "CD10", "BCL2", "CD5"}:
                    vals[f] = float(np.clip(rng.normal(mu, 0.12), 0, 1))
                else:
                    vals[f] = float(max(0.0, rng.normal(mu, mu * 0.20)))
            rows.append({"CaseID": f"{subtype}_{i:03d}", "Subtype": subtype, **vals})
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
    ap.add_argument("--output", default="examples")
    args = ap.parse_args()

    if args.demo:
        features = synthetic_lymphoma_cohort()
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
    print("\nNow generate the model:")
    print("  epinet --nodes examples/lymphoma_nodes.csv --edges examples/lymphoma_edges.csv \\")
    print(f"    --outcome-column {args.label_col} --no-run-paths \\")
    print("    --n-iterations 30 --permutation-test 200 \\")
    print("    --run-clusters --distance-metric mahalanobis --cluster-labeled-only \\")
    print("    --interactive-network --output-dir examples/lymphoma_outputs")


if __name__ == "__main__":
    main()
