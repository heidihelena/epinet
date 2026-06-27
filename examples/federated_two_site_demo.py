"""Two-site federated-fit demonstration.

Splits the bundled synthetic cohort across two notional sites, exchanges ONLY
per-site aggregates (count / sum / sum-of-squares, plus per-class count / sum),
and shows that the combined scaler and class centroids match a centralized run
to floating-point precision — proving the "aggregates-only" federation
reconstructs exactly what EpiNet would compute on pooled data.

No record ever leaves a site in this protocol; only the aggregate messages do.
Run:  python examples/federated_two_site_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vahtian.epinet import federated as efed
from vahtian.epinet import toolkit as et


def build_design(nodes_csv: str, edges_csv: str) -> tuple[pd.DataFrame, pd.Series]:
    nodes, edges = et.load_tables(nodes_csv, edges_csv)
    graph = et.build_graph(nodes, edges)
    features = et.generate_graph_features(graph)
    X = et.build_design_matrix(features, nodes, id_column="ID", outcome_column="Outcome")
    y = nodes.assign(ID=nodes["ID"].astype(str)).set_index("ID")["Outcome"]
    y = y.reindex(X.index)
    return X, y


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    X, y = build_design(str(repo / "synthetic_nodes.csv"), str(repo / "synthetic_edges.csv"))

    # Deterministic two-site partition (alternate rows -> Site A / Site B).
    site_labels = np.where(np.arange(len(X)) % 2 == 0, "Site A", "Site B")

    result = efed.simulate(X, y, site_labels, min_cell=0)

    print("Two-site federated fit on the synthetic cohort")
    print("-" * 52)
    print(f"total nodes pooled (never moved): {result['n_total']}")
    for site, info in result["sites"].items():
        print(f"  {site}: n={info['n']}, class counts={info['classes']}")
    print(f"classes: {result['classes']}")
    print(f"retained features: {len(result['kept_columns'])}")
    print()
    print("federated vs centralized (max absolute difference):")
    print(f"  global mean : {result['max_mean_diff']:.2e}")
    print(f"  global sd   : {result['max_sd_diff']:.2e}")
    print(f"  centroids   : {result['max_centroid_diff']:.2e}")
    print()

    worst = max(result["max_mean_diff"], result["max_sd_diff"], result["max_centroid_diff"])
    if worst < 1e-9:
        print(f"MATCH — aggregates-only reconstruction is exact (max diff {worst:.2e}).")
        return 0
    print(f"MISMATCH — investigate (max diff {worst:.2e}).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
