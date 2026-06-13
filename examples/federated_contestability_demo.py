"""Federated contestability demonstration (Tier-2 derived dataset).

Builds on the federated fit: after the pooled scaler + centroids are recovered
from per-site aggregates, each site computes flip-distance LOCALLY against the
global centroids and returns only a de-identified summary. This shows:

1. the per-node contestability SCORES match a centralized run exactly (~1e-14),
   because the standardized vectors and centroids were reconstructed exactly; and
2. the federated summary (mean/std/min/max, runner-up counts, value-of-
   information) matches centralized exactly, with the quantile-based contested
   threshold the one approximate piece (read off a summed histogram).

No per-node record crosses a site boundary — only the summary does.
Run:  python examples/federated_contestability_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epinet import contest as ecn
from epinet import federated as efed
from epinet import toolkit as et


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    nodes, edges = et.load_tables(
        str(repo / "synthetic_nodes.csv"), str(repo / "synthetic_edges.csv")
    )
    graph = et.build_graph(nodes, edges)
    feats = et.generate_graph_features(graph)
    X = et.build_design_matrix(feats, nodes, id_column="ID", outcome_column="Outcome")
    y = nodes.assign(ID=nodes["ID"].astype(str)).set_index("ID")["Outcome"].reindex(X.index)

    site_labels = pd.Series(np.where(np.arange(len(X)) % 2 == 0, "Site A", "Site B"), index=X.index)

    # Round 1 — federated fit (aggregates only).
    aggs = [
        efed.site_aggregates(X.loc[site_labels == s], y.loc[site_labels == s])
        for s in sorted(site_labels.unique())
    ]
    fit = efed.combine_aggregates(aggs)

    # Round 2 — each site scores locally and returns a de-identified summary.
    summaries = []
    local_flip = {}
    for s in sorted(site_labels.unique()):
        rows = site_labels == s
        summaries.append(efed.site_contestability(X.loc[rows], y.loc[rows], fit))
        fd = efed.local_flip_distances(X.loc[rows], fit)
        local_flip.update(dict(zip(X.loc[rows].index, fd)))
    fed = efed.combine_contestability(summaries, contest_quantile=0.1)

    # Centralized reference.
    central = ecn.contestability(X, y=y, metric="euclidean", contest_quantile=0.1)
    cframe = central["assignments"].set_index("ID")
    csum = central["summary"]

    fed_flip = pd.Series(local_flip).reindex(cframe.index)
    score_diff = float(np.max(np.abs(fed_flip.to_numpy() - cframe["flip_distance"].to_numpy())))

    print("Federated contestability on the synthetic cohort")
    print("-" * 52)
    print(f"nodes scored (never moved): {fed['n_scored']}")
    print()
    print("1) per-node flip-distance scores (computed site-locally):")
    print(f"   max |federated - centralized| = {score_diff:.2e}   <- exact")
    print()
    print("2) federated summary vs centralized (exact additive aggregates):")
    print(f"   mean : fed {fed['flip_distance']['mean']:.6f}  vs  central {csum['flip_distance']['mean']:.6f}")
    print(f"   std  : fed {fed['flip_distance']['std']:.6f}  vs  central {csum['flip_distance']['std']:.6f}")
    print(f"   min  : fed {fed['flip_distance']['min']:.6f}  vs  central {csum['flip_distance']['min']:.6f}")
    print(f"   max  : fed {fed['flip_distance']['max']:.6f}  vs  central {csum['flip_distance']['max']:.6f}")
    print(f"   runner-up counts: fed {fed['runner_up_counts']}  vs  central {csum['runner_up_counts']}")
    fed_top = next(iter(fed["feature_voi"]))
    central_top = next(iter(csum["feature_leverage"]))
    print(f"   top value-of-information feature: fed '{fed_top}'  vs  central '{central_top}'")
    print()
    print("3) contested set (the one APPROXIMATE piece — histogram quantile):")
    print(f"   fed approx threshold  = {fed['flip_distance']['approx_contest_threshold']:.6f}"
          f"  vs  central exact = {csum['flip_distance']['contest_threshold']:.6f}")
    print(f"   fed approx n_contested = {fed['flip_distance']['approx_n_contested']}"
          f"  vs  central exact = {csum['flip_distance']['n_contested']}  (both ~= q*N)")

    if score_diff < 1e-9 and fed_top == central_top:
        print()
        print(f"MATCH — contestability scores federate exactly (max diff {score_diff:.2e}).")
        return 0
    print("MISMATCH — investigate.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
