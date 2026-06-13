"""Registry schema adapter demonstration.

Fabricates a small flat "registry export" (one row per case, coded columns),
maps it to EpiNet's canonical node/edge schema with a declarative profile, and
runs the result through EpiNet to show the export flows end to end — no bespoke
glue, formatting only.

The same profile, applied at each site, is the shared feature contract the
federated fit needs; this demo runs a single site for clarity.
Run:  python examples/registry_adapter_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epinet import registry as ereg
from epinet import toolkit as et


def fake_registry_export(n=60, seed=0) -> pd.DataFrame:
    """A stand-in flat registry table: demographics + coded fields + outcome."""
    rng = np.random.default_rng(seed)
    stage = rng.integers(1, 5, size=n)
    rows = {
        "case_id": [f"CASE-{i:03d}" for i in range(n)],
        "age": rng.integers(45, 85, size=n),
        "stage_num": stage,
        "biomarker_level": rng.normal(stage * 1.5, 1.0, size=n).round(2),
        "pack_years": rng.integers(0, 60, size=n),
        "hospital": rng.choice(["North", "South", "East"], size=n),  # non-numeric, dropped
        # Outcome correlates with stage so there is something to find.
        "vital_status": np.where(stage >= 3, "deceased", "alive"),
    }
    return pd.DataFrame(rows)


def main() -> int:
    table = fake_registry_export()

    profile = ereg.RegistryProfile.from_dict({
        "id_column": "case_id",
        "outcome_column": "vital_status",
        "feature_columns": None,        # auto: numeric columns
        "edge_strategy": "knn",
        "knn_k": 5,
    })
    result = ereg.adapt(table, profile)
    nodes, edges, manifest = result["nodes"], result["edges"], result["manifest"]

    print("Registry adapter — flat export -> EpiNet schema")
    print("-" * 52)
    for key in ["n_cases", "feature_columns", "edge_strategy", "n_edges",
                "dropped_columns", "edge_note"]:
        print(f"  {key}: {manifest[key]}")
    print(f"  source_sha256: {manifest['source_sha256'][:16]}...")
    print()
    print("nodes (head):")
    print(nodes.head(3).to_string(index=False))
    print()

    # Flow straight into EpiNet: build the graph and compute features.
    graph = et.build_graph(nodes, edges, weight_column="Weight")
    features = et.generate_graph_features(graph)
    print(f"EpiNet ingested it: {graph.number_of_nodes()} nodes, "
          f"{graph.number_of_edges()} edges, {features.shape[1] - 1} graph features.")

    ok = (
        list(nodes.columns[:2]) == ["ID", "Outcome"]
        and list(edges.columns) == ["SourceID", "TargetID", "Weight"]
        and graph.number_of_nodes() == manifest["n_cases"]
    )
    print()
    print("OK — registry export adapts to canonical schema and runs through EpiNet."
          if ok else "MISMATCH — investigate.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
