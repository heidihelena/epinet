"""Representation baseline comparison on the lymphoma cohort.

Runs EpiNet's graph features against a learned node embedding (spectral) and a
no-information floor, all under the same honest evaluation harness, so you can
see whether the hand-crafted features actually earn their place.

Run:  python examples/baseline_comparison_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_lymphoma_workflow as blw

from vahtian.epinet import baselines as eb


def main() -> int:
    features = blw.synthetic_lymphoma_cohort(n_per_class=40, seed=0, grey_zone=12)
    nodes, edges = blw.build_similarity_graph(features, id_col="CaseID", label_col="Subtype", k=6)

    result = eb.compare_representations(
        nodes, edges, outcome_column="Subtype", n_iterations=10, random_state=42,
    )
    comparison = result["comparison"]

    print("Representation comparison on the lymphoma cohort")
    print("-" * 60)
    print(comparison.to_string(index=False))
    print()
    print("Same harness, same seed; only the node representation varies.")
    print("The no-information row is the floor; graph features and the spectral")
    print("embedding should both clear it on a cohort with real structure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
