"""External validation on two independent lymphoma cohorts.

Fits the model on a development cohort (seed 0) and evaluates it, untouched, on
an independently generated external cohort (seed 99). Reports internal vs
external metrics and the drift between them — the honest test of transport.

Run:  python examples/external_validation_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_lymphoma_workflow as blw

from vahtian.epinet import validation as ev


def cohort(seed: int, *, marker_shift: float = 0.0):
    features = blw.synthetic_lymphoma_cohort(n_per_class=40, seed=seed, grey_zone=12)
    if marker_shift:
        # Simulate a different site/scanner: a systematic offset on two markers.
        for col in ("Ki67", "CD10"):
            features[col] = (features[col] + marker_shift).clip(0.0, 1.0)
    return blw.build_similarity_graph(features, id_col="CaseID", label_col="Subtype", k=6)


def main() -> int:
    dev_nodes, dev_edges = cohort(seed=0)
    # External cohort: independently drawn AND shifted, to mimic a real
    # cross-site difference rather than a same-distribution resample.
    ext_nodes, ext_edges = cohort(seed=99, marker_shift=0.15)

    result = ev.external_validation(
        dev_nodes, dev_edges, ext_nodes, ext_edges, outcome_column="Subtype", random_state=42,
    )

    print("External validation: develop on cohort A, validate on independent cohort B")
    print("-" * 72)
    metrics = ["roc_auc", "average_precision", "balanced_accuracy", "f1_weighted"]
    print(f"{'metric':<22}{'internal (A)':>14}{'external (B)':>14}{'drift':>10}")
    for m in metrics:
        i = result["internal"].get(m)
        e = result["external"].get(m)
        d = result["drift_internal_minus_external"].get(m)
        fi = f"{i:.3f}" if isinstance(i, float) else "—"
        fe = f"{e:.3f}" if isinstance(e, float) else "—"
        fd = f"{d:+.3f}" if isinstance(d, float) else "—"
        print(f"{m:<22}{fi:>14}{fe:>14}{fd:>10}")
    print()
    print(f"external cohort size: {result['external']['n_external']}")
    print(result["note"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
