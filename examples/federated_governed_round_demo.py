"""Governed federated round — the mandatory-gate path end to end.

Each site produces a SEALED contribution that cannot be shipped directly; the
only way to obtain a payload is .disclose(policy, consent), which runs the
governance gate (suppression, tier, consent, manifest, audit). The coordinator
combines only disclosed payloads. This is the real cross-boundary path.

Run:  python examples/federated_governed_round_demo.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epinet import federated as efed
from epinet import governance as eg
from epinet import toolkit as et


def consent_for(site: str, controller: str) -> eg.Consent:
    return eg.Consent(
        site=site, controller=controller,
        lawful_basis="GDPR Art 9(2)(j) — scientific research",
        dpia_reference="DPIA-2026-014", purpose="federated contestability research",
        version="v1.0", allowed_tier="aggregate", coi_acknowledged=True,
        expires="2027-01-01",
    )


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
    controllers = {"Site A": "A Hospital Trust", "Site B": "B University Hospital"}

    policy = eg.DisclosurePolicy(min_cell=5, allowed_tier="aggregate")
    audit = eg.AuditLedger()
    when = dict(now=date(2026, 6, 11), timestamp="2026-06-11T00:00:00+00:00")

    print("Governed federated round (mandatory gate)")
    print("-" * 52)

    # A sealed contribution cannot be shipped without disclosing.
    sealed = efed.contribute_aggregate(X.loc[site_labels == "Site A"], y.loc[site_labels == "Site A"])
    print(f"sealed object: {sealed!r}")
    try:
        __import__("json").dumps(sealed)
        print("  [FAIL] sealed contribution was serializable")
    except TypeError:
        print("  [ok]   sealed contribution is NOT directly serializable")
    print()

    # Round 1 — fit: each site discloses through the gate; coordinator combines.
    disclosed_fit = []
    for s in sorted(site_labels.unique()):
        rows = site_labels == s
        contrib = efed.contribute_aggregate(X.loc[rows], y.loc[rows])
        disclosed_fit.append(contrib.disclose(policy=policy, consent=consent_for(s, controllers[s]),
                                               audit=audit, **when))
    fit = efed.combine_aggregates(disclosed_fit)
    print(f"fit reconstructed from disclosed payloads: classes={fit['classes']}, "
          f"n={fit['n_total']}, features={len(fit['kept_columns'])}")

    # Round 2 — contestability: same gate.
    disclosed_cont = []
    for s in sorted(site_labels.unique()):
        rows = site_labels == s
        contrib = efed.contribute_contestability(X.loc[rows], y.loc[rows], fit)
        disclosed_cont.append(contrib.disclose(policy=policy, consent=consent_for(s, controllers[s]),
                                               audit=audit, **when))
    summary = efed.combine_contestability(disclosed_cont)
    fd = summary["flip_distance"]
    print(f"federated contestability: mean flip={fd['mean']:.4f}, "
          f"top VOI={next(iter(summary['feature_voi']))}, "
          f"runner-up counts={summary['runner_up_counts']}")
    print()

    print(f"audit ledger entries: {len(audit.entries)} (one per disclosure)")
    print(f"audit verifies: {audit.verify()}")
    print()
    print("OK — nothing left a site except through the gate; coordinator combined disclosed payloads.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
