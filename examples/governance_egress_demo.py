"""Governance / consent gate demonstration.

Shows that a federated aggregate only leaves a site through check_egress, which
fails closed: it suppresses small cells, refuses without valid consent or above
the permitted tier, blocks identifying fields, and records a tamper-evident
audit entry with a disclosure manifest of exactly what crossed.

Run:  python examples/governance_egress_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vahtian.epinet import federated as efed
from vahtian.epinet import governance as eg


def main() -> int:
    # A site aggregate with one healthy class and one rare class (n=3).
    rng = np.random.default_rng(0)
    big = pd.DataFrame(rng.normal(size=(40, 3)), columns=["f0", "f1", "f2"])
    rare = pd.DataFrame(rng.normal(5, 1, size=(3, 3)), columns=["f0", "f1", "f2"])
    X = pd.concat([big, rare], ignore_index=True)
    y = pd.Series(["common"] * 40 + ["rare"] * 3, name="Outcome")
    agg = efed.site_aggregates(X, y)

    policy = eg.DisclosurePolicy(min_cell=5, allowed_tier="aggregate")
    consent = eg.Consent(
        site="Site A", controller="Site A Hospital Trust",
        lawful_basis="GDPR Art 9(2)(j) — scientific research",
        dpia_reference="DPIA-2026-014", purpose="federated contestability research",
        version="v1.0", allowed_tier="aggregate", coi_acknowledged=True,
        expires="2027-01-01",
    )
    audit = eg.AuditLedger()

    print("Governance gate on a federated aggregate")
    print("-" * 52)
    redacted, manifest = eg.check_egress(
        agg, policy=policy, consent=consent, tier="aggregate",
        audit=audit, now=__import__("datetime").date(2026, 6, 11),
        timestamp="2026-06-11T00:00:00+00:00",
    )
    print("EGRESS ALLOWED. disclosure manifest:")
    for key in ["tier", "record_count", "suppressed_cells", "controller",
                "lawful_basis", "consent_version", "payload_sha256"]:
        print(f"  {key}: {manifest[key]}")
    print(f"  rare class still present? {'rare' in redacted.get('class_n', {})}  (suppressed below min_cell)")
    print()

    print("Fail-closed checks (each must be refused):")
    fixed = dict(now=__import__("datetime").date(2026, 6, 11))

    def refused(label, **kw):
        try:
            eg.check_egress(agg, policy=policy, **kw, **fixed)
            print(f"  [FAIL] {label}: was ALLOWED")
            return False
        except eg.GovernanceError as e:
            print(f"  [ok]   {label}: refused ({str(e)[:60]})")
            return True

    all_ok = all([
        refused("no COI acknowledgement", consent=eg.Consent(
            site="A", controller="C", lawful_basis="L", dpia_reference="D",
            purpose="P", version="v1", coi_acknowledged=False)),
        refused("expired consent", consent=eg.Consent(
            site="A", controller="C", lawful_basis="L", dpia_reference="D",
            purpose="P", version="v1", coi_acknowledged=True, expires="2020-01-01")),
        refused("tier above consent", consent=consent, tier="derived"),
        refused("identifying tier", consent=consent, tier="identifiable"),
    ])

    print()
    print(f"audit ledger verifies: {audit.verify()}")
    # Tamper with a recorded manifest and show the chain breaks.
    audit.entries[0]["event"]["manifest"]["record_count"] = 9999
    print(f"after tampering, verifies: {audit.verify()}  (tamper-evident)")

    print()
    print("OK — gate fails closed and discloses exactly what crosses."
          if all_ok else "MISMATCH — a fail-closed check did not refuse.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
