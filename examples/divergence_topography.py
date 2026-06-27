"""Divergence topography: where do two labelings of the same nodules disagree?

Motivated by the view that we never measure truth directly, only the distance
between representations. LIDC gives up to four independent radiologist readings
per nodule, so we can construct *two independent labelings of the same objects*
by split-half assignment of readers, then ask whether their disagreement is
**structured in feature space** (locatable contested regions) or idiosyncratic
noise.

The output is a derived node set whose Outcome is ``concordant`` / ``discordant``
(do the two labelings assign the same malignancy tier?). Running the standard
EpiNet pipeline on it answers the question: if morphology predicts discordance
above a permutation null under a patient-aware split, the disagreement lives in
specific regions of the feature space — a topography of contestation. If not,
the divergence is unstructured.

Two modes:

* **reader-split** (default): two halves of the radiologist panel. Real,
  in-sandbox-verifiable data; a methodological stand-in for a second labeling.
* **pathology** (``--pathology lidc_diagnosis.csv``): the radiologist tier vs a
  pathology-derived tier. The pathology file (TCIA
  ``tcia-diagnosis-data-2012-04-20``) is NOT bundled and must be supplied; this
  module does not fabricate it. The expected schema and code mapping are below.

Pathology file schema (when supplied as lidc_diagnosis.csv):
    columns: pid, diagnosis  (patient- or nodule-level)
    diagnosis codes (TCIA): 0=unknown, 1=benign/non-malignant,
                            2=malignant primary, 3=malignant metastatic
    mapping -> tier: 1 -> benign_low ; 2,3 -> suspicious_high ; 0 -> dropped
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the cohort's tier thresholds so labelings are comparable.
from build_lidc_cohort import malignancy_tier  # noqa: E402


def split_half_tiers(reader_malignancies: list[int], seed_index: int) -> tuple[str, str] | None:
    """Deterministically split readers into two halves and tier each half."""
    if len(reader_malignancies) < 2:
        return None
    order = sorted(range(len(reader_malignancies)),
                   key=lambda i: ((i + seed_index) * 2654435761) % 2**32)
    half = len(order) // 2
    group_a = [reader_malignancies[i] for i in order[:max(1, half)]]
    group_b = [reader_malignancies[i] for i in order[max(1, half):]]
    if not group_b:
        return None
    return (malignancy_tier(statistics.median(group_a)),
            malignancy_tier(statistics.median(group_b)))


def build_reader_split(nodes: pd.DataFrame, prov: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, r in prov.iterrows():
        readers = [int(x) for x in str(r["ReaderMalignancies"]).split(";") if x != ""]
        tiers = split_half_tiers(readers, seed_index=i)
        if tiers is None:
            continue
        tier_a, tier_b = tiers
        rows.append({"ID": r["ID"], "tier_a": tier_a, "tier_b": tier_b,
                     "Outcome": "concordant" if tier_a == tier_b else "discordant"})
    return pd.DataFrame(rows)


def build_pathology_split(prov: pd.DataFrame, diagnosis_path: Path) -> pd.DataFrame:
    diag = pd.read_csv(diagnosis_path)
    code_to_tier = {1: "benign_low", 2: "suspicious_high", 3: "suspicious_high"}
    diag = diag[diag["diagnosis"].isin(code_to_tier)].copy()
    diag["path_tier"] = diag["diagnosis"].map(code_to_tier)
    merged = prov.merge(diag[["pid", "path_tier"]], on="pid", how="inner")
    merged["rad_tier"] = merged["tier"].map(
        lambda t: "benign_low" if t == "benign_low" else
        ("suspicious_high" if t == "suspicious_high" else "indeterminate")
    )
    merged = merged[merged["rad_tier"] != "indeterminate"]
    merged["Outcome"] = np.where(merged["rad_tier"] == merged["path_tier"],
                                 "concordant", "discordant")
    return merged[["ID", "rad_tier", "path_tier", "Outcome"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--pathology", default="", help="Path to lidc_diagnosis.csv (pid,diagnosis)")
    args = ap.parse_args()

    here = Path(args.dir)
    nodes = pd.read_csv(here / "lidc_nodes.csv")
    prov = pd.read_csv(here / "lidc_provenance.csv")

    if args.pathology:
        labels = build_pathology_split(prov, Path(args.pathology))
        mode = "pathology (radiologist tier vs pathology tier)"
    else:
        labels = build_reader_split(nodes, prov)
        mode = "reader-split (two independent halves of the radiologist panel)"

    # Keep every node so the edge list stays valid: nodules we cannot label
    # (fewer than two readers, or no pathology match) remain unlabeled scaffold.
    nodule_feats = nodes[nodes["NodeType"] == "Nodule"].drop(columns=["Outcome"])
    nodules = nodule_feats.merge(labels[["ID", "Outcome"]], on="ID", how="left")
    nodules["Outcome"] = nodules["Outcome"].fillna("")
    scaffold = nodes[nodes["NodeType"] != "Nodule"].copy()
    scaffold["Outcome"] = ""
    out_nodes = pd.concat([scaffold, nodules], ignore_index=True)
    out_nodes.to_csv(here / "lidc_divergence_nodes.csv", index=False)

    n = int((nodules["Outcome"] != "").sum())
    disc = int((nodules["Outcome"] == "discordant").sum())
    print(f"Mode: {mode}")
    print(f"Labeled nodules: {n} | discordant: {disc} ({100*disc/n:.1f}%) | "
          f"concordant: {n - disc}")
    print(f"Wrote {here / 'lidc_divergence_nodes.csv'} (reuse lidc_edges.csv).")
    print("\nNext: run the standard pipeline to test whether discordance is")
    print("structured in feature space (predictable above a permutation null):")
    print("  python -m vahtian.epinet.toolkit \\")
    print("    --nodes examples/lidc_divergence_nodes.csv --edges examples/lidc_edges.csv \\")
    print("    --outcome-column Outcome --include-centrality --no-run-paths \\")
    print("    --split-strategy community --n-iterations 30 --permutation-test 200 \\")
    print("    --run-clusters --distance-metric mahalanobis --cluster-labeled-only \\")
    print("    --output-dir examples/lidc_divergence_outputs")


if __name__ == "__main__":
    main()
