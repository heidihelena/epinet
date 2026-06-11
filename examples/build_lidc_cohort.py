"""Extract a real pulmonary-nodule cohort from LIDC-IDRI as an EpiNet network.

Uses the annotation database bundled with ``pylidc`` (no DICOM images needed):
1018 CT scans, 6859 radiologist annotations. Each physical nodule was read by
up to four radiologists, each assigning a malignancy score (1-5) and eight
semantic characteristics on ordinal scales.

This script clusters annotations into physical nodules, aggregates the
per-radiologist ratings, and writes the cohort as a node/edge network for the
EpiNet feature-space clustering. Unlike the synthetic cohort, this is real,
biased data:

- Malignancy is a *subjective radiologist gestalt*, not pathology confirmation.
- Readers hedge heavily toward "3" (indeterminate), inflating the middle tier.
- Only annotatable nodules >= 3 mm are included (selection bias).
- Inter-reader disagreement is real and is recorded (``MalignancySpread``,
  ``NReaders``) so it can be inspected rather than hidden.

Outcome = malignancy tier from the median reader malignancy:
    benign_low (< 2.5), indeterminate (2.5-3.5), suspicious_high (> 3.5)

The clustering uses the eight semantic characteristics + diameter; the
malignancy rating itself is the label and is kept in the provenance file, so
predicting the tier from morphology is not circular (malignancy is a separate
subjective rating, correlated with but not derived from the eight features).

Requires: pip install pylidc
"""

from __future__ import annotations

import csv
import statistics
import warnings
from pathlib import Path

import numpy as np

# pylidc 0.2.3 uses the long-deprecated np.int / np.bool aliases.
for _alias, _builtin in {"int": int, "float": float, "bool": bool, "object": object}.items():
    if not hasattr(np, _alias):
        setattr(np, _alias, _builtin)

warnings.filterwarnings("ignore")

import pylidc as pl  # noqa: E402

SEMANTIC_FEATURES = [
    "subtlety", "internalStructure", "calcification", "sphericity",
    "margin", "lobulation", "spiculation", "texture",
]


def malignancy_tier(median_malignancy: float) -> str:
    if median_malignancy < 2.5:
        return "benign_low"
    if median_malignancy > 3.5:
        return "suspicious_high"
    return "indeterminate"


def extract_nodules(max_scans: int | None = None, luna16: bool = False) -> list[dict]:
    """Cluster annotations into nodules and aggregate reader ratings.

    ``luna16`` reconstructs the LUNA16 cohort *definition* from LIDC: scans with
    slice thickness <= 2.5 mm and nodules accepted by >= 3 of 4 radiologists.
    This is a reconstruction from LIDC (which we have), not the LUNA16 download;
    it omits LUNA16's extra slice-spacing-consistency exclusion, so scan counts
    are ~1% above LUNA16's published 888.
    """
    nodules: list[dict] = []
    scans = pl.query(pl.Scan).all()
    if max_scans is not None:
        scans = scans[:max_scans]

    for si, scan in enumerate(scans):
        if luna16 and (scan.slice_thickness is None or scan.slice_thickness > 2.5):
            continue
        try:
            clusters = scan.cluster_annotations(verbose=False)
        except Exception:
            continue
        for ni, anns in enumerate(clusters):
            if not anns:
                continue
            if luna16 and len(anns) < 3:   # LUNA16: >=3 of 4 readers
                continue
            malignancies = [a.malignancy for a in anns]
            median_mal = float(statistics.median(malignancies))
            record = {
                "nid": f"ND_{scan.id}_{ni}",
                "pid": scan.patient_id,
                "tier": malignancy_tier(median_mal),
                "NReaders": len(anns),
                "MalignancyMedian": median_mal,
                "MalignancySpread": int(max(malignancies) - min(malignancies)),
                "ReaderMalignancies": ";".join(str(m) for m in malignancies),
            }
            for feature in SEMANTIC_FEATURES:
                vals = [getattr(a, feature) for a in anns]
                record[feature] = float(statistics.median(vals))
            try:
                record["DiameterMm"] = round(float(np.mean([a.diameter for a in anns])), 2)
            except Exception:
                record["DiameterMm"] = ""
            nodules.append(record)
        if (si + 1) % 100 == 0:
            print(f"  processed {si + 1} scans, {len(nodules)} nodules so far")
    return nodules


def write_cohort(out_dir: Path, max_scans: int | None = None, luna16: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "luna16" if luna16 else "lidc"
    nodules = extract_nodules(max_scans=max_scans, luna16=luna16)
    nodules = [n for n in nodules if n["DiameterMm"] != ""]

    patients = sorted({n["pid"] for n in nodules})
    feature_cols = SEMANTIC_FEATURES + ["DiameterMm"]
    node_fields = ["ID", "NodeType", "Outcome", "Label"] + feature_cols

    with (out_dir / f"{prefix}_nodes.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=node_fields)
        w.writeheader()
        for pid in patients:
            w.writerow({"ID": pid, "NodeType": "Scan", "Outcome": "",
                        "Label": pid, **{c: "" for c in feature_cols}})
        for n in nodules:
            w.writerow({"ID": n["nid"], "NodeType": "Nodule", "Outcome": n["tier"],
                        "Label": f"d={n['DiameterMm']}mm mal={n['MalignancyMedian']}",
                        **{c: n[c] for c in feature_cols}})

    with (out_dir / f"{prefix}_edges.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["SourceID", "TargetID", "Relationship", "Weight"])
        w.writeheader()
        by_patient: dict[str, list[str]] = {}
        for n in nodules:
            w.writerow({"SourceID": n["pid"], "TargetID": n["nid"],
                        "Relationship": "in_scan", "Weight": 1.0})
            by_patient.setdefault(n["pid"], []).append(n["nid"])
        for siblings in by_patient.values():
            for i in range(len(siblings)):
                for j in range(i + 1, len(siblings)):
                    w.writerow({"SourceID": siblings[i], "TargetID": siblings[j],
                                "Relationship": "same_scan", "Weight": 1.0})

    with (out_dir / f"{prefix}_provenance.csv").open("w", newline="") as fh:
        prov = ["ID", "pid", "tier", "NReaders", "MalignancyMedian", "MalignancySpread",
                "ReaderMalignancies", "DiameterMm"]
        w = csv.DictWriter(fh, fieldnames=prov)
        w.writeheader()
        for n in nodules:
            w.writerow({"ID": n["nid"], "pid": n["pid"], "tier": n["tier"],
                        "NReaders": n["NReaders"], "MalignancyMedian": n["MalignancyMedian"],
                        "MalignancySpread": n["MalignancySpread"],
                        "ReaderMalignancies": n["ReaderMalignancies"], "DiameterMm": n["DiameterMm"]})

    tiers = [n["tier"] for n in nodules]
    spreads = [n["MalignancySpread"] for n in nodules]
    label = "LUNA16-defined subset (reconstructed from LIDC)" if luna16 else "LIDC"
    print(f"\n[{label}] Wrote {len(patients)} scans and {len(nodules)} nodules "
          f"to {prefix}_*.csv")
    if luna16:
        print("  (LUNA16 published: 888 scans, 1186 nodules; ~1% more here as "
              "slice-spacing-consistency exclusion is not applied)")
    print("Malignancy tiers:", {t: tiers.count(t) for t in
                                 ("benign_low", "indeterminate", "suspicious_high")})
    print(f"Inter-reader malignancy spread: mean={statistics.mean(spreads):.2f}, "
          f">=2 in {sum(s >= 2 for s in spreads)}/{len(spreads)} nodules")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--luna16", action="store_true",
                    help="Reconstruct the LUNA16 cohort definition (<=2.5mm scans, >=3 readers) from LIDC")
    ap.add_argument("--max-scans", type=int, default=None)
    args = ap.parse_args()
    write_cohort(Path(__file__).resolve().parent, max_scans=args.max_scans, luna16=args.luna16)
