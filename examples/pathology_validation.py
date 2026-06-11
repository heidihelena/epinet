"""Radiologist-vs-pathology validation on the real LIDC-IDRI diagnosis subset.

This replaces the reader-split stand-in with a genuinely lower-variance
reference: the TCIA LIDC-IDRI diagnosis data (tcia-diagnosis-data-2012-04-20),
patient-level, restricted to histopathology confirmation (biopsy or surgical
resection) where requested.

It is deliberately patient-level: the diagnosis sheet's "Nodule 1..5" columns
cannot be reliably linked to pylidc nodule clusters, so matching is by TCIA
patient id only. For each patient the radiologist representation is the
*index nodule* (the nodule with the highest median-reader malignancy), compared
to the patient's tissue diagnosis.

Honest limitations surfaced by the data itself:
- Tissue is obtained mainly when malignancy is suspected, so the histopath
  subset is overwhelmingly malignant (~77 malignant vs ~7 benign here).
  Sensitivity is estimable; specificity rests on a handful of benigns and has a
  very wide interval. The "ground truth" is itself selection-biased.
- Diagnosis method 1 ("2 years radiological stability") is not tissue; including
  it makes the benign reference partly radiological and is reported separately.

Usage:
  python pathology_validation.py --diagnosis <tcia .xls or tidy .csv> \
      --provenance examples/lidc_provenance.csv
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

DX_TO_TIER = {1: "benign", 2: "malignant", 3: "malignant"}  # 0 = unknown -> drop
HISTOPATH_METHODS = {2, 3}  # biopsy, surgical resection


def load_diagnosis(path: Path) -> pd.DataFrame:
    """Read either the raw TCIA .xls or a tidy csv with pid,pt_dx,pt_method."""
    if path.suffix.lower() in {".xls", ".xlsx"}:
        d = pd.read_excel(path, header=None, skiprows=1)
        d = d.rename(columns={0: "pid", 1: "pt_dx", 2: "pt_method"})
        d = d[["pid", "pt_dx", "pt_method"]]
    else:
        d = pd.read_csv(path)[["pid", "pt_dx", "pt_method"]]
    d = d[d["pid"].astype(str).str.startswith("LIDC")].copy()
    d["pt_dx"] = pd.to_numeric(d["pt_dx"], errors="coerce")
    d["pt_method"] = pd.to_numeric(d["pt_method"], errors="coerce")
    return d


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def patient_table(prov: pd.DataFrame, diag: pd.DataFrame, *, histopath_only: bool) -> pd.DataFrame:
    d = diag[diag["pt_dx"].isin(DX_TO_TIER)].copy()
    if histopath_only:
        d = d[d["pt_method"].isin(HISTOPATH_METHODS)]
    d["path_tier"] = d["pt_dx"].map(DX_TO_TIER)

    # Index nodule per patient: highest median-reader malignancy.
    idx = prov.sort_values("MalignancyMedian").groupby("pid").tail(1)
    idx = idx.rename(columns={"MalignancyMedian": "index_malignancy",
                              "tier": "rad_tier", "DiameterMm": "index_diameter"})
    merged = d.merge(idx[["pid", "rad_tier", "index_malignancy", "index_diameter"]],
                     on="pid", how="inner")
    return merged


def analyse(merged: pd.DataFrame) -> dict:
    n = len(merged)
    # 3x2 confusion: radiologist tier vs pathology
    conf = pd.crosstab(merged["rad_tier"], merged["path_tier"]).reindex(
        index=["benign_low", "indeterminate", "suspicious_high"],
        columns=["benign", "malignant"], fill_value=0)

    malignant = merged["path_tier"] == "malignant"
    benign = merged["path_tier"] == "benign"
    n_mal, n_ben = int(malignant.sum()), int(benign.sum())

    # Two operating points for "radiologist positive".
    out = {"n_patients": n, "n_malignant": n_mal, "n_benign": n_ben,
           "confusion_rad_x_path": conf.to_dict(orient="index")}
    for name, positive in {
        "suspicious_high_only": merged["rad_tier"] == "suspicious_high",
        "indeterminate_or_high": merged["rad_tier"].isin(["indeterminate", "suspicious_high"]),
    }.items():
        tp = int((positive & malignant).sum()); fn = int((~positive & malignant).sum())
        tn = int((~positive & benign).sum()); fp = int((positive & benign).sum())
        sens = tp / n_mal if n_mal else float("nan")
        spec = tn / n_ben if n_ben else float("nan")
        out[name] = {
            "sensitivity": round(sens, 3), "sensitivity_ci": [round(x, 3) for x in wilson_ci(tp, n_mal)],
            "specificity": round(spec, 3), "specificity_ci": [round(x, 3) for x in wilson_ci(tn, n_ben)],
            "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        }

    # The hedge-bucket question: of radiologist-"indeterminate" patients, how many were malignant?
    ind = merged[merged["rad_tier"] == "indeterminate"]
    if len(ind):
        mal_in_ind = int((ind["path_tier"] == "malignant").sum())
        out["indeterminate_resolution"] = {
            "n": len(ind), "malignant": mal_in_ind,
            "malignant_fraction": round(mal_in_ind / len(ind), 3),
            "ci": [round(x, 3) for x in wilson_ci(mal_in_ind, len(ind))],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnosis", required=True)
    ap.add_argument("--provenance", default="examples/lidc_provenance.csv")
    ap.add_argument("--output", default="examples/lidc_pathology_outputs")
    args = ap.parse_args()

    prov = pd.read_csv(args.provenance)
    diag = load_diagnosis(Path(args.diagnosis))
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)

    report = {}
    for label, histo in [("histopath_only", True), ("all_confirmed", False)]:
        merged = patient_table(prov, diag, histopath_only=histo)
        report[label] = analyse(merged)
        merged.to_csv(out_dir / f"pathology_patients_{label}.csv", index=False)

    (out_dir / "pathology_validation.json").write_text(json.dumps(report, indent=2) + "\n")

    for label in ("histopath_only", "all_confirmed"):
        r = report[label]
        print(f"\n=== {label}: {r['n_patients']} patients "
              f"({r['n_malignant']} malignant, {r['n_benign']} benign) ===")
        print("radiologist tier x pathology:")
        for tier, row in r["confusion_rad_x_path"].items():
            print(f"  {tier:16s} benign={row.get('benign',0):3d}  malignant={row.get('malignant',0):3d}")
        for op in ("suspicious_high_only", "indeterminate_or_high"):
            s = r[op]
            print(f"  [{op}] sensitivity={s['sensitivity']} CI{s['sensitivity_ci']} | "
                  f"specificity={s['specificity']} CI{s['specificity_ci']} "
                  f"(tn={s['tn']}/{r['n_benign']})")
        if "indeterminate_resolution" in r:
            ir = r["indeterminate_resolution"]
            print(f"  hedge bucket: {ir['malignant']}/{ir['n']} radiologist-indeterminate patients "
                  f"were MALIGNANT ({ir['malignant_fraction']*100:.0f}%, CI{ir['ci']})")


if __name__ == "__main__":
    main()
