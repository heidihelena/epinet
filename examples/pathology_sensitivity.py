"""Specification-curve sensitivity analysis for the pathology validation.

The headline finding (radiologist-"indeterminate" patients are mostly malignant
on tissue) rests on three analyst choices. This script sweeps all of them and
reports the range of the key metrics across every specification, so we can see
whether the finding is robust or an artifact of one path through the garden of
forking choices.

Choices swept:
  1. index-nodule aggregation: most-suspicious | largest | mean-malignancy
  2. malignancy -> tier thresholds: wide indeterminate (2.5-3.5) | strict (==3)
  3. pathology "malignant" set: primary+metastatic {2,3} | primary only {2}

Restricted to the histopathology-confirmed subset (biopsy/resection), the
tissue-anchored reference.

Usage:
  python pathology_sensitivity.py --diagnosis <tcia .xls/.csv> \
      --provenance examples/lidc_provenance.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathology_validation import HISTOPATH_METHODS, load_diagnosis, wilson_ci  # noqa: E402

AGGREGATIONS = ("most_suspicious", "largest", "mean_malignancy")
THRESHOLDS = {"wide(2.5-3.5)": (2.5, 3.5), "strict(==3)": (2.999, 3.001)}
PATH_SETS = {"primary+met{2,3}": {2, 3}, "primary_only{2}": {2}}


def tier_of(value: float, lo: float, hi: float) -> str:
    if value < lo:
        return "benign_low"
    if value > hi:
        return "suspicious_high"
    return "indeterminate"


def index_nodule(prov: pd.DataFrame, how: str) -> pd.DataFrame:
    if how == "most_suspicious":
        idx = prov.sort_values(["MalignancyMedian", "DiameterMm"]).groupby("pid").tail(1)
        idx = idx.assign(rad_value=idx["MalignancyMedian"])
    elif how == "largest":
        idx = prov.sort_values(["DiameterMm", "MalignancyMedian"]).groupby("pid").tail(1)
        idx = idx.assign(rad_value=idx["MalignancyMedian"])
    elif how == "mean_malignancy":
        m = prov.groupby("pid")["MalignancyMedian"].mean().rename("rad_value").reset_index()
        idx = m
    else:
        raise ValueError(how)
    return idx[["pid", "rad_value"]]


def run_spec(prov: pd.DataFrame, diag: pd.DataFrame, agg: str,
             thr: tuple[float, float], path_set: set[int]) -> dict:
    d = diag[diag["pt_method"].isin(HISTOPATH_METHODS)].copy()
    d = d[d["pt_dx"].isin({1, 2, 3})]
    d["path_malignant"] = d["pt_dx"].isin(path_set)
    # When metastatic is excluded, dx==3 patients are dropped (neither pos nor a
    # clean benign), to avoid mislabelling them benign.
    if path_set == {2}:
        d = d[d["pt_dx"] != 3]

    idx = index_nodule(prov, agg)
    m = d.merge(idx, on="pid", how="inner")
    m["rad_tier"] = m["rad_value"].map(lambda v: tier_of(v, *thr))

    mal = m["path_malignant"]
    n_mal, n_ben = int(mal.sum()), int((~mal).sum())
    ind = m[m["rad_tier"] == "indeterminate"]
    hedge_n = len(ind)
    hedge_mal = int(ind["path_malignant"].sum())

    pos_sh = m["rad_tier"] == "suspicious_high"
    pos_ih = m["rad_tier"].isin(["indeterminate", "suspicious_high"])
    return {
        "aggregation": agg, "thresholds": thr, "path_set": sorted(path_set),
        "n": len(m), "n_malignant": n_mal, "n_benign": n_ben,
        "hedge_n": hedge_n, "hedge_malignant": hedge_mal,
        "hedge_frac": round(hedge_mal / hedge_n, 3) if hedge_n else None,
        "hedge_ci": [round(x, 3) for x in wilson_ci(hedge_mal, hedge_n)] if hedge_n else None,
        "sens_suspicious_only": round(int((pos_sh & mal).sum()) / n_mal, 3) if n_mal else None,
        "sens_indeterminate_or_high": round(int((pos_ih & mal).sum()) / n_mal, 3) if n_mal else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnosis", required=True)
    ap.add_argument("--provenance", default="examples/lidc_provenance.csv")
    ap.add_argument("--output", default="examples/lidc_pathology_outputs")
    args = ap.parse_args()

    prov = pd.read_csv(args.provenance)
    diag = load_diagnosis(Path(args.diagnosis))
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)

    specs = []
    for agg, (tname, thr), (pname, pset) in itertools.product(
            AGGREGATIONS, THRESHOLDS.items(), PATH_SETS.items()):
        r = run_spec(prov, diag, agg, thr, pset)
        r["threshold_name"], r["path_name"] = tname, pname
        specs.append(r)

    df = pd.DataFrame(specs)
    df.to_csv(out_dir / "pathology_sensitivity.csv", index=False)
    (out_dir / "pathology_sensitivity.json").write_text(json.dumps(specs, indent=2) + "\n")

    print(f"{'aggregation':16s} {'thresh':12s} {'path':16s} "
          f"{'n':>4s} {'hedge':>12s} {'hedgeFrac':>9s} {'sensSH':>7s} {'sensIH':>7s}")
    for r in specs:
        hedge = f"{r['hedge_malignant']}/{r['hedge_n']}"
        print(f"{r['aggregation']:16s} {r['threshold_name']:12s} {r['path_name']:16s} "
              f"{r['n']:>4d} {hedge:>12s} {str(r['hedge_frac']):>9s} "
              f"{str(r['sens_suspicious_only']):>7s} {str(r['sens_indeterminate_or_high']):>7s}")

    fr = [r["hedge_frac"] for r in specs if r["hedge_frac"] is not None]
    ss = [r["sens_suspicious_only"] for r in specs if r["sens_suspicious_only"] is not None]
    print(f"\nAcross {len(specs)} specifications:")
    print(f"  hedge-bucket malignant fraction: {min(fr):.2f} - {max(fr):.2f} (median {sorted(fr)[len(fr)//2]:.2f})")
    print(f"  sensitivity (suspicious-only):   {min(ss):.2f} - {max(ss):.2f} "
          f"-> {sum(s < 0.75 for s in ss)}/{len(ss)} specs below 0.75")


if __name__ == "__main__":
    main()
