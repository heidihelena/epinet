"""Parallel comparison of imaging-computable nodule scores vs pathology truth.

IMPORTANT honest-scope note. The established clinical risk models (Brock, Mayo)
and the NTOG research scores cannot be computed on LIDC-IDRI: they require age,
sex, smoking, family history, emphysema, lobe location, longitudinal growth, and
environmental/host factors, NONE of which are in LIDC's public annotations. This
module therefore does NOT fabricate those inputs. It compares only the predictors
that LIDC genuinely supports, against the real TCIA pathology diagnosis:

  * gestalt   - median radiologist malignancy rating (the established LIDC
                reference standard; a holistic human judgement)
  * size      - nodule diameter (the established imaging benchmark; Fleischner/BTS)
  * lit_morph - a transparent, literature-weighted morphology composite built
                only from LIDC semantic features, with documented directions:
                  + diameter, + spiculation, + lobulation, - margin (sharp=benign),
                  + calcification-absent (code 6). Standardised and summed.
                This is the imaging-feature-evidence analogue of the NTOG
                feature philosophy, restricted to what LIDC measures.

Each score is computed on the patient's index nodule (highest median-reader
malignancy) and evaluated against the pathology label by ROC-AUC with bootstrap
95% CIs; pairwise AUC differences are bootstrapped on paired resamples.

To run the FULL Brock/Mayo/NTOG comparison you need a cohort with demographics;
build_nodule_cohort.py provides a synthetic one for that purpose.

Usage:
  python score_comparison.py --diagnosis <tcia .xls/.csv> \
      --nodes examples/lidc_nodes.csv --provenance examples/lidc_provenance.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathology_validation import HISTOPATH_METHODS, load_diagnosis  # noqa: E402


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 1e-9 else s * 0.0


def build_scores(nodes: pd.DataFrame, prov: pd.DataFrame) -> pd.DataFrame:
    feats = nodes[nodes["NodeType"] == "Nodule"].merge(
        prov[["ID", "pid", "MalignancyMedian"]], on="ID", how="inner")
    # Index nodule per patient = highest median-reader malignancy.
    idx = feats.sort_values(["MalignancyMedian", "DiameterMm"]).groupby("pid").tail(1).copy()

    idx["gestalt"] = idx["MalignancyMedian"]
    idx["size"] = idx["DiameterMm"]
    idx["lit_morph"] = (
        zscore(idx["DiameterMm"])
        + zscore(idx["spiculation"])
        + zscore(idx["lobulation"])
        - zscore(idx["margin"])
        + (idx["calcification"] == 6).astype(float)
    )
    return idx[["pid", "gestalt", "size", "lit_morph"]]


def auc_ci(y: np.ndarray, score: np.ndarray, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    point = roc_auc_score(y, score)
    boots = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], score[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"auc": round(float(point), 3), "ci": [round(float(lo), 3), round(float(hi), 3)],
            "n_boot": len(boots)}


def auc_diff(y: np.ndarray, a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> dict:
    """Bootstrap distribution of AUC(a) - AUC(b) on paired resamples."""
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())  # two-sided bootstrap p
    return {"delta_auc": round(float(diffs.mean()), 3),
            "ci": [round(float(lo), 3), round(float(hi), 3)],
            "p_two_sided": round(float(p), 3)}


def evaluate(scores: pd.DataFrame, diag: pd.DataFrame, *, histopath_only: bool,
             n_boot: int, seed: int) -> dict:
    d = diag[diag["pt_dx"].isin({1, 2, 3})].copy()
    if histopath_only:
        d = d[d["pt_method"].isin(HISTOPATH_METHODS)]
    d["y"] = d["pt_dx"].isin({2, 3}).astype(int)
    m = d.merge(scores, on="pid", how="inner")
    y = m["y"].to_numpy()

    cols = ["gestalt", "size", "lit_morph"]
    res = {"n": int(len(m)), "n_malignant": int(y.sum()), "n_benign": int((1 - y).sum()),
           "auc": {c: auc_ci(y, m[c].to_numpy(), n_boot, seed) for c in cols}}
    res["vs_gestalt"] = {
        "size": auc_diff(y, m["size"].to_numpy(), m["gestalt"].to_numpy(), n_boot, seed),
        "lit_morph": auc_diff(y, m["lit_morph"].to_numpy(), m["gestalt"].to_numpy(), n_boot, seed),
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnosis", required=True)
    ap.add_argument("--nodes", default="examples/lidc_nodes.csv")
    ap.add_argument("--provenance", default="examples/lidc_provenance.csv")
    ap.add_argument("--output", default="examples/lidc_pathology_outputs")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    nodes = pd.read_csv(args.nodes)
    prov = pd.read_csv(args.provenance)
    diag = load_diagnosis(Path(args.diagnosis))
    scores = build_scores(nodes, prov)

    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for label, histo in [("all_confirmed", False), ("histopath_only", True)]:
        report[label] = evaluate(scores, diag, histopath_only=histo,
                                 n_boot=args.n_boot, seed=42)
    (out_dir / "score_comparison.json").write_text(json.dumps(report, indent=2) + "\n")

    for label in ("all_confirmed", "histopath_only"):
        r = report[label]
        print(f"\n=== {label}: n={r['n']} ({r['n_malignant']} malignant, {r['n_benign']} benign) ===")
        print(f"  {'score':10s} {'AUC':>6s}  95% CI")
        for c, a in r["auc"].items():
            print(f"  {c:10s} {a['auc']:>6.3f}  [{a['ci'][0]:.3f}, {a['ci'][1]:.3f}]")
        for c, dd in r["vs_gestalt"].items():
            print(f"  {c} - gestalt: ΔAUC={dd['delta_auc']:+.3f} "
                  f"CI[{dd['ci'][0]:+.3f},{dd['ci'][1]:+.3f}] p={dd['p_two_sided']}")


if __name__ == "__main__":
    main()
