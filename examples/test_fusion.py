"""Multi-test centroid fusion: what each test sees, and whether combining helps.

Treats each risk test as a separate view of a case and asks two things against a
truth label: (1) where do the tests diverge, and (2) does combining them beat the
best single test? Two fusion methods are compared:

  * centroid  - standardise each test (orient higher = more malignant) and
                average. This is the equal-weight axis through the class
                centroids in test-space; it is label-free, so there is no
                fitting and no leakage.
  * logistic  - a cross-validated logistic combination of the tests
                (out-of-fold predictions only, so the AUC is honest).

Run on two cohorts:
  * REAL LIDC tissue truth, with the only predictors LIDC supports (radiologist
    gestalt, diameter/size, a literature morphology composite). Brock/Mayo/NTOG
    are NOT computable on LIDC and are absent here.
  * SYNTHETIC cohort, with the full Brock + Mayo + NTOG against the independent
    latent label (generator-dependent; demonstrates machinery, not validity).

Usage:
  python test_fusion.py --diagnosis <tcia .xls/.csv> --dir examples
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathology_validation import load_diagnosis  # noqa: E402
from score_comparison import auc_ci, auc_diff, build_scores  # noqa: E402
from score_comparison_synthetic import ntog_post_ct  # noqa: E402


def _z(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return StandardScaler().fit_transform(df[cols].to_numpy(dtype=float))


def centroid_fusion(df: pd.DataFrame, tests: list[str]) -> np.ndarray:
    """Equal-weight average of standardised tests (label-free)."""
    return _z(df, tests).mean(axis=1)


def logistic_oof(df: pd.DataFrame, tests: list[str], y: np.ndarray,
                 *, splits: int = 5, repeats: int = 20, seed: int = 0) -> np.ndarray:
    """Out-of-fold predicted probabilities from a CV logistic combination."""
    X = df[tests].to_numpy(dtype=float)
    oof = np.zeros((repeats, len(y)))
    rskf = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=seed)
    rep = 0
    for i, (tr, te) in enumerate(rskf.split(X, y)):
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000).fit(scaler.transform(X[tr]), y[tr])
        oof[rep, te] = clf.predict_proba(scaler.transform(X[te]))[:, 1]
        if (i + 1) % splits == 0:
            rep += 1
    return oof.mean(axis=0)


def evaluate(df: pd.DataFrame, tests: list[str], y: np.ndarray, *, n_boot: int) -> dict:
    df = df.reset_index(drop=True)
    res = {"n": int(len(y)), "n_positive": int(y.sum()), "tests": tests}

    # Single tests.
    res["auc"] = {t: auc_ci(y, df[t].to_numpy(dtype=float), n_boot, 42) for t in tests}
    best = max(tests, key=lambda t: res["auc"][t]["auc"])
    res["best_single"] = best

    # Fusions.
    fused_c = centroid_fusion(df, tests)
    fused_l = logistic_oof(df, tests, y)
    res["auc"]["centroid_fusion"] = auc_ci(y, fused_c, n_boot, 42)
    res["auc"]["logistic_fusion"] = auc_ci(y, fused_l, n_boot, 42)

    res["fusion_vs_best_single"] = {
        "centroid": auc_diff(y, fused_c, df[best].to_numpy(dtype=float), n_boot, 42),
        "logistic": auc_diff(y, fused_l, df[best].to_numpy(dtype=float), n_boot, 42),
    }

    # Where the tests disagree: not all on the same side of their own mean.
    z = _z(df, tests)
    signs = np.sign(z)
    disagree = ~(np.all(signs >= 0, axis=1) | np.all(signs <= 0, axis=1))
    res["disagreement"] = {
        "n_disagree": int(disagree.sum()),
        "fraction": round(float(disagree.mean()), 3),
    }
    if disagree.sum() >= 10 and len(np.unique(y[disagree])) == 2:
        res["disagreement"]["best_single_auc"] = round(
            float(roc_auc_score(y[disagree], df[best].to_numpy(dtype=float)[disagree])), 3)
        res["disagreement"]["centroid_fusion_auc"] = round(
            float(roc_auc_score(y[disagree], fused_c[disagree])), 3)
        res["disagreement"]["logistic_fusion_auc"] = round(
            float(roc_auc_score(y[disagree], fused_l[disagree])), 3)
    return res


def real_cohort(diagnosis: Path, here: Path) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    nodes = pd.read_csv(here / "lidc_nodes.csv")
    prov = pd.read_csv(here / "lidc_provenance.csv")
    diag = load_diagnosis(diagnosis)
    scores = build_scores(nodes, prov)
    d = diag[diag["pt_dx"].isin({1, 2, 3})].copy()
    d["y"] = d["pt_dx"].isin({2, 3}).astype(int)
    m = d.merge(scores, on="pid", how="inner")
    return m, ["gestalt", "size", "lit_morph"], m["y"].to_numpy()


def synthetic_cohort(here: Path) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    prov = pd.read_csv(here / "nodule_risk_scores.csv")
    prov["NTOG"] = [ntog_post_ct(b, mo, v) for b, mo, v in
                    zip(prov["BrockProb"], prov["MayoProb"], prov["GrowthVDTdays"])]
    prov = prov.rename(columns={"BrockProb": "Brock", "MayoProb": "Mayo"})
    return prov, ["Brock", "Mayo", "NTOG"], prov["LatentMalignant"].to_numpy()


def report(name: str, r: dict) -> None:
    print(f"\n=== {name}: n={r['n']} ({r['n_positive']} positive) ===")
    for t, a in r["auc"].items():
        star = "  <- best single" if t == r["best_single"] else ""
        print(f"  {t:18s} AUC {a['auc']:.3f}  CI[{a['ci'][0]:.3f}, {a['ci'][1]:.3f}]{star}")
    for kind, d in r["fusion_vs_best_single"].items():
        print(f"  {kind}_fusion - best single: ΔAUC={d['delta_auc']:+.3f} "
              f"CI[{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}] p={d['p_two_sided']}")
    dz = r["disagreement"]
    print(f"  tests disagree on {dz['n_disagree']}/{r['n']} cases ({dz['fraction']*100:.0f}%)")
    if "best_single_auc" in dz:
        print(f"    on disagreement subset: best_single={dz['best_single_auc']} "
              f"centroid={dz['centroid_fusion_auc']} logistic={dz['logistic_fusion_auc']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnosis", required=True)
    ap.add_argument("--dir", default="examples")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    here = Path(args.dir)

    out = {}
    df, tests, y = real_cohort(Path(args.diagnosis), here)
    out["real_lidc_tissue"] = evaluate(df, tests, y, n_boot=args.n_boot)
    df, tests, y = synthetic_cohort(here)
    out["synthetic_latent"] = evaluate(df, tests, y, n_boot=args.n_boot)

    (here / "lidc_pathology_outputs").mkdir(parents=True, exist_ok=True)
    (here / "lidc_pathology_outputs" / "test_fusion.json").write_text(json.dumps(out, indent=2) + "\n")
    report("REAL LIDC tissue (imaging predictors)", out["real_lidc_tissue"])
    report("SYNTHETIC latent (Brock/Mayo/NTOG)", out["synthetic_latent"])


if __name__ == "__main__":
    main()
