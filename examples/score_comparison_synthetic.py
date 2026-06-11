"""Full Brock vs Mayo vs NTOG comparison on the synthetic cohort.

The real LIDC cohort cannot run these scores (no demographics). The synthetic
cohort (build_nodule_cohort.py) carries the clinical inputs they need, plus an
INDEPENDENT latent malignancy label (`LatentMalignant`) generated from a
different functional form than any score, so the comparison is not circular.

Scores compared (per nodule):
  * Brock  - PanCan malignancy probability (established)
  * Mayo   - Swensen pretest probability (established)
  * NTOG   - post-CT research score, Nodule + Growth domains only, normalised by
             available weight per the NTOG rule ("missing domains not imputed").
             N = banded max(Brock,Mayo); G = banded VDT. P/S/E are unavailable
             even here and are honestly omitted.

Reported:
  1. Rank concordance (Spearman) between the three scores - needs no truth.
  2. ROC-AUC vs the latent label, with bootstrap CIs and pairwise differences.
  3. Where NTOG departs from Brock (the Growth contribution).

HONEST CAVEAT, stated up front: AUCs here measure recovery of THIS generator's
latent label. They demonstrate the scoring machinery and the scores' agreement;
they CANNOT adjudicate real-world validity, which is a property of the generator
choice. A demographically complete real cohort (e.g. NLST) is required for that.

Usage:
  python score_comparison_synthetic.py --dir examples
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_comparison import auc_ci, auc_diff  # noqa: E402


def ntog_nodule_domain(prob_pct: float) -> int:
    for hi, pts in [(1, 1), (5, 4), (10, 10), (30, 18), (65, 28)]:
        if prob_pct < hi:
            return pts
    return 35


def ntog_growth_domain(vdt_days: float) -> int:
    if vdt_days >= 9999:            # sentinel: stable / no growth
        return 0
    if vdt_days < 100:
        return 15
    if vdt_days < 400:
        return 13
    if vdt_days < 600:
        return 10
    return 8                        # measured but slow growth


def ntog_post_ct(brock: float, mayo: float, vdt_days: float) -> float:
    """N + G domains only, normalised to 0-100 by available weight (50)."""
    n = ntog_nodule_domain(100 * max(brock, mayo))
    g = ntog_growth_domain(vdt_days)
    available_max = 35 + 15
    return 100.0 * (n + g) / available_max


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="examples")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    here = Path(args.dir)

    prov = pd.read_csv(here / "nodule_risk_scores.csv")
    prov["NTOG"] = [ntog_post_ct(b, m, v) for b, m, v in
                    zip(prov["BrockProb"], prov["MayoProb"], prov["GrowthVDTdays"])]
    prov = prov.rename(columns={"BrockProb": "Brock", "MayoProb": "Mayo"})
    y = prov["LatentMalignant"].to_numpy()

    scores = ["Brock", "Mayo", "NTOG"]
    report = {"n": int(len(prov)), "n_latent_malignant": int(y.sum())}

    # 1. Rank concordance (no truth needed).
    report["spearman"] = prov[scores].corr(method="spearman").round(3).to_dict()

    # 2. AUC vs latent truth.
    report["auc"] = {s: auc_ci(y, prov[s].to_numpy(), args.n_boot, 42) for s in scores}
    report["pairwise_delta_auc"] = {
        "NTOG_minus_Brock": auc_diff(y, prov["NTOG"].to_numpy(), prov["Brock"].to_numpy(), args.n_boot, 42),
        "NTOG_minus_Mayo": auc_diff(y, prov["NTOG"].to_numpy(), prov["Mayo"].to_numpy(), args.n_boot, 42),
        "Brock_minus_Mayo": auc_diff(y, prov["Brock"].to_numpy(), prov["Mayo"].to_numpy(), args.n_boot, 42),
    }

    # 3. Where NTOG departs from Brock: the growth domain.
    nt_rank = prov["NTOG"].rank()
    br_rank = prov["Brock"].rank()
    growing = prov["GrowthVDTdays"] < 9999
    report["ntog_vs_brock_rank_shift_corr_with_growthrate"] = round(float(
        np.corrcoef((nt_rank - br_rank)[growing],
                    -prov["GrowthVDTdays"][growing])[0, 1]), 3)

    (here / "score_comparison_synthetic.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"n={report['n']} nodules, latent malignant={report['n_latent_malignant']} "
          f"({100*report['n_latent_malignant']/report['n']:.0f}%)\n")
    print("Rank concordance (Spearman):")
    for a in scores:
        print("  " + "  ".join(f"{report['spearman'][a][b]:+.2f}" for b in scores) + f"   {a}")
    print("\nDiscrimination vs INDEPENDENT latent truth (generator-dependent):")
    for s in scores:
        a = report["auc"][s]
        print(f"  {s:6s} AUC {a['auc']:.3f}  CI[{a['ci'][0]:.3f}, {a['ci'][1]:.3f}]")
    print("\nPairwise AUC differences:")
    for name, d in report["pairwise_delta_auc"].items():
        print(f"  {name:18s} ΔAUC={d['delta_auc']:+.3f} CI[{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}] p={d['p_two_sided']}")
    print(f"\nNTOG-vs-Brock rank shift tracks growth rate: r="
          f"{report['ntog_vs_brock_rank_shift_corr_with_growthrate']} "
          f"(faster growth -> NTOG ranks higher than Brock, as designed)")


if __name__ == "__main__":
    main()
