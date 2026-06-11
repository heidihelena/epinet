"""Numerical validation of the nodule risk-model port.

Before any nodule result is trustworthy, the Python port in
``build_nodule_cohort.py`` must be shown to reproduce the risk models it claims
to implement. This harness validates the port three independent ways:

1. **Source equivalence.** An independent, verbatim transcription of the formula
   from the NTOG static tool (``ntog_lungrisk_static_tools_v3.html``) is
   evaluated on a large random grid of inputs and compared to the port. They
   must agree to floating-point tolerance. This proves "the port == the tool".

2. **Literature anchoring.** Each model coefficient is a published log-odds, so
   ``exp(coefficient)`` must equal the odds ratio reported for the Brock/PanCan
   (McWilliams et al., NEJM 2013) and Mayo/Swensen (Arch Intern Med 1997)
   models. This proves "the tool's coefficients == the published models".

3. **Worked cases and properties.** Hand-traceable cases (a defined linear
   predictor -> probability), the size-term zero point, monotonicity in
   diameter, the nodule-type ordering, and a hand example of volume doubling
   time. This catches silent drift in either direction.

Run directly for a printed report; the same checks run in the test suite.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_nodule_cohort as bnc  # noqa: E402

# --- 1. Independent verbatim transcription of the NTOG tool formula ----------
# Transcribed coefficient-by-coefficient from ntog_lungrisk_static_tools_v3.html
# (kept separate from the port so the two cannot share a bug).

REFERENCE_TYPE_TERM = {"solid": 0.0, "part_solid": 0.377, "non_solid": -0.1276}


def reference_brock(*, age, female, family_history, emphysema, diameter_mm,
                    nodule_type, upper_lobe, nodule_count, spiculation) -> float:
    lp = (
        -6.7892
        + 0.0287 * (age - 62)
        + 0.6011 * female
        + 0.2961 * family_history
        + 0.2953 * emphysema
        - 5.3854 * (math.pow(diameter_mm / 10.0, -0.5) - 1.58113883)
        + REFERENCE_TYPE_TERM[nodule_type]
        + 0.6581 * upper_lobe
        - 0.0824 * (nodule_count - 4)
        + 0.7729 * spiculation
    )
    return 1.0 / (1.0 + math.exp(-lp))


def reference_mayo(*, age, ever_smoker, prior_cancer, diameter_mm,
                   spiculation, upper_lobe) -> float:
    lp = (
        -6.8272
        + 0.0391 * age
        + 0.7917 * ever_smoker
        + 1.3388 * prior_cancer
        + 0.1274 * diameter_mm
        + 1.0407 * spiculation
        + 0.7838 * upper_lobe
    )
    return 1.0 / (1.0 + math.exp(-lp))


# --- 2. Published odds ratios (exp of each log-odds coefficient) --------------
# Brock full model with spiculation (McWilliams 2013) and Mayo (Swensen 1997).
PUBLISHED_ODDS_RATIOS = {
    "brock": {
        "female": (0.6011, 1.82),
        "family_history": (0.2961, 1.34),
        "emphysema": (0.2953, 1.34),
        "upper_lobe": (0.6581, 1.93),
        "spiculation": (0.7729, 2.17),
        "part_solid_vs_solid": (0.377, 1.46),
    },
    "mayo": {
        "ever_smoker": (0.7917, 2.21),
        "prior_cancer": (1.3388, 3.81),
        "spiculation": (1.0407, 2.83),
        "upper_lobe": (0.7838, 2.19),
    },
}


def check_source_equivalence(n: int = 5000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    max_brock_err = max_mayo_err = 0.0
    for _ in range(n):
        age = float(rng.uniform(40, 90))
        diameter = float(rng.uniform(4, 40))
        ntype = rng.choice(["solid", "part_solid", "non_solid"])
        female = int(rng.integers(0, 2))
        family = int(rng.integers(0, 2))
        emph = int(rng.integers(0, 2))
        smoker = int(rng.integers(0, 2))
        prior = int(rng.integers(0, 2))
        upper = int(rng.integers(0, 2))
        spic = int(rng.integers(0, 2))
        count = int(rng.integers(1, 6))

        port_b = bnc.brock_probability(
            age=age, female=female, family_history=family, emphysema=emph,
            diameter_mm=diameter, nodule_type=ntype, upper_lobe=upper,
            nodule_count=count, spiculation=spic)
        ref_b = reference_brock(
            age=age, female=female, family_history=family, emphysema=emph,
            diameter_mm=diameter, nodule_type=ntype, upper_lobe=upper,
            nodule_count=count, spiculation=spic)
        port_m = bnc.mayo_probability(
            age=age, ever_smoker=smoker, prior_cancer=prior,
            diameter_mm=diameter, spiculation=spic, upper_lobe=upper)
        ref_m = reference_mayo(
            age=age, ever_smoker=smoker, prior_cancer=prior,
            diameter_mm=diameter, spiculation=spic, upper_lobe=upper)
        max_brock_err = max(max_brock_err, abs(port_b - ref_b))
        max_mayo_err = max(max_mayo_err, abs(port_m - ref_m))
    return {"n": n, "max_brock_abs_error": max_brock_err, "max_mayo_abs_error": max_mayo_err}


def check_odds_ratios(tol: float = 0.02) -> dict:
    results = {}
    for model, terms in PUBLISHED_ODDS_RATIOS.items():
        for name, (coef, published_or) in terms.items():
            computed = math.exp(coef)
            results[f"{model}.{name}"] = {
                "coef": coef,
                "computed_or": round(computed, 3),
                "published_or": published_or,
                "ok": abs(computed - published_or) <= max(tol, 0.01 * published_or),
            }
    return results


def check_worked_cases() -> dict:
    out = {}
    # Size-term zero point: at 4 mm the diameter term vanishes by construction,
    # since (0.4)^-0.5 == 1.58113883.
    out["size_term_zero_at_4mm"] = abs(math.pow(0.4, -0.5) - 1.58113883) < 1e-7

    # A fully hand-traced Brock case. Solid 8 mm nodule, age 62 (centered to 0),
    # male, no risk factors, lower lobe, no spiculation, count 4 (centered to 0):
    #   lp = -6.7892 - 5.3854*((0.8)^-0.5 - 1.58113883)
    d_term = -5.3854 * (math.pow(0.8, -0.5) - 1.58113883)
    lp_expected = -6.7892 + d_term
    p_expected = 1.0 / (1.0 + math.exp(-lp_expected))
    p_port = bnc.brock_probability(
        age=62, female=0, family_history=0, emphysema=0, diameter_mm=8,
        nodule_type="solid", upper_lobe=0, nodule_count=4, spiculation=0)
    out["worked_brock_8mm"] = {
        "expected_p": round(p_expected, 6),
        "port_p": round(p_port, 6),
        "ok": abs(p_expected - p_port) < 1e-9,
    }

    # Nodule-type ordering: part-solid > solid > non-solid at fixed everything.
    common = dict(age=65, female=0, family_history=0, emphysema=0, diameter_mm=12,
                  upper_lobe=1, nodule_count=1, spiculation=0)
    p_solid = bnc.brock_probability(nodule_type="solid", **common)
    p_part = bnc.brock_probability(nodule_type="part_solid", **common)
    p_non = bnc.brock_probability(nodule_type="non_solid", **common)
    out["type_ordering_partsolid_solid_nonsolid"] = p_part > p_solid > p_non

    # Monotonic increase in diameter.
    diams = [5, 10, 15, 20, 30]
    probs = [bnc.brock_probability(nodule_type="solid", age=65, female=0,
             family_history=0, emphysema=0, diameter_mm=d, upper_lobe=1,
             nodule_count=1, spiculation=0) for d in diams]
    out["monotonic_in_diameter"] = all(b > a for a, b in zip(probs, probs[1:]))

    # VDT hand example: a doubling-of-volume over 100 days has VDT == 100 days;
    # a 4x volume increase over 300 days has VDT == 150 days.
    out["vdt_doubling_100d"] = abs(
        bnc.volume_doubling_time(diameter_mm=2 ** (1 / 3), prior_diameter_mm=1.0, days=100) - 100
    ) < 1e-6
    out["vdt_quadruple_300d"] = abs(
        bnc.volume_doubling_time(diameter_mm=4 ** (1 / 3), prior_diameter_mm=1.0, days=300) - 150
    ) < 1e-6
    return out


def run_all() -> dict:
    return {
        "source_equivalence": check_source_equivalence(),
        "odds_ratios": check_odds_ratios(),
        "worked_cases": check_worked_cases(),
    }


def main() -> None:
    report = run_all()
    eq = report["source_equivalence"]
    print("1. Source equivalence (port vs verbatim NTOG formula)")
    print(f"   {eq['n']} random inputs | max |Δ| Brock = {eq['max_brock_abs_error']:.2e}, "
          f"Mayo = {eq['max_mayo_abs_error']:.2e}")
    print(f"   -> {'PASS' if max(eq['max_brock_abs_error'], eq['max_mayo_abs_error']) < 1e-9 else 'FAIL'}")

    print("\n2. Coefficient -> published odds ratio")
    all_or_ok = True
    for name, r in report["odds_ratios"].items():
        flag = "ok" if r["ok"] else "MISMATCH"
        all_or_ok &= r["ok"]
        print(f"   {name:28s} exp({r['coef']:+.4f}) = {r['computed_or']:.2f} "
              f"vs published {r['published_or']:.2f}  [{flag}]")
    print(f"   -> {'PASS' if all_or_ok else 'FAIL'}")

    print("\n3. Worked cases and properties")
    wc = report["worked_cases"]
    for name, val in wc.items():
        ok = val["ok"] if isinstance(val, dict) else val
        detail = f" (expected {val['expected_p']}, port {val['port_p']})" if isinstance(val, dict) and "port_p" in val else ""
        print(f"   {name:38s} {'PASS' if ok else 'FAIL'}{detail}")


if __name__ == "__main__":
    main()
