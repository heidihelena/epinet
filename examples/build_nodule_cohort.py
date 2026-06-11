"""Generate a synthetic pulmonary-nodule cohort as an EpiNet network.

This is a faithful Python port of the published nodule risk models implemented
in the NTOG static lung-risk tools (ntog.org), used here to *generate* a
synthetic cohort that the EpiNet feature-space clustering can analyse. It is the
bridge between the NTOG nodule work (feature vector -> risk score) and EpiNet's
centroid clustering (feature vector -> distance to risk-tier centroid).

IMPORTANT: every patient and nodule here is synthetic, sampled from plausible
distributions. No real patient data is used. The risk-model *coefficients* are
the published ones (cited below); the Python port should be cross-checked
against the live NTOG tool before any clinical or publication use.

Risk models reproduced
-----------------------
Brock / PanCan parsimonious full model (McWilliams et al., NEJM 2013):
    x = -6.7892 + 0.0287*(age-62) + 0.6011*female + 0.2961*family_history
        + 0.2953*emphysema - 5.3854*((diameter_mm/10)^-0.5 - 1.58113883)
        + type_term + 0.6581*upper_lobe - 0.0824*(nodule_count-4)
        + 0.7729*spiculation
    P = 1 / (1 + exp(-x))
    type_term: solid = 0 (ref), part-solid = +0.377, non-solid = -0.1276

Mayo / Swensen pretest probability (Swensen et al., Arch Intern Med 1997):
    x = -6.8272 + 0.0391*age + 0.7917*ever_smoker
        + 1.3388*prior_extrathoracic_cancer + 0.1274*diameter_mm
        + 1.0407*spiculation + 0.7838*upper_lobe

Volume doubling time (for a follow-up scan):
    VDT = days * ln(2) / ln(volume_ratio)
    growth bands: rapid <100d, suspicious 100-399, slow 400-599, very slow >=600

Network model
-------------
Nodes: patients (scaffold, unlabeled) and nodules (labeled by Brock risk tier).
Edges: membership (patient-nodule) and co-occurrence (nodule-nodule in the same
patient). A nodule's degree therefore encodes multifocality, and a community
(patient-aware) split holds whole patients out of evaluation.

Risk tier (the modelled outcome) is set from the Brock probability using an
illustrative three-band scheme (low < 0.10, intermediate 0.10-0.50, high >= 0.50;
NOT a clinical standard - BTS/Fleischner use a single ~10% action threshold).
The clustering uses only the *raw* features (morphology + patient factors); the
Brock/Mayo probabilities are written to a separate provenance file so the
centroid analysis is not circular.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

TYPE_TERM = {"solid": 0.0, "part_solid": 0.377, "non_solid": -0.1276}
TYPE_LABELS = list(TYPE_TERM)


def brock_probability(*, age, female, family_history, emphysema, diameter_mm,
                      nodule_type, upper_lobe, nodule_count, spiculation) -> float:
    x = (
        -6.7892
        + 0.0287 * (age - 62)
        + 0.6011 * female
        + 0.2961 * family_history
        + 0.2953 * emphysema
        - 5.3854 * ((diameter_mm / 10) ** -0.5 - 1.58113883)
        + TYPE_TERM[nodule_type]
        + 0.6581 * upper_lobe
        - 0.0824 * (nodule_count - 4)
        + 0.7729 * spiculation
    )
    return 1.0 / (1.0 + math.exp(-x))


def mayo_probability(*, age, ever_smoker, prior_cancer, diameter_mm,
                     spiculation, upper_lobe) -> float:
    x = (
        -6.8272
        + 0.0391 * age
        + 0.7917 * ever_smoker
        + 1.3388 * prior_cancer
        + 0.1274 * diameter_mm
        + 1.0407 * spiculation
        + 0.7838 * upper_lobe
    )
    return 1.0 / (1.0 + math.exp(-x))


def volume_doubling_time(diameter_mm: float, prior_diameter_mm: float, days: float) -> float | None:
    if prior_diameter_mm <= 0 or diameter_mm <= prior_diameter_mm:
        return None  # stable or shrinking -> no doubling time
    volume_ratio = (diameter_mm / prior_diameter_mm) ** 3
    return days * math.log(2) / math.log(volume_ratio)


def latent_malignancy_prob(*, diameter_mm, spiculation, upper_lobe, age,
                           ever_smoker, nodule_type, family_history) -> float:
    """An INDEPENDENT synthetic ground-truth probability of malignancy.

    Deliberately a *different functional form* from Brock and Mayo (saturating
    sqrt-diameter rather than Brock's inverse-sqrt or Mayo's linear term, and its
    own weights) so that no score has privileged access to the label. It shares
    predictors with the scores only because they describe the same biology. With
    Bernoulli sampling this yields a noisy label none of the scores can fit
    exactly -- a fair, if generator-dependent, target.
    """
    x = (
        -4.2
        + 0.62 * math.sqrt(diameter_mm)
        + 0.85 * spiculation
        + 0.45 * upper_lobe
        + 0.035 * (age - 60)
        + 0.55 * ever_smoker
        + 0.7 * (nodule_type == "part_solid")
        - 0.5 * (nodule_type == "non_solid")
        + 0.45 * family_history
    )
    return 1.0 / (1.0 + math.exp(-x))


def risk_tier(brock_p: float) -> str:
    # Illustrative three-band scheme for this demo (NOT a clinical standard;
    # BTS/Fleischner use a single Brock action threshold around 10%).
    if brock_p < 0.10:
        return "low"
    if brock_p < 0.50:
        return "intermediate"
    return "high"


def sample_cohort(n_patients: int = 70, seed: int = 7):
    rng = np.random.default_rng(seed)
    # Independent stream for the latent label so it does not perturb the cohort
    # sampling (features stay identical whether or not the label is drawn).
    latent_rng = np.random.default_rng(seed + 1000)
    patients, nodules = [], []

    for p in range(n_patients):
        age = int(np.clip(rng.normal(67, 9), 40, 90))
        female = int(rng.random() < 0.45)
        family_history = int(rng.random() < 0.15)
        emphysema = int(rng.random() < 0.35)
        ever_smoker = int(rng.random() < 0.82)
        prior_cancer = int(rng.random() < 0.10)
        count = int(np.clip(1 + rng.poisson(0.6), 1, 5))

        pid = f"PT_{p:03d}"
        patients.append({
            "pid": pid, "age": age, "female": female,
            "family_history": family_history, "emphysema": emphysema,
            "ever_smoker": ever_smoker, "prior_cancer": prior_cancer, "count": count,
        })

        for k in range(count):
            # ~28% are larger "index" lesions; the rest are small incidentals.
            if rng.random() < 0.28:
                diameter = float(np.clip(rng.lognormal(mean=math.log(21), sigma=0.5), 6, 40))
            else:
                diameter = float(np.clip(rng.lognormal(mean=math.log(7), sigma=0.45), 4, 25))
            nodule_type = TYPE_LABELS[int(rng.choice(len(TYPE_LABELS), p=[0.70, 0.18, 0.12]))]
            # Spiculation and upper-lobe location correlate with size.
            spiculation = int(rng.random() < min(0.7, 0.08 + diameter / 30))
            upper_lobe = int(rng.random() < 0.60)

            # Synthetic follow-up scan for VDT: larger/spiculated nodules grow faster.
            interval_days = int(rng.uniform(120, 400))
            growth_factor = 1.0 + max(0.0, rng.normal(0.10 + 0.20 * spiculation, 0.12))
            prior_diameter = diameter / growth_factor
            vdt = volume_doubling_time(diameter, prior_diameter, interval_days)

            brock = brock_probability(
                age=age, female=female, family_history=family_history,
                emphysema=emphysema, diameter_mm=diameter, nodule_type=nodule_type,
                upper_lobe=upper_lobe, nodule_count=count, spiculation=spiculation,
            )
            mayo = mayo_probability(
                age=age, ever_smoker=ever_smoker, prior_cancer=prior_cancer,
                diameter_mm=diameter, spiculation=spiculation, upper_lobe=upper_lobe,
            )
            p_true = latent_malignancy_prob(
                diameter_mm=diameter, spiculation=spiculation, upper_lobe=upper_lobe,
                age=age, ever_smoker=ever_smoker, nodule_type=nodule_type,
                family_history=family_history,
            )
            latent_malignant = int(latent_rng.random() < p_true)
            nodules.append({
                "nid": f"ND_{p:03d}_{k}", "pid": pid, "tier": risk_tier(brock),
                "LatentMalignant": latent_malignant,
                "DiameterMm": round(diameter, 1),
                "TypePartSolid": int(nodule_type == "part_solid"),
                "TypeNonSolid": int(nodule_type == "non_solid"),
                "Spiculation": spiculation, "UpperLobe": upper_lobe,
                "NoduleCount": count,
                "GrowthVDTdays": round(vdt, 1) if vdt is not None else 9999,
                "Age": age, "Female": female, "Emphysema": emphysema,
                "FamilyHistory": family_history, "EverSmoker": ever_smoker,
                "BrockProb": round(brock, 4), "MayoProb": round(mayo, 4),
                "NoduleType": nodule_type,
            })

    return patients, nodules


def write_cohort(out_dir: Path) -> None:
    patients, nodules = sample_cohort()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Nodes: patients (unlabeled scaffold) + nodules (labeled by risk tier).
    # Raw features only on nodules; Brock/Mayo kept out of the modelled set.
    feature_cols = ["DiameterMm", "TypePartSolid", "TypeNonSolid", "Spiculation",
                    "UpperLobe", "NoduleCount", "GrowthVDTdays", "Age", "Female",
                    "Emphysema", "FamilyHistory", "EverSmoker"]
    node_fields = ["ID", "NodeType", "Outcome", "Label"] + feature_cols
    with (out_dir / "nodule_nodes.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=node_fields)
        w.writeheader()
        for pt in patients:
            w.writerow({"ID": pt["pid"], "NodeType": "Patient", "Outcome": "",
                        "Label": f"patient age {pt['age']}",
                        **{c: "" for c in feature_cols}})
        for nd in nodules:
            w.writerow({"ID": nd["nid"], "NodeType": "Nodule", "Outcome": nd["tier"],
                        "Label": f"{nd['NoduleType']} {nd['DiameterMm']}mm",
                        **{c: nd[c] for c in feature_cols}})

    # Edges: membership (patient-nodule) and co-occurrence (nodule-nodule).
    with (out_dir / "nodule_edges.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["SourceID", "TargetID", "Relationship", "Weight"])
        w.writeheader()
        by_patient: dict[str, list[str]] = {}
        for nd in nodules:
            w.writerow({"SourceID": nd["pid"], "TargetID": nd["nid"],
                        "Relationship": "has_nodule", "Weight": 1.0})
            by_patient.setdefault(nd["pid"], []).append(nd["nid"])
        for siblings in by_patient.values():
            for i in range(len(siblings)):
                for j in range(i + 1, len(siblings)):
                    w.writerow({"SourceID": siblings[i], "TargetID": siblings[j],
                                "Relationship": "same_patient", "Weight": 1.0})

    # Provenance: the computed risk scores + independent latent ground truth,
    # kept out of the modelled features.
    with (out_dir / "nodule_risk_scores.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["ID", "NoduleType", "DiameterMm",
                                           "BrockProb", "MayoProb", "GrowthVDTdays",
                                           "RiskTier", "LatentMalignant"])
        w.writeheader()
        for nd in nodules:
            w.writerow({"ID": nd["nid"], "NoduleType": nd["NoduleType"],
                        "DiameterMm": nd["DiameterMm"], "BrockProb": nd["BrockProb"],
                        "MayoProb": nd["MayoProb"], "GrowthVDTdays": nd["GrowthVDTdays"],
                        "RiskTier": nd["tier"], "LatentMalignant": nd["LatentMalignant"]})

    tiers = [nd["tier"] for nd in nodules]
    lat = [nd["LatentMalignant"] for nd in nodules]
    print(f"Wrote {len(patients)} patients and {len(nodules)} nodules.")
    print("Risk tiers:", {t: tiers.count(t) for t in ("low", "intermediate", "high")})
    print(f"Latent malignant prevalence: {sum(lat)}/{len(lat)} ({100*sum(lat)/len(lat):.1f}%)")


if __name__ == "__main__":
    write_cohort(Path(__file__).resolve().parent)
