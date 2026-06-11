"""Build the Nordic lung cancer quality-indicator capability network.

This emits two CSVs (`nordic_lung_cancer_qi_nodes.csv`,
`nordic_lung_cancer_qi_edges.csv`) describing lung cancer diagnostic- and
treatment-pathway quality indicators (QIs) across the five Nordic countries as
a network the EpiNet toolkit can analyse.

Sources
-------
- The per-country capture matrix (which registry currently measures which
  indicator) is transcribed from a curated Nordic Lung Cancer Diagnostic
  Pathway QI table (DLCR Arsrapport 2023; Swedish Lung Cancer Registry manual;
  Norwegian national lung cancer quality register; Finnish Cancer Registry;
  Icelandic Cancer Registry).
- Country five-year relative survival is approximate, from NORDCAN 2.0 / the
  Acta Oncologica registry-comparison literature (2016-2020 cohorts).

Model
-----
Three node types:
  * Country  - the five national registries (scaffold; no outcome label)
  * DataSource - the registry/linkage infrastructure an indicator draws on
                 (scaffold; no outcome label)
  * Indicator - a quality indicator, labeled by *feasibility tier*

Edges:
  * captures  : Country -> Indicator, present where that country currently
                measures (or directly derives) the indicator. Weight encodes
                measurement quality (explicit definition > derivable).
  * requires  : Indicator -> DataSource, the data infrastructure it needs.

Feasibility tier (the modelled outcome) is set purely by capture breadth:
  * broad   - measured by >= 4 of 5 countries
  * partial - measured by 2-3 countries
  * gap     - measured by <= 1 country

Crucially, the *edges* that generate an indicator's graph features come from
shared data infrastructure and country capture, while the clinical *features*
(Donabedian type, linkage need, modern-content flag) are independent of the
tier. The interesting question the toolkit then answers is whether an
indicator's position in the capability network predicts its feasibility tier,
and which indicators are misclassified - the harmonization targets.
"""

from __future__ import annotations

import csv
from pathlib import Path

# country_id -> (label, five-year relative survival %, dedicated lung registry?, capture quality weight)
COUNTRIES = {
    "DK_DLCR": ("Denmark (DLCR)", 24.0, 1, 1.00),
    "SE_SLCR": ("Sweden (Lungcancerregistret)", 24.0, 1, 0.90),
    "NO_NLCR": ("Norway (lung cancer quality register)", 27.0, 1, 0.75),
    "FI_FICAN": ("Finland (Finnish Cancer Registry / FICAN)", 17.0, 0, 0.60),
    "IS_ICR": ("Iceland (Icelandic Cancer Registry)", 22.0, 0, 0.60),
}

DATA_SOURCES = {
    "DS_ClinicalRegistry": "Clinical lung cancer quality registry",
    "DS_CancerRegistry": "Population cancer registry",
    "DS_PathologyRegistry": "Pathology registry (margin / morphology)",
    "DS_HospitalDischarge": "Hospital discharge / procedure codes",
    "DS_MortalityLinkage": "Civil registration / cause-of-death linkage",
    "DS_RadiotherapyRegistry": "Radiotherapy registry / module",
    "DS_SystemicTherapyData": "Systemic therapy administration / prescription data",
    "DS_PROMmodule": "Patient-reported outcome (PROM) module",
    "DS_PalliativeServices": "Specialist palliative care service data",
}

# indicator_id -> dict(label, stage, donabedian_outcome, requires_linkage,
#                       modern_content, invasive_data, countries[], sources[])
INDICATORS = {
    "IND_StageCompleteness": dict(
        label="Stage completeness (valid stage recorded)", stage="Staging",
        outcome_type=0, requires_linkage=0, modern_content=0, invasive_data=0,
        countries=["DK_DLCR", "SE_SLCR"],
        sources=["DS_ClinicalRegistry", "DS_CancerRegistry", "DS_PathologyRegistry"]),
    "IND_TimeToDiagnosis": dict(
        label="Time to diagnosis (referral to diagnosis)", stage="Diagnosis",
        outcome_type=0, requires_linkage=0, modern_content=0, invasive_data=0,
        countries=["DK_DLCR", "SE_SLCR"],
        sources=["DS_ClinicalRegistry", "DS_HospitalDischarge"]),
    "IND_TimeToTreatment": dict(
        label="Time to treatment (diagnosis to first treatment)", stage="Timeliness",
        outcome_type=0, requires_linkage=0, modern_content=0, invasive_data=0,
        countries=["DK_DLCR", "SE_SLCR"],
        sources=["DS_ClinicalRegistry", "DS_HospitalDischarge",
                 "DS_RadiotherapyRegistry", "DS_SystemicTherapyData"]),
    "IND_MDT_Documented": dict(
        label="MDT discussion documented", stage="TreatmentPlanning",
        outcome_type=0, requires_linkage=0, modern_content=0, invasive_data=0,
        countries=["DK_DLCR", "SE_SLCR"],
        sources=["DS_ClinicalRegistry", "DS_HospitalDischarge"]),
    "IND_SurgeryRate_EarlyNSCLC": dict(
        label="Surgery rate in early-stage NSCLC (I-II)", stage="Surgery",
        outcome_type=0, requires_linkage=0, modern_content=0, invasive_data=0,
        countries=["DK_DLCR", "SE_SLCR"],
        sources=["DS_ClinicalRegistry", "DS_HospitalDischarge", "DS_PathologyRegistry"]),
    "IND_R0_ResectionRate": dict(
        label="R0 resection rate (clear margin)", stage="Surgery",
        outcome_type=1, requires_linkage=0, modern_content=0, invasive_data=1,
        countries=[],
        sources=["DS_ClinicalRegistry", "DS_HospitalDischarge", "DS_PathologyRegistry"]),
    "IND_30dPostopMortality": dict(
        label="30-day postoperative mortality", stage="SurgeryOutcomes",
        outcome_type=1, requires_linkage=1, modern_content=0, invasive_data=0,
        countries=["DK_DLCR", "SE_SLCR"],
        sources=["DS_HospitalDischarge", "DS_MortalityLinkage"]),
    "IND_PostopComplications": dict(
        label="Postoperative complications / reoperation", stage="Complications",
        outcome_type=1, requires_linkage=0, modern_content=0, invasive_data=0,
        countries=["DK_DLCR"],
        sources=["DS_HospitalDischarge"]),
    "IND_RadiotherapyUtil": dict(
        label="Radiotherapy utilisation", stage="Radiotherapy",
        outcome_type=0, requires_linkage=0, modern_content=0, invasive_data=0,
        countries=["DK_DLCR", "SE_SLCR"],
        sources=["DS_RadiotherapyRegistry", "DS_CancerRegistry", "DS_ClinicalRegistry"]),
    "IND_SystemicTherapy_IV": dict(
        label="Systemic therapy utilisation in stage IV NSCLC", stage="SystemicTherapy",
        outcome_type=0, requires_linkage=0, modern_content=1, invasive_data=0,
        countries=["DK_DLCR", "SE_SLCR"],
        sources=["DS_SystemicTherapyData", "DS_ClinicalRegistry", "DS_CancerRegistry"]),
    "IND_CurativeIntentRate": dict(
        label="Curative-intent treatment rate", stage="Treatment",
        outcome_type=0, requires_linkage=0, modern_content=0, invasive_data=0,
        countries=["DK_DLCR", "NO_NLCR", "SE_SLCR"],
        sources=["DS_ClinicalRegistry", "DS_HospitalDischarge", "DS_RadiotherapyRegistry"]),
    "IND_SmokingStatusRecorded": dict(
        label="Smoking status recorded", stage="Diagnosis",
        outcome_type=0, requires_linkage=0, modern_content=0, invasive_data=0,
        countries=["DK_DLCR", "SE_SLCR"],
        sources=["DS_ClinicalRegistry"]),
    "IND_EarlyMortality_30_90": dict(
        label="Early mortality after diagnosis (30/90-day)", stage="Outcomes",
        outcome_type=1, requires_linkage=1, modern_content=0, invasive_data=0,
        countries=["DK_DLCR", "FI_FICAN", "IS_ICR", "NO_NLCR", "SE_SLCR"],
        sources=["DS_CancerRegistry", "DS_MortalityLinkage"]),
    "IND_1yrOS": dict(
        label="1-year overall survival after diagnosis", stage="Outcomes",
        outcome_type=1, requires_linkage=1, modern_content=0, invasive_data=0,
        countries=["DK_DLCR", "FI_FICAN", "IS_ICR", "NO_NLCR", "SE_SLCR"],
        sources=["DS_CancerRegistry", "DS_MortalityLinkage"]),
    "IND_5yrOS": dict(
        label="5-year overall survival after diagnosis", stage="Outcomes",
        outcome_type=1, requires_linkage=1, modern_content=0, invasive_data=0,
        countries=["DK_DLCR", "FI_FICAN", "IS_ICR", "NO_NLCR", "SE_SLCR"],
        sources=["DS_CancerRegistry", "DS_MortalityLinkage"]),
    "IND_PROM_Capture": dict(
        label="PROM capture / completion", stage="PROMs",
        outcome_type=0, requires_linkage=0, modern_content=1, invasive_data=0,
        countries=["DK_DLCR", "NO_NLCR"],
        sources=["DS_PROMmodule", "DS_ClinicalRegistry"]),
    "IND_PalliativeReferralTiming": dict(
        label="Palliative care referral timing", stage="PalliativeCare",
        outcome_type=0, requires_linkage=1, modern_content=0, invasive_data=0,
        countries=[],
        sources=["DS_PalliativeServices", "DS_HospitalDischarge", "DS_MortalityLinkage"]),
}

CAPTURE_QUALITY = {cid: weight for cid, (_, _, _, weight) in COUNTRIES.items()}
REQUIRES_WEIGHT = 0.8


def feasibility_tier(n_countries: int) -> str:
    if n_countries >= 4:
        return "broad"
    if n_countries >= 2:
        return "partial"
    return "gap"


def build_rows() -> tuple[list[dict], list[dict]]:
    node_rows: list[dict] = []
    for cid, (label, survival, dedicated, _weight) in COUNTRIES.items():
        node_rows.append({
            "ID": cid, "NodeType": "Country", "Outcome": "", "Label": label,
            "PathwayStage": "", "OutcomeType": "", "RequiresLinkage": "",
            "ModernContent": "", "InvasiveData": "",
            "FiveYearSurvival": survival, "DedicatedRegistry": dedicated,
        })
    for sid, label in DATA_SOURCES.items():
        node_rows.append({
            "ID": sid, "NodeType": "DataSource", "Outcome": "", "Label": label,
            "PathwayStage": "", "OutcomeType": "", "RequiresLinkage": "",
            "ModernContent": "", "InvasiveData": "",
            "FiveYearSurvival": "", "DedicatedRegistry": "",
        })
    for iid, spec in INDICATORS.items():
        node_rows.append({
            "ID": iid, "NodeType": "Indicator",
            "Outcome": feasibility_tier(len(spec["countries"])),
            "Label": spec["label"], "PathwayStage": spec["stage"],
            "OutcomeType": spec["outcome_type"],
            "RequiresLinkage": spec["requires_linkage"],
            "ModernContent": spec["modern_content"],
            "InvasiveData": spec["invasive_data"],
            "FiveYearSurvival": "", "DedicatedRegistry": "",
        })

    edge_rows: list[dict] = []
    for iid, spec in INDICATORS.items():
        for cid in spec["countries"]:
            edge_rows.append({
                "SourceID": cid, "TargetID": iid, "Relationship": "captures",
                "Weight": CAPTURE_QUALITY[cid],
            })
        for sid in spec["sources"]:
            edge_rows.append({
                "SourceID": iid, "TargetID": sid, "Relationship": "requires",
                "Weight": REQUIRES_WEIGHT,
            })
    return node_rows, edge_rows


def main() -> None:
    here = Path(__file__).resolve().parent
    node_rows, edge_rows = build_rows()

    node_fields = ["ID", "NodeType", "Outcome", "Label", "PathwayStage",
                   "OutcomeType", "RequiresLinkage", "ModernContent",
                   "InvasiveData", "FiveYearSurvival", "DedicatedRegistry"]
    with (here / "nordic_lung_cancer_qi_nodes.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=node_fields)
        writer.writeheader()
        writer.writerows(node_rows)

    edge_fields = ["SourceID", "TargetID", "Relationship", "Weight"]
    with (here / "nordic_lung_cancer_qi_edges.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=edge_fields)
        writer.writeheader()
        writer.writerows(edge_rows)

    tiers = [r["Outcome"] for r in node_rows if r["NodeType"] == "Indicator"]
    print(f"Wrote {len(node_rows)} nodes and {len(edge_rows)} edges.")
    print("Indicator feasibility tiers:",
          {tier: tiers.count(tier) for tier in ("broad", "partial", "gap")})


if __name__ == "__main__":
    main()
