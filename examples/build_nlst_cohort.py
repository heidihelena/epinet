"""Ingestion harness for the NLST dataset (run when CDAS data arrives).

NLST is NOT public: it is released only through the NCI Cancer Data Access
System (https://cdas.cancer.gov) after an approved data request and signed data
transfer agreement. This module does not and cannot download it. What it does is
turn the approved NLST CSVs into the same node/edge cohort format the rest of
this toolkit consumes, so the existing analyses (score_comparison_synthetic,
test_fusion, pathology_validation, epinet centroids) run with one command.

Why NLST is the cohort the project has been building toward: it carries the
inputs every earlier example lacked together in one place —
  * demographics + smoking + COPD + family history  -> Brock, Mayo, PLCOm2012,
    and the NTOG Person/Smoking/Emerging domains all become computable
  * per-nodule morphology (size, attenuation, margins, lobe)
  * three annual screens (T0/T1/T2) -> real volume-doubling-time growth
  * confirmed lung-cancer outcomes (truth) with adequate benign representation

IMPORTANT: the column names below are best-effort from published NLST
documentation and MUST be confirmed against the data dictionary of the specific
dataset version you receive (CDAS ships versioned datasets, e.g. the "NLST 780"
release). They are isolated in COLUMN_MAP so reconciling them is a config edit,
not a rewrite. Nothing here fabricates data; run `--demo` to exercise the
transform on a synthetic NLST-shaped fixture.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# --- Column map: VERIFY against your CDAS data dictionary before a real run. ---
COLUMN_MAP = {
    "participant": {
        "pid": "pid",                 # participant id
        "age": "age",                 # age at randomization
        "female": "gender",           # NLST gender: 1=male, 2=female (recoded below)
        "pack_years": "pkyr",         # pack-years
        "smoke_years": "smokeyr",     # years smoked
        "cig_per_day": "smokeday",    # cigarettes/day
        "current_smoker": "cigsmok",  # 1=current, 0=former
        "age_quit": "age_quit",       # age stopped smoking
        "copd": "diagcopd",           # COPD/emphysema history
        "family_history": "famslc",   # family history of lung cancer (1=yes)
        "prior_cancer": "cancblad",   # placeholder: any prior cancer flag (verify)
        "lung_cancer": "canclung",    # confirmed lung cancer (1=yes)
        "cancer_days": "candx_days",  # days from randomization to diagnosis
    },
    "abnormalities": {            # spiral-CT non-calcified nodule abnormalities
        "pid": "pid",
        "study_year": "study_yr",     # 0=T0, 1=T1, 2=T2
        "diameter_mm": "sct_long_dia",  # longest diameter (mm); verify exact name
        "attenuation": "sct_ab_attn",   # 1=soft tissue/solid, 2=ground glass, 3=mixed
        "margins": "sct_ab_marg",       # margin code
        "lobe": "sct_epi_loc",          # location code (upper-lobe derived below)
    },
}

# NLST attenuation code -> our nodule_type. Verify codes against the dictionary.
ATTENUATION_TO_TYPE = {1: "solid", 2: "non_solid", 3: "part_solid"}
# Upper-lobe location codes (RUL/LUL); verify against sct_epi_loc coding.
UPPER_LOBE_CODES = {1, 4}


def _col(df: pd.DataFrame, table: str, key: str) -> pd.Series:
    name = COLUMN_MAP[table][key]
    if name not in df.columns:
        raise KeyError(f"NLST {table} column '{name}' (for '{key}') not found; "
                       f"edit COLUMN_MAP to match your data dictionary.")
    return df[name]


def assemble(participant: pd.DataFrame, abnormalities: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build node/edge/provenance frames from NLST participant + abnormality tables."""
    p = pd.DataFrame({
        "pid": _col(participant, "participant", "pid").astype(str),
        "Age": pd.to_numeric(_col(participant, "participant", "age"), errors="coerce"),
        "Female": (pd.to_numeric(_col(participant, "participant", "female"), errors="coerce") == 2).astype(int),
        "PackYears": pd.to_numeric(_col(participant, "participant", "pack_years"), errors="coerce"),
        "SmokeYears": pd.to_numeric(_col(participant, "participant", "smoke_years"), errors="coerce"),
        "CurrentSmoker": pd.to_numeric(_col(participant, "participant", "current_smoker"), errors="coerce").fillna(0).astype(int),
        "Emphysema": pd.to_numeric(_col(participant, "participant", "copd"), errors="coerce").fillna(0).astype(int),
        "FamilyHistory": pd.to_numeric(_col(participant, "participant", "family_history"), errors="coerce").fillna(0).astype(int),
        "EverSmoker": 1,  # NLST enrolled current/former heavy smokers by design
        "LungCancer": pd.to_numeric(_col(participant, "participant", "lung_cancer"), errors="coerce").fillna(0).astype(int),
    })
    p["Outcome"] = np.where(p["LungCancer"] == 1, "suspicious_high", "benign_low")

    a = pd.DataFrame({
        "pid": _col(abnormalities, "abnormalities", "pid").astype(str),
        "study_yr": pd.to_numeric(_col(abnormalities, "abnormalities", "study_year"), errors="coerce"),
        "DiameterMm": pd.to_numeric(_col(abnormalities, "abnormalities", "diameter_mm"), errors="coerce"),
        "attn": pd.to_numeric(_col(abnormalities, "abnormalities", "attenuation"), errors="coerce"),
        "lobe": pd.to_numeric(_col(abnormalities, "abnormalities", "lobe"), errors="coerce"),
    }).dropna(subset=["DiameterMm"])
    a["NoduleType"] = a["attn"].map(ATTENUATION_TO_TYPE).fillna("solid")
    a["TypePartSolid"] = (a["NoduleType"] == "part_solid").astype(int)
    a["TypeNonSolid"] = (a["NoduleType"] == "non_solid").astype(int)
    a["UpperLobe"] = a["lobe"].isin(UPPER_LOBE_CODES).astype(int)
    a["Spiculation"] = 0  # NLST abnormality table has no spiculation flag; left 0

    # Growth (real VDT) when the same participant has a nodule at consecutive
    # screens: pair each participant's largest nodule across study years.
    a = a.sort_values(["pid", "study_yr"])
    grow = a.groupby("pid").agg(d_first=("DiameterMm", "first"),
                                d_last=("DiameterMm", "last"),
                                yr_first=("study_yr", "first"),
                                yr_last=("study_yr", "last")).reset_index()
    grow["GrowthVDTdays"] = 9999.0
    mask = (grow["yr_last"] > grow["yr_first"]) & (grow["d_last"] > grow["d_first"])
    days = (grow["yr_last"] - grow["yr_first"]) * 365.0
    ratio = (grow["d_last"] / grow["d_first"]) ** 3
    grow.loc[mask, "GrowthVDTdays"] = (days * np.log(2) / np.log(ratio))[mask].round(1)

    # One nodule node per participant-nodule; index by largest per study year.
    a["nid"] = a["pid"] + "_y" + a["study_yr"].astype(int).astype(str) + "_" + a.groupby(["pid", "study_yr"]).cumcount().astype(str)
    nod = a.merge(p, on="pid", how="left").merge(grow[["pid", "GrowthVDTdays"]], on="pid", how="left")
    nod["NoduleCount"] = nod.groupby("pid")["nid"].transform("count")

    feature_cols = ["DiameterMm", "TypePartSolid", "TypeNonSolid", "Spiculation",
                    "UpperLobe", "NoduleCount", "GrowthVDTdays", "Age", "Female",
                    "Emphysema", "FamilyHistory", "EverSmoker"]

    nodes_rows = []
    for _, r in p.iterrows():
        nodes_rows.append({"ID": r["pid"], "NodeType": "Participant", "Outcome": "",
                           "Label": f"participant {r['pid']}", **{c: "" for c in feature_cols}})
    for _, r in nod.iterrows():
        nodes_rows.append({"ID": r["nid"], "NodeType": "Nodule", "Outcome": r["Outcome"],
                           "Label": f"{r['NoduleType']} {r['DiameterMm']}mm",
                           **{c: r.get(c, "") for c in feature_cols}})
    nodes = pd.DataFrame(nodes_rows, columns=["ID", "NodeType", "Outcome", "Label"] + feature_cols)

    edges_rows = []
    for pid_, sib in nod.groupby("pid")["nid"]:
        sib = list(sib)
        for nid in sib:
            edges_rows.append({"SourceID": pid_, "TargetID": nid, "Relationship": "has_nodule", "Weight": 1.0})
        for i in range(len(sib)):
            for j in range(i + 1, len(sib)):
                edges_rows.append({"SourceID": sib[i], "TargetID": sib[j], "Relationship": "same_participant", "Weight": 1.0})
    edges = pd.DataFrame(edges_rows, columns=["SourceID", "TargetID", "Relationship", "Weight"])

    provenance = nod[["nid", "pid", "Outcome", "DiameterMm", "NoduleType", "GrowthVDTdays",
                      "Age", "Female", "PackYears", "SmokeYears", "CurrentSmoker",
                      "Emphysema", "FamilyHistory", "LungCancer"]].rename(columns={"nid": "ID"})
    return {"nodes": nodes, "edges": edges, "provenance": provenance}


def demo_fixture(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic NLST-shaped tables to exercise the transform without real data."""
    rng = np.random.default_rng(seed)
    n = 40
    participant = pd.DataFrame({
        "pid": [f"P{i:04d}" for i in range(n)],
        "age": rng.integers(55, 75, n),
        "gender": rng.integers(1, 3, n),
        "pkyr": rng.integers(30, 90, n),
        "smokeyr": rng.integers(20, 50, n),
        "smokeday": rng.integers(10, 40, n),
        "cigsmok": rng.integers(0, 2, n),
        "age_quit": rng.integers(40, 70, n),
        "diagcopd": rng.integers(0, 2, n),
        "famslc": rng.integers(0, 2, n),
        "cancblad": 0,
        "canclung": (rng.random(n) < 0.25).astype(int),
        "candx_days": rng.integers(100, 2000, n),
    })
    rows = []
    for i in range(n):
        for yr in range(rng.integers(1, 4)):
            rows.append({"pid": f"P{i:04d}", "study_yr": yr,
                         "sct_long_dia": float(np.clip(rng.lognormal(2.2, 0.5), 4, 40)),
                         "sct_ab_attn": rng.integers(1, 4), "sct_ab_marg": rng.integers(1, 4),
                         "sct_epi_loc": rng.integers(1, 7)})
    return participant, pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--participant", help="NLST participant CSV (from CDAS)")
    ap.add_argument("--abnormalities", help="NLST spiral-CT abnormality CSV (from CDAS)")
    ap.add_argument("--output", default="examples")
    ap.add_argument("--demo", action="store_true", help="Run on a synthetic NLST-shaped fixture")
    args = ap.parse_args()

    if args.demo:
        participant, abnormalities = demo_fixture()
    elif args.participant and args.abnormalities:
        participant = pd.read_csv(args.participant)
        abnormalities = pd.read_csv(args.abnormalities)
    else:
        ap.error("provide --participant and --abnormalities CSVs, or --demo")

    frames = assemble(participant, abnormalities)
    out = Path(args.output)
    frames["nodes"].to_csv(out / "nlst_nodes.csv", index=False)
    frames["edges"].to_csv(out / "nlst_edges.csv", index=False)
    frames["provenance"].to_csv(out / "nlst_provenance.csv", index=False)

    nd = frames["nodes"][frames["nodes"]["NodeType"] == "Nodule"]
    print(f"Wrote {len(frames['nodes'])} nodes, {len(frames['edges'])} edges, "
          f"{len(nd)} nodules.")
    print("Outcome:", nd["Outcome"].value_counts().to_dict())
    print("\nNext: compute Brock/Mayo/NTOG from nlst_provenance.csv (demographics now\n"
          "present) and run score_comparison + test_fusion against the LungCancer label.")


if __name__ == "__main__":
    main()
