# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Heidi Andersén

"""Registry schema adapter: map a flat clinical / registry export onto EpiNet's
canonical node/edge schema, driven by a declarative profile.

Registries and MDT exports are flat tables — one row per case, many coded
columns — not graphs. This adapter turns such a table into canonical nodes
(``ID``, ``Outcome``, numeric features) and edges (built by a chosen strategy),
so an export drops straight into EpiNet without bespoke glue.

It FORMATS only. It does not compute risk, stage, or treatment and makes no
clinical decision — those are out of scope (and out of regulatory tier). Every
run emits a manifest of exactly what it did, plus a content hash of the source.

Federation tie-in: the profile IS the shared feature contract. When every site
adapts its own export with the SAME profile, the per-site feature tables are
column-compatible by construction — which is precisely the precondition the
federated fit (``epinet_federated``) requires. So one profile drives both the
local ingest and the cross-site federation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from vahtian.epinet import cluster as epinet_cluster
from vahtian.epinet import common as epinet_common

EDGE_COLUMNS = ["SourceID", "TargetID", "Weight"]


@dataclass
class RegistryProfile:
    """Declarative mapping from a registry export to the canonical schema.

    ``feature_columns=None`` auto-selects the numeric columns (minus id/outcome).
    ``edge_strategy`` is one of ``knn`` (k-nearest-neighbour similarity graph in
    standardized feature space), ``shared`` (link cases sharing a value in
    ``shared_column``), or ``none`` (isolated nodes).
    """

    id_column: str
    outcome_column: str | None = None
    feature_columns: list[str] | None = None
    edge_strategy: str = "knn"
    knn_k: int = 5
    shared_column: str | None = None
    id_prefix: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "RegistryProfile":
        """Build a profile from a plain dict (e.g. a JSON registry profile)."""
        allowed = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown profile keys: {sorted(unknown)}")
        return cls(**data)


def _knn_edges(nodes: pd.DataFrame, feature_columns: list[str], k: int) -> tuple[pd.DataFrame, str]:
    from sklearn.neighbors import NearestNeighbors

    Xz, kept = epinet_cluster.standardize(nodes[feature_columns])
    if Xz.shape[0] < 2 or not kept:
        return pd.DataFrame(columns=EDGE_COLUMNS), "no edges (need >=2 cases and a varying feature)"

    k_eff = min(k, Xz.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(Xz)
    dist, idx = nn.kneighbors(Xz)
    ids = nodes["ID"].to_numpy()

    seen: set[tuple[int, int]] = set()
    rows = []
    for i in range(len(ids)):
        for pos in range(1, k_eff + 1):  # position 0 is the point itself
            j = int(idx[i, pos])
            a, b = sorted((i, j))
            if (a, b) in seen:
                continue
            seen.add((a, b))
            rows.append({"SourceID": ids[a], "TargetID": ids[b],
                         "Weight": float(1.0 / (1.0 + dist[i, pos]))})
    note = f"k-NN similarity graph (k={k_eff}), undirected, weight=1/(1+distance)"
    return pd.DataFrame(rows, columns=EDGE_COLUMNS), note


def _shared_edges(nodes: pd.DataFrame, column: str) -> tuple[pd.DataFrame, str]:
    if not column or column not in nodes.columns:
        raise ValueError(f"shared_column not available as a node column: {column!r}")
    ids = nodes["ID"].to_numpy()
    groups: dict[object, list[int]] = defaultdict(list)
    for i, value in enumerate(nodes[column].to_numpy()):
        groups[value].append(i)
    rows = []
    for members in groups.values():
        for ai in range(len(members)):
            for bi in range(ai + 1, len(members)):
                rows.append({"SourceID": ids[members[ai]], "TargetID": ids[members[bi]],
                             "Weight": 1.0})
    note = f"shared-attribute edges on {column!r} (cases with equal values are linked)"
    return pd.DataFrame(rows, columns=EDGE_COLUMNS), note


def adapt(table: pd.DataFrame, profile: RegistryProfile) -> dict[str, object]:
    """Map a flat registry table to canonical nodes/edges + a manifest.

    Returns ``{"nodes": DataFrame, "edges": DataFrame, "manifest": dict}``. Raises
    a clear ``ValueError`` when a declared column is missing or no usable numeric
    feature exists.
    """
    if profile.id_column not in table.columns:
        raise ValueError(f"id column not found: {profile.id_column!r}")
    if profile.outcome_column and profile.outcome_column not in table.columns:
        raise ValueError(f"outcome column not found: {profile.outcome_column!r}")

    reserved = {profile.id_column}
    if profile.outcome_column:
        reserved.add(profile.outcome_column)

    if profile.feature_columns is not None:
        missing = [c for c in profile.feature_columns if c not in table.columns]
        if missing:
            raise ValueError(f"feature columns not found: {missing}")
        feature_columns = list(profile.feature_columns)
    else:
        numeric = table.drop(columns=[c for c in reserved if c in table.columns], errors="ignore")
        feature_columns = list(numeric.select_dtypes(include=[np.number]).columns)
    if not feature_columns:
        raise ValueError("no numeric feature columns found; set feature_columns explicitly")

    ids = (profile.id_prefix + table[profile.id_column].astype(str)).tolist()
    if len(set(ids)) != len(ids):
        raise ValueError("case IDs are not unique after prefixing")

    nodes = pd.DataFrame({"ID": ids})
    if profile.outcome_column:
        outcome = table[profile.outcome_column]
        blank = epinet_common.blank_label_mask(outcome).to_numpy()
        nodes["Outcome"] = np.where(blank, "", outcome.astype("string").fillna("").to_numpy())
    for col in feature_columns:
        nodes[col] = pd.to_numeric(table[col], errors="coerce").fillna(0.0).to_numpy()

    # Carry the shared-attribute column as a node attribute if it is not already
    # a feature, so "shared" edges can reference it.
    if profile.edge_strategy == "shared" and profile.shared_column:
        if profile.shared_column not in nodes.columns:
            if profile.shared_column not in table.columns:
                raise ValueError(f"shared_column not found: {profile.shared_column!r}")
            nodes[profile.shared_column] = table[profile.shared_column].astype("string").to_numpy()

    if profile.edge_strategy == "knn":
        edges, edge_note = _knn_edges(nodes, feature_columns, profile.knn_k)
    elif profile.edge_strategy == "shared":
        edges, edge_note = _shared_edges(nodes, profile.shared_column)
    elif profile.edge_strategy == "none":
        edges, edge_note = pd.DataFrame(columns=EDGE_COLUMNS), "no edges (isolated nodes)"
    else:
        raise ValueError(f"unknown edge_strategy: {profile.edge_strategy!r}")

    dropped = [c for c in table.columns if c not in reserved and c not in feature_columns]
    manifest = {
        "n_cases": int(len(nodes)),
        "feature_columns": feature_columns,
        "outcome_column": profile.outcome_column,
        "edge_strategy": profile.edge_strategy,
        "n_edges": int(len(edges)),
        "edge_note": edge_note,
        "dropped_columns": dropped,
        "source_sha256": epinet_common.sha256_frame(table),
        "note": (
            "Formatting only — no risk, stage, or treatment computed; no clinical "
            "decision made. Edges are a similarity scaffold, not a clinical relationship."
        ),
    }
    return {"nodes": nodes, "edges": edges, "manifest": manifest}
