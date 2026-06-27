"""Tabular adapter for language bindings (notably the ``vahtian.epinet`` R package).

EpiNet's core is graph-shaped, but many callers — especially from R — have an
ordinary table: one row per subject, an outcome column, and predictor columns.
These adapters are thin, **feature-space** entry points over that shape: they
build a design matrix from the named predictors (one-hot encoding non-numeric
ones) and call the same tested toolkit functions the rest of EpiNet uses.

Every adapter returns a plain, JSON-friendly ``dict`` so a binding layer
(reticulate, etc.) can wrap it without reaching into pandas/numpy types. This is
deliberately the only surface the R package depends on, so the algorithms stay
single-sourced in tested Python and cannot silently diverge across languages.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from vahtian.epinet import contest as econtest
from vahtian.epinet import federated as efed
from vahtian.epinet import toolkit


def _design_matrix(data, outcome: str, predictors=None):
    """Encode a flat table into (X, y, features_used, predictors).

    Numeric predictors pass through; non-numeric predictors are one-hot encoded.
    Rows with a missing outcome are dropped. ``X`` and ``y`` share a string row
    index so downstream toolkit calls line up.
    """
    data = pd.DataFrame(data).copy()
    if outcome not in data.columns:
        raise ValueError(f"outcome column {outcome!r} is not in the data")
    if predictors is None:
        predictors = [c for c in data.columns if c != outcome]
    predictors = list(dict.fromkeys(predictors))  # de-dup, preserve order
    missing = [c for c in predictors if c not in data.columns]
    if missing:
        raise ValueError(f"predictor columns not in the data: {missing}")
    if not predictors:
        raise ValueError("need at least one predictor")

    work = data[[*predictors, outcome]].copy()
    work = work[work[outcome].notna()]
    if work.empty:
        raise ValueError("no rows with a non-missing outcome")

    X = work[predictors]
    numeric = X.select_dtypes(include="number")
    categorical = X.select_dtypes(exclude="number")
    parts = [numeric]
    if not categorical.empty:
        parts.append(pd.get_dummies(categorical.astype("category")).astype(float))
    X_enc = pd.concat(parts, axis=1)
    if X_enc.shape[1] == 0:
        raise ValueError("no usable predictor columns after encoding")

    ids = [str(i) for i in range(len(work))]
    X_enc.index = ids
    y = pd.Series(work[outcome].to_numpy(), index=ids, name=outcome)
    return X_enc, y, list(X_enc.columns), predictors


def fit(
    data,
    outcome: str,
    predictors=None,
    *,
    n_iterations: int = 1,
    n_permutations: int = 0,
    n_bootstrap: int = 1000,
    test_size: float = 0.2,
    random_state: int = 42,
    tune_threshold: bool = False,
) -> dict:
    """Fit EpiNet's honest outcome model on a flat table.

    Returns a dict with ``outcome``, ``predictors``, ``features_used``, ``n``,
    the full ``metrics`` summary (discrimination, classification, calibration,
    bootstrap CI, permutation null, data warnings), and ``importance``.
    """
    X_enc, y, features_used, predictors = _design_matrix(data, outcome, predictors)

    ids = list(X_enc.index)
    nodes = X_enc.reset_index(drop=True).copy()
    nodes.insert(0, "ID", ids)
    nodes[outcome] = y.to_numpy()
    features = pd.DataFrame({"ID": ids})

    with tempfile.TemporaryDirectory() as tmp:
        result = toolkit.train_outcome_model(
            nodes,
            features,
            id_column="ID",
            outcome_column=outcome,
            output_dir=Path(tmp),
            n_iterations=n_iterations,
            n_permutations=n_permutations,
            n_bootstrap=n_bootstrap,
            test_size=test_size,
            random_state=random_state,
            tune_threshold=tune_threshold,
        )

    return {
        "outcome": outcome,
        "predictors": predictors,
        "features_used": features_used,
        "n": int(len(X_enc)),
        "metrics": result["metrics"],
        "importance": result["importance"].to_dict(orient="records"),
    }


def contestability(
    data,
    outcome: str,
    predictors=None,
    *,
    metric: str = "euclidean",
    contest_quantile: float = 0.1,
) -> dict:
    """Score every row's contestability against the outcome-class centroids.

    For each row: the nearest-centroid class, the closed-form flip-distance (how
    far it would have to move to flip class), the runner-up class, and the
    most decision-relevant feature. Returns per-row vectors for plotting plus a
    summary (flip-distance stats, the contested-quantile threshold, and a
    per-feature value-of-information ranking).
    """
    X_enc, y, features_used, predictors = _design_matrix(data, outcome, predictors)
    res = econtest.contestability(
        X_enc, y=y, metric=metric, contest_quantile=contest_quantile
    )
    assignments = res["assignments"]
    summary = res["summary"]
    fd = summary["flip_distance"]
    return {
        "outcome": outcome,
        "predictors": predictors,
        "features_used": features_used,
        "n": int(len(X_enc)),
        "metric": metric,
        "contest_quantile": contest_quantile,
        "flip_distance": [float(v) for v in assignments["flip_distance"].to_numpy()],
        "contested": [bool(v) for v in assignments["contested"].to_numpy()],
        "contest_threshold": fd.get("contest_threshold"),
        "flip_summary": {
            k: fd.get(k) for k in ("mean", "std", "min", "median", "max", "n_contested")
        },
        "feature_voi": {str(k): float(v) for k, v in summary["feature_leverage"].items()},
        "assignments": assignments.to_dict(orient="records"),
        "caveats": summary.get("caveats"),
    }


def graph(
    nodes,
    edges,
    outcome: str,
    *,
    id_column: str = "ID",
    source_column: str = "SourceID",
    target_column: str = "TargetID",
    directed: bool = False,
    weight_column=None,
    include_centrality: bool = False,
    n_iterations: int = 1,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> dict:
    """Build the graph, derive graph features, and fit the honest outcome model.

    Returns the model ``metrics`` and ``importance`` plus everything needed to
    draw the network natively in R: ``nodes`` (id, degree, community, outcome)
    and ``edges`` (source, target). Graph features (degree, clustering,
    component size, optional centrality) join the node attributes to form the
    design matrix, exactly as the rest of EpiNet does.
    """
    nodes = pd.DataFrame(nodes).copy()
    edges = pd.DataFrame(edges).copy()
    if id_column not in nodes.columns:
        raise ValueError(f"id_column {id_column!r} is not in nodes")
    if outcome not in nodes.columns:
        raise ValueError(f"outcome column {outcome!r} is not in nodes")
    for col in (source_column, target_column):
        if col not in edges.columns:
            raise ValueError(f"edge column {col!r} is not in edges")

    g = toolkit.build_graph(
        nodes, edges, id_column=id_column, source_column=source_column,
        target_column=target_column, directed=directed, weight_column=weight_column,
    )
    feats = toolkit.generate_graph_features(g, include_centrality=include_centrality)

    with tempfile.TemporaryDirectory() as tmp:
        result = toolkit.train_outcome_model(
            nodes, feats, id_column=id_column, outcome_column=outcome,
            output_dir=Path(tmp), n_iterations=n_iterations,
            n_bootstrap=n_bootstrap, random_state=random_state,
        )

    communities = toolkit.community_labels(g)
    degrees = dict(g.degree())
    node_ids = nodes[id_column].astype(str).tolist()
    outcomes = nodes[outcome].tolist()
    node_records = [
        {
            "id": nid,
            "degree": float(degrees.get(nid, 0)),
            "community": int(communities.get(nid, -1)),
            "outcome": (None if pd.isna(oc) else str(oc)),
        }
        for nid, oc in zip(node_ids, outcomes)
    ]
    edge_records = [
        {"source": str(s), "target": str(t)}
        for s, t in zip(edges[source_column].astype(str), edges[target_column].astype(str))
    ]

    return {
        "outcome": outcome,
        "id_column": id_column,
        "directed": bool(directed),
        "n_nodes": int(g.number_of_nodes()),
        "n_edges": int(g.number_of_edges()),
        "metrics": result["metrics"],
        "importance": result["importance"].to_dict(orient="records"),
        "feature_columns": [c for c in feats.columns if c != "ID"],
        "nodes": node_records,
        "edges": edge_records,
    }


def federated(
    data,
    outcome: str,
    predictors=None,
    *,
    site=None,
    n_sites: int = 2,
    metric: str = "euclidean",
    contest_quantile: float = 0.1,
    random_state: int = 42,
) -> dict:
    """Federate the fit across sites and check it reconstructs the centralized run.

    Partitions the rows across sites (by the ``site`` column if given, else into
    ``n_sites`` balanced random groups), then runs EpiNet's federated
    reconstruction: only per-site aggregates cross, and the result is compared to
    the centralized fit. Returns the per-site sizes and the max absolute
    fit-reconstruction differences (mean/sd/centroid) plus, when computable, the
    contestability round-trip differences — all of which should be at
    floating-point level, demonstrating the federation is exact.
    """
    data = pd.DataFrame(data).copy()
    X, y, features_used, predictors = _design_matrix(data, outcome, predictors)
    n = len(X)

    if site is not None:
        if site not in data.columns:
            raise ValueError(f"site column {site!r} is not in the data")
        labels = data.loc[data[outcome].notna(), site].astype(str).to_numpy()
        if len(labels) != n:
            raise ValueError("site labels do not align with the labeled rows")
    else:
        n_sites = int(n_sites)
        if n_sites < 2:
            raise ValueError("need at least 2 sites")
        rng = np.random.default_rng(random_state)
        labels = np.empty(n, dtype=object)
        order = rng.permutation(n)
        for rank, idx in enumerate(order):
            labels[idx] = f"site{rank % n_sites + 1}"

    if len(set(labels.tolist())) < 2:
        raise ValueError("need at least 2 distinct sites to federate")

    sim = efed.simulate(X, y, labels)
    out = {
        "outcome": outcome,
        "predictors": predictors,
        "n": n,
        "metric": metric,
        "n_sites": int(len(set(labels.tolist()))),
        "sites": {s: int(info["n"]) for s, info in sim["sites"].items()},
        "fit_diffs": {
            "mean": float(sim["max_mean_diff"]),
            "sd": float(sim["max_sd_diff"]),
            "centroid": float(sim["max_centroid_diff"]),
        },
    }
    try:
        sc = efed.simulate_contestability(
            X, y, labels, metric=metric, contest_quantile=contest_quantile
        )
        out["contestability_diffs"] = {
            "mean": (None if sc["max_mean_diff"] is None else float(sc["max_mean_diff"])),
            "std": (None if sc["max_std_diff"] is None else float(sc["max_std_diff"])),
        }
        out["runner_up_match"] = bool(sc["runner_up_match"])
        out["top_voi_match"] = bool(sc["top_voi_match"])
    except Exception:  # contestability is best-effort here; the fit check is core
        out["contestability_diffs"] = None
    return out
