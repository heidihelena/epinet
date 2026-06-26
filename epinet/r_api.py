"""Tabular adapter for language bindings (notably the ``epinetR`` R package).

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

import pandas as pd

from epinet import contest as econtest
from epinet import toolkit


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
