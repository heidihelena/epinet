"""Tabular adapter for language bindings (notably the ``epinetR`` R package).

EpiNet's core is graph-shaped, but many callers — especially from R — have an
ordinary table: one row per subject, an outcome column, and predictor columns.
``fit`` is a thin, **feature-space** entry point over that shape: it builds a
design matrix from the named predictors (one-hot encoding non-numeric ones) and
runs the same honestly-evaluated outcome model the rest of the toolkit uses
(imbalance-aware tuning, calibration, bootstrap CI, permutation null, importance).

It returns a plain, JSON-friendly ``dict`` so a binding layer (reticulate, etc.)
can wrap it in a native object without reaching into pandas/numpy types. This is
deliberately the only surface the R package depends on, so the algorithms stay
single-sourced in tested Python and cannot silently diverge across languages.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from epinet import toolkit


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

    Parameters mirror the R ``epinet()`` call: ``data`` is a table (anything
    ``pandas.DataFrame`` accepts — an R ``data.frame`` arrives this way through
    reticulate), ``outcome`` is the label column, and ``predictors`` is the list
    of feature columns (default: every column except the outcome). Non-numeric
    predictors are one-hot encoded. Rows with a missing outcome are dropped.

    Returns a dict with ``outcome``, ``predictors``, ``features_used``, ``n``,
    the full ``metrics`` summary (discrimination, classification, calibration,
    bootstrap CI, permutation null, data warnings), and ``importance`` (a list of
    per-feature records).
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

    # Drop rows without an outcome (the supervised target must be present).
    work = data[[*predictors, outcome]].copy()
    work = work[work[outcome].notna()]
    if work.empty:
        raise ValueError("no rows with a non-missing outcome")

    # Design matrix: numeric predictors pass through; non-numeric are one-hot
    # encoded so categorical predictors (sex, smoking, …) are usable directly.
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
    nodes = X_enc.reset_index(drop=True).copy()
    nodes.insert(0, "ID", ids)
    nodes[outcome] = work[outcome].to_numpy()
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

    importance = result["importance"]
    return {
        "outcome": outcome,
        "predictors": predictors,
        "features_used": list(X_enc.columns),
        "n": int(len(work)),
        "metrics": result["metrics"],
        "importance": importance.to_dict(orient="records"),
    }
