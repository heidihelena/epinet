"""Federated fit: reconstruct EpiNet's scaler and class centroids from per-site
aggregates, without pooling any record.

The standardization + nearest-centroid spine is built entirely from *additive
sufficient statistics*: per-feature count / sum / sum-of-squares (for the global
z-score) and per-class count / sum (for the centroids). Sums are additive, so a
coordinator can combine one small aggregate message per site and recover EXACTLY
the scaler and centroids a centralized run would compute — while the records
never leave the site. This is the "federated fit" step that makes a *derived*
federated dataset (features + contestability) coherent across sites.

What this composes
- Global mean and population standard deviation -> the shared scaler.
- Per-class standardized centroids. Because z-scoring is affine, the
  standardized centroid of class c equals (raw_class_mean - global_mean) /
  global_sd, so per-class raw sums + counts are sufficient — no need to ship
  standardized vectors.

What this does NOT do
- It assumes a shared feature contract: the same feature columns, defined the
  same way, at every site. Whether a graph feature is *comparable* across
  differently-structured sites is a modelling question, not a math one.
- The RandomForest outcome model is not mean-poolable and is out of scope. This
  is the centroid / contestability layer only — which is exactly EpiNet's
  differentiated spine.

Disclosure note: the messages here are counts and sums. ``min_cell`` enforces a
minimum per-class count (small-cell suppression) so a sum over one or two
patients cannot leak an individual value. Set it before any real deployment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import epinet_cluster
import epinet_common

VARIANCE_TOL = 1e-12  # matches epinet_cluster.standardize's zero-variance cutoff


def site_aggregates(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    min_cell: int = 0,
) -> dict[str, object]:
    """Compute one site's aggregate message — counts and sums only, no records.

    Global statistics (for the scaler) are taken over every row of ``X``
    (scaffold included, mirroring ``epinet_cluster.standardize``); per-class
    statistics are taken over labeled rows only. Classes with fewer than
    ``min_cell`` labeled rows are suppressed and listed in ``suppressed``.
    """
    columns = list(X.columns)
    values = X.to_numpy(dtype=float)
    n = int(values.shape[0])

    labeled = epinet_common.labeled_mask(y.reindex(X.index))
    y_labeled = y.reindex(X.index)

    class_n: dict[str, int] = {}
    class_sum: dict[str, list[float]] = {}
    suppressed: list[str] = []
    for cls in sorted(y_labeled[labeled].astype(str).unique(), key=str):
        mask = labeled.to_numpy() & (y_labeled.astype(str).to_numpy() == cls)
        count = int(mask.sum())
        if count < min_cell:
            suppressed.append(cls)
            continue
        class_n[cls] = count
        class_sum[cls] = values[mask].sum(axis=0).tolist()

    return {
        "columns": columns,
        "n": n,
        "sum": values.sum(axis=0).tolist(),
        "sumsq": (values**2).sum(axis=0).tolist(),
        "class_n": class_n,
        "class_sum": class_sum,
        "suppressed": suppressed,
    }


def combine_aggregates(aggregates: list[dict[str, object]]) -> dict[str, object]:
    """Combine per-site aggregate messages into the shared scaler + centroids.

    Returns the global mean/sd over the retained (non-constant) feature columns
    and the standardized class centroids, in the same column order and class
    order (sorted by label) that a centralized ``standardize`` +
    ``class_centroids`` would produce.
    """
    if not aggregates:
        raise ValueError("need at least one site aggregate")
    columns = aggregates[0]["columns"]
    for agg in aggregates:
        if agg["columns"] != columns:
            raise ValueError("sites disagree on the feature contract (column set/order)")

    d = len(columns)
    total_n = sum(int(agg["n"]) for agg in aggregates)
    total_sum = np.sum([np.asarray(agg["sum"], dtype=float) for agg in aggregates], axis=0)
    total_sumsq = np.sum([np.asarray(agg["sumsq"], dtype=float) for agg in aggregates], axis=0)

    mean = total_sum / total_n
    var = np.clip(total_sumsq / total_n - mean**2, 0.0, None)
    sd = np.sqrt(var)

    kept_idx = [i for i in range(d) if var[i] > VARIANCE_TOL]
    kept_columns = [columns[i] for i in kept_idx]
    mean_kept = mean[kept_idx]
    sd_kept = sd[kept_idx]

    class_totals_n: dict[str, int] = {}
    class_totals_sum: dict[str, np.ndarray] = {}
    for agg in aggregates:
        for cls, count in agg["class_n"].items():
            class_totals_n[cls] = class_totals_n.get(cls, 0) + int(count)
            contribution = np.asarray(agg["class_sum"][cls], dtype=float)
            if cls in class_totals_sum:
                class_totals_sum[cls] = class_totals_sum[cls] + contribution
            else:
                class_totals_sum[cls] = contribution

    classes = sorted(class_totals_n, key=str)
    centroids = np.vstack(
        [
            (class_totals_sum[cls][kept_idx] / class_totals_n[cls] - mean_kept) / sd_kept
            for cls in classes
        ]
    ) if classes else np.empty((0, len(kept_columns)))

    return {
        "n_total": total_n,
        "classes": classes,
        "class_counts": {cls: class_totals_n[cls] for cls in classes},
        "kept_columns": kept_columns,
        "mean": mean_kept,
        "sd": sd_kept,
        "centroids": centroids,
    }


def centralized_fit(X: pd.DataFrame, y: pd.Series) -> dict[str, object]:
    """The reference: scaler + standardized centroids from pooled data.

    Uses the exact production primitives (``epinet_cluster.standardize`` and
    ``class_centroids``), so a match proves the federated path reconstructs what
    EpiNet would compute centrally.
    """
    Xz, kept_columns = epinet_cluster.standardize(X)
    y_labeled = y.reindex(X.index).mask(epinet_common.blank_label_mask(y.reindex(X.index)))
    classes, centroids = epinet_cluster.class_centroids(Xz, y_labeled.reset_index(drop=True))
    return {"kept_columns": kept_columns, "classes": classes, "centroids": centroids}


def simulate(
    X: pd.DataFrame,
    y: pd.Series,
    site_labels: list[object] | pd.Series | np.ndarray,
    *,
    min_cell: int = 0,
) -> dict[str, object]:
    """Partition rows across sites, federate the fit, and compare to centralized.

    ``site_labels`` assigns each row of ``X`` to a site. Only per-site aggregate
    messages cross; the function returns both fits and the maximum absolute
    difference in mean, sd, and centroids — which should be at floating-point
    level if the aggregation composes.
    """
    site_labels = pd.Series(np.asarray(site_labels), index=X.index)
    aggregates = []
    site_summary = {}
    for site in sorted(site_labels.unique(), key=str):
        rows = site_labels == site
        agg = site_aggregates(X.loc[rows], y.loc[rows], min_cell=min_cell)
        aggregates.append(agg)
        site_summary[str(site)] = {"n": agg["n"], "classes": agg["class_n"]}

    fed = combine_aggregates(aggregates)
    central = centralized_fit(X, y)

    # Align on the shared kept columns + classes, then compare element-wise.
    if fed["kept_columns"] != central["kept_columns"]:
        raise AssertionError(
            f"kept-column mismatch: federated {fed['kept_columns']} vs central {central['kept_columns']}"
        )
    central_centroids = {cls: central["centroids"][i] for i, cls in enumerate(central["classes"])}
    fed_centroids = {cls: fed["centroids"][i] for i, cls in enumerate(fed["classes"])}
    shared_classes = [c for c in fed["classes"] if c in central_centroids]
    max_centroid_diff = max(
        (float(np.max(np.abs(fed_centroids[c] - central_centroids[c]))) for c in shared_classes),
        default=0.0,
    )

    # Centralized mean/sd over the kept columns, recomputed for the comparison.
    Xz_cols = X[central["kept_columns"]].to_numpy(dtype=float)
    central_mean = Xz_cols.mean(axis=0)
    central_sd = Xz_cols.std(axis=0, ddof=0)

    return {
        "sites": site_summary,
        "n_total": fed["n_total"],
        "classes": fed["classes"],
        "kept_columns": fed["kept_columns"],
        "max_mean_diff": float(np.max(np.abs(fed["mean"] - central_mean))),
        "max_sd_diff": float(np.max(np.abs(fed["sd"] - central_sd))),
        "max_centroid_diff": max_centroid_diff,
        "federated": fed,
        "centralized": central,
    }
