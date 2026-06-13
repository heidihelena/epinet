"""Contestability and flip-distance for the nearest-centroid classifier.

This is the *analytic* complement to ``epinet_cluster``'s centroid view. Where
the cluster lens reports **which** class centroid a node is nearest to, this lens
reports **how contestable** that assignment is:

1. ``flip_distance`` — the smallest move in standardized feature space that would
   flip the node to a different class,
2. the binding ``runner_up`` class — the class it would flip *to*, and
3. a per-feature **value-of-information** ranking — which single input the
   decision is most sensitive to, i.e. the measurement you would most cheaply
   need to refine to move (or settle) the call.

Why this has a closed form. For a nearest-centroid (Rocchio) classifier the
decision boundary between two classes is the perpendicular bisector hyperplane of
their centroids, so the distance from a point to the boundary is exact rather
than estimated. With nearest centroid ``a`` and competing centroid ``k`` the
distance to their shared boundary is

    (d_k**2 - d_a**2) / (2 * ||c_k - c_a||)

where ``d_a``/``d_k`` are the point's distances to the two centroids. The
flip-distance is the minimum of this over all competing classes. This is the
toolkit's documented ``|s(x) - tau| / ||grad s(x)||`` made *exact* for this
surface, and it holds in both the Euclidean and shared-covariance Mahalanobis
metrics (in the latter, distances and centroid separations are simply measured in
the whitened metric, where the boundary is again a hyperplane).

Honest reading — surfaced in the output, not buried:

- ``flip_distance`` is in **standardized-feature units** (standard deviations).
  It is contestable in a way that *matters* only when it is smaller than the
  real-world measurement error of the inputs expressed in the same units
  ("this call reverses if the nodule were measured 0.4 SD differently"). The
  module reports the number; comparing it to measurement error is a domain step.
- It measures the geometry of the **centroid surface, not ground truth**. A small
  flip-distance says the model's call is fragile here, not that the underlying
  case is genuinely borderline. On EpiNet's deliberately unvalidated ported
  scores, flip-distance measures the fragility of the score, full stop.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from epinet import cluster as ec
from epinet import common as epinet_common

CAVEATS = (
    "flip_distance is in standardized-feature units (SDs); it is decision-relevant "
    "only when smaller than the inputs' real-world measurement error in the same units.",
    "flip_distance measures the centroid surface, not ground truth: a small value "
    "means the model's call is fragile here, not that the case is truly borderline.",
)


def flip_distances(
    Xz: np.ndarray,
    centroids: np.ndarray,
    *,
    metric: str = "euclidean",
    inv_cov: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Exact flip-distance and per-feature leverage against centroid boundaries.

    Operates on an already-standardized matrix ``Xz`` (n_samples x n_features)
    and ``centroids`` (n_classes x n_features). Returns per-sample arrays:

    - ``nearest`` / ``runner_up`` — index of the assigned class and of the class
      whose boundary is nearest (the one a flip would cross),
    - ``flip_distance`` — distance to that nearest boundary, in metric units,
    - ``most_relevant_feature`` — the feature the decision is most sensitive to,
      ranked by ``|grad_j| / sqrt(M_jj)`` so the ranking is correct in the metric
      (the cheapest single-axis move to the binding boundary), not just Euclidean,
    - ``single_axis_flip_distance`` — the metric length of the move along that one
      feature needed to flip the call, measured against the binding (nearest,
      runner_up) boundary. If a *third* class is nearer along that axis the true
      flip can be shorter; this is an upper bound for the binding pair,
    - ``leverage`` — ``|margin gradient| / sqrt(M_jj)`` per feature (the
      metric-correct value-of-information weights), and ``distances`` — the full
      sample-to-centroid distance matrix.
    """
    centroids = np.atleast_2d(centroids)
    n_samples, n_features = Xz.shape
    n_classes = centroids.shape[0]
    if n_classes < 2:
        raise ValueError("flip-distance needs at least two class centroids")
    if metric == "mahalanobis" and inv_cov is None:
        inv_cov = ec._mahalanobis_inverse_cov(Xz)

    distances = ec.distances_to_points(Xz, centroids, metric=metric, inv_cov=inv_cov)
    separation = ec.distances_to_points(centroids, centroids, metric=metric, inv_cov=inv_cov)

    rows = np.arange(n_samples)
    nearest = distances.argmin(axis=1)
    d_a = distances[rows, nearest]

    # Distance from each sample to the boundary with every class, via the
    # bisector formula. The sample's own class (separation 0) and any degenerate
    # identical centroids are pushed to +inf so they never bind.
    sep_from_nearest = separation[nearest]  # (n_samples, n_classes)
    with np.errstate(divide="ignore", invalid="ignore"):
        to_boundary = (distances**2 - d_a[:, None] ** 2) / (2.0 * sep_from_nearest)
    to_boundary[rows, nearest] = np.inf
    to_boundary[~np.isfinite(to_boundary)] = np.inf

    runner_up = to_boundary.argmin(axis=1)
    flip_distance = np.clip(to_boundary[rows, runner_up], 0.0, None)

    # Per-feature value-of-information for the binding (nearest, runner_up) pair.
    # The margin m(x) = d_runner**2 - d_nearest**2 has gradient 2 * M (c_a - c_b).
    # A single-axis move of raw size t along feature j has *metric* length
    # |t|*sqrt(M_jj), and reaches the boundary at |t| = m / (2|grad_j|). So the
    # flip cost in the same units as flip_distance is (m/2) * sqrt(M_jj)/|grad_j|,
    # and the cheapest axis to flip maximizes |grad_j| / sqrt(M_jj). For the
    # Euclidean metric (M = I, sqrt(M_jj) = 1) this reduces to |grad_j|.
    metric_matrix = inv_cov if metric == "mahalanobis" else np.eye(n_features)
    metric_diag = np.sqrt(np.clip(np.diag(metric_matrix), 1e-12, None))  # sqrt(M_jj)
    centroid_diff = centroids[nearest] - centroids[runner_up]  # c_a - c_b
    grad = centroid_diff @ metric_matrix
    abs_grad = np.abs(grad)
    leverage = abs_grad / metric_diag  # value of information per feature (metric-correct)
    most_relevant_feature = leverage.argmax(axis=1)
    margin = distances[rows, runner_up] ** 2 - d_a**2  # >= 0
    top_grad = abs_grad[rows, most_relevant_feature]
    top_diag = metric_diag[most_relevant_feature]
    with np.errstate(divide="ignore", invalid="ignore"):
        # raw step m/(2|grad_j|) scaled to metric length by sqrt(M_jj).
        single_axis = margin / (2.0 * top_grad) * top_diag
    single_axis = np.where(np.isfinite(single_axis), single_axis, np.inf)

    return {
        "nearest": nearest,
        "runner_up": runner_up,
        "flip_distance": flip_distance,
        "most_relevant_feature": most_relevant_feature,
        "single_axis_flip_distance": single_axis,
        "leverage": leverage,
        "distances": distances,
    }


def contestability(
    X: pd.DataFrame,
    *,
    y: pd.Series,
    metric: str = "euclidean",
    contest_quantile: float = 0.1,
) -> dict[str, object]:
    """Score every node's contestability against the outcome-class centroids.

    Class centroids are built from the labeled nodes; every node in ``X``
    (scaffold included) is then scored, since "where would this unlabeled node
    fall, and how contestably" is a meaningful question. ``contest_quantile``
    flags the most fragile fraction of nodes (default: the lowest-decile
    flip-distance) — a dataset-relative cut, because an absolute, measurement-
    error-based threshold needs domain input the toolkit does not have.

    Returns ``assignments`` (DataFrame) and a JSON-serializable ``summary``.
    """
    if not 0.0 < contest_quantile < 1.0:
        raise ValueError("contest_quantile must be in (0, 1)")

    # Missing features silently poison centroids (NaN mean) and produce all-inf
    # flip-distances; refuse rather than return a confidently-wrong "nothing is
    # contestable". The caller must impute or drop deliberately.
    numeric = X.select_dtypes(include=[np.number])
    nan_cols = [c for c in numeric.columns if numeric[c].isna().any()]
    if nan_cols:
        raise ValueError(
            "Contestability requires non-missing numeric features; NaN found in "
            f"columns {nan_cols[:10]}. Impute or drop these before scoring."
        )

    Xz, kept_columns = ec.standardize(X)
    if Xz.shape[0] < 2 or not kept_columns:
        raise ValueError("Contestability needs at least two nodes and one varying feature")

    labeled = y.reindex(X.index)
    labeled = labeled.mask(epinet_common.blank_label_mask(labeled))
    if labeled.dropna().nunique() < 2:
        raise ValueError("Contestability needs at least two outcome classes among labeled nodes")

    classes, centroids = ec.class_centroids(Xz, labeled)
    inv_cov = ec._mahalanobis_inverse_cov(Xz) if metric == "mahalanobis" else None
    result = flip_distances(Xz, centroids, metric=metric, inv_cov=inv_cov)

    flip = result["flip_distance"]
    finite = flip[np.isfinite(flip)]
    threshold = float(np.quantile(finite, contest_quantile)) if finite.size else float("inf")
    contested = flip <= threshold

    assignments = pd.DataFrame(index=X.index)
    assignments.index.name = "ID"
    assignments["nearest_class_centroid"] = [classes[i] for i in result["nearest"]]
    assignments["runner_up_class"] = [classes[i] for i in result["runner_up"]]
    assignments["flip_distance"] = flip
    assignments["contested"] = contested
    assignments["most_decision_relevant_feature"] = [
        kept_columns[i] for i in result["most_relevant_feature"]
    ]
    assignments["single_axis_flip_distance"] = result["single_axis_flip_distance"]
    for j, cls in enumerate(classes):
        assignments[f"dist_to_{cls}"] = result["distances"][:, j]
    assignments["outcome"] = labeled.to_numpy()
    # Agreement is only defined for labeled nodes; scaffold rows stay None so the
    # comparison never touches a missing label (pd.NA has no truth value).
    label_present = labeled.notna().to_numpy()
    nearest_names = assignments["nearest_class_centroid"].to_numpy()
    outcome_names = labeled.astype("string").to_numpy()
    matches = np.full(len(assignments), None, dtype=object)
    matches[label_present] = nearest_names[label_present] == outcome_names[label_present]
    assignments["nearest_matches_outcome"] = matches

    # Global value-of-information: mean per-feature share of the flip gradient.
    leverage = result["leverage"]
    row_totals = leverage.sum(axis=1, keepdims=True)
    shares = np.divide(leverage, row_totals, out=np.zeros_like(leverage), where=row_totals > 0)
    mean_share = shares.mean(axis=0)
    feature_leverage = dict(
        sorted(
            ((kept_columns[j], float(mean_share[j])) for j in range(len(kept_columns))),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )

    runner_counts = (
        assignments["runner_up_class"].value_counts().to_dict()
    )
    summary: dict[str, object] = {
        "distance_metric": metric,
        "n_scored": int(len(assignments)),
        "n_classes": len(classes),
        "classes": classes,
        "n_features": len(kept_columns),
        "feature_columns": kept_columns,
        "flip_distance": {
            "mean": float(np.mean(finite)) if finite.size else None,
            "std": float(np.std(finite)) if finite.size else None,
            "min": float(np.min(finite)) if finite.size else None,
            "median": float(np.median(finite)) if finite.size else None,
            "max": float(np.max(finite)) if finite.size else None,
            "contest_quantile": contest_quantile,
            "contest_threshold": threshold,
            "n_contested": int(contested.sum()),
        },
        "runner_up_counts": {str(k): int(v) for k, v in runner_counts.items()},
        "feature_leverage": feature_leverage,
        "caveats": list(CAVEATS),
    }
    return {"assignments": assignments.reset_index(), "summary": summary}


def run_contestability(
    X: pd.DataFrame,
    output_dir: Path,
    *,
    y: pd.Series,
    metric: str = "euclidean",
    contest_quantile: float = 0.1,
) -> dict[str, object]:
    """Score contestability and write the CSV, JSON, and markdown report."""
    result = contestability(X, y=y, metric=metric, contest_quantile=contest_quantile)
    output_dir.mkdir(parents=True, exist_ok=True)
    result["assignments"].to_csv(output_dir / "node_contestability.csv", index=False)
    (output_dir / "contest_summary.json").write_text(
        json.dumps(result["summary"], indent=2) + "\n"
    )
    (output_dir / "contestability_report.md").write_text(
        contestability_report(result["assignments"], result["summary"]) + "\n"
    )
    return result


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    """A minimal GitHub-flavored markdown table (no third-party dependency)."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def contestability_report(
    assignments: pd.DataFrame,
    summary: dict,
    *,
    top_n: int = 10,
) -> str:
    """Render a human-readable markdown report: most-contested cases + the
    value-of-information ranking + the caveats. Companion to the CSV/JSON."""
    flip = summary["flip_distance"]
    threshold = flip["contest_threshold"]
    threshold_text = f"{threshold:.3g}" if np.isfinite(threshold) else "∞"

    out = [
        "# Contestability report",
        "",
        f"Metric: **{summary['distance_metric']}** · scored {summary['n_scored']} "
        f"cases · {flip['n_contested']} contested "
        f"(flip-distance ≤ {threshold_text}, lowest {flip['contest_quantile']:.0%}).",
        "",
        "> **Read this first.** Flip-distance is in *standardized-feature* units: it is "
        "decision-relevant only when smaller than the real-world measurement error of "
        "the features in the same units (comparing it to assay/measurement variability "
        "is your step). It measures the **classifier's fragility, not ground truth** — a "
        "small value says the model is unsure here, not that the case is truly borderline.",
        "",
        "## Most contested cases",
        "",
        "The calls nearest to flipping, with the single feature that would most "
        "cheaply settle each (value of information).",
        "",
    ]
    contested = assignments.sort_values("flip_distance").head(top_n)
    case_rows = []
    for _, row in contested.iterrows():
        value = row["flip_distance"]
        case_rows.append([
            row["ID"],
            row["nearest_class_centroid"],
            row["runner_up_class"],
            f"{value:.3g}" if np.isfinite(value) else "∞",
            row["most_decision_relevant_feature"],
        ])
    out.append(_markdown_table(
        ["case", "call", "would flip to", "flip-distance", "decisive feature"],
        case_rows,
    ))

    out += [
        "",
        "## Value of information",
        "",
        "Features that most drive boundary flips across the cohort "
        "(mean flip-gradient share).",
        "",
    ]
    voi_rows = [
        [name, f"{share:.3f}"]
        for name, share in list(summary.get("feature_leverage", {}).items())[:top_n]
    ]
    out.append(_markdown_table(["feature", "leverage"], voi_rows))

    out += ["", "## Caveats", ""]
    out += [f"- {caveat}" for caveat in summary.get("caveats", [])]
    return "\n".join(out)
