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

from dataclasses import dataclass

import numpy as np
import pandas as pd

from epinet import cluster as epinet_cluster
from epinet import common as epinet_common
from epinet import contest as epinet_contest
from epinet import governance as governance

VARIANCE_TOL = 1e-12  # matches epinet_cluster.standardize's zero-variance cutoff

# Shared histogram bins for the federated flip-distance quantile (in SD units).
# Part of the contract: every site bins against the same edges so the coordinator
# can sum the histograms. The last bin absorbs the (rare) tail beyond the range.
DEFAULT_FLIP_BINS = np.linspace(0.0, 8.0, 33)


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

    # Feature-contract guard: a single non-finite value silently poisons the
    # additive statistics — one NaN/inf propagates through sum/mean/m2/comoment
    # and corrupts the whole federated fit (and every other site's view of it).
    # Reject it at the site with a clear, column-named error rather than shipping
    # a broken aggregate that fails far away in the coordinator.
    if n and not np.isfinite(values).all():
        bad = [columns[j] for j in range(values.shape[1]) if not np.isfinite(values[:, j]).all()]
        raise ValueError(f"feature columns contain non-finite values (NaN/inf): {bad}")

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

    # Centered (mean-subtracted) moments rather than raw sum-of-squares. The site
    # has its own records, so it can subtract its own mean before squaring, which
    # avoids the catastrophic cancellation of ``sumsq/n - mean**2`` for features
    # whose mean dwarfs their spread. ``m2`` (per-feature sum of squared
    # deviations) and ``comoment`` (centered co-moment matrix) combine ACROSS
    # sites exactly via the parallel/Chan update in ``combine_aggregates`` — still
    # additive, still aggregate-only, no record leaves. ``sum`` is kept because
    # the mean (and the class centroids) are well-conditioned as plain sums.
    if n:
        mean = values.mean(axis=0)
        centered = values - mean
        m2 = (centered**2).sum(axis=0)
        comoment = centered.T @ centered
    else:
        d = len(columns)
        mean = np.zeros(d)
        m2 = np.zeros(d)
        comoment = np.zeros((d, d))

    return {
        "columns": columns,
        "n": n,
        "sum": values.sum(axis=0).tolist(),
        "mean": mean.tolist(),
        # Per-feature sum of squared deviations from this site's mean.
        "m2": m2.tolist(),
        # Centered co-moment (sum of centered outer products). With m2 + n + mean
        # it reconstructs the pooled covariance for the Mahalanobis metric.
        "comoment": comoment.tolist(),
        "class_n": class_n,
        "class_sum": class_sum,
        "suppressed": suppressed,
    }


def combine_aggregates(
    aggregates: list[dict[str, object]],
    *,
    shrinkage: float = 0.0,
) -> dict[str, object]:
    """Combine per-site aggregate messages into the shared scaler + centroids.

    Returns the global mean/sd over the retained (non-constant) feature columns
    and the standardized class centroids, in the same column order and class
    order (sorted by label) that a centralized ``standardize`` +
    ``class_centroids`` would produce.

    ``shrinkage`` (0–1) optionally regularizes the standardized covariance used
    for the Mahalanobis precision, shrinking it toward the identity:
    ``(1 - shrinkage) * cov + shrinkage * I``. Because the features are
    standardized the covariance has unit diagonal, so the identity is its natural
    Ledoit-Wolf target — and the shrink is computed from the already-shipped
    centered co-moment, needing no extra aggregate and no record access. It
    conditions ``inv_cov`` when features are collinear or a site is small. The
    default ``0.0`` reproduces the exact empirical reconstruction.
    """
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError(f"shrinkage must be in [0, 1], got {shrinkage}")
    if not aggregates:
        raise ValueError("need at least one site aggregate")
    aggregates = _unwrap(aggregates)
    columns = aggregates[0]["columns"]
    for agg in aggregates:
        if agg["columns"] != columns:
            raise ValueError("sites disagree on the feature contract (column set/order)")

    d = len(columns)
    total_n = sum(int(agg["n"]) for agg in aggregates)
    if total_n == 0:
        raise ValueError("all site aggregates are empty (total n = 0); nothing to combine")
    total_sum = np.sum([np.asarray(agg["sum"], dtype=float) for agg in aggregates], axis=0)
    mean = total_sum / total_n

    # Numerically stable pooled moments via the parallel (Chan et al.) update.
    # Folding centered per-site moments never forms ``sumsq/n - mean**2``, so the
    # reconstruction stays exact even when a feature's mean dwarfs its variance.
    # Each merge of accumulator A with site B uses delta = mean_B - mean_A:
    #   M2  += M2_B + delta**2          * (n_A n_B / n)
    #   C   += C_B  + outer(delta,delta)* (n_A n_B / n)
    run_n = 0
    run_mean = np.zeros(d)
    run_m2 = np.zeros(d)
    run_c = np.zeros((d, d))
    for agg in aggregates:
        n_b = int(agg["n"])
        if n_b == 0:
            continue
        mean_b = np.asarray(agg["mean"], dtype=float)
        m2_b = np.asarray(agg["m2"], dtype=float)
        c_b = np.asarray(agg["comoment"], dtype=float)
        if run_n == 0:
            run_n, run_mean, run_m2, run_c = n_b, mean_b, m2_b, c_b
            continue
        delta = mean_b - run_mean
        combined_n = run_n + n_b
        scale = run_n * n_b / combined_n
        run_m2 = run_m2 + m2_b + delta**2 * scale
        run_c = run_c + c_b + np.outer(delta, delta) * scale
        run_mean = run_mean + delta * (n_b / combined_n)
        run_n = combined_n

    var = np.clip(run_m2 / total_n, 0.0, None)
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

    # Pooled covariance of the standardized features, for the Mahalanobis metric.
    # cov(X) = C/n from the folded centered co-moment (population); cov(Xz)
    # divides by the sd outer product. This is the EMPIRICAL covariance — exactly
    # federatable and, via the centered co-moment, free of the cancellation that
    # ``second_moment/n - mean.mean^T`` suffered. ``shrinkage`` below applies a
    # specified-intensity shrink toward the identity; matching production
    # Ledoit-Wolf's DATA-DRIVEN intensity exactly would additionally need
    # 4th-moment aggregates (noted, not built).
    cov_full = run_c / total_n
    cov_kept = cov_full[np.ix_(kept_idx, kept_idx)]
    cov_z = cov_kept / np.outer(sd_kept, sd_kept)
    # Optional shrinkage toward the identity (the standardized covariance has unit
    # diagonal, so I is its natural target). lambda=0 leaves cov_z untouched.
    eye = np.eye(len(kept_idx))
    cov_reg = (1.0 - shrinkage) * cov_z + shrinkage * eye
    ridge = 1e-6 * eye
    inv_cov = np.linalg.pinv(cov_reg + ridge)

    return {
        "n_total": total_n,
        "classes": classes,
        "class_counts": {cls: class_totals_n[cls] for cls in classes},
        "kept_columns": kept_columns,
        "mean": mean_kept,
        "sd": sd_kept,
        "centroids": centroids,
        "cov_standardized": cov_z,
        "inv_cov": inv_cov,
        "shrinkage": float(shrinkage),
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
    shrinkage: float = 0.0,
) -> dict[str, object]:
    """Partition rows across sites, federate the fit, and compare to centralized.

    ``site_labels`` assigns each row of ``X`` to a site. Only per-site aggregate
    messages cross; the function returns both fits and the maximum absolute
    difference in mean, sd, and centroids — which should be at floating-point
    level if the aggregation composes. ``shrinkage`` is passed to
    ``combine_aggregates`` (it conditions ``inv_cov`` only; mean/sd/centroids,
    and thus the comparison below, are unaffected).
    """
    site_labels = pd.Series(np.asarray(site_labels), index=X.index)
    aggregates = []
    site_summary = {}
    for site in sorted(site_labels.unique(), key=str):
        rows = site_labels == site
        agg = site_aggregates(X.loc[rows], y.loc[rows], min_cell=min_cell)
        aggregates.append(agg)
        site_summary[str(site)] = {"n": agg["n"], "classes": agg["class_n"]}

    fed = combine_aggregates(aggregates, shrinkage=shrinkage)
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


# --- Federated contestability ---------------------------------------------
# Stage 2 of the derived federated dataset. Given the pooled fit (scaler +
# centroids) from combine_aggregates, each site computes flip-distance LOCALLY
# against the global centroids — and because both the standardized node vectors
# and the centroids were reconstructed exactly, those per-node scores are
# identical to a centralized run. Only de-identified summaries cross.


def local_flip_distances(
    X: pd.DataFrame,
    fit: dict[str, object],
    *,
    metric: str = "euclidean",
) -> np.ndarray:
    """Per-node flip-distance computed at a site against the GLOBAL fit.

    Standardizes the site's rows with the pooled mean/sd and scores them against
    the pooled standardized centroids. Stays local — this is what proves the
    scores federate exactly; in deployment only the summary below leaves.
    """
    mean = np.asarray(fit["mean"], dtype=float)
    sd = np.asarray(fit["sd"], dtype=float)
    centroids = np.asarray(fit["centroids"], dtype=float)
    inv_cov = np.asarray(fit["inv_cov"], dtype=float) if metric == "mahalanobis" else None
    Xz = (X[fit["kept_columns"]].to_numpy(dtype=float) - mean) / sd
    return epinet_contest.flip_distances(Xz, centroids, metric=metric, inv_cov=inv_cov)["flip_distance"]


def site_contestability(
    X: pd.DataFrame,
    y: pd.Series,
    fit: dict[str, object],
    *,
    metric: str = "euclidean",
    bin_edges: np.ndarray = DEFAULT_FLIP_BINS,
) -> dict[str, object]:
    """One site's de-identified contestability summary against the global fit.

    Carries additive statistics only: flip-distance count/sum/sumsq/min/max and a
    shared-bin histogram, runner-up class counts, a per-feature value-of-
    information sum, and nearest-centroid agreement counts. No per-node row.
    """
    mean = np.asarray(fit["mean"], dtype=float)
    sd = np.asarray(fit["sd"], dtype=float)
    centroids = np.asarray(fit["centroids"], dtype=float)
    inv_cov = np.asarray(fit["inv_cov"], dtype=float) if metric == "mahalanobis" else None
    classes = list(fit["classes"])
    kept_columns = list(fit["kept_columns"])

    Xz = (X[kept_columns].to_numpy(dtype=float) - mean) / sd
    res = epinet_contest.flip_distances(Xz, centroids, metric=metric, inv_cov=inv_cov)
    flip = res["flip_distance"]
    finite = np.isfinite(flip)
    flip_f = flip[finite]

    # Value-of-information: per-node feature share of the flip gradient, summed.
    leverage = res["leverage"]
    row_totals = leverage.sum(axis=1, keepdims=True)
    shares = np.divide(leverage, row_totals, out=np.zeros_like(leverage), where=row_totals > 0)

    runner_up_counts: dict[str, int] = {}
    for idx in res["runner_up"][finite]:
        cls = classes[int(idx)]
        runner_up_counts[cls] = runner_up_counts.get(cls, 0) + 1

    # Nearest-centroid agreement on labeled nodes only.
    labeled = epinet_common.labeled_mask(y.reindex(X.index)).to_numpy()
    nearest_names = np.array([classes[int(i)] for i in res["nearest"]])
    y_str = y.reindex(X.index).astype("string").to_numpy()
    agree = int(np.sum(labeled & (nearest_names == y_str)))

    counts, _ = np.histogram(np.clip(flip_f, bin_edges[0], bin_edges[-1]), bins=bin_edges)

    return {
        "columns": kept_columns,
        "n": int(len(flip)),
        "flip_count": int(finite.sum()),
        "flip_sum": float(flip_f.sum()),
        "flip_sumsq": float((flip_f**2).sum()),
        "flip_min": float(flip_f.min()) if flip_f.size else None,
        "flip_max": float(flip_f.max()) if flip_f.size else None,
        "flip_hist": counts.tolist(),
        "runner_up_counts": runner_up_counts,
        "leverage_sum": shares.sum(axis=0).tolist(),
        "leverage_n": int(shares.shape[0]),
        "agree_count": agree,
        "labeled_count": int(labeled.sum()),
    }


def combine_contestability(
    summaries: list[dict[str, object]],
    *,
    contest_quantile: float = 0.1,
    bin_edges: np.ndarray = DEFAULT_FLIP_BINS,
) -> dict[str, object]:
    """Pool per-site contestability summaries into a federated summary.

    Flip-distance mean/std, runner-up counts, value-of-information, and agreement
    are EXACT additive aggregates. min/max are exact too WHEN present, but the
    egress gate withholds them as extreme single-record values (then this returns
    ``None`` for them). The contested threshold is the one approximate piece: it
    reads a quantile off the summed shared-bin histogram (bin-resolution
    accuracy), because a quantile is an order statistic, not a sum.
    """
    if not summaries:
        raise ValueError("need at least one site summary")
    summaries = _unwrap(summaries)
    columns = summaries[0]["columns"]
    for summary in summaries:
        if summary["columns"] != columns:
            raise ValueError("sites disagree on the feature contract")

    count = sum(int(s["flip_count"]) for s in summaries)
    flip_sum = sum(float(s["flip_sum"]) for s in summaries)
    flip_sumsq = sum(float(s["flip_sumsq"]) for s in summaries)
    mean = flip_sum / count if count else None
    std = float(np.sqrt(max(flip_sumsq / count - mean**2, 0.0))) if count else None
    mins = [s["flip_min"] for s in summaries if s.get("flip_min") is not None]
    maxs = [s["flip_max"] for s in summaries if s.get("flip_max") is not None]

    hist = np.sum([np.asarray(s["flip_hist"], dtype=float) for s in summaries], axis=0)
    # Approximate quantile threshold from the summed histogram, interpolating
    # within the crossing bin (assume-uniform). The contested COUNT is q*N by
    # definition (it is a quantile cut); the threshold VALUE is the approximate
    # quantity, and its error is bounded by the bin width.
    cum = np.cumsum(hist)
    total = float(cum[-1]) if cum.size else 0.0
    threshold = None
    n_contested = 0
    if total > 0:
        target = contest_quantile * total
        k = min(int(np.searchsorted(cum, target, side="left")), len(hist) - 1)
        cum_before = float(cum[k - 1]) if k > 0 else 0.0
        bin_count = float(hist[k])
        frac = min(max((target - cum_before) / bin_count, 0.0), 1.0) if bin_count > 0 else 0.0
        threshold = float(bin_edges[k] + frac * (bin_edges[k + 1] - bin_edges[k]))
        n_contested = int(round(target))

    runner_up_counts: dict[str, int] = {}
    for summary in summaries:
        for cls, n in summary["runner_up_counts"].items():
            runner_up_counts[cls] = runner_up_counts.get(cls, 0) + int(n)

    leverage_sum = np.sum([np.asarray(s["leverage_sum"], dtype=float) for s in summaries], axis=0)
    leverage_n = sum(int(s["leverage_n"]) for s in summaries)
    mean_share = leverage_sum / leverage_n if leverage_n else leverage_sum
    feature_voi = dict(
        sorted(zip(columns, mean_share.tolist()), key=lambda kv: kv[1], reverse=True)
    )

    agree = sum(int(s["agree_count"]) for s in summaries)
    labeled = sum(int(s["labeled_count"]) for s in summaries)

    return {
        "n_scored": sum(int(s["n"]) for s in summaries),
        "flip_distance": {
            "mean": mean,
            "std": std,
            "min": min(mins) if mins else None,
            "max": max(maxs) if maxs else None,
            "contest_quantile": contest_quantile,
            "approx_contest_threshold": threshold,
            "approx_n_contested": n_contested,
        },
        "runner_up_counts": runner_up_counts,
        "feature_voi": feature_voi,
        "nearest_centroid_agreement": (agree / labeled) if labeled else None,
    }


def simulate_contestability(
    X: pd.DataFrame,
    y: pd.Series,
    site_labels: list[object] | pd.Series | np.ndarray,
    *,
    metric: str = "euclidean",
    contest_quantile: float = 0.1,
    min_cell: int = 0,
    bin_edges: np.ndarray = DEFAULT_FLIP_BINS,
) -> dict[str, object]:
    """Federate the contestability summary and compare to centralized — the
    stage-2 analogue of ``simulate``.

    Builds the pooled fit from per-site aggregates, has each site summarize its
    flip-distances against that shared fit (only aggregate messages cross), pools
    the summaries, and checks the federated flip-distance mean/std, runner-up
    counts, and top value-of-information feature against ``contest.contestability``
    on the pooled data. The mean/std diffs should be at floating-point level if
    the contestability layer composes; the contested-threshold value is the one
    histogram-approximate piece and is not asserted here.
    """
    site_labels = pd.Series(np.asarray(site_labels), index=X.index)
    ordered_sites = sorted(site_labels.unique(), key=str)

    aggregates = [
        site_aggregates(X.loc[site_labels == s], y.loc[site_labels == s], min_cell=min_cell)
        for s in ordered_sites
    ]
    fit = combine_aggregates(aggregates)

    summaries = []
    site_summary = {}
    for s in ordered_sites:
        rows = site_labels == s
        summary = site_contestability(
            X.loc[rows], y.loc[rows], fit, metric=metric, bin_edges=bin_edges
        )
        summaries.append(summary)
        site_summary[str(s)] = {"n": summary["n"], "flip_count": summary["flip_count"]}

    fed = combine_contestability(summaries, contest_quantile=contest_quantile, bin_edges=bin_edges)
    central = epinet_contest.contestability(
        X, y=y, metric=metric, contest_quantile=contest_quantile
    )["summary"]

    fed_fd, cen_fd = fed["flip_distance"], central["flip_distance"]
    def _abs_diff(a, b):
        return float(abs(a - b)) if a is not None and b is not None else None

    return {
        "sites": site_summary,
        "metric": metric,
        "federated": fed,
        "centralized": central,
        "max_mean_diff": _abs_diff(fed_fd["mean"], cen_fd["mean"]),
        "max_std_diff": _abs_diff(fed_fd["std"], cen_fd["std"]),
        "runner_up_match": fed["runner_up_counts"]
        == {str(k): int(v) for k, v in central["runner_up_counts"].items()},
        "top_voi_match": next(iter(fed["feature_voi"]), None)
        == next(iter(central["feature_leverage"]), None),
    }


# --- Mandatory egress gate -------------------------------------------------
# The functions above compute aggregates *within* a site's trust boundary (and
# are reused for the in-boundary two-site simulation and local scoring). The
# moment an aggregate is meant to LEAVE a site, it must go through the governance
# gate. We make that structural: the cross-boundary producers below return a
# sealed SiteContribution that cannot be serialized or shipped — the only way to
# obtain a shippable payload is ``.disclose(policy, consent)``, which runs
# ``epinet_governance.check_egress`` (small-cell suppression, tier ceiling,
# consent, disclosure manifest, audit). This closes the "forgot to call the
# gate" gap for accidental egress.


@dataclass
class DisclosedContribution:
    """A site aggregate that has passed the egress gate — safe to ship.

    Carries the redacted ``payload`` (what crosses) and the ``manifest`` (the
    disclosure record). This is what a coordinator receives and feeds to the
    ``combine_*`` functions.
    """

    payload: dict
    manifest: dict


class SiteContribution:
    """A sealed, not-yet-disclosed site aggregate.

    Holds the raw aggregate privately. It is deliberately not JSON-serializable
    and its repr hides the data: the only sanctioned way to get a shippable
    payload is ``disclose()``, which runs the governance gate. This makes the
    gate mandatory on the egress path rather than an optional extra step.
    """

    def __init__(self, aggregate: dict, kind: str):
        self._aggregate = aggregate
        self.kind = kind

    def disclose(
        self,
        *,
        policy: governance.DisclosurePolicy,
        consent: governance.Consent,
        tier: str = "aggregate",
        audit: governance.AuditLedger | None = None,
        now=None,
        timestamp: str | None = None,
    ) -> DisclosedContribution:
        """Run the egress gate and return a shippable, disclosed contribution."""
        payload, manifest = governance.check_egress(
            self._aggregate, policy=policy, consent=consent, tier=tier,
            audit=audit, now=now, timestamp=timestamp,
        )
        return DisclosedContribution(payload=payload, manifest=manifest)

    def __repr__(self) -> str:
        return (
            f"SiteContribution(kind={self.kind!r}, sealed) — call "
            ".disclose(policy=..., consent=...) to obtain a shippable payload"
        )


def contribute_aggregate(X: pd.DataFrame, y: pd.Series, *, min_cell: int = 0) -> SiteContribution:
    """Egress entry point for the federated fit: a sealed fit contribution."""
    return SiteContribution(site_aggregates(X, y, min_cell=min_cell), kind="fit")


def contribute_contestability(
    X: pd.DataFrame,
    y: pd.Series,
    fit: dict[str, object],
    *,
    metric: str = "euclidean",
    bin_edges: np.ndarray = DEFAULT_FLIP_BINS,
) -> SiteContribution:
    """Egress entry point for federated contestability: a sealed summary."""
    return SiteContribution(
        site_contestability(X, y, fit, metric=metric, bin_edges=bin_edges),
        kind="contestability",
    )


def _unwrap(items: list[object]) -> list[dict]:
    """Accept raw aggregates (in-boundary) or DisclosedContribution (post-gate)."""
    return [it.payload if isinstance(it, DisclosedContribution) else it for it in items]
