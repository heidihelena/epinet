"""Feature-space clustering and centroid distances for network nodes.

This is the attribute-space counterpart to the toolkit's graph shortest paths.
Where ``shortest_path_records`` measures *topological* distance (hops/weights
along edges), this module measures *feature-space* distance: each node becomes
a standardized vector of graph features plus numeric attributes, and we

1. group nodes with k-means centroids (unsupervised structure),
2. measure each node's distance to its cluster centroid, and
3. measure each node's distance to every outcome-class centroid — a transparent
   nearest-centroid (Rocchio) view, analogous to scoring a lung nodule by its
   distance to risk-tier centroids in a standardized feature space.

Two metrics are supported: ``euclidean`` (isotropic) and ``mahalanobis``
(accounts for feature correlation and scale via the pooled covariance), the
latter being the natural choice when features are correlated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import epinet_common


def standardize(X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Z-score the feature matrix, dropping zero-variance columns.

    Constant features carry no distance information and would make the
    covariance singular, so they are removed before scaling.
    """
    variances = X.var(axis=0, ddof=0)
    keep = [column for column in X.columns if variances[column] > 1e-12]
    scaled = StandardScaler().fit_transform(X[keep].to_numpy(dtype=float))
    return scaled, keep


def _mahalanobis_inverse_cov(Xz: np.ndarray) -> np.ndarray:
    """Ridge-regularised pseudo-inverse of the covariance of standardized X."""
    cov = np.cov(Xz, rowvar=False)
    cov = np.atleast_2d(cov)
    ridge = 1e-6 * np.eye(cov.shape[0])
    return np.linalg.pinv(cov + ridge)


def distances_to_points(
    Xz: np.ndarray,
    points: np.ndarray,
    *,
    metric: str = "euclidean",
    inv_cov: np.ndarray | None = None,
) -> np.ndarray:
    """Distance from every row of Xz to every centroid in ``points``.

    Returns an (n_samples, n_points) array.
    """
    points = np.atleast_2d(points)
    if metric == "euclidean":
        diff = Xz[:, None, :] - points[None, :, :]
        return np.sqrt((diff**2).sum(axis=2))
    if metric == "mahalanobis":
        if inv_cov is None:
            inv_cov = _mahalanobis_inverse_cov(Xz)
        diff = Xz[:, None, :] - points[None, :, :]
        # sqrt( diff @ inv_cov @ diff ) per (sample, point)
        tmp = np.einsum("npd,de->npe", diff, inv_cov)
        sq = np.einsum("npe,npe->np", tmp, diff)
        return np.sqrt(np.clip(sq, 0.0, None))
    raise ValueError(f"Unknown distance metric: {metric}")


def choose_k(Xz: np.ndarray, k_range: range) -> tuple[int, dict[int, float]]:
    """Pick the number of clusters maximizing the silhouette score."""
    scores: dict[int, float] = {}
    for k in k_range:
        if k < 2 or k >= len(Xz):
            continue
        labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(Xz)
        if len(set(labels)) < 2:
            continue
        scores[k] = float(silhouette_score(Xz, labels))
    if not scores:
        return min(2, len(Xz)), scores
    return max(scores, key=scores.get), scores


def class_centroids(Xz: np.ndarray, y: pd.Series) -> tuple[list[str], np.ndarray]:
    """Mean feature vector per outcome class (Rocchio centroids)."""
    classes = sorted(y.dropna().unique(), key=str)
    centroids = np.vstack([Xz[(y == cls).to_numpy()].mean(axis=0) for cls in classes])
    return [str(c) for c in classes], centroids


def cluster_nodes(
    X: pd.DataFrame,
    *,
    y: pd.Series | None = None,
    n_clusters: int = 0,
    metric: str = "euclidean",
    random_state: int = 42,
    labeled_only: bool = False,
) -> dict[str, object]:
    """Cluster nodes in feature space and compute centroid distances.

    ``n_clusters <= 0`` selects k automatically: the number of outcome classes
    when labels are available, otherwise a silhouette search over 2..8.

    ``labeled_only`` restricts clustering to nodes with a non-blank outcome.
    Use it when the scaffold nodes carry no meaningful features (e.g. patient
    hub nodes with empty attributes) and would otherwise form a degenerate
    cluster.

    Returns a dict with ``assignments`` (DataFrame), ``centroids`` (DataFrame),
    and ``summary`` (JSON-serializable dict).
    """
    if labeled_only:
        if y is None:
            raise ValueError("labeled_only requires an outcome series")
        keep = epinet_common.labeled_mask(y.reindex(X.index))
        X = X.loc[keep]

    Xz, kept_columns = standardize(X)
    n_samples = Xz.shape[0]
    if n_samples < 2 or not kept_columns:
        raise ValueError("Clustering needs at least two nodes and one varying feature")

    labeled = None
    if y is not None:
        labeled = y.reindex(X.index)
        labeled = labeled.mask(epinet_common.blank_label_mask(labeled))

    silhouettes: dict[int, float] = {}
    if n_clusters <= 0:
        if labeled is not None and labeled.dropna().nunique() >= 2:
            n_clusters = int(labeled.dropna().nunique())
        else:
            n_clusters, silhouettes = choose_k(Xz, range(2, min(8, n_samples) + 1))
    n_clusters = max(2, min(n_clusters, n_samples - 1))

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit(Xz)
    cluster_labels = kmeans.labels_
    cluster_centroids = kmeans.cluster_centers_

    inv_cov = _mahalanobis_inverse_cov(Xz) if metric == "mahalanobis" else None
    dist_to_clusters = distances_to_points(Xz, cluster_centroids, metric=metric, inv_cov=inv_cov)
    dist_to_own = dist_to_clusters[np.arange(n_samples), cluster_labels]

    assignments = pd.DataFrame(index=X.index)
    assignments.index.name = "ID"
    assignments["cluster"] = cluster_labels
    assignments["distance_to_cluster_centroid"] = dist_to_own

    summary: dict[str, object] = {
        "n_clusters": int(n_clusters),
        "distance_metric": metric,
        "n_features": len(kept_columns),
        "feature_columns": kept_columns,
        "inertia": float(kmeans.inertia_),
        "cluster_sizes": {int(c): int((cluster_labels == c).sum()) for c in range(n_clusters)},
    }
    if silhouettes:
        summary["silhouette_by_k"] = {int(k): v for k, v in silhouettes.items()}
    if len(set(cluster_labels)) >= 2:
        summary["silhouette"] = float(silhouette_score(Xz, cluster_labels))

    # Outcome-class centroids: nearest-centroid classifier in feature space.
    if labeled is not None and labeled.dropna().nunique() >= 2:
        classes, centroids = class_centroids(Xz, labeled)
        dist_to_classes = distances_to_points(Xz, centroids, metric=metric, inv_cov=inv_cov)
        nearest_idx = dist_to_classes.argmin(axis=1)
        assignments["nearest_class_centroid"] = [classes[i] for i in nearest_idx]
        for j, cls in enumerate(classes):
            assignments[f"dist_to_{cls}"] = dist_to_classes[:, j]
        assignments["outcome"] = labeled.to_numpy()

        labeled_mask = labeled.notna().to_numpy()
        agree = (
            assignments.loc[labeled_mask, "nearest_class_centroid"].to_numpy()
            == labeled[labeled_mask].astype(str).to_numpy()
        )
        # Pairwise separation between class centroids.
        sep = distances_to_points(centroids, centroids, metric=metric, inv_cov=inv_cov)
        summary["class_centroids"] = {
            "classes": classes,
            "nearest_centroid_insample_accuracy": float(agree.mean()) if agree.size else None,
            "centroid_separation": {
                f"{classes[a]}|{classes[b]}": float(sep[a, b])
                for a in range(len(classes))
                for b in range(a + 1, len(classes))
            },
        }
        # Cross-tab of kmeans cluster vs outcome (composition of each cluster).
        composition = (
            pd.crosstab(assignments["cluster"], assignments["outcome"]).to_dict(orient="index")
        )
        summary["cluster_outcome_composition"] = {
            int(k): {str(c): int(n) for c, n in row.items()} for k, row in composition.items()
        }

    centroid_frame = pd.DataFrame(cluster_centroids, columns=kept_columns)
    centroid_frame.insert(0, "cluster", range(n_clusters))

    return {
        "assignments": assignments.reset_index(),
        "centroids": centroid_frame,
        "summary": summary,
        "_standardized": Xz,
        "_kept_columns": kept_columns,
        "_cluster_labels": cluster_labels,
    }


def run_clustering(
    X: pd.DataFrame,
    output_dir: Path,
    *,
    y: pd.Series | None = None,
    n_clusters: int = 0,
    metric: str = "euclidean",
    random_state: int = 42,
    labeled_only: bool = False,
) -> dict[str, object]:
    """Cluster and write node_clusters.csv, cluster_centroids.csv, cluster_summary.json."""
    result = cluster_nodes(
        X, y=y, n_clusters=n_clusters, metric=metric,
        random_state=random_state, labeled_only=labeled_only,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result["assignments"].to_csv(output_dir / "node_clusters.csv", index=False)
    result["centroids"].to_csv(output_dir / "cluster_centroids.csv", index=False)
    (output_dir / "cluster_summary.json").write_text(json.dumps(result["summary"], indent=2) + "\n")
    return result
