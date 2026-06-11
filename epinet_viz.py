"""Visualization counterpart to epinet_toolkit.

The toolkit produces tabular outputs (features, paths, model metrics); this module
renders them as figures so a run can be inspected at a glance:

- network overview colored by outcome, with target nodes and nearest paths highlighted
- degree distribution
- model feature importance (with cross-iteration variability when available)
- metric stability across evaluation iterations
- confusion matrix heatmap

All functions write PNG files and return the written path. Matplotlib runs with the
Agg backend so plotting works in headless environments.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# Colorblind-friendly cycle (Okabe-Ito) for categorical node outcomes.
CATEGORY_COLORS = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
    "#999999",
]


def _save(fig: plt.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_network(
    graph: nx.Graph,
    output_path: Path,
    *,
    outcome_attribute: str | None = None,
    target_nodes: list[str] | None = None,
    paths: list[list[str]] | None = None,
    seed: int = 42,
) -> Path:
    """Draw the graph with outcomes as colors, targets outlined, and paths overlaid."""
    pos = nx.spring_layout(graph, seed=seed)
    fig, ax = plt.subplots(figsize=(10, 8))

    if outcome_attribute:
        def _normalize(value: object) -> str:
            text = str(value).strip()
            return "unlabeled" if text in ("", "nan", "None") else text

        outcomes = [_normalize(graph.nodes[node].get(outcome_attribute)) for node in graph.nodes()]
        # Blank/NaN outcomes are scaffold nodes (semi-supervised graphs); give
        # them a neutral gray and sort them last so labeled classes keep the
        # categorical palette.
        labeled_categories = sorted({o for o in outcomes if o != "unlabeled"})
        color_map = {
            category: CATEGORY_COLORS[i % len(CATEGORY_COLORS)]
            for i, category in enumerate(labeled_categories)
        }
        if "unlabeled" in outcomes:
            color_map["unlabeled"] = "#BBBBBB"
        node_colors = [color_map[outcome] for outcome in outcomes]
        for category in labeled_categories + (["unlabeled"] if "unlabeled" in outcomes else []):
            label = "unlabeled (scaffold)" if category == "unlabeled" else f"{outcome_attribute}={category}"
            ax.scatter([], [], c=color_map[category], label=label)
        ax.legend(loc="upper left", fontsize=8)
    else:
        node_colors = CATEGORY_COLORS[0]

    node_size = max(30, 3000 // max(graph.number_of_nodes(), 1))
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.25, width=0.8)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=node_size)

    targets = [node for node in (target_nodes or []) if node in graph]
    if targets:
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=targets,
            ax=ax,
            node_color="none",
            edgecolors="red",
            linewidths=2.0,
            node_size=node_size * 2,
        )

    if paths:
        path_edges = [
            (u, v)
            for path in paths
            for u, v in zip(path[:-1], path[1:])
            if graph.has_edge(u, v)
        ]
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=path_edges,
            ax=ax,
            edge_color="red",
            width=1.8,
            alpha=0.8,
        )

    if graph.number_of_nodes() <= 50:
        nx.draw_networkx_labels(graph, pos, ax=ax, font_size=7)

    ax.set_title("Network overview" + (" (targets outlined, nearest paths in red)" if targets else ""))
    ax.set_axis_off()
    return _save(fig, output_path)


def plot_degree_distribution(graph: nx.Graph, output_path: Path) -> Path:
    degrees = [degree for _, degree in graph.degree()]
    fig, ax = plt.subplots(figsize=(7, 5))
    if degrees:
        bins = np.arange(min(degrees), max(degrees) + 2) - 0.5
        ax.hist(degrees, bins=bins, color=CATEGORY_COLORS[0], edgecolor="white")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Node count")
    ax.set_title("Degree distribution")
    return _save(fig, output_path)


def plot_feature_importance(
    importance: pd.DataFrame,
    output_path: Path,
    *,
    top_n: int = 15,
) -> Path:
    """Horizontal bar chart of feature importance.

    Expects columns `feature` and `importance`; an optional `importance_std`
    column (from iterative evaluation) is drawn as error bars.
    """
    top = importance.nlargest(top_n, "importance").iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(top))))
    errors = top["importance_std"] if "importance_std" in top.columns else None
    ax.barh(top["feature"], top["importance"], xerr=errors, color=CATEGORY_COLORS[0], capsize=3)
    ax.set_xlabel("Importance")
    ax.set_title("Model feature importance")
    return _save(fig, output_path)


def plot_metric_stability(iteration_metrics: pd.DataFrame, output_path: Path) -> Path:
    """Box plot of model metrics across evaluation iterations."""
    metric_columns = [
        column
        for column in ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
        if column in iteration_metrics.columns
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(
        [iteration_metrics[column] for column in metric_columns],
        tick_labels=metric_columns,
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Metric stability across {len(iteration_metrics)} iterations")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, output_path)


def plot_permutation_null(
    permutation_metrics: pd.DataFrame,
    observed_mean: float,
    p_value: float,
    output_path: Path,
    *,
    metric: str = "f1_weighted",
) -> Path:
    """Histogram of the label-permutation null distribution vs the observed score."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(
        permutation_metrics[metric],
        bins=min(20, max(5, len(permutation_metrics) // 5)),
        color=CATEGORY_COLORS[0],
        alpha=0.8,
        label=f"null ({len(permutation_metrics)} permuted outcomes)",
    )
    ax.axvline(
        observed_mean,
        color="red",
        linewidth=2,
        label=f"observed mean = {observed_mean:.3f}",
    )
    ax.set_xlabel(metric)
    ax.set_ylabel("Permutations")
    ax.set_title(f"Permutation test: {metric} (p = {p_value:.3f})")
    ax.legend()
    return _save(fig, output_path)


def plot_clusters_pca(
    standardized: np.ndarray,
    cluster_labels: np.ndarray,
    output_path: Path,
    *,
    outcomes: list[str] | None = None,
    metric: str = "euclidean",
) -> Path:
    """Project the standardized feature space to 2D (PCA) and show clusters.

    Points are colored by k-means cluster; centroids are drawn as black crosses.
    If outcome labels are supplied, marker shape encodes the outcome class so
    cluster/label agreement is visible at a glance.
    """
    from sklearn.decomposition import PCA

    n_components = min(2, standardized.shape[1])
    coords = PCA(n_components=n_components, random_state=0).fit_transform(standardized)
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])

    if outcomes is not None:
        outcomes = ["unlabeled" if pd.isna(o) else str(o) for o in outcomes]

    fig, ax = plt.subplots(figsize=(8, 6))
    clusters = sorted(set(cluster_labels))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    outcome_categories = sorted(set(outcomes)) if outcomes else [None]

    for ci, cluster in enumerate(clusters):
        color = CATEGORY_COLORS[ci % len(CATEGORY_COLORS)]
        for mi, category in enumerate(outcome_categories):
            mask = cluster_labels == cluster
            if outcomes is not None:
                mask = mask & (np.asarray(outcomes) == category)
            if not mask.any():
                continue
            label = f"cluster {cluster}" if category is None else f"cluster {cluster} / {category}"
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=color,
                marker=markers[mi % len(markers)],
                edgecolors="black",
                linewidths=0.4,
                s=70,
                label=label,
            )
        centroid = coords[cluster_labels == cluster].mean(axis=0)
        ax.scatter(centroid[0], centroid[1], c="black", marker="X", s=160, zorder=5)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Feature-space clusters ({metric} centroids; black X = cluster centroid)")
    ax.legend(fontsize=7, loc="best")
    return _save(fig, output_path)


def plot_confusion_matrix(
    matrix: list[list[int]],
    classes: list[str],
    output_path: Path,
) -> Path:
    array = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(array, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (held-out test set)")
    threshold = array.max() / 2 if array.size else 0
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            ax.text(
                j,
                i,
                str(array[i, j]),
                ha="center",
                va="center",
                color="white" if array[i, j] > threshold else "black",
            )
    return _save(fig, output_path)


def generate_run_plots(
    graph: nx.Graph,
    output_dir: Path,
    *,
    outcome_attribute: str | None = None,
    target_nodes: list[str] | None = None,
    nearest: pd.DataFrame | None = None,
    metrics: dict | None = None,
    importance: pd.DataFrame | None = None,
    iteration_metrics: pd.DataFrame | None = None,
    permutation_metrics: pd.DataFrame | None = None,
    clustering: dict | None = None,
    seed: int = 42,
) -> list[Path]:
    """Render every figure supported by the available run artifacts."""
    plots_dir = output_dir / "plots"
    written: list[Path] = []

    paths = None
    if nearest is not None and not nearest.empty:
        paths = [
            str(path).split(" -> ")
            for path in nearest["path"].fillna("")
            if path
        ]
    written.append(
        plot_network(
            graph,
            plots_dir / "network_overview.png",
            outcome_attribute=outcome_attribute,
            target_nodes=target_nodes,
            paths=paths,
            seed=seed,
        )
    )
    written.append(plot_degree_distribution(graph, plots_dir / "degree_distribution.png"))

    if importance is not None and not importance.empty:
        written.append(plot_feature_importance(importance, plots_dir / "feature_importance.png"))
    if metrics is not None and metrics.get("confusion_matrix"):
        written.append(
            plot_confusion_matrix(
                metrics["confusion_matrix"],
                metrics.get("classes", []),
                plots_dir / "confusion_matrix.png",
            )
        )
    if iteration_metrics is not None and len(iteration_metrics) > 1:
        written.append(
            plot_metric_stability(iteration_metrics, plots_dir / "metric_stability.png")
        )
    if (
        permutation_metrics is not None
        and not permutation_metrics.empty
        and metrics is not None
        and "permutation_test" in metrics
    ):
        f1_summary = metrics["permutation_test"]["metrics"]["f1_weighted"]
        written.append(
            plot_permutation_null(
                permutation_metrics,
                f1_summary["observed_mean"],
                f1_summary["p_value"],
                plots_dir / "permutation_null.png",
            )
        )
    if clustering is not None and "_standardized" in clustering:
        outcomes = None
        assignments = clustering.get("assignments")
        if assignments is not None and "outcome" in assignments.columns:
            outcomes = (
                assignments["outcome"].astype("string").fillna("unlabeled").tolist()
            )
        written.append(
            plot_clusters_pca(
                clustering["_standardized"],
                clustering["_cluster_labels"],
                plots_dir / "feature_clusters.png",
                outcomes=outcomes,
                metric=clustering["summary"].get("distance_metric", "euclidean"),
            )
        )

    return written
