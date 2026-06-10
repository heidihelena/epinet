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
        outcomes = [str(graph.nodes[node].get(outcome_attribute, "")) for node in graph.nodes()]
        categories = sorted(set(outcomes))
        color_map = {
            category: CATEGORY_COLORS[i % len(CATEGORY_COLORS)]
            for i, category in enumerate(categories)
        }
        node_colors = [color_map[outcome] for outcome in outcomes]
        for category in categories:
            ax.scatter([], [], c=color_map[category], label=f"{outcome_attribute}={category}")
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

    return written
