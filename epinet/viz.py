"""Visualization counterpart to epinet_toolkit.

The toolkit produces tabular outputs (features, paths, model metrics); this module
renders them as figures so a run can be inspected at a glance:

- network overview colored by outcome, with target nodes and nearest paths highlighted
- degree distribution
- model feature importance (with cross-iteration variability when available)
- metric stability across evaluation iterations
- confusion matrix heatmap

All functions write a figure and return the written path. Matplotlib runs with the
Agg backend so plotting works in headless environments.

Publication standards: a single house style (``HOUSE_STYLE`` / ``apply_house_style``)
gives every figure consistent typography, removes top/right spines, and uses a
colorblind-friendly palette. Figures render at 300 DPI by default (``DEFAULT_DPI``)
and can be written as raster (PNG) or vector (PDF/SVG) via ``image_format``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from epinet import common as epinet_common

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
HIGHLIGHT = "#D81B60"  # accessible magenta for targets/paths (distinct from palette)

# Print-quality default; override per call or via the CLI (--plot-dpi).
DEFAULT_DPI = 300

# A single house style applied to every figure, so typography, spines, and grid
# are consistent across the whole figure set (publication standard).
HOUSE_STYLE = {
    "figure.dpi": 150,
    "savefig.dpi": DEFAULT_DPI,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.grid": False,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "figure.titlesize": 14,
}


def apply_house_style() -> None:
    """Apply the shared figure style to matplotlib's global rcParams."""
    plt.rcParams.update(HOUSE_STYLE)


apply_house_style()


def _mpl_at_least(major: int, minor: int) -> bool:
    """True if the installed matplotlib is at least major.minor."""
    parts = matplotlib.__version__.split(".")
    try:
        return (int(parts[0]), int(parts[1])) >= (major, minor)
    except (ValueError, IndexError):
        return True  # unrecognizable (dev/rc) version string: assume modern


# ``boxplot`` renamed the ``labels`` keyword to ``tick_labels`` in matplotlib 3.9;
# the repo supports matplotlib >= 3.3.2, so pick the keyword the runtime accepts.
_BOXPLOT_LABEL_KW = "tick_labels" if _mpl_at_least(3, 9) else "labels"


def _save(fig: plt.Figure, output_path: Path, *, dpi: int | None = None) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Vector formats ignore dpi; raster honours DEFAULT_DPI unless overridden.
    fig.savefig(output_path, dpi=dpi or DEFAULT_DPI)
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
            return "unlabeled" if epinet_common.is_blank_value(value) else str(value).strip()

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
            edgecolors=HIGHLIGHT,
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
            edge_color=HIGHLIGHT,
            width=1.8,
            alpha=0.85,
        )

    if graph.number_of_nodes() <= 50:
        nx.draw_networkx_labels(graph, pos, ax=ax, font_size=7)

    ax.set_title("Network overview" + (" (targets outlined, nearest paths highlighted)" if targets else ""))
    ax.set_axis_off()
    return _save(fig, output_path)


_INTERACTIVE_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  body {{ margin: 0; font-family: sans-serif; }}
  #net {{ width: 100vw; height: 92vh; border-bottom: 1px solid #ddd; }}
  #legend {{ padding: 6px 12px; font-size: 13px; }}
  .swatch {{ display: inline-block; width: 12px; height: 12px; margin: 0 4px 0 12px;
            border-radius: 2px; vertical-align: middle; }}
</style></head>
<body>
<div id="net"></div>
<div id="legend">{legend}</div>
<script>
  const nodes = new vis.DataSet({nodes});
  const edges = new vis.DataSet({edges});
  new vis.Network(document.getElementById("net"), {{nodes, edges}}, {{
    nodes: {{ shape: "dot", size: 12, font: {{ size: 12 }} }},
    edges: {{ color: {{ color: "#cccccc" }}, smooth: false }},
    physics: {{ stabilization: true, barnesHut: {{ gravitationalConstant: -8000 }} }},
    interaction: {{ hover: true, tooltipDelay: 120 }}
  }});
</script></body></html>
"""


def network_to_html(
    graph: nx.Graph,
    output_path: Path,
    *,
    outcome_attribute: str | None = None,
    title: str = "EpiNet interactive network",
) -> Path:
    """Write a self-contained, draggable/zoomable HTML network (vis-network via CDN).

    No extra Python dependency: the graph is embedded as JSON and rendered by
    vis-network loaded from a CDN. Useful for graphs too large for a readable
    static spring layout. Nodes are colored by outcome (blank = gray scaffold).
    """
    import json

    def _norm(value: object) -> str:
        return "unlabeled" if epinet_common.is_blank_value(value) else str(value).strip()

    outcomes = {node: _norm(graph.nodes[node].get(outcome_attribute)) if outcome_attribute else ""
                for node in graph.nodes()}
    categories = sorted({o for o in outcomes.values() if o and o != "unlabeled"})
    color_map = {c: CATEGORY_COLORS[i % len(CATEGORY_COLORS)] for i, c in enumerate(categories)}
    color_map["unlabeled"] = "#BBBBBB"
    color_map[""] = CATEGORY_COLORS[0]

    nodes_json = json.dumps([
        {"id": str(n), "label": str(n),
         "color": color_map.get(outcomes[n], CATEGORY_COLORS[0]),
         "title": f"{n}" + (f" — {outcome_attribute}={outcomes[n]}" if outcome_attribute else "")}
        for n in graph.nodes()
    ])
    edges_json = json.dumps([{"from": str(u), "to": str(v)} for u, v in graph.edges()])

    legend = ""
    if outcome_attribute:
        for c in categories + (["unlabeled"] if "unlabeled" in outcomes.values() else []):
            label = "unlabeled (scaffold)" if c == "unlabeled" else f"{outcome_attribute}={c}"
            legend += f'<span class="swatch" style="background:{color_map[c]}"></span>{label}'

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_INTERACTIVE_TEMPLATE.format(
        title=title, legend=legend or "drag to reposition, scroll to zoom",
        nodes=nodes_json, edges=edges_json))
    return output_path


def plot_degree_distribution(graph: nx.Graph, output_path: Path) -> Path:
    degrees = [degree for _, degree in graph.degree()]
    fig, ax = plt.subplots(figsize=(7, 5))
    if degrees:
        bins = np.arange(min(degrees), max(degrees) + 2) - 0.5
        ax.hist(degrees, bins=bins, color=CATEGORY_COLORS[0], edgecolor="white")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Node count")
    ax.set_title("Degree distribution")
    ax.grid(axis="y", alpha=0.3)
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
    ax.set_xlabel("Importance" + (" (mean ± sd)" if errors is not None else ""))
    ax.set_title("Model feature importance")
    ax.grid(axis="x", alpha=0.3)
    ax.margins(y=0.01)
    return _save(fig, output_path)


def plot_metric_stability(iteration_metrics: pd.DataFrame, output_path: Path) -> Path:
    """Box plot of model metrics across evaluation iterations.

    Restricted to metrics that share the 0..1 higher-is-better scale (so the box
    plot is readable); Brier and MCC live on different scales and are reported in
    the metrics JSON / model card instead.
    """
    metric_columns = [
        column
        for column in ["accuracy", "balanced_accuracy", "f1_weighted", "roc_auc"]
        if column in iteration_metrics.columns and iteration_metrics[column].notna().any()
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [iteration_metrics[column].dropna() for column in metric_columns]
    labels = [c.replace("_weighted", "").replace("balanced_accuracy", "bal_acc") for c in metric_columns]
    ax.boxplot(data, medianprops={"color": HIGHLIGHT}, **{_BOXPLOT_LABEL_KW: labels})
    # Overlay individual iterations (jittered) so the spread is shown, not hidden.
    rng = np.random.default_rng(0)
    for i, column in enumerate(metric_columns, start=1):
        values = iteration_metrics[column].dropna()
        x = rng.normal(i, 0.04, size=len(values))
        ax.scatter(x, values, s=12, color=CATEGORY_COLORS[0], alpha=0.4, zorder=3)
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
        color=HIGHLIGHT,
        linewidth=2,
        label=f"observed mean = {observed_mean:.3f}",
    )
    ax.set_xlabel(metric)
    ax.set_ylabel("Permutations")
    ax.set_title(f"Permutation test: {metric} (p = {p_value:.3f})")
    ax.grid(axis="y", alpha=0.3)
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
    pca = PCA(n_components=n_components, random_state=0)
    coords = pca.fit_transform(standardized)
    evr = list(pca.explained_variance_ratio_)
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])
        evr.append(0.0)

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

    ax.set_xlabel(f"PC1 ({evr[0] * 100:.0f}% variance)")
    ax.set_ylabel(f"PC2 ({evr[1] * 100:.0f}% variance)")
    ax.set_title(f"Feature-space clusters ({metric} centroids; black X = cluster centroid)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, loc="best")
    return _save(fig, output_path)


def plot_confusion_matrix(
    matrix: list[list[int]],
    classes: list[str],
    output_path: Path,
) -> Path:
    array = np.asarray(matrix, dtype=float)
    row_totals = array.sum(axis=1, keepdims=True)
    row_norm = np.divide(array, row_totals, out=np.zeros_like(array), where=row_totals > 0)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    image = ax.imshow(row_norm, cmap="Blues", vmin=0, vmax=1)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized (recall)")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (held-out test set)")
    # Minor gridlines between cells for readability.
    ax.set_xticks(np.arange(-0.5, len(classes), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(classes), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            # Each cell shows the count and the row-normalized percentage.
            label = f"{int(array[i, j])}\n{row_norm[i, j] * 100:.0f}%"
            ax.text(j, i, label, ha="center", va="center",
                    color="white" if row_norm[i, j] > 0.5 else "black")
    return _save(fig, output_path)


def plot_contestability(
    assignments: pd.DataFrame,
    summary: dict,
    output_path: Path,
    *,
    top_n: int = 12,
) -> Path:
    """Two-panel contestability figure for the ``--run-contest`` lens.

    Left: histogram of per-case flip-distance with the contested region (at or
    below the threshold) shaded in the highlight color. Right: the
    value-of-information ranking — the features that most drive boundary flips.
    """
    flip = assignments["flip_distance"].to_numpy(dtype=float)
    flip = flip[np.isfinite(flip)]
    flip_summary = summary.get("flip_distance", {})
    threshold = flip_summary.get("contest_threshold")
    n_contested = flip_summary.get("n_contested", 0)
    leverage = summary.get("feature_leverage", {})

    fig, (ax_hist, ax_voi) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Left panel: flip-distance distribution, contested tail shaded.
    if flip.size:
        bins = min(30, max(8, flip.size // 5))
        _, edges, patches = ax_hist.hist(flip, bins=bins, color=CATEGORY_COLORS[0], alpha=0.85)
        if threshold is not None and np.isfinite(threshold):
            for patch, left_edge in zip(patches, edges[:-1]):
                if left_edge < threshold:
                    patch.set_facecolor(HIGHLIGHT)
            ax_hist.axvline(
                threshold,
                color=HIGHLIGHT,
                linewidth=2,
                label=f"contested ≤ {threshold:.3g} (n={n_contested})",
            )
            ax_hist.legend()
    ax_hist.set_xlabel("flip-distance (standardized-feature units)")
    ax_hist.set_ylabel("Cases")
    ax_hist.set_title("How far is each call from flipping?")
    ax_hist.grid(axis="y", alpha=0.3)

    # Right panel: value-of-information — features that most drive boundary flips.
    items = list(leverage.items())[:top_n][::-1]
    if items:
        names = [name for name, _ in items]
        shares = [value for _, value in items]
        ax_voi.barh(names, shares, color=CATEGORY_COLORS[2])
        ax_voi.set_xlabel("mean flip-gradient share")
        ax_voi.set_title("Value of information: decisive features")
        ax_voi.grid(axis="x", alpha=0.3)
        ax_voi.margins(y=0.01)
    else:
        ax_voi.axis("off")

    fig.tight_layout()
    return _save(fig, output_path)


def plot_calibration(
    y_true: np.ndarray,
    proba_pos: np.ndarray,
    output_path: Path,
    *,
    pos_label: object = None,
    brier: float | None = None,
    n_bins: int = 10,
) -> Path:
    """Reliability diagram for a binary risk score (predicted vs observed risk).

    Points on the diagonal are perfectly calibrated; points below mean the model
    over-estimates risk in that probability band, above means under-estimates.
    The histogram strip shows where the predicted probabilities actually fall, so
    sparsely-populated bins are not over-read. A discriminating model can still be
    badly off this line — which is exactly why calibration is reported.
    """
    from sklearn.calibration import calibration_curve

    y = (np.asarray(y_true) == pos_label).astype(int) if pos_label is not None else np.asarray(y_true)
    n_bins = max(2, min(n_bins, len(np.unique(proba_pos))))
    fig, (ax, ax_hist) = plt.subplots(
        2, 1, figsize=(6, 6.4), height_ratios=[4, 1], sharex=True
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="perfect calibration")
    try:
        prob_true, prob_pred = calibration_curve(y, proba_pos, n_bins=n_bins, strategy="quantile")
        ax.plot(prob_pred, prob_true, marker="o", color=HIGHLIGHT,
                label="model" + (f" (Brier={brier:.3f})" if brier is not None else ""))
    except ValueError:
        pass
    ax.set_ylabel("Observed frequency")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Calibration (reliability diagram)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    ax_hist.hist(proba_pos, bins=np.linspace(0, 1, n_bins + 1), color=CATEGORY_COLORS[0], alpha=0.8)
    ax_hist.set_xlabel("Predicted probability")
    ax_hist.set_ylabel("Count")
    ax_hist.set_xlim(-0.02, 1.02)
    ax_hist.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, output_path)


def plot_learning_curve(learning_curve: dict, output_path: Path) -> Path:
    """Cross-validated F1 vs training-set size, with ±1 sd bands.

    Makes the small-n regime visible: if the validation curve is still climbing
    at the largest training size, the model is data-starved and more cases would
    help; if train and validation curves sit far apart, it is over-fitting.
    """
    sizes = np.asarray(learning_curve["train_sizes"], dtype=float)
    train_mean = np.asarray(learning_curve["train_mean"], dtype=float)
    train_std = np.asarray(learning_curve["train_std"], dtype=float)
    test_mean = np.asarray(learning_curve["test_mean"], dtype=float)
    test_std = np.asarray(learning_curve["test_std"], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sizes, train_mean, marker="o", color=CATEGORY_COLORS[1], label="training score")
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std,
                    color=CATEGORY_COLORS[1], alpha=0.15)
    ax.plot(sizes, test_mean, marker="s", color=CATEGORY_COLORS[0], label="cross-validation score")
    ax.fill_between(sizes, test_mean - test_std, test_mean + test_std,
                    color=CATEGORY_COLORS[0], alpha=0.15)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("F1 (weighted)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Learning curve")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
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
    calibration: dict | None = None,
    learning_curve: dict | None = None,
    clustering: dict | None = None,
    contestability: dict | None = None,
    seed: int = 42,
    image_format: str = "png",
    interactive: bool = False,
) -> list[Path]:
    """Render every figure supported by the available run artifacts.

    ``image_format`` selects the file type for every figure: ``png`` (default,
    raster at DEFAULT_DPI) or a vector format (``pdf``/``svg``) for print.
    ``interactive`` additionally writes a draggable/zoomable ``network.html``.
    """
    plots_dir = output_dir / "plots"
    written: list[Path] = []

    def fp(name: str) -> Path:
        return plots_dir / f"{name}.{image_format}"

    if interactive:
        written.append(
            network_to_html(graph, plots_dir / "network.html", outcome_attribute=outcome_attribute)
        )

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
            fp("network_overview"),
            outcome_attribute=outcome_attribute,
            target_nodes=target_nodes,
            paths=paths,
            seed=seed,
        )
    )
    written.append(plot_degree_distribution(graph, fp("degree_distribution")))

    if importance is not None and not importance.empty:
        written.append(plot_feature_importance(importance, fp("feature_importance")))
    if metrics is not None and metrics.get("confusion_matrix"):
        written.append(
            plot_confusion_matrix(
                metrics["confusion_matrix"],
                metrics.get("classes", []),
                fp("confusion_matrix"),
            )
        )
    if iteration_metrics is not None and len(iteration_metrics) > 1:
        written.append(
            plot_metric_stability(iteration_metrics, fp("metric_stability"))
        )
    if calibration is not None and len(calibration.get("proba_pos", [])) > 0:
        written.append(
            plot_calibration(
                calibration["y_true"],
                calibration["proba_pos"],
                fp("calibration"),
                pos_label=calibration.get("pos_label"),
                brier=calibration.get("brier"),
            )
        )
    if learning_curve is not None:
        written.append(plot_learning_curve(learning_curve, fp("learning_curve")))
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
                fp("permutation_null"),
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
                fp("feature_clusters"),
                outcomes=outcomes,
                metric=clustering["summary"].get("distance_metric", "euclidean"),
            )
        )
    if (
        contestability is not None
        and contestability.get("assignments") is not None
        and "flip_distance" in contestability["assignments"].columns
    ):
        written.append(
            plot_contestability(
                contestability["assignments"],
                contestability["summary"],
                fp("contestability"),
            )
        )

    return written
