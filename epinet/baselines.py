"""Representation baselines for the EpiNet outcome model.

A reviewer of any graph-ML method asks the obvious question: do the hand-crafted
graph features actually beat a learned node embedding, or a no-information floor?
This module answers it by running the SAME honest evaluation harness
(``epinet_toolkit.train_outcome_model`` — calibration, repeated splits, optional
permutation null) across several node representations and reporting them side by
side, at one shared seed so the only thing that varies is the representation:

- ``no_information`` — a constant feature (majority-class floor);
- ``graph_features`` — EpiNet's degree/clustering/component features;
- ``spectral_embedding`` — a learned node embedding (Laplacian eigenmaps of the
  graph, via scikit-learn ``SpectralEmbedding``);
- ``graph_features+spectral`` — both concatenated.

Scope note on GNNs. A full message-passing GNN (GCN/GraphSAGE) is the heavier
comparison and needs a deep-learning dependency that the single-file toolkit
deliberately avoids; on the small cohorts EpiNet targets, GNNs also tend to
overfit and rarely beat well-chosen features. The spectral embedding here is the
representative *node-embedding* baseline; a GNN baseline behind an optional
extra is reasonable future work.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from epinet import toolkit as et

COMPARISON_METRICS = ["roc_auc", "average_precision", "balanced_accuracy", "f1_weighted", "accuracy"]


def spectral_node_embeddings(
    graph: nx.Graph,
    *,
    n_components: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """Laplacian-eigenmap node embedding from the (weighted) graph.

    Returns a DataFrame with an ``ID`` column and ``spectral_0..k`` columns. The
    weighted adjacency is used as the affinity; edge weights act as similarities.
    Robust to small graphs by capping the embedding dimension.
    """
    from sklearn.manifold import SpectralEmbedding

    nodelist = [str(node) for node in graph.nodes()]
    n = len(nodelist)
    if n < 3:
        raise ValueError("spectral embedding needs at least 3 nodes")
    adjacency = nx.to_numpy_array(graph, nodelist=nodelist, weight="weight")
    adjacency = np.maximum(adjacency, adjacency.T)  # symmetrize for affinity
    k = max(1, min(n_components, n - 2))
    embedding = SpectralEmbedding(
        n_components=k, affinity="precomputed", random_state=seed
    ).fit_transform(adjacency)
    frame = pd.DataFrame(embedding, columns=[f"spectral_{i}" for i in range(k)])
    frame.insert(0, "ID", nodelist)
    return frame


def _no_information_features(graph: nx.Graph) -> pd.DataFrame:
    """A single constant feature — the majority-class / no-information floor."""
    return pd.DataFrame({"ID": [str(node) for node in graph.nodes()], "constant": 0.0})


def compare_representations(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    id_column: str = "ID",
    outcome_column: str = "Outcome",
    source_column: str = "SourceID",
    target_column: str = "TargetID",
    n_components: int = 8,
    n_iterations: int = 10,
    n_permutations: int = 0,
    random_state: int = 42,
    output_dir: Path | None = None,
) -> dict[str, object]:
    """Evaluate several node representations under the identical honest harness.

    Each representation is fed to ``train_outcome_model`` with a node table
    stripped to id + outcome, so the design matrix is the representation alone
    (raw node attributes are excluded for a clean representation-vs-representation
    comparison). Returns a comparison DataFrame and the per-representation metric
    dicts; writes ``baseline_comparison.{csv,md}`` when ``output_dir`` is given.
    """
    graph = et.build_graph(
        nodes, edges, id_column=id_column,
        source_column=source_column, target_column=target_column,
    )
    nodes_minimal = nodes[[id_column, outcome_column]].copy()

    representations: dict[str, pd.DataFrame] = {
        "no_information": _no_information_features(graph),
        "graph_features": et.generate_graph_features(graph),
    }
    try:
        spectral = spectral_node_embeddings(graph, n_components=n_components, seed=random_state)
        representations["spectral_embedding"] = spectral
        combined = et.generate_graph_features(graph).merge(spectral, on="ID", how="left").fillna(0.0)
        representations["graph_features+spectral"] = combined
    except (ValueError, RuntimeError):
        # Spectral embedding can fail on degenerate graphs; report what we can.
        pass

    rows: list[dict[str, object]] = []
    per_rep: dict[str, dict] = {}
    for name, features in representations.items():
        target = (output_dir / "baselines" / name) if output_dir else Path("/tmp") / f"epinet_baseline_{name}"
        result = et.train_outcome_model(
            nodes_minimal, features,
            id_column=id_column, outcome_column=outcome_column,
            output_dir=target, n_iterations=n_iterations,
            random_state=random_state, n_permutations=n_permutations, n_bootstrap=0,
        )
        metrics = result["metrics"]
        per_rep[name] = metrics
        row = {"representation": name, "n_features": int(features.shape[1] - 1)}
        row.update({m: metrics.get(m) for m in COMPARISON_METRICS})
        if n_permutations and "permutation_test" in metrics:
            row["roc_auc_p"] = metrics["permutation_test"]["metrics"].get("roc_auc", {}).get("p_value")
        rows.append(row)

    comparison = pd.DataFrame(rows)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(output_dir / "baseline_comparison.csv", index=False)
        (output_dir / "baseline_comparison.md").write_text(_markdown(comparison) + "\n")

    return {"comparison": comparison, "metrics": per_rep}


def _markdown(comparison: pd.DataFrame) -> str:
    headers = list(comparison.columns)
    lines = [
        "# Representation comparison",
        "",
        "Same honest harness (calibration, repeated splits), one shared seed; only "
        "the node representation varies.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in comparison.iterrows():
        cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
