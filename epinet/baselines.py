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
    test_size: float = 0.2,
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
    per_rep_iterations: dict[str, pd.DataFrame] = {}
    for name, features in representations.items():
        target = (output_dir / "baselines" / name) if output_dir else Path("/tmp") / f"epinet_baseline_{name}"
        result = et.train_outcome_model(
            nodes_minimal, features,
            id_column=id_column, outcome_column=outcome_column,
            output_dir=target, n_iterations=n_iterations, test_size=test_size,
            random_state=random_state, n_permutations=n_permutations, n_bootstrap=0,
        )
        metrics = result["metrics"]
        per_rep[name] = metrics
        per_rep_iterations[name] = result["iteration_metrics"]
        row = {"representation": name, "n_features": int(features.shape[1] - 1)}
        row.update({m: metrics.get(m) for m in COMPARISON_METRICS})
        if n_permutations and "permutation_test" in metrics:
            row["roc_auc_p"] = metrics["permutation_test"]["metrics"].get("roc_auc", {}).get("p_value")
        rows.append(row)

    comparison = pd.DataFrame(rows)
    paired = _paired_difference_tests(per_rep_iterations)
    paired_baseline = _paired_baseline_margin(per_rep_iterations, test_size=test_size)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(output_dir / "baseline_comparison.csv", index=False)
        md = _markdown(comparison)
        if paired.get("comparisons"):
            md += "\n\n" + _paired_markdown(paired)
        if paired_baseline is not None:
            md += "\n\n" + _paired_baseline_markdown(paired_baseline)
        (output_dir / "baseline_comparison.md").write_text(md + "\n")

    return {"comparison": comparison, "metrics": per_rep,
            "paired_tests": paired, "paired_baseline": paired_baseline}


def _paired_baseline_margin(
    per_rep_iterations: dict[str, pd.DataFrame],
    *,
    test_size: float,
    metric: str = "roc_auc",
    fallback_metric: str = "balanced_accuracy",
    model_rep: str = "graph_features",
    floor_rep: str = "no_information",
    threshold: float = 0.02,
) -> dict[str, object] | None:
    """Paired model-vs-floor AUROC margin on shared splits (Nadeau–Bengio corrected).

    The model representation and the no-information floor are scored on the *same*
    splits — ``_split_indices`` depends only on ``(n, seed, y)``, never on the
    feature values, so iteration ``i`` is the identical partition for both. The
    margin is therefore paired per split::

        Δ_i = AUROC_model(i) − AUROC_floor(i)        # same split, same nodes

    and split-difficulty variance (some test sets are just easy) cancels in the
    pairing — the very noise that makes a raw 0.02 line meaningless. We use
    ``graph_features`` (EpiNet's core representation) as the model leg rather than
    the best-scoring one, so the claim is not cherry-picked over representations.

    Honesty correction: the re-splits overlap, so a naïve t-interval on the Δ_i is
    optimistic (Nadeau & Bengio, 2003). We inflate the variance by ``1/k + ρ``
    with ``ρ = test/train`` before forming the interval, giving a directional-but-
    not-overconfident band. (A within-split cluster-paired bootstrap would be the
    fully honest community-mode version, but this harness uses random splits.)
    """
    model_df = per_rep_iterations.get(model_rep)
    floor_df = per_rep_iterations.get(floor_rep)
    if model_df is None or floor_df is None:
        return None
    chosen = metric
    if not all(
        m in df.columns and df[m].notna().any()
        for df in (model_df, floor_df) for m in (metric,)
    ):
        chosen = fallback_metric
    if not all(chosen in df.columns and df[chosen].notna().any() for df in (model_df, floor_df)):
        return None

    m = model_df.set_index("iteration")[chosen]
    f = floor_df.set_index("iteration")[chosen]
    paired = pd.concat([m, f], axis=1, join="inner").dropna()
    k = int(paired.shape[0])
    if k < 2:
        return {"metric": chosen, "model_representation": model_rep, "n_pairs": k,
                "mean_margin": None, "margin_ci_lower": None, "margin_ci_upper": None,
                "threshold": threshold, "note": "Too few paired splits for an interval."}

    diff = (paired.iloc[:, 0] - paired.iloc[:, 1]).to_numpy()
    mean = float(diff.mean())
    var = float(diff.var(ddof=1))
    rho = test_size / (1.0 - test_size)
    corrected_se = float(np.sqrt((1.0 / k + rho) * var))
    try:
        from scipy.stats import t

        tcrit = float(t.ppf(0.975, df=k - 1))
    except (ImportError, ValueError):
        tcrit = 1.96
    lo, hi = mean - tcrit * corrected_se, mean + tcrit * corrected_se
    return {
        "metric": chosen,
        "model_representation": model_rep,
        "n_pairs": k,
        "mean_margin": mean,
        "margin_ci_lower": float(lo),
        "margin_ci_upper": float(hi),
        "threshold": threshold,
        "correction": f"Nadeau–Bengio resampled-t (rho={rho:.3f}, df={k - 1})",
    }


def _paired_baseline_markdown(pb: dict[str, object]) -> str:
    if pb.get("mean_margin") is None:
        return (f"## Paired baseline margin\n\nOnly {pb['n_pairs']} paired split(s) — "
                "too few for a margin interval.")
    return (
        "## Paired baseline margin\n\n"
        f"`{pb['model_representation']}` vs `no_information` on the **same** "
        f"{pb['n_pairs']} splits (paired per split, so split-difficulty variance "
        f"cancels): mean **{pb['metric']}** margin {pb['mean_margin']:+.3f}, 95% CI "
        f"[{pb['margin_ci_lower']:+.3f}, {pb['margin_ci_upper']:+.3f}] "
        f"({pb['correction']}). The claim gate reads whether this CI clears the "
        f"{pb['threshold']:.2f} line or straddles it (not resolvable at this n)."
    )


def _paired_difference_tests(
    per_rep_iterations: dict[str, pd.DataFrame],
    metric: str = "roc_auc",
    fallback_metric: str = "balanced_accuracy",
) -> dict[str, object]:
    """Paired signed-rank test of the best representation against every other.

    The representations share one seed, so iteration ``i`` is the *same* held-out
    split for every representation — the per-iteration scores are paired. That
    pairing removes split-to-split variance from the comparison, so a difference
    that survives it is about the representation, not the luck of the split. This
    is what backs the "graph features earn their place" claim: side-by-side means
    can differ purely by split noise; a paired test cannot be fooled that way.

    Uses a Wilcoxon signed-rank test (non-parametric: no normality assumption on
    the typically small number of iterations). Reports the best representation,
    and for each other one the mean paired difference, the number of splits the
    best representation wins, and the two-sided p-value.
    """
    # Which metric is actually present across iterations?
    chosen = metric
    if not all(metric in df.columns and df[metric].notna().any() for df in per_rep_iterations.values()):
        chosen = fallback_metric
    usable = {
        name: df.set_index("iteration")[chosen]
        for name, df in per_rep_iterations.items()
        if chosen in df.columns and df[chosen].notna().any()
    }
    if len(usable) < 2:
        return {"metric": chosen, "comparisons": [],
                "note": "Not enough representations with a defined metric to compare."}

    best = max(usable, key=lambda n: float(usable[n].mean()))
    best_scores = usable[best]
    comparisons: list[dict[str, object]] = []
    for name, scores in usable.items():
        if name == best:
            continue
        paired = pd.concat([best_scores, scores], axis=1, join="inner").dropna()
        if paired.shape[0] < 2:
            comparisons.append({"representation": name, "n_pairs": int(paired.shape[0]),
                                "mean_difference": None, "best_wins": None, "p_value": None,
                                "note": "Too few paired splits for a signed-rank test."})
            continue
        diff = (paired.iloc[:, 0] - paired.iloc[:, 1]).to_numpy()
        wins = int((diff > 0).sum())
        try:
            import warnings

            from scipy.stats import wilcoxon

            # All-zero differences (identical scores) make the test undefined;
            # treat that as "no detectable difference" (p = 1). The small-n
            # zero-handling/normal-approximation warnings are expected here and the
            # caveat is already carried by n_pairs, so silence just those.
            if np.allclose(diff, 0.0):
                p_value = 1.0
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    p_value = float(wilcoxon(diff).pvalue)
        except (ImportError, ValueError):
            p_value = None
        comparisons.append({
            "representation": name,
            "n_pairs": int(len(diff)),
            "mean_difference": float(diff.mean()),
            "best_wins": wins,
            "p_value": p_value,
        })

    return {"metric": chosen, "best_representation": best, "comparisons": comparisons,
            "test": "Wilcoxon signed-rank (paired across shared-seed splits)"}


def _paired_markdown(paired: dict[str, object]) -> str:
    best = paired.get("best_representation")
    lines = [
        "## Paired difference test",
        "",
        f"Best representation by **{paired['metric']}**: `{best}`. Because all "
        "representations share one seed, each iteration is the same held-out split, "
        "so the per-split scores are paired — this removes split noise from the "
        "comparison (Wilcoxon signed-rank, two-sided).",
        "",
        "| representation | n pairs | mean Δ (best − this) | best wins | p-value |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c in paired["comparisons"]:
        md = "—" if c["mean_difference"] is None else f"{c['mean_difference']:+.3f}"
        wins = "—" if c["best_wins"] is None else f"{c['best_wins']}/{c['n_pairs']}"
        p = "—" if c.get("p_value") is None else f"{c['p_value']:.3f}"
        lines.append(f"| {c['representation']} | {c['n_pairs']} | {md} | {wins} | {p} |")
    return "\n".join(lines)


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
