"""General network analysis toolkit for node/edge CSV data.

This script keeps two analytic lenses side by side:

1. graph-derived node features, optionally used for a simple outcome model
2. shortest-path summaries from source nodes to target/outcome nodes

It is intentionally domain-neutral. Epidemiology is one use case, not a hard-coded
assumption.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, train_test_split


def split_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_tables(nodes_file: str, edges_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(nodes_file), pd.read_csv(edges_file)


def validate_tables(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    id_column: str,
    source_column: str,
    target_column: str,
    outcome_column: str | None = None,
) -> dict[str, int | list[str]]:
    required_nodes = {id_column}
    if outcome_column:
        required_nodes.add(outcome_column)
    required_edges = {source_column, target_column}

    missing_node_cols = sorted(required_nodes - set(nodes.columns))
    missing_edge_cols = sorted(required_edges - set(edges.columns))
    if missing_node_cols or missing_edge_cols:
        raise ValueError(
            f"Missing columns: nodes={missing_node_cols or 'ok'}, "
            f"edges={missing_edge_cols or 'ok'}"
        )

    node_ids = set(nodes[id_column].astype(str))
    edge_endpoints = set(edges[source_column].astype(str)) | set(edges[target_column].astype(str))
    missing_endpoints = sorted(edge_endpoints - node_ids)
    duplicate_node_ids = int(nodes[id_column].astype(str).duplicated().sum())
    duplicate_edges = int(edges[[source_column, target_column]].astype(str).duplicated().sum())

    if duplicate_node_ids:
        raise ValueError(f"Node IDs must be unique; found {duplicate_node_ids} duplicates")
    if missing_endpoints:
        raise ValueError(f"Edges reference unknown node IDs: {missing_endpoints[:10]}")

    return {
        "node_count": len(nodes),
        "edge_rows": len(edges),
        "duplicate_edge_rows": duplicate_edges,
        "missing_endpoints": missing_endpoints,
    }


def build_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    id_column: str = "ID",
    source_column: str = "SourceID",
    target_column: str = "TargetID",
    directed: bool = False,
    weight_column: str | None = None,
) -> nx.Graph:
    graph: nx.Graph = nx.DiGraph() if directed else nx.Graph()

    for attrs in nodes.to_dict("records"):
        node_id = str(attrs.pop(id_column))
        graph.add_node(node_id, **attrs)

    for attrs in edges.to_dict("records"):
        source = str(attrs.pop(source_column))
        target = str(attrs.pop(target_column))
        if weight_column and weight_column in attrs:
            attrs["weight"] = float(attrs[weight_column])
        graph.add_edge(source, target, **attrs)

    return graph


def graph_summary(graph: nx.Graph) -> dict[str, int | float | bool]:
    undirected = graph.to_undirected()
    degrees = dict(graph.degree())
    return {
        "directed": graph.is_directed(),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "components": nx.number_connected_components(undirected),
        "isolates": nx.number_of_isolates(graph),
        "density": nx.density(graph),
        "mean_degree": float(np.mean(list(degrees.values()))) if degrees else 0.0,
        "max_degree": max(degrees.values()) if degrees else 0,
    }


def generate_graph_features(graph: nx.Graph, *, include_centrality: bool = False) -> pd.DataFrame:
    undirected = graph.to_undirected()
    component_sizes: dict[str, int] = {}
    for component in nx.connected_components(undirected):
        size = len(component)
        for node in component:
            component_sizes[str(node)] = size

    features = pd.DataFrame(index=[str(node) for node in graph.nodes()])
    features.index.name = "ID"
    features["degree"] = pd.Series(dict(graph.degree()), dtype=float)
    features["weighted_degree"] = pd.Series(dict(graph.degree(weight="weight")), dtype=float)
    features["clustering"] = pd.Series(nx.clustering(undirected), dtype=float)
    features["component_size"] = pd.Series(component_sizes, dtype=float)
    features["is_isolate"] = features["degree"].eq(0).astype(int)

    if include_centrality:
        features["betweenness"] = pd.Series(nx.betweenness_centrality(graph), dtype=float)
        features["closeness"] = pd.Series(nx.closeness_centrality(graph), dtype=float)
        try:
            features["pagerank"] = pd.Series(nx.pagerank(graph, weight="weight"), dtype=float)
        except nx.NetworkXException:
            features["pagerank"] = np.nan

    return features.reset_index()


def numeric_node_attributes(
    nodes: pd.DataFrame,
    *,
    id_column: str,
    exclude_columns: Iterable[str],
) -> pd.DataFrame:
    excluded = set(exclude_columns)
    numeric = nodes.drop(columns=[c for c in excluded if c in nodes.columns], errors="ignore")
    numeric = numeric.select_dtypes(include=[np.number]).copy()
    numeric[id_column] = nodes[id_column].astype(str)
    return numeric.set_index(id_column)


def build_design_matrix(
    features: pd.DataFrame,
    nodes: pd.DataFrame,
    *,
    id_column: str,
    outcome_column: str | None = None,
) -> pd.DataFrame:
    """Join graph features with numeric node attributes into one matrix.

    This is the shared feature representation used by both the supervised
    outcome model and the feature-space clustering, indexed by node ID and
    covering every node in the graph (scaffold included).
    """
    graph_features = features.set_index("ID")
    exclude = [id_column] + ([outcome_column] if outcome_column else [])
    node_numeric = numeric_node_attributes(nodes, id_column=id_column, exclude_columns=exclude)
    return graph_features.join(node_numeric, how="left").fillna(0)


def community_labels(graph: nx.Graph) -> pd.Series:
    """Assign each node a community id via greedy modularity maximization.

    Used for community-aware train/test splitting: connected nodes share
    information through their graph features, so splitting them across train
    and test leaks structure. Keeping whole communities on one side of the
    split gives a more honest estimate of generalization to unseen regions
    of the network.
    """
    undirected = graph.to_undirected()
    labels: dict[str, int] = {}
    for i, community in enumerate(nx.community.greedy_modularity_communities(undirected)):
        for node in community:
            labels[str(node)] = i
    return pd.Series(labels, name="community")


def _split_indices(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    random_state: int,
    stratify_ok: bool,
    groups: pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        return next(splitter.split(X, y, groups))
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify_ok else None,
    )
    return train_idx, test_idx


def _split_and_score(
    X: pd.DataFrame,
    y: pd.Series,
    model: RandomForestClassifier,
    *,
    test_size: float,
    random_state: int,
    stratify_ok: bool,
    groups: pd.Series | None = None,
) -> tuple[RandomForestClassifier, dict[str, object], pd.Series, pd.Series]:
    train_idx, test_idx = _split_indices(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify_ok=stratify_ok,
        groups=groups,
    )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    fitted = clone(model).set_params(random_state=random_state).fit(X_train, y_train)
    predictions = fitted.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )
    metrics = {
        "random_state": random_state,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    return fitted, metrics, y_test, pd.Series(predictions, index=y_test.index)


def train_outcome_model(
    nodes: pd.DataFrame,
    features: pd.DataFrame,
    *,
    id_column: str,
    outcome_column: str,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    n_iterations: int = 1,
    groups: pd.Series | None = None,
    n_permutations: int = 0,
) -> dict[str, object]:
    """Fit and evaluate the outcome model, optionally over repeated splits.

    A single train/test split is a noisy estimate on small networks, so when
    ``n_iterations > 1`` the model is re-evaluated on ``n_iterations`` different
    splits (seeds ``random_state .. random_state + n_iterations - 1``) and the
    summary reports the mean, standard deviation, and range of each metric.
    Hyperparameters are tuned once on the primary split and held fixed across
    iterations so the loop measures split variance, not tuning variance.

    Nodes whose outcome is blank/NaN are treated as unlabeled scaffold: they
    contribute to the graph features but are excluded from training and
    evaluation (semi-supervised setting). The counts are reported as
    ``labeled_rows`` and ``unlabeled_excluded``.

    ``groups`` (a node-id -> community-id Series, e.g. from
    ``community_labels``) switches every split to GroupShuffleSplit so train
    and test never share a community, avoiding leakage through graph features.

    ``n_permutations > 0`` runs a label-permutation null model: the outcome is
    shuffled and re-evaluated with the same tuned configuration and split
    scheme. The summary reports an empirical one-sided p-value per metric —
    the chance that shuffled labels score at least as well as the observed
    mean. If the observed metrics sit inside the null distribution, the
    features carry no detectable signal for the outcome.

    Returns a dict with keys ``metrics`` (JSON-serializable summary),
    ``importance``, ``iteration_metrics``, and ``permutation_metrics``
    (DataFrames; the last is None unless permutations were requested).
    """
    if outcome_column not in nodes.columns:
        raise ValueError(f"Outcome column not found: {outcome_column}")
    if n_iterations < 1:
        raise ValueError("n_iterations must be at least 1")
    if n_permutations < 0:
        raise ValueError("n_permutations must be non-negative")

    X = build_design_matrix(features, nodes, id_column=id_column, outcome_column=outcome_column)

    y = nodes.assign(**{id_column: nodes[id_column].astype(str)}).set_index(id_column)[outcome_column]

    # Partially-labeled graphs: nodes with a blank/NaN outcome are scaffold
    # (e.g. infrastructure or context nodes). They still shaped the graph
    # features computed above, but are excluded from supervised training and
    # evaluation. This is the common semi-supervised setting where only some
    # node types carry the label of interest.
    labeled = y.notna()
    if not pd.api.types.is_numeric_dtype(y):
        text = y.astype("string").fillna("").str.strip().str.lower()
        labeled &= ~text.isin(["", "nan", "none"])
    n_unlabeled = int((~labeled).sum())
    if not labeled.any():
        raise ValueError("Outcome modeling needs at least one labeled node")
    y = y[labeled]
    X = X.loc[y.index]
    if y.dtype == "object":
        y = y.astype("category")

    class_counts = y.value_counts()
    if len(class_counts) < 2:
        raise ValueError("Outcome modeling needs at least two outcome classes")

    split_note: str | None = None
    if groups is not None:
        groups = groups.reindex(X.index)
        groups = groups.fillna(-1)
        if groups.nunique() < 2:
            split_note = "community split requested but graph has a single community; using random splits"
            groups = None

    stratify_ok = bool(class_counts.min() >= 2) and groups is None

    # Tune hyperparameters once on the primary split.
    train_idx, _ = _split_indices(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify_ok=stratify_ok,
        groups=groups,
    )
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    min_train_class = y_train.value_counts().min()
    cv = min(5, int(min_train_class))
    base_model = RandomForestClassifier(random_state=random_state, n_jobs=1)

    if cv >= 2:
        search = GridSearchCV(
            base_model,
            {"n_estimators": [100, 200], "max_depth": [None, 5, 10]},
            cv=cv,
            n_jobs=1,
            scoring="f1_weighted",
        )
        search.fit(X_train, y_train)
        tuned_model = search.best_estimator_
        best_params = search.best_params_
    else:
        tuned_model = base_model
        best_params = {"note": "insufficient class counts for cross-validation"}

    # Iterative evaluation: re-split and re-fit with the tuned configuration.
    iteration_rows: list[dict[str, object]] = []
    importance_runs: list[np.ndarray] = []
    primary_fit: RandomForestClassifier | None = None
    primary_truth: pd.Series | None = None
    primary_predictions: pd.Series | None = None
    for i in range(n_iterations):
        fitted, row, truth, predictions = _split_and_score(
            X,
            y,
            tuned_model,
            test_size=test_size,
            random_state=random_state + i,
            stratify_ok=stratify_ok,
            groups=groups,
        )
        row["iteration"] = i
        iteration_rows.append(row)
        importance_runs.append(fitted.feature_importances_)
        if i == 0:
            primary_fit = fitted
            primary_truth = truth
            primary_predictions = predictions

    iteration_metrics = pd.DataFrame(iteration_rows)
    metric_columns = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    metrics: dict[str, object] = {
        column: float(iteration_metrics[column].iloc[0]) for column in metric_columns
    }
    metrics.update(
        {
            "best_params": best_params,
            "n_iterations": n_iterations,
            "split_strategy": "community" if groups is not None else "random",
            "labeled_rows": int(len(y)),
            "unlabeled_excluded": n_unlabeled,
            "train_rows": int(iteration_metrics["train_rows"].iloc[0]),
            "test_rows": int(iteration_metrics["test_rows"].iloc[0]),
            "classes": [str(c) for c in primary_fit.classes_],
            "confusion_matrix": confusion_matrix(primary_truth, primary_predictions).tolist(),
        }
    )
    if groups is not None:
        metrics["n_groups"] = int(groups.nunique())
    if split_note:
        metrics["split_note"] = split_note
    if n_iterations > 1:
        metrics["iteration_summary"] = {
            column: {
                "mean": float(iteration_metrics[column].mean()),
                "std": float(iteration_metrics[column].std(ddof=1)),
                "min": float(iteration_metrics[column].min()),
                "max": float(iteration_metrics[column].max()),
            }
            for column in metric_columns
        }

    importance_array = np.vstack(importance_runs)
    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": importance_array.mean(axis=0),
        }
    )
    if n_iterations > 1:
        importance["importance_std"] = importance_array.std(axis=0, ddof=1)
    importance = importance.sort_values("importance", ascending=False)

    # Permutation null model: shuffle the outcome and re-evaluate with the
    # same tuned configuration and split scheme.
    permutation_metrics: pd.DataFrame | None = None
    if n_permutations > 0:
        rng = np.random.default_rng(random_state)
        permutation_rows: list[dict[str, object]] = []
        for i in range(n_permutations):
            y_permuted = pd.Series(rng.permutation(y.to_numpy()), index=y.index, name=y.name)
            _, row, _, _ = _split_and_score(
                X,
                y_permuted,
                tuned_model,
                test_size=test_size,
                random_state=random_state + i,
                stratify_ok=stratify_ok,
                groups=groups,
            )
            row["permutation"] = i
            permutation_rows.append(row)
        permutation_metrics = pd.DataFrame(permutation_rows)
        metrics["permutation_test"] = {
            "n_permutations": n_permutations,
            "metrics": {
                column: {
                    "observed_mean": float(iteration_metrics[column].mean()),
                    "null_mean": float(permutation_metrics[column].mean()),
                    "null_std": float(permutation_metrics[column].std(ddof=1)),
                    # Empirical one-sided p-value with add-one smoothing.
                    "p_value": float(
                        (1 + int((permutation_metrics[column] >= iteration_metrics[column].mean()).sum()))
                        / (n_permutations + 1)
                    ),
                }
                for column in metric_columns
            },
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    importance.to_csv(output_dir / "model_feature_importance.csv", index=False)
    if n_iterations > 1:
        iteration_metrics.to_csv(output_dir / "model_iteration_metrics.csv", index=False)
    if permutation_metrics is not None:
        permutation_metrics.to_csv(output_dir / "model_permutation_metrics.csv", index=False)

    return {
        "metrics": metrics,
        "importance": importance,
        "iteration_metrics": iteration_metrics,
        "permutation_metrics": permutation_metrics,
    }


def select_target_nodes(
    nodes: pd.DataFrame,
    *,
    id_column: str,
    outcome_column: str | None,
    target_outcome: str | None,
    target_nodes: list[str],
) -> list[str]:
    if target_nodes:
        return [str(node) for node in target_nodes]
    if not outcome_column:
        return []
    if outcome_column not in nodes.columns:
        raise ValueError(f"Outcome column not found: {outcome_column}")

    outcome = nodes[outcome_column]
    if target_outcome is None:
        if pd.api.types.is_numeric_dtype(outcome):
            target_value = outcome.max()
        else:
            raise ValueError("Pass --target-outcome for non-numeric outcomes")
    else:
        target_value = target_outcome
        if pd.api.types.is_numeric_dtype(outcome):
            target_value = pd.to_numeric(pd.Series([target_outcome]), errors="raise").iloc[0]

    return nodes.loc[outcome.eq(target_value), id_column].astype(str).tolist()


def shortest_path_records(
    graph: nx.Graph,
    *,
    source_nodes: list[str],
    target_nodes: list[str],
    path_mode: str = "hops",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if path_mode == "hops":
        weight = None
    elif path_mode == "distance":
        weight = "weight"
    elif path_mode == "strength":
        weight = strength_as_cost
    else:
        raise ValueError(f"Unknown path mode: {path_mode}")

    sources = source_nodes or [str(node) for node in graph.nodes() if str(node) not in set(target_nodes)]
    targets = [str(node) for node in target_nodes]
    target_set = set(targets)

    pair_records: list[dict[str, object]] = []
    nearest_records: list[dict[str, object]] = []

    for source in sources:
        if source not in graph:
            continue
        best: dict[str, object] | None = None
        reachable_count = 0
        for target in targets:
            if target not in graph or source == target:
                continue
            try:
                length = nx.shortest_path_length(graph, source, target, weight=weight)
                path = nx.shortest_path(graph, source, target, weight=weight)
                strength = path_strength(graph, path)
                record = {
                    "source": source,
                    "target": target,
                    "path_mode": path_mode,
                    "path_cost": float(length),
                    "distance": float(length),
                    "hops": len(path) - 1,
                    "path_strength": strength,
                    "path": " -> ".join(map(str, path)),
                }
            except nx.NetworkXNoPath:
                record = {
                    "source": source,
                    "target": target,
                    "path_mode": path_mode,
                    "path_cost": np.nan,
                    "distance": np.nan,
                    "hops": np.nan,
                    "path_strength": np.nan,
                    "path": "",
                }
            pair_records.append(record)
            if record["path"]:
                reachable_count += 1
                if (
                    best is None
                    or float(record["distance"]) < float(best["distance"])
                    or (
                        float(record["distance"]) == float(best["distance"])
                        and str(record["target"]) < str(best["target"])
                    )
                ):
                    best = record

        nearest_records.append(
            {
                "source": source,
                "nearest_target": best["target"] if best else "",
                "path_mode": path_mode,
                "path_cost": best["path_cost"] if best else np.nan,
                "distance": best["distance"] if best else np.nan,
                "hops": best["hops"] if best else np.nan,
                "path_strength": best["path_strength"] if best else np.nan,
                "path": best["path"] if best else "",
                "reachable_target_count": reachable_count,
                "target_count": len(target_set),
            }
        )

    return pd.DataFrame(pair_records), pd.DataFrame(nearest_records)


def target_coverage_records(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per-target counterpart to the per-source nearest-target table.

    `nearest_targets.csv` answers "how far is each source from its nearest
    target?"; this answers the reverse question: "how well is each target
    covered by the sources?" — how many sources reach it and at what cost.
    """
    if pairs.empty:
        return pd.DataFrame(
            columns=[
                "target",
                "source_count",
                "reachable_source_count",
                "coverage",
                "min_distance",
                "mean_distance",
                "max_distance",
                "mean_hops",
            ]
        )

    records = []
    for target, group in pairs.groupby("target", sort=True):
        reachable = group[group["path"].astype(str).ne("") & group["path"].notna()]
        records.append(
            {
                "target": target,
                "source_count": int(len(group)),
                "reachable_source_count": int(len(reachable)),
                "coverage": float(len(reachable) / len(group)) if len(group) else 0.0,
                "min_distance": float(reachable["distance"].min()) if not reachable.empty else np.nan,
                "mean_distance": float(reachable["distance"].mean()) if not reachable.empty else np.nan,
                "max_distance": float(reachable["distance"].max()) if not reachable.empty else np.nan,
                "mean_hops": float(reachable["hops"].mean()) if not reachable.empty else np.nan,
            }
        )
    return pd.DataFrame(records)


def strength_as_cost(_u: str, _v: str, attrs: dict[str, object]) -> float:
    """Convert edge strength into a non-negative path cost.

    This is useful only when `weight` is a normalized 0..1 relationship strength.
    The path optimizer minimizes the sum of -log(strength), which is equivalent
    to maximizing the product of edge strengths.
    """
    try:
        strength = float(attrs.get("weight", 1.0))
    except (TypeError, ValueError):
        strength = 1.0
    strength = min(max(strength, 1e-12), 1.0)
    return -math.log(strength)


def path_strength(graph: nx.Graph, path: list[str]) -> float:
    if len(path) < 2:
        return 1.0
    product = 1.0
    for source, target in zip(path[:-1], path[1:]):
        attrs = graph.get_edge_data(source, target, default={})
        if graph.is_multigraph():
            attrs = next(iter(attrs.values()), {})
        try:
            product *= float(attrs.get("weight", 1.0))
        except (TypeError, ValueError):
            product *= 1.0
    return float(product)


def run(args: argparse.Namespace) -> dict[str, object]:
    nodes, edges = load_tables(args.nodes, args.edges)
    validation = validate_tables(
        nodes,
        edges,
        id_column=args.id_column,
        source_column=args.source_column,
        target_column=args.target_column,
        outcome_column=args.outcome_column if args.run_model else None,
    )
    graph = build_graph(
        nodes,
        edges,
        id_column=args.id_column,
        source_column=args.source_column,
        target_column=args.target_column,
        directed=args.directed,
        weight_column=args.weight_column,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {"validation": validation, "graph": graph_summary(graph)}
    (output_dir / "graph_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    features = generate_graph_features(graph, include_centrality=args.include_centrality)
    features.to_csv(output_dir / "node_features.csv", index=False)

    targets: list[str] = []
    nearest: pd.DataFrame | None = None
    if args.run_paths:
        targets = select_target_nodes(
            nodes,
            id_column=args.id_column,
            outcome_column=args.outcome_column,
            target_outcome=args.target_outcome,
            target_nodes=split_csv_list(args.target_nodes),
        )
        if not targets:
            raise ValueError("Shortest-path analysis needs --target-nodes or --outcome-column")
        pairs, nearest = shortest_path_records(
            graph,
            source_nodes=split_csv_list(args.source_nodes),
            target_nodes=targets,
            path_mode=args.path_mode,
        )
        coverage = target_coverage_records(pairs)
        pairs.to_csv(output_dir / "shortest_paths.csv", index=False)
        nearest.to_csv(output_dir / "nearest_targets.csv", index=False)
        coverage.to_csv(output_dir / "target_coverage.csv", index=False)
        summary["shortest_paths"] = {
            "source_count": int(nearest.shape[0]),
            "target_count": len(targets),
            "reachable_sources": int(nearest["path"].ne("").sum()) if not nearest.empty else 0,
            "fully_covered_targets": int(coverage["coverage"].eq(1.0).sum()) if not coverage.empty else 0,
            "unreached_targets": int(coverage["reachable_source_count"].eq(0).sum()) if not coverage.empty else 0,
        }

    model_result: dict[str, object] | None = None
    if args.run_model:
        if not args.outcome_column:
            raise ValueError("Outcome modeling needs --outcome-column")
        groups = None
        if getattr(args, "split_strategy", "random") == "community":
            groups = community_labels(graph)
        model_result = train_outcome_model(
            nodes,
            features,
            id_column=args.id_column,
            outcome_column=args.outcome_column,
            output_dir=output_dir,
            test_size=args.test_size,
            random_state=args.random_state,
            n_iterations=getattr(args, "n_iterations", 1),
            groups=groups,
            n_permutations=getattr(args, "permutation_test", 0),
        )
        summary["model"] = model_result["metrics"]

    cluster_result: dict[str, object] | None = None
    if getattr(args, "run_clusters", False):
        import epinet_cluster

        design = build_design_matrix(
            features,
            nodes,
            id_column=args.id_column,
            outcome_column=args.outcome_column,
        )
        y_cluster = None
        if args.outcome_column and args.outcome_column in nodes.columns:
            y_cluster = (
                nodes.assign(**{args.id_column: nodes[args.id_column].astype(str)})
                .set_index(args.id_column)[args.outcome_column]
            )
        cluster_result = epinet_cluster.run_clustering(
            design,
            output_dir,
            y=y_cluster,
            n_clusters=getattr(args, "n_clusters", 0),
            metric=getattr(args, "distance_metric", "euclidean"),
            random_state=args.random_state,
            labeled_only=getattr(args, "cluster_labeled_only", False),
        )
        summary["clusters"] = cluster_result["summary"]

    if getattr(args, "make_plots", False):
        import epinet_viz

        plots = epinet_viz.generate_run_plots(
            graph,
            output_dir,
            outcome_attribute=args.outcome_column if args.outcome_column in nodes.columns else None,
            target_nodes=targets,
            nearest=nearest,
            metrics=model_result["metrics"] if model_result else None,
            importance=model_result["importance"] if model_result else None,
            iteration_metrics=model_result["iteration_metrics"] if model_result else None,
            permutation_metrics=model_result["permutation_metrics"] if model_result else None,
            clustering=cluster_result,
            seed=args.random_state,
        )
        summary["plots"] = [str(path.relative_to(output_dir)) for path in plots]

    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="General node/edge network analysis toolkit")
    parser.add_argument("--nodes", default="synthetic_nodes.csv", help="Node CSV path")
    parser.add_argument("--edges", default="synthetic_edges.csv", help="Edge CSV path")
    parser.add_argument("--output-dir", default="epinet_outputs", help="Directory for generated CSV/JSON outputs")
    parser.add_argument("--id-column", default="ID")
    parser.add_argument("--source-column", default="SourceID")
    parser.add_argument("--target-column", default="TargetID")
    parser.add_argument("--outcome-column", default="Outcome", help="Outcome/target column for model and default paths")
    parser.add_argument("--target-outcome", default="1", help="Outcome value treated as target nodes for paths")
    parser.add_argument("--source-nodes", default="", help="Comma-separated source node IDs; default is all non-target nodes")
    parser.add_argument("--target-nodes", default="", help="Comma-separated target node IDs; overrides target outcome")
    parser.add_argument("--weight-column", default="", help="Optional edge column to copy into graph as numeric weight")
    parser.add_argument(
        "--path-mode",
        choices=["hops", "distance", "strength"],
        default="hops",
        help=(
            "Path objective: hops ignores weights; distance minimizes the copied weight column; "
            "strength maximizes normalized 0..1 edge strengths via -log transform"
        ),
    )
    parser.add_argument(
        "--use-weighted-paths",
        action="store_true",
        help="Deprecated alias for --path-mode distance",
    )
    parser.add_argument("--directed", action="store_true", help="Treat edges as directed SourceID -> TargetID")
    parser.add_argument("--include-centrality", action="store_true", help="Add betweenness, closeness, and PageRank features")
    parser.add_argument("--run-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-paths", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--run-clusters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Feature-space k-means clustering with centroid distances (attribute-space counterpart to shortest paths)",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=0,
        help="Number of feature-space clusters; 0 = number of outcome classes, else silhouette-selected",
    )
    parser.add_argument(
        "--distance-metric",
        choices=["euclidean", "mahalanobis"],
        default="euclidean",
        help="Centroid distance metric; mahalanobis accounts for feature correlation/scale",
    )
    parser.add_argument(
        "--cluster-labeled-only",
        action="store_true",
        help="Cluster only nodes with a non-blank outcome (skip feature-less scaffold nodes)",
    )
    parser.add_argument(
        "--make-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write PNG figures (network, degree distribution, model diagnostics) to <output-dir>/plots",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=10,
        help=(
            "Repeated train/test evaluations of the outcome model; >1 reports "
            "mean/std/min/max per metric to expose split-to-split variance"
        ),
    )
    parser.add_argument(
        "--split-strategy",
        choices=["random", "community"],
        default="random",
        help=(
            "random: stratified random train/test splits; community: detect graph "
            "communities and keep each one entirely in train or test, so scores "
            "estimate generalization to unseen regions of the network"
        ),
    )
    parser.add_argument(
        "--permutation-test",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Run N label-permutation null evaluations and report an empirical "
            "p-value per metric; if observed scores sit inside the null "
            "distribution, the features carry no detectable outcome signal"
        ),
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.weight_column = args.weight_column or None
    args.outcome_column = args.outcome_column or None
    if args.use_weighted_paths:
        args.path_mode = "distance"
    summary = run(args)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote outputs to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
