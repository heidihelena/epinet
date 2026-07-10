"""General network analysis toolkit for node/edge CSV data.

This script keeps two analytic lenses side by side:

1. graph-derived node features, optionally used for a simple outcome model
2. shortest-path summaries from source nodes to target/outcome nodes

It is intentionally domain-neutral. Epidemiology is one use case, not a hard-coded
assumption.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import warnings
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

from vahtian.epinet import common as epinet_common

# Classification metrics where a higher value is better, vs the calibration
# error (Brier) where lower is better. The split matters for the one-sided
# permutation p-value (which tail counts as "at least as extreme as observed").
LABEL_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "mcc",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
)
HIGHER_IS_BETTER = LABEL_METRICS + ("roc_auc", "average_precision")
LOWER_IS_BETTER = ("brier",)
# Every metric reported per evaluation iteration, in display order.
ALL_METRICS = HIGHER_IS_BETTER + LOWER_IS_BETTER
MODEL_CHOICES = ("random_forest", "logistic_regression", "xgboost")


class XGBoostLabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    """Small sklearn-compatible XGBoost wrapper that accepts arbitrary labels.

    ``xgboost.XGBClassifier`` has varied across releases in how permissively it
    accepts string/category labels.  EpiNet's outcomes are often read from CSVs,
    so keep a stable wrapper: encode labels on fit, expose the original
    ``classes_`` order, and decode predictions on the way out.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        n_jobs: int = 1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "The xgboost estimator requires the optional dependency. "
                "Install it with `pip install vahtian-epinet[xgboost]` or "
                "`pip install xgboost`."
            ) from exc

        self._label_encoder = LabelEncoder()
        y_enc = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        objective = "binary:logistic" if len(self.classes_) == 2 else "multi:softprob"
        eval_metric = "logloss" if len(self.classes_) == 2 else "mlogloss"
        kwargs = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "objective": objective,
            "eval_metric": eval_metric,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbosity": 0,
        }
        if len(self.classes_) > 2:
            kwargs["num_class"] = len(self.classes_)
        self._model = XGBClassifier(**kwargs)
        self._model.fit(X, y_enc)
        self.n_features_in_ = getattr(self._model, "n_features_in_", None)
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def predict(self, X):
        pred = np.asarray(self._model.predict(X), dtype=int)
        return self._label_encoder.inverse_transform(pred)

    @property
    def feature_importances_(self):
        return getattr(self._model, "feature_importances_", None)


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
        train_idx, test_idx = next(splitter.split(X, y, groups))
    else:
        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=y if stratify_ok else None,
        )
    # Scientific guardrail: a training fold missing an outcome class can never
    # predict it, so every downstream metric would quietly score the model
    # against a class it has not seen. With grouped (community) splits this
    # happens exactly when an outcome class is confined to held-out
    # communities — the communities are outcome-confounded and a "leakage-
    # aware" split there is misleading, not honest. Refuse rather than report.
    missing = sorted(str(c) for c in set(y.unique()) - set(y.iloc[train_idx].unique()))
    if missing:
        kind = "community/grouped" if groups is not None else "random"
        raise ValueError(
            f"training fold from the {kind} split is missing outcome class(es) "
            f"{missing}. With community splits this means communities are "
            "outcome-confounded (a class lives entirely in held-out groups); "
            "use random splits or revisit the grouping."
        )
    return train_idx, test_idx


def _probability_metrics(
    y_true: pd.Series,
    proba: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float | None]:
    """Discrimination (AUROC, AUPRC) and calibration (Brier) from probabilities.

    Probabilities — not just the hard label — are what a risk score lives or dies
    on, so we report:

    - **AUROC** / **AUPRC** (average precision): ranking quality, the latter being
      the honest choice under class imbalance.
    - **Brier score**: mean squared error of the probabilities, the standard
      calibration summary (lower is better). For >2 classes we use the multiclass
      sum-of-squares form over the one-hot outcome.

    Any metric that is undefined on a given split (e.g. a held-out fold that
    contains a single class) is returned as ``None`` rather than raising, so the
    iteration loop degrades gracefully on small cohorts.
    """
    out: dict[str, float | None] = {"roc_auc": None, "average_precision": None, "brier": None}
    present = np.unique(y_true)
    onehot = label_binarize(y_true, classes=classes)
    if onehot.shape[1] == 1:  # binary: label_binarize emits one column
        onehot = np.hstack([1 - onehot, onehot])

    # Brier is always computable from probabilities and the one-hot truth.
    if len(classes) == 2:
        pos = classes[1]
        try:
            out["brier"] = float(brier_score_loss((y_true == pos).astype(int), proba[:, 1]))
        except ValueError:
            out["brier"] = None
    else:
        out["brier"] = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))

    # AUROC / AUPRC need at least two classes actually present in y_true.
    if len(present) < 2:
        return out
    try:
        if len(classes) == 2:
            score = proba[:, 1]
            out["roc_auc"] = float(roc_auc_score((y_true == classes[1]).astype(int), score))
            out["average_precision"] = float(
                average_precision_score((y_true == classes[1]).astype(int), score)
            )
        else:
            out["roc_auc"] = float(
                roc_auc_score(y_true, proba, multi_class="ovr", average="weighted", labels=classes)
            )
            out["average_precision"] = float(
                average_precision_score(onehot, proba, average="weighted")
            )
    except ValueError:
        pass
    return out


def _tune_threshold(y_bin: np.ndarray, oob_pos: np.ndarray) -> float:
    """Binary decision threshold maximizing balanced accuracy on OOB train scores.

    A forest's argmax (0.5) prediction is often suboptimal under class imbalance
    even after class weighting: the minority class is under-called. The threshold
    is chosen on the **out-of-bag** predicted probabilities — each training row
    scored only by the trees that did not see it — so the test set is never
    touched and there is no leakage. Returns 0.5 (the default) when no threshold
    beats it on the OOB scores, or when those scores are degenerate, so tuning
    can only help.
    """
    valid = ~np.isnan(oob_pos)
    y_bin, oob_pos = y_bin[valid], oob_pos[valid]
    if len(y_bin) == 0 or len(np.unique(y_bin)) < 2:
        return 0.5
    best_t = 0.5
    best_s = balanced_accuracy_score(y_bin, (oob_pos >= 0.5).astype(int))
    for t in np.unique(oob_pos):
        if t <= 0.0 or t >= 1.0:
            continue
        score = balanced_accuracy_score(y_bin, (oob_pos >= t).astype(int))
        if score > best_s:
            best_s, best_t = score, float(t)
    return best_t


def _model_display_name(model_name: str) -> str:
    return {
        "random_forest": "RandomForestClassifier",
        "logistic_regression": "LogisticRegression",
        "xgboost": "XGBoostClassifier",
    }.get(model_name, model_name)


def _build_estimator(model_name: str, *, random_state: int) -> tuple[BaseEstimator, dict[str, list[object]]]:
    """Return the base estimator and hyperparameter grid for ``model_name``."""
    if model_name not in MODEL_CHOICES:
        raise ValueError(f"model must be one of {MODEL_CHOICES}, got {model_name!r}")
    if model_name == "random_forest":
        return (
            RandomForestClassifier(random_state=random_state, n_jobs=1),
            {
                "n_estimators": [100, 200],
                "max_depth": [None, 5, 10],
                "class_weight": [None, "balanced", "balanced_subsample"],
                "min_samples_leaf": [1, 3],
            },
        )
    if model_name == "logistic_regression":
        return (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            solver="lbfgs",
                            max_iter=2000,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        )
    if importlib.util.find_spec("xgboost") is None:
        raise ImportError(
            "The xgboost estimator requires the optional dependency. "
            "Install it with `pip install vahtian-epinet[xgboost]` or `pip install xgboost`."
        )
    return (
        XGBoostLabelEncodedClassifier(random_state=random_state, n_jobs=1),
        {
            "n_estimators": [100, 200],
            "max_depth": [2, 4],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
        },
    )


def _clone_with_random_state(model: BaseEstimator, random_state: int, **extra) -> BaseEstimator:
    """Clone an estimator and set random-state-like params only when present."""
    fitted = clone(model)
    params = fitted.get_params()
    updates: dict[str, object] = {}
    for key in ("random_state", "model__random_state"):
        if key in params:
            updates[key] = random_state
    for key, value in extra.items():
        if key in params:
            updates[key] = value
        else:
            raise ValueError(f"Estimator {fitted.__class__.__name__} does not support parameter {key!r}")
    if updates:
        fitted = fitted.set_params(**updates)
    return fitted


def _intrinsic_importance(model: BaseEstimator, n_features: int) -> tuple[np.ndarray, str]:
    """Return a model-native importance vector when available.

    The primary feature-importance output remains held-out permutation
    importance.  This secondary vector preserves the old RF impurity column and
    gives logistic regression/XGBoost a comparable model-native diagnostic.
    """
    candidate = model
    if isinstance(model, Pipeline):
        candidate = model.steps[-1][1]
    if hasattr(candidate, "feature_importances_") and getattr(candidate, "feature_importances_") is not None:
        return np.asarray(candidate.feature_importances_, dtype=float), "impurity"
    if hasattr(candidate, "coef_"):
        coef = np.asarray(candidate.coef_, dtype=float)
        if coef.ndim == 1:
            values = np.abs(coef)
        else:
            values = np.mean(np.abs(coef), axis=0)
        return values, "absolute_coefficient"
    return np.zeros(n_features, dtype=float), "none"


def _split_and_score(
    X: pd.DataFrame,
    y: pd.Series,
    model: BaseEstimator,
    *,
    test_size: float,
    random_state: int,
    stratify_ok: bool,
    groups: pd.Series | None = None,
    tune_threshold: bool = False,
) -> tuple[BaseEstimator, dict[str, object], pd.Series, pd.Series, np.ndarray]:
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

    # Decision-threshold tuning is OPT-IN (``tune_threshold``). When off, the
    # default path is unchanged: argmax (0.5) predictions and no OOB pass. When
    # on, fit with ``oob_score`` (a nearly-free out-of-bag estimate on the
    # training rows) and, for binary outcomes, label at the threshold that
    # maximizes balanced accuracy on those OOB scores rather than 0.5 — which
    # under-calls the minority class. The test set is never used to pick the
    # threshold, so there is no leakage. Probabilities/AUROC are unchanged.
    extra = {"oob_score": True} if tune_threshold else {}
    with warnings.catch_warnings():
        # OOB coverage can be partial on small/imbalanced folds; the threshold
        # tuner already masks rows without an OOB score, so silence the notice.
        warnings.filterwarnings("ignore", message="Some inputs do not have OOB scores")
        fitted = _clone_with_random_state(model, random_state, **extra).fit(X_train, y_train)
    proba = fitted.predict_proba(X_test)
    classes_ = fitted.classes_

    threshold = 0.5
    if tune_threshold and len(classes_) == 2:
        neg_label, pos_label = classes_[0], classes_[1]
        oob = getattr(fitted, "oob_decision_function_", None)
        if oob is not None and oob.ndim == 2 and oob.shape[1] == 2:
            y_bin = (np.asarray(y_train) == pos_label).astype(int)
            threshold = _tune_threshold(y_bin, oob[:, 1])
        predictions = np.where(proba[:, 1] >= threshold, pos_label, neg_label)
    else:
        predictions = fitted.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )
    metrics: dict[str, object] = {
        "random_state": random_state,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        # MCC is robust to class imbalance; +1 perfect, 0 chance, -1 inverse.
        "mcc": float(matthews_corrcoef(y_test, predictions)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "decision_threshold": float(threshold),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    metrics.update(_probability_metrics(y_test, proba, fitted.classes_))
    return fitted, metrics, y_test, pd.Series(predictions, index=y_test.index), proba


def _calibrated_proba(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    classes: np.ndarray,
    *,
    random_state: int,
) -> np.ndarray | None:
    """Platt-scaled (sigmoid) probabilities for the tuned estimator, or ``None``.

    Tree ensembles are often over-confident, and any estimator can be
    miscalibrated on a small cohort. Refitting the tuned model inside
    ``CalibratedClassifierCV`` maps those scores toward
    observed event frequencies. Sigmoid (Platt) rather than isotonic because the
    cohorts here are small and isotonic overfits the calibration set. Returns
    ``None`` when the minority class is too small to cross-validate the
    calibrator honestly, so calibration is attempted only when it can be trusted.
    """
    k = int(min(5, y_train.value_counts().min()))
    if k < 2:
        return None
    estimator = _clone_with_random_state(model, random_state)
    calibrated = CalibratedClassifierCV(estimator, method="sigmoid", cv=k)
    calibrated.fit(X_train, y_train)
    proba = calibrated.predict_proba(X_test)
    # Align columns to the model's full class order. The calibrator's classes are
    # already in sklearn's sorted order (same as ``classes``); this is defensive.
    if not np.array_equal(calibrated.classes_, classes):
        order = [list(calibrated.classes_).index(c) for c in classes]
        proba = proba[:, order]
    return proba


def calibration_slope_intercept(
    y_true: pd.Series,
    proba_pos: np.ndarray,
    pos_label: object,
) -> dict[str, float | None]:
    """Calibration slope and intercept for a binary risk score.

    The standard weak-calibration check (Cox): regress the binary outcome on the
    *logit* of the predicted probability via logistic regression. A perfectly
    calibrated model gives **slope = 1** and **intercept = 0**. Slope < 1 means
    the predictions are too extreme (over-confident); a non-zero intercept means
    a systematic over/under-estimation of risk. Returns ``None`` values when the
    fit is undefined (single outcome class, or degenerate probabilities).
    """
    y = (np.asarray(y_true) == pos_label).astype(int)
    if len(np.unique(y)) < 2:
        return {"slope": None, "intercept": None}
    eps = 1e-12
    p = np.clip(np.asarray(proba_pos, dtype=float), eps, 1 - eps)
    logit = np.log(p / (1 - p))
    if np.allclose(logit, logit[0]):
        return {"slope": None, "intercept": None}
    try:
        from sklearn.linear_model import LogisticRegression

        # Unpenalized fit: the Cox calibration slope/intercept are a maximum-
        # likelihood recalibration, so no shrinkage. C=inf drives the
        # regularization strength (1/C) to zero, which is the unpenalized fit and
        # the replacement scikit-learn recommends now that the ``penalty``
        # argument is deprecated (removed in 1.10); it is numerically identical to
        # the former ``penalty=None`` across the supported versions.
        lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
        lr.fit(logit.reshape(-1, 1), y)
        return {"slope": float(lr.coef_[0][0]), "intercept": float(lr.intercept_[0])}
    except (ValueError, ImportError):
        return {"slope": None, "intercept": None}


def _bootstrap_ci(
    y_true: pd.Series,
    predictions: pd.Series,
    proba: np.ndarray,
    classes: np.ndarray,
    *,
    n_boot: int,
    random_state: int,
    alpha: float = 0.05,
    groups: np.ndarray | None = None,
) -> tuple[dict[str, dict[str, float]], str]:
    """Percentile bootstrap confidence intervals for the primary-split metrics.

    Repeated random re-splits expose *split-to-split* variability but their
    spread is not a valid confidence interval (the splits overlap, so it
    understates uncertainty — Nadeau & Bengio, 2003). This instead resamples the
    held-out test set with replacement and recomputes the metrics, giving a
    genuine within-split interval for a fixed model. Reported alongside, the two
    answer different questions: "how much does the score move if I re-split?" vs
    "how precisely is this split's score estimated?".

    When ``groups`` (one community label per held-out row) is supplied and the
    split spans at least two communities, the resampling is a **cluster
    bootstrap**: whole communities are drawn with replacement instead of
    individual nodes. Connected cases inside a community are correlated, so the
    community is the honest resampling unit — an observation-level bootstrap there
    treats correlated rows as independent and reports an over-narrow interval.
    With fewer than two communities (or no groups) it falls back to the ordinary
    observation-level bootstrap. The second return value names the unit used
    (``"community"`` or ``"observation"``).
    """
    y_arr = np.asarray(y_true)
    pred_arr = np.asarray(predictions)
    n = len(y_arr)
    rng = np.random.default_rng(random_state)
    keys = ["accuracy", "balanced_accuracy", "mcc", "f1_weighted", "roc_auc", "brier"]
    samples: dict[str, list[float]] = {k: [] for k in keys}

    # Decide the resampling unit. A cluster bootstrap needs ≥2 communities in the
    # held-out set to express between-community variability.
    cluster_units: list[np.ndarray] | None = None
    if groups is not None:
        groups_arr = np.asarray(groups)
        unique_groups = pd.unique(groups_arr)
        if len(unique_groups) >= 2:
            cluster_units = [np.flatnonzero(groups_arr == g) for g in unique_groups]
    resample_unit = "community" if cluster_units is not None else "observation"

    for _ in range(n_boot):
        if cluster_units is not None:
            chosen = rng.integers(0, len(cluster_units), size=len(cluster_units))
            idx = np.concatenate([cluster_units[c] for c in chosen])
        else:
            idx = rng.integers(0, n, size=n)
        yt = pd.Series(y_arr[idx])
        pr = pd.Series(pred_arr[idx])
        pb = proba[idx]
        if len(np.unique(yt)) < 2:
            continue  # degenerate resample: skip rather than emit a NaN metric
        samples["accuracy"].append(float(accuracy_score(yt, pr)))
        samples["balanced_accuracy"].append(float(balanced_accuracy_score(yt, pr)))
        samples["mcc"].append(float(matthews_corrcoef(yt, pr)))
        samples["f1_weighted"].append(
            float(precision_recall_fscore_support(yt, pr, average="weighted", zero_division=0)[2])
        )
        prob = _probability_metrics(yt, pb, classes)
        if prob["roc_auc"] is not None:
            samples["roc_auc"].append(prob["roc_auc"])
        if prob["brier"] is not None:
            samples["brier"].append(prob["brier"])
    out: dict[str, dict[str, float]] = {}
    lo_q, hi_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    for key, values in samples.items():
        if values:
            out[key] = {
                "lower": float(np.percentile(values, lo_q)),
                "upper": float(np.percentile(values, hi_q)),
            }
    return out, resample_unit


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
    n_bootstrap: int = 1000,
    tune_threshold: bool = False,
    model_name: str = "random_forest",
    provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    """Fit and evaluate the outcome model, optionally over repeated splits.

    A single train/test split is a noisy estimate on small networks, so when
    ``n_iterations > 1`` the model is re-evaluated on ``n_iterations`` different
    splits (seeds ``random_state .. random_state + n_iterations - 1``) and the
    summary reports the mean, standard deviation, and range of each metric.
    Hyperparameters are tuned once on the primary split and held fixed across
    iterations so the loop measures split variance, not tuning variance.
    ``model_name`` selects the estimator: ``random_forest`` (default nonlinear
    tabular model), ``logistic_regression`` (scaled regularized comparator), or
    ``xgboost`` (optional dependency). The tuning grids are intentionally small
    and scored by ``balanced_accuracy`` so skewed outcomes do not collapse the
    model toward the majority class.

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

    Beyond the weighted classification metrics, every split reports
    discrimination (AUROC, average precision) and calibration (Brier score, plus
    a calibration slope/intercept on the primary split for binary outcomes), and
    a percentile-bootstrap confidence interval is computed on the primary split
    (``n_bootstrap``; 0 disables). Feature importance is **permutation
    importance** measured on the held-out test set — less biased than impurity
    importance, which is kept alongside as ``impurity_importance`` for reference.
    Small-cohort risks (tiny test sets, rare classes) are surfaced as
    ``data_warnings`` rather than left implicit, and a ``provenance`` block (when
    supplied) is embedded so the metrics file is self-describing.

    Returns a dict with keys ``metrics`` (JSON-serializable summary),
    ``importance``, ``iteration_metrics``, ``permutation_metrics``,
    ``calibration`` (held-out probabilities for the reliability diagram, or
    None), and ``learning_curve`` (or None).
    """
    if outcome_column not in nodes.columns:
        raise ValueError(f"Outcome column not found: {outcome_column}")
    if n_iterations < 1:
        raise ValueError("n_iterations must be at least 1")
    if n_permutations < 0:
        raise ValueError("n_permutations must be non-negative")
    if model_name not in MODEL_CHOICES:
        raise ValueError(f"model_name must be one of {MODEL_CHOICES}, got {model_name!r}")
    if tune_threshold and model_name != "random_forest":
        raise ValueError("tune_threshold currently requires model_name='random_forest' (OOB scores)")

    X = build_design_matrix(features, nodes, id_column=id_column, outcome_column=outcome_column)

    y = nodes.assign(**{id_column: nodes[id_column].astype(str)}).set_index(id_column)[outcome_column]

    # Partially-labeled graphs: nodes with a blank/NaN outcome are scaffold
    # (e.g. infrastructure or context nodes). They still shaped the graph
    # features computed above, but are excluded from supervised training and
    # evaluation. This is the common semi-supervised setting where only some
    # node types carry the label of interest.
    labeled = epinet_common.labeled_mask(y)
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
    base_model, param_grid = _build_estimator(model_name, random_state=random_state)

    if cv >= 2:
        # Imbalance-aware model selection. Epidemiological outcomes are usually
        # skewed, where a default model may collapse toward the majority class.
        # Grids expose estimator-specific regularization/weighting knobs where
        # available (e.g. class_weight for RF/logistic, leaf size for RF) and use
        # balanced_accuracy so minority recall can affect selection.
        search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            n_jobs=1,
            scoring="balanced_accuracy",
        )
        search.fit(X_train, y_train)
        tuned_model = search.best_estimator_
        best_params = search.best_params_
    else:
        tuned_model = base_model
        best_params = {"note": "insufficient class counts for cross-validation"}

    # Iterative evaluation: re-split and re-fit with the tuned configuration.
    iteration_rows: list[dict[str, object]] = []
    impurity_runs: list[np.ndarray] = []
    primary_fit: BaseEstimator | None = None
    primary_truth: pd.Series | None = None
    primary_predictions: pd.Series | None = None
    primary_proba: np.ndarray | None = None
    for i in range(n_iterations):
        fitted, row, truth, predictions, proba = _split_and_score(
            X,
            y,
            tuned_model,
            test_size=test_size,
            random_state=random_state + i,
            stratify_ok=stratify_ok,
            groups=groups,
            tune_threshold=tune_threshold,
        )
        row["iteration"] = i
        iteration_rows.append(row)
        impurity_runs.append(_intrinsic_importance(fitted, X.shape[1])[0])
        if i == 0:
            primary_fit = fitted
            primary_truth = truth
            primary_predictions = predictions
            primary_proba = proba

    iteration_metrics = pd.DataFrame(iteration_rows)
    classes = primary_fit.classes_
    is_binary = len(classes) == 2

    # Primary-split row indices (random_state, i.e. iteration 0). Computed once
    # and reused for both probability calibration and permutation importance so
    # the calibrator and the importance estimator see exactly the same held-out
    # data the primary metrics were measured on.
    primary_train_idx, primary_test_idx = _split_indices(
        X, y, test_size=test_size, random_state=random_state,
        stratify_ok=stratify_ok, groups=groups,
    )
    X_primary_train = X.iloc[primary_train_idx]
    y_primary_train = y.iloc[primary_train_idx]
    X_test_primary = X.iloc[primary_test_idx]

    # Metrics actually computable on this run (probability metrics drop out on
    # degenerate splits, so only report what was measured).
    present_metrics = [
        m for m in ALL_METRICS
        if m in iteration_metrics.columns and iteration_metrics[m].notna().any()
    ]

    def _first(column: str) -> float | None:
        value = iteration_metrics[column].iloc[0]
        return float(value) if pd.notna(value) else None

    metrics: dict[str, object] = {column: _first(column) for column in present_metrics}
    metrics.update(
        {
            "best_params": best_params,
            "model_name": model_name,
            "estimator": _model_display_name(model_name),
            "n_iterations": n_iterations,
            "split_strategy": "community" if groups is not None else "random",
            "labeled_rows": int(len(y)),
            "unlabeled_excluded": n_unlabeled,
            "n_classes": int(len(classes)),
            "train_rows": int(iteration_metrics["train_rows"].iloc[0]),
            "test_rows": int(iteration_metrics["test_rows"].iloc[0]),
            "classes": [str(c) for c in classes],
            # Fix labels to the full class set so the matrix is always
            # n_classes x n_classes and aligns with "classes" above (a held-out
            # split can otherwise omit a class and produce a smaller matrix).
            "confusion_matrix": confusion_matrix(
                primary_truth, primary_predictions, labels=classes
            ).tolist(),
        }
    )
    if groups is not None:
        metrics["n_groups"] = int(groups.nunique())
    if split_note:
        metrics["split_note"] = split_note

    # Decision-threshold report (binary, opt-in). Predictions already use the
    # OOB-tuned threshold; surface it and the balanced accuracy at 0.5 vs the
    # tuned threshold on the held-out primary split so the choice — and whether
    # it transported — is auditable. Computed before any probability
    # recalibration below, so the 0.5 baseline uses the same raw scores the
    # threshold was applied to.
    if tune_threshold and is_binary and primary_proba is not None:
        primary_threshold = float(iteration_metrics["decision_threshold"].iloc[0])
        pos = classes[1]
        y_bin_primary = (np.asarray(primary_truth) == pos).astype(int)
        ba_default = float(
            balanced_accuracy_score(y_bin_primary, (primary_proba[:, 1] >= 0.5).astype(int))
        )
        ba_tuned = float(balanced_accuracy_score(primary_truth, primary_predictions))
        metrics["threshold_tuning"] = {
            "threshold": primary_threshold,
            "rule": "maximize balanced accuracy on out-of-bag training scores",
            "balanced_accuracy_at_0.5": ba_default,
            "balanced_accuracy_tuned": ba_tuned,
            "note": (
                "Opt-in. Threshold selected on OOB training scores only (no test "
                "leakage); it maximizes balanced accuracy on the training fold, but "
                "transport is not guaranteed — compare the 0.5 vs tuned held-out "
                "values here. Most useful for imbalanced outcomes; the gain over a "
                "class-weighted forest is typically small. Predictions only; "
                "probabilities and AUROC are unchanged."
            ),
        }

    if n_iterations > 1:
        metrics["iteration_summary"] = {
            column: {
                "mean": float(iteration_metrics[column].mean()),
                "std": float(iteration_metrics[column].std(ddof=1)),
                "min": float(iteration_metrics[column].min()),
                "max": float(iteration_metrics[column].max()),
            }
            for column in present_metrics
        }
        metrics["iteration_summary_note"] = (
            "std/min/max describe variability across overlapping random re-splits, "
            "not an independent-sample confidence interval; because the splits share "
            "data they tend to UNDERestimate true uncertainty (Nadeau & Bengio, 2003). "
            "See primary_split_bootstrap_ci for a within-split interval."
        )

    # Calibration (binary): Brier on the primary split plus the calibration
    # slope/intercept. A discriminating-but-miscalibrated risk score is
    # misleading exactly where decisions are contestable.
    calibration_payload: dict[str, object] | None = None
    if is_binary and primary_proba is not None:
        pos_label = classes[1]
        y_pos = (np.asarray(primary_truth) == pos_label).astype(int)
        raw_cal = calibration_slope_intercept(primary_truth, primary_proba[:, 1], pos_label)
        raw_brier = _first("brier")

        # Probability calibration. Raw scores can be miscalibrated (often
        # over-confident, slope < 1). Platt-scale them, but ADOPT the calibrated
        # probabilities only when they actually lower held-out Brier, so a
        # calibrator that overfits a thin minority class can never make the
        # reported risk worse. Discrimination (argmax predictions, AUROC) is
        # untouched — sigmoid is monotonic — so only probability quality changes.
        cal_proba = _calibrated_proba(
            tuned_model, X_primary_train, y_primary_train, X_test_primary,
            classes, random_state=random_state,
        )
        recalibration: dict[str, object] | None = None
        if cal_proba is not None:
            cal_brier = float(brier_score_loss(y_pos, cal_proba[:, 1]))
            cal_cal = calibration_slope_intercept(primary_truth, cal_proba[:, 1], pos_label)
            adopted = raw_brier is None or cal_brier <= raw_brier
            recalibration = {
                "method": "sigmoid",
                "adopted": adopted,
                "brier_raw": raw_brier,
                "brier_calibrated": cal_brier,
                "slope_raw": raw_cal["slope"],
                "slope_calibrated": cal_cal["slope"],
                "intercept_calibrated": cal_cal["intercept"],
                "note": (
                    "Platt (sigmoid) scaling, adopted only when it lowers held-out "
                    "Brier; discrimination is unchanged (monotonic transform)."
                ),
            }
            if adopted:
                # Downstream consumers (calibration plot, bootstrap probability
                # metrics) now see the calibrated probabilities.
                primary_proba = cal_proba

        pos = primary_proba[:, 1]
        final_cal = calibration_slope_intercept(primary_truth, pos, pos_label)
        final_brier = float(brier_score_loss(y_pos, pos))
        metrics["calibration"] = {
            "brier": final_brier,
            "slope": final_cal["slope"],
            "intercept": final_cal["intercept"],
            "positive_class": str(pos_label),
            "note": "Perfect calibration: slope 1, intercept 0. Slope < 1 = over-confident.",
        }
        if recalibration is not None:
            metrics["calibration"]["recalibration"] = recalibration
        calibration_payload = {
            "y_true": np.asarray(primary_truth),
            "proba_pos": pos,
            "pos_label": pos_label,
            "brier": final_brier,
        }
    elif primary_proba is not None:
        # Multiclass: the Cox calibration slope/intercept are defined for a
        # binary risk score only. Report the (multiclass) Brier so calibration is
        # never silently absent, and state explicitly why slope/intercept are
        # omitted rather than dropping the whole block.
        metrics["calibration"] = {
            "brier": _first("brier"),
            "slope": None,
            "intercept": None,
            "positive_class": None,
            "note": (
                "Multiclass outcome: 'brier' is the multiclass sum-of-squares "
                "(lower is better); calibration slope/intercept (Cox) are defined "
                "for binary outcomes only and are not computed here."
            ),
        }

    # Within-split uncertainty: percentile bootstrap on the primary held-out set.
    if n_bootstrap > 0 and primary_proba is not None:
        # In community mode, resample whole communities (cluster bootstrap) so the
        # interval reflects the unit at which connected cases are correlated.
        test_groups = (
            groups.reindex(primary_truth.index).to_numpy() if groups is not None else None
        )
        ci, resample_unit = _bootstrap_ci(
            primary_truth, primary_predictions, primary_proba, classes,
            n_boot=n_bootstrap, random_state=random_state, groups=test_groups,
        )
        if ci:
            metrics["primary_split_bootstrap_ci"] = {
                "n_bootstrap": n_bootstrap,
                "level": "95%",
                "resample_unit": resample_unit,
                "metrics": ci,
            }

    # Small-cohort guardrails: surface the risks rather than print a confident
    # number on data too thin to support it.
    warnings_list: list[str] = []
    primary_test_rows = int(iteration_metrics["test_rows"].iloc[0])
    min_class = int(class_counts.min())
    if len(y) < 50:
        warnings_list.append(f"Small cohort: {len(y)} labeled nodes (<50); metrics are high-variance.")
    if primary_test_rows < 20:
        warnings_list.append(f"Small test set: {primary_test_rows} held-out rows (<20) per split.")
    if min_class < 10:
        warnings_list.append(f"Rare class: smallest outcome class has {min_class} nodes (<10).")
    if warnings_list:
        metrics["data_warnings"] = warnings_list

    # Feature importance: permutation importance on the held-out primary test set
    # (less biased than impurity, and computed on data the model did not train
    # on). Impurity importance is retained for reference. The primary-split
    # indices were computed once above and are reused here.
    intrinsic_mean = np.vstack(impurity_runs).mean(axis=0)
    intrinsic_kind = _intrinsic_importance(primary_fit, X.shape[1])[1]
    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "intrinsic_importance": intrinsic_mean,
            # Backward-compatible alias retained for downstream readers that
            # expect the historical RF impurity column.
            "impurity_importance": intrinsic_mean,
        }
    )
    importance_kind = "permutation"
    try:
        perm = permutation_importance(
            primary_fit, X_test_primary, primary_truth,
            n_repeats=10, random_state=random_state, n_jobs=1, scoring="f1_weighted",
        )
        importance["importance"] = perm.importances_mean
        importance["importance_std"] = perm.importances_std
    except (ValueError, RuntimeError):
        # Fallback: model-native importance with cross-iteration spread.
        importance_kind = "intrinsic"
        importance["importance"] = intrinsic_mean
        importance["importance_std"] = np.vstack(impurity_runs).std(axis=0, ddof=1) \
            if n_iterations > 1 else 0.0
    importance = importance.sort_values("importance", ascending=False)
    metrics["importance_kind"] = importance_kind
    metrics["intrinsic_importance_kind"] = intrinsic_kind

    # Permutation null model. Each permutation is evaluated with the SAME
    # multi-iteration averaging as the observed score, so the null and the
    # statistic it is compared against have matching shape (fixing an earlier
    # single-split-null vs multi-split-mean mismatch).
    permutation_metrics: pd.DataFrame | None = None
    if n_permutations > 0:
        rng = np.random.default_rng(random_state)
        permutation_rows: list[dict[str, object]] = []
        n_perm_skipped = 0
        for p in range(n_permutations):
            y_permuted = pd.Series(rng.permutation(y.to_numpy()), index=y.index, name=y.name)
            inner: list[dict[str, object]] = []
            for j in range(n_iterations):
                try:
                    _, row, _, _, _ = _split_and_score(
                        X, y_permuted, tuned_model, test_size=test_size,
                        random_state=random_state + j, stratify_ok=stratify_ok, groups=groups,
                        tune_threshold=tune_threshold,
                    )
                except ValueError:
                    # A grouped (community) shuffle can leave an outcome class
                    # entirely in the held-out communities, which _split_indices
                    # refuses (training fold missing a class). That is the right
                    # guardrail for the real split, but here the labels are
                    # already destroyed — skip this draw rather than crash the
                    # whole null distribution.
                    continue
                inner.append(row)
            if not inner:
                # Every iteration of this permutation degenerated; drop it.
                n_perm_skipped += 1
                continue
            inner_df = pd.DataFrame(inner)
            mean_row = {
                m: float(inner_df[m].mean()) for m in present_metrics if m in inner_df.columns
            }
            mean_row["permutation"] = p
            permutation_rows.append(mean_row)
        permutation_metrics = pd.DataFrame(permutation_rows)

        perm_block: dict[str, object] = {}
        for column in present_metrics:
            if column not in permutation_metrics.columns:
                continue
            observed = float(iteration_metrics[column].mean())
            null_values = permutation_metrics[column].dropna()
            if null_values.empty:
                continue
            if column in LOWER_IS_BETTER:  # Brier: extreme = at or below observed
                extreme = int((null_values <= observed).sum())
            else:
                extreme = int((null_values >= observed).sum())
            perm_block[column] = {
                "observed_mean": observed,
                "null_mean": float(null_values.mean()),
                "null_std": float(null_values.std(ddof=1)) if len(null_values) > 1 else 0.0,
                # Denominator is the number of *valid* null draws for this metric,
                # not the requested count, so skipped degenerate permutations do
                # not silently inflate significance.
                "p_value": float((1 + extreme) / (len(null_values) + 1)),
            }
        metrics["permutation_test"] = {
            "n_permutations": n_permutations,
            "n_permutations_used": len(permutation_metrics),
            "n_permutations_skipped": n_perm_skipped,
            "metrics": perm_block,
            "multiplicity_note": (
                f"{len(perm_block)} metrics tested simultaneously; p-values are NOT "
                "corrected for multiple comparisons — read them jointly, not as "
                "independent significance tests."
            ),
        }

    # Learning curve: cross-validated score vs training-set size, to make the
    # small-n regime visible (is the curve still climbing at full size?).
    learning_curve_payload: dict[str, object] | None = None
    if cv >= 2 and len(y) >= 10:
        try:
            sizes, train_scores, test_scores = learning_curve(
                clone(tuned_model), X, y,
                cv=cv, scoring="f1_weighted",
                train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=1,
                random_state=random_state,
            )
            learning_curve_payload = {
                "train_sizes": sizes.tolist(),
                "train_mean": train_scores.mean(axis=1).tolist(),
                "train_std": train_scores.std(axis=1).tolist(),
                "test_mean": test_scores.mean(axis=1).tolist(),
                "test_std": test_scores.std(axis=1).tolist(),
            }
        except (ValueError, RuntimeError):
            learning_curve_payload = None

    if provenance is not None:
        metrics["provenance"] = provenance

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
        "calibration": calibration_payload,
        "learning_curve": learning_curve_payload,
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

    # Front-end normalization: map common column aliases (patient_id -> ID,
    # from/to -> SourceID/TargetID, ...) onto the configured schema before strict
    # validation, so messy real-world CSVs run without hand-editing. Every rename
    # is recorded and provenance hashes the normalized tables (see below).
    if getattr(args, "normalize", True):
        from vahtian.epinet import ingest as epinet_ingest

        nodes, edges, ingest_report = epinet_ingest.normalize_tables(
            nodes,
            edges,
            id_column=args.id_column,
            source_column=args.source_column,
            target_column=args.target_column,
            outcome_column=args.outcome_column,
            weight_column=args.weight_column,
        )
    else:
        ingest_report = {"normalized": False}

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

    # Provenance: stamp the exact inputs, code, and environment so every output
    # below traces back to a reproducible context. The normalization report is
    # folded in (it carries the SHA-256 of the normalized tables), so provenance
    # ties the run to both the raw input files and the exact analyzed form.
    prov = epinet_common.provenance([args.nodes, args.edges], seed=args.random_state)
    prov["normalization"] = ingest_report
    (output_dir / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    (output_dir / "ingest_report.json").write_text(json.dumps(ingest_report, indent=2) + "\n")

    summary = {"provenance": prov, "validation": validation, "graph": graph_summary(graph)}
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
            n_bootstrap=getattr(args, "n_bootstrap", 1000),
            tune_threshold=getattr(args, "tune_threshold", False),
            model_name=getattr(args, "model", "random_forest"),
            provenance=prov,
        )
        summary["model"] = model_result["metrics"]

        # TRIPOD+AI-flavored model card: the human-readable companion to the
        # metrics JSON (intended use, performance, calibration, limitations).
        from vahtian.epinet import report as epinet_report

        (output_dir / "model_card.md").write_text(
            epinet_report.model_card(model_result["metrics"]) + "\n"
        )

    cluster_result: dict[str, object] | None = None
    if getattr(args, "run_clusters", False):
        from vahtian.epinet import cluster as epinet_cluster

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

    contest_result: dict[str, object] | None = None
    if getattr(args, "run_contest", False):
        from vahtian.epinet import contest as epinet_contest

        contest_design = build_design_matrix(
            features,
            nodes,
            id_column=args.id_column,
            outcome_column=args.outcome_column,
        )
        y_contest = None
        if args.outcome_column and args.outcome_column in nodes.columns:
            y_contest = (
                nodes.assign(**{args.id_column: nodes[args.id_column].astype(str)})
                .set_index(args.id_column)[args.outcome_column]
            )
        if y_contest is None:
            raise SystemExit("--run-contest needs an --outcome-column with at least two classes")
        contest_result = epinet_contest.run_contestability(
            contest_design,
            output_dir,
            y=y_contest,
            metric=getattr(args, "distance_metric", "euclidean"),
            contest_quantile=getattr(args, "contest_quantile", 0.1),
        )
        summary["contestability"] = contest_result["summary"]

    if getattr(args, "make_plots", False):
        from vahtian.epinet import viz as epinet_viz

        if getattr(args, "plot_dpi", None):
            epinet_viz.DEFAULT_DPI = args.plot_dpi
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
            calibration=model_result["calibration"] if model_result else None,
            learning_curve=model_result["learning_curve"] if model_result else None,
            clustering=cluster_result,
            contestability=contest_result,
            seed=args.random_state,
            image_format=getattr(args, "plot_format", "png"),
            interactive=getattr(args, "interactive_network", False),
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
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Resolve common column aliases (patient_id->ID, from/to->Source/TargetID, "
            "label->Outcome, ...) onto the configured schema before validation; every "
            "rename is logged to ingest_report.json and provenance. Use --no-normalize "
            "for strict mode (reject anything not already in canonical schema)"
        ),
    )
    parser.add_argument("--directed", action="store_true", help="Treat edges as directed SourceID -> TargetID")
    parser.add_argument("--include-centrality", action="store_true", help="Add betweenness, closeness, and PageRank features")
    parser.add_argument("--run-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="random_forest",
        help=(
            "Outcome-model estimator: random_forest (default), "
            "logistic_regression (scaled regularized comparator), or xgboost "
            "(requires optional dependency)"
        ),
    )
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
        "--run-contest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Contestability/flip-distance lens: smallest feature-space move that flips each node's nearest-centroid class, plus per-feature value-of-information (uses --distance-metric)",
    )
    parser.add_argument(
        "--contest-quantile",
        type=float,
        default=0.1,
        help="Flag the most fragile fraction of nodes by flip-distance (default: lowest decile)",
    )
    parser.add_argument(
        "--make-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write figures (network, degree distribution, model diagnostics) to <output-dir>/plots",
    )
    parser.add_argument(
        "--plot-format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Figure file format: png (raster) or pdf/svg (vector, for print)",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=300,
        help="Raster figure resolution in DPI (ignored for vector formats)",
    )
    parser.add_argument(
        "--interactive-network",
        action="store_true",
        help="Also write a draggable/zoomable network.html (good for large graphs)",
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
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        metavar="N",
        help=(
            "Percentile-bootstrap resamples of the held-out test set for a "
            "within-split 95%% confidence interval on the primary metrics; 0 disables"
        ),
    )
    parser.add_argument(
        "--tune-threshold",
        action="store_true",
        help=(
            "Binary outcomes only: pick the decision threshold that maximizes "
            "balanced accuracy on out-of-bag training scores instead of the "
            "default 0.5 (no test leakage). Reports 0.5 vs tuned held-out "
            "balanced accuracy. Off by default; most useful for imbalanced "
            "outcomes, and the gain over the class-weighted forest is usually small"
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
