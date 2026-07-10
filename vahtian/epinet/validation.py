"""External validation for the EpiNet outcome model.

Internal evaluation (repeated splits, permutation null, calibration) guards
against leakage and chance *within one dataset*. It cannot tell you whether a
model transports to a genuinely independent cohort — different site, scanner,
era, or case mix. That is the question external validation answers, and it is the
bar a clinical-prediction claim has to clear (see TRIPOD+AI).

This module fits the model on a development cohort and evaluates it, untouched,
on an independent external cohort: discrimination (AUROC, AUPRC), classification
(balanced accuracy, MCC, F1), and — for binary outcomes — calibration (Brier,
slope/intercept). It reports the external metrics alongside the honest *internal*
metrics on the development data and the drift between them, because the expected
and informative result is that external performance is **lower**.

This is framing and measurement, not a guarantee: external validity is only as
good as how independent the external cohort really is.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV

from vahtian.epinet import common as epinet_common
from vahtian.epinet import toolkit as et


def _design(nodes, edges, *, id_column, outcome_column, source_column, target_column,
            include_centrality):
    graph = et.build_graph(
        nodes, edges, id_column=id_column,
        source_column=source_column, target_column=target_column,
    )
    features = et.generate_graph_features(graph, include_centrality=include_centrality)
    X = et.build_design_matrix(features, nodes, id_column=id_column, outcome_column=outcome_column)
    y = nodes.assign(**{id_column: nodes[id_column].astype(str)}).set_index(id_column)[outcome_column]
    return features, X, y.reindex(X.index)


def external_validation(
    dev_nodes: pd.DataFrame,
    dev_edges: pd.DataFrame,
    ext_nodes: pd.DataFrame,
    ext_edges: pd.DataFrame,
    *,
    id_column: str = "ID",
    outcome_column: str = "Outcome",
    source_column: str = "SourceID",
    target_column: str = "TargetID",
    include_centrality: bool = False,
    random_state: int = 42,
    model_name: str = "random_forest",
    output_dir: Path | None = None,
) -> dict[str, object]:
    """Fit on the development cohort; evaluate on the independent external cohort.

    Returns ``internal`` (honest within-development metrics), ``external``
    (metrics on the untouched external cohort), and ``drift`` (internal minus
    external, where larger = worse transport). Writes
    ``external_validation.json`` when ``output_dir`` is given.
    """
    dev_features, X_dev, y_dev = _design(
        dev_nodes, dev_edges, id_column=id_column, outcome_column=outcome_column,
        source_column=source_column, target_column=target_column,
        include_centrality=include_centrality,
    )
    _, X_ext, y_ext = _design(
        ext_nodes, ext_edges, id_column=id_column, outcome_column=outcome_column,
        source_column=source_column, target_column=target_column,
        include_centrality=include_centrality,
    )

    # Honest internal metrics on the development cohort (repeated splits etc.).
    target = Path(output_dir) / "internal" if output_dir else Path("/tmp/epinet_extval_internal")
    internal = et.train_outcome_model(
        dev_nodes, dev_features, id_column=id_column, outcome_column=outcome_column,
        output_dir=target, n_iterations=10, random_state=random_state, n_bootstrap=0,
        model_name=model_name,
    )["metrics"]

    # Fit on the FULL labeled development set, then apply to the external cohort.
    dev_labeled = epinet_common.labeled_mask(y_dev).to_numpy()
    X_dev_l, y_dev_l = X_dev[dev_labeled], y_dev[dev_labeled].astype(str)
    if y_dev_l.nunique() < 2:
        raise ValueError("development cohort needs at least two outcome classes")
    base_model, param_grid = et._build_estimator(model_name, random_state=random_state)
    cv = min(5, int(y_dev_l.value_counts().min()))
    if cv >= 2:
        search = GridSearchCV(
            base_model, param_grid, cv=cv, n_jobs=1, scoring="balanced_accuracy"
        )
        model = search.fit(X_dev_l, y_dev_l).best_estimator_
        best_params = search.best_params_
    else:
        model = base_model.fit(X_dev_l, y_dev_l)
        best_params = {"note": "insufficient class counts for cross-validation"}

    # Align external features to the development columns; score the labeled rows.
    X_ext = X_ext.reindex(columns=X_dev.columns, fill_value=0.0)
    ext_labeled = epinet_common.labeled_mask(y_ext).to_numpy()
    X_ext_l, y_ext_l = X_ext[ext_labeled], y_ext[ext_labeled].astype(str)
    if len(y_ext_l) == 0:
        raise ValueError("external cohort has no labeled rows to validate against")

    proba = model.predict_proba(X_ext_l)
    preds = model.predict(X_ext_l)
    classes = model.classes_

    external: dict[str, object] = {
        "n_external": int(len(y_ext_l)),
        "balanced_accuracy": float(balanced_accuracy_score(y_ext_l, preds)),
        "mcc": float(matthews_corrcoef(y_ext_l, preds)),
        "f1_weighted": float(precision_recall_fscore_support(
            y_ext_l, preds, average="weighted", zero_division=0)[2]),
    }
    external.update(et._probability_metrics(y_ext_l, proba, classes))
    if len(classes) == 2:
        cal = et.calibration_slope_intercept(y_ext_l, proba[:, 1], classes[1])
        external["calibration"] = {"brier": external.get("brier"), **cal}

    drift = {
        m: (None if internal.get(m) is None or external.get(m) is None
            else float(internal[m] - external[m]))
        for m in ("roc_auc", "average_precision", "balanced_accuracy", "f1_weighted")
    }

    result = {
        "internal": {m: internal.get(m) for m in
                     ("roc_auc", "average_precision", "balanced_accuracy", "f1_weighted", "accuracy")},
        "external": external,
        "model_name": model_name,
        "estimator": et._model_display_name(model_name),
        "best_params": best_params,
        "drift_internal_minus_external": drift,
        "note": (
            "External performance is expected to be lower than internal; large "
            "positive drift means the model does not transport. External validity "
            "is only as strong as the independence of the external cohort."
        ),
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "external_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
