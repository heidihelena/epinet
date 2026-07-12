# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Heidi Andersén

"""Model card generation for the EpiNet outcome model.

A model card is the reporting counterpart to the metrics JSON: a human-readable
summary of *what the model is, how it was evaluated, and where it must not be
trusted*. The structure follows the spirit of the TRIPOD+AI reporting guideline
for prediction models — intended use, data and provenance, model, performance
(discrimination AND calibration), validation, and limitations — so a reader can
judge a result without reverse-engineering the code.

This is deliberately a *card*, not a certificate: EpiNet produces research
demonstrators, and the card says so in the first line.
"""

from __future__ import annotations


def _fmt(value: object, digits: int = 3) -> str:
    """Format a metric for the card; ``—`` when it was not measurable."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _ci_text(ci: dict | None, key: str) -> str:
    if not ci:
        return ""
    entry = ci.get("metrics", {}).get(key)
    if not entry:
        return ""
    return f" (95% CI {entry['lower']:.3f}–{entry['upper']:.3f})"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def model_card(metrics: dict, *, title: str = "EpiNet outcome model") -> str:
    """Render a TRIPOD+AI-flavored markdown model card from a metrics dict.

    Reads the keys written by ``train_outcome_model`` (classification,
    discrimination, calibration, bootstrap CI, permutation test, provenance,
    data warnings) and degrades gracefully when any are absent.
    """
    ci = metrics.get("primary_split_bootstrap_ci")
    calibration = metrics.get("calibration", {})
    provenance = metrics.get("provenance", {})
    out: list[str] = [f"# Model card — {title}", ""]

    # 1. Intended use — stated first, deliberately.
    out += [
        "## Intended use",
        "",
        "**Research and education demonstrator. Not clinical or public-health "
        "decision support.** Any model produced here must be validated on "
        "independent, outcome-linked data before it carries clinical meaning. "
        "The figures below describe this run on this dataset, not external "
        "performance.",
        "",
    ]

    # 2. Data.
    out += [
        "## Data",
        "",
        _table(
            ["property", "value"],
            [
                ["Labeled nodes", _fmt(metrics.get("labeled_rows"))],
                ["Unlabeled scaffold (excluded)", _fmt(metrics.get("unlabeled_excluded"))],
                ["Outcome classes", _fmt(metrics.get("n_classes"))],
                ["Class labels", ", ".join(metrics.get("classes", [])) or "—"],
                ["Train / test rows (primary split)",
                 f"{_fmt(metrics.get('train_rows'))} / {_fmt(metrics.get('test_rows'))}"],
            ],
        ),
        "",
    ]
    if metrics.get("data_warnings"):
        out += ["**Data warnings:**", ""]
        out += [f"- {w}" for w in metrics["data_warnings"]]
        out += [""]

    # 3. Model + evaluation design.
    out += [
        "## Model & evaluation",
        "",
        _table(
            ["property", "value"],
            [
                ["Estimator", _fmt(metrics.get("estimator") or metrics.get("model_name") or "RandomForestClassifier")],
                ["Tuned hyperparameters", _fmt(metrics.get("best_params"))],
                ["Split strategy", _fmt(metrics.get("split_strategy"))],
                ["Evaluation iterations", _fmt(metrics.get("n_iterations"))],
                ["Importance method", _fmt(metrics.get("importance_kind"))],
            ],
        ),
        "",
    ]
    if metrics.get("split_note"):
        out += [f"_{metrics['split_note']}_", ""]

    # 4. Performance — discrimination, classification, calibration.
    multiclass = int(metrics.get("n_classes", 2)) > 2
    auroc_label = "AUROC (macro OvR)" if multiclass else "AUROC"
    ap_label = "Average precision (macro)" if multiclass else "Average precision (AUPRC)"
    out += ["## Performance", "", "### Discrimination & classification", ""]
    perf_rows = []
    for key, label in [
        ("roc_auc", auroc_label),
        ("average_precision", ap_label),
        ("balanced_accuracy", "Balanced accuracy"),
        ("mcc", "Matthews corr. coef."),
        ("f1_weighted", "F1 (weighted)"),
        ("accuracy", "Accuracy"),
    ]:
        if key in metrics:
            perf_rows.append([label, _fmt(metrics.get(key)) + _ci_text(ci, key)])
    out += [_table(["metric", "value"], perf_rows), ""]

    out += ["### Calibration", ""]
    if calibration and calibration.get("slope") is not None:
        out += [
            _table(
                ["metric", "value"],
                [
                    ["Brier score (lower better)", _fmt(calibration.get("brier")) + _ci_text(ci, "brier")],
                    ["Calibration slope (ideal 1)", _fmt(calibration.get("slope"))],
                    ["Calibration intercept (ideal 0)", _fmt(calibration.get("intercept"))],
                    ["Positive class", _fmt(calibration.get("positive_class"))],
                ],
            ),
            "",
            f"_{calibration.get('note', '')}_",
            "",
        ]
    elif calibration:
        # Multiclass: Brier is defined, slope/intercept are not. Report Brier and
        # say so explicitly rather than printing empty slope/intercept rows.
        out += [
            _table(
                ["metric", "value"],
                [["Brier score (lower better)", _fmt(calibration.get("brier")) + _ci_text(ci, "brier")]],
            ),
            "",
            f"_{calibration.get('note', '')}_",
            "",
        ]
    else:
        out += [
            "Calibration slope/intercept are reported for binary outcomes only; "
            "the Brier score above (if present) still summarizes probability error.",
            "",
        ]

    # 5. Validation — permutation null.
    out += ["## Validation", ""]
    perm = metrics.get("permutation_test")
    if perm:
        rows = []
        for key, entry in perm.get("metrics", {}).items():
            rows.append([key, _fmt(entry.get("observed_mean")), _fmt(entry.get("null_mean")),
                         _fmt(entry.get("p_value"))])
        out += [
            f"Label-permutation null model ({perm['n_permutations']} permutations):",
            "",
            _table(["metric", "observed", "null mean", "p-value"], rows),
            "",
            f"_{perm.get('multiplicity_note', '')}_",
            "",
        ]
    else:
        out += [
            "No permutation null was run. Run with `--permutation-test N` to test "
            "the observed scores against chance.",
            "",
        ]
    if ci:
        out += [
            f"Within-split uncertainty: {ci['level']} percentile bootstrap "
            f"({ci['n_bootstrap']} resamples of the held-out test set).",
            "",
        ]

    # 6. Limitations.
    out += [
        "## Limitations",
        "",
        "- Performance is in-sample to this dataset; external validity is unknown.",
        "- Repeated-split spread understates true uncertainty (overlapping splits); "
        "the bootstrap CI is a within-split estimate, not external.",
        "- A good discrimination score does not imply good calibration — check both.",
        "- On small cohorts, all metrics are high-variance; see any data warnings above.",
        "",
    ]

    # 7. Provenance.
    if provenance:
        git = provenance.get("git", {})
        pkgs = provenance.get("packages", {})
        out += [
            "## Provenance",
            "",
            _table(
                ["property", "value"],
                [
                    ["EpiNet version", _fmt(provenance.get("epinet_version"))],
                    ["Git commit", _fmt(git.get("commit"))],
                    ["Working tree", "dirty" if git.get("dirty") else "clean"
                     if git.get("available") else "—"],
                    ["Python", _fmt(provenance.get("python_version"))],
                    ["scikit-learn", _fmt(pkgs.get("scikit-learn"))],
                    ["numpy / pandas", f"{_fmt(pkgs.get('numpy'))} / {_fmt(pkgs.get('pandas'))}"],
                    ["Random seed", _fmt(provenance.get("random_seed"))],
                    ["Generated (UTC)", _fmt(provenance.get("created_utc"))],
                ],
            ),
            "",
        ]
        if provenance.get("input_sha256"):
            out += ["Input SHA-256:", ""]
            for path, digest in provenance["input_sha256"].items():
                out += [f"- `{path}`: `{digest}`"]
            out += [""]

    return "\n".join(out)
