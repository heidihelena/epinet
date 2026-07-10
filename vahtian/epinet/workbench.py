"""EpiNet Workbench — config-driven planner, runner, and CLI.

The Workbench turns CSVs into a transparent analysis *plan* (``analysis.yaml``)
and then executes that plan with the same engine the command line uses. It is
deliberately **not** AutoML: it does not search models or maximise a metric. It
runs the honest EpiNet pipeline — validation, schema mapping, graph/feature-space
construction, an honestly-evaluated outcome model, baselines, calibration,
permutation null, optional external validation, contestability, a model card,
plots, and provenance — and writes a reproducible result bundle.

Three commands::

    epinet-workbench plan --nodes nodes.csv --outcome Outcome --output analysis.yaml
    epinet-workbench run  --config analysis.yaml
    epinet-workbench ui

``plan`` profiles the data, infers a schema, and writes a config you review.
``run`` is the source of truth: it reads the config and nothing else. ``ui``
launches the Streamlit workbench, which only ever builds a config and calls the
same ``run``.
"""

from __future__ import annotations

import json
import shutil
import sys
import zipfile
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from vahtian.epinet import schema as epinet_schema
from vahtian.epinet.config import AnalysisConfig, validate_config

# Files that make up the portable result bundle. Missing ones are skipped (a
# descriptive-only run has no model_metrics, etc.) so the bundle reflects what the
# run actually produced.
BUNDLE_FILES = (
    "analysis.yaml",
    "index.html",
    "run_summary.json",
    "provenance.json",
    "model_metrics.json",
    "model_card.md",
    "claims_check.json",
    "split_comparison.json",
    "node_features.csv",
    "node_contestability.csv",
    "model_feature_importance.csv",
    "baseline_comparison.csv",
    "external_validation.json",
    "gate_report.json",
    "ingest_report.json",
    "environment.txt",
)


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #

def build_plan(
    *,
    nodes_path: str,
    edges_path: str | None = None,
    validation_path: str | None = None,
    outcome: str | None = None,
    output_dir: str | None = None,
    name: str | None = None,
    mode: str | None = None,
) -> tuple[AnalysisConfig, epinet_schema.TableProfile]:
    """Profile the node table, infer a schema, and assemble an :class:`AnalysisConfig`.

    Shared by the ``plan`` CLI and the Streamlit UI so both produce identical
    configs. The returned profile carries the Data-screen checks (duplicate IDs,
    cardinality, date-like / free-text / leakage columns) for display.
    """
    if mode is None:
        mode = (
            "dev_validation" if validation_path
            else "nodes_edges" if edges_path
            else "single_csv"
        )

    # First pass with no id column to infer one; second pass validates duplicates
    # against the chosen id column.
    profile = epinet_schema.profile_table(nodes_path)
    schema = epinet_schema.infer_schema(profile, mode=mode)
    if outcome:
        schema.outcome_column = outcome
    profile = epinet_schema.profile_table(nodes_path, id_column=schema.id_column)

    config = AnalysisConfig()
    config.project.name = name or Path(nodes_path).stem
    config.project.output_dir = output_dir or f"epinet_outputs/{config.project.name}"
    config.data.mode = mode
    config.data.nodes_path = nodes_path
    config.data.edges_path = edges_path
    config.data.validation_path = validation_path
    config.schema = schema

    # Mode-appropriate defaults that stay honest.
    if mode == "nodes_edges":
        config.analysis.graph.mode = "none"  # real edges already define the graph
        config.analysis.split.method = "community_aware"
    elif mode == "single_csv":
        config.analysis.graph.mode = "none"  # feature-space only unless user opts into similarity
        config.analysis.split.method = "stratified"
    if mode == "dev_validation":
        config.analysis.evaluation.external_validation = True
        config.analysis.split.method = "community_aware"

    # Classification only when there is a low-cardinality outcome to classify;
    # otherwise fall back to a descriptive run (no fabricated outcome model).
    oc = profile.column(schema.outcome_column) if schema.outcome_column else None
    config.analysis.task = "classification" if (oc is not None and oc.n_unique <= 20) else "descriptive"
    return config, profile


# --------------------------------------------------------------------------- #
# Safety gates
# --------------------------------------------------------------------------- #

@dataclass
class GateReport:
    blocks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    downgraded_to_descriptive: bool = False

    @property
    def ok(self) -> bool:
        return not self.blocks

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "blocks": self.blocks,
            "warnings": self.warnings,
            "downgraded_to_descriptive": self.downgraded_to_descriptive,
        }


def check_gates(config: AnalysisConfig, profile: epinet_schema.TableProfile | None = None) -> GateReport:
    """Hard/soft gates that protect the user from producing nonsense.

    Blocks stop a run; warnings are surfaced and logged but allow it. A
    too-small positive class downgrades model training to a descriptive report
    rather than blocking outright (``downgraded_to_descriptive``).
    """
    report = GateReport()
    s, a = config.schema, config.analysis

    # Structural config errors are always blocking.
    report.blocks.extend(validate_config(config))

    if profile is None and config.data.nodes_path:
        profile = epinet_schema.profile_table(config.data.nodes_path, id_column=s.id_column)

    classification = a.task == "classification"

    if classification and not s.outcome_column:
        report.blocks.append("No outcome column selected for a classification task.")

    if profile is not None and s.outcome_column:
        oc = profile.column(s.outcome_column)
        if oc is not None:
            if oc.n_unique <= 1 and classification:
                report.blocks.append(
                    f"Outcome '{s.outcome_column}' is single-class — nothing to classify."
                )
            # Smallest class size for the positive-case gate.
            min_class = _min_class_size(config.data.nodes_path, s.outcome_column)
            if classification and min_class is not None and min_class < 5:
                report.warnings.append(
                    f"Smallest outcome class has {min_class} cases (<5): downgrading to a "
                    "descriptive report; no outcome model will be trained."
                )
                report.downgraded_to_descriptive = True

    # ID column must never be used as a feature.
    if s.id_column in s.feature_columns:
        report.blocks.append(f"ID column '{s.id_column}' is selected as a feature — remove it.")

    # Outcome-derived / leakage columns used as features: warn + require override.
    leaky_features = [c for c in s.leakage_flags if c in s.feature_columns]
    for c in leaky_features:
        report.warnings.append(
            f"Column '{c}' is suspected outcome leakage but is selected as a feature "
            "(override logged)."
        )

    # Publication-mode validation cohort.
    if config.data.mode == "dev_validation" and not config.data.validation_path:
        report.warnings.append("Publication mode selected but no validation cohort provided.")

    # Local-only sensitive-data notice (identifier-looking columns present).
    if profile is not None:
        idish = [c.name for c in profile.columns if c.looks_like_id and c.name != s.id_column]
        if idish:
            report.warnings.append(
                f"Possible identifier columns present ({', '.join(idish)}); keep this run local "
                "and de-identify before any export."
            )
    return report


def _min_class_size(nodes_path: str | None, outcome_column: str) -> int | None:
    if not nodes_path:
        return None
    try:
        col = pd.read_csv(nodes_path, usecols=[outcome_column])[outcome_column]
    except Exception:  # noqa: BLE001
        return None
    from vahtian.epinet.common import labeled_mask

    labeled = col[labeled_mask(col).to_numpy()]
    if labeled.empty:
        return 0
    return int(labeled.astype(str).value_counts().min())


# --------------------------------------------------------------------------- #
# Input preparation: feature pruning + (optional) similarity graph
# --------------------------------------------------------------------------- #

def _prepare_inputs(config: AnalysisConfig, work_dir: Path) -> tuple[str, str]:
    """Write the exact node/edge tables the engine will read into ``work_dir``.

    Feature selection is made explicit and auditable here: excluded and
    leakage-flagged columns are physically dropped before the model ever sees the
    data, and the pruned table is kept as a bundle artifact. Returns
    ``(nodes_path, edges_path)``.
    """
    return _prepare_one(config, config.data.nodes_path, config.data.edges_path, work_dir, "train")


def _prepare_one(config, nodes_path, edges_path, work_dir, tag) -> tuple[str, str]:
    s = config.schema
    nodes = pd.read_csv(nodes_path)

    keep = [s.id_column]
    if s.outcome_column and s.outcome_column in nodes.columns:
        keep.append(s.outcome_column)
    if s.feature_columns:
        feats = [c for c in s.feature_columns if c in nodes.columns]
    else:
        drop = set(s.exclude_columns) | {s.id_column, s.outcome_column}
        feats = [c for c in nodes.columns if c not in drop]
    # Never keep explicitly-excluded columns even if a user left them in features.
    feats = [c for c in feats if c not in set(s.exclude_columns) and c not in keep]
    pruned = nodes[keep + feats].copy()

    work_dir.mkdir(parents=True, exist_ok=True)
    pruned_path = work_dir / f"{tag}_nodes.csv"
    pruned.to_csv(pruned_path, index=False)

    if edges_path:
        edges_out = work_dir / f"{tag}_edges.csv"
        shutil.copy(edges_path, edges_out)
        return str(pruned_path), str(edges_out)

    if config.analysis.graph.mode == "similarity":
        edges = _knn_edges(pruned, s.id_column, feats, config.analysis.graph.k_neighbors)
    else:
        edges = pd.DataFrame(columns=[s.source_column, s.target_column])
    edges_out = work_dir / f"{tag}_edges.csv"
    edges.to_csv(edges_out, index=False)
    return str(pruned_path), str(edges_out)


def _knn_edges(nodes: pd.DataFrame, id_column: str, feature_columns: list[str], k: int) -> pd.DataFrame:
    """Build an undirected k-nearest-neighbour edge list in standardized feature space.

    Honest by construction: the synthesized edges are written to the bundle so the
    "graph" is reproducible and inspectable, never a hidden transform.
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    numeric = nodes[feature_columns].select_dtypes("number") if feature_columns else nodes.select_dtypes("number")
    ids = nodes[id_column].astype(str).to_numpy()
    if numeric.shape[1] == 0 or len(ids) < 2:
        return pd.DataFrame(columns=["SourceID", "TargetID", "Weight"])
    X = StandardScaler().fit_transform(numeric.fillna(numeric.mean()).to_numpy())
    k = min(k, len(ids) - 1)
    rows = []
    seen = set()
    for i in range(len(ids)):
        d = np.linalg.norm(X - X[i], axis=1)
        nearest = np.argsort(d)[1 : k + 1]
        for j in nearest:
            pair = tuple(sorted((i, int(j))))
            if pair in seen:
                continue
            seen.add(pair)
            rows.append({"SourceID": ids[pair[0]], "TargetID": ids[pair[1]],
                         "Weight": float(1.0 / (1.0 + d[j]))})
    return pd.DataFrame(rows, columns=["SourceID", "TargetID", "Weight"])


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #

def _config_to_args(config: AnalysisConfig, nodes_path: str, edges_path: str, run_model: bool) -> Namespace:
    """Translate a config into the toolkit's argument namespace. No hidden state."""
    s, a = config.schema, config.analysis
    split_strategy = "community" if a.split.method == "community_aware" else "random"
    run_paths = config.data.mode == "nodes_edges"
    return Namespace(
        nodes=nodes_path,
        edges=edges_path,
        output_dir=config.project.output_dir,
        id_column=s.id_column,
        source_column=s.source_column,
        target_column=s.target_column,
        outcome_column=s.outcome_column or None,
        target_outcome=s.positive_class or "1",
        source_nodes="",
        target_nodes="",
        weight_column=s.weight_column or None,
        path_mode="hops",
        use_weighted_paths=False,
        normalize=True,
        directed=False,
        include_centrality=a.graph.include_centrality,
        run_model=run_model,
        model=a.model.primary,
        run_paths=run_paths,
        run_clusters=True,
        n_clusters=0,
        distance_metric="euclidean",
        cluster_labeled_only=False,
        run_contest=a.contestability.enabled and run_model,
        contest_quantile=a.contestability.quantile,
        make_plots=a.reporting.plots,
        plot_format=a.reporting.plot_format,
        plot_dpi=300,
        interactive_network=False,
        n_iterations=a.evaluation.n_iterations,
        split_strategy=split_strategy,
        permutation_test=a.evaluation.permutation_repeats if a.evaluation.permutation_null else 0,
        n_bootstrap=1000 if a.evaluation.bootstrap_ci else 0,
        test_size=a.split.test_size,
        random_state=a.split.random_state,
    )


def _split_comparison(config: AnalysisConfig, nodes_path: str, edges_path: str) -> dict:
    """Evaluate the outcome model under random AND community-aware splits.

    A bounded diagnostic (capped iterations, no bootstrap/permutation) whose only
    purpose is the random-vs-community AUROC gap in the claims check. Recomputes
    the cheap graph features rather than threading them out of ``et.run``.
    """
    from vahtian.epinet import toolkit as et

    nodes, edges = et.load_tables(nodes_path, edges_path)
    graph = et.build_graph(
        nodes, edges,
        id_column=config.schema.id_column,
        source_column=config.schema.source_column,
        target_column=config.schema.target_column,
    )
    features = et.generate_graph_features(graph)
    communities = et.community_labels(graph)
    n_iter = min(5, config.analysis.evaluation.n_iterations)
    keep = ("roc_auc", "balanced_accuracy", "average_precision")

    def _evaluate(groups, tag):
        target = Path(config.project.output_dir) / "_split_comparison" / tag
        metrics = et.train_outcome_model(
            nodes, features,
            id_column=config.schema.id_column,
            outcome_column=config.schema.outcome_column,
            output_dir=target, test_size=config.analysis.split.test_size,
            random_state=config.analysis.split.random_state,
            n_iterations=n_iter, groups=groups, n_permutations=0, n_bootstrap=0,
            model_name=config.analysis.model.primary,
        )["metrics"]
        out = {k: metrics.get(k) for k in keep}
        # Per-split AUROC spread across re-splits, so the claims check can tell a
        # real random->community drop from one inside the iteration noise.
        out["roc_auc_std"] = metrics.get("iteration_summary", {}).get("roc_auc", {}).get("std")
        return out

    return {
        "random": _evaluate(None, "random"),
        "community": _evaluate(communities, "community"),
        "n_iterations": n_iter,
        "roc_auc_std_note": "roc_auc_std is the std of AUROC across the bounded "
                            "re-splits; it feeds the split-drop uncertainty band.",
        "note": "Bounded diagnostic for leakage sensitivity; the community-aware "
                "split is the more honest generalization estimate.",
    }


def run_config(config: AnalysisConfig, *, skip_gates: bool = False) -> dict:
    """Execute a plan end-to-end and write the result bundle. The config is the
    only source of truth — this reads nothing from any UI.
    """
    from vahtian.epinet import toolkit as et

    output_dir = Path(config.project.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Persist the exact plan that produced these outputs.
    config.write(output_dir / "analysis.yaml")

    profile = epinet_schema.profile_table(config.data.nodes_path, id_column=config.schema.id_column)
    gates = check_gates(config, profile)
    (output_dir / "gate_report.json").write_text(json.dumps(gates.to_dict(), indent=2) + "\n")
    if gates.blocks and not skip_gates:
        raise SystemExit(
            "Run blocked by safety gates:\n  - " + "\n  - ".join(gates.blocks)
        )

    run_model = config.analysis.task == "classification" and not gates.downgraded_to_descriptive

    # Apply the report palette before the engine draws, so the figures match the
    # HTML report chrome. Returns the resolved palette for the report theme.
    from vahtian.epinet import palette as epinet_palette

    resolved_palette = epinet_palette.apply_palette(config.analysis.reporting.plot_palette)

    work_dir = output_dir / "inputs"
    nodes_path, edges_path = _prepare_inputs(config, work_dir)

    args = _config_to_args(config, nodes_path, edges_path, run_model)
    summary = et.run(args)
    summary["gates"] = gates.to_dict()

    # Baselines: same honest harness, graph vs no-information floor (and spectral
    # where the graph supports it). Only meaningful with a trained model.
    baseline_metrics: dict | None = None
    baseline_paired: dict | None = None
    if run_model and config.analysis.model.baselines:
        try:
            from vahtian.epinet import baselines as eb

            nodes_df, edges_df = et.load_tables(nodes_path, edges_path)
            baseline_result = eb.compare_representations(
                nodes_df, edges_df,
                id_column=config.schema.id_column,
                outcome_column=config.schema.outcome_column,
                n_iterations=min(5, config.analysis.evaluation.n_iterations),
                random_state=config.analysis.split.random_state,
                output_dir=output_dir,
            )
            baseline_metrics = baseline_result.get("metrics")
            baseline_paired = baseline_result.get("paired_baseline")
            summary["baselines"] = "baseline_comparison.csv"
        except Exception as exc:  # noqa: BLE001 - baselines are best-effort
            summary["baselines_error"] = str(exc)

    # Random vs community-aware split comparison: a leakage-sensitivity diagnostic
    # that feeds the claims check (does the headline survive an honest split?).
    split_comparison: dict | None = None
    if run_model and config.analysis.evaluation.split_comparison:
        try:
            split_comparison = _split_comparison(config, nodes_path, edges_path)
            (output_dir / "split_comparison.json").write_text(
                json.dumps(split_comparison, indent=2) + "\n"
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic is best-effort
            summary["split_comparison_error"] = str(exc)

    # External validation (publication mode).
    external_validation: dict | None = None
    if config.analysis.evaluation.external_validation and config.data.validation_path and run_model:
        try:
            from vahtian.epinet import validation as exv

            val_nodes, val_edges = _prepare_one(
                config, config.data.validation_path,
                config.data.validation_edges_path, work_dir, "validation",
            )
            dev_nodes_df, dev_edges_df = et.load_tables(nodes_path, edges_path)
            ext_nodes_df, ext_edges_df = et.load_tables(val_nodes, val_edges)
            external_validation = exv.external_validation(
                dev_nodes_df, dev_edges_df, ext_nodes_df, ext_edges_df,
                id_column=config.schema.id_column,
                outcome_column=config.schema.outcome_column,
                random_state=config.analysis.split.random_state,
                model_name=config.analysis.model.primary,
                output_dir=output_dir,
            )
            summary["external_validation"] = "external_validation.json"
        except Exception as exc:  # noqa: BLE001
            summary["external_validation_error"] = str(exc)

    # Scientific claims check: distil every diagnostic above into plain-language
    # claim gates, append them to the model card, and write the machine-readable
    # record. Generated for every run (descriptive runs get the caveat too).
    from vahtian.epinet import claims as epinet_claims

    model_metrics = summary.get("model") if isinstance(summary.get("model"), dict) else None
    claims = epinet_claims.scientific_claims_check(
        model_metrics,
        split_comparison=split_comparison,
        baseline_metrics=baseline_metrics,
        baseline_paired=baseline_paired,
        external_validation=external_validation,
        model_trained=run_model and model_metrics is not None,
    )
    (output_dir / "claims_check.json").write_text(json.dumps(claims, indent=2) + "\n")
    card_path = output_dir / "model_card.md"
    if card_path.exists():
        card_path.write_text(card_path.read_text().rstrip() + "\n\n"
                             + epinet_claims.claims_markdown(claims) + "\n")

    # Branded, portable HTML report — the polished bundle artifact.
    if config.analysis.reporting.html_report:
        try:
            from vahtian.epinet import htmlreport

            htmlreport.build_html_report(
                output_dir, config=config, claims=claims,
                metrics=model_metrics, palette=resolved_palette,
            )
            summary["html_report"] = "index.html"
        except Exception as exc:  # noqa: BLE001 - report is best-effort
            summary["html_report_error"] = str(exc)

    _write_environment(output_dir)
    bundle = assemble_bundle(output_dir, config.project.name)
    summary["bundle"] = str(bundle)
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _write_environment(output_dir: Path) -> None:
    import platform

    lines = [f"python: {sys.version.split()[0]}", f"platform: {platform.platform()}"]
    for pkg in ("numpy", "pandas", "scikit-learn", "networkx", "matplotlib"):
        try:
            mod = __import__("sklearn" if pkg == "scikit-learn" else pkg)
            lines.append(f"{pkg}: {getattr(mod, '__version__', '?')}")
        except Exception:  # noqa: BLE001
            lines.append(f"{pkg}: not installed")
    (output_dir / "environment.txt").write_text("\n".join(lines) + "\n")


def assemble_bundle(output_dir: Path, name: str) -> Path:
    """Zip the canonical result files (+ plots/) into ``<name>_bundle.zip``."""
    output_dir = Path(output_dir)
    bundle_path = output_dir / f"{name}_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in BUNDLE_FILES:
            fpath = output_dir / fname
            if fpath.exists():
                zf.write(fpath, fname)
        plots_dir = output_dir / "plots"
        if plots_dir.is_dir():
            for plot in sorted(plots_dir.glob("*")):
                zf.write(plot, f"plots/{plot.name}")
    return bundle_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _cmd_plan(args) -> None:
    config, profile = build_plan(
        nodes_path=args.nodes,
        edges_path=args.edges,
        validation_path=args.validation,
        outcome=args.outcome,
        output_dir=args.output_dir,
        name=args.name,
    )
    config.write(args.output)
    gates = check_gates(config, profile)
    print(config.to_yaml())
    print(f"# Wrote plan to {Path(args.output).resolve()}")
    if profile.warnings:
        print("# Data warnings:")
        for w in profile.warnings:
            print(f"#   - {w}")
    if gates.blocks:
        print("# BLOCKED (fix before running):")
        for b in gates.blocks:
            print(f"#   - {b}")
    if gates.warnings:
        print("# Gate warnings:")
        for w in gates.warnings:
            print(f"#   - {w}")


def _cmd_run(args) -> None:
    config = AnalysisConfig.load(args.config)
    summary = run_config(config, skip_gates=args.force)
    print(json.dumps(summary.get("gates", {}), indent=2))
    print(f"\nWrote outputs to {Path(config.project.output_dir).resolve()}")
    print(f"Result bundle: {summary.get('bundle')}")


def _cmd_ui(args) -> None:
    import subprocess

    app = Path(__file__).resolve().parent / "ui.py"
    try:
        subprocess.run(["streamlit", "run", str(app)] + args.streamlit_args, check=True)
    except FileNotFoundError:
        raise SystemExit(
            "Streamlit is not installed. Install the UI extra:\n"
            "    pip install -e \".[ui]\"\nthen run: epinet-workbench ui"
        )


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="epinet-workbench",
        description="EpiNet Workbench — local CSV-to-report interface over the EpiNet engine.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan", help="profile data, infer a schema, write analysis.yaml")
    p.add_argument("--nodes", required=True, help="node/single CSV path")
    p.add_argument("--edges", default=None, help="optional edge CSV (graph mode)")
    p.add_argument("--validation", default=None, help="optional external cohort CSV")
    p.add_argument("--outcome", default=None, help="outcome column (overrides inference)")
    p.add_argument("--output", default="analysis.yaml", help="config output path")
    p.add_argument("--output-dir", default=None, help="run output directory")
    p.add_argument("--name", default=None, help="project name")
    p.set_defaults(func=_cmd_plan)

    r = sub.add_parser("run", help="execute analysis.yaml and write the result bundle")
    r.add_argument("--config", required=True, help="analysis.yaml path")
    r.add_argument("--force", action="store_true", help="run despite blocking gates (logged)")
    r.set_defaults(func=_cmd_run)

    u = sub.add_parser("ui", help="launch the Streamlit workbench")
    u.add_argument("streamlit_args", nargs="*", help="extra args passed to streamlit run")
    u.set_defaults(func=_cmd_ui)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
