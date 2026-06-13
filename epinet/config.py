"""EpiNet Workbench configuration schema — the single source of truth for a run.

The Workbench (CLI ``epinet-workbench`` and the Streamlit UI) is a *thin wrapper*
over this file. Every interactive choice ends up in an ``analysis.yaml`` that is
reviewed before execution and runs identically with or without the UI::

    epinet-workbench plan --nodes nodes.csv --outcome Outcome --output analysis.yaml
    epinet-workbench run --config analysis.yaml

The config — not hidden UI state — drives the analysis, so a run can always be
reproduced from the YAML alone. This module owns the schema (dataclasses with
honest defaults), YAML load/dump that round-trips losslessly, and structural
validation of the config itself (distinct from *data* validation, which lives in
``epinet_schema``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

# Three input modes, matching the Workbench UI. ``single_csv`` is feature-space
# analysis for an ordinary table; ``nodes_edges`` is the full graph pipeline;
# ``dev_validation`` adds a held-out external cohort and runs external validation
# by default (the mode for publishable work).
INPUT_MODES = ("single_csv", "nodes_edges", "dev_validation")
TASKS = ("classification", "descriptive")
SPLIT_METHODS = ("random", "stratified", "community_aware")
GRAPH_MODES = ("none", "similarity")
SIMILARITY_METRICS = ("euclidean", "mixed")


@dataclass
class Project:
    name: str = "epinet_run"
    output_dir: str = "epinet_outputs/epinet_run"


@dataclass
class Data:
    mode: str = "single_csv"
    nodes_path: str | None = None        # the CSV (single_csv) or node table
    edges_path: str | None = None        # optional edge table (nodes_edges)
    validation_path: str | None = None   # external cohort nodes (dev_validation)
    validation_edges_path: str | None = None


@dataclass
class Schema:
    id_column: str = "ID"
    outcome_column: str | None = "Outcome"
    positive_class: str | None = None
    feature_columns: list[str] = field(default_factory=list)  # empty = all numeric
    exclude_columns: list[str] = field(default_factory=list)
    site_column: str | None = None
    time_column: str | None = None
    # Graph wiring (nodes_edges mode). Defaults match the toolkit's canonical schema.
    source_column: str = "SourceID"
    target_column: str = "TargetID"
    weight_column: str | None = None
    # Columns the planner flagged as likely outcome leakage. Kept so the run record
    # shows what was warned about; only *excluded* columns are actually dropped.
    leakage_flags: list[str] = field(default_factory=list)


@dataclass
class Split:
    method: str = "stratified"
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class Graph:
    mode: str = "none"               # none = feature-space only; similarity = k-NN graph
    similarity_metric: str = "euclidean"
    k_neighbors: int = 10
    include_centrality: bool = False


@dataclass
class Model:
    primary: str = "random_forest"
    baselines: list[str] = field(
        default_factory=lambda: ["no_information", "graph_features"]
    )


@dataclass
class Evaluation:
    calibration: bool = True
    bootstrap_ci: bool = True
    permutation_null: bool = True
    permutation_repeats: int = 1000
    external_validation: bool = False
    n_iterations: int = 10
    # Run the model under both random and community-aware splits and report the
    # gap in the claims check (a leakage-sensitivity diagnostic).
    split_comparison: bool = True


@dataclass
class Contestability:
    enabled: bool = True
    method: str = "nearest_centroid"
    quantile: float = 0.1


@dataclass
class Reporting:
    model_card: bool = True
    provenance: bool = True
    plots: bool = True
    plot_format: str = "png"
    # Branded HTML report written into the bundle (index.html). The theme below
    # may recolour/retitle it, but the caveats, claims check, and provenance are
    # always rendered and cannot be themed away.
    html_report: bool = True
    brand_name: str = "EpiNet"
    report_title: str = "EpiNet Analysis Report"
    logo_path: str | None = None
    primary_color: str = "#5E4F99"
    accent_color: str = "#8273C0"
    plot_palette: str = "wong"


@dataclass
class Analysis:
    task: str = "classification"
    split: Split = field(default_factory=Split)
    graph: Graph = field(default_factory=Graph)
    model: Model = field(default_factory=Model)
    evaluation: Evaluation = field(default_factory=Evaluation)
    contestability: Contestability = field(default_factory=Contestability)
    reporting: Reporting = field(default_factory=Reporting)


@dataclass
class AnalysisConfig:
    project: Project = field(default_factory=Project)
    data: Data = field(default_factory=Data)
    schema: Schema = field(default_factory=Schema)
    analysis: Analysis = field(default_factory=Analysis)

    # --- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml())
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisConfig":
        return _build(cls, data or {})

    @classmethod
    def from_yaml(cls, text: str) -> "AnalysisConfig":
        return cls.from_dict(yaml.safe_load(text) or {})

    @classmethod
    def load(cls, path: str | Path) -> "AnalysisConfig":
        return cls.from_yaml(Path(path).read_text())


def _build(klass: type, data: dict[str, Any]) -> Any:
    """Recursively build a (possibly nested) dataclass from a plain dict.

    Unknown keys are ignored rather than raising, so a config written by a newer
    Workbench still loads on an older engine (forward-compatible); missing keys
    fall back to the dataclass defaults.
    """
    kwargs: dict[str, Any] = {}
    type_hints = {f.name: f.type for f in fields(klass)}
    nested = {f.name: f.type for f in fields(klass) if is_dataclass(_resolve(f.type))}
    for name in type_hints:
        if name not in data:
            continue
        value = data[name]
        target = _resolve(type_hints[name])
        if name in nested and isinstance(value, dict):
            kwargs[name] = _build(target, value)
        else:
            kwargs[name] = value
    return klass(**kwargs)


def _resolve(annotation: Any) -> Any:
    """Best-effort resolve a field annotation to a class (handles string hints)."""
    if isinstance(annotation, str):
        return globals().get(annotation, annotation)
    return annotation


def validate_config(config: AnalysisConfig) -> list[str]:
    """Structural validation of the config itself; returns a list of error strings.

    This checks the *plan* is internally coherent (valid enums, required paths for
    the chosen mode, a sane split). It deliberately does not touch the data — that
    is ``epinet_schema``'s job and runs against the actual CSVs.
    """
    errors: list[str] = []
    d, s, a = config.data, config.schema, config.analysis

    if d.mode not in INPUT_MODES:
        errors.append(f"data.mode must be one of {INPUT_MODES}, got {d.mode!r}")
    if not d.nodes_path:
        errors.append("data.nodes_path is required")
    if d.mode == "dev_validation" and not d.validation_path:
        errors.append("dev_validation mode requires data.validation_path")
    if d.mode == "nodes_edges" and not d.edges_path:
        errors.append("nodes_edges mode requires data.edges_path")

    if a.task not in TASKS:
        errors.append(f"analysis.task must be one of {TASKS}, got {a.task!r}")
    if a.task == "classification" and not s.outcome_column:
        errors.append("classification task requires schema.outcome_column")

    if a.split.method not in SPLIT_METHODS:
        errors.append(f"analysis.split.method must be one of {SPLIT_METHODS}")
    if not 0.0 < a.split.test_size < 1.0:
        errors.append("analysis.split.test_size must be between 0 and 1")

    if a.graph.mode not in GRAPH_MODES:
        errors.append(f"analysis.graph.mode must be one of {GRAPH_MODES}")
    if a.graph.mode == "similarity" and a.graph.k_neighbors < 1:
        errors.append("analysis.graph.k_neighbors must be >= 1")
    if a.graph.similarity_metric not in SIMILARITY_METRICS:
        errors.append(f"analysis.graph.similarity_metric must be one of {SIMILARITY_METRICS}")

    if not 0.0 < a.contestability.quantile < 1.0:
        errors.append("analysis.contestability.quantile must be between 0 and 1")

    return errors
