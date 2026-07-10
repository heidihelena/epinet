"""EpiNet Workbench — Streamlit UI.

A *thin* local interface over the same engine as the CLI. It never becomes the
source of truth: every screen edits an :class:`epinet_config.AnalysisConfig`, the
user reviews the resulting ``analysis.yaml``, and the run goes through the exact
same :func:`epinet_workbench.run_config` that ``epinet-workbench run`` uses. There
is no hidden UI state that changes the analysis.

Launch with::

    epinet-workbench ui          # or: streamlit run epinet/ui.py

Five screens: Data → Schema → Plan → Run → Report. Research and education only —
not clinical decision support.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - exercised only when extra missing
    raise SystemExit(
        "Streamlit is required for the UI. Install it with:\n"
        "    pip install -e \".[ui]\"\nthen run: epinet-workbench ui"
    )

from vahtian.epinet import workbench as wb
from vahtian.epinet.config import AnalysisConfig

SCREENS = ["1. Data", "2. Schema", "3. Plan", "4. Run", "5. Report"]


def _save_upload(uploaded, suffix: str = ".csv") -> str:
    """Persist a Streamlit upload to a temp file so the engine can read a path."""
    tmp_dir = Path(tempfile.gettempdir()) / "epinet_workbench_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dest = tmp_dir / uploaded.name
    dest.write_bytes(uploaded.getbuffer())
    return str(dest)


def _state() -> dict:
    if "wb" not in st.session_state:
        st.session_state.wb = {"config": None, "profile": None, "summary": None}
    return st.session_state.wb


def screen_data(state: dict) -> None:
    st.header("Data")
    st.caption("Upload CSVs and choose an output folder. Files stay local.")
    nodes = st.file_uploader("nodes.csv (or a single CSV)", type=["csv"], key="nodes")
    edges = st.file_uploader("edges.csv (optional — graph mode)", type=["csv"], key="edges")
    validation = st.file_uploader(
        "validation.csv (optional — external cohort)", type=["csv"], key="validation"
    )
    validation_edges = st.file_uploader(
        "validation_edges.csv (optional)", type=["csv"], key="validation_edges"
    )
    output_dir = st.text_input("Output folder", value="epinet_outputs/workbench_run")

    if st.button("Validate & continue", type="primary", disabled=nodes is None):
        nodes_path = _save_upload(nodes)
        edges_path = _save_upload(edges) if edges else None
        validation_path = _save_upload(validation) if validation else None
        validation_edges_path = _save_upload(validation_edges) if validation_edges else None
        config, profile = wb.build_plan(
            nodes_path=nodes_path,
            edges_path=edges_path,
            validation_path=validation_path,
            validation_edges_path=validation_edges_path,
            output_dir=output_dir,
        )
        state["config"] = config
        state["profile"] = profile
        st.session_state.screen = "2. Schema"
        st.rerun()

    profile = state.get("profile")
    if profile is not None:
        st.subheader("Validation checks")
        st.write(f"Rows: {profile.n_rows} · Columns: {profile.n_cols} · "
                 f"Duplicate IDs: {profile.duplicate_id_count}")
        for e in profile.errors:
            st.error(e)
        for w in profile.warnings:
            st.warning(w)


def screen_schema(state: dict) -> None:
    st.header("Schema")
    config: AnalysisConfig | None = state.get("config")
    profile = state.get("profile")
    if config is None or profile is None:
        st.info("Upload data on the Data screen first.")
        return

    names = [c.name for c in profile.columns]
    s = config.schema
    st.caption("Confirm the inferred mappings. Suspected-leakage columns are flagged.")

    s.id_column = st.selectbox("ID column", names, index=_idx(names, s.id_column))
    outcome_opts = ["(none)"] + names
    s.outcome_column = st.selectbox(
        "Outcome column", outcome_opts, index=_idx(outcome_opts, s.outcome_column or "(none)")
    )
    s.outcome_column = None if s.outcome_column == "(none)" else s.outcome_column

    config.analysis.task = st.radio("Task", ["classification", "descriptive"],
                                    index=0 if config.analysis.task == "classification" else 1)

    feature_default = [c for c in s.feature_columns if c in names]
    s.feature_columns = st.multiselect(
        "Feature columns", [n for n in names if n not in (s.id_column, s.outcome_column)],
        default=feature_default,
    )
    s.exclude_columns = st.multiselect(
        "Excluded columns", names, default=[c for c in s.exclude_columns if c in names]
    )

    site_opts = ["(none)"] + names
    s.site_column = st.selectbox("Site / cluster column (optional)", site_opts,
                                 index=_idx(site_opts, s.site_column or "(none)"))
    s.site_column = None if s.site_column == "(none)" else s.site_column
    s.time_column = st.selectbox("Time column (optional)", site_opts,
                                 index=_idx(site_opts, s.time_column or "(none)"))
    s.time_column = None if s.time_column == "(none)" else s.time_column

    if s.leakage_flags:
        st.warning("Suspected outcome leakage (default-excluded): " + ", ".join(s.leakage_flags))
    leaky_in = [c for c in s.leakage_flags if c in s.feature_columns]
    if leaky_in:
        st.error("Leakage columns kept as features (override will be logged): " + ", ".join(leaky_in))

    if st.button("Confirm schema → Plan", type="primary"):
        st.session_state.screen = "3. Plan"
        st.rerun()


def screen_plan(state: dict) -> None:
    st.header("Analysis plan")
    config: AnalysisConfig | None = state.get("config")
    if config is None:
        st.info("Complete the Schema screen first.")
        return

    a = config.analysis
    st.caption("This is the full analysis.yaml. No hidden defaults — review before running.")

    col1, col2 = st.columns(2)
    with col1:
        a.split.method = st.selectbox("Split method",
                                      ["stratified", "random", "community_aware"],
                                      index=_idx(["stratified", "random", "community_aware"], a.split.method))
        a.split.test_size = st.slider("Test size", 0.1, 0.5, a.split.test_size, 0.05)
        a.graph.mode = st.selectbox("Graph mode", ["none", "similarity"],
                                    index=_idx(["none", "similarity"], a.graph.mode))
        if a.graph.mode == "similarity":
            a.graph.k_neighbors = st.number_input("k neighbours", 1, 50, a.graph.k_neighbors)
    with col2:
        a.evaluation.calibration = st.checkbox("Calibration", a.evaluation.calibration)
        a.evaluation.bootstrap_ci = st.checkbox("Bootstrap CI", a.evaluation.bootstrap_ci)
        a.evaluation.permutation_null = st.checkbox("Permutation null", a.evaluation.permutation_null)
        a.evaluation.permutation_repeats = st.number_input(
            "Permutation repeats", 0, 5000, a.evaluation.permutation_repeats, 100
        )
        a.evaluation.external_validation = st.checkbox(
            "External validation", a.evaluation.external_validation,
            disabled=config.data.validation_path is None,
        )
        a.contestability.enabled = st.checkbox("Contestability", a.contestability.enabled)
        a.evaluation.split_comparison = st.checkbox(
            "Random vs community split comparison", a.evaluation.split_comparison
        )

    with st.expander("Report & branding"):
        r = a.reporting
        r.html_report = st.checkbox("Branded HTML report (index.html)", r.html_report)
        r.plot_palette = st.selectbox(
            "Plot palette", ["wong", "vahtian"],
            index=_idx(["wong", "vahtian"], r.plot_palette),
            help="wong = colourblind-safe Okabe–Ito; vahtian = Sentinel brand palette",
        )
        r.brand_name = st.text_input("Brand name", r.brand_name)
        r.report_title = st.text_input("Report title", r.report_title)
        c1, c2 = st.columns(2)
        with c1:
            r.primary_color = st.color_picker("Primary colour", r.primary_color)
        with c2:
            r.accent_color = st.color_picker("Accent colour", r.accent_color)
        st.caption("Theme changes colour/title only — caveats, claims check, and "
                   "provenance are always rendered.")

    gates = wb.check_gates(config, state.get("profile"))
    for b in gates.blocks:
        st.error(f"BLOCKED: {b}")
    for w in gates.warnings:
        st.warning(w)

    st.code(config.to_yaml(), language="yaml")
    st.download_button("Download analysis.yaml", config.to_yaml(), file_name="analysis.yaml")

    if st.button("Run full EpiNet scope →", type="primary", disabled=not gates.ok):
        st.session_state.screen = "4. Run"
        st.rerun()


def screen_run(state: dict) -> None:
    st.header("Run")
    config: AnalysisConfig | None = state.get("config")
    if config is None:
        st.info("Build a plan first.")
        return

    stages = [
        "Validate schema", "Normalize inputs", "Build graph / feature space",
        "Compute graph features", "Train primary model", "Run baselines",
        "Run calibration", "Run permutation null", "Run external validation",
        "Run contestability", "Generate plots", "Generate model card", "Write provenance",
    ]
    st.write("Pipeline stages:")
    st.write("  ·  ".join(stages))

    if st.button("Execute", type="primary"):
        with st.spinner("Running the full EpiNet scope…"):
            try:
                summary = wb.run_config(config)
                state["summary"] = summary
                st.success("Run complete.")
                st.session_state.screen = "5. Report"
                st.rerun()
            except SystemExit as exc:
                st.error(str(exc))


def screen_report(state: dict) -> None:
    st.header("Report")
    config: AnalysisConfig | None = state.get("config")
    summary = state.get("summary")
    if config is None or summary is None:
        st.info("Run the analysis first.")
        return

    import json

    out = Path(config.project.output_dir)

    # Scientific claims check — the plain-language headline and gates, first.
    claims_path = out / "claims_check.json"
    if claims_path.exists():
        cc = json.loads(claims_path.read_text())
        st.subheader("Scientific claims check")
        st.info(cc.get("headline", ""))
        for label, key in [("Permutation null", "permutation"),
                           ("Split sensitivity", "split_comparison"),
                           ("Baseline floor", "baselines"),
                           ("External validation", "external_validation"),
                           ("Graph semantics", "graph_semantics")]:
            gate = cc.get(key, {})
            st.markdown(f"- **{label} — {gate.get('status', '—')}**: {gate.get('statement', '')}")
        st.warning(cc.get("clinical_caveat", ""))

    # Branded HTML report download.
    report_path = out / "index.html"
    if report_path.exists():
        st.download_button("Download HTML report (index.html)",
                           report_path.read_bytes(), file_name="index.html",
                           mime="text/html")

    metrics_path = out / "model_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        st.subheader("Metrics")
        st.json({k: metrics.get(k) for k in
                 ("roc_auc", "average_precision", "balanced_accuracy", "mcc", "brier")})

    plots_dir = out / "plots"
    if plots_dir.is_dir():
        st.subheader("Plots")
        for plot in sorted(plots_dir.glob("*.png")):
            st.image(str(plot), caption=plot.name)

    card = out / "model_card.md"
    if card.exists():
        st.subheader("Model card")
        st.markdown(card.read_text())

    bundle = summary.get("bundle")
    if bundle and Path(bundle).exists():
        st.download_button(
            "Download result bundle (.zip)", Path(bundle).read_bytes(),
            file_name=Path(bundle).name, mime="application/zip",
        )


def _idx(options: list, value) -> int:
    try:
        return options.index(value)
    except (ValueError, AttributeError):
        return 0


def main() -> None:
    st.set_page_config(page_title="EpiNet Workbench", layout="wide")
    st.title("EpiNet Workbench")
    st.caption("Local CSV-to-report interface over the EpiNet engine. "
               "Research and education only — not clinical decision support.")

    if "screen" not in st.session_state:
        st.session_state.screen = SCREENS[0]
    state = _state()

    st.session_state.screen = st.sidebar.radio("Workflow", SCREENS,
                                               index=_idx(SCREENS, st.session_state.screen))
    screen = st.session_state.screen
    if screen == "1. Data":
        screen_data(state)
    elif screen == "2. Schema":
        screen_schema(state)
    elif screen == "3. Plan":
        screen_plan(state)
    elif screen == "4. Run":
        screen_run(state)
    elif screen == "5. Report":
        screen_report(state)


if __name__ == "__main__":
    main()
