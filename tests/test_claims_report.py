"""Tests for the scientific claims check, palettes, and HTML report layer."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from epinet import claims, htmlreport, palette
from epinet import workbench as wb
from epinet.config import AnalysisConfig


class ClaimsGateTests(unittest.TestCase):
    def test_permutation_gate_signal_above_null(self):
        metrics = {"permutation_test": {"n_permutations": 100, "metrics": {
            "roc_auc": {"observed_mean": 0.82, "null_mean": 0.50, "p_value": 0.001}}}}
        gate = claims.permutation_gate(metrics)
        self.assertEqual(gate["status"], "signal above null")
        self.assertIn("above null", gate["statement"])

    def test_permutation_gate_signal_not_detected(self):
        metrics = {"permutation_test": {"n_permutations": 100, "metrics": {
            "roc_auc": {"observed_mean": 0.52, "null_mean": 0.50, "p_value": 0.40}}}}
        gate = claims.permutation_gate(metrics)
        self.assertEqual(gate["status"], "signal not detected")

    def test_permutation_gate_not_run(self):
        self.assertEqual(claims.permutation_gate({})["status"], "not run")

    def test_split_gate_flags_leakage_sensitivity(self):
        sc = {"random": {"roc_auc": 0.90}, "community": {"roc_auc": 0.70}}
        gate = claims.split_gate(sc)
        self.assertEqual(gate["status"], "leakage-sensitive")
        self.assertAlmostEqual(gate["drop"], 0.20, places=6)

    def test_split_gate_stable(self):
        sc = {"random": {"roc_auc": 0.80}, "community": {"roc_auc": 0.78}}
        self.assertEqual(claims.split_gate(sc)["status"], "stable")

    def test_baseline_gate_beats_and_at_floor(self):
        self.assertEqual(
            claims.baseline_gate({"no_information": {"roc_auc": 0.50}}, {"roc_auc": 0.80})["status"],
            "beats floor")
        self.assertEqual(
            claims.baseline_gate({"no_information": {"roc_auc": 0.50}}, {"roc_auc": 0.505})["status"],
            "at floor")

    def _paired(self, lo, hi, mean, thr=0.02):
        return {"n_pairs": 8, "mean_margin": mean, "margin_ci_lower": lo,
                "margin_ci_upper": hi, "threshold": thr, "correction": "NB",
                "model_representation": "graph_features"}

    def test_baseline_gate_paired_three_verdicts(self):
        # CI clears the line -> beats; below -> at floor; straddles -> inconclusive.
        self.assertEqual(claims.baseline_gate(None, {}, paired=self._paired(0.05, 0.12, 0.085))["status"],
                         "beats floor")
        self.assertEqual(claims.baseline_gate(None, {}, paired=self._paired(-0.03, 0.01, -0.01))["status"],
                         "at floor")
        straddle = claims.baseline_gate(None, {}, paired=self._paired(-0.01, 0.06, 0.025))
        self.assertEqual(straddle["status"], "not resolvable")
        self.assertFalse(straddle["resolvable_at_this_n"])

    def test_paired_baseline_overrides_scalar(self):
        # Even with a scalar floor that would read "beats", a straddling paired CI wins.
        out = claims.baseline_gate({"no_information": {"roc_auc": 0.50}}, {"roc_auc": 0.80},
                                   paired=self._paired(-0.01, 0.06, 0.025))
        self.assertEqual(out["status"], "not resolvable")

    def test_headline_inconclusive_on_not_resolvable(self):
        metrics = {"roc_auc": 0.60, "permutation_test": {"n_permutations": 100, "metrics": {
            "roc_auc": {"observed_mean": 0.60, "null_mean": 0.50, "p_value": 0.001}}}}
        out = claims.scientific_claims_check(metrics, baseline_paired=self._paired(-0.01, 0.06, 0.025))
        self.assertEqual(out["baselines"]["status"], "not resolvable")
        self.assertIn("Inconclusive", out["headline"])

    def test_html_status_not_resolvable_is_not_green(self):
        # Regression guard: an inconclusive gate must NOT render as a pass (gate-ok).
        self.assertEqual(htmlreport._status_class("not resolvable"), "gate-warn")
        self.assertEqual(htmlreport._status_class("beats floor"), "gate-ok")

    def test_headline_downgrades_on_failed_gate(self):
        metrics = {"roc_auc": 0.52, "permutation_test": {"n_permutations": 100, "metrics": {
            "roc_auc": {"observed_mean": 0.52, "null_mean": 0.50, "p_value": 0.4}}}}
        out = claims.scientific_claims_check(metrics)
        self.assertIn("No usable signal", out["headline"])
        # The clinical caveat is always present.
        self.assertIn("Do not claim", out["clinical_caveat"])

    def test_claims_markdown_always_carries_caveat(self):
        out = claims.scientific_claims_check(None, model_trained=False)
        md = claims.claims_markdown(out)
        self.assertIn("Scientific claims check", md)
        self.assertIn("Do not claim", md)


class PaletteTests(unittest.TestCase):
    def test_apply_palette_swaps_figure_colors(self):
        from epinet import viz

        palette.apply_palette("vahtian")
        self.assertEqual(viz.CATEGORY_COLORS[0], "#5E4F99")
        self.assertEqual(viz.HIGHLIGHT, "#2D2440")
        palette.apply_palette("wong")
        self.assertEqual(viz.CATEGORY_COLORS[0], "#0072B2")

    def test_unknown_palette_falls_back_to_wong(self):
        self.assertEqual(palette.get_palette("nope")["name"], "wong")


class HtmlRenderTests(unittest.TestCase):
    def test_md_to_html_renders_headers_and_tables(self):
        md = "## Title\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\n- item\n"
        out = htmlreport.md_to_html(md)
        self.assertIn("<h2>Title</h2>", out)
        self.assertIn("<th>a</th>", out)
        self.assertIn("<li>item</li>", out)

    def test_build_report_always_has_caveat_claims_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            (out / "model_card.md").write_text("# Model card\n\n## Provenance\n\nx\n")
            (out / "provenance.json").write_text('{"epinet_version": "0.4.0", "git": {}}')
            config = AnalysisConfig()
            config.analysis.reporting.primary_color = "#123456"
            cc = claims.scientific_claims_check(None, model_trained=False)
            path = htmlreport.build_html_report(out, config=config, claims=cc,
                                                palette=palette.get_palette("vahtian"))
            doc = path.read_text()
            self.assertIn("Do not claim", doc)            # caveat
            self.assertIn("Scientific claims check", doc)  # claims gates
            self.assertIn("Provenance", doc)               # provenance
            self.assertIn("--vh-primary:#123456", doc)     # theme override applied


class RunConfigReportTests(unittest.TestCase):
    def test_run_writes_html_and_claims_into_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            rng = np.random.default_rng(0)
            n = 80
            y = rng.integers(0, 2, n)
            pd.DataFrame({
                "ID": [f"n{i}" for i in range(n)], "Outcome": y,
                "F1": rng.random(n) + y * 0.5, "F2": rng.random(n),
            }).to_csv(Path(td) / "nodes.csv", index=False)

            out = Path(td) / "out"
            config, _ = wb.build_plan(nodes_path=str(Path(td) / "nodes.csv"),
                                      outcome="Outcome", output_dir=str(out), name="rep")
            config.analysis.evaluation.permutation_repeats = 10
            config.analysis.evaluation.n_iterations = 2
            config.analysis.evaluation.bootstrap_ci = False
            summary = wb.run_config(config)

            self.assertTrue((out / "index.html").exists())
            self.assertTrue((out / "claims_check.json").exists())
            self.assertTrue((out / "split_comparison.json").exists())
            self.assertEqual(summary.get("html_report"), "index.html")
            import zipfile
            members = zipfile.ZipFile(summary["bundle"]).namelist()
            self.assertIn("index.html", members)
            self.assertIn("claims_check.json", members)


if __name__ == "__main__":
    unittest.main()
