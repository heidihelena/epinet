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
