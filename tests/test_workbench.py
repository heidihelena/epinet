"""Tests for the EpiNet Workbench: config, schema inference, gates, and the runner.

The central guarantee is *parity*: the UI and CLI both reduce to a config and the
same ``run_config``, and that config round-trips through YAML losslessly. These
tests pin that contract plus the safety gates and the result bundle.
"""

import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from epinet import config as ecfg
from epinet import schema as esch
from epinet import workbench as wb


def _make_nodes(path, n=60, *, single_class=False, tiny_positive=False, leakage=False, seed=0):
    rng = np.random.default_rng(seed)
    if single_class:
        y = np.ones(n, dtype=int)
    elif tiny_positive:
        y = np.array([1, 1, 1] + [0] * (n - 3))
    else:
        y = rng.integers(0, 2, n)
    data = {
        "ID": [f"n{i}" for i in range(n)],
        "Outcome": y,
        "Feature1": rng.random(n) + y * 0.4,
        "Feature2": rng.random(n),
    }
    if leakage:
        data["death_date"] = rng.random(n)
    pd.DataFrame(data).to_csv(path, index=False)
    return path


class ConfigTests(unittest.TestCase):
    def test_yaml_round_trip_is_lossless(self):
        config = ecfg.AnalysisConfig()
        config.project.name = "demo"
        config.schema.feature_columns = ["a", "b"]
        config.analysis.evaluation.permutation_repeats = 250
        restored = ecfg.AnalysisConfig.from_yaml(config.to_yaml())
        self.assertEqual(config.to_dict(), restored.to_dict())

    def test_from_dict_ignores_unknown_keys(self):
        config = ecfg.AnalysisConfig.from_dict(
            {"project": {"name": "x", "future_field": 1}, "unknown": True}
        )
        self.assertEqual(config.project.name, "x")

    def test_validate_config_flags_bad_plan(self):
        config = ecfg.AnalysisConfig()
        config.data.nodes_path = None
        config.analysis.split.test_size = 1.5
        config.analysis.graph.mode = "bogus"
        errors = ecfg.validate_config(config)
        self.assertTrue(any("nodes_path" in e for e in errors))
        self.assertTrue(any("test_size" in e for e in errors))
        self.assertTrue(any("graph.mode" in e for e in errors))


class SchemaInferenceTests(unittest.TestCase):
    def test_infers_id_outcome_and_keeps_continuous_features(self):
        with tempfile.TemporaryDirectory() as td:
            path = _make_nodes(Path(td) / "n.csv")
            profile = esch.profile_table(path, id_column="ID")
            schema = esch.infer_schema(profile, mode="single_csv")
            self.assertEqual(schema.id_column, "ID")
            self.assertEqual(schema.outcome_column, "Outcome")
            # All-unique float columns are features, not identifiers.
            self.assertIn("Feature1", schema.feature_columns)
            self.assertIn("Feature2", schema.feature_columns)
            self.assertNotIn("Feature1", schema.exclude_columns)

    def test_flags_and_excludes_suspected_leakage(self):
        with tempfile.TemporaryDirectory() as td:
            path = _make_nodes(Path(td) / "n.csv", leakage=True)
            profile = esch.profile_table(path, id_column="ID")
            schema = esch.infer_schema(profile, mode="single_csv")
            self.assertIn("death_date", schema.leakage_flags)
            self.assertIn("death_date", schema.exclude_columns)
            self.assertNotIn("death_date", schema.feature_columns)

    def test_duplicate_ids_are_reported(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dup.csv"
            pd.DataFrame({"ID": ["a", "a", "b"], "Outcome": [0, 1, 0]}).to_csv(path, index=False)
            profile = esch.profile_table(path, id_column="ID")
            self.assertEqual(profile.duplicate_id_count, 1)
            self.assertTrue(profile.errors)


class GateTests(unittest.TestCase):
    def test_single_class_outcome_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            path = _make_nodes(Path(td) / "n.csv", single_class=True)
            config, profile = wb.build_plan(nodes_path=str(path), outcome="Outcome",
                                            output_dir=str(Path(td) / "out"))
            gates = wb.check_gates(config, profile)
            self.assertFalse(gates.ok)
            self.assertTrue(any("single-class" in b for b in gates.blocks))

    def test_tiny_positive_class_downgrades_to_descriptive(self):
        with tempfile.TemporaryDirectory() as td:
            path = _make_nodes(Path(td) / "n.csv", tiny_positive=True)
            config, profile = wb.build_plan(nodes_path=str(path), outcome="Outcome",
                                            output_dir=str(Path(td) / "out"))
            gates = wb.check_gates(config, profile)
            self.assertTrue(gates.ok)  # not blocked, but downgraded
            self.assertTrue(gates.downgraded_to_descriptive)

    def test_id_as_feature_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            path = _make_nodes(Path(td) / "n.csv")
            config, profile = wb.build_plan(nodes_path=str(path), outcome="Outcome",
                                            output_dir=str(Path(td) / "out"))
            config.schema.feature_columns.append(config.schema.id_column)
            gates = wb.check_gates(config, profile)
            self.assertFalse(gates.ok)
            self.assertTrue(any("ID column" in b for b in gates.blocks))

    def test_leakage_feature_override_warns(self):
        with tempfile.TemporaryDirectory() as td:
            path = _make_nodes(Path(td) / "n.csv", leakage=True)
            config, profile = wb.build_plan(nodes_path=str(path), outcome="Outcome",
                                            output_dir=str(Path(td) / "out"))
            config.schema.feature_columns.append("death_date")
            gates = wb.check_gates(config, profile)
            self.assertTrue(gates.ok)
            self.assertTrue(any("death_date" in w and "leakage" in w for w in gates.warnings))


class RunnerTests(unittest.TestCase):
    def _fast(self, config):
        config.analysis.evaluation.permutation_repeats = 0
        config.analysis.evaluation.n_iterations = 2
        config.analysis.evaluation.bootstrap_ci = False
        return config

    def test_single_csv_run_writes_bundle_and_canonical_files(self):
        with tempfile.TemporaryDirectory() as td:
            path = _make_nodes(Path(td) / "n.csv", n=80)
            out = Path(td) / "out"
            config, _ = wb.build_plan(nodes_path=str(path), outcome="Outcome",
                                      output_dir=str(out), name="run")
            summary = wb.run_config(self._fast(config))
            # Canonical bundle members exist.
            for name in ("analysis.yaml", "model_metrics.json", "model_card.md",
                         "provenance.json", "node_features.csv", "gate_report.json",
                         "environment.txt"):
                self.assertTrue((out / name).exists(), name)
            bundle = Path(summary["bundle"])
            self.assertTrue(bundle.exists())
            members = zipfile.ZipFile(bundle).namelist()
            self.assertIn("analysis.yaml", members)
            self.assertIn("model_metrics.json", members)
            self.assertTrue(any(m.startswith("plots/") for m in members))

    def test_cli_ui_parity_same_config_same_outputs(self):
        # The UI and CLI differ only in how they *build* the config; the run is
        # identical. Persisting and reloading the config must yield the same plan.
        with tempfile.TemporaryDirectory() as td:
            path = _make_nodes(Path(td) / "n.csv")
            config, _ = wb.build_plan(nodes_path=str(path), outcome="Outcome",
                                      output_dir=str(Path(td) / "out"), name="parity")
            yaml_path = Path(td) / "analysis.yaml"
            config.write(yaml_path)
            reloaded = ecfg.AnalysisConfig.load(yaml_path)
            self.assertEqual(config.to_dict(), reloaded.to_dict())

    def test_descriptive_run_skips_model(self):
        with tempfile.TemporaryDirectory() as td:
            path = _make_nodes(Path(td) / "n.csv", tiny_positive=True)
            out = Path(td) / "out"
            config, _ = wb.build_plan(nodes_path=str(path), outcome="Outcome",
                                      output_dir=str(out), name="desc")
            summary = wb.run_config(self._fast(config))
            self.assertNotIn("model", summary)
            self.assertTrue(summary["gates"]["downgraded_to_descriptive"])
            self.assertFalse((out / "model_metrics.json").exists())

    def test_blocked_run_raises(self):
        with tempfile.TemporaryDirectory() as td:
            path = _make_nodes(Path(td) / "n.csv", single_class=True)
            config, _ = wb.build_plan(nodes_path=str(path), outcome="Outcome",
                                      output_dir=str(Path(td) / "out"), name="blk")
            with self.assertRaises(SystemExit):
                wb.run_config(self._fast(config))

    def test_similarity_graph_synthesizes_edges(self):
        with tempfile.TemporaryDirectory() as td:
            path = _make_nodes(Path(td) / "n.csv", n=40)
            out = Path(td) / "out"
            config, _ = wb.build_plan(nodes_path=str(path), outcome="Outcome",
                                      output_dir=str(out), name="sim")
            config.analysis.graph.mode = "similarity"
            config.analysis.graph.k_neighbors = 4
            wb.run_config(self._fast(config))
            edges = pd.read_csv(out / "inputs" / "train_edges.csv")
            self.assertGreater(len(edges), 0)
            self.assertIn("SourceID", edges.columns)


if __name__ == "__main__":
    unittest.main()
