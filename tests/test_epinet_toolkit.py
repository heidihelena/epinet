import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

import epinet_cluster as ec
import epinet_common as ecommon
import epinet_toolkit as et
import epinet_viz as ev


class CommonHelperTests(unittest.TestCase):
    def test_blank_label_mask_handles_dtypes(self):
        # Object dtype with the blank tokens.
        s = pd.Series(["a", "", "nan", "None", "b", None])
        self.assertEqual(ecommon.blank_label_mask(s).tolist(),
                         [False, True, True, True, False, True])
        # Modern pandas string dtype with a real NA (the bug this guards against).
        s2 = pd.Series(["x", pd.NA, "y"], dtype="string")
        self.assertEqual(ecommon.labeled_mask(s2).tolist(), [True, False, True])
        # Numeric labels are all considered present.
        s3 = pd.Series([0, 1, 2])
        self.assertTrue(ecommon.labeled_mask(s3).all())

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import validate_nodule_models as vnm  # noqa: E402


class ToolkitTests(unittest.TestCase):
    def test_graph_features_and_shortest_paths(self):
        nodes = pd.DataFrame(
            [
                {"ID": "A", "Outcome": 0, "Value": 1.0},
                {"ID": "B", "Outcome": 0, "Value": 2.0},
                {"ID": "C", "Outcome": 1, "Value": 3.0},
                {"ID": "D", "Outcome": 0, "Value": 4.0},
            ]
        )
        edges = pd.DataFrame(
            [
                {"SourceID": "A", "TargetID": "B", "Weight": 1.0},
                {"SourceID": "B", "TargetID": "C", "Weight": 1.0},
            ]
        )

        et.validate_tables(
            nodes,
            edges,
            id_column="ID",
            source_column="SourceID",
            target_column="TargetID",
            outcome_column="Outcome",
        )
        graph = et.build_graph(nodes, edges, weight_column="Weight")
        features = et.generate_graph_features(graph)
        self.assertEqual(set(features["ID"]), {"A", "B", "C", "D"})
        self.assertEqual(float(features.loc[features["ID"].eq("B"), "degree"].iloc[0]), 2.0)

        targets = et.select_target_nodes(
            nodes,
            id_column="ID",
            outcome_column="Outcome",
            target_outcome="1",
            target_nodes=[],
        )
        pairs, nearest = et.shortest_path_records(graph, source_nodes=["A", "D"], target_nodes=targets)
        self.assertEqual(pairs.loc[pairs["source"].eq("A"), "path"].iloc[0], "A -> B -> C")
        self.assertEqual(nearest.loc[nearest["source"].eq("A"), "hops"].iloc[0], 2)
        self.assertEqual(nearest.loc[nearest["source"].eq("D"), "path"].iloc[0], "")

    def test_cli_run_writes_expected_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            nodes = root / "nodes.csv"
            edges = root / "edges.csv"
            out = root / "out"
            nodes.write_text(
                "ID,Outcome,Value\n"
                "A,0,1\n"
                "B,1,2\n"
                "C,0,3\n"
                "D,1,4\n"
                "E,0,5\n"
                "F,1,6\n"
            )
            edges.write_text(
                "SourceID,TargetID\n"
                "A,B\n"
                "B,C\n"
                "C,D\n"
                "D,E\n"
                "E,F\n"
            )
            args = Namespace(
                nodes=str(nodes),
                edges=str(edges),
                output_dir=str(out),
                id_column="ID",
                source_column="SourceID",
                target_column="TargetID",
                outcome_column="Outcome",
                target_outcome="1",
                source_nodes="",
                target_nodes="",
                weight_column=None,
                use_weighted_paths=False,
                path_mode="hops",
                directed=False,
                include_centrality=False,
                run_model=False,
                run_paths=True,
                test_size=0.2,
                random_state=42,
            )
            summary = et.run(args)
            self.assertEqual(summary["graph"]["nodes"], 6)
            self.assertTrue((out / "node_features.csv").exists())
            self.assertTrue((out / "nearest_targets.csv").exists())
            self.assertTrue((out / "run_summary.json").exists())

    def test_citematch_example_finds_contrast_path(self):
        repo = Path(__file__).resolve().parents[1]
        nodes = repo / "examples" / "citematch_nodes.csv"
        edges = repo / "examples" / "citematch_edges.csv"

        with tempfile.TemporaryDirectory() as td:
            args = Namespace(
                nodes=str(nodes),
                edges=str(edges),
                output_dir=td,
                id_column="ID",
                source_column="SourceID",
                target_column="TargetID",
                outcome_column="Outcome",
                target_outcome="contrast_evidence",
                source_nodes="claim_chemo_required,claim_osimertinib_dfs",
                target_nodes="",
                weight_column=None,
                use_weighted_paths=False,
                path_mode="hops",
                directed=False,
                include_centrality=False,
                run_model=False,
                run_paths=True,
                test_size=0.2,
                random_state=42,
            )
            summary = et.run(args)
            self.assertEqual(summary["shortest_paths"]["target_count"], 1)

            nearest = pd.read_csv(Path(td) / "nearest_targets.csv")
            direct = nearest.loc[nearest["source"].eq("claim_chemo_required")].iloc[0]
            self.assertEqual(direct["nearest_target"], "paper_contrast_editorial")
            self.assertEqual(direct["hops"], 1)

    def test_strength_path_prefers_stronger_two_step_route(self):
        nodes = pd.DataFrame(
            [
                {"ID": "A"},
                {"ID": "B"},
                {"ID": "C"},
                {"ID": "T"},
            ]
        )
        edges = pd.DataFrame(
            [
                {"SourceID": "A", "TargetID": "T", "Weight": 0.2},
                {"SourceID": "A", "TargetID": "B", "Weight": 0.9},
                {"SourceID": "B", "TargetID": "T", "Weight": 0.8},
                {"SourceID": "A", "TargetID": "C", "Weight": 0.7},
                {"SourceID": "C", "TargetID": "T", "Weight": 0.7},
            ]
        )
        graph = et.build_graph(nodes, edges, weight_column="Weight")
        _, nearest = et.shortest_path_records(
            graph,
            source_nodes=["A"],
            target_nodes=["T"],
            path_mode="strength",
        )
        row = nearest.iloc[0]
        self.assertEqual(row["path"], "A -> B -> T")
        self.assertAlmostEqual(row["path_strength"], 0.72)

    def test_target_coverage_summarizes_per_target_reachability(self):
        nodes = pd.DataFrame([{"ID": node} for node in ["A", "B", "C", "T1", "T2"]])
        edges = pd.DataFrame(
            [
                {"SourceID": "A", "TargetID": "B"},
                {"SourceID": "B", "TargetID": "T1"},
                # C and T2 are isolated from the main component.
                {"SourceID": "C", "TargetID": "T2"},
            ]
        )
        graph = et.build_graph(nodes, edges)
        pairs, nearest = et.shortest_path_records(
            graph,
            source_nodes=["A", "B", "C"],
            target_nodes=["T1", "T2"],
        )
        coverage = et.target_coverage_records(pairs).set_index("target")

        self.assertEqual(coverage.loc["T1", "reachable_source_count"], 2)
        self.assertEqual(coverage.loc["T2", "reachable_source_count"], 1)
        self.assertAlmostEqual(coverage.loc["T1", "coverage"], 2 / 3)
        self.assertAlmostEqual(coverage.loc["T1", "min_distance"], 1.0)
        self.assertAlmostEqual(coverage.loc["T1", "mean_distance"], 1.5)
        # reachable_target_count in nearest must match the pair table.
        self.assertEqual(
            int(nearest.loc[nearest["source"].eq("A"), "reachable_target_count"].iloc[0]), 1
        )

    def _synthetic_run_args(self, root: Path, out: Path, **overrides) -> Namespace:
        nodes = root / "nodes.csv"
        edges = root / "edges.csv"
        rng = np.random.default_rng(0)
        node_rows = ["ID,Outcome,Value"]
        for i in range(30):
            node_rows.append(f"N{i},{i % 2},{rng.random():.4f}")
        nodes.write_text("\n".join(node_rows) + "\n")
        edge_rows = ["SourceID,TargetID"]
        for i in range(29):
            edge_rows.append(f"N{i},N{i + 1}")
        edges.write_text("\n".join(edge_rows) + "\n")
        args = Namespace(
            nodes=str(nodes),
            edges=str(edges),
            output_dir=str(out),
            id_column="ID",
            source_column="SourceID",
            target_column="TargetID",
            outcome_column="Outcome",
            target_outcome="1",
            source_nodes="",
            target_nodes="",
            weight_column=None,
            use_weighted_paths=False,
            path_mode="hops",
            directed=False,
            include_centrality=False,
            run_model=True,
            run_paths=True,
            run_clusters=False,
            n_clusters=0,
            distance_metric="euclidean",
            make_plots=False,
            n_iterations=1,
            split_strategy="random",
            permutation_test=0,
            test_size=0.2,
            random_state=42,
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_iterative_model_evaluation_reports_metric_spread(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "out"
            args = self._synthetic_run_args(root, out, n_iterations=3)
            summary = et.run(args)

            model = summary["model"]
            self.assertEqual(model["n_iterations"], 3)
            self.assertIn("iteration_summary", model)
            self.assertIn("mean", model["iteration_summary"]["f1_weighted"])

            iteration_metrics = pd.read_csv(out / "model_iteration_metrics.csv")
            self.assertEqual(len(iteration_metrics), 3)
            self.assertEqual(iteration_metrics["random_state"].tolist(), [42, 43, 44])

            importance = pd.read_csv(out / "model_feature_importance.csv")
            self.assertIn("importance_std", importance.columns)

            persisted = json.loads((out / "model_metrics.json").read_text())
            self.assertIn("iteration_summary", persisted)

    def test_single_iteration_keeps_legacy_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "out"
            summary = et.run(self._synthetic_run_args(root, out, n_iterations=1))
            self.assertNotIn("iteration_summary", summary["model"])
            self.assertFalse((out / "model_iteration_metrics.csv").exists())
            self.assertTrue((out / "target_coverage.csv").exists())

    def test_run_writes_plots(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "out"
            args = self._synthetic_run_args(root, out, n_iterations=3, make_plots=True)
            summary = et.run(args)

            self.assertIn("plots", summary)
            expected = {
                "network_overview.png",
                "degree_distribution.png",
                "feature_importance.png",
                "confusion_matrix.png",
                "metric_stability.png",
            }
            written = {Path(p).name for p in summary["plots"]}
            self.assertEqual(written, expected)
            for plot in summary["plots"]:
                self.assertGreater((out / plot).stat().st_size, 0)

    def _two_cluster_graph(self):
        # Two triangles joined by a single bridge edge.
        nodes = pd.DataFrame([{"ID": n} for n in ["A", "B", "C", "D", "E", "F"]])
        edges = pd.DataFrame(
            [
                {"SourceID": "A", "TargetID": "B"},
                {"SourceID": "B", "TargetID": "C"},
                {"SourceID": "C", "TargetID": "A"},
                {"SourceID": "D", "TargetID": "E"},
                {"SourceID": "E", "TargetID": "F"},
                {"SourceID": "F", "TargetID": "D"},
                {"SourceID": "C", "TargetID": "D"},
            ]
        )
        return et.build_graph(nodes, edges)

    def test_community_labels_separate_two_clusters(self):
        labels = et.community_labels(self._two_cluster_graph())
        self.assertEqual(labels[["A", "B", "C"]].nunique(), 1)
        self.assertEqual(labels[["D", "E", "F"]].nunique(), 1)
        self.assertNotEqual(labels["A"], labels["D"])

    def test_group_split_keeps_communities_intact(self):
        labels = et.community_labels(self._two_cluster_graph())
        X = pd.DataFrame({"x": range(len(labels))}, index=labels.index)
        y = pd.Series([0, 1] * 3, index=labels.index)
        train_idx, test_idx = et._split_indices(
            X, y, test_size=0.5, random_state=0, stratify_ok=False, groups=labels
        )
        train_groups = set(labels.iloc[train_idx])
        test_groups = set(labels.iloc[test_idx])
        self.assertTrue(train_groups)
        self.assertTrue(test_groups)
        self.assertEqual(train_groups & test_groups, set())

    def test_community_split_strategy_is_recorded(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "out"
            args = self._synthetic_run_args(
                root, out, n_iterations=2, split_strategy="community"
            )
            summary = et.run(args)
            self.assertEqual(summary["model"]["split_strategy"], "community")
            self.assertGreaterEqual(summary["model"]["n_groups"], 2)

    def test_permutation_test_reports_null_distribution(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "out"
            args = self._synthetic_run_args(
                root, out, n_iterations=2, permutation_test=10, make_plots=True
            )
            summary = et.run(args)

            test = summary["model"]["permutation_test"]
            self.assertEqual(test["n_permutations"], 10)
            f1 = test["metrics"]["f1_weighted"]
            self.assertGreater(f1["p_value"], 0.0)
            self.assertLessEqual(f1["p_value"], 1.0)
            self.assertIn("observed_mean", f1)
            self.assertIn("null_mean", f1)

            null_rows = pd.read_csv(out / "model_permutation_metrics.csv")
            self.assertEqual(len(null_rows), 10)

            plot_names = {Path(p).name for p in summary["plots"]}
            self.assertIn("permutation_null.png", plot_names)

    def test_partially_labeled_outcome_excludes_scaffold(self):
        # Two labeled classes among "indicator" nodes plus unlabeled scaffold.
        rng = np.random.default_rng(1)
        node_rows = []
        for i in range(20):
            node_rows.append({"ID": f"I{i}", "Outcome": "a" if i % 2 else "b",
                              "Value": float(rng.random())})
        for j in range(8):
            node_rows.append({"ID": f"S{j}", "Outcome": "", "Value": float(rng.random())})
        nodes = pd.DataFrame(node_rows)
        edges = pd.DataFrame(
            [{"SourceID": f"I{i}", "TargetID": f"S{i % 8}"} for i in range(20)]
            + [{"SourceID": f"S{j}", "TargetID": f"S{(j + 1) % 8}"} for j in range(8)]
        )
        graph = et.build_graph(nodes, edges)
        features = et.generate_graph_features(graph)
        with tempfile.TemporaryDirectory() as td:
            result = et.train_outcome_model(
                nodes, features, id_column="ID", outcome_column="Outcome",
                output_dir=Path(td), n_iterations=3,
            )
            metrics = result["metrics"]
            self.assertEqual(metrics["labeled_rows"], 20)
            self.assertEqual(metrics["unlabeled_excluded"], 8)
            self.assertEqual(set(metrics["classes"]), {"a", "b"})

    def test_plot_network_marks_unlabeled_scaffold(self):
        nodes = pd.DataFrame(
            [{"ID": "A", "Outcome": "broad"}, {"ID": "B", "Outcome": ""},
             {"ID": "C", "Outcome": "gap"}]
        )
        edges = pd.DataFrame([{"SourceID": "A", "TargetID": "B"},
                              {"SourceID": "B", "TargetID": "C"}])
        graph = et.build_graph(nodes, edges)
        with tempfile.TemporaryDirectory() as td:
            path = ev.plot_network(graph, Path(td) / "net.png", outcome_attribute="Outcome")
            self.assertGreater(path.stat().st_size, 0)

    def _separable_design(self):
        # Two well-separated blobs in a 3-feature space, labeled by blob.
        rng = np.random.default_rng(3)
        a = rng.normal(0.0, 0.3, size=(12, 3))
        b = rng.normal(5.0, 0.3, size=(12, 3))
        X = pd.DataFrame(
            np.vstack([a, b]),
            columns=["f1", "f2", "f3"],
            index=[f"n{i}" for i in range(24)],
        )
        y = pd.Series(["a"] * 12 + ["b"] * 12, index=X.index, name="Outcome")
        return X, y

    def test_distance_metrics_match_known_geometry(self):
        X = pd.DataFrame({"f1": [0.0, 3.0], "f2": [0.0, 4.0]}, index=["p", "q"])
        Xz, _ = ec.standardize(X)
        d = ec.distances_to_points(Xz, Xz, metric="euclidean")
        # Symmetric, zero on the diagonal, and equal off-diagonal.
        self.assertAlmostEqual(d[0, 0], 0.0)
        self.assertAlmostEqual(d[0, 1], d[1, 0])
        self.assertGreater(d[0, 1], 0.0)

    def test_cluster_recovers_blobs_and_centroid_classifier(self):
        X, y = self._separable_design()
        for metric in ("euclidean", "mahalanobis"):
            result = ec.cluster_nodes(X, y=y, n_clusters=0, metric=metric)
            summary = result["summary"]
            self.assertEqual(summary["n_clusters"], 2)
            # Each k-means cluster should be pure in outcome composition.
            for comp in summary["cluster_outcome_composition"].values():
                self.assertEqual(sum(1 for v in comp.values() if v > 0), 1)
            # Nearest-centroid recovers the labels exactly on separable blobs.
            self.assertEqual(
                summary["class_centroids"]["nearest_centroid_insample_accuracy"], 1.0
            )
            assignments = result["assignments"]
            self.assertIn("dist_to_a", assignments.columns)
            self.assertIn("nearest_class_centroid", assignments.columns)

    def test_cluster_labeled_only_skips_scaffold(self):
        # 12 labeled nodes with features + 6 feature-less scaffold nodes.
        X = pd.DataFrame(
            {"f1": list(range(12)) + [0] * 6, "f2": list(range(12)) + [0] * 6},
            index=[f"n{i}" for i in range(12)] + [f"s{j}" for j in range(6)],
            dtype=float,
        )
        y = pd.Series(
            ["a"] * 6 + ["b"] * 6 + [""] * 6, index=X.index, name="Outcome"
        )
        result = ec.cluster_nodes(X, y=y, n_clusters=2, labeled_only=True)
        # Only the 12 labeled nodes should appear in the assignments.
        self.assertEqual(len(result["assignments"]), 12)
        self.assertTrue(result["assignments"]["ID"].str.startswith("n").all())

    def test_cluster_excludes_constant_features(self):
        X = pd.DataFrame(
            {"varies": [0.0, 1.0, 2.0, 3.0], "constant": [5.0, 5.0, 5.0, 5.0]},
            index=list("abcd"),
        )
        result = ec.cluster_nodes(X, n_clusters=2, metric="euclidean")
        self.assertEqual(result["summary"]["feature_columns"], ["varies"])

    def test_run_writes_cluster_outputs_and_plot(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "out"
            args = self._synthetic_run_args(
                root, out, run_clusters=True, distance_metric="mahalanobis",
                n_clusters=0, make_plots=True,
            )
            summary = et.run(args)
            self.assertIn("clusters", summary)
            self.assertTrue((out / "node_clusters.csv").exists())
            self.assertTrue((out / "cluster_centroids.csv").exists())
            self.assertTrue((out / "cluster_summary.json").exists())
            self.assertIn("feature_clusters.png", {Path(p).name for p in summary["plots"]})

    def test_run_writes_vector_plots(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "out"
            args = self._synthetic_run_args(
                root, out, n_iterations=3, make_plots=True,
                plot_format="pdf", plot_dpi=200,
            )
            summary = et.run(args)
            self.assertTrue(all(p.endswith(".pdf") for p in summary["plots"]))
            for plot in summary["plots"]:
                self.assertTrue((out / plot).exists())

    def test_house_style_removes_chartjunk(self):
        import matplotlib.pyplot as plt

        ev.apply_house_style()
        self.assertFalse(plt.rcParams["axes.spines.top"])
        self.assertFalse(plt.rcParams["axes.spines.right"])
        self.assertFalse(plt.rcParams["legend.frameon"])

    def test_plot_confusion_matrix_handles_empty_row(self):
        # A class with no actual samples (zero row) must not divide-by-zero.
        with tempfile.TemporaryDirectory() as td:
            path = ev.plot_confusion_matrix(
                [[3, 1], [0, 0]], ["a", "b"], Path(td) / "cm.png")
            self.assertGreater(path.stat().st_size, 0)

    def test_plot_network_handles_missing_outcome(self):
        graph = et.build_graph(
            pd.DataFrame([{"ID": "A"}, {"ID": "B"}]),
            pd.DataFrame([{"SourceID": "A", "TargetID": "B"}]),
        )
        with tempfile.TemporaryDirectory() as td:
            path = ev.plot_network(graph, Path(td) / "net.png")
            self.assertGreater(path.stat().st_size, 0)


class NoduleModelValidationTests(unittest.TestCase):
    def test_port_matches_verbatim_ntog_formula(self):
        eq = vnm.check_source_equivalence(n=2000, seed=1)
        self.assertLess(eq["max_brock_abs_error"], 1e-9)
        self.assertLess(eq["max_mayo_abs_error"], 1e-9)

    def test_coefficients_match_published_odds_ratios(self):
        for name, r in vnm.check_odds_ratios().items():
            self.assertTrue(r["ok"], f"{name}: exp(coef)={r['computed_or']} vs {r['published_or']}")

    def test_worked_cases_and_properties(self):
        wc = vnm.check_worked_cases()
        self.assertTrue(wc["size_term_zero_at_4mm"])
        self.assertTrue(wc["worked_brock_8mm"]["ok"])
        self.assertTrue(wc["type_ordering_partsolid_solid_nonsolid"])
        self.assertTrue(wc["monotonic_in_diameter"])
        self.assertTrue(wc["vdt_doubling_100d"])
        self.assertTrue(wc["vdt_quadruple_300d"])


class NlstLoaderTests(unittest.TestCase):
    def test_demo_fixture_assembles_cohort(self):
        import build_nlst_cohort as bnl

        participant, abnormalities = bnl.demo_fixture(seed=1)
        frames = bnl.assemble(participant, abnormalities)
        nodes, edges, prov = frames["nodes"], frames["edges"], frames["provenance"]

        nodules = nodes[nodes["NodeType"] == "Nodule"]
        self.assertGreater(len(nodules), 0)
        # Nodules are labeled; participants are unlabeled scaffold.
        self.assertTrue((nodules["Outcome"] != "").all())
        self.assertTrue((nodes[nodes["NodeType"] == "Participant"]["Outcome"] == "").all())
        self.assertEqual(set(nodules["Outcome"]) - {"benign_low", "suspicious_high"}, set())
        # The demographics that make Brock/Mayo/NTOG computable are carried.
        for col in ["Age", "PackYears", "CurrentSmoker", "FamilyHistory", "LungCancer"]:
            self.assertIn(col, prov.columns)
        # Edges reference only known node IDs.
        ids = set(nodes["ID"])
        self.assertTrue(set(edges["SourceID"]) <= ids and set(edges["TargetID"]) <= ids)

    def test_missing_column_raises_actionable_error(self):
        import build_nlst_cohort as bnl

        participant, abnormalities = bnl.demo_fixture(seed=2)
        participant = participant.drop(columns=["canclung"])
        with self.assertRaises(KeyError):
            bnl.assemble(participant, abnormalities)


if __name__ == "__main__":
    unittest.main()
