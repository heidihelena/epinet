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
import epinet_contest as ecn
import epinet_federated as efed
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
                # Binary outcome -> calibration reliability diagram; learning
                # curve is produced whenever cross-validation is feasible.
                "calibration.png",
                "learning_curve.png",
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

    def test_network_to_html_is_self_describing(self):
        nodes = pd.DataFrame([{"ID": "A", "Outcome": "x"}, {"ID": "B", "Outcome": ""},
                              {"ID": "C", "Outcome": "y"}])
        edges = pd.DataFrame([{"SourceID": "A", "TargetID": "B"},
                              {"SourceID": "B", "TargetID": "C"}])
        graph = et.build_graph(nodes, edges)
        with tempfile.TemporaryDirectory() as td:
            path = ev.network_to_html(graph, Path(td) / "net.html", outcome_attribute="Outcome")
            html = path.read_text()
            self.assertIn("vis.DataSet", html)
            self.assertEqual(html.count('"id":'), 3)
            self.assertIn("unlabeled (scaffold)", html)  # blank-outcome legend entry

    def test_lymphoma_workflow_builds_runnable_cohort(self):
        import build_lymphoma_workflow as blw

        features = blw.synthetic_lymphoma_cohort(n_per_class=12, seed=1)
        nodes, edges = blw.build_similarity_graph(
            features, id_col="CaseID", label_col="Subtype", k=4)
        # Outcome stored under its real column name; graph is connected enough to model.
        self.assertIn("Subtype", nodes.columns)
        self.assertEqual(set(nodes["Subtype"]), {"DLBCL", "FL", "CLL", "MCL", "BL"})
        self.assertGreater(len(edges), 0)
        ids = set(nodes["ID"])
        self.assertTrue(set(edges["SourceID"]) <= ids and set(edges["TargetID"]) <= ids)

    def test_lymphoma_grey_zone_cases_are_the_contested_ones(self):
        import build_lymphoma_workflow as blw

        cohort = blw.synthetic_lymphoma_cohort(n_per_class=30, seed=0, grey_zone=12)
        grey = cohort["CaseID"].str.startswith("GZ_").to_numpy()
        self.assertEqual(int(grey.sum()), 12)

        X = cohort.set_index("CaseID")[blw._FEATURES]
        y = cohort.set_index("CaseID")["Subtype"]
        result = ecn.contestability(X, y=y, metric="mahalanobis", contest_quantile=0.1)
        frame = result["assignments"].set_index("ID")
        frame["is_grey"] = grey

        # Grey-zone cases sit on a boundary: much smaller flip-distance than the bulk.
        self.assertLess(
            frame.loc[frame["is_grey"], "flip_distance"].mean(),
            frame.loc[~frame["is_grey"], "flip_distance"].mean(),
        )
        grey = frame.loc[frame["is_grey"]]
        # The lens parks them on the CLL/MCL axis: most are nearest one of the pair.
        on_axis = grey["nearest_class_centroid"].isin({"CLL", "MCL"}).mean()
        self.assertGreaterEqual(on_axis, 0.8)
        # And the marker it most often names to resolve them is cyclin D1 — the
        # real discriminator (t(11;14) defines MCL vs CLL), learned from features.
        self.assertEqual(grey["most_decision_relevant_feature"].value_counts().idxmax(), "CyclinD1")

    def test_lymphoma_grey_zone_pair_must_be_known(self):
        import build_lymphoma_workflow as blw

        with self.assertRaises(ValueError):
            blw.synthetic_lymphoma_cohort(n_per_class=5, grey_zone=3,
                                          grey_zone_pair=("CLL", "Hodgkin"))

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


class ContestabilityTests(unittest.TestCase):
    def test_flip_distance_matches_closed_form_geometry(self):
        # Two centroids 4 apart on f0. The bisector sits at f0 = 2.
        centroids = np.array([[0.0, 0.0], [4.0, 0.0]])
        Xz = np.array(
            [
                [0.0, 0.0],  # at centroid 0: half the separation away -> 2.0
                [1.0, 0.0],  # |2 - 1| = 1.0 from the boundary
                [2.0, 0.0],  # exactly on the boundary -> 0.0
            ]
        )
        res = ecn.flip_distances(Xz, centroids, metric="euclidean")
        np.testing.assert_allclose(res["flip_distance"], [2.0, 1.0, 0.0], atol=1e-9)
        # Class 0 nodes flip toward class 1; both points sit off-boundary toward 0.
        self.assertEqual(res["nearest"].tolist(), [0, 0, 0])
        self.assertEqual(res["runner_up"].tolist(), [1, 1, 1])

    def test_value_of_information_picks_the_separating_axis(self):
        # Centroids differ only on feature index 1 -> that is the decisive axis,
        # and a single-axis flip along it equals the full flip-distance.
        centroids = np.array([[0.0, 0.0, 0.0], [0.0, 6.0, 0.0]])
        Xz = np.array([[0.0, 1.0, 0.0]])
        res = ecn.flip_distances(Xz, centroids, metric="euclidean")
        self.assertEqual(res["most_relevant_feature"][0], 1)
        self.assertAlmostEqual(res["flip_distance"][0], res["single_axis_flip_distance"][0])
        # The decisive feature carries all the leverage; the inert axes carry none.
        leverage = res["leverage"][0]
        self.assertGreater(leverage[1], 0.0)
        self.assertAlmostEqual(leverage[0], 0.0)
        self.assertAlmostEqual(leverage[2], 0.0)

    def test_mahalanobis_with_identity_cov_matches_euclidean(self):
        centroids = np.array([[0.0, 0.0], [4.0, 0.0]])
        Xz = np.array([[1.0, 0.0], [3.5, 0.0]])
        euc = ecn.flip_distances(Xz, centroids, metric="euclidean")
        mah = ecn.flip_distances(Xz, centroids, metric="mahalanobis", inv_cov=np.eye(2))
        np.testing.assert_allclose(euc["flip_distance"], mah["flip_distance"], atol=1e-9)

    def test_nearest_boundary_binds_not_nearest_centroid(self):
        # Three classes; the binding flip is to whichever boundary is closest,
        # which need not be the second-nearest centroid. Check against brute force.
        centroids = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 3.0]])
        Xz = np.array([[1.0, 0.5]])
        res = ecn.flip_distances(Xz, centroids, metric="euclidean")
        d = np.linalg.norm(Xz[0][None, :] - centroids, axis=1)
        a = int(d.argmin())
        expected = min(
            (d[k] ** 2 - d[a] ** 2) / (2 * np.linalg.norm(centroids[k] - centroids[a]))
            for k in range(3)
            if k != a
        )
        self.assertAlmostEqual(res["flip_distance"][0], expected)

    def test_flip_distance_is_never_negative(self):
        rng = np.random.default_rng(7)
        centroids = rng.normal(size=(4, 5))
        Xz = rng.normal(size=(50, 5))
        res = ecn.flip_distances(Xz, centroids, metric="euclidean")
        self.assertTrue((res["flip_distance"] >= 0).all())

    def test_contestability_flags_the_fragile_decile_and_carries_caveats(self):
        # Labeled blobs plus one point parked on the boundary: it must be flagged.
        X = pd.DataFrame(
            {"f1": [0.0, 0.1, 0.2, 5.0, 5.1, 5.2, 2.5], "f2": [0.0] * 7},
            index=[f"n{i}" for i in range(7)],
        )
        y = pd.Series(["a", "a", "a", "b", "b", "b", "a"], index=X.index, name="Outcome")
        result = ecn.contestability(X, y=y, contest_quantile=0.2)
        frame = result["assignments"].set_index("ID")
        # The midpoint node has the smallest flip-distance and is flagged contested.
        self.assertEqual(frame["flip_distance"].idxmin(), "n6")
        self.assertTrue(bool(frame.loc["n6", "contested"]))
        summary = result["summary"]
        self.assertEqual(summary["flip_distance"]["n_contested"], result["assignments"]["contested"].sum())
        self.assertEqual(len(summary["caveats"]), 2)
        self.assertIn("measurement error", summary["caveats"][0])

    def test_contestability_scores_blank_scaffold_without_na_error(self):
        # Scaffold nodes carry a blank outcome; they get scored, but agreement is
        # only defined for labeled nodes (regression: pd.NA == str raised before).
        X = pd.DataFrame(
            {"f1": [0.0, 0.2, 5.0, 5.2, 2.6], "f2": [0.0, 0.1, 5.0, 4.9, 2.5]},
            index=["a1", "a2", "b1", "b2", "scaffold"],
        )
        y = pd.Series(["a", "a", "b", "b", ""], index=X.index, name="Outcome")
        result = ecn.contestability(X, y=y)
        frame = result["assignments"].set_index("ID")
        self.assertTrue(np.isfinite(frame.loc["scaffold", "flip_distance"]))
        self.assertIsNone(frame.loc["scaffold", "nearest_matches_outcome"])
        # Labeled nodes still get a real agreement boolean.
        self.assertIn(frame.loc["a1", "nearest_matches_outcome"], (True, False))

    def test_contestability_requires_two_classes(self):
        X = pd.DataFrame({"f1": [0.0, 1.0, 2.0]}, index=list("abc"))
        y = pd.Series(["a", "a", "a"], index=X.index, name="Outcome")
        with self.assertRaises(ValueError):
            ecn.contestability(X, y=y)

    def test_contestability_report_is_markdown_with_tables(self):
        X = pd.DataFrame(
            {"f1": [0.0, 0.2, 5.0, 5.2, 2.5], "f2": [0.0, 0.1, 5.0, 4.9, 2.4]},
            index=["a1", "a2", "b1", "b2", "mid"],
        )
        y = pd.Series(["a", "a", "b", "b", "a"], index=X.index, name="Outcome")
        result = ecn.contestability(X, y=y)
        report = ecn.contestability_report(result["assignments"], result["summary"])
        self.assertIn("# Contestability report", report)
        self.assertIn("## Most contested cases", report)
        self.assertIn("## Value of information", report)
        self.assertIn("## Caveats", report)
        self.assertIn("| case | call | would flip to | flip-distance | decisive feature |", report)
        # The most-contested case (the midpoint) heads the table.
        body = report.split("## Most contested cases", 1)[1]
        self.assertIn("mid", body.split("## Value of information", 1)[0])

    def test_run_contestability_writes_markdown_report(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            X = pd.DataFrame(
                {"f1": [0.0, 0.3, 5.0, 5.3, 2.5], "f2": [0.0, 0.2, 5.0, 4.8, 2.5]},
                index=[f"n{i}" for i in range(5)],
            )
            y = pd.Series(["a", "a", "b", "b", "a"], index=X.index, name="Outcome")
            ecn.run_contestability(X, out, y=y)
            self.assertTrue((out / "contestability_report.md").exists())
            self.assertIn("Contestability report", (out / "contestability_report.md").read_text())

    def test_plot_contestability_writes_panel(self):
        with tempfile.TemporaryDirectory() as td:
            X = pd.DataFrame(
                {"f1": list(np.linspace(0, 5, 10)), "f2": list(np.linspace(0, 5, 10))},
                index=[f"n{i}" for i in range(10)],
            )
            y = pd.Series(["a"] * 5 + ["b"] * 5, index=X.index, name="Outcome")
            result = ecn.contestability(X, y=y)
            path = ev.plot_contestability(
                result["assignments"], result["summary"], Path(td) / "contestability.png"
            )
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)

    def test_cli_run_writes_contestability_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "out"
            args = ToolkitTests._synthetic_run_args(
                self, root, out, run_clusters=False, run_contest=True, contest_quantile=0.1,
            )
            summary = et.run(args)
            self.assertIn("contestability", summary)
            self.assertTrue((out / "node_contestability.csv").exists())
            self.assertTrue((out / "contest_summary.json").exists())
            self.assertTrue((out / "contestability_report.md").exists())
            frame = pd.read_csv(out / "node_contestability.csv")
            for column in ["flip_distance", "runner_up_class", "contested",
                           "most_decision_relevant_feature"]:
                self.assertIn(column, frame.columns)
            self.assertTrue((frame["flip_distance"] >= 0).all())


class ScientificStandardsTests(unittest.TestCase):
    """Discrimination + calibration metrics, honest uncertainty, provenance,
    permutation importance, shrinkage, learning curves, and the model card."""

    def _binary_cohort(self, n=40, seed=0):
        # A binary outcome with real-but-imperfect signal in `Value`, on a chain
        # graph so graph features are well-defined.
        rng = np.random.default_rng(seed)
        rows = [
            {"ID": f"N{i}", "Outcome": i % 2, "Value": float(rng.normal((i % 2) * 2.0, 1.0))}
            for i in range(n)
        ]
        nodes = pd.DataFrame(rows)
        edges = pd.DataFrame([{"SourceID": f"N{i}", "TargetID": f"N{i + 1}"} for i in range(n - 1)])
        graph = et.build_graph(nodes, edges)
        return nodes, et.generate_graph_features(graph)

    def test_discrimination_and_calibration_metrics_reported(self):
        nodes, features = self._binary_cohort()
        with tempfile.TemporaryDirectory() as td:
            result = et.train_outcome_model(
                nodes, features, id_column="ID", outcome_column="Outcome",
                output_dir=Path(td), n_iterations=3, n_bootstrap=200,
            )
            m = result["metrics"]
            for key in ["roc_auc", "average_precision", "balanced_accuracy", "mcc", "brier"]:
                self.assertIn(key, m)
                self.assertIsNotNone(m[key])
            self.assertIn("calibration", m)
            self.assertEqual(m["calibration"]["positive_class"], "1")
            # Importance is permutation-based, with impurity retained for reference.
            self.assertEqual(m["importance_kind"], "permutation")
            self.assertIn("impurity_importance", result["importance"].columns)
            self.assertIn("importance_std", result["importance"].columns)
            # The reliability-diagram payload is returned for one held-out split.
            self.assertIsNotNone(result["calibration"])
            self.assertEqual(len(result["calibration"]["proba_pos"]), m["test_rows"])

    def test_bootstrap_ci_is_within_split_interval(self):
        nodes, features = self._binary_cohort()
        with tempfile.TemporaryDirectory() as td:
            result = et.train_outcome_model(
                nodes, features, id_column="ID", outcome_column="Outcome",
                output_dir=Path(td), n_iterations=1, n_bootstrap=300,
            )
            ci = result["metrics"]["primary_split_bootstrap_ci"]
            self.assertEqual(ci["n_bootstrap"], 300)
            acc = ci["metrics"]["accuracy"]
            self.assertLessEqual(acc["lower"], acc["upper"])

    def test_iteration_summary_carries_uncertainty_caveat(self):
        nodes, features = self._binary_cohort()
        with tempfile.TemporaryDirectory() as td:
            result = et.train_outcome_model(
                nodes, features, id_column="ID", outcome_column="Outcome",
                output_dir=Path(td), n_iterations=3, n_bootstrap=0,
            )
            m = result["metrics"]
            self.assertIn("iteration_summary_note", m)
            self.assertIn("Nadeau", m["iteration_summary_note"])
            # n_bootstrap=0 disables the within-split CI.
            self.assertNotIn("primary_split_bootstrap_ci", m)

    def test_permutation_test_is_direction_aware_and_averaged(self):
        nodes, features = self._binary_cohort()
        with tempfile.TemporaryDirectory() as td:
            result = et.train_outcome_model(
                nodes, features, id_column="ID", outcome_column="Outcome",
                output_dir=Path(td), n_iterations=2, n_permutations=8, n_bootstrap=0,
            )
            perm = result["metrics"]["permutation_test"]
            self.assertEqual(perm["n_permutations"], 8)
            self.assertIn("multiplicity_note", perm)
            # One averaged row per permutation (not one per inner split).
            self.assertEqual(len(result["permutation_metrics"]), 8)
            for entry in perm["metrics"].values():
                self.assertGreater(entry["p_value"], 0.0)
                self.assertLessEqual(entry["p_value"], 1.0)
            # Brier (lower-is-better) is in the null comparison via its own tail.
            self.assertIn("brier", perm["metrics"])

    def test_small_cohort_emits_data_warnings(self):
        nodes, features = self._binary_cohort(n=12, seed=2)
        with tempfile.TemporaryDirectory() as td:
            result = et.train_outcome_model(
                nodes, features, id_column="ID", outcome_column="Outcome",
                output_dir=Path(td), n_iterations=1, n_bootstrap=0,
            )
            warnings = result["metrics"].get("data_warnings", [])
            self.assertTrue(any("Small cohort" in w for w in warnings))

    def test_provenance_is_captured(self):
        prov = ecommon.provenance([], seed=7)
        self.assertEqual(prov["random_seed"], 7)
        for key in ["epinet_version", "git", "python_version", "packages", "created_utc"]:
            self.assertIn(key, prov)
        self.assertIn("scikit-learn", prov["packages"])

    def test_sha256_file_is_deterministic_and_stamped(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "data.csv"
            p.write_text("a,b\n1,2\n")
            h1, h2 = ecommon.sha256_file(p), ecommon.sha256_file(p)
            self.assertEqual(h1, h2)
            self.assertEqual(len(h1), 64)
            prov = ecommon.provenance([p], seed=1)
            self.assertEqual(prov["input_sha256"][str(p)], h1)

    def test_run_writes_provenance_and_model_card(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "out"
            args = ToolkitTests._synthetic_run_args(self, root, out, n_iterations=2)
            summary = et.run(args)
            self.assertIn("provenance", summary)
            self.assertTrue((out / "provenance.json").exists())
            self.assertTrue((out / "model_card.md").exists())
            card = (out / "model_card.md").read_text()
            for section in ["# Model card", "## Intended use", "## Performance",
                            "Calibration", "## Validation", "## Provenance"]:
                self.assertIn(section, card)

    def test_ledoit_wolf_precision_is_well_conditioned(self):
        # Near-collinear features make the naive covariance ill-conditioned;
        # shrinkage must still return a finite, symmetric precision matrix.
        rng = np.random.default_rng(0)
        base = rng.normal(size=(8, 1))
        Xz = np.hstack([base, base + 1e-6 * rng.normal(size=(8, 1)), rng.normal(size=(8, 1))])
        prec = ec._mahalanobis_inverse_cov(Xz)
        self.assertEqual(prec.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(prec)))
        np.testing.assert_allclose(prec, prec.T, atol=1e-8)

    def test_learning_curve_payload_present(self):
        nodes, features = self._binary_cohort(n=40)
        with tempfile.TemporaryDirectory() as td:
            result = et.train_outcome_model(
                nodes, features, id_column="ID", outcome_column="Outcome",
                output_dir=Path(td), n_iterations=1, n_bootstrap=0,
            )
            lc = result["learning_curve"]
            self.assertIsNotNone(lc)
            self.assertEqual(len(lc["train_sizes"]), len(lc["test_mean"]))

    def test_calibration_slope_intercept_handles_degenerate(self):
        # Single outcome class -> slope/intercept undefined (None), no crash.
        res = et.calibration_slope_intercept(pd.Series([1, 1, 1]), np.array([0.2, 0.6, 0.8]), 1)
        self.assertIsNone(res["slope"])
        # Two classes with separating probabilities -> a finite slope.
        res2 = et.calibration_slope_intercept(
            pd.Series([0, 0, 1, 1]), np.array([0.1, 0.3, 0.7, 0.9]), 1)
        self.assertIsNotNone(res2["slope"])

    def test_calibration_and_learning_curve_plots_render(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            ev.plot_calibration(
                np.array([0, 0, 1, 1, 0, 1]),
                np.array([0.1, 0.3, 0.7, 0.9, 0.2, 0.8]),
                out / "cal.png", pos_label=1, brier=0.1,
            )
            ev.plot_learning_curve(
                {"train_sizes": [4, 8, 12], "train_mean": [0.9, 0.92, 0.95],
                 "train_std": [0.02, 0.02, 0.01], "test_mean": [0.6, 0.7, 0.75],
                 "test_std": [0.05, 0.04, 0.03]},
                out / "lc.png",
            )
            self.assertGreater((out / "cal.png").stat().st_size, 0)
            self.assertGreater((out / "lc.png").stat().st_size, 0)


class FederatedFitTests(unittest.TestCase):
    """Aggregates-only federation reconstructs the centralized scaler + centroids."""

    def _labeled_matrix(self, n=40, d=4, seed=0):
        rng = np.random.default_rng(seed)
        X = pd.DataFrame(
            rng.normal(size=(n, d)),
            columns=[f"f{i}" for i in range(d)],
            index=[f"n{i}" for i in range(n)],
        )
        y = pd.Series(["x", "y"] * (n // 2), index=X.index, name="Outcome")
        return X, y

    def test_two_site_fit_matches_centralized(self):
        X, y = self._labeled_matrix()
        sites = np.where(np.arange(len(X)) % 2 == 0, "A", "B")
        res = efed.simulate(X, y, sites)
        # Federated == centralized to floating-point precision.
        self.assertLess(res["max_mean_diff"], 1e-9)
        self.assertLess(res["max_sd_diff"], 1e-9)
        self.assertLess(res["max_centroid_diff"], 1e-9)
        self.assertEqual(res["n_total"], len(X))

    def test_three_way_split_still_matches(self):
        X, y = self._labeled_matrix(n=60, seed=2)
        sites = np.array([f"S{i % 3}" for i in range(len(X))])
        res = efed.simulate(X, y, sites)
        self.assertEqual(len(res["sites"]), 3)
        self.assertLess(res["max_centroid_diff"], 1e-9)

    def test_aggregate_message_is_only_counts_and_sums(self):
        X, y = self._labeled_matrix(n=10)
        agg = efed.site_aggregates(X, y)
        # The message carries no per-row data — only counts and summed vectors.
        self.assertEqual(
            set(agg), {"columns", "n", "sum", "sumsq", "class_n", "class_sum", "suppressed"}
        )
        self.assertEqual(len(agg["sum"]), X.shape[1])
        self.assertEqual(agg["n"], len(X))
        self.assertEqual(sum(agg["class_n"].values()), len(X))

    def test_small_cell_suppression_drops_rare_class(self):
        X = pd.DataFrame(
            {"f1": [0.0, 1, 2, 3, 4, 5], "f2": [1.0, 1, 0, 0, 1, 2]},
            index=[f"n{i}" for i in range(6)],
        )
        y = pd.Series(["a", "a", "b", "b", "a", "rare"], index=X.index, name="Outcome")
        agg = efed.site_aggregates(X, y, min_cell=2)
        self.assertIn("rare", agg["suppressed"])
        self.assertNotIn("rare", agg["class_n"])
        combined = efed.combine_aggregates([agg])
        self.assertNotIn("rare", combined["classes"])
        self.assertIn("a", combined["classes"])

    def test_contract_mismatch_raises(self):
        a = efed.site_aggregates(pd.DataFrame({"f": [1.0, 2.0]}), pd.Series(["a", "b"]))
        b = efed.site_aggregates(pd.DataFrame({"g": [1.0, 2.0]}), pd.Series(["a", "b"]))
        with self.assertRaises(ValueError):
            efed.combine_aggregates([a, b])

    def test_bundled_synthetic_cohort_federates_exactly(self):
        repo = Path(__file__).resolve().parents[1]
        nodes, edges = et.load_tables(
            str(repo / "synthetic_nodes.csv"), str(repo / "synthetic_edges.csv")
        )
        graph = et.build_graph(nodes, edges)
        feats = et.generate_graph_features(graph)
        X = et.build_design_matrix(feats, nodes, id_column="ID", outcome_column="Outcome")
        y = nodes.assign(ID=nodes["ID"].astype(str)).set_index("ID")["Outcome"].reindex(X.index)
        sites = np.where(np.arange(len(X)) % 2 == 0, "A", "B")
        res = efed.simulate(X, y, sites)
        self.assertLess(res["max_centroid_diff"], 1e-9)
        self.assertLess(res["max_mean_diff"], 1e-9)


class IngestNormalizationTests(unittest.TestCase):
    """Front-end column-alias normalization (epinet_ingest)."""

    def test_aliases_are_resolved_to_canonical_schema(self):
        import epinet_ingest as ein

        nodes = pd.DataFrame([{"patient_id": "p1", "label": 1}, {"patient_id": "p2", "label": 0}])
        edges = pd.DataFrame([{"from": "p1", "to": "p2"}])
        out_nodes, out_edges, report = ein.normalize_tables(
            nodes, edges, id_column="ID", source_column="SourceID",
            target_column="TargetID", outcome_column="Outcome",
        )
        self.assertIn("ID", out_nodes.columns)
        self.assertIn("Outcome", out_nodes.columns)
        self.assertIn("SourceID", out_edges.columns)
        self.assertIn("TargetID", out_edges.columns)
        self.assertEqual(report["n_operations"], 4)
        # Inputs are not mutated.
        self.assertIn("patient_id", nodes.columns)

    def test_canonical_input_is_a_noop_but_still_hashed(self):
        import epinet_ingest as ein

        nodes = pd.DataFrame([{"ID": "a", "Outcome": 1}])
        edges = pd.DataFrame([{"SourceID": "a", "TargetID": "a"}])
        _, _, report = ein.normalize_tables(
            nodes, edges, id_column="ID", source_column="SourceID",
            target_column="TargetID", outcome_column="Outcome",
        )
        self.assertEqual(report["n_operations"], 0)
        self.assertEqual(len(report["normalized_nodes_sha256"]), 64)

    def test_sha256_frame_is_deterministic(self):
        df = pd.DataFrame([{"a": 1, "b": 2}])
        self.assertEqual(ecommon.sha256_frame(df), ecommon.sha256_frame(df.copy()))

    def test_run_normalizes_aliased_input_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "out"
            nodes = root / "nodes.csv"
            edges = root / "edges.csv"
            # Aliased columns: from/to and patient_id/label.
            nodes.write_text("patient_id,label\nA,1\nB,0\nC,1\nD,0\n")
            edges.write_text("from,to\nA,B\nB,C\nC,D\n")
            args = Namespace(
                nodes=str(nodes), edges=str(edges), output_dir=str(out),
                id_column="ID", source_column="SourceID", target_column="TargetID",
                outcome_column="Outcome", target_outcome="1", source_nodes="",
                target_nodes="", weight_column=None, use_weighted_paths=False,
                path_mode="hops", directed=False, include_centrality=False,
                run_model=False, run_paths=True, test_size=0.2, random_state=42,
            )
            summary = et.run(args)
            self.assertEqual(summary["graph"]["nodes"], 4)
            self.assertTrue((out / "ingest_report.json").exists())
            ops = summary["provenance"]["normalization"]["operations"]
            self.assertEqual(summary["provenance"]["normalization"]["n_operations"], 4)
            roles = {op["role"] for op in ops}
            self.assertEqual(roles, {"node_id", "outcome", "edge_source", "edge_target"})

    def test_no_normalize_flag_keeps_strict_validation(self):
        import epinet_ingest as ein  # noqa: F401  (module exists; flag bypasses it)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            nodes = root / "nodes.csv"
            edges = root / "edges.csv"
            nodes.write_text("patient_id,label\nA,1\nB,0\n")
            edges.write_text("from,to\nA,B\n")
            args = Namespace(
                nodes=str(nodes), edges=str(edges), output_dir=str(root / "out"),
                id_column="ID", source_column="SourceID", target_column="TargetID",
                outcome_column="Outcome", target_outcome="1", source_nodes="",
                target_nodes="", weight_column=None, use_weighted_paths=False,
                path_mode="hops", directed=False, include_centrality=False,
                run_model=False, run_paths=True, test_size=0.2, random_state=42,
                normalize=False,
            )
            # Strict mode: aliased columns are not recognized -> clear error.
            with self.assertRaises(ValueError):
                et.run(args)


try:
    from hypothesis import given, settings
    from hypothesis import strategies as st
    from hypothesis.extra import numpy as hnp

    _HAS_HYPOTHESIS = True
except ImportError:
    _HAS_HYPOTHESIS = False

if _HAS_HYPOTHESIS:
    _finite = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)

    class FlipDistancePropertyTests(unittest.TestCase):
        """Property-based invariants for the closed-form flip-distance."""

        @settings(max_examples=50, deadline=None)
        @given(
            hnp.arrays(np.float64, (3, 2), elements=_finite),
            hnp.arrays(np.float64, (6, 2), elements=_finite),
        )
        def test_flip_distance_is_never_negative(self, centroids, Xz):
            res = ecn.flip_distances(Xz, centroids, metric="euclidean")
            self.assertTrue(np.all(res["flip_distance"] >= -1e-9))

        @settings(max_examples=50, deadline=None)
        @given(
            hnp.arrays(np.float64, (3, 3), elements=_finite),
            hnp.arrays(np.float64, (5, 3), elements=_finite),
        )
        def test_single_axis_flip_at_least_full_flip_distance(self, centroids, Xz):
            # Moving one feature can never settle a call more cheaply than the
            # unconstrained shortest move to the boundary.
            res = ecn.flip_distances(Xz, centroids, metric="euclidean")
            finite = np.isfinite(res["single_axis_flip_distance"]) & np.isfinite(res["flip_distance"])
            self.assertTrue(
                np.all(res["single_axis_flip_distance"][finite] >= res["flip_distance"][finite] - 1e-6)
            )


if __name__ == "__main__":
    unittest.main()
