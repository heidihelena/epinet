import json
import sys
import tempfile
import unittest
from argparse import Namespace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from epinet import baselines as eb
from epinet import cluster as ec
from epinet import common as ecommon
from epinet import contest as ecn
from epinet import federated as efed
from epinet import governance as eg
from epinet import report as epinet_report
from epinet import toolkit as et
from epinet import validation as exv
from epinet import viz as ev


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

    def test_outcome_confounded_community_split_is_refused(self):
        # Outcome identical to community: any group split leaves the training
        # fold without one class entirely. The guard must refuse with an
        # explanation rather than score a model against an unseen class.
        labels = et.community_labels(self._two_cluster_graph())
        X = pd.DataFrame({"x": range(len(labels))}, index=labels.index)
        y = labels.astype(str)  # class == community, i.e. fully confounded
        with self.assertRaisesRegex(ValueError, "outcome-confounded"):
            et._split_indices(
                X, y, test_size=0.5, random_state=0, stratify_ok=False, groups=labels
            )

    def test_random_split_missing_class_is_refused(self):
        # Unstratified random split where one class is so rare it can land
        # entirely in the test fold; the guard refuses that too.
        X = pd.DataFrame({"x": range(8)})
        y = pd.Series(["a"] * 7 + ["b"])
        refused = False
        for seed in range(50):
            try:
                train_idx, _ = et._split_indices(
                    X, y, test_size=0.5, random_state=seed, stratify_ok=False
                )
            except ValueError as err:
                self.assertIn("missing outcome class", str(err))
                refused = True
            else:
                self.assertEqual(set(y.iloc[train_idx].unique()), {"a", "b"})
        self.assertTrue(refused, "expected at least one seed to drop the rare class")

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

    def test_single_axis_flip_distance_is_metric_correct_under_mahalanobis(self):
        # The single-axis flip distance must be the *metric* length of the move to
        # the binding boundary, i.e. include the sqrt(M_jj) factor. Verify against
        # a direct linear-boundary computation for the (nearest, runner_up) pair.
        rng = np.random.default_rng(7)
        for _ in range(5):
            a = rng.normal(size=(3, 3))
            M = a @ a.T + 0.5 * np.eye(3)  # SPD precision
            cents = rng.normal(size=(3, 3))
            x = rng.normal(size=3)
            res = ecn.flip_distances(x.reshape(1, -1), cents, metric="mahalanobis", inv_cov=M)
            ci, ri = int(res["nearest"][0]), int(res["runner_up"][0])
            j = int(res["most_relevant_feature"][0])

            def margin(p):
                return (p - cents[ri]) @ M @ (p - cents[ri]) - (p - cents[ci]) @ M @ (p - cents[ci])

            e = np.zeros(3); e[j] = 1.0
            m0 = margin(x)
            dm = (margin(x + 1e-6 * e) - m0) / 1e-6
            true_len = abs(-m0 / dm) * np.sqrt(M[j, j])  # |t| * sqrt(M_jj)
            self.assertAlmostEqual(float(res["single_axis_flip_distance"][0]), true_len, places=5)

    def test_contestability_refuses_nan_features(self):
        X = pd.DataFrame(
            {"f1": [0.0, 1.0, np.nan, 5.0, 5.1], "f2": [0.0, 0.1, 0.2, 5.0, 5.2]},
            index=[f"n{i}" for i in range(5)],
        )
        y = pd.Series(["a", "a", "a", "b", "b"], index=X.index)
        with self.assertRaises(ValueError):
            ecn.contestability(X, y=y, metric="euclidean")

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

    def _multiclass_cohort(self, n=60, seed=0):
        rng = np.random.default_rng(seed)
        rows = [
            {"ID": f"N{i}", "Outcome": ["a", "b", "c"][i % 3],
             "Value": float(rng.normal((i % 3) * 3.0, 1.0))}
            for i in range(n)
        ]
        nodes = pd.DataFrame(rows)
        edges = pd.DataFrame([{"SourceID": f"N{i}", "TargetID": f"N{i + 1}"} for i in range(n - 1)])
        graph = et.build_graph(nodes, edges)
        return nodes, et.generate_graph_features(graph)

    def _imbalanced_cohort(self, n=80, minority=16, seed=0):
        # A skewed binary outcome (minority count fixed for a deterministic split)
        # with real-but-imperfect signal in `Value`, on a chain graph.
        rng = np.random.default_rng(seed)
        rows = [
            {"ID": f"N{i}", "Outcome": 1 if i < minority else 0,
             "Value": float(rng.normal((1 if i < minority else 0) * 2.0, 1.0))}
            for i in range(n)
        ]
        nodes = pd.DataFrame(rows)
        edges = pd.DataFrame([{"SourceID": f"N{i}", "TargetID": f"N{i + 1}"} for i in range(n - 1)])
        graph = et.build_graph(nodes, edges)
        return nodes, et.generate_graph_features(graph)

    def test_tuning_is_imbalance_aware(self):
        # The hyperparameter search must expose class_weight so a skewed outcome
        # can be weighted rather than collapsing to the majority class.
        nodes, features = self._imbalanced_cohort()
        with tempfile.TemporaryDirectory() as td:
            result = et.train_outcome_model(
                nodes, features, id_column="ID", outcome_column="Outcome",
                output_dir=Path(td), n_iterations=1,
            )
            best = result["metrics"]["best_params"]
            self.assertIn("class_weight", best)
            self.assertIn(best["class_weight"], [None, "balanced", "balanced_subsample"])
            # The grid also regularizes via min_samples_leaf.
            self.assertIn("min_samples_leaf", best)
            self.assertIn(best["min_samples_leaf"], [1, 3])

    def test_multiclass_calibration_block_is_present_and_honest(self):
        # Calibration must not be silently absent for multiclass: Brier is
        # reported, slope/intercept are explicitly None with a stated reason.
        nodes, features = self._multiclass_cohort()
        with tempfile.TemporaryDirectory() as td:
            result = et.train_outcome_model(
                nodes, features, id_column="ID", outcome_column="Outcome",
                output_dir=Path(td), n_iterations=2,
            )
            m = result["metrics"]
            self.assertGreater(m["n_classes"], 2)
            self.assertIn("calibration", m)                      # not silently dropped
            self.assertIsNotNone(m["calibration"]["brier"])      # Brier still reported
            self.assertIsNone(m["calibration"]["slope"])         # binary-only, explicit
            self.assertIsNone(m["calibration"]["intercept"])
            self.assertIn("binary outcomes only", m["calibration"]["note"])
            # The model card surfaces it without empty slope/intercept rows.
            card = epinet_report.model_card(m)
            self.assertIn("AUROC (macro OvR)", card)
            self.assertNotIn("Calibration slope (ideal 1)", card)

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

    def test_probability_calibration_is_reported_and_safe(self):
        # Binary outcomes get a Platt-scaling recalibration block. Calibrated
        # probabilities are ADOPTED only when they do not worsen held-out Brier,
        # and the reported calibration Brier must reflect that decision.
        nodes, features = self._binary_cohort()
        with tempfile.TemporaryDirectory() as td:
            result = et.train_outcome_model(
                nodes, features, id_column="ID", outcome_column="Outcome",
                output_dir=Path(td), n_iterations=1, n_bootstrap=0,
            )
            cal = result["metrics"]["calibration"]
            self.assertIn("recalibration", cal)
            rc = cal["recalibration"]
            self.assertEqual(rc["method"], "sigmoid")
            self.assertIn(rc["adopted"], (True, False))
            # The adoption gate: calibration is taken only when Brier does not
            # increase, and the headline Brier matches the adopted probabilities.
            if rc["adopted"]:
                self.assertLessEqual(rc["brier_calibrated"], rc["brier_raw"])
                self.assertAlmostEqual(cal["brier"], rc["brier_calibrated"])
            else:
                self.assertAlmostEqual(cal["brier"], rc["brier_raw"])

    def test_threshold_tuning_is_opt_in_and_does_not_touch_probabilities(self):
        # Decision-threshold tuning is OFF by default (no block, default argmax
        # predictions). When ON, it reports a threshold in [0, 1] with the
        # 0.5-vs-tuned held-out comparison, and — being a labels-only change —
        # must leave the probability metric (AUROC) untouched.
        nodes, features = self._imbalanced_cohort()
        with tempfile.TemporaryDirectory() as td:
            base = et.train_outcome_model(
                nodes, features, id_column="ID", outcome_column="Outcome",
                output_dir=Path(td), n_iterations=1, n_bootstrap=0,
            )
        with tempfile.TemporaryDirectory() as td:
            tuned = et.train_outcome_model(
                nodes, features, id_column="ID", outcome_column="Outcome",
                output_dir=Path(td), n_iterations=1, n_bootstrap=0,
                tune_threshold=True,
            )
        self.assertNotIn("threshold_tuning", base["metrics"])
        tt = tuned["metrics"]["threshold_tuning"]
        self.assertGreaterEqual(tt["threshold"], 0.0)
        self.assertLessEqual(tt["threshold"], 1.0)
        self.assertIn("balanced_accuracy_at_0.5", tt)
        self.assertIn("balanced_accuracy_tuned", tt)
        # Thresholding relabels probabilities; it must not change the ranking, so
        # AUROC is identical with and without tuning.
        self.assertAlmostEqual(base["metrics"]["roc_auc"], tuned["metrics"]["roc_auc"])

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
        # The message carries no per-row data — only counts, sums, and centered
        # moments (mean-subtracted second moments for a stable variance/covariance).
        self.assertEqual(
            set(agg),
            {"columns", "n", "sum", "mean", "m2", "comoment", "class_n", "class_sum", "suppressed"},
        )
        self.assertEqual(len(agg["sum"]), X.shape[1])
        self.assertEqual(len(agg["m2"]), X.shape[1])
        self.assertEqual(np.asarray(agg["comoment"]).shape, (X.shape[1], X.shape[1]))
        self.assertEqual(agg["n"], len(X))
        self.assertEqual(sum(agg["class_n"].values()), len(X))

    def test_federated_fit_is_stable_under_large_offset_feature(self):
        # A feature whose mean dwarfs its spread is the classic catastrophic-
        # cancellation case for sumsq/n - mean**2. The centered-moment fit must
        # still reconstruct the centralized scaler/centroids to ~fp precision.
        rng = np.random.default_rng(7)
        n = 400
        X = pd.DataFrame({
            "big": 1e6 + rng.normal(0.0, 1.0, size=n),   # offset >> spread
            "ok": rng.normal(0.0, 1.0, size=n),
        }, index=[f"n{i}" for i in range(n)])
        y = pd.Series((rng.random(n) < 0.5).astype(int), index=X.index, name="Outcome")
        sites = np.array([f"S{i % 4}" for i in range(n)])
        res = efed.simulate(X, y, sites)
        # Naive sumsq reconstruction drifted ~1e-4 on sd here; centered moments
        # bring it back to floating-point agreement with the centralized fit.
        self.assertLess(res["max_sd_diff"], 1e-9)
        self.assertLess(res["max_mean_diff"], 1e-6)
        self.assertLess(res["max_centroid_diff"], 1e-9)

    def test_covariance_shrinkage_conditions_inverse_and_is_opt_in(self):
        # Near-collinear features make the standardized covariance ill-conditioned,
        # so the Mahalanobis precision (inv_cov) blows up. Opt-in shrinkage toward
        # the identity tames it; the default (0.0) leaves the empirical cov intact.
        rng = np.random.default_rng(1)
        n = 200
        a = rng.normal(0.0, 1.0, n)
        X = pd.DataFrame({
            "a": a,
            "b": a + rng.normal(0.0, 1e-3, n),   # ~collinear with a
            "c": rng.normal(0.0, 1.0, n),
        }, index=[f"n{i}" for i in range(n)])
        y = pd.Series((rng.random(n) < 0.5).astype(int), index=X.index, name="Outcome")
        aggs = [efed.site_aggregates(X.iloc[i::4], y.iloc[i::4]) for i in range(4)]

        plain = efed.combine_aggregates(aggs)
        shrunk = efed.combine_aggregates(aggs, shrinkage=0.1)

        self.assertEqual(plain["shrinkage"], 0.0)
        self.assertEqual(shrunk["shrinkage"], 0.1)
        # Shrinkage strictly improves conditioning of the precision matrix.
        self.assertLess(
            np.linalg.cond(shrunk["inv_cov"]), np.linalg.cond(plain["inv_cov"])
        )
        # The reported empirical covariance is unchanged — shrinkage only feeds inv_cov.
        np.testing.assert_allclose(plain["cov_standardized"], shrunk["cov_standardized"])
        # Out-of-range intensity is rejected.
        with self.assertRaises(ValueError):
            efed.combine_aggregates(aggs, shrinkage=1.5)

    def test_non_finite_features_are_rejected_with_named_columns(self):
        # A NaN/inf would silently poison sum/m2/comoment and corrupt the fit, so
        # site_aggregates must refuse it and name the offending column(s).
        X = pd.DataFrame(
            {"good": [1.0, 2.0, 3.0], "bad": [0.0, np.nan, 1.0]},
            index=["a", "b", "c"],
        )
        y = pd.Series(["x", "y", "x"], index=X.index, name="Outcome")
        with self.assertRaises(ValueError) as ctx:
            efed.site_aggregates(X, y)
        self.assertIn("bad", str(ctx.exception))
        self.assertNotIn("good", str(ctx.exception))
        # An inf is caught the same way.
        X.loc["b", "bad"] = np.inf
        with self.assertRaises(ValueError):
            efed.site_aggregates(X, y)

    def test_combine_rejects_all_empty_aggregates(self):
        # Empty sites must raise rather than divide by zero into NaN stats.
        empty = pd.DataFrame({"f": []}, dtype=float)
        ey = pd.Series([], dtype="object", name="Outcome")
        agg = efed.site_aggregates(empty, ey)
        self.assertEqual(agg["n"], 0)
        with self.assertRaises(ValueError):
            efed.combine_aggregates([agg])

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


class FederatedContestabilityTests(unittest.TestCase):
    """Flip-distance scores federate exactly; only de-identified summaries cross."""

    def _design(self):
        repo = Path(__file__).resolve().parents[1]
        nodes, edges = et.load_tables(
            str(repo / "synthetic_nodes.csv"), str(repo / "synthetic_edges.csv")
        )
        graph = et.build_graph(nodes, edges)
        feats = et.generate_graph_features(graph)
        X = et.build_design_matrix(feats, nodes, id_column="ID", outcome_column="Outcome")
        y = nodes.assign(ID=nodes["ID"].astype(str)).set_index("ID")["Outcome"].reindex(X.index)
        sites = pd.Series(np.where(np.arange(len(X)) % 2 == 0, "A", "B"), index=X.index)
        return X, y, sites

    def _fit(self, X, y, sites):
        aggs = [
            efed.site_aggregates(X.loc[sites == s], y.loc[sites == s])
            for s in sorted(sites.unique())
        ]
        return efed.combine_aggregates(aggs)

    def test_scores_federate_exactly(self):
        X, y, sites = self._design()
        fit = self._fit(X, y, sites)
        local = {}
        for s in sorted(sites.unique()):
            rows = sites == s
            local.update(dict(zip(X.loc[rows].index, efed.local_flip_distances(X.loc[rows], fit))))
        central = ecn.contestability(X, y=y, metric="euclidean")["assignments"].set_index("ID")
        fed_flip = pd.Series(local).reindex(central.index).to_numpy()
        diff = float(np.max(np.abs(fed_flip - central["flip_distance"].to_numpy())))
        self.assertLess(diff, 1e-9)

    def test_federated_summary_matches_centralized(self):
        X, y, sites = self._design()
        fit = self._fit(X, y, sites)
        summaries = [
            efed.site_contestability(X.loc[sites == s], y.loc[sites == s], fit)
            for s in sorted(sites.unique())
        ]
        fed = efed.combine_contestability(summaries, contest_quantile=0.1)
        central = ecn.contestability(X, y=y, metric="euclidean", contest_quantile=0.1)["summary"]
        self.assertAlmostEqual(fed["flip_distance"]["mean"], central["flip_distance"]["mean"], places=9)
        self.assertAlmostEqual(fed["flip_distance"]["std"], central["flip_distance"]["std"], places=6)
        self.assertEqual(
            fed["runner_up_counts"],
            {str(k): int(v) for k, v in central["runner_up_counts"].items()},
        )
        # Same top value-of-information feature, and contested count ~= q*N.
        self.assertEqual(next(iter(fed["feature_voi"])), next(iter(central["feature_leverage"])))
        self.assertEqual(fed["flip_distance"]["approx_n_contested"], central["flip_distance"]["n_contested"])

    def test_simulate_contestability_matches_centralized(self):
        # The stage-2 analogue of simulate(): one call partitions, federates the
        # contestability summary, and compares to a centralized run.
        X, y, sites = self._design()
        res = efed.simulate_contestability(X, y, sites, contest_quantile=0.1)
        self.assertEqual(set(res["sites"]), {"A", "B"})
        self.assertLess(res["max_mean_diff"], 1e-9)
        self.assertLess(res["max_std_diff"], 1e-6)
        self.assertTrue(res["runner_up_match"])
        self.assertTrue(res["top_voi_match"])

    def test_site_summary_is_aggregate_only(self):
        X, y, _ = self._design()
        fit = self._fit(X, y, pd.Series("A", index=X.index))
        summary = efed.site_contestability(X, y, fit)
        self.assertEqual(
            set(summary),
            {"columns", "n", "flip_count", "flip_sum", "flip_sumsq", "flip_min", "flip_max",
             "flip_hist", "runner_up_counts", "leverage_sum", "leverage_n",
             "agree_count", "labeled_count"},
        )

    def test_mahalanobis_covariance_federates(self):
        # The pooled covariance reconstructs from per-site second moments, so the
        # Mahalanobis flip-distance matches a centralized EMPIRICAL-covariance run
        # (the unshrunk reference; production Ledoit-Wolf is a separate refinement).
        X, y, sites = self._design()
        fit = self._fit(X, y, sites)
        local = {}
        for s in sorted(sites.unique()):
            rows = sites == s
            fd = efed.local_flip_distances(X.loc[rows], fit, metric="mahalanobis")
            local.update(dict(zip(X.loc[rows].index, fd)))

        Xz, kept = ec.standardize(X)
        y_lab = y.reindex(X.index).mask(ecommon.blank_label_mask(y.reindex(X.index)))
        classes, centroids = ec.class_centroids(Xz, y_lab.reset_index(drop=True))
        ridge = 1e-6 * np.eye(Xz.shape[1])
        inv_cov_emp = np.linalg.pinv(np.cov(Xz, rowvar=False, bias=True) + ridge)
        central = ecn.flip_distances(Xz, centroids, metric="mahalanobis", inv_cov=inv_cov_emp)
        central_flip = pd.Series(central["flip_distance"], index=X.index)

        fed_flip = pd.Series(local).reindex(X.index).to_numpy()
        diff = float(np.max(np.abs(fed_flip - central_flip.to_numpy())))
        self.assertLess(diff, 1e-6)


class BaselineAndValidationTests(unittest.TestCase):
    """Representation baselines and external validation."""

    def _two_community_graph(self):
        # Two rings (communities) joined by one bridge; outcome = community.
        # Degree/clustering are ~constant, so graph-topology features carry little;
        # a node embedding that recovers the community structure should win.
        ids_a = [f"a{i}" for i in range(12)]
        ids_b = [f"b{i}" for i in range(12)]
        nodes = pd.DataFrame(
            [{"ID": i, "Outcome": 0} for i in ids_a]
            + [{"ID": i, "Outcome": 1} for i in ids_b]
        )

        def ring(ids):
            return [{"SourceID": ids[i], "TargetID": ids[(i + 1) % len(ids)]} for i in range(len(ids))]

        edges = pd.DataFrame(ring(ids_a) + ring(ids_b) + [{"SourceID": "a0", "TargetID": "b0"}])
        return nodes, edges

    def test_spectral_embedding_shape(self):
        nodes, edges = self._two_community_graph()
        graph = et.build_graph(nodes, edges)
        emb = eb.spectral_node_embeddings(graph, n_components=4, seed=0)
        self.assertEqual(emb.shape[0], 24)
        self.assertIn("ID", emb.columns)
        self.assertEqual(sum(c.startswith("spectral_") for c in emb.columns), 4)

    def test_compare_representations_floors_and_beats(self):
        nodes, edges = self._two_community_graph()
        result = eb.compare_representations(
            nodes, edges, n_components=4, n_iterations=3, random_state=0,
        )
        comp = result["comparison"].set_index("representation")
        self.assertIn("no_information", comp.index)
        self.assertIn("spectral_embedding", comp.index)
        # No-information is a chance-level floor; the embedding clears it.
        self.assertLessEqual(comp.loc["no_information", "roc_auc"], 0.7)
        self.assertGreater(comp.loc["spectral_embedding", "roc_auc"], 0.75)
        # Paired baseline margin: graph_features vs floor on the SAME splits.
        pb = result["paired_baseline"]
        self.assertEqual(pb["model_representation"], "graph_features")
        self.assertEqual(pb["n_pairs"], 3)
        self.assertLessEqual(pb["margin_ci_lower"], pb["margin_ci_upper"])
        self.assertIn("Nadeau", pb["correction"])

    def test_external_validation_reports_internal_external_drift(self):
        dev_nodes, dev_edges = self._two_community_graph()
        ext_nodes, ext_edges = self._two_community_graph()
        result = exv.external_validation(dev_nodes, dev_edges, ext_nodes, ext_edges, random_state=0)
        self.assertIn("roc_auc", result["external"])
        self.assertIn("roc_auc", result["internal"])
        self.assertIn("roc_auc", result["drift_internal_minus_external"])
        self.assertEqual(result["external"]["n_external"], 24)

    def test_external_validation_aligns_missing_columns(self):
        # External cohort built the same way still validates (column alignment path).
        dev_nodes, dev_edges = self._two_community_graph()
        ext_nodes, ext_edges = self._two_community_graph()
        with tempfile.TemporaryDirectory() as td:
            result = exv.external_validation(
                dev_nodes, dev_edges, ext_nodes, ext_edges, random_state=0, output_dir=Path(td))
            self.assertTrue((Path(td) / "external_validation.json").exists())
            self.assertIsNotNone(result["external"]["balanced_accuracy"])


class FederatedEgressTests(unittest.TestCase):
    """The egress gate is mandatory: contributions are sealed until disclosed."""

    NOW = date(2026, 6, 11)

    def _consent(self, **overrides):
        base = dict(
            site="A", controller="C", lawful_basis="GDPR Art 9(2)(j)",
            dpia_reference="D", purpose="research", version="v1",
            coi_acknowledged=True, expires="2027-01-01",
        )
        base.update(overrides)
        return eg.Consent(**base)

    def _data(self, n=40, seed=0):
        rng = np.random.default_rng(seed)
        X = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"f{i}" for i in range(4)],
                         index=[f"n{i}" for i in range(n)])
        y = pd.Series(["x", "y"] * (n // 2), index=X.index, name="Outcome")
        return X, y

    def test_contribution_is_sealed_and_unserializable(self):
        X, y = self._data()
        contrib = efed.contribute_aggregate(X, y)
        self.assertIsInstance(contrib, efed.SiteContribution)
        # Cannot be shipped directly — the only way out is .disclose().
        with self.assertRaises(TypeError):
            json.dumps(contrib)
        self.assertIn("disclose", repr(contrib))

    def test_disclose_runs_gate_and_combine_reconstructs(self):
        X, y = self._data()
        policy = eg.DisclosurePolicy(min_cell=2)
        sites = np.where(np.arange(len(X)) % 2 == 0, "A", "B")
        disclosed = []
        for s in ("A", "B"):
            rows = sites == s
            contrib = efed.contribute_aggregate(X.loc[rows], y.loc[rows])
            disclosed.append(contrib.disclose(policy=policy, consent=self._consent(), now=self.NOW))
        self.assertIsInstance(disclosed[0], efed.DisclosedContribution)
        self.assertIn("payload_sha256", disclosed[0].manifest)
        fit = efed.combine_aggregates(disclosed)
        self.assertEqual(set(fit["classes"]), {"x", "y"})
        self.assertEqual(fit["n_total"], len(X))

    def test_disclose_refused_without_valid_consent(self):
        X, y = self._data()
        contrib = efed.contribute_aggregate(X, y)
        with self.assertRaises(eg.GovernanceError):
            contrib.disclose(policy=eg.DisclosurePolicy(min_cell=2),
                             consent=self._consent(coi_acknowledged=False), now=self.NOW)

    def test_combine_accepts_disclosed_and_raw_equivalently(self):
        X, y = self._data()
        policy = eg.DisclosurePolicy(min_cell=2)  # no class is this small -> no suppression
        raw = [efed.site_aggregates(X.iloc[:20], y.iloc[:20]),
               efed.site_aggregates(X.iloc[20:], y.iloc[20:])]
        disclosed = [
            efed.contribute_aggregate(X.iloc[:20], y.iloc[:20]).disclose(
                policy=policy, consent=self._consent(), now=self.NOW),
            efed.contribute_aggregate(X.iloc[20:], y.iloc[20:]).disclose(
                policy=policy, consent=self._consent(), now=self.NOW),
        ]
        fit_raw = efed.combine_aggregates(raw)
        fit_disclosed = efed.combine_aggregates(disclosed)
        np.testing.assert_allclose(fit_raw["centroids"], fit_disclosed["centroids"], atol=1e-12)
        np.testing.assert_allclose(fit_raw["mean"], fit_disclosed["mean"], atol=1e-12)


class GovernanceGateTests(unittest.TestCase):
    """The egress gate fails closed and discloses exactly what crosses."""

    NOW = date(2026, 6, 11)

    def _consent(self, **overrides):
        base = dict(
            site="A", controller="A Trust", lawful_basis="GDPR Art 9(2)(j)",
            dpia_reference="DPIA-1", purpose="research", version="v1",
            allowed_tier="aggregate", coi_acknowledged=True, expires="2027-01-01",
        )
        base.update(overrides)
        return eg.Consent(**base)

    def _payload(self, common=40, rare=3):
        return {"n": common + rare, "class_n": {"common": common, "rare": rare},
                "class_sum": {"common": [1.0], "rare": [1.0]}}

    def test_valid_egress_returns_manifest_and_suppresses_small_cells(self):
        policy = eg.DisclosurePolicy(min_cell=5)
        redacted, manifest = eg.check_egress(
            self._payload(), policy=policy, consent=self._consent(), now=self.NOW)
        self.assertNotIn("rare", redacted["class_n"])     # suppressed (3 < 5)
        self.assertIn("common", redacted["class_n"])
        self.assertIn("class_n[rare]=3", manifest["suppressed_cells"])
        self.assertEqual(len(manifest["payload_sha256"]), 64)

    def test_extreme_flip_values_are_withheld_at_egress(self):
        # flip_min / flip_max are one node's exact score apiece (least/most
        # contestable patient). The gate must withhold them even when the cohort
        # clears the record floor, while distribution-shape stats still cross.
        policy = eg.DisclosurePolicy(min_cell=5)
        payload = {
            "n": 50, "flip_count": 50, "flip_sum": 100.0, "flip_sumsq": 250.0,
            "flip_min": 0.12, "flip_max": 7.84, "flip_hist": [10, 20, 20],
            "runner_up_counts": {"x": 30, "y": 20},
        }
        redacted, manifest = eg.check_egress(
            payload, policy=policy, consent=self._consent(), now=self.NOW)
        self.assertIsNone(redacted["flip_min"])
        self.assertIsNone(redacted["flip_max"])
        self.assertTrue(any("flip_min" in s for s in manifest["suppressed_cells"]))
        self.assertTrue(any("flip_max" in s for s in manifest["suppressed_cells"]))
        # Distribution shape/spread still cross — only the extremes are withheld.
        self.assertEqual(redacted["flip_sum"], 100.0)
        self.assertEqual(redacted["flip_hist"], [10, 20, 20])

    def test_suppression_survives_complementary_subtraction(self):
        # The suppressed cell must not be recoverable as total - sum(retained).
        policy = eg.DisclosurePolicy(min_cell=5)
        redacted, manifest = eg.check_egress(
            self._payload(common=40, rare=3), policy=policy,
            consent=self._consent(), now=self.NOW)
        retained = sum(redacted["class_n"].values())
        self.assertEqual(redacted["n"], retained)          # total reduced to retained sum
        self.assertEqual(manifest["record_count"], retained)
        self.assertEqual(redacted["n"] - retained, 0)      # nothing to subtract
        self.assertTrue(any("block subtraction" in s for s in manifest["suppressed_cells"]))

    def test_record_floor_uses_true_total_not_reduced(self):
        # A genuinely tiny aggregate is refused on its TRUE total, even though
        # secondary suppression would otherwise shrink the disclosed total.
        policy = eg.DisclosurePolicy(min_cell=5)
        with self.assertRaises(eg.GovernanceError):
            eg.check_egress({"n": 4, "class_n": {"a": 4}}, policy=policy,
                            consent=self._consent(), now=self.NOW)

    def test_malformed_expiry_fails_closed_with_governance_error(self):
        policy = eg.DisclosurePolicy(min_cell=5)
        with self.assertRaises(eg.GovernanceError):
            eg.check_egress(self._payload(), policy=policy,
                            consent=self._consent(expires="not-a-date"), now=self.NOW)

    def test_missing_required_field_is_refused(self):
        with self.assertRaises(eg.GovernanceError):
            eg.check_egress(self._payload(), policy=eg.DisclosurePolicy(),
                            consent=self._consent(dpia_reference=""), now=self.NOW)

    def test_coi_not_acknowledged_is_refused(self):
        with self.assertRaises(eg.GovernanceError):
            eg.check_egress(self._payload(), policy=eg.DisclosurePolicy(),
                            consent=self._consent(coi_acknowledged=False), now=self.NOW)

    def test_expired_consent_is_refused(self):
        with self.assertRaises(eg.GovernanceError):
            eg.check_egress(self._payload(), policy=eg.DisclosurePolicy(),
                            consent=self._consent(expires="2020-01-01"), now=self.NOW)

    def test_tier_above_allowance_is_refused(self):
        with self.assertRaises(eg.GovernanceError):
            eg.check_egress(self._payload(), policy=eg.DisclosurePolicy(allowed_tier="aggregate"),
                            consent=self._consent(allowed_tier="derived"), tier="derived",
                            now=self.NOW)

    def test_identifiable_tier_is_always_refused(self):
        with self.assertRaises(eg.GovernanceError):
            eg.check_egress(self._payload(),
                            policy=eg.DisclosurePolicy(allowed_tier="identifiable"),
                            consent=self._consent(allowed_tier="identifiable"),
                            tier="identifiable", now=self.NOW)

    def test_identifying_field_in_payload_is_refused(self):
        payload = {"n": 50, "columns": ["patient_id", "age"]}
        with self.assertRaises(eg.GovernanceError):
            eg.check_egress(payload, policy=eg.DisclosurePolicy(),
                            consent=self._consent(), now=self.NOW)

    def test_record_floor_is_enforced(self):
        # An aggregate over fewer than min_cell records is refused outright.
        with self.assertRaises(eg.GovernanceError):
            eg.check_egress({"n": 3, "class_n": {"a": 3}}, policy=eg.DisclosurePolicy(min_cell=5),
                            consent=self._consent(), now=self.NOW)

    def test_audit_chain_verifies_and_detects_tampering(self):
        audit = eg.AuditLedger()
        eg.check_egress(self._payload(), policy=eg.DisclosurePolicy(min_cell=5),
                        consent=self._consent(), audit=audit, now=self.NOW,
                        timestamp="2026-06-11T00:00:00+00:00")
        self.assertTrue(audit.verify())
        audit.entries[0]["event"]["manifest"]["record_count"] = 9999
        self.assertFalse(audit.verify())

    def test_suppress_small_cells_handles_runner_up_counts(self):
        payload = {"n_scored": 100, "runner_up_counts": {"a": 80, "b": 2}}
        redacted, suppressed = eg.suppress_small_cells(payload, min_cell=5)
        self.assertNotIn("b", redacted["runner_up_counts"])
        self.assertIn("a", redacted["runner_up_counts"])
        self.assertEqual(suppressed, ["runner_up_counts[b]=2"])


class RegistryAdapterTests(unittest.TestCase):
    """Flat registry export -> canonical EpiNet schema (epinet_registry)."""

    def _table(self, n=12):
        return pd.DataFrame({
            "case_id": [f"C{i}" for i in range(n)],
            "age": list(range(40, 40 + n)),
            "stage": [1, 2, 3, 4] * (n // 4),
            "site": (["North", "South"] * (n // 2)),  # non-numeric
            "status": (["alive", "deceased"] * (n // 2)),
        })

    def test_adapts_flat_table_to_canonical_schema(self):
        from epinet import registry as ereg

        profile = ereg.RegistryProfile(id_column="case_id", outcome_column="status")
        result = ereg.adapt(self._table(), profile)
        nodes, edges, manifest = result["nodes"], result["edges"], result["manifest"]
        self.assertEqual(list(nodes.columns[:2]), ["ID", "Outcome"])
        self.assertEqual(list(edges.columns), ["SourceID", "TargetID", "Weight"])
        # Auto feature selection keeps numerics, drops the non-numeric "site".
        self.assertEqual(set(manifest["feature_columns"]), {"age", "stage"})
        self.assertIn("site", manifest["dropped_columns"])
        self.assertEqual(manifest["n_cases"], 12)
        self.assertEqual(len(manifest["source_sha256"]), 64)

    def test_missing_id_column_raises(self):
        from epinet import registry as ereg

        with self.assertRaises(ValueError):
            ereg.adapt(self._table(), ereg.RegistryProfile(id_column="nope"))

    def test_edge_strategy_none_yields_no_edges(self):
        from epinet import registry as ereg

        result = ereg.adapt(self._table(), ereg.RegistryProfile(id_column="case_id", edge_strategy="none"))
        self.assertEqual(len(result["edges"]), 0)

    def test_shared_attribute_edges_link_matching_cases(self):
        from epinet import registry as ereg

        profile = ereg.RegistryProfile(
            id_column="case_id", edge_strategy="shared", shared_column="site",
        )
        result = ereg.adapt(self._table(), profile)
        # Two "site" groups of 6 -> each fully connected: 2 * C(6,2) = 30 edges.
        self.assertEqual(len(result["edges"]), 30)

    def test_output_runs_through_epinet(self):
        from epinet import registry as ereg

        result = ereg.adapt(self._table(), ereg.RegistryProfile(id_column="case_id", outcome_column="status"))
        graph = et.build_graph(result["nodes"], result["edges"], weight_column="Weight")
        features = et.generate_graph_features(graph)
        self.assertEqual(graph.number_of_nodes(), 12)
        self.assertEqual(len(features), 12)

    def test_same_profile_makes_sites_column_compatible(self):
        from epinet import registry as ereg

        table = self._table(n=12)
        profile = ereg.RegistryProfile(id_column="case_id", outcome_column="status")
        a = ereg.adapt(table.iloc[:6], profile)["nodes"]
        b = ereg.adapt(table.iloc[6:], profile)["nodes"]
        # The federation precondition: identical columns from the same profile.
        self.assertEqual(list(a.columns), list(b.columns))

    def test_profile_from_dict_rejects_unknown_keys(self):
        from epinet import registry as ereg

        with self.assertRaises(ValueError):
            ereg.RegistryProfile.from_dict({"id_column": "x", "bogus": 1})


class IngestNormalizationTests(unittest.TestCase):
    """Front-end column-alias normalization (epinet_ingest)."""

    def test_aliases_are_resolved_to_canonical_schema(self):
        from epinet import ingest as ein

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
        from epinet import ingest as ein

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
        from epinet import ingest as ein  # noqa: F401  (module exists; flag bypasses it)

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
