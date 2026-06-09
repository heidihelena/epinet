import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import pandas as pd

import epinet_toolkit as et


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


if __name__ == "__main__":
    unittest.main()
