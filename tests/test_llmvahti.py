"""Tests for the experimental LLMvahti blinded judge audit."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from epinet import llmvahti as elv


def _ratings(n=40, flip_every=5, seed=7):
    """Synthetic two-class audit: judge mostly agrees, disagrees periodically."""
    rng = np.random.default_rng(seed)
    human = np.where(np.arange(n) % 2 == 0, "pass", "fail")
    judge = human.copy()
    judge[::flip_every] = np.where(judge[::flip_every] == "pass", "fail", "pass")
    # Criterion scores that separate the judge's classes, with noise.
    base = np.where(judge == "pass", 1.0, -1.0)
    human_df = pd.DataFrame({"item_id": np.arange(n), "human_label": human})
    judge_df = pd.DataFrame(
        {
            "item_id": np.arange(n),
            "judge_label": judge,
            "judge_confidence": np.clip(rng.normal(0.8, 0.1, n), 0.05, 0.95),
            "criterion_accuracy": base + rng.normal(0, 0.3, n),
            "criterion_completeness": base + rng.normal(0, 0.5, n),
            "criterion_style": rng.normal(0, 1.0, n),  # irrelevant criterion
        }
    )
    return human_df, judge_df


class AgreementStatTests(unittest.TestCase):
    def test_kappa_perfect_and_chance(self):
        a = pd.Series(["x", "y", "x", "y"])
        self.assertAlmostEqual(elv.cohens_kappa(a, a.copy()), 1.0)
        # Constant raters on the same label: expected agreement 1 -> undefined.
        c = pd.Series(["x", "x", "x"])
        self.assertIsNone(elv.cohens_kappa(c, c.copy()))

    def test_kappa_known_value(self):
        # Classic 2x2 example: po=0.7, pe=0.5 -> kappa=0.4.
        a = pd.Series(["y"] * 5 + ["n"] * 5)
        b = pd.Series(["y", "y", "y", "y", "n", "n", "n", "n", "n", "y"])
        po = float(np.mean(a.to_numpy() == b.to_numpy()))
        self.assertAlmostEqual(po, 0.8)
        self.assertAlmostEqual(elv.cohens_kappa(a, b), (0.8 - 0.5) / 0.5)

    def test_alpha_bounds_and_undefined(self):
        a = pd.Series(["x", "y", "x", "y", "x", "y"])
        self.assertAlmostEqual(elv.krippendorff_alpha(a, a.copy()), 1.0)
        c = pd.Series(["x", "x"])
        self.assertIsNone(elv.krippendorff_alpha(c, c.copy()))

    def test_blank_labels_are_dropped_not_compared(self):
        a = pd.Series(["x", "", "y", None])
        b = pd.Series(["x", "y", "y", "x"])
        result = elv.agreement(a, b)
        self.assertEqual(result["n_items"], 2)
        self.assertAlmostEqual(result["raw_agreement"], 1.0)


class AgreementBootstrapTests(unittest.TestCase):
    def _pair(self, n=60, disagree_every=4, seed=3):
        rng = np.random.default_rng(seed)
        human = pd.Series(rng.choice(["pass", "fail"], size=n))
        judge = human.copy()
        flip = np.arange(0, n, disagree_every)
        judge.iloc[flip] = judge.iloc[flip].map({"pass": "fail", "fail": "pass"})
        return human, judge

    def test_ci_block_present_and_brackets_point_estimates(self):
        human, judge = self._pair()
        result = elv.agreement(human, judge)
        ci = result["confidence_intervals"]
        self.assertIsNotNone(ci)
        for key in ("raw_agreement", "cohens_kappa", "krippendorff_alpha"):
            lo, hi = ci[key]
            self.assertLessEqual(lo, hi)
            self.assertLessEqual(lo, result[key])
            self.assertGreaterEqual(hi, result[key])
        self.assertEqual(ci["n_boot"], 1000)
        self.assertEqual(ci["ci_level"], 0.95)

    def test_ci_is_reproducible_under_fixed_seed(self):
        human, judge = self._pair()
        a = elv.agreement(human, judge, random_state=11)["confidence_intervals"]
        b = elv.agreement(human, judge, random_state=11)["confidence_intervals"]
        self.assertEqual(a["cohens_kappa"], b["cohens_kappa"])

    def test_small_sample_reports_no_interval(self):
        a = pd.Series(["x", "y", "x", "y", "x"])  # n=5 < _MIN_ITEMS_FOR_CI
        self.assertIsNone(elv.agreement(a, a.copy())["confidence_intervals"])

    def test_n_boot_zero_skips_interval(self):
        human, judge = self._pair()
        self.assertIsNone(elv.agreement(human, judge, n_boot=0)["confidence_intervals"])


class HumanPanelTests(unittest.TestCase):
    def test_multi_rater_alpha_reduces_to_two_rater(self):
        a = pd.Series(["y"] * 5 + ["n"] * 5)
        b = pd.Series(["y", "y", "y", "y", "n", "n", "n", "n", "n", "y"])
        two = elv.krippendorff_alpha(a, b)
        multi = elv._krippendorff_alpha_nominal([[x, y] for x, y in zip(a, b)])
        self.assertAlmostEqual(two, multi)

    def test_unanimous_panel_is_one(self):
        self.assertAlmostEqual(
            elv._krippendorff_alpha_nominal([["x", "x", "x"], ["y", "y", "y"]]), 1.0
        )

    def test_single_human_rater_has_no_panel(self):
        df = pd.DataFrame({"human_label": ["x", "y", "x"]}, index=pd.Index([0, 1, 2], name="item_id"))
        self.assertIsNone(elv.human_panel_agreement(df))

    def test_panel_reports_splits_and_alpha(self):
        n = 12
        df = pd.DataFrame(
            {
                "human_label": (["x", "y"] * (n // 2)),
                "human_label_b": (["x", "y"] * (n // 2)),
                "human_label_c": (["x", "y"] * (n // 2)),
            },
            index=pd.Index(range(n), name="item_id"),
        )
        df.loc[2, "human_label_c"] = "y"  # one split item
        panel = elv.human_panel_agreement(df)
        self.assertEqual(panel["n_raters"], 3)
        self.assertEqual(panel["n_split"], 1)
        self.assertIn("2", panel["split_item_ids"])
        self.assertLess(panel["krippendorff_alpha"], 1.0)

    def test_panel_flows_through_audit_and_report(self):
        human_df, judge_df = _ratings()
        # Add a second and third human rater that mostly agree with human_label.
        human_df["human_label_b"] = human_df["human_label"]
        human_df["human_label_c"] = human_df["human_label"]
        human_df.loc[3, "human_label_b"] = "fail"  # plant a panel split
        audit = elv.BlindedAudit()
        audit.seal_human(human_df)
        audit.add_judge(judge_df)
        results = audit.results()
        self.assertIn("human_panel", results)
        self.assertEqual(results["human_panel"]["n_raters"], 3)
        report = elv.audit_report(results)
        self.assertIn("Human panel", report)


class ConformalVerdictSetTests(unittest.TestCase):
    def _binary(self, n=120, seed=2):
        rng = np.random.default_rng(seed)
        hl = rng.choice(["pass", "fail"], n)
        conf = np.clip(rng.beta(5, 2, n), 0.05, 0.99)
        correct = rng.random(n) < conf
        jl = np.where(correct, hl, np.where(hl == "pass", "fail", "pass"))
        return pd.Series(hl), pd.Series(jl), pd.Series(conf)

    def test_sets_are_well_formed_and_partition(self):
        h, j, c = self._binary()
        out = elv.conformal_verdict_sets(h, j, c, alpha=0.1, random_state=0)
        self.assertEqual(out["n_singleton"] + out["n_ambiguous"] + out["n_empty"], out["n_items"])
        sizes = out["_assignments"]["set_size"].to_numpy()
        self.assertTrue(set(np.unique(sizes)).issubset({0, 1, 2}))
        self.assertTrue(0.0 <= out["empirical_coverage_heldout"] <= 1.0)

    def test_coverage_near_target(self):
        # Held-out coverage should land near 1 - alpha (allow sampling slack).
        h, j, c = self._binary(n=200)
        out = elv.conformal_verdict_sets(h, j, c, alpha=0.1, random_state=0)
        self.assertGreater(out["empirical_coverage_heldout"], 0.8)

    def test_reproducible_under_seed(self):
        h, j, c = self._binary()
        a = elv.conformal_verdict_sets(h, j, c, random_state=7)["nonconformity_threshold"]
        b = elv.conformal_verdict_sets(h, j, c, random_state=7)["nonconformity_threshold"]
        self.assertEqual(a, b)

    def test_non_binary_and_small_n_return_none(self):
        tri = pd.Series(["a", "b", "c"] * 10)
        self.assertIsNone(elv.conformal_verdict_sets(tri, tri.copy(), pd.Series([0.8] * 30)))
        small = pd.Series(["pass", "fail"] * 5)
        self.assertIsNone(elv.conformal_verdict_sets(small, small.copy(), pd.Series([0.8] * 10)))

    def test_alpha_out_of_range_raises(self):
        h, j, c = self._binary()
        with self.assertRaises(ValueError):
            elv.conformal_verdict_sets(h, j, c, alpha=1.5)

    def test_conformal_flows_through_run_and_writes_csv(self):
        rng = np.random.default_rng(4)
        n = 60
        hl = rng.choice(["pass", "fail"], n)
        conf = np.clip(rng.beta(5, 2, n), 0.05, 0.99)
        correct = rng.random(n) < conf
        jl = np.where(correct, hl, np.where(hl == "pass", "fail", "pass"))
        human_df = pd.DataFrame({"item_id": np.arange(n), "human_label": hl})
        judge_df = pd.DataFrame({"item_id": np.arange(n), "judge_label": jl, "judge_confidence": conf})
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            human_df.to_csv(tmp / "h.csv", index=False)
            judge_df.to_csv(tmp / "j.csv", index=False)
            results = elv.run_blinded_audit(tmp / "h.csv", tmp / "j.csv", tmp / "out")
            self.assertIn("conformal_verdict_sets", results)
            self.assertTrue((tmp / "out" / "conformal_sets.csv").exists())
            self.assertIn("Conformal verdict sets", (tmp / "out" / "judge_audit.md").read_text())


class BlindedProtocolTests(unittest.TestCase):
    def test_judge_before_seal_is_refused(self):
        human_df, judge_df = _ratings()
        audit = elv.BlindedAudit()
        with self.assertRaisesRegex(RuntimeError, "blinded protocol violation"):
            audit.add_judge(judge_df)

    def test_audit_closes_after_results(self):
        human_df, judge_df = _ratings()
        audit = elv.BlindedAudit()
        audit.seal_human(human_df)
        audit.add_judge(judge_df)
        audit.results()
        with self.assertRaises(RuntimeError):
            audit.add_judge(judge_df)

    def test_seal_hash_is_deterministic(self):
        human_df, _ = _ratings()
        sha1 = elv.BlindedAudit().seal_human(human_df)
        sha2 = elv.BlindedAudit().seal_human(human_df.copy())
        self.assertEqual(sha1, sha2)

    def test_duplicate_item_ids_are_refused(self):
        human_df, _ = _ratings()
        dup = pd.concat([human_df, human_df.head(1)])
        with self.assertRaisesRegex(ValueError, "duplicate item_id"):
            elv.BlindedAudit().seal_human(dup)


class AuditResultTests(unittest.TestCase):
    def setUp(self):
        self.human_df, self.judge_df = _ratings()
        audit = elv.BlindedAudit()
        audit.seal_human(self.human_df)
        audit.add_judge(self.judge_df)
        self.results = audit.results()

    def test_agreement_reflects_planted_disagreements(self):
        agree = self.results["agreement"]
        n = len(self.human_df)
        planted = len(range(0, n, 5))
        self.assertAlmostEqual(agree["raw_agreement"], (n - planted) / n)
        self.assertIsNotNone(agree["cohens_kappa"])

    def test_calibration_block_present_and_sane(self):
        cal = self.results["judge_calibration"]
        self.assertEqual(cal["n_items"], len(self.human_df))
        self.assertTrue(0.0 <= cal["brier_score"] <= 1.0)

    def test_calibration_ci_brackets_point_estimates(self):
        cal = self.results["judge_calibration"]
        ci = cal["confidence_intervals"]
        self.assertIsNotNone(ci)
        for key in ("brier_score", "judge_accuracy_vs_human"):
            lo, hi = ci[key]
            self.assertLessEqual(lo, cal[key])
            self.assertGreaterEqual(hi, cal[key])
        self.assertEqual(ci["n_boot"], 1000)
        self.assertIn("n_undefined_calibration", ci)

    def test_calibration_ci_reproducible_and_skippable(self):
        human = self.human_df["human_label"]
        judge = self.judge_df.set_index("item_id")["judge_label"].reindex(self.human_df["item_id"])
        conf = self.judge_df.set_index("item_id")["judge_confidence"].reindex(self.human_df["item_id"])
        a = elv.judge_calibration(human, judge, conf, random_state=9)["confidence_intervals"]
        b = elv.judge_calibration(human, judge, conf, random_state=9)["confidence_intervals"]
        self.assertEqual(a["brier_score"], b["brier_score"])
        skipped = elv.judge_calibration(human, judge, conf, n_boot=0)["confidence_intervals"]
        self.assertIsNone(skipped)

    def test_contestability_targets_judge_verdicts(self):
        summary = self.results["verdict_contestability"]
        self.assertEqual(summary["n_scored"], len(self.human_df))
        # The separating criteria should out-lever the irrelevant style criterion.
        leverage = summary["feature_leverage"]
        self.assertGreater(leverage["criterion_accuracy"], leverage["criterion_style"])

    def test_grey_zone_counts_contested_disagreements(self):
        assignments = self.results["_assignments"]
        grey = assignments[assignments["contested"] & assignments["human_disagrees"]]
        self.assertEqual(self.results["n_grey_zone_disagreements"], len(grey))

    def test_caveats_lead_with_human_standard(self):
        self.assertIn("human standard", self.results["caveats"][0])


class SubgroupErrorFunnelTests(unittest.TestCase):
    def test_normal_quantile_matches_known_values(self):
        self.assertAlmostEqual(elv._normal_quantile(0.975), 1.959964, places=5)
        self.assertAlmostEqual(elv._normal_quantile(0.999), 3.090232, places=5)
        self.assertAlmostEqual(elv._normal_quantile(0.5), 0.0, places=9)
        self.assertAlmostEqual(elv._normal_quantile(0.01), -elv._normal_quantile(0.99), places=9)

    def test_planted_high_error_stratum_is_flagged(self):
        # Deterministic counts: site_bad disagrees 10/30 (33%), site_ok 13/270
        # (4.8%), pooled 7.7%. The bad stratum must clear its alarm limit; the
        # clean majority must stay inside (a planted effect mild enough not to
        # drag the pooled benchmark, which would legitimately flag site_ok low).
        groups = pd.Series(["site_bad"] * 30 + ["site_ok"] * 270)
        human = pd.Series(["pass"] * 300)
        judge = pd.Series(["pass"] * 300)
        judge.iloc[:10] = "fail"
        judge.iloc[30:43] = "fail"
        funnel = elv.subgroup_error_funnel(human, judge, groups)
        by_name = {s["group"]: s for s in funnel["strata"]}
        self.assertEqual(by_name["site_bad"]["flag"], "high")
        self.assertIsNone(by_name["site_ok"]["flag"])
        self.assertFalse(by_name["site_ok"]["outside_warn"])
        self.assertEqual(funnel["n_flagged_high"], 1)

    def test_tiny_stratum_at_pooled_rate_is_not_flagged(self):
        # A 3-item stratum with one disagreement should sit inside its wide limits.
        human = pd.Series(["a"] * 103)
        judge = pd.Series(["a"] * 103)
        judge.iloc[:10] = "b"  # pooled rate ~9.7%
        groups = pd.Series(["big"] * 100 + ["tiny"] * 3)
        judge.iloc[100] = "b"  # tiny: 1/3 disagree
        funnel = elv.subgroup_error_funnel(human, judge, groups)
        tiny = next(s for s in funnel["strata"] if s["group"] == "tiny")
        self.assertIsNone(tiny["flag"])

    def test_funnel_caveats_disclaim_causal_bias(self):
        human = pd.Series(["a", "b"] * 10)
        judge = pd.Series(["a", "b"] * 10)
        groups = pd.Series(["g1", "g2"] * 10)
        funnel = elv.subgroup_error_funnel(human, judge, groups)
        self.assertIn("not proof of causal bias", funnel["caveats"][0])

    def test_group_columns_flow_through_audit_and_report(self):
        human_df, judge_df = _ratings()
        judge_df["group_site"] = np.where(judge_df["item_id"] < 20, "north", "south")
        audit = elv.BlindedAudit()
        audit.seal_human(human_df)
        audit.add_judge(judge_df)
        results = audit.results()
        self.assertIn("group_site", results["subgroup_error_funnel"])
        self.assertEqual(results["subgroup_error_funnel"]["group_site"]["n_strata"], 2)
        report = elv.audit_report(results)
        self.assertIn("Subgroup error funnel", report)


class RunBlindedAuditTests(unittest.TestCase):
    def test_end_to_end_writes_report_and_provenance(self):
        human_df, judge_df = _ratings()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            human_csv = tmp / "human.csv"
            judge_csv = tmp / "judge.csv"
            human_df.to_csv(human_csv, index=False)
            judge_df.to_csv(judge_csv, index=False)
            results = elv.run_blinded_audit(human_csv, judge_csv, tmp / "out")
            self.assertTrue((tmp / "out" / "judge_audit.json").exists())
            self.assertTrue((tmp / "out" / "verdict_assignments.csv").exists())
            report = (tmp / "out" / "judge_audit.md").read_text()
            self.assertIn("blinded second rater", report)
            self.assertIn("Criterion leverage", report)
            self.assertIn("percentile bootstrap", report)
            self.assertIsNotNone(results["agreement"]["confidence_intervals"])
            self.assertIn("provenance", results)

    def test_cli_writes_bundle(self):
        human_df, judge_df = _ratings()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            human_df.to_csv(tmp / "h.csv", index=False)
            judge_df.to_csv(tmp / "j.csv", index=False)
            out = tmp / "out"
            elv.main(
                [
                    "--human", str(tmp / "h.csv"),
                    "--judge", str(tmp / "j.csv"),
                    "--output-dir", str(out),
                    "--n-boot", "200",
                    "--random-state", "1",
                ]
            )
            self.assertTrue((out / "judge_audit.json").exists())
            self.assertTrue((out / "judge_audit.md").exists())

    def test_cli_requires_both_rating_files(self):
        with self.assertRaises(SystemExit):
            elv.main(["--human", "only.csv"])

    def test_no_criteria_still_audits_agreement(self):
        human_df, judge_df = _ratings()
        judge_only = judge_df[["item_id", "judge_label"]]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            human_df.to_csv(tmp / "h.csv", index=False)
            judge_only.to_csv(tmp / "j.csv", index=False)
            results = elv.run_blinded_audit(tmp / "h.csv", tmp / "j.csv", tmp / "out")
            self.assertIn("agreement", results)
            self.assertNotIn("verdict_contestability", results)


if __name__ == "__main__":
    unittest.main()
