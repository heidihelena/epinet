"""Tests for the experimental LLMvahti blinded judge audit."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import epinet_llmvahti as elv


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
            self.assertIn("provenance", results)

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
