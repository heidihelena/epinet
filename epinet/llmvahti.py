"""LLMvahti (experimental): blinded-second-rater audit of LLM-judge verdicts.

EpiNet's organising question — not just what a model predicts but how
well-founded each call is — applied to LLM-as-judge evaluation. The human is
the **primary rater**; the LLM judge is a **second rater**, and the audit is
honest about both how often they agree and how fragile the judge's verdicts
are:

1. **Blinded protocol** — human ratings are sealed (content-hashed) *before*
   the judge ratings enter the audit, and the audit refuses to run otherwise.
   The seal makes the ordering tamper-evident within a run; keeping the human
   genuinely unexposed to the judge's output is a process responsibility the
   software cannot verify, and the report says so.
2. **Inter-rater agreement** — raw agreement, Cohen's kappa, and a two-rater
   nominal Krippendorff's alpha between human and judge, each with a seeded
   percentile-bootstrap confidence interval (the small-sample audits LLMvahti is
   built for make a bare kappa point estimate badly under-determined), with the
   confusion matrix and every disagreeing item listed rather than averaged away.
3. **Judge calibration** — when the judge reports a confidence, it is scored
   against *being right by the human standard*: Brier score plus the standard
   weak-calibration slope/intercept (reusing the toolkit's Cox check), each
   with the same seeded percentile-bootstrap confidence interval as the
   agreement block.
4. **Verdict contestability** — the nearest-centroid flip-distance lens
   (``epinet_contest``) pointed at the judge's verdicts in rubric-criterion
   space: how small a move in criterion scores would flip each verdict, which
   criterion the verdict is most sensitive to, and which verdicts sit in the
   contested grey zone. Grey-zone *and* human-disagreeing verdicts are the
   audit's headline table: the calls most worth a human second look.

Honest reading — surfaced in the output, not buried:

- Agreement with the human rater is the audit's ground truth *by design*; it
  measures alignment with the human standard, not correctness in any absolute
  sense. Where the human standard is itself uncertain, so is the audit.
- Contestability here measures the geometry of the judge's verdicts in rubric
  space, not the quality of the underlying responses: a small flip-distance
  says the verdict is fragile, not that the response is genuinely borderline.
- This module is an experimental research demonstrator, not a benchmark, a
  leaderboard, or a substitute for human review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from epinet import common as epinet_common
from epinet import contest as ecn

CAVEATS = (
    "Agreement is measured against the human rater by design: it quantifies alignment "
    "with the human standard, not absolute correctness.",
    "The seal makes the human-before-judge ordering tamper-evident within this run; true "
    "blinding (the human never seeing the judge's output) is a process responsibility "
    "the software cannot verify.",
    "Verdict flip-distance measures the geometry of the judge's verdicts in rubric-criterion "
    "space; a small value means the verdict is fragile, not that the response is borderline.",
) + ecn.CAVEATS


def _kappa_arr(a: np.ndarray, b: np.ndarray) -> float | None:
    """Cohen's kappa on two aligned, blank-free rating arrays (``None`` if undefined)."""
    if len(a) == 0:
        return None
    labels = sorted(set(a) | set(b))
    observed = float(np.mean(a == b))
    pa = pd.Series(a).value_counts(normalize=True)
    pb = pd.Series(b).value_counts(normalize=True)
    expected = float(sum(pa.get(lab, 0.0) * pb.get(lab, 0.0) for lab in labels))
    if np.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1.0 - expected)


def _alpha_arr(a: np.ndarray, b: np.ndarray) -> float | None:
    """Two-rater nominal Krippendorff's alpha on aligned, blank-free arrays."""
    n = len(a)
    if n == 0:
        return None
    pooled = pd.Series(np.concatenate([a, b]))
    freqs = pooled.value_counts()
    total = float(len(pooled))
    if len(freqs) < 2:
        return None
    do = float(np.mean(a != b))
    de = 1.0 - float(((freqs / total) ** 2).sum())
    # Small-sample correction for the expected disagreement (Krippendorff):
    # use n_pooled/(n_pooled-1) * (1 - sum p^2) ... simplified pairwise form.
    de = de * total / (total - 1.0)
    if np.isclose(de, 0.0):
        return None
    return 1.0 - do / de


def cohens_kappa(a: pd.Series, b: pd.Series) -> float | None:
    """Cohen's kappa for two nominal raters; ``None`` when chance-undefined.

    Kappa is undefined when expected agreement is 1 (both raters constant on the
    same label); returning ``None`` rather than 0 keeps "undefined" distinct from
    "exactly chance-level".
    """
    a, b = _aligned_pair(a, b)
    return _kappa_arr(a, b)


def krippendorff_alpha(a: pd.Series, b: pd.Series) -> float | None:
    """Two-rater nominal Krippendorff's alpha; ``None`` when undefined.

    Nominal metric over the pooled value distribution: alpha = 1 - Do/De, where
    Do is observed disagreement across paired ratings and De the disagreement
    expected from the pooled label frequencies. Undefined (``None``) when fewer
    than two distinct values appear in the pool.
    """
    a, b = _aligned_pair(a, b)
    return _alpha_arr(a, b)


def _aligned_pair(a: pd.Series, b: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Align two rating series on their shared index and drop blank labels."""
    a = pd.Series(a).astype("string")
    b = pd.Series(b).astype("string")
    frame = pd.DataFrame({"a": a, "b": b})
    keep = ~(epinet_common.blank_label_mask(frame["a"]) | epinet_common.blank_label_mask(frame["b"]))
    frame = frame[keep]
    return frame["a"].to_numpy(), frame["b"].to_numpy()


# Below this many jointly-rated items a bootstrap interval is more noise than
# signal, so the CI block is reported as null with the count instead. The
# LLMvahti regime (25-50 golden items) sits comfortably above this floor.
_MIN_ITEMS_FOR_CI = 10


def _bootstrap_agreement(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int,
    random_state: int,
    alpha: float = 0.05,
) -> dict[str, object] | None:
    """Percentile-bootstrap CIs for the agreement metrics on aligned arrays.

    Resamples the paired items with replacement and recomputes raw agreement,
    Cohen's kappa, and Krippendorff's alpha on each replicate. Kappa and alpha
    are undefined on a degenerate resample (e.g. one that draws a single label);
    those replicates are excluded from the interval and counted, so the reader
    can see how often the statistic was estimable. Returns ``None`` when there
    are too few items or ``n_boot`` is non-positive — an honest "no interval"
    rather than a falsely tight one.
    """
    n = len(a)
    if n < _MIN_ITEMS_FOR_CI or n_boot <= 0:
        return None
    rng = np.random.default_rng(random_state)
    raw: list[float] = []
    kappa: list[float] = []
    alpha_vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ra, rb = a[idx], b[idx]
        raw.append(float(np.mean(ra == rb)))
        k = _kappa_arr(ra, rb)
        if k is not None:
            kappa.append(k)
        al = _alpha_arr(ra, rb)
        if al is not None:
            alpha_vals.append(al)
    lo, hi = 100 * alpha / 2.0, 100 * (1.0 - alpha / 2.0)

    def _ci(values: list[float]) -> list[float] | None:
        return [float(np.percentile(values, lo)), float(np.percentile(values, hi))] if values else None

    return {
        "method": "percentile bootstrap over paired items",
        "n_boot": int(n_boot),
        "random_state": int(random_state),
        "ci_level": round(1.0 - alpha, 4),
        "raw_agreement": _ci(raw),
        "cohens_kappa": _ci(kappa),
        "krippendorff_alpha": _ci(alpha_vals),
        "n_undefined_kappa": int(n_boot - len(kappa)),
        "n_undefined_alpha": int(n_boot - len(alpha_vals)),
    }


def agreement(
    human: pd.Series,
    judge: pd.Series,
    *,
    n_boot: int = 1000,
    random_state: int = 0,
) -> dict[str, object]:
    """Inter-rater agreement between the primary (human) and second (judge) rater.

    Point estimates always; a seeded percentile-bootstrap ``confidence_intervals``
    block when there are enough jointly-rated items (see ``_MIN_ITEMS_FOR_CI``).
    The interval matters here because the LLMvahti regime is small (tens of
    golden items), where a bare kappa point estimate is badly under-determined.
    """
    a, b = _aligned_pair(human, judge)
    n = len(a)
    if n == 0:
        raise ValueError("Agreement needs at least one item rated by both raters")
    labels = sorted(set(a) | set(b))
    confusion = {ha: {ja: int(np.sum((a == ha) & (b == ja))) for ja in labels} for ha in labels}
    return {
        "n_items": n,
        "labels": labels,
        "raw_agreement": float(np.mean(a == b)),
        "cohens_kappa": _kappa_arr(a, b),
        "krippendorff_alpha": _alpha_arr(a, b),
        "confusion_human_rows_judge_cols": confusion,
        "confidence_intervals": _bootstrap_agreement(
            a, b, n_boot=n_boot, random_state=random_state
        ),
    }


def _bootstrap_calibration(
    correct: np.ndarray,
    conf: np.ndarray,
    *,
    n_boot: int,
    random_state: int,
    alpha: float = 0.05,
) -> dict[str, object] | None:
    """Percentile-bootstrap CIs for the judge-calibration metrics.

    Resamples the paired (correct, confidence) items and recomputes the Brier
    score, judge accuracy, and the Cox slope/intercept on each replicate. The
    slope/intercept are jointly undefined on a degenerate resample (one class of
    ``correct``, or constant confidence); those replicates are excluded and
    counted. Returns ``None`` below ``_MIN_ITEMS_FOR_CI`` items or when
    ``n_boot`` is non-positive — an honest "no interval" rather than a falsely
    tight one, matching the agreement-metric bootstrap.
    """
    from epinet.toolkit import calibration_slope_intercept

    n = len(correct)
    if n < _MIN_ITEMS_FOR_CI or n_boot <= 0:
        return None
    rng = np.random.default_rng(random_state)
    brier: list[float] = []
    accuracy: list[float] = []
    slope: list[float] = []
    intercept: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        c, p = correct[idx], conf[idx]
        brier.append(float(np.mean((p - c) ** 2)))
        accuracy.append(float(c.mean()))
        cal = calibration_slope_intercept(pd.Series(c), p, 1)
        if cal["slope"] is not None:
            slope.append(cal["slope"])
        if cal["intercept"] is not None:
            intercept.append(cal["intercept"])
    lo, hi = 100 * alpha / 2.0, 100 * (1.0 - alpha / 2.0)

    def _ci(values: list[float]) -> list[float] | None:
        return [float(np.percentile(values, lo)), float(np.percentile(values, hi))] if values else None

    return {
        "method": "percentile bootstrap over paired items",
        "n_boot": int(n_boot),
        "random_state": int(random_state),
        "ci_level": round(1.0 - alpha, 4),
        "brier_score": _ci(brier),
        "judge_accuracy_vs_human": _ci(accuracy),
        "calibration_slope": _ci(slope),
        "calibration_intercept": _ci(intercept),
        "n_undefined_calibration": int(n_boot - len(slope)),
    }


def judge_calibration(
    human: pd.Series,
    judge: pd.Series,
    confidence: pd.Series,
    *,
    n_boot: int = 1000,
    random_state: int = 0,
) -> dict[str, object]:
    """Score the judge's confidence against being right by the human standard.

    ``correct`` is 1 when judge and human agree on an item. A well-calibrated
    judge's confidence should track that probability: Brier score plus the Cox
    weak-calibration slope/intercept (slope < 1 = overconfident). Reuses the
    toolkit's logistic check so the reading matches the outcome-model report.
    Point estimates always; a seeded percentile-bootstrap ``confidence_intervals``
    block when there are enough jointly-rated items (same small-sample rationale
    as the agreement metrics — a Brier or slope on tens of items is noisy).
    """
    from epinet.toolkit import calibration_slope_intercept

    frame = pd.DataFrame(
        {
            "human": pd.Series(human).astype("string"),
            "judge": pd.Series(judge).astype("string"),
            "conf": pd.to_numeric(pd.Series(confidence), errors="coerce"),
        }
    )
    keep = ~(epinet_common.blank_label_mask(frame["human"]) | epinet_common.blank_label_mask(frame["judge"]))
    frame = frame[keep]
    if frame.empty:
        raise ValueError("calibration needs at least one item rated by both raters")
    a = frame["human"].to_numpy()
    b = frame["judge"].to_numpy()
    conf = frame["conf"].to_numpy(dtype=float)
    if not np.isfinite(conf).all():
        raise ValueError("judge confidence contains missing values; drop or impute deliberately")
    if conf.min() < 0.0 or conf.max() > 1.0:
        raise ValueError("judge confidence must be in [0, 1]")
    correct = (a == b).astype(int)
    brier = float(np.mean((conf - correct) ** 2))
    cal = calibration_slope_intercept(pd.Series(correct), conf, 1)
    return {
        "n_items": int(len(correct)),
        "judge_accuracy_vs_human": float(correct.mean()),
        "mean_confidence": float(conf.mean()),
        "brier_score": brier,
        "calibration_slope": cal["slope"],
        "calibration_intercept": cal["intercept"],
        "confidence_intervals": _bootstrap_calibration(
            correct, conf, n_boot=n_boot, random_state=random_state
        ),
    }


FUNNEL_CAVEATS = (
    "The subgroup error funnel is an exploratory differential-error screen: it flags strata "
    "where judge disagreement with the sealed human standard is unusually high or low after "
    "accounting for subgroup size. It is not proof of causal bias and inherits the "
    "limitations of the human standard.",
    "Funnel limits are normal-approximation control limits with continuity correction around "
    "the pooled rate; with many strata some excursions are expected by chance, which is why "
    "only the outer (alarm) limit flags.",
)


def _normal_quantile(p: float) -> float:
    """Inverse standard-normal CDF (Beasley–Springer–Moro approximation).

    Keeps the funnel scipy-free; |error| < 3e-9 over (0, 1), far below the
    resolution a screening funnel needs.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("quantile probability must be in (0, 1)")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > p_high:
        return -_normal_quantile(1 - p)
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def subgroup_error_funnel(
    human: pd.Series,
    judge: pd.Series,
    groups: pd.Series,
    *,
    warn_level: float = 0.95,
    alarm_level: float = 0.998,
) -> dict[str, object]:
    """Exploratory differential-error screen over subgroups (funnel plot logic).

    For each stratum of ``groups``, compares the judge-vs-human disagreement
    rate against funnel control limits around the pooled rate at the stratum's
    size — the quality-indicator funnel, pointed at the judge. Strata outside
    the outer (``alarm_level``) limits are flagged ``"high"``/``"low"``; the
    inner (``warn_level``) excursions are reported but not flagged, because
    with many strata some inner excursions are expected by chance.

    This is a screen, not an inference: a flag says "this stratum's error rate
    is unusual given its size", not that the judge is causally biased, and the
    whole reading inherits the limitations of the human standard.
    """
    if not 0.5 < warn_level < alarm_level < 1.0:
        raise ValueError("need 0.5 < warn_level < alarm_level < 1")
    frame = pd.DataFrame(
        {
            "human": pd.Series(human).astype("string"),
            "judge": pd.Series(judge).astype("string"),
            "group": pd.Series(groups).astype("string"),
        }
    )
    keep = ~(
        epinet_common.blank_label_mask(frame["human"])
        | epinet_common.blank_label_mask(frame["judge"])
        | epinet_common.blank_label_mask(frame["group"])
    )
    frame = frame[keep]
    if frame.empty:
        raise ValueError("funnel needs at least one item with both ratings and a group")
    frame["disagree"] = frame["human"] != frame["judge"]
    pooled = float(frame["disagree"].mean())
    z_warn = _normal_quantile(0.5 + warn_level / 2.0)
    z_alarm = _normal_quantile(0.5 + alarm_level / 2.0)

    strata = []
    for name, sub in frame.groupby("group", sort=True):
        n = int(len(sub))
        k = int(sub["disagree"].sum())
        rate = k / n
        se = float(np.sqrt(pooled * (1.0 - pooled) / n))
        cc = 0.5 / n  # continuity correction
        limits = {
            "warn_low": max(0.0, pooled - z_warn * se - cc),
            "warn_high": min(1.0, pooled + z_warn * se + cc),
            "alarm_low": max(0.0, pooled - z_alarm * se - cc),
            "alarm_high": min(1.0, pooled + z_alarm * se + cc),
        }
        flag = None
        if rate > limits["alarm_high"]:
            flag = "high"
        elif rate < limits["alarm_low"]:
            flag = "low"
        strata.append(
            {
                "group": str(name),
                "n": n,
                "n_disagree": k,
                "disagreement_rate": rate,
                **{key: float(v) for key, v in limits.items()},
                "outside_warn": bool(rate > limits["warn_high"] or rate < limits["warn_low"]),
                "flag": flag,
            }
        )

    return {
        "pooled_disagreement_rate": pooled,
        "n_items": int(len(frame)),
        "n_strata": len(strata),
        "warn_level": warn_level,
        "alarm_level": alarm_level,
        "strata": strata,
        "n_flagged_high": sum(1 for s in strata if s["flag"] == "high"),
        "n_flagged_low": sum(1 for s in strata if s["flag"] == "low"),
        "caveats": list(FUNNEL_CAVEATS),
    }


class BlindedAudit:
    """Order-enforcing container: human ratings seal before judge ratings enter.

    ``seal_human`` content-hashes the human ratings; ``add_judge`` refuses to run
    before the seal exists, and ``results`` refuses until both raters are in.
    After ``results`` is computed the audit is closed and further mutation
    raises — the same fail-closed posture as the governance gate.
    """

    def __init__(self) -> None:
        self._human: pd.DataFrame | None = None
        self._human_sha: str | None = None
        self._judge: pd.DataFrame | None = None
        self._closed = False

    @staticmethod
    def _check_frame(frame: pd.DataFrame, label_col: str) -> pd.DataFrame:
        if "item_id" not in frame.columns or label_col not in frame.columns:
            raise ValueError(f"ratings need 'item_id' and '{label_col}' columns")
        if frame["item_id"].duplicated().any():
            raise ValueError("duplicate item_id in ratings")
        return frame.set_index("item_id")

    def seal_human(self, frame: pd.DataFrame) -> str:
        if self._closed or self._human is not None:
            raise RuntimeError("human ratings are already sealed")
        self._human = self._check_frame(frame, "human_label")
        self._human_sha = epinet_common.sha256_frame(frame)
        return self._human_sha

    def add_judge(self, frame: pd.DataFrame) -> None:
        if self._closed:
            raise RuntimeError("audit is closed")
        if self._human_sha is None:
            raise RuntimeError(
                "blinded protocol violation: seal the human ratings before judge ratings enter"
            )
        self._judge = self._check_frame(frame, "judge_label")

    def results(
        self,
        *,
        metric: str = "euclidean",
        contest_quantile: float = 0.1,
        n_boot: int = 1000,
        random_state: int = 0,
    ) -> dict[str, object]:
        if self._human is None or self._judge is None:
            raise RuntimeError("audit needs sealed human ratings and judge ratings")
        self._closed = True

        human = self._human["human_label"]
        judge = self._judge["judge_label"].reindex(human.index)
        out: dict[str, object] = {
            "human_ratings_sha256": self._human_sha,
            "agreement": agreement(human, judge, n_boot=n_boot, random_state=random_state),
            "caveats": list(CAVEATS),
        }

        if "judge_confidence" in self._judge.columns:
            out["judge_calibration"] = judge_calibration(
                human,
                judge,
                self._judge["judge_confidence"].reindex(human.index),
                n_boot=n_boot,
                random_state=random_state,
            )

        criteria = self._judge.reindex(human.index).filter(regex="^criterion_")
        criteria = criteria.select_dtypes(include=[np.number])
        if criteria.shape[1] >= 1:
            contest = ecn.contestability(
                criteria,
                y=self._judge["judge_label"].reindex(human.index),
                metric=metric,
                contest_quantile=contest_quantile,
            )
            assignments = contest["assignments"].set_index("ID")
            disagrees = human.astype("string").reindex(assignments.index) != judge.astype("string").reindex(
                assignments.index
            )
            assignments["human_disagrees"] = disagrees.to_numpy()
            grey = assignments[assignments["contested"] & assignments["human_disagrees"]]
            out["verdict_contestability"] = contest["summary"]
            out["criterion_leverage"] = contest["summary"]["feature_leverage"]
            out["n_grey_zone_disagreements"] = int(len(grey))
            out["_assignments"] = assignments.reset_index()

        group_cols = [c for c in self._judge.columns if c.startswith("group_")]
        if group_cols:
            out["subgroup_error_funnel"] = {
                col: subgroup_error_funnel(human, judge, self._judge[col].reindex(human.index))
                for col in group_cols
            }
        return out


def audit_report(results: dict[str, object]) -> str:
    """Render the audit as Markdown, disagreements and grey zone first."""
    agree = results["agreement"]
    lines = [
        "# LLMvahti judge audit (experimental)",
        "",
        "Human is the primary rater; the LLM judge is a blinded second rater. "
        f"Human ratings sealed at SHA-256 `{results['human_ratings_sha256']}`.",
        "",
        "## Inter-rater agreement",
        "",
        f"- items rated by both: **{agree['n_items']}**",
        f"- raw agreement: **{agree['raw_agreement']:.3f}**{_ci_suffix(agree, 'raw_agreement')}",
        f"- Cohen's kappa: **{_fmt(agree['cohens_kappa'])}**{_ci_suffix(agree, 'cohens_kappa')}",
        f"- Krippendorff's alpha (nominal): **{_fmt(agree['krippendorff_alpha'])}**"
        f"{_ci_suffix(agree, 'krippendorff_alpha')}",
    ]
    ci = agree.get("confidence_intervals")
    if ci is None:
        lines.append(
            f"- *(no bootstrap interval: fewer than {_MIN_ITEMS_FOR_CI} jointly-rated items)*"
        )
    else:
        lines.append(
            f"- intervals: {ci['ci_level']:.0%} percentile bootstrap, "
            f"{ci['n_boot']} resamples (seed {ci['random_state']})"
        )
    cal = results.get("judge_calibration")
    if cal:
        lines += [
            "",
            "## Judge calibration (confidence vs. being right by the human standard)",
            "",
            f"- judge accuracy vs. human: **{cal['judge_accuracy_vs_human']:.3f}**"
            f"{_ci_suffix(cal, 'judge_accuracy_vs_human')} "
            f"at mean confidence **{cal['mean_confidence']:.3f}**",
            f"- Brier score: **{cal['brier_score']:.3f}**{_ci_suffix(cal, 'brier_score')}",
            f"- calibration slope / intercept: **{_fmt(cal['calibration_slope'])} / "
            f"{_fmt(cal['calibration_intercept'])}** (slope < 1 = overconfident)"
            f"{_ci_suffix(cal, 'calibration_slope')}",
        ]
        cal_ci = cal.get("confidence_intervals")
        if cal_ci is None:
            lines.append(
                f"- *(no bootstrap interval: fewer than {_MIN_ITEMS_FOR_CI} jointly-rated items)*"
            )
        else:
            lines.append(
                f"- intervals: {cal_ci['ci_level']:.0%} percentile bootstrap, "
                f"{cal_ci['n_boot']} resamples (seed {cal_ci['random_state']}); "
                f"slope CI shown after the slope/intercept pair"
            )
    summary = results.get("verdict_contestability")
    if summary:
        flip = summary["flip_distance"]
        lines += [
            "",
            "## Verdict contestability (judge verdicts in rubric-criterion space)",
            "",
            f"- verdicts scored: {summary['n_scored']}; contested (lowest "
            f"{flip['contest_quantile']:.0%} flip-distance): **{flip['n_contested']}**",
            f"- grey zone *and* human disagrees — the calls to re-review first: "
            f"**{results['n_grey_zone_disagreements']}**",
            "",
            "Criterion leverage (which rubric criterion drives verdict flips):",
            "",
        ]
        lines += [
            f"- `{name}`: {share:.3f}" for name, share in list(summary["feature_leverage"].items())[:10]
        ]
    funnels = results.get("subgroup_error_funnel")
    if funnels:
        lines += ["", "## Subgroup error funnel (exploratory differential-error screen)", ""]
        for col, funnel in funnels.items():
            lines.append(
                f"- `{col}`: pooled disagreement **{funnel['pooled_disagreement_rate']:.3f}** "
                f"across {funnel['n_strata']} strata; flagged high/low at the "
                f"{funnel['alarm_level']:.1%} limit: **{funnel['n_flagged_high']}** / "
                f"**{funnel['n_flagged_low']}**"
            )
            for s in funnel["strata"]:
                if s["flag"]:
                    lines.append(
                        f"  - `{s['group']}` flagged **{s['flag']}**: "
                        f"{s['n_disagree']}/{s['n']} = {s['disagreement_rate']:.3f} "
                        f"(alarm limits {s['alarm_low']:.3f}–{s['alarm_high']:.3f})"
                    )
        lines += ["", *(f"- {c}" for c in FUNNEL_CAVEATS)]
    lines += ["", "## Caveats", ""]
    lines += [f"- {c}" for c in results["caveats"]]
    return "\n".join(lines) + "\n"


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "undefined"
    return f"{float(value):.{digits}f}"


def _ci_suffix(agreement_block: dict[str, object], key: str, digits: int = 3) -> str:
    """`` [lo, hi]`` for a metric's bootstrap CI, or empty when unavailable."""
    ci = agreement_block.get("confidence_intervals")
    if not ci:
        return ""
    interval = ci.get(key)
    if not interval:
        return ""
    return f" [{interval[0]:.{digits}f}, {interval[1]:.{digits}f}]"


def run_blinded_audit(
    human_csv: str | Path,
    judge_csv: str | Path,
    out_dir: str | Path,
    *,
    metric: str = "euclidean",
    contest_quantile: float = 0.1,
    n_boot: int = 1000,
    random_state: int = 0,
) -> dict[str, object]:
    """End-to-end blinded audit from two CSVs, with provenance and a report.

    ``human_csv``: ``item_id, human_label``. ``judge_csv``: ``item_id,
    judge_label`` plus optional ``judge_confidence`` (in [0, 1]), any number of
    numeric ``criterion_*`` rubric columns, and any number of categorical
    ``group_*`` columns (each gets a subgroup error funnel). Writes
    ``judge_audit.json``, ``judge_audit.md``, and ``verdict_assignments.csv``.

    ``n_boot`` and ``random_state`` control the seeded percentile bootstrap for
    the agreement-metric confidence intervals (set ``n_boot=0`` to skip it).
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    auditor = BlindedAudit()
    auditor.seal_human(pd.read_csv(human_csv))
    auditor.add_judge(pd.read_csv(judge_csv))
    results = auditor.results(
        metric=metric, contest_quantile=contest_quantile, n_boot=n_boot, random_state=random_state
    )

    assignments = results.pop("_assignments", None)
    if assignments is not None:
        assignments.to_csv(out_path / "verdict_assignments.csv", index=False)
    results["provenance"] = epinet_common.provenance([human_csv, judge_csv])
    (out_path / "judge_audit.json").write_text(json.dumps(results, indent=2, default=str))
    (out_path / "judge_audit.md").write_text(audit_report(results))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "LLMvahti (experimental): blinded-second-rater audit of LLM-judge verdicts. "
            "Human is the primary rater; the judge is a blinded second rater."
        )
    )
    parser.add_argument("--human", required=True, help="Human ratings CSV: item_id, human_label")
    parser.add_argument(
        "--judge",
        required=True,
        help=(
            "Judge ratings CSV: item_id, judge_label, plus optional judge_confidence in [0,1], "
            "numeric criterion_* rubric columns, and categorical group_* strata"
        ),
    )
    parser.add_argument("--output-dir", default="llmvahti_outputs", help="Directory for the audit bundle")
    parser.add_argument(
        "--metric",
        default="euclidean",
        choices=["euclidean", "mahalanobis"],
        help="Distance metric for the verdict-contestability flip-distance",
    )
    parser.add_argument(
        "--contest-quantile",
        type=float,
        default=0.1,
        help="Lowest-flip-distance fraction of verdicts flagged as the contested grey zone",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=1000,
        help="Bootstrap resamples for the agreement-metric confidence intervals (0 to skip)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Seed for the agreement-metric bootstrap, for reproducible intervals",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    results = run_blinded_audit(
        args.human,
        args.judge,
        args.output_dir,
        metric=args.metric,
        contest_quantile=args.contest_quantile,
        n_boot=args.n_boot,
        random_state=args.random_state,
    )
    print(audit_report(results))
    print(f"Wrote audit bundle to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
