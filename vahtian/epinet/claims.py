"""Scientific claims check — turn evaluation outputs into plain-language gates.

The methods produce numbers; this layer states, in words a non-statistician can
act on, *what may and may not be claimed* from them. It is deliberately
conservative: it never says a model "works", only whether a specific, narrow
claim clears a specific bar.

Five gates plus a standing caveat:

1. **Permutation null** — "signal above null" vs "signal not detected".
2. **Split sensitivity** — random vs community(-aware) split; a large drop means
   the headline number leaned on leakage between connected cases.
3. **Baseline** — does the model beat the no-information floor?
4. **External validation** — run or not, and how far performance transported.
5. **Graph semantics** — whether graph-shaped claims are licensed by the edge
   meaning and edge timing.
6. **Clinical-utility caveat** — generated into every report, always.

Each gate returns a ``status`` and a one-line ``statement``; :func:`claims_markdown`
renders them for the model card and :func:`scientific_claims_check` assembles the
machine-readable record written as ``claims_check.json``.
"""

from __future__ import annotations

import math

# The standing caveat, generated into every report and never suppressible by the
# theme. Phrased as preconditions, not reassurance.
CLINICAL_CAVEAT = (
    "Do not claim clinical, public-health, or operational utility from this "
    "report unless ALL of the following hold: (1) the model is validated on "
    "independent, outcome-linked data from the deployment population; (2) it is "
    "calibrated in that population, not only discriminating; (3) it has been "
    "evaluated prospectively against the actual decision it would inform; "
    "(4) subgroup performance, bias, and failure modes have been examined; and "
    "(5) data governance, consent, and lawful basis have been reviewed. This is "
    "a research and education demonstrator: absent the above, treat every number "
    "here as exploratory evidence, not a decision."
)

_SIG_ALPHA = 0.05          # permutation gate threshold
_SPLIT_DROP_WARN = 0.10    # AUROC drop random->community that flags leakage sensitivity
_BASELINE_MARGIN = 0.02    # AUROC margin over the no-information floor to count as "beats"


def _f(x, d=3):
    return "—" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def _roc_auc_spread(metrics: dict | None) -> tuple[float | None, int | None]:
    """Per-re-split AUROC std and the re-split count, or (None, None) if absent.

    The std lives in ``iteration_summary`` (only present for >1 re-split), the
    count in ``n_iterations``. Together they give the standard error of a mean
    AUROC, which is what lets a gate put an error bar on its own threshold.
    """
    if not metrics:
        return None, None
    std = metrics.get("iteration_summary", {}).get("roc_auc", {}).get("std")
    n = metrics.get("n_iterations")
    return std, n


def _diff_se(std_a, n_a, std_b, n_b) -> float | None:
    """Standard error of a difference of two mean AUROCs, or None if undefined.

    Treats the two means as independent — a conservative simplification: when the
    estimates are positively correlated (shared splits) this OVERstates the
    spread, widening the band, so the gate errs toward "not resolvable" rather
    than over-claiming. Underlying re-splits overlap (Nadeau & Bengio), so even
    each per-estimate SE is itself a mild underestimate; read the band as a
    rough resolvability check, not an exact CI.
    """
    if None in (std_a, n_a, std_b, n_b) or n_a < 2 or n_b < 2:
        return None
    return math.sqrt(std_a ** 2 / n_a + std_b ** 2 / n_b)


def permutation_gate(metrics: dict) -> dict:
    """Gate the headline metric against the label-permutation null."""
    perm = metrics.get("permutation_test")
    if not perm:
        return {
            "status": "not run",
            "statement": "No permutation null was run, so the score has not been "
            "tested against chance. Enable it before drawing any signal claim.",
            "p_values": {},
        }
    pmetrics = perm.get("metrics", {})
    p_values = {k: v.get("p_value") for k, v in pmetrics.items()}
    headline = pmetrics.get("roc_auc") or next(iter(pmetrics.values()), {})
    p = headline.get("p_value")
    observed = headline.get("observed_mean")
    null_mean = headline.get("null_mean")
    if p is not None and p < _SIG_ALPHA:
        status = "signal above null"
        statement = (
            f"Signal above null: the observed score ({_f(observed)}) sits outside "
            f"the permuted-label null (mean {_f(null_mean)}, p={_f(p)} over "
            f"{perm.get('n_permutations')} permutations). The features carry "
            "detectable outcome information on this dataset."
        )
    else:
        status = "signal not detected"
        statement = (
            f"Signal NOT detected: the observed score ({_f(observed)}) is consistent "
            f"with the permuted-label null (mean {_f(null_mean)}, p={_f(p)}). On this "
            "dataset the features carry no outcome signal distinguishable from chance "
            "— do not report the headline metric as evidence of an effect."
        )
    return {"status": status, "statement": statement, "p_values": p_values}


def split_gate(split_comparison: dict | None) -> dict:
    """Compare random vs community(-aware) split to expose leakage sensitivity."""
    if not split_comparison:
        return {
            "status": "not compared",
            "statement": "Random and community-aware splits were not compared.",
            "random_roc_auc": None, "community_roc_auc": None, "drop": None,
        }
    rnd = split_comparison.get("random", {}).get("roc_auc")
    com = split_comparison.get("community", {}).get("roc_auc")
    rnd_std = split_comparison.get("random", {}).get("roc_auc_std")
    com_std = split_comparison.get("community", {}).get("roc_auc_std")
    n_iter = split_comparison.get("n_iterations")
    drop = (rnd - com) if (rnd is not None and com is not None) else None

    # Uncertainty of the drop: each per-split AUROC is a mean over n_iter
    # re-splits, so the drop is a difference of two means. Enough to tell a real
    # drop from iteration noise (see _diff_se for the caveats).
    drop_se = _diff_se(rnd_std, n_iter, com_std, n_iter)
    half = 2 * drop_se if drop_se is not None else None  # ~95% half-width
    band = f" (±{_f(half)} at ~2 SE over {n_iter} re-splits)" if half is not None else ""
    within_noise = half is not None and drop is not None and abs(drop) < half

    if drop is None:
        status, statement = "incomplete", "Split comparison did not produce both scores."
    elif drop >= _SPLIT_DROP_WARN:
        status = "leakage-sensitive"
        hedge = (
            " The band reaches below the flag threshold, so read this as suggestive "
            "rather than firm." if half is not None and (drop - half) < _SPLIT_DROP_WARN else ""
        )
        statement = (
            f"Leakage-sensitive: AUROC falls {_f(drop)}{band} from random ({_f(rnd)}) to "
            f"community-aware splitting ({_f(com)}). Connected cases share structure; "
            f"the community-aware number is the more honest estimate of generalization.{hedge}"
        )
    elif within_noise:
        status = "stable"
        statement = (
            f"Stable across splits: the {_f(drop)} AUROC change from random ({_f(rnd)}) "
            f"to community-aware ({_f(com)}) is within iteration noise{band}, so it is "
            "not distinguishable from zero — no evidence of leakage between connected cases."
        )
    else:
        status = "stable"
        statement = (
            f"Stable across splits: AUROC changes only {_f(drop)}{band} from random "
            f"({_f(rnd)}) to community-aware ({_f(com)}), so the headline is not "
            "driven by leakage between connected cases."
        )
    return {"status": status, "statement": statement,
            "random_roc_auc": rnd, "community_roc_auc": com,
            "drop": drop, "drop_se": drop_se}


def _paired_baseline_gate(paired: dict) -> dict:
    """Verdict from a paired model-vs-floor margin CI (the resolvable-at-n form)."""
    mean = paired["mean_margin"]
    lo, hi = paired["margin_ci_lower"], paired["margin_ci_upper"]
    thr = paired.get("threshold", _BASELINE_MARGIN)
    k = paired["n_pairs"]
    ci = f"[{_f(lo)}, {_f(hi)}]"
    base = (f"paired per-split margin {_f(mean)}, 95% CI {ci} over {k} shared "
            f"splits ({paired.get('correction', 'resampled-t')})")
    if lo >= thr:
        status = "beats floor"
        statement = (f"Beats the no-information floor: {base}, entirely above the "
                     f"{_f(thr)} line.")
        resolvable = True
    elif hi < thr:
        status = "at floor"
        statement = (f"At the no-information floor: {base}, below the {_f(thr)} line. "
                     "The representation adds no measurable signal — do not claim predictive value.")
        resolvable = True
    else:
        status = "not resolvable"
        statement = (f"Not resolvable at this n: {base}, straddles the {_f(thr)} line, so "
                     "the data cannot say whether the model beats the floor. Do not claim "
                     "predictive value on this evidence — gather more before deciding.")
        resolvable = False
    return {"status": status, "statement": statement, "margin": mean,
            "margin_ci": [lo, hi], "n_pairs": k, "resolvable_at_this_n": resolvable,
            "model_representation": paired.get("model_representation")}


def baseline_gate(
    baseline_metrics: dict | None,
    model_metrics: dict,
    *,
    paired: dict | None = None,
) -> dict:
    """Does the model clear the no-information floor under the same harness?

    When a ``paired`` model-vs-floor margin is supplied (per-split Δ on identical
    splits, see ``baselines._paired_baseline_margin``), the verdict is read off
    that paired CI: it clears the threshold, sits below it, or *straddles* it — in
    which case the honest answer is "not resolvable at this n" rather than a bright
    pass/fail. This is preferred over the scalar fallback because pairing cancels
    split-difficulty variance and guarantees the two legs share the same splits.
    """
    if paired is not None and paired.get("n_pairs", 0) >= 2 and paired.get("mean_margin") is not None:
        return _paired_baseline_gate(paired)
    if not baseline_metrics:
        return {"status": "not run",
                "statement": "No baseline comparison was run.", "margin": None}
    floor = baseline_metrics.get("no_information", {}).get("roc_auc")
    model = model_metrics.get("roc_auc")
    if floor is None or model is None:
        return {"status": "incomplete",
                "statement": "Baseline comparison is missing an AUROC.", "margin": None}
    margin = model - floor

    # Error bar on the margin, so "beats floor" vs "at floor" reports whether the
    # line is resolvable at this n, not just which side of 0.02 the point lands on.
    model_std, n_model = _roc_auc_spread(model_metrics)
    floor_std, n_floor = _roc_auc_spread(baseline_metrics.get("no_information"))
    margin_se = _diff_se(model_std, n_model, floor_std, n_floor)
    half = 2 * margin_se if margin_se is not None else None  # ~95% half-width
    band = f" (±{_f(half)} at ~2 SE)" if half is not None else ""
    within_noise = half is not None and margin < half

    if margin >= _BASELINE_MARGIN:
        status = "beats floor"
        hedge = (
            " The band reaches below the margin threshold, so treat this as "
            "suggestive rather than firm." if half is not None and (margin - half) < _BASELINE_MARGIN else ""
        )
        statement = (
            f"Beats the no-information floor: model AUROC {_f(model)} vs "
            f"{_f(floor)} (margin {_f(margin)}{band}).{hedge}"
        )
    elif within_noise:
        status = "at floor"
        statement = (
            f"At the no-information floor: model AUROC {_f(model)} vs {_f(floor)} "
            f"(margin {_f(margin)}{band}). The margin is within iteration noise — not "
            "distinguishable from zero at this n — so do not claim predictive value."
        )
    else:
        status = "at floor"
        statement = (
            f"At the no-information floor: model AUROC {_f(model)} vs {_f(floor)} "
            f"(margin {_f(margin)}{band}). The representation adds no measurable signal "
            "over the floor — do not claim predictive value."
        )
    return {"status": status, "statement": statement, "margin": margin,
            "margin_se": margin_se, "model_roc_auc": model, "floor_roc_auc": floor}


def external_validation_gate(extval: dict | None) -> dict:
    """Report transport to an independent cohort (or that it was not tested)."""
    if not extval:
        return {
            "status": "not run",
            "statement": "No external validation was run; external validity is "
            "unknown. In-sample scores do not establish that the model transports.",
            "drift": None,
        }
    internal = (extval.get("internal") or {}).get("roc_auc")
    external = (extval.get("external") or {}).get("roc_auc")
    drift = (extval.get("drift_internal_minus_external") or {}).get("roc_auc")
    statement = (
        f"External validation run: internal AUROC {_f(internal)} vs external "
        f"{_f(external)} (drift {_f(drift)}). External performance is the honest "
        "estimate; large positive drift means the model does not transport."
    )
    return {"status": "run", "statement": statement,
            "internal_roc_auc": internal, "external_roc_auc": external, "drift": drift}


def graph_semantics_gate(graph_semantics: dict | None = None) -> dict:
    """Gate graph-shaped claims by edge meaning and edge timing.

    A similarity graph is a feature transformation, not evidence that relations
    transmitted risk or information. Observed relation graphs can license
    network claims only when the edge was knowable before the outcome.
    """
    graph_semantics = graph_semantics or {}
    mode = graph_semantics.get("graph_mode") or graph_semantics.get("mode") or "none"
    semantics = graph_semantics.get("semantics") or "unspecified"
    timing = graph_semantics.get("edge_timing") or graph_semantics.get("timing") or "unknown"
    has_edge_table = bool(graph_semantics.get("has_edge_table"))

    if mode == "none" and semantics == "unspecified" and not has_edge_table:
        return {
            "status": "not applicable",
            "statement": "No graph relation was supplied or synthesized; do not make a relational network claim.",
            "semantics": semantics,
            "edge_timing": timing,
        }
    if mode == "similarity" or semantics == "similarity":
        return {
            "status": "exploratory only",
            "statement": (
                "Similarity graph: edges are derived from feature proximity. Treat graph outputs as "
                "feature-space exploration; do not claim relation, contagion, referral, or message-passing "
                "effects from these edges."
            ),
            "semantics": "similarity",
            "edge_timing": timing,
        }
    if semantics == "observed_relation":
        if timing == "pre_outcome":
            return {
                "status": "relation documented",
                "statement": (
                    "Observed relation graph with pre-outcome edge timing documented. Network-aware "
                    "predictive claims may be tested, but still depend on community-aware and external validation."
                ),
                "semantics": semantics,
                "edge_timing": timing,
            }
        return {
            "status": "timing unresolved",
            "statement": (
                "Observed relation graph supplied, but edge timing is not documented as pre-outcome. "
                "Do not claim predictive graph utility until post-outcome or mixed-time leakage is ruled out."
            ),
            "semantics": semantics,
            "edge_timing": timing,
        }
    return {
        "status": "not specified",
        "statement": (
            "Graph semantics were not specified. State whether edges are feature-derived similarity "
            "or observed relations before making graph-shaped epidemiological claims."
        ),
        "semantics": semantics,
        "edge_timing": timing,
    }


def scientific_claims_check(
    metrics: dict | None,
    *,
    split_comparison: dict | None = None,
    baseline_metrics: dict | None = None,
    baseline_paired: dict | None = None,
    external_validation: dict | None = None,
    graph_semantics: dict | None = None,
    model_trained: bool = True,
) -> dict:
    """Assemble the full machine-readable claims record."""
    graph = graph_semantics_gate(graph_semantics)
    if not model_trained or not metrics:
        return {
            "model_trained": False,
            "headline": "No outcome model was trained (descriptive report only); "
            "no predictive claim can be made.",
            "permutation": {"status": "not run", "statement": "No model.", "p_values": {}},
            "split_comparison": split_gate(None),
            "baselines": {"status": "not run", "statement": "No model.", "margin": None},
            "external_validation": external_validation_gate(external_validation),
            "graph_semantics": graph,
            "clinical_caveat": CLINICAL_CAVEAT,
        }

    perm = permutation_gate(metrics)
    split = split_gate(split_comparison)
    base = baseline_gate(baseline_metrics, metrics, paired=baseline_paired)
    extv = external_validation_gate(external_validation)

    # Headline: the most conservative reading. Any failed gate downgrades it.
    if perm["status"] == "signal not detected" or base["status"] == "at floor":
        headline = ("No usable signal: the model is at or below chance/no-information "
                    "on this dataset. Report this as a negative result, not a model.")
    elif base["status"] == "not resolvable":
        headline = ("Inconclusive vs the no-information floor: the model-minus-floor "
                    "margin is not resolvable at this sample size. Neither claim it "
                    "works nor that it fails — the data cannot tell yet.")
    elif split["status"] == "leakage-sensitive":
        headline = ("Signal present but leakage-sensitive: trust the community-aware "
                    "split number, and do not generalize beyond connected structure "
                    "without external validation.")
    elif extv["status"] != "run":
        headline = ("Signal above null on this dataset, but external validity is "
                    "untested. Exploratory only until validated on independent data.")
    else:
        headline = ("Signal above null and externally validated on the provided "
                    "cohort; interpret within that cohort's limits and the caveat below.")

    return {
        "model_trained": True,
        "headline": headline,
        "permutation": perm,
        "split_comparison": split,
        "baselines": base,
        "external_validation": extv,
        "graph_semantics": graph,
        "clinical_caveat": CLINICAL_CAVEAT,
    }


def claims_markdown(claims: dict) -> str:
    """Render the claims check as a markdown section for the model card."""
    out = ["## Scientific claims check", "",
           f"**Headline:** {claims['headline']}", ""]
    graph = claims.get("graph_semantics") or graph_semantics_gate(None)
    rows = [
        ("Permutation null", claims["permutation"]["status"], claims["permutation"]["statement"]),
        ("Split sensitivity", claims["split_comparison"]["status"],
         claims["split_comparison"]["statement"]),
        ("Baseline floor", claims["baselines"]["status"], claims["baselines"]["statement"]),
        ("External validation", claims["external_validation"]["status"],
         claims["external_validation"]["statement"]),
        ("Graph semantics", graph["status"], graph["statement"]),
    ]
    out.append("| gate | status | reading |")
    out.append("| --- | --- | --- |")
    for name, status, statement in rows:
        out.append(f"| {name} | **{status}** | {statement} |")
    out += ["", "### Do not over-claim", "", claims["clinical_caveat"], ""]
    return "\n".join(out)
