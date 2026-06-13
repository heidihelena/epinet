"""Scientific claims check — turn evaluation outputs into plain-language gates.

The methods produce numbers; this layer states, in words a non-statistician can
act on, *what may and may not be claimed* from them. It is deliberately
conservative: it never says a model "works", only whether a specific, narrow
claim clears a specific bar.

Four gates plus a standing caveat:

1. **Permutation null** — "signal above null" vs "signal not detected".
2. **Split sensitivity** — random vs community(-aware) split; a large drop means
   the headline number leaned on leakage between connected cases.
3. **Baseline** — does the model beat the no-information floor?
4. **External validation** — run or not, and how far performance transported.
5. **Clinical-utility caveat** — generated into every report, always.

Each gate returns a ``status`` and a one-line ``statement``; :func:`claims_markdown`
renders them for the model card and :func:`scientific_claims_check` assembles the
machine-readable record written as ``claims_check.json``.
"""

from __future__ import annotations

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
    drop = (rnd - com) if (rnd is not None and com is not None) else None
    if drop is None:
        status, statement = "incomplete", "Split comparison did not produce both scores."
    elif drop >= _SPLIT_DROP_WARN:
        status = "leakage-sensitive"
        statement = (
            f"Leakage-sensitive: AUROC falls {_f(drop)} from random ({_f(rnd)}) to "
            f"community-aware splitting ({_f(com)}). Connected cases share structure; "
            "the community-aware number is the more honest estimate of generalization."
        )
    else:
        status = "stable"
        statement = (
            f"Stable across splits: AUROC changes only {_f(drop)} from random "
            f"({_f(rnd)}) to community-aware ({_f(com)}), so the headline is not "
            "driven by leakage between connected cases."
        )
    return {"status": status, "statement": statement,
            "random_roc_auc": rnd, "community_roc_auc": com, "drop": drop}


def baseline_gate(baseline_metrics: dict | None, model_metrics: dict) -> dict:
    """Does the model clear the no-information floor under the same harness?"""
    if not baseline_metrics:
        return {"status": "not run",
                "statement": "No baseline comparison was run.", "margin": None}
    floor = baseline_metrics.get("no_information", {}).get("roc_auc")
    model = model_metrics.get("roc_auc")
    if floor is None or model is None:
        return {"status": "incomplete",
                "statement": "Baseline comparison is missing an AUROC.", "margin": None}
    margin = model - floor
    if margin >= _BASELINE_MARGIN:
        status = "beats floor"
        statement = (
            f"Beats the no-information floor: model AUROC {_f(model)} vs "
            f"{_f(floor)} (margin {_f(margin)})."
        )
    else:
        status = "at floor"
        statement = (
            f"At the no-information floor: model AUROC {_f(model)} vs {_f(floor)} "
            f"(margin {_f(margin)}). The representation adds no measurable signal "
            "over the floor — do not claim predictive value."
        )
    return {"status": status, "statement": statement, "margin": margin,
            "model_roc_auc": model, "floor_roc_auc": floor}


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


def scientific_claims_check(
    metrics: dict | None,
    *,
    split_comparison: dict | None = None,
    baseline_metrics: dict | None = None,
    external_validation: dict | None = None,
    model_trained: bool = True,
) -> dict:
    """Assemble the full machine-readable claims record."""
    if not model_trained or not metrics:
        return {
            "model_trained": False,
            "headline": "No outcome model was trained (descriptive report only); "
            "no predictive claim can be made.",
            "permutation": {"status": "not run", "statement": "No model.", "p_values": {}},
            "split_comparison": split_gate(None),
            "baselines": {"status": "not run", "statement": "No model.", "margin": None},
            "external_validation": external_validation_gate(external_validation),
            "clinical_caveat": CLINICAL_CAVEAT,
        }

    perm = permutation_gate(metrics)
    split = split_gate(split_comparison)
    base = baseline_gate(baseline_metrics, metrics)
    extv = external_validation_gate(external_validation)

    # Headline: the most conservative reading. Any failed gate downgrades it.
    if perm["status"] == "signal not detected" or base["status"] == "at floor":
        headline = ("No usable signal: the model is at or below chance/no-information "
                    "on this dataset. Report this as a negative result, not a model.")
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
        "clinical_caveat": CLINICAL_CAVEAT,
    }


def claims_markdown(claims: dict) -> str:
    """Render the claims check as a markdown section for the model card."""
    out = ["## Scientific claims check", "",
           f"**Headline:** {claims['headline']}", ""]
    rows = [
        ("Permutation null", claims["permutation"]["status"], claims["permutation"]["statement"]),
        ("Split sensitivity", claims["split_comparison"]["status"],
         claims["split_comparison"]["statement"]),
        ("Baseline floor", claims["baselines"]["status"], claims["baselines"]["statement"]),
        ("External validation", claims["external_validation"]["status"],
         claims["external_validation"]["statement"]),
    ]
    out.append("| gate | status | reading |")
    out.append("| --- | --- | --- |")
    for name, status, statement in rows:
        out.append(f"| {name} | **{status}** | {statement} |")
    out += ["", "### Do not over-claim", "", claims["clinical_caveat"], ""]
    return "\n".join(out)
