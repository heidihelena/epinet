# Brock vs Mayo vs NTOG — full comparison on the synthetic cohort

The real LIDC cohort cannot run these scores (no demographics). The synthetic
cohort carries the clinical inputs they need, so this is where the three scores
can actually be put side by side. `score_comparison_synthetic.py` does it.

## How circularity is avoided

The synthetic risk *tier* is derived from Brock, so it cannot be the target —
Brock would win by construction. Instead the generator now draws an
**independent latent malignancy label** (`LatentMalignant`) from a deliberately
different functional form (saturating √diameter, its own weights, Bernoulli
noise) on its own RNG stream. No score has privileged access to it, and none can
fit it exactly. Prevalence: 39/117 (33%).

## Scores

- **Brock** — PanCan probability (established).
- **Mayo** — Swensen pretest probability (established).
- **NTOG** — post-CT research score, **Nodule + Growth domains only**, normalised
  by available weight per the NTOG rule (*"missing domains are not imputed"*).
  The Person/Smoking-architecture/Emerging-risk domains are unavailable even in
  the synthetic cohort and are honestly omitted. N = banded max(Brock, Mayo);
  G = banded volume-doubling-time.

## Result

**Rank concordance (Spearman):** Brock–Mayo 0.78, Mayo–NTOG 0.87, Brock–NTOG 0.66.
NTOG tracks Mayo most closely and departs from Brock the most — because its
Growth domain moves nodules Brock cannot see.

**Discrimination vs the independent latent truth** (bootstrap 95% CI, n=117):

| score | AUC | 95% CI |
|-------|-----|--------|
| Brock | 0.703 | 0.60–0.80 |
| Mayo  | 0.716 | 0.62–0.81 |
| NTOG  | 0.711 | 0.61–0.81 |

- NTOG − Brock: ΔAUC +0.009, CI [−0.084, +0.110], p = 0.88
- NTOG − Mayo: ΔAUC −0.005, p = 0.87
- Brock − Mayo: ΔAUC −0.013, p = 0.76

**All three are statistically indistinguishable.** And the NTOG Growth domain
does what it is designed to: the amount by which NTOG re-ranks a nodule relative
to Brock correlates with its growth rate (r = 0.24) — faster growth pushes a
nodule up the NTOG ranking.

## Reading — what this can and cannot say

- **It can say:** the three scores compute correctly, agree strongly in ranking
  (0.66–0.87), and discriminate a held-out latent target comparably; NTOG's
  distinctive growth contribution is real and isolable.
- **It cannot say which is better.** The AUCs are a property of *this generator's*
  latent label, which I chose. Decisively: my latent truth contains **no growth
  term**, so NTOG's Growth domain can only add variance here, not signal — that is
  exactly why NTOG does not pull ahead. Had the true biology rewarded growth,
  the growth-aware score would gain. The ranking is an artifact of the generator,
  not evidence of validity.

That last point is the honest core: a synthetic head-to-head demonstrates
machinery and behaviour, never validity. The real adjudication of NTOG against
Brock/Mayo requires a demographically complete cohort with tissue or follow-up
outcomes — a screening cohort such as NLST (smoking/age present, benign
representation adequate, growth measurable). This synthetic run is the
methods-demonstration substitute, and it is labelled as such.

## Boundaries

- Synthetic cohort (117 nodules); AUC CIs are wide.
- NTOG is the N+G subset only; P/S/E domains are not evaluated.
- The latent label is one generator choice; results shift if its form changes
  (that sensitivity *is* the lesson).
