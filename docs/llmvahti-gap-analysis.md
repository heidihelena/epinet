# LLMvahti — cited gap analysis of the LLM-as-judge evaluation landscape

*Deep-research report, 2026-06-12. Five parallel search angles; load-bearing claims
adversarially verified against primary sources. Claims that failed verification were
corrected or excluded (see Verification notes).*

## Verdict: **partially met — with one genuinely open niche**

The market need is **not unmet in general**: "LLM evaluation" is heavily funded and
crowded, and *judge-vs-human calibration tracking* — which a year ago looked like a
gap — is now shipping in mainstream tools. But the specific vahti stance —
**human as primary rater, LLM judge as blinded second rater, with per-verdict
contestability (how close the verdict was to flipping, and which rubric criterion
drove it)** — exists nowhere as a product feature. It is an active *research* frontier
(2025–2026) that no production tool has operationalized.

---

## 1. What is already covered (do not compete here)

**Generic eval harnesses are saturated.** promptfoo (acquired by OpenAI, March 9 2026
[[OpenAI](https://openai.com/index/openai-to-acquire-promptfoo/)]), LangSmith,
Braintrust, DeepEval/Confident AI, Ragas, Langfuse, Comet Opik, Arize Phoenix, MLflow,
W&B Weave, UK AISI Inspect, lm-eval-harness, HELM. Funding confirms the crowding:
Braintrust $36M Series A (a16z, Oct 2024, ~$150M post
[[Braintrust](https://www.braintrust.dev/blog/announcing-series-a)]), Galileo $45M
Series B (Oct 2024
[[PR Newswire](https://www.prnewswire.com/news-releases/galileo-raises-45m-series-b-funding-to-bring-evaluation-intelligence-to-generative-ai-teams-everywhere-302276383.html)]),
Arize $70M Series C (Feb 2025
[[PR Newswire](https://www.prnewswire.com/news-releases/arize-ai-secures-70m-series-c-to-fix-ais-biggest-problem-making-llms-and-ai-agents-work-in-the-real-world-302381601.html)]),
Confident AI $2.2M seed (Mar 2025
[[Confident AI](https://www.confident-ai.com/blog/how-i-closed-confident-ais-2-2m-seed-round-in-5-days)]).

**Judge–human agreement tracking is now mainstream.** This is the part of the original
thesis that is *no longer* a gap:

- **LangSmith "Align Evals"** (July 29 2025): explicit alignment score between
  human-graded data and LLM judge scores, tracked over prompt iterations
  [[LangChain](https://blog.langchain.com/introducing-align-evals/)].
- **DeepEval/Confident AI**: mandatory calibration against 25–50 human-annotated
  golden examples, TP/TN/FP/FN reporting
  [[DeepEval](https://deepeval.com/guides/guides-llm-as-a-judge)].
- **Langfuse "Score Analytics"** (Nov 2025): Pearson/Spearman correlation, MAE, RMSE
  between human annotations and automated evaluators
  [[Langfuse](https://langfuse.com/docs/evaluation/scores/score-analytics)].
- **Braintrust**: human-vs-judge score comparison on the same traces; annotation
  queues as the gatekeeping layer
  [[Braintrust](https://www.braintrust.dev/articles/best-human-in-the-loop-llm-evaluation-platforms-2026)].

**Human annotation with LLM assist is mainstream.** Prodigy ("LLM pre-annotation for
human review"), Argilla, Label Studio (built-in Cohen's kappa between human raters),
LangSmith/Langfuse annotation queues.

## 2. What is genuinely open (the LLMvahti niche)

Across every production tool surveyed, **no tool ships per-verdict contestability**:

1. **No flip-distance.** No tool answers "how close was this judge verdict to going
   the other way?" — no decision margins, no grey-zone flagging by default, no
   segregation of borderline verdicts.
2. **No per-verdict criterion attribution.** No tool shows *which rubric criterion*
   drove a verdict or would flip it (the value-of-information question).
3. **No formal blinding.** Every "human-primary" workflow surveyed shows the human
   the LLM's pre-annotation (Prodigy, Argilla) or routes by judge confidence
   (Maxim, Langfuse). **None implements the blinded-second-rater design** —
   independent ratings compared afterwards via inter-rater statistics — which is the
   standard rigorous design in clinical research and exactly citevahti's pattern.
4. **Research exists, productization doesn't.** Conformal prediction intervals for
   LLM judges (EMNLP 2025
   [[ACL](https://aclanthology.org/2025.emnlp-main.569/)]), linear-probe judge
   calibration ([arXiv 2512.22245](https://arxiv.org/abs/2512.22245)), IRT-based
   judge reliability diagnosis ([arXiv 2602.00521](https://arxiv.org/abs/2602.00521)).
   The closest library, Haize Labs' Verdict
   ([arXiv 2502.18018](https://arxiv.org/abs/2502.18018);
   [GitHub](https://github.com/haizelabs/verdict)), names miscalibrated uncertainty
   as a problem but optimizes judge *accuracy*, not contestability or human primacy.
5. **"Contestable AI by design" has no LLM-eval implementation.** The framework is
   established academically
   [[Springer](https://link.springer.com/article/10.1007/s11023-022-09611-z)] but
   no general-purpose LLM-judge tool operationalizes it.

## 3. Evidence the need is real (verified)

- **Judges are unreliable enough to need this.** RAND's Judge Reliability Harness:
  *no* evaluated judge was uniformly reliable across benchmarks; consistency breaks
  on formatting, paraphrase, and verbosity changes
  [[RAND TLA4547-1](https://www.rand.org/pubs/tools/TLA4547-1.html)].
- **Self-preference bias is mechanistic**: LLM evaluators recognize and favor their
  own generations, with a linear correlation between self-recognition and
  self-preference (NeurIPS 2024
  [[arXiv 2404.13076](https://arxiv.org/abs/2404.13076)]).
- **Twelve distinct judge biases** including position and verbosity quantified at
  ICLR 2025 [[arXiv 2410.02736](https://arxiv.org/abs/2410.02736)].
- **Expert domains are where judges fail**: judge–expert agreement 68% (dietetics)
  and 64% (mental health) vs. 75%/72% inter-expert agreement
  [[arXiv 2410.20266](https://arxiv.org/abs/2410.20266)] — against ~80–85%
  agreement (human–human 81%) on general chat tasks
  [[arXiv 2306.05685](https://arxiv.org/abs/2306.05685)]. **The vahti audience
  (clinical, registry, scientific) sits exactly in the failing regime.**
- **Practitioner pain**: 93% of teams report struggling with LLM-judge consistency,
  cost, or bias (Galileo State of Eval Engineering, Q1 2026
  [[Galileo](https://galileo.ai/state-of-eval-engineering-report)]).
- **Regulated domains mandate the stance**: 2025 clinical consensus frameworks
  (e.g. CLEVER) require human oversight, regular judge audits, uncertainty
  calibration, and audit trails for LLM evaluation in healthcare
  [[PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12677871/)].

## 4. Positioning recommendations for LLMvahti

1. **Do not build an eval harness.** Build the *audit layer that sits on top of*
   existing judges — promptfoo/LangSmith/DeepEval users bring their verdicts,
   LLMvahti tells them which ones to distrust.
2. **Lead with contestability, not calibration.** Calibration dashboards are being
   commoditized (Align Evals, Score Analytics). Flip-distance + criterion-level
   value-of-information per verdict is the unduplicated feature — and it is EpiNet's
   existing closed-form machinery pointed at judge outputs.
3. **Make the blinded-second-rater design the brand.** Human rates first, blind to
   the judge; judge rates blind to the human; LLMvahti reports Cohen's κ /
   Krippendorff's α, disagreement clusters, and the contestable grey zone. No
   surveyed tool does formal blinding — and it is the same design citevahti uses
   for citation integrity, making the family story coherent.
4. **Target expert/regulated domains first** (clinical NLP, registry methods,
   scientific claim grading) — where judges demonstrably fail (64–68% agreement),
   where consensus frameworks already *require* human-primary auditing, and where
   the vahti/EpiNet governance posture (fail-closed, audit-ledger, no-compliance-
   claims) is already credible.
5. **Borrow the research, cite it, don't reinvent it**: conformal intervals
   (EMNLP 2025) for per-verdict uncertainty bands; EpiNet's Rocchio flip-distance
   on embedded (prompt, response, rubric) space for contestability — the additive,
   federatable statistics carry over unchanged.
6. **Window: act within ~6–12 months.** Align Evals and Score Analytics show
   platforms moving up this stack; contestability is publishable now and
   defensible as the wedge, but the calibration half is being absorbed.

## Verification notes

Adversarially verified (3 independent checking passes against primary sources):
- **Verified**: OpenAI–promptfoo acquisition; Align Evals; Humanloop sunset
  (Sept 8 2025); Verdict library; RAND harness; Langfuse Score Analytics; funding
  rounds for Braintrust/Galileo/Arize/Confident AI; Galileo 93% stat; NeurIPS 2024
  self-preference; ICLR 2025 bias quantification; EMNLP 2025 conformal prediction.
- **Corrected**: expert-domain agreement is 64–68% (not "60–68%"); Zheng et al.
  GPT-4 agreement ~83–85% in main settings; Braintrust's "OpenAI participation"
  was Greg Brockman personally, not OpenAI institutionally.
- **Excluded as unverifiable**: "Judge's Verdict" 85%-vs-81–82% figures (paper
  exists, arXiv 2510.09738, but reports Cohen's κ, not those percentages);
  Patronus $50M Series B and March-2026 Agent Evaluation Suite (secondary
  sources only); "frontier models fail 50%+ of bias tests" (vendor blogs only —
  treat as marketing until traced to the underlying study).
