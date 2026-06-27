---
name: iterative-improvement
description: >-
  Drive durable improvement to a codebase, model, or research artifact with an
  eval-driven loop: set a goal and stop condition, lock a baseline, change one
  thing, measure against held-out evals and regression tests, keep only verified
  wins, and capture the workflow as a reusable skill. Use when asked to "improve
  X", to run an iterative loop of improvements, to set up evaluation-first
  development, or when work involves goals, loops, ablations, benchmarks, or
  regression tests.
---

# Iterative, eval-driven improvement

Improvement is only real if a check that could have failed didn't. Optimize the
*measurement* first, then the artifact — otherwise you are guessing.

## The loop

1. **Goal + stop condition.** State what "better" means and when to stop
   ("balanced accuracy up with calibration intact", "R CMD check clean",
   "5 iterations"). A goal without a stop condition runs forever or quits early.
2. **Baseline + eval harness, first.** Before changing anything, build/seed the
   evaluation and record the baseline number. No eval → no improvement, only
   opinion. Reuse the project's tests as a regression floor.
3. **One change at a time.** Smallest coherent change. Multiple simultaneous
   changes make gains unattributable — ablate to know what actually helped.
4. **Measure vs the eval.** Held-out / cross-validated metric for the gain;
   full test suite for safety. Both must be green.
5. **Adopt only verified, transporting wins.** Keep a change only if it beats
   baseline *and* generalizes (held-out, multiple seeds), not just on the data
   you tuned on. If a gain is within noise, say so — make it opt-in or drop it.
6. **Commit → PR → CI gate → merge.** Land each verified step on its own; let CI
   re-verify across environments. Then sync and repeat.
7. **Capture it.** When a workflow recurs, write it down as a skill so the method
   compounds instead of being re-derived.

## Tools that power the loop

- **Goals** give the loop a destination and stop condition.
- **Self-paced loops** keep work moving across CI waits and context resets
  without supervision; re-check state on each wake and act on what's unblocked.
- **Deep research** grounds a *new* direction in evidence before you build.
- **Skills** capture a proven method so it is reusable.

## Honesty gates (what keeps the loop trustworthy)

- **Beat a baseline and a null**, not just your own previous attempt.
- **Report marginal/negative results.** A change that doesn't transport should be
  opt-in or reverted, and the finding stated — e.g. decision-threshold tuning on
  top of class weighting was within noise here, so it shipped opt-in, off by
  default, with the gain reported.
- **No silent scope creep.** Each iteration does the one thing the goal names.

## Anti-patterns

- **Goodhart / metric-chasing**: optimizing one number until it stops meaning
  anything. Track a small basket (incl. calibration, regressions), not one scalar.
- **Overfitting the eval**: tuning against the test set, or peeking. Hold data out;
  refresh evals; never select on the final test.
- **No regression net**: "improving" while silently breaking something else. Keep
  the suite green every iteration.
- **Big-bang changes**: many edits at once with no ablation — unattributable and
  hard to revert.

## When to stop

Stop when the goal's condition holds, when further iterations return within-noise
gains (loop-until-dry), or at the iteration cap. Report what changed, what was
verified, and what was tried and rejected.
