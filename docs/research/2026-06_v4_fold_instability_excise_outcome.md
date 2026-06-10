# v4 — fold-instability deep dive: the excise hypothesis is falsified, and that's the interesting part

**Date:** 2026-06-10
**Branch:** `claude/v4-fold-instability`
**Status:** complete — hypothesis falsified against the clean baseline; richer mechanism identified
**See also:** [`2026-06_v3.0_phase3_closing_summary.md`](2026-06_v3.0_phase3_closing_summary.md)

## The question

Phase 3 #2 (the seed-noise floor experiment) found that **folds 3 (2022-23) and 6 (2025-26) had 3-5× higher per-fold seed-stdev** than folds 1/2/4/5 — under the *original v1/v2 hyperparameters*. We documented the finding but never asked why.

The hypothesis: **fold_3's instability is the September 28, 2022 fuel excise restoration.** Australia halved the federal fuel excise (44.2 → 22.1 c/L) on March 30, 2022, then restored it with CPI catch-up (→ ~46 c/L) on September 29. Fold_3's test window (May 2022 – April 2023) spans both periods, but fold_3's *training data* ends April 2022 — the model never saw the change. With no feature encoding the tax level, the model extrapolates over a ~24 c/L structural break it can't see. That's a plausible source of seed-sensitive instability.

The test: add `cal_fuel_excise_cents_per_litre` to the calendar block, re-run the 6-seed seed-noise protocol, compare per-fold seed-stdev.

## The result: falsified — but the path there is the finding

### What we expected
Fold_3's seed-stdev drops sharply when the excise feature is added → hypothesis confirmed → roll the feature into production.

### What actually happened

Three-way per-fold seed-stdev comparison (the key table):

| Fold | OLD defaults (Phase 3 #2) | NEW tuned (no excise) | NEW tuned + excise |
|------|--------------------------:|----------------------:|-------------------:|
| fold_1 | 0.0544 | 0.0191 | 0.0250 |
| fold_2 | 0.0345 | 0.0307 | 0.0314 |
| **fold_3** | **0.1626** | **0.0436** | **0.0520** |
| fold_4 | 0.0330 | 0.0398 | 0.0339 |
| fold_5 | 0.0687 | 0.0454 | 0.0891 |
| fold_6 | 0.1782 | 0.1002 | 0.1954 |

The hyperparameter retune (Phase 3 #4, already shipped) **had already fixed fold_3's instability**:

- **Retune effect on fold_3: 0.1626 → 0.0436 (−73%).** The smaller, more-regularized trees (num_leaves 63→31, min_data_in_leaf 200→544, no row bagging, L1 penalty) simply don't overfit fold_3's within-window noise the way the v1/v2 trees did.
- **Excise-feature effect on fold_3: 0.0436 → 0.0520 (+19%).** Adding the feature on top of the retuned model makes fold_3 *slightly worse*, not better.

**Fold_3 was a regularization problem, not a missing-feature problem.** The original hypothesis correctly identified *which fold* was unstable and *when* the structural break was, but the mechanism was wrong: the instability was the under-regularized model overfitting, not the model lacking the excise signal.

### The trap I nearly fell into

The experiment script's first version compared against the Phase 3 #2 baseline (OLD spec defaults). Against *that* baseline, fold_3 with the excise feature looks like a 0.1626 → 0.052 = **−68% drop** — and the script dutifully printed "Hypothesis: CONFIRMED" for the first several seeds.

That comparison is **wrong**: it conflates two changes (the hyperparameter retune *and* the new feature) and attributes both to the feature. The clean comparison holds hyperparameters fixed and changes only the feature — and against that, the feature does nothing for fold_3 (+19% stdev, i.e. marginally worse).

This is itself a methodology lesson worth recording: **when testing a new feature on a model whose hyperparameters have also changed since the last baseline, the baseline must use the new hyperparameters.** Otherwise the retune's effect masquerades as the feature's effect. The corrected script (`v4_excise_fold_instability.py`) uses the hyperopt-validation per-seed MAE (new tuned defaults, no excise) as the baseline, and keeps the OLD-defaults numbers only for the explicit cumulative view.

## The excise feature is net-harmful

Beyond not helping fold_3, the feature actively hurts:

**Seed-stdev (feature-only effect):** worse on 4 of 6 folds (fold_1 +31%, fold_3 +19%, fold_5 +96%, fold_6 +95%); roughly flat on fold_2; only fold_4 improves (−15%).

**Mean MAE (feature-only effect):** worse on 5 of 6 folds — fold_6 by **+0.40 c/L**, a major regression.

The pattern is clearest on folds 5 and 6, where the excise rate is **near-constant within the test window** (49.6 → 50.6 c/L for fold_5; 50.8 → 51.6 for fold_6 — small CPI indexation steps, no structural break). There, the feature provides no signal but adds tree capacity, and the model overfits whatever spurious variation the seed produces against an essentially-constant column.

## The general principle

**A feature that is near-constant within a fold's test window can be net-negative.** It adds model capacity without adding signal for that fold, and the spare capacity gets spent fitting seed-dependent noise. The excise feature "works" (has variance to fit) only on folds that span the 2022 break; on every other fold it's dead weight that increases variance.

This generalizes the v3.0 finding: feature value isn't a property of the feature alone — it's a property of the feature *relative to the evaluation windows*. A feature can be informative globally and harmful per-fold.

## Fold_6 remains unexplained

The retune halved fold_6's instability (0.1782 → 0.1002) but it's still the most unstable fold by a wide margin, and the excise feature nearly doubled it again. Fold_6 (2025-26) is the recent oil-shock / 2026 price-spike period. Its instability is **not** a domestic-policy artifact — there's no excise-style lever the model can hook into. The next hypothesis for fold_6 would be Brent/crude volatility regime features (realized volatility of the upstream price series, or a regime indicator), not policy features.

## Decision

- **Do NOT ship the raw excise feature.** It's net-harmful: worsens both mean MAE and seed-stdev on most folds, and its one upside (fold_3 stability) was already delivered by the hyperparameter retune.
- **The fold_3 instability is resolved** — by the Phase 3 #4 retune that already shipped in `spec.md` §8.2 / `config.py`. No further action needed for fold_3.
- **A smarter encoding might still be worth a future spike:** `cal_days_since_excise_change` (a recency-decaying signal that captures the discontinuity *moment* without leaving a constant column afterwards) would avoid the near-constant-within-fold trap. Low priority — the production model is already stable on fold_3.
- **Fold_6 needs a different hypothesis** — crude-volatility regime features, not policy. Backlog candidate for a v4.1 spike.

## What's pinned in this branch

- `tools/research/v4_excise_fold_instability.py` — the experiment (excise schedule + seed-noise runner + corrected three-way comparison)
- `results/v4_excise_fold_instability_summary.md` — auto-generated per-fold tables
- `results/v4_excise_fold_instability.json` — raw metrics
- This outcome doc

Per-experiment `models_kfold_v4_excise_seed_*/` dirs are gitignored (`models_*/` pattern).

## Sources

- [`docs/research/2026-06_v3.0_phase3_closing_summary.md`](2026-06_v3.0_phase3_closing_summary.md) — Phase 3 #2 (seed-noise floor) + #4 (hyperopt retune)
- `results/v3_phase3_seed_noise.json` — OLD-defaults baseline (cumulative comparison)
- `results/v3_phase3_hyperopt_validation.json` — clean baseline (new tuned defaults, no excise)
- **Excise schedule:** Australian Government Federal Treasury March 2022 budget papers; ATO fuel excise indexation tables. The 2022-03-30 cut and 2022-09-29 restoration are the load-bearing dates.
