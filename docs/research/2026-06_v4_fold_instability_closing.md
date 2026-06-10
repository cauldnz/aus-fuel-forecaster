# v4 fold-instability investigation — closing summary

**Date:** 2026-06-11
**Branch:** `claude/v4-fold-instability`
**Status:** complete — three hypotheses tested; none shippable; instability is retune-resolved + residually irreducible
**See also:** [`2026-06_v4_fold_instability_excise_outcome.md`](2026-06_v4_fold_instability_excise_outcome.md), [`2026-06_v3.0_phase3_closing_summary.md`](2026-06_v3.0_phase3_closing_summary.md)

## The question

Phase 3 #2 found that **folds 3 (2022-23) and 6 (2025-26) had 3-5× higher per-fold seed-stdev** than the other folds — under the *original* v1/v2 hyperparameters. This investigation asked: what causes that instability, and can a feature fix it?

Three hypotheses, each tested with the same 6-seed protocol (Model A only, per-fold seed-stdev, **matched-seed comparison against the clean hyperopt-validation baseline** — new tuned defaults, no extra feature):

1. **v4** — fold_3 is the 2022 fuel excise cut/restore. Add `cal_fuel_excise_cents_per_litre`.
2. **v4.1** — fold_6 is a crude-volatility regime. Add Brent realized-vol (14d/30d) + vol-ratio.
3. **v4.2** — fold_6 is a fat-tail / jump-risk regime (user pointer). Add Brent return skew (30d) + kurtosis (30d/60d), nested on v4.1.

## The headline result

**Two things fixed the folds, and neither was a new feature:**

1. **The Phase 3 #4 hyperparameter retune did the real work.** It dropped fold_3 seed-stdev 0.163 → 0.044 (−73%) and fold_6 0.178 → 0.100 (−44%), purely from regularization (smaller trees, no row bagging, heavier leaf reg, L1). This was already shipped in spec §8.2.

2. **No added feature improves aggregate stability beyond the retune.** Every feature tested merely *relocates* instability between folds:

| Fold | clean (retuned, no feat) | v4 excise | v4.1 vol | v4.2 vol+kurt |
|------|-------------------------:|----------:|---------:|--------------:|
| fold_1 | 0.0191 | 0.0250 | 0.0316 | 0.0332 |
| fold_2 | 0.0307 | 0.0314 | 0.0187 | 0.0065 |
| fold_3 | 0.0436 | 0.0520 | **0.0832** | 0.0452 |
| fold_4 | 0.0398 | 0.0339 | 0.0395 | **0.0751** |
| fold_5 | 0.0454 | 0.0891 | 0.0499 | 0.0506 |
| fold_6 | **0.1002** | 0.1954 | **0.0506** | 0.0775 |
| **SUM** | **0.2788** | 0.331* | 0.2734 | 0.2881 |

*excise sum vs OLD-defaults baseline in its own doc; shown here re-based on the clean baseline for comparability.

Aggregate stdev: clean 0.279 → vol 0.273 (−2%, flat) → vol+kurt 0.288 (+3%, worse). Mean-of-fold-means MAE: clean 4.857 → vol 4.875 → vol+kurt 4.873 — every feature set is slightly *worse* on accuracy.

## What each hypothesis taught us

### v4 — excise: FALSIFIED
fold_3's instability was a **regularization** problem, not a missing-feature problem. The retune already fixed it; adding the excise feature made fold_3 marginally worse and badly destabilised folds 5+6 (where excise is near-constant within the test window — capacity without signal). See the [excise outcome doc](2026-06_v4_fold_instability_excise_outcome.md).

### v4.1 — realized volatility: PARTIAL but not shippable
fold_6 genuinely halves (0.100 → 0.051, −50%, converged across seeds). **Crude volatility IS a real source of fold_6's instability** — the hypothesis is confirmed. But broad vol "lights up" in *both* the 2022 (fold_3) and 2026 (fold_6) shock periods, so it stabilises fold_6 while **destabilising fold_3** (0.044 → 0.083, +91%). Net aggregate −2% (flat); mean MAE +0.017 worse. A redistribution, not an improvement.

### v4.2 — kurtosis + skew: NEUTRAL-to-NEGATIVE beyond vol
The user's intuition (kurtosis = jump risk, often informative) is sound, and the features are well-formed (excess kurtosis mean +2.4, genuinely variable in fold_6). But on this data the higher moments add **no marginal value beyond vol** on the target fold — they actually pull fold_6 back up (0.051 → 0.077). Kurtosis and realized vol are correlated (high-vol windows are usually fat-tailed), so they carry overlapping regime signal; vol got there first.

**The interesting sub-finding:** kurtosis has a *different fold-sensitivity* than vol — it **undoes vol's fold_3 damage** (0.083 → 0.045, back to baseline) and sharply stabilises fold_2 (0.031 → 0.007), but introduces *new* fold_4 damage (0.040 → 0.075). So the higher moments aren't useless — they're sensitive to different periods than vol — they just don't net out positive on the fold we were targeting.

## Why fold_6 resists

fold_6 (2025-26) is the most unstable fold even after everything. The retune halved it; vol halved it again but at fold_3's expense; kurtosis gives most of that back. The residual instability is most likely the **2026 structural-break magnitude itself** — the model trains on 2017–2025-04 and tests on a price spike larger than anything in its training history. That's an extrapolation problem no *regime* feature fixes, because the issue isn't "the model can't see the regime" but "the model has never seen a shock this size." It's plausibly irreducible without 2026-like data in the training window — which, mechanically, fold_6 (the most recent fold) can never have.

## Decisions

- **Ship nothing from the v4 line.** No feature (excise, vol, kurtosis, skew) improves aggregate stability or mean MAE over the already-shipped Phase 3 #4 retuned Model A.
- **fold_3 + fold_6 instability is as-resolved-as-it-gets** via the retune. The production model (spec §8.2 tuned defaults) is the right operating point.
- **The "feature relocates instability between folds" pattern is the durable methodology lesson** — generalises the v4 near-constant principle: a regime feature's value is per-fold, and a feature that helps the target fold can silently hurt another. Always read the *full* per-fold vector + aggregate, never just the target fold.
- **A general feature-evaluation methodology lesson worth keeping:** per-fold stdev at small seed counts is wildly unreliable — fold_6 read −94% at 2 seeds and converged to −50% at 6. And any feature-vs-baseline comparison must hold seed-count fixed (the v4.2 script initially compared partial-vs-full and spuriously reported "higher moments help"). Both now baked into the runners.

## Future (low priority)

- **fold_6 extrapolation** could be probed with a magnitude-of-break feature (e.g. deviation of current Brent from a long trailing mean) rather than a volatility/regime feature — but the v4 line suggests the ceiling is low. Backlog only.
- A `cal_days_since_excise_change` recency-decay encoding (avoids the near-constant trap) remains a cleaner-than-raw-excise option if fold_3 ever resurfaces; low priority since fold_3 is retune-stable.

## What's pinned in this branch

- `tools/research/v4_excise_fold_instability.py` + `v4_1_brent_vol_fold6.py` + `v4_2_brent_kurtosis_fold6.py`
- `results/v4_excise_fold_instability_*` + `v4_1_brent_vol_fold6_*` + `v4_2_brent_kurtosis_fold6_*`
- This closing doc + the excise outcome doc

Per-experiment `models_kfold_v4*_seed_*/` dirs + `features_v4*.parquet` are gitignored.

## Sources

- `results/v3_phase3_hyperopt_validation.json` — clean baseline (new tuned defaults, no extra feature)
- [`2026-06_v3.0_phase3_closing_summary.md`](2026-06_v3.0_phase3_closing_summary.md) — Phase 3 #2 (seed-noise) + #4 (retune)
- User domain pointer: kurtosis of returns as a regime feature (v4.2 motivation)
