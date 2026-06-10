# v5 — 7-day horizon: the augmentor null generalises across horizons

**Date:** 2026-06-11
**Branch:** `claude/v5-7day-horizon`
**Status:** complete — augmentor still null at t+7; v3.0 ship-Model-A conclusion holds at a second horizon
**See also:** [`2026-06_v3.0_phase3_closing_summary.md`](2026-06_v3.0_phase3_closing_summary.md)

## The question

The v3.0 conclusion — ship Model A, the SA2 augmentor adds no robust lift — was established on the **1-day** target (`y_t1`). The most plausible "but maybe it helps elsewhere" escape hatch: at a **longer horizon** the lag features carry less information (tomorrow's price ≈ today's, but the t+1..t+7 mean is harder to pin from recent lags), so the augmentor's static demographic context might finally win where the lag advantage decays.

This experiment tests that directly: re-run the v3.0 Phase 2 PR-B-baseline A-vs-B comparison at the **7-day** target (`y_t1_t7` = mean of price[t+1..t+7]), 6-fold k-fold, v3.0 tuned defaults.

## Prerequisite fix — `horizon_days` leak

`KFoldConfig.horizon_days` was documented as widening the train/test gap but `fold_bounds()` ignored it (only `gap_days` was used). Harmless at t+1, but for `y_t1_t7` a train row at the boundary has a target spanning [t+1, t+7] that would leak into the test window. Fixed (commit `a710531`): the train cutoff now pulls back by `horizon_days - 1` extra days, so `horizon_days=7` widens the gap to 8 days and the 7-day target lands entirely in the gap. Backward-compatible (horizon_days=1 → no change); 26 fold tests pass.

## Result: still null

| Metric | t+1 (v3.0 Phase 2) | **t+7 (this run)** |
|---|---:|---:|
| Mean Δ MAE (B−A) | +0.215 | **+0.041** |
| Stdev across folds | 0.394 | 0.370 |
| Verdict | noise | **noise** |

`|Mean| (0.041) ≪ Stdev (0.370)` → squarely in the noise band. Model B (with the 15-col SA2 block) does not beat Model A at t+7.

Per-fold Δ MAE (B−A) at t+7:

| Fold | MAE A | MAE B | Δ MAE |
|------|------:|------:|------:|
| fold_1 | 8.217 | 8.211 | −0.007 |
| fold_2 | 13.835 | 14.172 | +0.337 |
| fold_3 | 25.157 | 25.217 | +0.060 |
| fold_4 | 16.162 | 16.752 | +0.590 |
| fold_5 | 8.817 | 8.666 | −0.152 |
| fold_6 | 18.807 | 18.222 | −0.585 |

Same scattered pattern as t+1 — 3 folds marginally favour B, 3 favour A, no consistency. (Note the per-fold MAE is 2-3× higher than t+1, confirming the 7-day target is genuinely harder — the premise held; the augmentor just didn't fill the gap.)

## Reading

**The premise was sound and the test was fair.** The 7-day target really is harder (per-fold MAE 8-25 c/L vs 4-13 at t+1; lag features genuinely weaker). But the harder target **did not open a gap the augmentor fills**. Whatever predicts the 7-day price mean beyond the lag/upstream/calendar/weather blocks, it isn't SA2 demographics.

This is the **strongest available confirmation of the v3.0 conclusion**: the augmentor null now holds at **two independent horizons**. A single-horizon null invites "maybe the horizon was wrong"; a two-horizon null closes that. The interesting twist — the augmentor is marginally *less* bad at t+7 (+0.041 vs +0.215) — is still firmly within noise and doesn't change the verdict.

## Decision — stop here, don't expand

The v3.0 Phase 2 protocol tested 8 augmentor variants at t+1; all were no better than the PR B baseline. At t+7 the PR B baseline (the augmentor's best-curated config) is flat. Running the full 8-variant + 6-seed-noise protocol at t+7 would cost ~5h of compute to confirm 8 nulls instead of 1 — not worth it given:
- the t+1 evidence already showed the other 7 variants don't beat the baseline, and
- the t+7 baseline is itself flat.

**Explicit scope note (no silent cap):** this experiment ran only the PR B baseline A-vs-B at t+7, not the full 8-variant sweep or the seed-noise floor. That's a deliberate stop based on the decisive baseline result, not an oversight. If a future reason emerges to revisit the augmentor at t+7 (e.g. a new feature family), the full protocol would be the next step.

## What ships

Nothing changes. Model A on v3.0 tuned defaults remains the production model, now validated as the right call at both the 1-day and 7-day horizons. The 7-day horizon itself (spec §13.8 backlog) remains a valid *product* direction — Model A could be trained for t+7 if a 7-day forecast is wanted — but it doesn't change the augmentor verdict.

## What's pinned in this branch

- `src/fuel_pred/train/folds.py` — the horizon_days gap fix (+ test)
- `tools/research/v5_7day_horizon.py` — the experiment
- `results/v5_7day_horizon_kfold.md` — full per-fold report
- `results/v5_7day_horizon_headline.md` + `.json` — headline + t+1 comparison
- This outcome doc

`models_kfold_v5_7day/` is gitignored (`models_*/`).

## Sources

- `results/v3_phase2_pr_b_baseline_kfold.md` — t+1 reference
- [`2026-06_v3.0_phase3_closing_summary.md`](2026-06_v3.0_phase3_closing_summary.md) — the v3.0 ship-Model-A decision this corroborates
