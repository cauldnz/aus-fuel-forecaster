# v4 hypothesis test — fuel excise feature → fold-3 stability

Adds ``cal_fuel_excise_cents_per_litre`` to the calendar block and re-runs the Phase 3 #2 seed-noise protocol. Tests whether fold_3's seed-instability (Phase 3 #2: 0.163 c/L stdev, 3-5× higher than the other folds) is the model failing to extrapolate over the Sept 28 2022 fuel excise restoration.

**Methodology note (corrected 2026-06-10):** the clean comparison is **against the hyperopt-validation baseline** (new tuned defaults WITHOUT the excise feature) — same hyperparameters on both sides, feature is the only experimental difference. The initial version of this script compared to the Phase 3 #2 OLD-defaults baseline, which conflated the retune's effect with the feature's effect. The OLD-defaults numbers are kept below as the **cumulative** (retune + feature) view for context.

## Clean comparison — excise feature ONLY (new tuned defaults both sides)

Per-fold seed-stdev under the new tuned defaults, with and without the excise feature:

| Fold | WITHOUT excise | WITH excise | Δ stdev | Δ % | Verdict |
|------|---------------:|------------:|--------:|----:|---------|
| fold_1 | 0.0191 | 0.0250 | +0.0059 |   +31% | worsened |
| fold_2 | 0.0307 | 0.0314 | +0.0007 |    +2% | unchanged |
| fold_3 | 0.0436 | 0.0520 | +0.0084 |   +19% | unchanged |
| fold_4 | 0.0398 | 0.0339 | -0.0058 |   -15% | unchanged |
| fold_5 | 0.0454 | 0.0891 | +0.0437 |   +96% | worsened |
| fold_6 | 0.1002 | 0.1954 | +0.0952 |   +95% | worsened |

## Cumulative comparison — retune + excise feature (vs Phase 3 #2 OLD defaults)

Shows the *total* improvement when both the hyperparameter retune AND the excise feature are added, vs the original v1/v2 spec defaults. This is the picture against the **original** Phase 3 #2 baseline.

| Fold | OLD defaults stdev | NEW (retune + excise) stdev | Δ stdev |
|------|-------------------:|----------------------------:|--------:|
| fold_1 | 0.0544 | 0.0250 | -0.0293 |
| fold_2 | 0.0345 | 0.0314 | -0.0032 |
| fold_3 | 0.1626 | 0.0520 | -0.1106 |
| fold_4 | 0.0330 | 0.0339 | +0.0009 |
| fold_5 | 0.0687 | 0.0891 | +0.0204 |
| fold_6 | 0.1782 | 0.1954 | +0.0172 |

## Per-fold seed-mean MAE — new vs baseline

Negative Δ = new feature reduced MAE. The headline test is the **stdev** column above; the mean is a secondary check that the feature isn't degrading overall MAE.

| Fold | Baseline mean | New mean | Δ mean |
|------|--------------:|---------:|-------:|
| fold_1 | 5.3335 | 5.3721 | +0.0387 |
| fold_2 | 4.7624 | 4.7781 | +0.0158 |
| fold_3 | 6.0898 | 6.0987 | +0.0089 |
| fold_4 | 5.0015 | 5.0464 | +0.0449 |
| fold_5 | 4.4115 | 4.3389 | -0.0726 |
| fold_6 | 3.5453 | 3.9448 | +0.3994 |

## Per-seed per-fold MAE_A (new feature)

| Seed | fold_1 | fold_2 | fold_3 | fold_4 | fold_5 | fold_6 | Wall-clock |
|------|---|---|---|---|---|---|----|
| 42 | 5.3918 | 4.8201 | 6.0661 | 5.0455 | 4.2589 | 4.0947 | 0.0 min (cached) |
| 1 | 5.3786 | 4.7509 | 6.1161 | 5.0514 | 4.4039 | 3.6444 | 0.0 min (cached) |
| 7 | 5.3665 | 4.8188 | 6.1228 | 4.9973 | 4.4807 | 4.0516 | 0.0 min (cached) |
| 13 | 5.3792 | 4.7676 | 6.1652 | 5.0177 | 4.3760 | 4.2082 | 0.0 min (cached) |
| 99 | 5.3961 | 4.7378 | 6.0017 | 5.1048 | 4.2245 | 3.9108 | 0.0 min (cached) |
| 123 | 5.3205 | 4.7737 | 6.1201 | 5.0617 | 4.2895 | 3.7589 | 0.0 min (cached) |

## Hypothesis verdict

**Hypothesis FALSIFIED.** Fold_3's seed-stdev is essentially unchanged. The fuel excise feature doesn't explain the instability. **Action:** investigate alternative hypotheses — maybe fold_3's variance is the 2022-23 LNG/oil shock, or training-data composition (fold_3 trains on 2017-2022-04 — the last 4 months are pre-cut, then test on cut+restore).

**Bonus — fold_6 verdict:** worsened. The 2026 spike-period instability has a different cause than the 2022 excise cut, but if it also shows movement, the excise feature might be capturing some broader regime-change signal worth investigating further.

## Sources

- `tools/research/v4_excise_fold_instability.py` — this script
- `results/v3_phase3_hyperopt_validation.json` — CLEAN baseline (new tuned defaults, no excise) — the fair feature-only comparison
- `results/v3_phase3_seed_noise.json` — OLD-defaults baseline (cumulative retune+feature comparison)
- `docs/research/2026-06_v3.0_phase3_closing_summary.md` — Phase 3 #2 + #4
- `docs/research/2026-06_v4_fold_instability_excise_outcome.md` — narrative

**Excise schedule sources:** Australian Government Federal Treasury, March 2022 budget papers; ATO indexation tables.
