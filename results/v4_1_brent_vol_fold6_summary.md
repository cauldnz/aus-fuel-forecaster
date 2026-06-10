# v4.1 hypothesis test — Brent realized-volatility features -> fold_6 stability

Adds 3 Brent realized-volatility features to the upstream block and re-runs the seed-noise protocol. Tests whether fold_6's residual instability (clean-baseline seed-stdev 0.100, still the worst fold after the Phase 3 #4 retune) is a crude-volatility regime the model couldn't see. Comparison is against the **clean** baseline (new tuned defaults, no extra features) — same hyperparameters both sides.

New features: `upstream_brent_realized_vol_14d`, `upstream_brent_realized_vol_30d`, `upstream_brent_vol_ratio_14_90`

## Per-fold seed-stdev — feature-only effect

| Fold | WITHOUT vol | WITH vol | Δ stdev | Δ % | Verdict |
|------|------------:|---------:|--------:|----:|---------|
| fold_1 | 0.0191 | 0.0316 | +0.0125 |   +65% | worsened |
| fold_2 | 0.0307 | 0.0187 | -0.0120 |   -39% | improved |
| fold_3 | 0.0436 | 0.0832 | +0.0395 |   +91% | worsened |
| fold_4 | 0.0398 | 0.0395 | -0.0003 |    -1% | unchanged |
| fold_5 | 0.0454 | 0.0499 | +0.0045 |   +10% | unchanged |
| fold_6 | 0.1002 | 0.0506 | -0.0496 |   -50% | improved |

## Per-fold seed-mean MAE — feature-only effect

Negative Δ = vol features reduced MAE.

| Fold | WITHOUT vol | WITH vol | Δ mean |
|------|------------:|---------:|-------:|
| fold_1 | 5.3335 | 5.3366 | +0.0031 |
| fold_2 | 4.7624 | 4.7777 | +0.0153 |
| fold_3 | 6.0898 | 6.0953 | +0.0055 |
| fold_4 | 5.0015 | 5.0772 | +0.0756 |
| fold_5 | 4.4115 | 4.4658 | +0.0542 |
| fold_6 | 3.5453 | 3.4948 | -0.0505 |

## Per-seed per-fold MAE_A (with vol features)

| Seed | fold_1 | fold_2 | fold_3 | fold_4 | fold_5 | fold_6 | Wall-clock |
|------|---|---|---|---|---|---|----|
| 42 | 5.3517 | 4.7694 | 6.1598 | 5.1139 | 4.4562 | 3.5223 | 11.3 min |
| 1 | 5.2961 | 4.7428 | 6.0650 | 5.0561 | 4.5266 | 3.5346 | 9.8 min |
| 7 | 5.3075 | 4.8030 | 5.9314 | 5.0198 | 4.5261 | 3.5086 | 9.1 min |
| 13 | 5.3248 | 4.7902 | 6.1685 | 5.1143 | 4.3814 | 3.3871 | 9.1 min |
| 99 | 5.3914 | 4.7814 | 6.1597 | 5.0414 | 4.4485 | 3.5290 | 10.4 min |
| 123 | 5.3481 | 4.7794 | 6.0876 | 5.1175 | 4.4557 | 3.4874 | 10.3 min |

## Hypothesis verdict

- fold_6 seed-stdev: 0.1002 (without) -> 0.0506 (with vol)
- **Hypothesis: PARTIAL**

**PARTIAL.** Vol features help fold_6 but don't fully stabilise it. Worth adding if mean MAE holds elsewhere, but fold_6 has residual instability from another source.

## Sources

- `tools/research/v4_1_brent_vol_fold6.py` — this script
- `results/v3_phase3_hyperopt_validation.json` — clean baseline
- `docs/research/2026-06_v4_fold_instability_excise_outcome.md` — v4 (excise) precursor
